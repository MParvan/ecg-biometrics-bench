import numpy as np
import pandas as pd
import os
from pathlib import Path
import glob
from typing import List, Optional, Dict
import datetime
import shutil
import zipfile
import tempfile
import requests
import yaml
import wfdb
import re
import patoolib
import collections
from tqdm import tqdm  # For real-time progress bars
from preprocessing import Preprocessing

# =============================================================================
# CONFIGURATION LOADING
# =============================================================================
def load_config(config_name: str = "config.yaml") -> dict:
    """
    Robust config loader that searches multiple locations.
    Priority:
    1. Same directory as this script.
    2. Current working directory (CWD).
    3. Parent directory of CWD.
    """
    search_paths = [
        Path(__file__).resolve().parent / config_name,
        Path.cwd() / config_name,
        Path.cwd().parent / config_name
    ]

    for path in search_paths:
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f)
                
    raise FileNotFoundError(f"Could not find '{config_name}'. Checked: {[str(p) for p in search_paths]}")

CONFIG = load_config()

# =============================================================================
# SHARED UTILITIES
# =============================================================================

def _download_and_extract(url: str, zip_path: Path, extract_to: Path, dataset_name: str, cleanup: bool = False):
    """
    Helper to download a file with a real-time progress bar and extract it.
    Includes auto-cleanup for corrupt files and handles nested ZIP/RAR archives.
    
    Args:
        url (str): Direct download link.
        zip_path (Path): Where to save the compressed file.
        extract_to (Path): Folder to extract contents into.
        dataset_name (str): For print logging.
        cleanup (bool): If True, deletes the zip/rar file after successful extraction.
    """
    extract_to.mkdir(parents=True, exist_ok=True)
    
    # 1. DOWNLOAD PHASE
    if not zip_path.exists():
        print(f"[INFO] Downloading {dataset_name}...")
        try:
            # Use browser headers to prevent 403 Forbidden errors from sites like Figshare
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            
            response = requests.get(url, stream=True, headers=headers)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            # Write file in chunks with progress bar
            with open(zip_path, "wb") as f, tqdm(
                desc=f"Downloading {dataset_name}", 
                total=total_size, 
                unit='iB', 
                unit_scale=True, 
                unit_divisor=1024
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)
                    
        except Exception as e:
            print(f"[ERR] Download failed: {e}")
            # Clean up partial file so we don't try to unzip a corrupt file later
            if zip_path.exists(): os.remove(zip_path)
            return

    # 2. EXTRACTION PHASE
    # Only extract if the target directory is empty
    if not any(extract_to.iterdir()):
        print(f"[INFO] Extracting {dataset_name}...")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Handle ZIP files
                if zip_path.suffix == ".zip":
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(temp_dir)
                        
                # Handle RAR files (requires 'patool' and system 7-Zip/Unrar)
                elif zip_path.suffix == ".rar":
                    patoolib.extract_archive(str(zip_path), outdir=temp_dir)
                
                # --- Intelligent Move Logic ---
                # Many archives wrap everything in a single top-level folder.
                # We detect this and move the *contents* up one level to keep paths clean.
                content = [d for d in os.listdir(temp_dir)]
                if len(content) == 1 and os.path.isdir(os.path.join(temp_dir, content[0])):
                    src = os.path.join(temp_dir, content[0])
                    for item in os.listdir(src):
                        shutil.move(os.path.join(src, item), extract_to)
                else:
                    # Move everything directly
                    for item in content:
                        shutil.move(os.path.join(temp_dir, item), extract_to)
                        
            print(f"[INFO] {dataset_name} ready.")
            
            # Optional cleanup
            if cleanup: 
                print(f"[INFO] Cleaning up zip file: {zip_path.name}")
                os.remove(zip_path)
                
        except Exception as e:
            print(f"[ERR] Extraction failed: {e}")
            print(f"[ACTION] Deleting corrupt file {zip_path.name} to force re-download on next run.")
            if zip_path.exists(): os.remove(zip_path)
            if extract_to.exists(): shutil.rmtree(extract_to)

# =============================================================================
# 1. ECG-ID
# =============================================================================
class load_ecgid_dataset():
    """
    Robust Loader for ECG-ID Database.
    
    This class handles downloading, parsing (via WFDB), and splitting the ECG-ID dataset
    according to strict biometric evaluation protocols.
    
    Attributes:
        enrollment_mode (str): Defines the biometric evaluation regime.
            - 'short_term': (Day 1 Stability)
                Enroll = First recording of the first day.
                Probe  = All subsequent recordings on the same day.
                *Tests immediate re-authentication capabilities.*
            
            - 'long_term':  (Template Aging)
                Enroll = All recordings from the first day.
                Probe  = All recordings from all future days.
                *Tests robustness to physiological changes over days/months.*
            
            - 'maximal':    (Very Long-Term)
                Enroll = The absolute first recording.
                Probe  = The absolute last recording available.
                *Tests the limits of the system over the maximum available time gap.*
                
        signal_type (str): 
            - 'filtered': Loads 'rec_1.dat' (Pre-filtered by original authors).
            - 'noisy': Loads 'rec_1n.dat' (Original raw signal with noise).
    """
    def __init__(self, num_beats=1, merge_strategy="average", enrollment_mode="long_term", 
                 signal_type=None, preprocessing_params=None, cleanup_zip=False):
        self.preprocessor = Preprocessing()
        self.cfg = CONFIG["datasets"]["ecgid"]
        project_dir = Path(__file__).resolve().parent
        self.data_root = (project_dir / CONFIG["project"]["data_root"]).resolve()
        self.dataset_root = self.data_root / self.cfg["root_dir"]
        self.zip_path = self.data_root / self.cfg["zip_name"]
        self.url = self.cfg["url"]
        
        self.signal_type = signal_type if signal_type else self.cfg.get("signal_type", "filtered")
        self.prep_params = preprocessing_params if preprocessing_params else self.cfg.get("preprocessing", {})
        
        self.num_beats = num_beats
        self.merge_strategy = merge_strategy
        self.cleanup_zip = cleanup_zip
        
        # Validate biometric regimes
        valid_modes = ["short_term", "long_term", "maximal"]
        if enrollment_mode not in valid_modes:
            # Backward compatibility mapping
            if enrollment_mode == "first_date": enrollment_mode = "long_term"
            elif enrollment_mode == "first_vs_last": enrollment_mode = "maximal"
            elif enrollment_mode == "first_record": enrollment_mode = "short_term"
            else: raise ValueError(f"Invalid mode: {enrollment_mode}. Use {valid_modes}")
        self.enrollment_mode = enrollment_mode

    def download(self):
        """Downloads and extracts the dataset if not already present."""
        _download_and_extract(self.url, self.zip_path, self.dataset_root, "ECG-ID", cleanup=self.cleanup_zip)

    def _extract_rec_number(self, filename):
        """Helper to extract '1' from 'rec_1.dat' for sorting."""
        match = re.search(r'rec_(\d+)', filename)
        return int(match.group(1)) if match else 9999

    def load_raw_data(self):
        """
        Scans the directory structure, reads WFDB headers, and parses recording dates.
        Returns a dictionary: { 'patient_id': [list of recording dicts sorted by time] }
        """
        if not self.dataset_root.exists() or not any(self.dataset_root.iterdir()):
            self.download()
        
        recordings = {}
        for subject_dir in tqdm(sorted(self.dataset_root.glob("Person*")), desc="Loading ECG-ID raw files"):
            sid = subject_dir.name.replace("Person_", "")
            recs = []
            hea_files = sorted(subject_dir.glob("*.hea"))
            
            for hea_path in hea_files:
                # Filter specific signal versions (clean vs noisy)
                is_noisy = "n.hea" in hea_path.name
                if self.signal_type == "filtered" and is_noisy: continue
                if self.signal_type == "noisy" and not is_noisy: continue
                
                try:
                    # Read header to get sampling rate and date
                    record = wfdb.rdrecord(str(hea_path.with_suffix("")))
                    rec_date = record.base_date
                    
                    # Robust date parsing from comments if base_date is missing
                    if rec_date is None:
                        for comment in record.comments:
                            if "date" in comment.lower():
                                try: 
                                    date_str = comment.split(":")[-1].strip()
                                    rec_date = datetime.datetime.strptime(date_str, "%d.%m.%Y").date()
                                except: pass
                    if rec_date is None: rec_date = datetime.date.min

                    recs.append({
                        'signal': record.p_signal[:, 0], # Lead I
                        'date': rec_date,
                        'fs': record.fs,
                        'filename': hea_path.name
                    })
                except Exception as e: pass

            # Sort chronologically to enable correct Short/Long term splitting
            recs.sort(key=lambda x: (x['date'], self._extract_rec_number(x['filename'])))
            recordings[sid] = recs
            
        return recordings

    def _process_signal(self, sig, fs):
        """
        Applies filtering, segmentation (beat vs blind), and normalization.
        Handles multi-beat merging (stacking or averaging) if configured.
        """
        beats = self.preprocessor.preprocess_ecg(
            ecg=sig, fs=fs, 
            mode=self.prep_params.get("mode", "beat"),
            window_s=self.prep_params.get("window_len", 5.0),
            stride_s=self.prep_params.get("stride", 1.0),
            pre_s=self.prep_params.get("pre_s", 0.2), 
            post_s=self.prep_params.get("post_s", 0.4), 
            filter_method="butter" if self.prep_params.get("bandpass") else None,
            filter_kwargs={'low': self.prep_params.get("lowcut", 0.5), 'high': self.prep_params.get("highcut", 40.0)},
            norm_method="zscore" if self.prep_params.get("normalize") else None
        )
        
        if self.num_beats == 1: return beats
        if len(beats) < self.num_beats: return np.empty((0, beats.shape[1])) 
        
        processed_samples = []
        for i in range(0, len(beats) - self.num_beats + 1):
            group = beats[i : i + self.num_beats] 
            if self.merge_strategy == "average":
                processed_samples.append(np.mean(group, axis=0))
            elif self.merge_strategy == "concat":
                processed_samples.append(group.flatten())
        return np.array(processed_samples)

    def load_all_sessions(self):
        """
        Loads the entire dataset without splitting. 
        Used primarily for random-split experiments.
        """
        data = self.load_raw_data()
        x_list, y_list = [], []
        for sid, recs in tqdm(data.items(), desc="Processing signals"):
            for rec in recs:
                segments = self._process_signal(rec['signal'], rec['fs'])
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))

    def load_session(self, session_name):
        """
        Loads 'Session_1' (Enrollment) or 'Session_2' (Probe) based on the 
        configured enrollment_mode (Short-Term, Long-Term, or Maximal).
        """
        data = self.load_raw_data()
        x_list, y_list = [], []
        is_enrollment = (session_name == "Session_1")
        
        for sid, recs in tqdm(data.items(), desc=f"Processing {session_name}"):
            if not recs: continue
            target_recs = []

            # --- MODE 1: SHORT-TERM (Intra-Day) ---
            # Objective: Test stability within a single session/day.
            # Split: Record 1 of Day 1 (Enroll) vs Rest of Day 1 (Probe).
            if self.enrollment_mode == "short_term":
                first_date = recs[0]['date']
                # Isolate only the recordings from the very first day
                day1_recs = [r for r in recs if r['date'] == first_date]
                
                if len(day1_recs) < 2: 
                    # EDGE CASE: Subject has only 1 recording on Day 1.
                    # Fallback: Split that single recording into First Half (Enroll) vs Last Half (Probe).
                    if len(day1_recs) == 1:
                        target_recs = day1_recs
                        # Note: The splitting logic happens in the 'EXTRACTION' loop below.
                    else: 
                        continue # No valid data for this subject
                else:
                    if is_enrollment: target_recs = [day1_recs[0]]
                    else: target_recs = day1_recs[1:]

            # --- MODE 2: LONG-TERM (Inter-Day) ---
            # Objective: Test robustness to template aging (days/weeks/months).
            # Split: All Day 1 records (Enroll) vs All Future records (Probe).
            elif self.enrollment_mode == "long_term":
                first_date = recs[0]['date']
                dates = sorted(list(set(r['date'] for r in recs)))
                
                # Requires at least 2 distinct dates
                if len(dates) < 2: continue 
                
                if is_enrollment: 
                    target_recs = [r for r in recs if r['date'] == first_date]
                else:
                    target_recs = [r for r in recs if r['date'] > first_date]

            # --- MODE 3: MAXIMAL (First vs Last - STRICT VLT) ---
            # Objective: Test the maximum possible time gap (Extreme Aging).
            # Constraint: Must be CROSS-DAY. Skips subjects with only 1 day of data.
            elif self.enrollment_mode == "maximal":
                if len(recs) < 2: continue
                
                # Ensure it is strictly cross-day
                if recs[0]['date'] == recs[-1]['date']:
                    continue # Skip this subject (Short-term context)

                if is_enrollment: target_recs = [recs[0]]
                else: target_recs = [recs[-1]]

            # --- EXTRACTION & SIGNAL PROCESSING ---
            for rec in target_recs:
                segments = self._process_signal(rec['signal'], rec['fs'])
                
                # EDGE CASE LOGIC: Single-Record Intra-Split
                # This only triggers if we are in Short-Term mode AND forced to use a single file.
                if self.enrollment_mode == "short_term" and len(target_recs) == 1 and len(recs) == 1:
                     if len(segments) > 1:
                        mid = len(segments) // 2
                        if is_enrollment: segments = segments[:mid] # First 50%
                        else: segments = segments[mid:]             # Last 50%
                
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))

        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))

# =============================================================================
# 2. HeartPrint
# =============================================================================
class load_heartprint_dataset():
    """
    Flexible Loader for HeartPrint Dataset.
    
    Structure:
        - Session 1 (S1): Baseline/Rest.
        - Session 2 (S2): Rest (Time gap from S1).
        - Session 3R (S3R): Reading Condition (State Change).
        - Session 3L (S3L): Long Interval (Maximal Time Gap).
    
    Biometric Regimes (enrollment_mode):
        - 'standard' (Long-Term): 
             Enroll = S1. Probe = S2.
        - 'reverse' (Long-Term):
             Enroll = S2. Probe = S1.
        - 'state_robustness' (Reading Task):
             Enroll = S1. Probe = S3R (Reading).
             *Tests robustness to mental task/state changes.*
        - 'vlt' (Very Long-Term):
             Enroll = S1. Probe = S3L (Long Interval).
             *Tests robustness to maximal template aging.*
    """
    def __init__(self, num_beats=1, merge_strategy="average", enrollment_mode="standard", 
                 preprocessing_params=None, cleanup_zip=False):
        self.preprocessor = Preprocessing()
        self.cfg = CONFIG["datasets"]["heartprint"]
        project_dir = Path(__file__).resolve().parent
        self.data_root = (project_dir / CONFIG["project"]["data_root"]).resolve()
        self.dataset_root = self.data_root / self.cfg["root_dir"]
        self.zip_path = self.data_root / self.cfg["zip_name"]
        self.url = self.cfg["url"]
        
        self.sample_len = self.cfg.get("sample_length", 3747)
        self.prep_params = preprocessing_params if preprocessing_params else self.cfg.get("preprocessing", {})
        self.num_beats = num_beats
        self.merge_strategy = merge_strategy
        self.cleanup_zip = cleanup_zip
        
        valid_modes = ["standard", "reverse", "state_robustness", "vlt"]
        if enrollment_mode not in valid_modes:
             raise ValueError(f"Invalid HeartPrint mode: {enrollment_mode}. Use {valid_modes}")
        self.enrollment_mode = enrollment_mode

    def download(self):
        """
        Attempts robust download via Figshare API. 
        """
        # 1. Check if data already exists
        if self.dataset_root.exists():
            for root, dirs, files in os.walk(self.dataset_root):
                for d in dirs:
                    if "session" in d.lower(): return 

        self.dataset_root.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Attempting to download HeartPrint...")
        
        try:
            # 2. Download Phase
            if not self.zip_path.exists():
                match = re.search(r'articles/(\d+)/versions/(\d+)', self.url)
                if match:
                    aid, ver = match.groups()
                    api = f"https://api.figshare.com/v2/articles/{aid}/versions/{ver}"
                    r = requests.get(api); r.raise_for_status()
                    target = next((f for f in r.json()['files'] if 'zip' in f['name'] or 'rar' in f['name']), None)
                    if not target: raise ValueError("Archive not found")
                    dl_url = target['download_url']
                    size = target['size']
                else:
                    dl_url = self.url; size = 0

                with requests.get(dl_url, stream=True) as r:
                    r.raise_for_status()
                    if size == 0: size = int(r.headers.get('content-length', 0))
                    # Added tqdm here as requested
                    with open(self.zip_path, "wb") as f, tqdm(desc="Downloading", total=size, unit='iB', unit_scale=True) as bar:
                        for chunk in r.iter_content(8192): f.write(chunk); bar.update(len(chunk))

            # 3. Extraction Phase
            print(f"[INFO] Attempting extraction...")
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    patoolib.extract_archive(str(self.zip_path), outdir=temp_dir)
                except Exception:
                    with zipfile.ZipFile(self.zip_path, "r") as zf: zf.extractall(temp_dir)
                
                for item in os.listdir(temp_dir):
                    shutil.move(os.path.join(temp_dir, item), self.dataset_root)
            
            if self.cleanup_zip: os.remove(self.zip_path)

        except Exception as e:
            print(f"[WARN] Automated download failed: {e}")
            print("Please download manually and extract to:", self.dataset_root)

    def load_raw_data(self, target_folder_keyword):
        """
        Recursively searches for folder matching keyword (e.g., 'session3r').
        """
        if not self.dataset_root.exists() or not any(self.dataset_root.iterdir()):
            self.download()
        
        recordings = {}
        # Normalize: "Session-3R" -> "session3r"
        target_clean = target_folder_keyword.lower().replace("-", "").replace("_", "").replace(" ", "")
        found_path = None
        
        if self.dataset_root.exists():
            for root, dirs, files in os.walk(self.dataset_root):
                for d in dirs:
                    d_norm = d.lower().replace("-", "").replace("_", "").replace(" ", "")
                    # Strict check for 3L vs 3R
                    if target_clean in d_norm:
                        # Prevent "session3" from matching "session3r" incorrectly if ambiguous
                        if "3" in target_clean and "r" in target_clean and "r" not in d_norm: continue
                        if "3" in target_clean and "l" in target_clean and "l" not in d_norm: continue
                        
                        found_path = Path(root) / d
                        break
                if found_path: break
        
        if not found_path:
            # Fallback for some unzipped structures
            return {}

        for subj_dir in tqdm(sorted(os.listdir(found_path)), desc=f"Loading {target_folder_keyword}"):
            full_subj_path = found_path / subj_dir
            if not full_subj_path.is_dir(): continue
            
            sid = subj_dir
            if sid not in recordings: recordings[sid] = []
            
            for f in sorted(glob.glob(os.path.join(full_subj_path, "*"))): 
                if not os.path.isfile(f) or f.endswith(".rar") or f.endswith(".zip"): continue
                with open(f, 'r') as fp:
                    try:
                        lines = [l.strip() for l in fp.readlines()]
                        vals = [float(l) for l in lines if l][:self.sample_len]
                        if len(vals) > 0: recordings[sid].append(np.array(vals))
                    except ValueError: pass
        return recordings

    def _process_signal(self, sig, fs=250):
        beats = self.preprocessor.preprocess_ecg(
            sig, fs=fs, 
            mode=self.prep_params.get("mode", "beat"),
            window_s=self.prep_params.get("window_len", 5.0),
            stride_s=self.prep_params.get("stride", 1.0),
            pre_s=self.prep_params.get("pre_s", 0.2), post_s=self.prep_params.get("post_s", 0.4),
            filter_method="butter" if self.prep_params.get("bandpass") else None,
            filter_kwargs={'low': self.prep_params.get("lowcut", 0.5), 'high': self.prep_params.get("highcut", 40.0)},
            norm_method="zscore" if self.prep_params.get("normalize") else None
        )
        if self.num_beats == 1: return beats
        if len(beats) < self.num_beats: return np.empty((0, beats.shape[1]))
        
        processed_samples = []
        for i in range(0, len(beats) - self.num_beats + 1):
            group = beats[i : i + self.num_beats]
            if self.merge_strategy == "average":
                processed_samples.append(np.mean(group, axis=0))
            elif self.merge_strategy == "concat":
                processed_samples.append(group.flatten())
        return np.array(processed_samples)

    def load_all_sessions(self):
        """Loads S1 and S2 for random split."""
        data_s1 = self.load_raw_data("session1")
        data_s2 = self.load_raw_data("session2")
        
        for k, v in data_s2.items():
            if k in data_s1: data_s1[k].extend(v)
            else: data_s1[k] = v
            
        x_list, y_list = [], []
        # Added tqdm here
        for sid, recs in tqdm(data_s1.items(), desc="Processing all sessions"):
            for sig in recs:
                segments = self._process_signal(sig)
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
                    
        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))

    def load_session(self, session_name):
        """
        Mappings:
        - standard: S1 -> S2
        - reverse: S2 -> S1
        - state_robustness: S1 -> S3R
        - vlt: S1 -> S3L
        """
        target_folders = []
        
        if self.enrollment_mode == "standard":
            if session_name == "Session_1": target_folders = ["session1"]
            elif session_name == "Session_2": target_folders = ["session2"]

        elif self.enrollment_mode == "reverse":
            if session_name == "Session_1": target_folders = ["session2"]
            elif session_name == "Session_2": target_folders = ["session1"]

        elif self.enrollment_mode == "state_robustness":
            if session_name == "Session_1": target_folders = ["session1"]
            elif session_name == "Session_2": target_folders = ["session3r"] # Reading
        
        elif self.enrollment_mode == "vlt":
            if session_name == "Session_1": target_folders = ["session1"]
            elif session_name == "Session_2": target_folders = ["session3l"] # Long Interval
        
        x_list, y_list = [], []
        
        for folder in target_folders:
            data_dict = self.load_raw_data(folder)
            # Added tqdm here
            for sid, recs in tqdm(data_dict.items(), desc=f"Processing {folder} signals"):
                for sig in recs:
                    segments = self._process_signal(sig)
                    if len(segments) > 0:
                        x_list.append(segments)
                        y_list.extend([sid] * len(segments))
                        
        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))

# =============================================================================
# 3. PTB (Physikalisch-Technische Bundesanstalt)
# =============================================================================
class load_ptb_dataset():
    """
    Robust Loader for PTB Diagnostic ECG Database.
    
    This class manages the loading and splitting of the PTB database, which contains 
    varying numbers of records per patient (from 1 to >20) collected over months.
    
    Attributes:
        enrollment_mode (str): Defines the biometric evaluation regime.
            - 'short_term': (Day 1 Stability)
                Enroll = First recording of the first day.
                Probe  = All subsequent recordings on the same day.
                *Tests immediate re-authentication stability.*
            
            - 'long_term':  (Template Aging)
                Enroll = All recordings from the first day.
                Probe  = All recordings from all future days.
                *Tests robustness to physiological changes/disease progression over time.*
            
            - 'maximal':    (Very Long-Term)
                Enroll = The absolute first recording.
                Probe  = The absolute last recording available.
                *Tests the limits of the system over the maximum available time gap.*
        
        filter_subset (str): 
            - 'all': Uses all subjects.
            - 'multi_rec_only': Drops subjects with < 2 recordings.
            
        only_healthy (bool):
            - If True, filters dataset to include only Healthy Controls (HC).
    """
    def __init__(self, leads=['ii'], filter_subset="all", enrollment_mode="long_term",
                 only_healthy=False, num_beats=1, merge_strategy="average", preprocessing_params=None, cleanup_zip=False):
        """
        Args:
            leads (list): Target leads to load (e.g., ['ii', 'v5']).
            
            filter_subset (str):
                - 'all': Uses all available subjects.
                - 'multi_rec_only': Restricts dataset to subjects with at least 2 recordings.
                  (Required for strict Long-Term and Maximal evaluation).
            
            enrollment_mode (str):
                - 'short_term': Intra-day split (Record 1 vs Rest of Day 1).
                - 'long_term': Inter-day split (Day 1 vs Future Days).
                - 'maximal': First vs Last recording (Maximal temporal gap).
            
            only_healthy (bool): If True, filters out Pathological subjects (Myocardial Infarction, etc.)
                                 and keeps only Healthy Controls.
                                 
            num_beats (int): Number of beats to stack/average for the input template.
            preprocessing_params (dict): Configuration for the signal processing pipeline.
            cleanup_zip (bool): Remove .zip file after extraction to save space.
        """
        self.preprocessor = Preprocessing()
        self.cfg = CONFIG["datasets"]["ptb"]
        project_dir = Path(__file__).resolve().parent
        self.data_root = (project_dir / CONFIG["project"]["data_root"]).resolve()
        self.dataset_root = self.data_root / self.cfg["root_dir"]
        self.zip_path = self.data_root / self.cfg["zip_name"]
        self.url = self.cfg["url"]
        
        self.prep_params = preprocessing_params if preprocessing_params else self.cfg.get("preprocessing", {})
        self.target_leads = [l.lower() for l in leads] if isinstance(leads, list) else leads
        self.only_healthy = only_healthy
        self.num_beats = num_beats
        self.merge_strategy = merge_strategy
        self.cleanup_zip = cleanup_zip
        self.filter_subset = filter_subset
        
        valid_modes = ["short_term", "long_term", "maximal"]
        if enrollment_mode not in valid_modes:
            # Backward compatibility
            if enrollment_mode == "first_date": enrollment_mode = "long_term"
            elif enrollment_mode == "first_vs_last": enrollment_mode = "maximal"
            elif enrollment_mode == "first_record": enrollment_mode = "short_term"
            else: raise ValueError(f"Invalid mode: {enrollment_mode}. Use {valid_modes}")
        self.enrollment_mode = enrollment_mode

    def download(self):
        """Downloads and extracts the dataset if not already present."""
        _download_and_extract(self.url, self.zip_path, self.dataset_root, "PTB", cleanup=self.cleanup_zip)

    def _is_healthy(self, record):
        """Robust check for healthy status using multiple header fields."""
        healthy_keywords = ["healthy", "control", "volunteer", "donor", "normal"]
        target_fields = ["reason for admission", "clinical classification", "diagnose"]
        for comment in record.comments:
            c_lower = comment.lower()
            if any(field in c_lower for field in target_fields):
                if any(kw in c_lower for kw in healthy_keywords): return True
        return False

    def _parse_date_from_comments(self, comments):
        """Extracts date from PTB comments like '# ECG date: 05/06/1997'."""
        for comment in comments:
            if "ecg date" in comment.lower():
                try:
                    date_str = comment.split(":")[-1].strip()
                    for fmt in ["%d/%m/%Y", "%d-%b-%y", "%d.%m.%Y"]:
                        try: return datetime.datetime.strptime(date_str, fmt).date()
                        except ValueError: continue
                except: pass
        return None

    def _get_lead_indices(self, available_leads):
        """Maps requested lead names (e.g., 'ii', 'v5') to channel indices."""
        avail_norm = [l.lower().replace("ecg", "").strip() for l in available_leads]
        if self.target_leads == 'all': return list(range(len(available_leads)))
        indices = []
        for req in self.target_leads:
            req = req.strip().lower()
            try: indices.append(avail_norm.index(req))
            except ValueError: pass 
        return indices

    def load_raw_data(self):
        """
        Loads all WFDB records, parses metadata, and sorts chronologically.
        Returns: { 'patient_id': [list of recording dicts sorted by date] }
        """
        if not self.dataset_root.exists() or not any(self.dataset_root.iterdir()):
            self.download()
        
        recordings = {}
        # Find all .hea files recursively
        files = list(self.dataset_root.rglob("*.hea"))
        
        # Group files by patient folder (e.g., patient001)
        patient_groups = {}
        for f in files:
            pid = f.parent.name
            if pid not in patient_groups: patient_groups[pid] = []
            patient_groups[pid].append(f)

        for sid, p_files in tqdm(sorted(patient_groups.items()), desc="Loading PTB raw files"):
            recs = []
            p_files = sorted(p_files) 

            # Optional: Filter for Healthy Controls
            if self.only_healthy:
                try:
                    first_header = wfdb.rdheader(str(p_files[0].with_suffix("")))
                    if not self._is_healthy(first_header): continue 
                except: continue

            for hea in p_files:
                try:
                    rec_header = wfdb.rdheader(str(hea.with_suffix("")))
                    lead_indices = self._get_lead_indices(rec_header.sig_name)
                    if not lead_indices: continue
                    
                    data, _ = wfdb.rdsamp(str(hea.with_suffix("")), channels=lead_indices)
                    
                    dt = rec_header.base_date
                    if dt is None: dt = self._parse_date_from_comments(rec_header.comments)
                    if dt is None: dt = datetime.date.min
                    full_dt = datetime.datetime.combine(dt, datetime.time.min)
                    
                    recs.append({"signal": data, "fs": rec_header.fs, "date": full_dt, "filename": hea.name})
                except Exception as e: pass
            
            if recs:
                # Sort chronologically to enable Short/Long term splitting
                recs.sort(key=lambda x: (x["date"], x["filename"]))
                recordings[sid] = recs
        
        if self.filter_subset == "multi_rec_only":
            recordings = {k: v for k, v in recordings.items() if len(v) >= 2}
        return recordings

    def _process_signal(self, sig, fs):
        """Applies filters, segmentation, and normalization."""
        n_channels = sig.shape[1]
        processed_channels = []
        for c in range(n_channels):
            processed_channels.append(self.preprocessor.preprocess_ecg(
                sig[:, c], fs, 
                mode=self.prep_params.get("mode", "beat"),
                window_s=self.prep_params.get("window_len", 5.0),
                stride_s=self.prep_params.get("stride", 1.0),
                pre_s=self.prep_params.get("pre_s", 0.2), post_s=self.prep_params.get("post_s", 0.4),
                filter_method="butter" if self.prep_params.get("bandpass") else None,
                filter_kwargs={'low': self.prep_params.get("lowcut", 0.5), 'high': self.prep_params.get("highcut", 40.0)},
                norm_method="zscore" if self.prep_params.get("normalize") else None
            ))
        if not processed_channels: return np.empty((0, n_channels, 0))
        min_len = min([len(ch) for ch in processed_channels])
        if min_len == 0: return np.empty((0, n_channels, 0))
        
        beats_multi = np.stack([ch[:min_len] for ch in processed_channels], axis=1)
        if self.num_beats == 1:
            return beats_multi[:, 0, :] if n_channels == 1 else beats_multi
        if len(beats_multi) < self.num_beats: return np.empty((0, n_channels, 0))
        
        merged_samples = []
        for i in range(0, len(beats_multi) - self.num_beats + 1):
            group = beats_multi[i : i + self.num_beats]
            if self.merge_strategy == "average":
                merged = np.mean(group, axis=0)
                if n_channels == 1: merged = merged.squeeze(0)
                merged_samples.append(merged)
            elif self.merge_strategy == "concat":
                merged = group.transpose(1, 0, 2).reshape(n_channels, -1)
                if n_channels == 1: merged = merged.squeeze(0)
                merged_samples.append(merged)
        return np.array(merged_samples)

    def load_all_sessions(self):
        """Loads all data without splitting (for Random Split tasks)."""
        data = self.load_raw_data()
        x_list, y_list = [], []
        for sid, recs in tqdm(data.items(), desc="Processing signals"):
            for rec in recs:
                segments = self._process_signal(rec['signal'], rec['fs'])
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))

    def load_session(self, session_name, part="all"):
        """
        Loads Session 1 (Enrollment) or Session 2 (Probe) based on enrollment_mode.
        """
        data = self.load_raw_data()
        x_list, y_list = [], []
        
        is_enrollment = (session_name == "Session_1" or part == "enrollment")
        is_probe = (session_name == "Session_2" or part == "probe")
        is_all = (part == "all")

        for sid, recs in tqdm(data.items(), desc=f"Processing {session_name}"):
            if not recs: continue
            
            target_recs = []
            do_split_signal = False # Flag for single-record intra-split

            # Sort again just to be safe
            recs.sort(key=lambda x: (x["date"], x["filename"]))

            # -----------------------------------------------------------
            # MODE 1: SHORT-TERM (Intra-Day)
            # -----------------------------------------------------------
            # Objective: Test stability within Day 1.
            if self.enrollment_mode == "short_term":
                first_date = recs[0]['date']
                day1_recs = [r for r in recs if r['date'] == first_date]
                
                if len(day1_recs) > 1:
                    # Multi-Record Day 1: Rec 1 vs Rest
                    if is_enrollment: target_recs = [day1_recs[0]]
                    elif is_probe: target_recs = day1_recs[1:]
                elif len(day1_recs) == 1:
                    # Single-Record Day 1: Split the file 50/50
                    target_recs = day1_recs
                    if not is_all: do_split_signal = True
                else:
                    continue

            # -----------------------------------------------------------
            # MODE 2: LONG-TERM (Inter-Day)
            # -----------------------------------------------------------
            # Objective: Test template aging (Day 1 vs Future).
            elif self.enrollment_mode == "long_term":
                dates = sorted(list(set(r['date'] for r in recs)))
                if len(dates) < 2: 
                    continue # Skip subjects with only 1 day of data
                
                first_d = dates[0]
                if is_enrollment:
                    target_recs = [r for r in recs if r['date'] == first_d]
                elif is_probe:
                    target_recs = [r for r in recs if r['date'] > first_d]

            # -----------------------------------------------------------
            # MODE 3: MAXIMAL (First vs Last)
            # -----------------------------------------------------------
            # Objective: Test maximum time gap (Strictly Cross-Day).
            elif self.enrollment_mode == "maximal":
                if len(recs) < 2: continue 
                
                # Ensure First and Last are different days
                if recs[0]['date'] == recs[-1]['date']:
                    continue

                if is_enrollment: target_recs = [recs[0]]
                elif is_probe: target_recs = [recs[-1]]

            # -----------------------------------------------------------
            # Extract Segments
            # -----------------------------------------------------------
            for rec in target_recs:
                segments = self._process_signal(rec['signal'], rec['fs'])
                
                # Handle single-record intra-split logic (first half vs second half)
                if do_split_signal and len(segments) > 0:
                    mid = len(segments) // 2
                    if is_enrollment: segments = segments[:mid]
                    elif is_probe: segments = segments[mid:]
                
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))

        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))

# # =============================================================================
# # 4. CYBHi
# # =============================================================================
class load_cybhi_dataset():
    """
    Loader for CYBHi (Check Your Biosignals Here) Dataset.
    
    Subsets:
    1. 'short-term' (CI-A1-A2): ~3 days gap. Filenames like '20110718-LGM.txt'
    2. 'long-term' (A0): ~3 months gap. Filenames like '20120106-AA.txt'
    """
    def __init__(self, num_beats=1, merge_strategy="average", subset="long-term", 
                 enrollment_mode="standard", preprocessing_params=None, cleanup_zip=False):
        self.preprocessor = Preprocessing()
        self.cfg = CONFIG["datasets"]["cybhi"]
        project_dir = Path(__file__).resolve().parent
        self.data_root = (project_dir / CONFIG["project"]["data_root"]).resolve()
        self.dataset_root = self.data_root / self.cfg["root_dir"]
        self.zip_path = self.data_root / self.cfg["zip_name"]
        self.url = self.cfg["url"]
        
        self.prep_params = preprocessing_params if preprocessing_params else self.cfg.get("preprocessing", {})
        self.subset = subset.lower()
        self.enrollment_mode = enrollment_mode
        self.num_beats = num_beats
        self.merge_strategy = merge_strategy
        self.cleanup_zip = cleanup_zip
        
        if self.subset not in ["short-term", "long-term"]:
            raise ValueError(f"Invalid CYBHi subset: {self.subset}. Use 'short-term' or 'long-term'.")

    def download(self):
        _download_and_extract(self.url, self.zip_path, self.dataset_root, "CYBHi", cleanup=self.cleanup_zip)

    def _parse_filename(self, filename):
        """
        Robust Regex parser for CYBHi filenames.
        Handles:
        - '20120106-AA.txt'  (Long-Term Standard)
        - '20110718-LGM.txt' (Short-Term Standard)
        - '20110718-ST-LGM.txt' (Variant)
        - '2012-01-06-AA.txt' (Hyphenated Date Variant)
        """
        # Strategy 1: Look for 8-digit date followed by ID
        # Matches "20120106" -> sep -> "AA"
        match = re.search(r"(\d{8})[-_]+(?:ST[-_]+)?([A-Za-z]+)", filename)
        
        if match:
            date_str, sid = match.groups()
            try:
                rec_date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
                return rec_date, sid
            except: pass # Fallback if date is invalid

        # Strategy 2: Fallback for hyphenated dates (YYYY-MM-DD)
        # Matches "2012-01-06" -> sep -> "AA"
        match_hyphen = re.search(r"(\d{4}-\d{2}-\d{2})[-_]+(?:ST[-_]+)?([A-Za-z]+)", filename)
        if match_hyphen:
            date_str, sid = match_hyphen.groups()
            try:
                rec_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                return rec_date, sid
            except: pass

        # Strategy 3: Manual split fallback (Last resort)
        # Assumes ID is the LAST alphanumeric part
        clean = filename.replace(".txt", "")
        parts = re.split(r"[-_]", clean)
        sid = parts[-1] if parts else "unknown"
        # Try to find date in parts
        rec_date = datetime.date.min
        for p in parts:
            if len(p) == 8 and p.isdigit():
                try: rec_date = datetime.datetime.strptime(p, "%Y%m%d").date()
                except: pass
        
        return rec_date, sid

    def load_raw_data(self):
        if not self.dataset_root.exists() or not any(self.dataset_root.iterdir()):
            self.download()

        target_dir = None
        possible_names = [self.subset]
        if self.subset == "long-term": possible_names.append("data_A0")
        if self.subset == "short-term": possible_names.append("data_CI_A1_A2")

        for name in possible_names:
            matches = list(self.dataset_root.rglob(name))
            if matches: 
                target_dir = matches[0]
                break
        
        if not target_dir:
            keyword = "A0" if self.subset == "long-term" else "CI"
            for root, dirs, files in os.walk(self.dataset_root):
                for d in dirs:
                    if keyword in d:
                        target_dir = Path(root) / d
                        break
        
        if not target_dir or not target_dir.exists():
            print(f"[ERR] Could not find folder for subset '{self.subset}' in {self.dataset_root}")
            return {}

        files = list(target_dir.glob("*.txt"))
        recordings = {}
        
        for fpath in tqdm(files, desc=f"Loading CYBHi ({self.subset})"):
            try:
                rec_date, sid = self._parse_filename(fpath.name)
                
                # Header skipping
                start_line = 0
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as fp:
                    for i, line in enumerate(fp):
                        if "# EndOfHeader" in line: start_line = i + 1; break
                
                df = pd.read_csv(fpath, skiprows=start_line, sep=r"\s+", header=None, engine='python')
                
                # Column selection
                if df.shape[1] == 1: 
                    sig = df.iloc[:, 0].values.astype(float)
                else:
                    target_col = self.cfg.get("ecg_column", 2) 
                    if target_col >= df.shape[1]: target_col = 0 
                    sig = df.iloc[:, target_col].values.astype(float)

                sig = sig - np.mean(sig)
                
                if sid not in recordings: recordings[sid] = []
                recordings[sid].append({'signal': sig, 'date': rec_date, 'fs': 1000, 'filename': fpath.name})
            except Exception as e: pass
        
        for sid in recordings: recordings[sid].sort(key=lambda x: x['date'])
        return recordings

    def _process_signal(self, sig, fs=1000):
        # Sanity Checks
        if np.isnan(sig).any() or len(sig) < fs or np.std(sig) < 1e-5: 
            return np.empty((0, 0))

        beats = self.preprocessor.preprocess_ecg(
            sig, fs=fs, 
            mode=self.prep_params.get("mode", "beat"),
            window_s=self.prep_params.get("window_len", 5.0),
            stride_s=self.prep_params.get("stride", 1.0),
            pre_s=self.prep_params.get("pre_s", 0.2), post_s=self.prep_params.get("post_s", 0.4),
            filter_method="butter" if self.prep_params.get("bandpass") else None,
            filter_kwargs={'low': self.prep_params.get("lowcut", 0.5), 'high': self.prep_params.get("highcut", 40.0)},
            norm_method="zscore" if self.prep_params.get("normalize") else None
        )
        if self.num_beats == 1: return beats
        if len(beats) < self.num_beats: return np.empty((0, beats.shape[1]))
        
        processed_samples = []
        for i in range(0, len(beats) - self.num_beats + 1):
            group = beats[i : i + self.num_beats]
            if self.merge_strategy == "average": processed_samples.append(np.mean(group, axis=0))
            elif self.merge_strategy == "concat": processed_samples.append(group.flatten())
        return np.array(processed_samples)

    def load_all_sessions(self):
        data = self.load_raw_data()
        x_list, y_list = [], []
        for sid, recs in tqdm(data.items(), desc="Processing signals"):
            for rec in recs:
                segments = self._process_signal(rec['signal'], rec['fs'])
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))

    def load_session(self, session_name, part="all"):
        data = self.load_raw_data()
        x_list, y_list = [], []
        is_enrollment = (session_name == "Session_1" or part == "enrollment")
        is_probe = (session_name == "Session_2" or part == "probe")
        is_all = (part == "all")
        
        for sid, recs in tqdm(data.items(), desc=f"Processing {session_name}"):
            if not recs: continue
            
            dates = sorted(list(set([r['date'] for r in recs])))
            
            if len(dates) > 1:
                first_date = dates[0]
                recs_s1 = [r for r in recs if r['date'] == first_date]
                recs_s2 = [r for r in recs if r['date'] > first_date]
            else:
                recs_s1 = [recs[0]]
                recs_s2 = [recs[0]] 
            
            target_recs = []
            if is_enrollment: target_recs = recs_s1
            elif is_probe: target_recs = recs_s2
            elif is_all: target_recs = recs
            
            for rec in target_recs:
                segments = self._process_signal(rec['signal'], rec['fs'])
                
                # Intra-Session Split logic
                if len(dates) == 1 and len(segments) > 0:
                    mid = len(segments) // 2
                    if is_enrollment: segments = segments[:mid]
                    elif is_probe: segments = segments[mid:]
                
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
        
        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))

# =============================================================================
# 5. MIT-BIH Arrhythmia Database
# =============================================================================
class load_mitbih_dataset():
    """
    Robust Loader for MIT-BIH Arrhythmia Database.
    
    Characteristics:
    - 48 continuous recordings (~30 minutes each).
    - Sample rate: 360 Hz.
    
    Splitting Logic (Range-Based):
    - Uses percentage ranges (0.0 to 1.0) to define Enrollment and Probe sessions.
    - Default: 
        Enroll = First 50% (0.0 - 0.5)
        Probe  = Last 50%  (0.5 - 1.0)
    - Custom Regimes (e.g., Maximal Interval) can be defined by passing specific ranges 
      in __init__ (e.g., first 2 mins vs last 2 mins).
    """
    def __init__(self, leads=['MLII'], num_beats=1, merge_strategy="average", 
                 enrollment_range=(0.0, 0.5), probe_range=(0.5, 1.0),
                 preprocessing_params=None, cleanup_zip=False):
        """
        Args:
            enrollment_range (tuple): (start_pct, end_pct) for Session 1.
            probe_range (tuple): (start_pct, end_pct) for Session 2.
        """
        self.preprocessor = Preprocessing()
        self.cfg = CONFIG["datasets"]["mitbih"]
        project_dir = Path(__file__).resolve().parent
        self.data_root = (project_dir / CONFIG["project"]["data_root"]).resolve()
        self.dataset_root = self.data_root / self.cfg["root_dir"]
        self.zip_path = self.data_root / self.cfg["zip_name"]
        self.url = self.cfg["url"]
        
        self.prep_params = preprocessing_params if preprocessing_params else self.cfg.get("preprocessing", {})
        self.target_leads = [l.lower() for l in leads] if isinstance(leads, list) else leads
        self.num_beats = num_beats
        self.merge_strategy = merge_strategy
        self.cleanup_zip = cleanup_zip
        
        # User defined ranges (0.0 to 1.0)
        self.enrollment_range = enrollment_range
        self.probe_range = probe_range

    def download(self):
        """Downloads and extracts the dataset if missing."""
        _download_and_extract(self.url, self.zip_path, self.dataset_root, "MIT-BIH", cleanup=self.cleanup_zip)

    def _get_lead_indices(self, available_leads):
        """Finds indices for requested leads (e.g., 'MLII')."""
        avail_norm = [l.lower() for l in available_leads]
        if self.target_leads == 'all': return list(range(len(available_leads)))
        indices = []
        for req in self.target_leads:
            req = req.strip().lower()
            if req in avail_norm: indices.append(avail_norm.index(req))
            else:
                for i, avail in enumerate(avail_norm):
                    if req in avail or avail in req: indices.append(i); break
        return indices

    def load_raw_data(self):
        """
        Loads all raw files into memory.
        Note: MIT-BIH files are small (~30 mins), so we load the full signal once 
        and slice it later in memory. This reduces disk I/O overhead.
        """
        if not self.dataset_root.exists() or not any(self.dataset_root.iterdir()):
            self.download()
        recordings = {}
        files = list(self.dataset_root.rglob("*.hea"))
        
        for hea in tqdm(files, desc="Loading MIT-BIH raw files"):
            sid = hea.stem
            try:
                rec_header = wfdb.rdheader(str(hea.with_suffix("")))
                lead_indices = self._get_lead_indices(rec_header.sig_name)
                if not lead_indices: continue
                
                # Load full file
                data, _ = wfdb.rdsamp(str(hea.with_suffix("")), channels=lead_indices)
                
                if sid not in recordings: recordings[sid] = []
                recordings[sid].append({"signal": data, "fs": rec_header.fs, "filename": hea.name})
            except Exception as e: pass
        return recordings

    def _process_signal(self, sig, fs):
        """Standard Preprocessing Pipeline (Filter -> Segment -> Normalize)."""
        n_channels = sig.shape[1]
        processed_channels = []
        for c in range(n_channels):
            processed_channels.append(self.preprocessor.preprocess_ecg(
                sig[:, c], fs, 
                mode=self.prep_params.get("mode", "beat"),
                window_s=self.prep_params.get("window_len", 5.0),
                stride_s=self.prep_params.get("stride", 1.0),
                pre_s=self.prep_params.get("pre_s", 0.2), post_s=self.prep_params.get("post_s", 0.4),
                filter_method="butter" if self.prep_params.get("bandpass") else None,
                filter_kwargs={'low': self.prep_params.get("lowcut", 0.5), 'high': self.prep_params.get("highcut", 40.0)},
                norm_method="zscore" if self.prep_params.get("normalize") else None
            ))
        if not processed_channels: return np.empty((0, n_channels, 0))
        min_len = min([len(ch) for ch in processed_channels])
        if min_len == 0: return np.empty((0, n_channels, 0))
        
        beats_multi = np.stack([ch[:min_len] for ch in processed_channels], axis=1)
        if self.num_beats == 1: return beats_multi[:, 0, :] if n_channels == 1 else beats_multi
        if len(beats_multi) < self.num_beats: return np.empty((0, n_channels, 0))
        
        merged_samples = []
        for i in range(0, len(beats_multi) - self.num_beats + 1):
            group = beats_multi[i : i + self.num_beats]
            if self.merge_strategy == "average":
                merged = np.mean(group, axis=0)
                if n_channels == 1: merged = merged.squeeze(0)
                merged_samples.append(merged)
            elif self.merge_strategy == "concat":
                merged = group.transpose(1, 0, 2).reshape(n_channels, -1)
                if n_channels == 1: merged = merged.squeeze(0)
                merged_samples.append(merged)
        return np.array(merged_samples)

    def load_all_sessions(self):
        """Loads full data (0.0 to 1.0). Useful for Random Split benchmarks."""
        data = self.load_raw_data()
        x_list, y_list = [], []
        for sid, recs in tqdm(data.items(), desc="Processing signals"):
            for rec in recs:
                segments = self._process_signal(rec['signal'], rec['fs'])
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))

    def load_session(self, session_name, part="all"):
        """
        Loads and slices data based on the configured ranges.
        
        Logic:
        1. Preprocess the entire signal.
        2. Calculate start/end indices based on 'enrollment_range' or 'probe_range'.
        3. Slice the list of segments.
        """
        data = self.load_raw_data()
        x_list, y_list = [], []
        
        # Determine which range to use
        target_range = (0.0, 1.0)
        if session_name == "Session_1" or part == "enrollment": 
            target_range = self.enrollment_range
        elif session_name == "Session_2" or part == "probe": 
            target_range = self.probe_range

        for sid, recs in tqdm(data.items(), desc=f"Processing {session_name}"):
            if not recs: continue
            for rec in recs:
                segments = self._process_signal(rec['signal'], rec['fs'])
                if len(segments) == 0: continue
                
                # --- Slicing Logic ---
                N = len(segments)
                start_idx = int(N * target_range[0])
                end_idx = int(N * target_range[1])
                
                # Sanity Check Indices
                start_idx = max(0, start_idx)
                end_idx = min(N, end_idx)
                
                if start_idx >= end_idx: continue

                final_segments = segments[start_idx:end_idx]
                
                if len(final_segments) > 0:
                    x_list.append(final_segments)
                    y_list.extend([sid] * len(final_segments))
        
        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))


# =============================================================================
# 6. MIT-BIH NSRDB (Normal Sinus Rhythm)
# =============================================================================
class load_nsrdb_dataset():
    """
    Loader for MIT-BIH Normal Sinus Rhythm Database (NSRDB).
    
    Characteristics:
    - 18 long-term recordings (~24 hours continuous).
    - Sample rate: 128 Hz.
    
    Optimization (Disk I/O):
    - Unlike MIT-BIH, NSRDB files are huge.
    - This loader calculates the specific sample indices needed for the requested range
      and ONLY reads that slice from disk using `wfdb.rdsamp`.
      
    Default Ranges:
    - Enroll: First Hour (0.0 - 0.0417)
    - Probe:  Last Hour  (0.9583 - 1.0)
    """
    def __init__(self, leads=['ECG1'], num_beats=1, merge_strategy="average", 
                 enrollment_range=(0.0, 0.0417), probe_range=(0.9583, 1.0), 
                 preprocessing_params=None, cleanup_zip=False):
        self.preprocessor = Preprocessing()
        self.cfg = CONFIG["datasets"]["nsrdb"]
        project_dir = Path(__file__).resolve().parent
        self.data_root = (project_dir / CONFIG["project"]["data_root"]).resolve()
        self.dataset_root = self.data_root / self.cfg["root_dir"]
        self.zip_path = self.data_root / self.cfg["zip_name"]
        self.url = self.cfg["url"]
        
        self.prep_params = preprocessing_params if preprocessing_params else self.cfg.get("preprocessing", {})
        self.target_leads = [l.lower() for l in leads] if isinstance(leads, list) else leads
        self.num_beats = num_beats
        self.merge_strategy = merge_strategy
        self.cleanup_zip = cleanup_zip
        
        # User defined ranges (0.0 to 1.0)
        self.enrollment_range = enrollment_range
        self.probe_range = probe_range

    def download(self):
        """Downloads and extracts dataset."""
        _download_and_extract(self.url, self.zip_path, self.dataset_root, "NSRDB", cleanup=self.cleanup_zip)

    def _get_lead_indices(self, available_leads):
        """Finds channel indices."""
        avail_norm = [l.lower() for l in available_leads]
        if self.target_leads == 'all': return list(range(len(available_leads)))
        indices = []
        for req in self.target_leads:
            req = req.strip().lower()
            if req in avail_norm: indices.append(avail_norm.index(req))
            else:
                for i, avail in enumerate(avail_norm):
                    if req in avail: indices.append(i); break
        return indices

    def load_raw_data(self, start_ratio=0.0, end_ratio=1.0):
        """
        Loads ONLY the specified time slice from disk to save memory.
        Calculates 'sampfrom' and 'sampto' based on total file duration.
        """
        if not self.dataset_root.exists() or not any(self.dataset_root.iterdir()):
            self.download()
        recordings = {}
        files = list(self.dataset_root.rglob("*.hea"))
        
        for hea in tqdm(files, desc="Loading NSRDB raw files"):
            sid = hea.stem
            try:
                # 1. Read Header first (Lightweight) to get total length
                rec_header = wfdb.rdheader(str(hea.with_suffix("")))
                total_samples = rec_header.sig_len
                lead_indices = self._get_lead_indices(rec_header.sig_name)
                if not lead_indices: continue

                # 2. Calculate specific sample indices based on requested ratios
                sampfrom = int(total_samples * start_ratio)
                sampto = int(total_samples * end_ratio)
                
                # Validations
                if sampto > total_samples: sampto = total_samples
                if sampto <= sampfrom: continue 
                
                # 3. Load EFFICIENT slice directly from disk
                data, _ = wfdb.rdsamp(str(hea.with_suffix("")), channels=lead_indices, sampfrom=sampfrom, sampto=sampto)
                
                if sid not in recordings: recordings[sid] = []
                recordings[sid].append({"signal": data, "fs": rec_header.fs, "filename": hea.name})
            except Exception as e: pass
        return recordings

    def _process_signal(self, sig, fs):
        """Standard Preprocessing."""
        n_channels = sig.shape[1]
        processed_channels = []
        for c in range(n_channels):
            processed_channels.append(self.preprocessor.preprocess_ecg(
                sig[:, c], fs, 
                mode=self.prep_params.get("mode", "beat"),
                window_s=self.prep_params.get("window_len", 5.0),
                stride_s=self.prep_params.get("stride", 1.0),
                pre_s=self.prep_params.get("pre_s", 0.2), post_s=self.prep_params.get("post_s", 0.4),
                filter_method="butter" if self.prep_params.get("bandpass") else None,
                filter_kwargs={'low': self.prep_params.get("lowcut", 0.5), 'high': self.prep_params.get("highcut", 40.0)},
                norm_method="zscore" if self.prep_params.get("normalize") else None
            ))
        if not processed_channels: return np.empty((0, n_channels, 0))
        min_len = min([len(ch) for ch in processed_channels])
        if min_len == 0: return np.empty((0, n_channels, 0))
        
        beats_multi = np.stack([ch[:min_len] for ch in processed_channels], axis=1)
        if self.num_beats == 1: return beats_multi[:, 0, :] if n_channels == 1 else beats_multi
        if len(beats_multi) < self.num_beats: return np.empty((0, n_channels, 0))
        
        merged_samples = []
        for i in range(0, len(beats_multi) - self.num_beats + 1):
            group = beats_multi[i : i + self.num_beats]
            if self.merge_strategy == "average":
                merged = np.mean(group, axis=0)
                if n_channels == 1: merged = merged.squeeze(0)
                merged_samples.append(merged)
            elif self.merge_strategy == "concat":
                merged = group.transpose(1, 0, 2).reshape(n_channels, -1)
                if n_channels == 1: merged = merged.squeeze(0)
                merged_samples.append(merged)
        return np.array(merged_samples)

    def load_all_sessions(self):
        """Loads FULL 24h files (Warning: High Memory Usage)."""
        data = self.load_raw_data(start_ratio=0.0, end_ratio=1.0)
        x_list, y_list = [], []
        for sid, recs in tqdm(data.items(), desc="Processing signals"):
            for rec in recs:
                segments = self._process_signal(rec['signal'], rec['fs'])
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))

    def load_session(self, session_name, part="all"):
        """
        Loads optimized slice based on configured ranges.
        Does NOT load the whole file; only the needed slice.
        """
        x_list, y_list = [], []
        
        # Determine target range
        target_range = (0.0, 1.0)
        if session_name == "Session_1" or part == "enrollment": 
            target_range = self.enrollment_range
        elif session_name == "Session_2" or part == "probe": 
            target_range = self.probe_range
        
        # KEY OPTIMIZATION: Pass the range to load_raw_data
        # This ensures we only read the specific bytes from disk.
        data = self.load_raw_data(start_ratio=target_range[0], end_ratio=target_range[1])
        
        for sid, recs in tqdm(data.items(), desc=f"Processing {session_name}"):
            if not recs: continue
            for rec in recs:
                segments = self._process_signal(rec['signal'], rec['fs'])
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
                    
        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))

# =============================================================================
# 7. PTB-XL (Physikalisch-Technische Bundesanstalt XL)
# =============================================================================
class load_ptbxl_dataset():
    """
    Robust Loader for PTB-XL Dataset.
    
    Characteristics:
    - ~21,800 records from ~18,800 patients.
    - Duration: EXACTLY 10 seconds per record.
    - Sample Rate: 500 Hz (High) or 100 Hz (Low).
    
    Biometric Regimes:
    
    1. Very Short-Term (Intra-Record):
       - Target: ALL subjects.
       - Constraint: Most subjects have only one 10s record.
       - Method: Split the 10s file into two 5s halves.
         * Enroll: 0-5 seconds.
         * Probe:  5-10 seconds.
         
    2. Long-Term (Inter-Record):
       - Target: Multi-Record subjects only.
       - Method: 
         * Enroll: The very first recording (Rec 1).
         * Probe:  All subsequent recordings (Rec 2, 3...).
    """
    def __init__(self, leads=['II'], resolution='high', filter_subset="all", 
                 num_beats=1, merge_strategy="average", limit_records=None, 
                 preprocessing_params=None, cleanup_zip=False):
        self.preprocessor = Preprocessing()
        self.cfg = CONFIG["datasets"]["ptbxl"]
        project_dir = Path(__file__).resolve().parent
        self.data_root = (project_dir / CONFIG["project"]["data_root"]).resolve()
        self.dataset_root = self.data_root / self.cfg["root_dir"]
        self.zip_path = self.data_root / self.cfg["zip_name"]
        self.url = self.cfg["url"]
        
        self.prep_params = preprocessing_params if preprocessing_params else self.cfg.get("preprocessing", {})
        self.target_leads = [l.lower() for l in leads] if isinstance(leads, list) else leads
        self.resolution = resolution
        self.filter_subset = filter_subset
        self.num_beats = num_beats
        self.merge_strategy = merge_strategy
        self.limit_records = limit_records
        self.cleanup_zip = cleanup_zip

    def download(self):
        _download_and_extract(self.url, self.zip_path, self.dataset_root, "PTB-XL", cleanup=self.cleanup_zip)

    def _get_lead_indices(self, available_leads):
        avail_norm = [l.lower() for l in available_leads]
        if self.target_leads == 'all': return list(range(len(available_leads)))
        indices = []
        for req in self.target_leads:
            req = req.strip().lower()
            if req in avail_norm: indices.append(avail_norm.index(req))
            else:
                for i, avail in enumerate(avail_norm):
                    if req == avail: indices.append(i); break
        return indices

    def load_raw_data(self):
        if not self.dataset_root.exists() or not any(self.dataset_root.iterdir()):
            self.download()
        
        csv_path = self.dataset_root / 'ptbxl_database.csv'
        if not csv_path.exists(): raise FileNotFoundError(f"Database CSV not found at {csv_path}")
        
        df = pd.read_csv(csv_path, index_col='ecg_id')
        df['patient_id'] = df['patient_id'].astype(int)
        recordings = {}
        unique_patients = df['patient_id'].unique()
        
        if self.limit_records: 
            unique_patients = unique_patients[:self.limit_records]
        
        fname_col = 'filename_hr' if self.resolution == 'high' else 'filename_lr'

        for pid in tqdm(unique_patients, desc="Loading PTB-XL raw files"):
            # Sort chronologically is CRITICAL for Long-Term regime
            patient_recs = df[df['patient_id'] == pid].sort_values(by='recording_date')
            
            # Filter: If we only want multi-record subjects globally
            if self.filter_subset == "multi_rec_only" and len(patient_recs) < 2: continue

            recs_list = []
            for ecg_id, row in patient_recs.iterrows():
                fname_rel = row[fname_col]
                full_path = self.dataset_root / fname_rel
                
                if not (full_path.parent / (full_path.name + ".hea")).exists(): continue

                try:
                    rec_header = wfdb.rdheader(str(full_path))
                    lead_indices = self._get_lead_indices(rec_header.sig_name)
                    if not lead_indices: continue
                    
                    data, _ = wfdb.rdsamp(str(full_path), channels=lead_indices)
                    recs_list.append({"signal": data, "fs": rec_header.fs})
                except Exception as e: pass
            
            if recs_list: recordings[str(pid)] = recs_list
            
        return recordings

    def _process_signal(self, sig, fs):
        """Standard Preprocessing."""
        n_channels = sig.shape[1]
        processed_channels = []
        for c in range(n_channels):
            processed_channels.append(self.preprocessor.preprocess_ecg(
                sig[:, c], fs, 
                mode=self.prep_params.get("mode", "beat"),
                window_s=self.prep_params.get("window_len", 5.0),
                stride_s=self.prep_params.get("stride", 1.0),
                pre_s=self.prep_params.get("pre_s", 0.2), post_s=self.prep_params.get("post_s", 0.4),
                filter_method="butter" if self.prep_params.get("bandpass") else None,
                filter_kwargs={'low': self.prep_params.get("lowcut", 0.5), 'high': self.prep_params.get("highcut", 40.0)},
                norm_method="zscore" if self.prep_params.get("normalize") else None
            ))
        if not processed_channels: return np.empty((0, n_channels, 0))
        min_len = min([len(ch) for ch in processed_channels])
        if min_len == 0: return np.empty((0, n_channels, 0))
        
        beats_multi = np.stack([ch[:min_len] for ch in processed_channels], axis=1)
        if self.num_beats == 1: return beats_multi[:, 0, :] if n_channels == 1 else beats_multi
        if len(beats_multi) < self.num_beats: return np.empty((0, n_channels, 0))
        
        merged_samples = []
        for i in range(0, len(beats_multi) - self.num_beats + 1):
            group = beats_multi[i : i + self.num_beats]
            if self.merge_strategy == "average":
                merged = np.mean(group, axis=0)
                if n_channels == 1: merged = merged.squeeze(0)
                merged_samples.append(merged)
            elif self.merge_strategy == "concat":
                merged = group.transpose(1, 0, 2).reshape(n_channels, -1)
                if n_channels == 1: merged = merged.squeeze(0)
                merged_samples.append(merged)
        return np.array(merged_samples)

    def load_all_sessions(self):
        """Loads all data without splitting (for Random Split)."""
        data = self.load_raw_data()
        x_list, y_list = [], []
        for sid, recs in tqdm(data.items(), desc="Processing signals"):
            for rec in recs:
                segments = self._process_signal(rec['signal'], rec['fs'])
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))

    def load_session(self, session_name, split_mode="short_term"):
        """
        Loads data based on the regime.
        
        Args:
            split_mode (str):
             - 'short_term': Intra-Record Split (5s vs 5s).
               Used for Row 1 (All Subjects).
               Splits the 10s recording into First Half (0-5s) and Second Half (5-10s).
               
             - 'long_term': Inter-Record Split.
               Used for Row 2 (Multi-Record Subjects).
               Enroll = Rec 1. Probe = Rec 2+.
        """
        data = self.load_raw_data()
        x_list, y_list = [], []
        is_enrollment = (session_name == "Session_1")
        is_probe = (session_name == "Session_2")

        for sid, recs in tqdm(data.items(), desc=f"Processing {session_name}"):
            if not recs: continue
            
            target_recs = []
            intra_split = False 
            
            # --- LONG-TERM (Inter-Record) ---
            if split_mode == "long_term":
                # Must have multiple records to simulate aging
                if len(recs) < 2: continue
                
                if is_enrollment: target_recs = [recs[0]]     # Earliest Record
                elif is_probe: target_recs = recs[1:]         # All Later Records
                
                intra_split = False 
            
            # --- Very SHORT-TERM (Intra-Record 5s vs 5s) ---
            elif split_mode == "short_term":
                # We use ALL records (even if subject only has 1)
                target_recs = recs
                intra_split = True

            # --- PROCESS & SPLIT ---
            for rec in target_recs:
                segments = self._process_signal(rec['signal'], rec['fs'])
                
                # Apply 5s vs 5s split logic
                if intra_split and len(segments) > 0:
                    mid = len(segments) // 2
                    if is_enrollment: segments = segments[:mid]  # 0-5s
                    elif is_probe: segments = segments[mid:]     # 5-10s
                
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))

        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))
# =============================================================================