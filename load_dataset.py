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
    Robust Loader for the ECG-ID Database.
    Handles automatic downloading, parsing via WFDB, and filtering.

    This dataset consists of 310 recordings from 90 subjects. Recordings vary 
    in number per subject (from 2 to 20) and are taken over different days.

    Args:
        num_beats_to_merge (int): Number of consecutive beats to fuse into a single sample. 
            Default is 1 (no fusion).
        beat_merge_method (str): Strategy for fusing beats if `num_beats_to_merge` > 1.
            Options: 
                - 'average': Averages the morphology of N beats.
                - 'concat': Flattens N beats into a single continuous vector.
        data_split_mode (str): Evaluation regime mapping to strictly partition records.
            Options:
                - 'all-available': Loads every record (used for random beat-level splitting).
                - 'single-session': Loads ONLY the 1st record of each subject.
                - 'single-cross-session': 1st record = Train/Enroll, 2nd record = Test/Probe.
                - 'single-shot-short-term': Day 1's 1st record = Enroll, rest of Day 1 = Probe.
                - 'leave-last-out-short-term': Day 1's last record = Probe, rest of Day 1 = Enroll.
                - 'single-shot-long-term': All Day 1 records = Enroll, all future days = Probe.
                - 'leave-last-out-long-term': Last recording day = Probe, all past days = Enroll.
        signal_type (str): Which WFDB channel to extract.
            Options:
                - 'raw': Extracts the noisy/unfiltered channel (idx 0).
                - 'filtered': Extracts the hardware-filtered channel (idx 1).
        cleanup_zip (bool): If True, deletes the downloaded zip file after extraction.
        **preprocessing_params: kwargs passed directly to the Preprocessing class.
            Common options: mode='beat'|'blind', bandpass=True, normalize=True, window_len=5.0
    """
    def __init__(self, num_beats_to_merge=1, beat_merge_method="average", 
                 data_split_mode="all-available", signal_type="raw", 
                 cleanup_zip=False, **preprocessing_params):
        
        self.preprocessor = Preprocessing()
        self.cfg = CONFIG["datasets"]["ecgid"]
        project_dir = Path(__file__).resolve().parent
        self.data_root = (project_dir / CONFIG["project"]["data_root"]).resolve()
        self.dataset_root = self.data_root / self.cfg["root_dir"]
        self.zip_path = self.data_root / self.cfg["zip_name"]
        self.url = self.cfg["url"]
        
        self.signal_type = "noisy" if signal_type == "raw" else "filtered"
        self.prep_params = preprocessing_params if preprocessing_params else self.cfg.get("preprocessing", {})
        self.num_beats = num_beats_to_merge
        self.merge_strategy = beat_merge_method
        self.cleanup_zip = cleanup_zip
        
        valid_modes = [
            "all-available", "single-session", "single-cross-session", 
            "single-shot-short-term", "leave-last-out-short-term", 
            "single-shot-long-term", "leave-last-out-long-term"
        ]
        if data_split_mode not in valid_modes:
            raise ValueError(f"Invalid mode: {data_split_mode}. Use {valid_modes}")
        self.data_split_mode = data_split_mode

    def download(self):
        _download_and_extract(self.url, self.zip_path, self.dataset_root, "ECG-ID", cleanup=self.cleanup_zip)

    def _extract_rec_number(self, filename):
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
                try:
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

                    channel_idx = 0 if self.signal_type == "raw" else 1
                    
                    # Safety check just in case a file has only 1 channel
                    if record.p_signal.shape[1] <= channel_idx:
                        channel_idx = 0

                    recs.append({
                        'signal': record.p_signal[:, channel_idx], 
                        'date': rec_date,
                        'fs': record.fs,
                        'filename': hea_path.name
                    })
                except Exception as e: pass

            recs.sort(key=lambda x: (x['date'], self._extract_rec_number(x['filename'])))
            recordings[sid] = recs
            
        return recordings

    def _process_signal(self, sig, fs):
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

    def load_all_data(self):
        """
        Loads dataset for tasks that handle train/test splitting downstream.
        Applies to 'all-available' and 'single-session'.
        """
        if self.data_split_mode not in ["all-available", "single-session"]:
            print(f"[WARN] Calling load_all_data() but mode is '{self.data_split_mode}'.")
            
        data = self.load_raw_data()
        x_list, y_list = [], []
        
        for sid, recs in tqdm(data.items(), desc="Processing signals"):
            if not recs: continue
            
            target_recs = recs if self.data_split_mode == "all-available" else [recs[0]]
            
            for rec in target_recs:
                segments = self._process_signal(rec['signal'], rec['fs'])
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
                    
        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))

    def load_session(self, session_name):
        """
        Loads the partitioned data strictly based on temporal/record boundaries.
        Applies to cross-session and short/long-term tasks.
        """
        session_name = session_name.lower()
        if session_name in ["enrol", "train"]:
            is_enrollment = True
            log_name = "Train/Enrollment"
        elif session_name in ["probe", "test"]:
            is_enrollment = False
            log_name = "Test/Probe"
        else:
            raise ValueError("session_name must be 'enrol', 'train', 'probe', or 'test'.")
            
        data = self.load_raw_data()
        x_list, y_list = [], []
        
        kept_subjects, dropped_subjects = 0, 0

        for sid, recs in tqdm(data.items(), desc=f"Processing {log_name}"):
            if not recs: continue
            
            target_recs = []
            
            unique_dates = sorted(list(set(r['date'] for r in recs)))
            day1_date = unique_dates[0]
            day1_recs = [r for r in recs if r['date'] == day1_date]

            # --- TASK 3: SINGLE CROSS-SESSION ---
            if self.data_split_mode == "single-cross-session":
                if len(recs) < 2:
                    dropped_subjects += 1
                    continue

                kept_subjects += 1
                target_recs = [recs[0]] if is_enrollment else [recs[1]]

            # --- TASK 4: SINGLE-SHOT SHORT-TERM ---
            elif self.data_split_mode == "single-shot-short-term":
                if len(day1_recs) < 2:
                    dropped_subjects += 1
                    continue

                kept_subjects += 1
                target_recs = (
                    [day1_recs[0]]
                    if is_enrollment
                    else day1_recs[1:]
                )

            # --- TASK 5: LEAVE-LAST-OUT SHORT-TERM ---
            elif self.data_split_mode == "leave-last-out-short-term":
                if len(day1_recs) < 2:
                    dropped_subjects += 1
                    continue

                kept_subjects += 1
                target_recs = (
                    day1_recs[:-1]
                    if is_enrollment
                    else [day1_recs[-1]]
                )

            # --- TASK 6: SINGLE-SHOT LONG-TERM ---
            elif self.data_split_mode == "single-shot-long-term":
                if len(unique_dates) < 2: 
                    dropped_subjects += 1
                    continue
                kept_subjects += 1
                target_recs = day1_recs if is_enrollment else [r for r in recs if r['date'] > day1_date]

            # --- TASK 7: LEAVE-LAST-OUT LONG-TERM ---
            elif self.data_split_mode == "leave-last-out-long-term":
                if len(unique_dates) < 2:
                    dropped_subjects += 1
                    continue
                kept_subjects += 1
                last_date = unique_dates[-1]
                target_recs = [r for r in recs if r['date'] < last_date] if is_enrollment else [r for r in recs if r['date'] == last_date]

            # --- EXTRACTION & SPLITTING ---
            for rec in target_recs:
                segments = self._process_signal(rec['signal'], rec['fs'])
                
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))

        # Dynamic summary print for all structured tasks during enrollment
        if self.data_split_mode not in ["all-available", "single-session"] and is_enrollment:
            mode_title = self.data_split_mode.replace('-', ' ').title()
            print(f"\n[INFO] {mode_title} Summary: Kept {kept_subjects} subjects. Dropped {dropped_subjects} subjects.")

        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))

# =============================================================================
# 2. HeartPrint
# =============================================================================
class load_heartprint_dataset():
    """
    Dynamic Loader for the HeartPrint Dataset.
    
    HeartPrint is highly structured around distinct physiological and temporal sessions.
    This loader enforces strict mathematical intersection—a subject is only kept if they 
    possess valid data in ALL requested target sessions.

    Session Tags Available for Mapping:
      - 'session1'  (Baseline / Rest)
      - 'session2'  (Rest / Short-Term follow up)
      - 'session3r' (Reading Task / Cognitive State Change)
      - 'session3l' (Very Long-Term / Maximal Time Gap)

    Args:
        data_split_mode (str): The routing logic for data extraction.
            Options:
                - 'single-session': Extracts the exact sessions defined in `session_for_single_session_evaluation` 
                                    and pools them for downstream random-splitting.
                - 'cross-session': Maps data strictly to Train/Enroll/Probe groups based on the session arguments below.
        session_for_single_session_evaluation (str or list): Target session(s) to load if mode is 'single-session'.
            Example: 'session1' or ['session1', 'session2']
        train_sessions (str or list): Session(s) to load when requesting representation learning data. Can be None.
            Example: 'session1'
        enroll_sessions (str or list): Session(s) to load when requesting Gallery enrollment data. Can be None.
            Example: 'session1'
        probe_sessions (str or list): Session(s) to load when requesting Test query data. Can be None.
            Example: 'session3l'
        num_beats_to_merge (int): Number of consecutive beats to fuse into a single sample.
        beat_merge_method (str): Strategy for fusing beats. Options: ['average', 'concat']
        cleanup_zip (bool): If True, deletes the downloaded zip file after extraction.
        **preprocessing_params: kwargs passed directly to the Preprocessing class.
    """
    def __init__(self, data_split_mode="cross-session", 
                 session_for_single_session_evaluation=["session1"],
                 train_sessions=["session1"], 
                 enroll_sessions=["session1"],
                 probe_sessions=["session2"], 
                 num_beats_to_merge=1, beat_merge_method="average", 
                 cleanup_zip=False, **preprocessing_params):
       
        # --- KWARGS GUARD ---
        allowed_prep_kwargs = ["mode", "window_s", "stride_s", "pre_s", "post_s", "bandpass", "lowcut", "highcut", "normalize"]
        for k in preprocessing_params.keys():
            if k not in allowed_prep_kwargs:
                raise ValueError(
                    f"\n[ERROR] Unrecognized parameter: '{k}'.\n"
                    f"Did you misspell a class argument? (e.g., using 'enrol_sessions' instead of 'enroll_sessions')\n"
                    f"Allowed preprocessing kwargs: {allowed_prep_kwargs}"
                )

        self.preprocessor = Preprocessing()
        self.cfg = CONFIG["datasets"]["heartprint"]
        project_dir = Path(__file__).resolve().parent
        self.data_root = (project_dir / CONFIG["project"]["data_root"]).resolve()
        self.dataset_root = self.data_root / self.cfg["root_dir"]
        self.zip_path = self.data_root / self.cfg["zip_name"]
        self.url = self.cfg["url"]
        
        self.sample_len = self.cfg.get("sample_length", 3747)
        self.prep_params = preprocessing_params if preprocessing_params else self.cfg.get("preprocessing", {})
        self.num_beats = num_beats_to_merge
        self.merge_strategy = beat_merge_method
        self.cleanup_zip = cleanup_zip
        
        valid_modes = ["single-session", "cross-session"]
        if data_split_mode not in valid_modes:
            raise ValueError(f"Invalid mode: {data_split_mode}. Use {valid_modes}")
        self.data_split_mode = data_split_mode
        
        # Safely convert strings to lists and handle None
        to_list = lambda x: [x] if isinstance(x, str) else (x if x else [])
        
        self.session_for_single_session_evaluation = to_list(session_for_single_session_evaluation)
        self.train_sessions = to_list(train_sessions)
        self.enroll_sessions = to_list(enroll_sessions)
        self.probe_sessions = to_list(probe_sessions)
        
        # Enforce exact naming to match folder architectures
        self._normalize_sessions(self.session_for_single_session_evaluation)
        self._normalize_sessions(self.train_sessions)
        self._normalize_sessions(self.enroll_sessions)
        self._normalize_sessions(self.probe_sessions)
        
        self.required_cross_sessions = list(set(self.train_sessions + self.enroll_sessions + self.probe_sessions))
        
        if self.data_split_mode == "single-session" and not self.session_for_single_session_evaluation:
            raise ValueError("You must provide `session_for_single_session_evaluation`.")
        if self.data_split_mode == "cross-session" and not self.required_cross_sessions:
            raise ValueError("You must provide at least one valid train, enroll, or probe session.")

    def _normalize_sessions(self, session_list):
        """Standardizes session strings to match directory keys natively."""
        for i in range(len(session_list)):
            s = session_list[i].lower().replace("-", "").replace("_", "").replace(" ", "")
            session_list[i] = s

    def download(self):
        """Attempts robust download via Figshare API."""
        if self.dataset_root.exists():
            for root, dirs, files in os.walk(self.dataset_root):
                for d in dirs:
                    if "session" in d.lower(): return 

        self.dataset_root.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Attempting to download HeartPrint...")
        
        try:
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
                    with open(self.zip_path, "wb") as f, tqdm(desc="Downloading", total=size, unit='iB', unit_scale=True) as bar:
                        for chunk in r.iter_content(8192): f.write(chunk); bar.update(len(chunk))

            print(f"[INFO] Attempting extraction...")
            with tempfile.TemporaryDirectory() as temp_dir:
                try: patoolib.extract_archive(str(self.zip_path), outdir=temp_dir)
                except Exception:
                    with zipfile.ZipFile(self.zip_path, "r") as zf: zf.extractall(temp_dir)
                for item in os.listdir(temp_dir): shutil.move(os.path.join(temp_dir, item), self.dataset_root)
            if self.cleanup_zip: os.remove(self.zip_path)

        except Exception as e:
            print(f"[WARN] Automated download failed: {e}")
            print("Please download manually and extract to:", self.dataset_root)

    def load_raw_data(self):
        """
        Scans the HeartPrint directory and maps valid text files to their explicit Session and Subject ID.
        """
        if not self.dataset_root.exists() or not any(self.dataset_root.iterdir()):
            self.download()

        print("\n[INFO] Scanning directories and pooling HeartPrint files...")
        session_dirs = {}
        
        # 1. Identify the core session folders safely
        for root, dirs, files in os.walk(self.dataset_root):
            for d in dirs:
                d_norm = d.lower().replace("-", "").replace("_", "").replace(" ", "")
                if "session1" in d_norm: session_dirs["session1"] = Path(root) / d
                elif "session2" in d_norm: session_dirs["session2"] = Path(root) / d
                elif "session3r" in d_norm or ("session3" in d_norm and "r" in d_norm): session_dirs["session3r"] = Path(root) / d
                elif "session3l" in d_norm or ("session3" in d_norm and "l" in d_norm): session_dirs["session3l"] = Path(root) / d

        recordings = {}
        
        # 2. Extract records, ensuring we skip hidden macOS files
        for session_tag, sess_path in session_dirs.items():
            for sid in tqdm(os.listdir(sess_path), desc=f"Loading {session_tag.upper()} raw files"):
                subj_path = sess_path / sid
                if not subj_path.is_dir(): continue
                
                if sid not in recordings: recordings[sid] = {}
                if session_tag not in recordings[sid]: recordings[sid][session_tag] = []
                
                for f in os.listdir(subj_path):
                    if not f.endswith(".txt") or f.startswith("._"): continue
                    
                    fpath = subj_path / f
                    try:
                        # High-Speed parsing: Bypasses headers and stops at the sample length
                        df = pd.read_csv(fpath, comment='#', delim_whitespace=True, header=None, nrows=self.sample_len, on_bad_lines='skip')
                        if not df.empty:
                            sig = df.iloc[:, 0].dropna().values.astype(float)
                            sig = sig - np.mean(sig) # Zero-mean baseline
                            recordings[sid][session_tag].append({'signal': sig, 'fs': 250})
                    except Exception as e:
                        print(f"\n[WARN] Failed to read {f}: {e}")
                        
        return recordings

    def _process_signal(self, sig, fs=250):
        """Applies filters, segmentation, and multi-beat merging."""
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

    def load_all_data(self):
        """Safely routes generic all-data requests to the single-session logic."""
        if self.data_split_mode != "single-session":
            print(f"[WARN] Calling load_all_data() but mode is '{self.data_split_mode}'.")
        return self.load_session("train")

    def load_session(self, session_name):
        """
        Extracts requested sessions.
        In 'cross-session' mode, enforces strict mathematical intersection, ensuring a 
        subject is present across ALL specified train, enroll, and probe configurations.
        """
        session_name = session_name.lower()
        target_sessions = []
        is_primary_pass = False
        
        if self.data_split_mode == "single-session":
            if session_name in ["probe", "test"]:
                raise ValueError("Cannot load 'test' in single-session mode. Split upstream.")
            target_sessions = self.session_for_single_session_evaluation
            log_name = f"Single-Session Target(s): {target_sessions}"
            is_primary_pass = True
            
        elif self.data_split_mode == "cross-session":
            if session_name in ["train"]:
                target_sessions = self.train_sessions
                is_primary_pass = True
            elif session_name in ["enrol", "enrollment"]:
                target_sessions = self.enroll_sessions
                is_primary_pass = True if not self.train_sessions else False
            elif session_name in ["probe", "test"]:
                target_sessions = self.probe_sessions
            else:
                raise ValueError("session_name must be 'train', 'enrol', or 'test'.")
            log_name = f"Cross-Session ({session_name.title()}): {target_sessions}"

        if not target_sessions:
            return np.empty((0, 0)), np.empty((0,))
            
        data = self.load_raw_data()
        x_list, y_list = [], []
        
        kept_subjects, dropped_subjects = 0, 0

        for sid, tagged_sessions in tqdm(data.items(), desc=f"Processing {log_name}"):
            
            # --- STRICT INTERSECTION LOGIC ---
            if self.data_split_mode == "single-session":
                is_valid = all(s in tagged_sessions for s in target_sessions)
            elif self.data_split_mode == "cross-session":
                # Strict: Subject MUST have data in ALL requested global sets
                is_valid = all(s in tagged_sessions for s in self.required_cross_sessions)

            if is_valid:
                kept_subjects += 1
                for s in target_sessions:
                    # HeartPrint frequently has 2 to 6 files per session. This naturally pools them!
                    for signal_dict in tagged_sessions[s]:
                        segments = self._process_signal(signal_dict['signal'], signal_dict['fs'])
                        if len(segments) > 0:
                            x_list.append(segments)
                            y_list.extend([sid] * len(segments))
            else:
                dropped_subjects += 1

        if is_primary_pass:
            print(f"\n[INFO] HeartPrint Evaluation Summary ({self.data_split_mode.title()}):")
            print(f"       Kept {kept_subjects} mathematically matched subjects. Dropped {dropped_subjects} subjects due to missing session data.")

        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))

# =============================================================================
# 3. PTB (Physikalisch-Technische Bundesanstalt)
# =============================================================================
class load_ptb_dataset():
    """
    Robust Loader for the PTB Diagnostic ECG Database.
    Handles multi-lead parsing, clinical header filtering, and chronologically mapped biometric tasks.

    This dataset contains 549 records from 290 subjects. Many subjects have 
    severe clinical pathologies (e.g., Myocardial Infarction). 

    Args:
        leads (list of str): Target ECG leads to extract. 
            Options: Any valid 12-lead string (e.g., ['i', 'v5', 'ii']) or 'all' for all 15 available channels.
            Default: ['i']
        data_split_mode (str): Evaluation regime mapping.
            Options:
                - 'all-available': Loads every record (used for random beat-level splitting).
                - 'single-session': Loads ONLY the 1st record of each subject.
                - 'single-cross-session': 1st record = Train/Enroll, 2nd record = Test/Probe.
                - 'single-shot-short-term': Day 1's 1st record = Enroll, rest of Day 1 = Probe.
                - 'leave-last-out-short-term': Day 1's last record = Probe, rest of Day 1 = Enroll.
                - 'single-shot-long-term': All Day 1 records = Enroll, all future days = Probe.
                - 'leave-last-out-long-term': Last recording day = Probe, all past days = Enroll.
        only_healthy (bool): If True, strictly drops subjects with clinical pathologies, 
                             keeping only the ~52 healthy control volunteers.
        num_beats_to_merge (int): Number of consecutive beats to fuse into a single sample.
        beat_merge_method (str): Strategy for fusing beats. Options: ['average', 'concat']
        cleanup_zip (bool): If True, deletes the downloaded zip file after extraction.
        **preprocessing_params: kwargs passed directly to the Preprocessing class.
    """
    def __init__(self, leads=['i'], data_split_mode="all-available",
                 only_healthy=False, num_beats_to_merge=1, beat_merge_method="average", 
                 cleanup_zip=False, **preprocessing_params):
       
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
        self.num_beats = num_beats_to_merge
        self.merge_strategy = beat_merge_method
        self.cleanup_zip = cleanup_zip
        
        valid_modes = [
            "all-available", "single-session", "single-cross-session", 
            "single-shot-short-term", "leave-last-out-short-term", 
            "single-shot-long-term", "leave-last-out-long-term"
        ]
        if data_split_mode not in valid_modes:
            raise ValueError(f"Invalid mode: {data_split_mode}. Use {valid_modes}")
        self.data_split_mode = data_split_mode

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
        """Maps requested lead names (e.g., 'i', 'v5') to channel indices."""
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
        Injects synthetic dates for records missing timestamps to preserve them for evaluation.
        """
        if not self.dataset_root.exists() or not any(self.dataset_root.iterdir()):
            self.download()
        
        recordings = {}
        files = list(self.dataset_root.rglob("*.hea"))
        
        # Group files by patient folder (e.g., patient001)
        patient_groups = {}
        for f in files:
            pid = f.parent.name
            if pid not in patient_groups: patient_groups[pid] = []
            patient_groups[pid].append(f)

        # Base date for synthetic injection
        dummy_date = datetime.date(2099, 1, 1)

        for sid, p_files in tqdm(sorted(patient_groups.items()), desc="Loading PTB raw files"):
            recs = []
            p_files = sorted(p_files) 

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
                    
                    # Assign a synthetic sequential date if none is found
                    if dt is None:
                        dt = dummy_date
                        dummy_date += datetime.timedelta(days=1)
                        
                    full_dt = datetime.datetime.combine(dt, datetime.time.min)
                    recs.append({"signal": data, "fs": rec_header.fs, "date": full_dt, "filename": hea.name})
                except Exception: pass
            
            if recs:
                recs.sort(key=lambda x: (x["date"], x["filename"]))
                recordings[sid] = recs
        
        return recordings

    def _process_signal(self, sig, fs):
        """Applies filters, segmentation, and multi-beat merging."""
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

    def load_all_data(self):
        """
        Loads dataset for tasks that handle train/test splitting downstream.
        Applies to 'all-available' and 'single-session'.
        """
        if self.data_split_mode not in ["all-available", "single-session"]:
            print(f"[WARN] Calling load_all_data() but mode is '{self.data_split_mode}'.")
            
        data = self.load_raw_data()
        x_list, y_list = [], []
        
        for sid, recs in tqdm(data.items(), desc="Processing signals"):
            if not recs: continue
            
            target_recs = recs if self.data_split_mode == "all-available" else [recs[0]]
            
            for rec in target_recs:
                segments = self._process_signal(rec['signal'], rec['fs'])
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
                    
        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))

    def load_session(self, session_name):
        """
        Loads the partitioned data strictly based on temporal/record boundaries.
        Applies to cross-session and short/long-term tasks.
        """
        session_name = session_name.lower()
        if session_name in ["enrol", "train"]:
            is_enrollment = True
            log_name = "Train/Enrollment"
        elif session_name in ["probe", "test"]:
            is_enrollment = False
            log_name = "Test/Probe"
        else:
            raise ValueError("session_name must be 'enrol', 'train', 'probe', or 'test'.")
            
        data = self.load_raw_data()
        x_list, y_list = [], []
        
        kept_subjects, dropped_subjects = 0, 0

        for sid, recs in tqdm(data.items(), desc=f"Processing {log_name}"):
            if not recs: continue
            
            target_recs = []
            
            unique_dates = sorted(list(set(r['date'] for r in recs)))
            day1_date = unique_dates[0]
            day1_recs = [r for r in recs if r['date'] == day1_date]

            # --- TASK 3: SINGLE CROSS-SESSION ---
            if self.data_split_mode == "single-cross-session":
                if len(recs) < 2: 
                    dropped_subjects += 1
                    continue
                kept_subjects += 1
                target_recs = [recs[0]] if is_enrollment else [recs[1]]

            # --- TASK 4: SINGLE-SHOT SHORT-TERM ---
            elif self.data_split_mode == "single-shot-short-term":
                if len(day1_recs) < 2: 
                    dropped_subjects += 1
                    continue
                kept_subjects += 1
                target_recs = [day1_recs[0]] if is_enrollment else day1_recs[1:]

            # --- TASK 5: LEAVE-LAST-OUT SHORT-TERM ---
            elif self.data_split_mode == "leave-last-out-short-term":
                if len(day1_recs) < 2: 
                    dropped_subjects += 1
                    continue
                kept_subjects += 1
                target_recs = day1_recs[:-1] if is_enrollment else [day1_recs[-1]]

            # --- TASK 6: SINGLE-SHOT LONG-TERM ---
            elif self.data_split_mode == "single-shot-long-term":
                if len(unique_dates) < 2: 
                    dropped_subjects += 1
                    continue
                kept_subjects += 1
                target_recs = day1_recs if is_enrollment else [r for r in recs if r['date'] > day1_date]

            # --- TASK 7: LEAVE-LAST-OUT LONG-TERM ---
            elif self.data_split_mode == "leave-last-out-long-term":
                if len(unique_dates) < 2:
                    dropped_subjects += 1
                    continue
                kept_subjects += 1
                last_date = unique_dates[-1]
                target_recs = [r for r in recs if r['date'] < last_date] if is_enrollment else [r for r in recs if r['date'] == last_date]

            # --- EXTRACTION & SIGNAL PROCESSING ---
            for rec in target_recs:
                segments = self._process_signal(rec['signal'], rec['fs'])
                
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))

        # Dynamic summary print for all structured tasks during enrollment
        if self.data_split_mode not in ["all-available", "single-session"] and is_enrollment:
            mode_title = self.data_split_mode.replace('-', ' ').title()
            print(f"\n[INFO] {mode_title} Summary: Kept {kept_subjects} subjects. Dropped {dropped_subjects} subjects.")

        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))

# # =============================================================================
# # 4. CYBHi
# # =============================================================================
class load_cybhi_dataset():
    """
    Dynamic Loader for the CYBHi Dataset.
    
    CYBHi is designed to test biometric stability across intense physical/mental 
    interventions (Short-Term) and across a 3-month aging gap (Long-Term).

    Session Tags Available for Mapping:
      - 'short-term_CI' (Baseline Rest)
      - 'short-term_A1' (Intervention 1 - e.g., Physical Exercise)
      - 'short-term_A2' (Intervention 2 - e.g., Mental Stress)
      - 'long-term_S1'  (Month 0 Baseline)
      - 'long-term_S2'  (Month 3 Follow-up)

    Args:
        data_split_mode (str): The routing logic for data extraction.
            Options:
                - 'single-session': Extracts the exact sessions defined in `session_for_single_session_evaluation` 
                                    and pools them for downstream random-splitting.
                - 'cross-session': Maps data strictly to Train/Enroll/Probe groups based on the session arguments below.
        session_for_single_session_evaluation (str or list): Target session(s) to load if mode is 'single-session'.
            Example: 'long-term_S1'
        train_sessions (str or list): Session(s) to load for representation learning.
            Example: 'long-term_S1'
        enroll_sessions (str or list): Session(s) to load for Gallery enrollment.
            Example: 'long-term_S1'
        probe_sessions (str or list): Session(s) to load for Test queries.
            Example: 'long-term_S2'
        num_beats_to_merge (int): Number of consecutive beats to fuse into a single sample.
        beat_merge_method (str): Strategy for fusing beats. Options: ['average', 'concat']
        cleanup_zip (bool): If True, deletes the downloaded zip file after extraction.
        **preprocessing_params: kwargs passed directly to the Preprocessing class.
    """
    def __init__(self, data_split_mode="cross-session", 
                 session_for_single_session_evaluation=["long-term_S1"],
                 train_sessions=["long-term_S1"], 
                 enroll_sessions=["long-term_S1"],
                 probe_sessions=["long-term_S2"], 
                 num_beats_to_merge=1, beat_merge_method="average", 
                 cleanup_zip=False, **preprocessing_params):
        
        # --- KWARGS GUARD ---
        allowed_prep_kwargs = ["mode", "window_s", "stride_s", "pre_s", "post_s", "bandpass", "lowcut", "highcut", "normalize"]
        for k in preprocessing_params.keys():
            if k not in allowed_prep_kwargs:
                raise ValueError(f"\n[ERROR] Unrecognized parameter: '{k}'. Did you misspell an argument?")

        self.preprocessor = Preprocessing()
        self.cfg = CONFIG["datasets"]["cybhi"]
        project_dir = Path(__file__).resolve().parent
        self.data_root = (project_dir / CONFIG["project"]["data_root"]).resolve()
        self.dataset_root = self.data_root / self.cfg["root_dir"]
        self.zip_path = self.data_root / self.cfg["zip_name"]
        self.url = self.cfg["url"]
        
        self.prep_params = preprocessing_params if preprocessing_params else self.cfg.get("preprocessing", {})
        self.num_beats = num_beats_to_merge
        self.merge_strategy = beat_merge_method
        self.cleanup_zip = cleanup_zip
        
        valid_modes = ["single-session", "cross-session"]
        if data_split_mode not in valid_modes:
            raise ValueError(f"Invalid mode: {data_split_mode}.")
        self.data_split_mode = data_split_mode
        
        # Format variables to lists gracefully
        to_list = lambda x: [x] if isinstance(x, str) else (x if x else [])
        
        self.session_for_single_session_evaluation = to_list(session_for_single_session_evaluation)
        self.train_sessions = to_list(train_sessions)
        self.enroll_sessions = to_list(enroll_sessions)
        self.probe_sessions = to_list(probe_sessions)
        
        self.required_cross_sessions = list(set(self.train_sessions + self.enroll_sessions + self.probe_sessions))
        
        if self.data_split_mode == "single-session" and not self.session_for_single_session_evaluation:
            raise ValueError("You must provide `session_for_single_session_evaluation`.")
        if self.data_split_mode == "cross-session" and not self.required_cross_sessions:
            raise ValueError("You must provide at least one valid train, enroll, or probe session.")

    def download(self):
        _download_and_extract(self.url, self.zip_path, self.dataset_root, "CYBHi", cleanup=self.cleanup_zip)

    def _parse_file_info(self, filename):
        """
        Parses strictly based on CYBHi format seen in screenshots: 
        [Date] - [SID] - [Session/Intervention] - [Sensor/Extra]
        """
        clean = filename.replace('.txt', '').replace('._', '')
        parts = clean.split('-')
        
        rec_date = datetime.date.min
        sid = "UNKNOWN"
        session_code = "UNKNOWN"
        
        if len(parts) >= 3:
            # 1. Date (e.g., 20110715)
            if len(parts[0]) == 8 and parts[0].isdigit():
                try: 
                    rec_date = datetime.datetime.strptime(parts[0], "%Y%m%d").date()
                except ValueError: 
                    pass
            
            # 2. Subject ID (e.g., MLS)
            sid = parts[1].upper()
            
            # 3. Session Code (e.g., A1, CI, A0)
            session_code = parts[2].upper()
            
        return rec_date, sid, session_code

    def _read_signal(self, fpath):
        """Bulletproof Pandas reader using the fast C engine and native comment skipping."""
        try:
            # comment='#' prevents silent crashes on header info
            df = pd.read_csv(fpath, comment='#', delim_whitespace=True, header=None, on_bad_lines='skip')
            
            col_idx = self.cfg.get("ecg_column", 2)
            if col_idx >= df.shape[1]: col_idx = 0
            
            sig = df.iloc[:, col_idx].dropna().values.astype(float)
            sig = sig - np.mean(sig) # Zero-mean baseline
            return sig
        except Exception as e:
            print(f"\n[ERR] Failed to read {fpath.name}: {e}")
            return None

    def load_raw_data(self):
        if not self.dataset_root.exists() or not any(self.dataset_root.iterdir()):
            self.download()

        print("\n[INFO] Scanning directories and pooling CYBHi files...")
        st_pool = {} # Short-Term
        lt_pool = {} # Long-Term

        # 1. Distribute files to proper pools based on exact folder names
        for root, _, files in os.walk(self.dataset_root):
            path_str = str(root).lower()
            is_st = "short-term" in path_str or "ci" in path_str
            is_lt = "long-term" in path_str or "a0" in path_str

            if not is_st and not is_lt: continue

            for f in files:
                # IMPORTANT: Skip macOS hidden metadata files that crash the reader
                if not f.endswith(".txt") or f.startswith("._"): 
                    continue
                    
                fpath = Path(root) / f
                rec_date, sid, session_code = self._parse_file_info(f)
                
                if sid == "UNKNOWN": continue
                
                if is_st:
                    if sid not in st_pool: st_pool[sid] = []
                    st_pool[sid].append({"path": fpath, "date": rec_date, "code": session_code})
                elif is_lt:
                    if sid not in lt_pool: lt_pool[sid] = []
                    lt_pool[sid].append({"path": fpath, "date": rec_date, "code": session_code})

        recordings = {}

        # 2. Load Short-Term signals directly using explicit intervention tags
        for sid, recs in tqdm(st_pool.items(), desc="Loading short-term raw files"):
            if sid not in recordings: recordings[sid] = {}
            for rec in recs:
                tag = f"short-term_{rec['code']}" # Outputs: short-term_CI, short-term_A1, short-term_A2
                if tag not in recordings[sid]: recordings[sid][tag] = []
                
                sig = self._read_signal(rec['path'])
                if sig is not None:
                    recordings[sid][tag].append({'signal': sig, 'fs': 1000})

        # 3. Load Long-Term signals by date sequence (Month 0 vs Month 3)
        for sid, recs in tqdm(lt_pool.items(), desc="Loading long-term raw files"):
            if sid not in recordings: recordings[sid] = {}
            recs.sort(key=lambda x: x["date"])
            unique_dates = sorted(list(set([r['date'] for r in recs])))
            
            for rec in recs:
                # 1st Date = S1. 2nd Date = S2.
                date_idx = unique_dates.index(rec['date']) + 1
                tag = f"long-term_S{date_idx}" # Outputs: long-term_S1, long-term_S2
                if tag not in recordings[sid]: recordings[sid][tag] = []
                
                sig = self._read_signal(rec['path'])
                if sig is not None:
                    recordings[sid][tag].append({'signal': sig, 'fs': 1000})

        return recordings

    def _process_signal(self, sig, fs=1000):
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

    def load_all_data(self):
        if self.data_split_mode != "single-session":
            print(f"[WARN] Calling load_all_data() but mode is '{self.data_split_mode}'.")
        return self.load_session("train")

    def load_session(self, session_name):
        session_name = session_name.lower()
        target_sessions = []
        is_primary_pass = False
        
        if self.data_split_mode == "single-session":
            if session_name in ["probe", "test"]:
                raise ValueError("Cannot load 'test' in single-session mode. Split upstream.")
            target_sessions = self.session_for_single_session_evaluation
            log_name = f"Single-Session Target(s): {target_sessions}"
            is_primary_pass = True
            
        elif self.data_split_mode == "cross-session":
            if session_name in ["train"]:
                target_sessions = self.train_sessions
                is_primary_pass = True
            elif session_name in ["enrol", "enrollment"]:
                target_sessions = self.enroll_sessions
                is_primary_pass = True if not self.train_sessions else False
            elif session_name in ["probe", "test"]:
                target_sessions = self.probe_sessions
            else:
                raise ValueError("session_name must be 'train', 'enrol', or 'test'.")
            log_name = f"Cross-Session ({session_name.title()}): {target_sessions}"

        if not target_sessions:
            return np.empty((0, 0)), np.empty((0,))
            
        data = self.load_raw_data()
        x_list, y_list = [], []
        
        kept_subjects, dropped_subjects = 0, 0

        for sid, tagged_sessions in tqdm(data.items(), desc=f"Processing {log_name}"):
            
            if self.data_split_mode == "single-session":
                is_valid = all(s in tagged_sessions for s in target_sessions)
            elif self.data_split_mode == "cross-session":
                # Strict: Subject MUST have data in ALL requested global sets
                is_valid = all(s in tagged_sessions for s in self.required_cross_sessions)

            if is_valid:
                kept_subjects += 1
                for s in target_sessions:
                    # CYBHi has multiple sensors per session (e.g. 8B and 85). We pool them both!
                    for signal_dict in tagged_sessions[s]:
                        segments = self._process_signal(signal_dict['signal'], signal_dict['fs'])
                        if len(segments) > 0:
                            x_list.append(segments)
                            y_list.extend([sid] * len(segments))
            else:
                dropped_subjects += 1

        if is_primary_pass:
            print(f"\n[INFO] CYBHi Evaluation Summary ({self.data_split_mode.title()}):")
            print(f"       Kept {kept_subjects} matched subjects. Dropped {dropped_subjects} subjects.")

        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))

# =============================================================================
# 5. MIT-BIH Arrhythmia Database
# =============================================================================
class load_mitbih_dataset():
    """
    Robust Loader for the MIT-BIH Arrhythmia Database.
    
    This dataset consists of 48 continuous ~30-minute recordings from 47 subjects.
    Because there are no distinct "sessions" per subject, biometric evaluation 
    requires slicing the continuous timeline into discrete minute-based chunks.

    Args:
        leads (list of str): Target leads to extract.
            Options: Usually ['MLII'] or ['V1']. Pass 'all' for both available leads.
            Default: ['MLII']
        data_split_mode (str): Evaluation regime mapping.
            Options:
                - 'all-available': Loads the entire 30-minute continuous signal.
                - 'single-segment': Extracts a continuous chunk based on `single_segment_range`.
                - 'custom-split': Manually maps exact minute ranges to Train/Enroll/Probe regimes.
        single_segment_range (tuple): Used only if mode='single-segment'. Defines (start_min, end_min).
            Example: (0, 5) extracts the first 5 minutes.
        train_parts (list of tuples): Minute ranges for training data if mode='custom-split'.
            Example: [(0, 5), (10, 15)]
        enrol_parts (list of tuples): Minute ranges for template enrollment if mode='custom-split'.
        test_parts (list of tuples): Minute ranges for test probes if mode='custom-split'.
            Example: [(25, 30)] extracts the last 5 minutes of the tape.
        num_beats_to_merge (int): Number of consecutive beats to fuse.
        beat_merge_method (str): Strategy for fusing beats. Options: ['average', 'concat']
        cleanup_zip (bool): If True, deletes the downloaded zip file after extraction.
        **preprocessing_params: kwargs passed directly to the Preprocessing class.
    """
    def __init__(self, leads=['MLII'], data_split_mode="all-available", 
                 single_segment_range=(0, 5), train_parts=None, enrol_parts=None, 
                 test_parts=None, num_beats_to_merge=1, beat_merge_method="average", 
                 cleanup_zip=False, **preprocessing_params):
        
        self.preprocessor = Preprocessing()
        self.cfg = CONFIG["datasets"]["mitbih"]
        project_dir = Path(__file__).resolve().parent
        self.data_root = (project_dir / CONFIG["project"]["data_root"]).resolve()
        self.dataset_root = self.data_root / self.cfg["root_dir"]
        self.zip_path = self.data_root / self.cfg["zip_name"]
        self.url = self.cfg["url"]
        
        self.prep_params = preprocessing_params if preprocessing_params else self.cfg.get("preprocessing", {})
        self.target_leads = [l.lower() for l in leads] if isinstance(leads, list) else leads
        self.num_beats = num_beats_to_merge
        self.merge_strategy = beat_merge_method
        self.cleanup_zip = cleanup_zip
        
        valid_modes = ["all-available", "single-segment", "custom-split"]
        if data_split_mode not in valid_modes:
            raise ValueError(f"Invalid mode: {data_split_mode}. Use {valid_modes}")
        self.data_split_mode = data_split_mode
        
        # Segment mappings (in minutes)
        self.single_segment_range = single_segment_range
        self.train_parts = train_parts
        self.enrol_parts = enrol_parts
        self.test_parts = test_parts
        
        # Strict validation for custom-split
        if self.data_split_mode == "custom-split":
            if not self.train_parts or not self.test_parts:
                raise ValueError(
                    "For 'custom-split' mode, `train_parts` and `test_parts` cannot be None. "
                    "Please provide minute ranges. Example: train_parts=[(0, 5)], test_parts=[(25, 30)]"
                )

    def download(self):
        """Downloads and extracts the dataset if missing."""
        _download_and_extract(self.url, self.zip_path, self.dataset_root, "MIT-BIH", cleanup=self.cleanup_zip)

    def _get_lead_indices(self, available_leads):
        """Maps requested lead names (e.g., 'MLII') to channel indices."""
        avail_norm = [l.lower().strip() for l in available_leads]
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
        Files are kept continuous. Slicing occurs via index manipulation based on fs.
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
                
                data, _ = wfdb.rdsamp(str(hea.with_suffix("")), channels=lead_indices)
                recordings[sid] = {"signal": data, "fs": rec_header.fs, "filename": hea.name}
            except Exception: pass
            
        return recordings

    def _process_signal(self, sig, fs):
        """Applies filters, segmentation, and multi-beat merging."""
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

    def _slice_signal(self, raw_signal, fs, min_ranges):
        """
        Takes a continuous raw signal and extracts the requested minute boundaries.
        Returns a concatenated raw array to pass to the preprocessor.
        """
        if not min_ranges: return np.empty((0, raw_signal.shape[1]))
        
        sliced_chunks = []
        total_samples = raw_signal.shape[0]
        
        for (start_min, end_min) in min_ranges:
            start_idx = int(start_min * 60 * fs)
            end_idx = int(end_min * 60 * fs)
            
            # Boundary protections
            start_idx = max(0, start_idx)
            end_idx = min(total_samples, end_idx)
            
            if start_idx < end_idx:
                sliced_chunks.append(raw_signal[start_idx:end_idx, :])
                
        if sliced_chunks:
            return np.vstack(sliced_chunks)
        return np.empty((0, raw_signal.shape[1]))

    def load_all_data(self):
        """
        Handles dataset loading for downstream random-split tasks.
        Applies to 'all-available' and 'single-segment'.
        """
        if self.data_split_mode not in ["all-available", "single-segment"]:
            print(f"[WARN] Calling load_all_data() but mode is '{self.data_split_mode}'.")
            
        data = self.load_raw_data()
        x_list, y_list = [], []
        
        for sid, rec in tqdm(data.items(), desc="Processing signals"):
            raw_sig = rec['signal']
            fs = rec['fs']
            
            # Extract requested ranges before preprocessing
            if self.data_split_mode == "single-segment":
                target_signal = self._slice_signal(raw_sig, fs, [self.single_segment_range])
            else: # all-available
                target_signal = raw_sig
                
            if target_signal.shape[0] == 0: continue
            
            segments = self._process_signal(target_signal, fs)
            if len(segments) > 0:
                x_list.append(segments)
                y_list.extend([sid] * len(segments))
                
        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))

    def load_session(self, session_name):
        """
        Processes the customized minute-based ranges mapping to Train/Enrol/Test tasks.
        """
        if self.data_split_mode != "custom-split":
            raise ValueError("load_session() is only valid when data_split_mode='custom-split'.")
            
        session_name = session_name.lower()
        target_ranges = []
        
        # Route the correct minute ranges based on the requested session
        if session_name in ["train"]:
            target_ranges = self.train_parts
        elif session_name in ["enrol", "enrollment"]:
            if not self.enrol_parts: return np.empty((0, 0)), np.empty((0,))
            target_ranges = self.enrol_parts
        elif session_name in ["test", "probe"]:
            target_ranges = self.test_parts
        else:
            raise ValueError("session_name must be 'train', 'enrol', or 'test'.")

        data = self.load_raw_data()
        x_list, y_list = [], []
        
        kept_subjects = 0

        for sid, rec in tqdm(data.items(), desc=f"Processing {session_name}"):
            raw_sig = rec['signal']
            fs = rec['fs']
            
            # Slice the raw signal based on the assigned minutes
            target_signal = self._slice_signal(raw_sig, fs, target_ranges)
            if target_signal.shape[0] == 0: continue
            
            segments = self._process_signal(target_signal, fs)
            
            if len(segments) > 0:
                kept_subjects += 1
                x_list.append(segments)
                y_list.extend([sid] * len(segments))

        if session_name == "train":
            print(f"\n[INFO] Custom Split Summary: Extracted data for {kept_subjects} subjects.")

        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))

# =============================================================================
# 6. MIT-BIH NSRDB (Normal Sinus Rhythm)
# =============================================================================
class load_nsrdb_dataset():
    """
    Highly Optimized Loader for the MIT-BIH Normal Sinus Rhythm Database (NSRDB).
    
    This dataset consists of 18 extremely long-term Holter recordings (~24 hours continuous).
    To prevent memory overflow, this loader calculates exact byte boundaries and 
    reads ONLY the requested minute slices directly from the disk.

    Args:
        leads (list of str): Target leads to extract. 
            Options: Usually ['ECG1'] or ['ECG2'].
            Default: ['ECG1']
        data_split_mode (str): Evaluation regime mapping.
            Options:
                - 'all-available': Loads the ENTIRE 24-hour signal (Warning: High Memory/RAM usage).
                - 'single-segment': Extracts a continuous chunk based on `single_segment_range`.
                - 'custom-split': Manually maps exact minute ranges to Train/Enroll/Probe regimes.
        single_segment_range (tuple): Used only if mode='single-segment'. Defines (start_min, end_min).
            Example: (0, 60) extracts the first hour.
        train_parts (list of tuples): Minute ranges for training data if mode='custom-split'.
            Example: [(0, 120)] extracts the first 2 hours.
        enrol_parts (list of tuples): Minute ranges for template enrollment if mode='custom-split'.
        test_parts (list of tuples): Minute ranges for test probes if mode='custom-split'.
            Example: [(1380, 1440)] extracts the final hour of the 24-hour tape.
        num_beats_to_merge (int): Number of consecutive beats to fuse.
        beat_merge_method (str): Strategy for fusing beats. Options: ['average', 'concat']
        cleanup_zip (bool): If True, deletes the downloaded zip file after extraction.
        **preprocessing_params: kwargs passed directly to the Preprocessing class.
    """
    def __init__(self, leads=['ECG1'], data_split_mode="all-available", 
                 single_segment_range=(0, 60), train_parts=None, enrol_parts=None, 
                 test_parts=None, num_beats_to_merge=1, beat_merge_method="average", 
                 cleanup_zip=False, **preprocessing_params):
       
        self.preprocessor = Preprocessing()
        self.cfg = CONFIG["datasets"]["nsrdb"]
        project_dir = Path(__file__).resolve().parent
        self.data_root = (project_dir / CONFIG["project"]["data_root"]).resolve()
        self.dataset_root = self.data_root / self.cfg["root_dir"]
        self.zip_path = self.data_root / self.cfg["zip_name"]
        self.url = self.cfg["url"]
        
        self.prep_params = preprocessing_params if preprocessing_params else self.cfg.get("preprocessing", {})
        self.target_leads = [l.lower() for l in leads] if isinstance(leads, list) else leads
        self.num_beats = num_beats_to_merge
        self.merge_strategy = beat_merge_method
        self.cleanup_zip = cleanup_zip
        
        valid_modes = ["all-available", "single-segment", "custom-split"]
        if data_split_mode not in valid_modes:
            raise ValueError(f"Invalid mode: {data_split_mode}. Use {valid_modes}")
        self.data_split_mode = data_split_mode
        
        # Segment mappings (in minutes)
        self.single_segment_range = single_segment_range
        self.train_parts = train_parts
        self.enrol_parts = enrol_parts
        self.test_parts = test_parts
        
        # Strict validation for custom-split
        if self.data_split_mode == "custom-split":
            if not self.train_parts or not self.test_parts:
                raise ValueError(
                    "For 'custom-split' mode, `train_parts` and `test_parts` cannot be None. "
                    "Please provide minute ranges. Example: train_parts=[(0, 60)], test_parts=[(1380, 1440)]"
                )

    def download(self):
        """Downloads and extracts dataset."""
        _download_and_extract(self.url, self.zip_path, self.dataset_root, "NSRDB", cleanup=self.cleanup_zip)

    def _get_lead_indices(self, available_leads):
        """Finds channel indices."""
        avail_norm = [l.lower().strip() for l in available_leads]
        if self.target_leads == 'all': return list(range(len(available_leads)))
        indices = []
        for req in self.target_leads:
            req = req.strip().lower()
            if req in avail_norm: indices.append(avail_norm.index(req))
            else:
                for i, avail in enumerate(avail_norm):
                    if req in avail or avail in req: indices.append(i); break
        return indices

    def load_raw_data_slices(self, min_ranges=None):
        """
        Core I/O Optimizer: Reads ONLY the specified minute chunks from disk.
        If min_ranges is None, it loads the entire 24h file.
        """
        if not self.dataset_root.exists() or not any(self.dataset_root.iterdir()):
            self.download()
            
        recordings = {}
        files = list(self.dataset_root.rglob("*.hea"))
        
        for hea in tqdm(files, desc="Loading NSRDB specific slices"):
            sid = hea.stem
            try:
                # 1. Read lightweight header to get total length and sampling rate
                rec_header = wfdb.rdheader(str(hea.with_suffix("")))
                total_samples = rec_header.sig_len
                fs = rec_header.fs
                
                lead_indices = self._get_lead_indices(rec_header.sig_name)
                if not lead_indices: continue

                # If no ranges specified, load the entire massive file
                if min_ranges is None:
                    data, _ = wfdb.rdsamp(str(hea.with_suffix("")), channels=lead_indices)
                    if sid not in recordings: recordings[sid] = []
                    recordings[sid].append({"signal": data, "fs": fs, "filename": hea.name})
                    continue

                # 2. Iterate through requested chunks and pull them efficiently from disk
                for (start_min, end_min) in min_ranges:
                    sampfrom = int(start_min * 60 * fs)
                    sampto = int(end_min * 60 * fs)
                    
                    # Boundary protection
                    sampfrom = max(0, min(sampfrom, total_samples))
                    sampto = max(0, min(sampto, total_samples))
                    
                    if sampfrom < sampto:
                        data, _ = wfdb.rdsamp(str(hea.with_suffix("")), channels=lead_indices, sampfrom=sampfrom, sampto=sampto)
                        if sid not in recordings: recordings[sid] = []
                        recordings[sid].append({"signal": data, "fs": fs, "filename": hea.name})
                        
            except Exception: pass
            
        return recordings

    def _process_signal(self, sig, fs):
        """Applies filters, segmentation, and multi-beat merging."""
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

    def load_all_data(self):
        """
        Handles dataset loading for 'all-available' and 'single-segment'.
        """
        if self.data_split_mode not in ["all-available", "single-segment"]:
            print(f"[WARN] Calling load_all_data() but mode is '{self.data_split_mode}'.")
            
        # Determine ranges
        target_ranges = None
        if self.data_split_mode == "single-segment":
            target_ranges = [self.single_segment_range]
            
        # The optimizer perfectly fetches only what we need from disk
        data = self.load_raw_data_slices(min_ranges=target_ranges)
        
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
        Processes the customized minute-based ranges mapping to Train/Enrol/Test tasks.
        """
        if self.data_split_mode != "custom-split":
            raise ValueError("load_session() is only valid when data_split_mode='custom-split'.")
            
        session_name = session_name.lower()
        target_ranges = []
        
        # Route the correct minute ranges based on the requested session
        if session_name in ["train"]:
            target_ranges = self.train_parts
        elif session_name in ["enrol", "enrollment"]:
            if not self.enrol_parts: return np.empty((0, 0)), np.empty((0,))
            target_ranges = self.enrol_parts
        elif session_name in ["test", "probe"]:
            target_ranges = self.test_parts
        else:
            raise ValueError("session_name must be 'train', 'enrol', or 'test'.")

        # Fetch explicitly only the requested bytes from the massive 24h disk files
        data = self.load_raw_data_slices(min_ranges=target_ranges)
        
        x_list, y_list = [], []
        kept_subjects = 0

        for sid, recs in tqdm(data.items(), desc=f"Processing {session_name}"):
            if not recs: continue
            
            subject_has_data = False
            for rec in recs:
                segments = self._process_signal(rec['signal'], rec['fs'])
                if len(segments) > 0:
                    subject_has_data = True
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
                    
            if subject_has_data:
                kept_subjects += 1

        if session_name == "train":
            print(f"\n[INFO] Custom Split Summary: Extracted data for {kept_subjects} subjects.")

        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))

# =============================================================================
# 7. PTB-XL (Physikalisch-Technische Bundesanstalt XL)
# =============================================================================
class load_ptbxl_dataset():
    """
    Robust Loader for the PTB-XL Dataset.
    
    This is a massive clinical dataset (21k+ records). Every recording is exactly 10 seconds long.
    To ensure robust feature extraction, records are never split internally. Subjects lacking 
    the required number of discrete 10-second recordings for a given task are strictly dropped.

    Args:
        leads (list of str): Target leads to extract.
            Options: Any valid 12-lead string (e.g., ['i', 'v5', 'avf']) or 'all' for all 12 channels.
            Default: ['i']
        resolution (str): The sampling rate database to load.
            Options:
                - 'high': 500 Hz (Recommended for biometric fidelity).
                - 'low': 100 Hz.
        only_healthy (bool): If True, strictly evaluates the 'scp_codes' of the subject's 
                             baseline recording and drops any subject without a 'NORM' tag.
        data_split_mode (str): Evaluation regime mapping to strictly partition the 10-second records.
            Options:
                - 'all-available': Loads every record (used for random beat-level splitting).
                - 'single-session': Loads ONLY the 1st record of each subject.
                - 'single-cross-session': 1st record = Train/Enroll, 2nd record = Test/Probe.
                - 'single-shot-short-term': Day 1's 1st record = Enroll, rest of Day 1 = Probe.
                - 'leave-last-out-short-term': Day 1's last record = Probe, rest of Day 1 = Enroll.
                - 'single-shot-long-term': All Day 1 records = Enroll, all future days = Probe.
                - 'leave-last-out-long-term': Last recording day = Probe, all past days = Enroll.
        num_beats_to_merge (int): Number of consecutive beats to fuse into a single sample.
        beat_merge_method (str): Strategy for fusing beats. Options: ['average', 'concat']
        limit_records (int, optional): Hard limit on the number of patients to process (useful for fast debugging).
        cleanup_zip (bool): If True, deletes the downloaded zip file after extraction.
        **preprocessing_params: kwargs passed directly to the Preprocessing class.
    """
    def __init__(self, leads=['i'], resolution='high', only_healthy=False, 
                 data_split_mode="all-available", num_beats_to_merge=1, 
                 beat_merge_method="average", limit_records=None, 
                 cleanup_zip=False, **preprocessing_params):
       
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
        self.only_healthy = only_healthy
        self.num_beats = num_beats_to_merge
        self.merge_strategy = beat_merge_method
        self.limit_records = limit_records
        self.cleanup_zip = cleanup_zip
        
        valid_modes = [
            "all-available", "single-session", "single-cross-session", 
            "single-shot-short-term", "leave-last-out-short-term", 
            "single-shot-long-term", "leave-last-out-long-term"
        ]
        if data_split_mode not in valid_modes:
            raise ValueError(f"Invalid mode: {data_split_mode}. Use {valid_modes}")
        self.data_split_mode = data_split_mode

    def download(self):
        """Downloads and extracts the dataset if not already present."""
        _download_and_extract(self.url, self.zip_path, self.dataset_root, "PTB-XL", cleanup=self.cleanup_zip)

    def _get_lead_indices(self, available_leads):
        """Maps requested lead names (e.g., 'i', 'v5') to channel indices."""
        avail_norm = [l.lower().strip() for l in available_leads]
        if self.target_leads == 'all': return list(range(len(available_leads)))
        indices = []
        for req in self.target_leads:
            req = req.strip().lower()
            if req in avail_norm: indices.append(avail_norm.index(req))
            else:
                for i, avail in enumerate(avail_norm):
                    if req == avail: indices.append(i); break
        return indices

    def _is_healthy(self, scp_codes_str):
        """
        Checks if the 'NORM' (Normal ECG) superclass is present in the diagnostic codes.
        PTB-XL stores these as stringified dictionaries (e.g., "{'NORM': 100.0, ...}").
        """
        return "NORM" in str(scp_codes_str)

    def load_raw_data(self):
        """
        Parses the official ptbxl_database.csv, loads WFDB records, and groups by patient.
        Ensures strict chronological sorting using official metadata timestamps.
        """
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
            # Sort chronologically using precise metadata timestamps
            patient_recs = df[df['patient_id'] == pid].sort_values(by='recording_date')
            
            # --- Healthy Control Check ---
            # We evaluate the baseline (first) recording to determine subject eligibility
            if self.only_healthy:
                baseline_codes = patient_recs.iloc[0]['scp_codes']
                if not self._is_healthy(baseline_codes):
                    continue

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
                    
                    # Store exact datetime for Day 1 splits
                    rec_dt = pd.to_datetime(row['recording_date']).date()
                    recs_list.append({"signal": data, "fs": rec_header.fs, "date": rec_dt})
                except Exception: pass
            
            if recs_list: recordings[str(pid)] = recs_list
            
        return recordings

    def _process_signal(self, sig, fs):
        """Applies filters, segmentation, and multi-beat merging."""
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

    def load_all_data(self):
        """
        Loads dataset for tasks that handle train/test splitting downstream.
        Applies to 'all-available' and 'single-session'.
        """
        if self.data_split_mode not in ["all-available", "single-session"]:
            print(f"[WARN] Calling load_all_data() but mode is '{self.data_split_mode}'.")
            
        data = self.load_raw_data()
        x_list, y_list = [], []
        
        for sid, recs in tqdm(data.items(), desc="Processing signals"):
            if not recs: continue
            
            target_recs = recs if self.data_split_mode == "all-available" else [recs[0]]
            
            for rec in target_recs:
                segments = self._process_signal(rec['signal'], rec['fs'])
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
                    
        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))

    def load_session(self, session_name):
        """
        Loads the partitioned data strictly based on temporal/record boundaries.
        No intra-record splitting is permitted to preserve complete 10s windows.
        """
        session_name = session_name.lower()
        if session_name in ["enrol", "train"]:
            is_enrollment = True
            log_name = "Train/Enrollment"
        elif session_name in ["probe", "test"]:
            is_enrollment = False
            log_name = "Test/Probe"
        else:
            raise ValueError("session_name must be 'enrol', 'train', 'probe', or 'test'.")
            
        data = self.load_raw_data()
        x_list, y_list = [], []
        
        kept_subjects, dropped_subjects = 0, 0

        for sid, recs in tqdm(data.items(), desc=f"Processing {log_name}"):
            if not recs: continue
            
            target_recs = []
            
            unique_dates = sorted(list(set(r['date'] for r in recs)))
            day1_date = unique_dates[0]
            day1_recs = [r for r in recs if r['date'] == day1_date]

            # --- TASK 3: SINGLE CROSS-SESSION ---
            # Objective: Test immediate cross-record variation (Rec 1 vs Rec 2).
            if self.data_split_mode == "single-cross-session":
                if len(recs) < 2: 
                    dropped_subjects += 1
                    continue
                kept_subjects += 1
                target_recs = [recs[0]] if is_enrollment else [recs[1]]

            # --- TASK 4: SINGLE-SHOT SHORT-TERM ---
            # Objective: Single-shot enrollment vs all subsequent intra-day probes.
            elif self.data_split_mode == "single-shot-short-term":
                if len(day1_recs) < 2: 
                    dropped_subjects += 1
                    continue
                kept_subjects += 1
                target_recs = [day1_recs[0]] if is_enrollment else day1_recs[1:]

            # --- TASK 5: LEAVE-LAST-OUT SHORT-TERM ---
            # Objective: Multi-shot enrollment vs a single final intra-day probe.
            elif self.data_split_mode == "leave-last-out-short-term":
                if len(day1_recs) < 2: 
                    dropped_subjects += 1
                    continue
                kept_subjects += 1
                target_recs = day1_recs[:-1] if is_enrollment else [day1_recs[-1]]

            # --- TASK 6: SINGLE-SHOT LONG-TERM ---
            # Objective: Day 1 template aging tested against all future days.
            elif self.data_split_mode == "single-shot-long-term":
                if len(unique_dates) < 2: 
                    dropped_subjects += 1
                    continue
                kept_subjects += 1
                target_recs = day1_recs if is_enrollment else [r for r in recs if r['date'] > day1_date]

            # --- TASK 7: LEAVE-LAST-OUT LONG-TERM ---
            # Objective: Historical longitudinal data enrolled vs final day probe.
            elif self.data_split_mode == "leave-last-out-long-term":
                if len(unique_dates) < 2:
                    dropped_subjects += 1
                    continue
                kept_subjects += 1
                last_date = unique_dates[-1]
                target_recs = [r for r in recs if r['date'] < last_date] if is_enrollment else [r for r in recs if r['date'] == last_date]

            # --- EXTRACTION & SIGNAL PROCESSING ---
            for rec in target_recs:
                segments = self._process_signal(rec['signal'], rec['fs'])
                
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))

        # Dynamic summary print during the enrollment pass
        if self.data_split_mode not in ["all-available", "single-session"] and is_enrollment:
            mode_title = self.data_split_mode.replace('-', ' ').title()
            print(f"\n[INFO] {mode_title} Summary: Kept {kept_subjects} subjects. Dropped {dropped_subjects} subjects.")

        if x_list: return np.vstack(x_list), np.array(y_list)
        return np.empty((0, 0)), np.empty((0,))
# =============================================================================