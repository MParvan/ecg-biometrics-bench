import numpy as np
import neurokit2 as nk
from scipy.signal import resample
from filtering import Filtering  # Import your dedicated module

class Preprocessing:
    """
    Main ECG Preprocessing Pipeline.
    
    This class handles the core signal processing steps required to transform 
    raw ECG data into model-ready inputs. It supports two distinct segmentation paradigms:
    
    1. **Beat-Based (Fiducial):**
       - Traditional biometric approach.
       - Detects R-peaks, cuts a fixed window around them, and aligns them.
       - Focuses on QRS morphology.
       
    2. **Blind (Non-Fiducial):**
       - Modern Deep Learning approach.
       - Slides a fixed-length window (e.g., 5s, 10s) across the signal.
       - Captures Rhythm and Heart Rate Variability (HRV) context.
       - Ignores R-peak detection errors.
    
    Pipeline Steps:
    1. **Filtering:** Removes powerline noise, baseline wander, and EMG.
    2. **Segmentation:** Cuts the signal into 'samples' (Beats or Windows).
    3. **Normalization:** Scales the data (Z-Score or MinMax).
    4. **Resampling:** Ensures consistent input size for the neural network.
    """

    def __init__(self):
        self.filtering = Filtering()

    def detect_r_peaks(self, ecg: np.ndarray, fs: int, method: str = "pantompkins") -> np.ndarray:
        """
        Detects R-peaks using NeuroKit2 algorithms.
        
        Args:
            ecg (np.ndarray): The input ECG signal (1D).
            fs (int): Sampling frequency.
            method (str): Detection algorithm ('pantompkins', 'hamilton', 'engzee', etc.).
            
        Returns:
            np.ndarray: Array of indices where R-peaks are located.
        """
        try:
            # NeuroKit2 provides a robust processing pipeline. 
            # We extract just the R-peaks from the result.
            _, info = nk.ecg_process(ecg, sampling_rate=fs, method=method)
            r_peaks = info["ECG_R_Peaks"]
            
            # Sanity check: Remove NaNs if any appear
            r_peaks = r_peaks[~np.isnan(r_peaks)]
            return r_peaks.astype(int)
        except Exception as e:
            # If detection fails (e.g., extremely noisy signal), return empty.
            # The loader will skip this signal instead of crashing.
            # print(f"[WARN] R-peak detection failed: {e}") 
            return np.array([])

    def cut_beats(self, ecg: np.ndarray, r_locs: np.ndarray, fs: int, pre_s: float, post_s: float, align_peak: bool = True):
        """
        Segments a continuous ECG signal into individual heartbeats.
        
        Args:
            ecg (np.ndarray): The full ECG signal.
            r_locs (np.ndarray): Indices of R-peaks.
            fs (int): Sampling frequency.
            pre_s (float): Seconds to include BEFORE the R-peak (e.g., 0.2s for P-wave).
            post_s (float): Seconds to include AFTER the R-peak (e.g., 0.4s for T-wave).
            align_peak (bool): If True, performs local search to center the R-peak exactly 
                               at the defined center index, correcting small detection jitters.
                               
        Returns:
            list[np.ndarray]: List of individual beat segments.
        """
        pre = int(round(pre_s * fs))
        post = int(round(post_s * fs))
        beats = []
        N = len(ecg)

        for r in r_locs:
            s, e = r - pre, r + post
            
            # Boundary checks: skip beats at the very start/end if they get cut off
            if s < 0 or e > N:
                continue
            
            beat = ecg[s:e]
            
            if align_peak:
                # --- Peak Alignment Logic ---
                # R-peak detectors aren't perfect. They might mark the peak 1-2 samples off.
                # We search a small window (+/- 50ms) around the expected center to find the REAL max.
                center_idx = pre
                search_window = int(0.05 * fs) 
                
                # Extract the small window around the center
                local_window = beat[max(0, center_idx - search_window) : min(len(beat), center_idx + search_window)]
                
                if len(local_window) == 0: continue

                # Find the index of the maximum absolute value (handles inverted peaks too)
                peak_offset = np.argmax(np.abs(local_window))
                
                # Convert local offset back to global signal indices
                actual_peak_idx = (center_idx - search_window) + peak_offset
                shift = actual_peak_idx - center_idx
                
                # Apply the shift
                s += shift
                e += shift
                
                # Re-check boundaries after shifting
                if s < 0 or e > N:
                    continue
                    
                beat = ecg[s:e]

            beats.append(beat.copy())

        return beats

    def segment_blind(self, ecg: np.ndarray, fs: int, window_s: float, stride_s: float) -> list:
        """
        Segments the signal into fixed-length windows (Blind / Non-Fiducial Segmentation).
        
        This method DOES NOT use R-peaks. It blindly cuts the signal every X seconds.
        Useful for rhythm analysis (HRV) and deep learning models that learn from raw context.
        
        Args:
            ecg (np.ndarray): The full ECG signal.
            fs (int): Sampling frequency.
            window_s (float): Length of each window in seconds (e.g., 5.0).
            stride_s (float): Step size in seconds (e.g., 2.0). 
                              If stride < window, windows will overlap (data augmentation).
                              
        Returns:
            list[np.ndarray]: List of signal segments.
        """
        window_samples = int(window_s * fs)
        stride_samples = int(stride_s * fs)
        
        segments = []
        N = len(ecg)
        
        # If signal is shorter than one window, discard it
        if N < window_samples:
            return []

        # Sliding Window Loop
        for start in range(0, N - window_samples + 1, stride_samples):
            end = start + window_samples
            segment = ecg[start:end]
            segments.append(segment)
            
        return segments

    def normalize_signal(self, sig: np.ndarray, method: str = "zscore") -> np.ndarray:
        """
        Normalizes a single signal segment (beat or window).
        
        Args:
            sig (np.ndarray): Input segment.
            method (str): 'zscore' (Zero mean, Unit variance) or 'minmax' (0 to 1).
        """
        if method == "zscore":
            std = np.std(sig)
            if std < 1e-6: return sig # Avoid division by zero for flat lines
            return (sig - np.mean(sig)) / std
        
        elif method == "minmax":
            min_val, max_val = np.min(sig), np.max(sig)
            if max_val - min_val < 1e-6: return sig
            return (sig - min_val) / (max_val - min_val)
            
        return sig

    def resample_signal(self, sig: np.ndarray, target_len: int) -> np.ndarray:
        """
        Resamples a signal segment to a fixed length using Fourier method.
        Essential for stacking beats into a numpy array for training.
        """
        if len(sig) == target_len:
            return sig
        return resample(sig, target_len)

    def preprocess_ecg(self, ecg: np.ndarray, fs: int, 
                       mode: str = "beat", # 'beat' or 'blind'
                       # Beat-based params
                       pre_s: float = 0.2, post_s: float = 0.4, resample_len: int = None,
                       # Blind params
                       window_s: float = 5.0, stride_s: float = 1.0,
                       # Common params
                       filter_method: str = "butter", filter_kwargs: dict = {},
                       norm_method: str = "zscore") -> np.ndarray:
        """
        Master Preprocessing Pipeline.
        
        This function orchestrates the entire flow:
        1. Filters the raw signal.
        2. Segments it (using either Beat or Blind mode).
        3. Normalizes each segment.
        4. Stacks them into a final array.
        
        Args:
            ecg (np.ndarray): Raw input signal.
            fs (int): Sampling frequency.
            mode (str): 'beat' for R-peak segmentation, 'blind' for sliding window.
            
            [Beat Params]
            pre_s (float): Seconds before R-peak.
            post_s (float): Seconds after R-peak.
            resample_len (int): Forced length of output beat (optional).
            
            [Blind Params]
            window_s (float): Window size in seconds.
            stride_s (float): Stride size in seconds.
            
            [Filter Params]
            filter_method (str): 'butter', 'fir', 'notch', etc.
            norm_method (str): 'zscore', 'minmax', or None.
            
        Returns:
            np.ndarray: 2D array of shape (Num_Segments, Segment_Length).
                        Returns empty array if processing fails.
        """
        # -----------------------------------------------------------
        # STEP 1: Filtering
        # -----------------------------------------------------------
        if filter_method == "butter":
            ecg_clean = self.filtering.butter(ecg, fs, **filter_kwargs)
        elif filter_method == "fir":
            ecg_clean = self.filtering.fir(ecg, fs, **filter_kwargs)
        elif filter_method == "notch":
            ecg_clean = self.filtering.notch(ecg, fs, **filter_kwargs)
        elif filter_method == "savgol":
            ecg_clean = self.filtering.savgol(ecg, **filter_kwargs)
        else:
            ecg_clean = ecg # Raw signal

        segments = []

        # -----------------------------------------------------------
        # STEP 2: Segmentation
        # -----------------------------------------------------------
        if mode == "beat":
            # --- Mode A: Fiducial (Beat-Based) ---
            r_locs = self.detect_r_peaks(ecg_clean, fs)
            
            # Need at least 2 R-peaks to be sure of rhythm
            if len(r_locs) < 2:
                return np.empty((0, 0))
            
            raw_beats = self.cut_beats(ecg_clean, r_locs, fs, pre_s, post_s, align_peak=True)
            
            for b in raw_beats:
                # Optional resampling for beats to ensure strict input size
                if resample_len is not None:
                    b = self.resample_signal(b, resample_len)
                segments.append(b)

        elif mode == "blind":
            # --- Mode B: Non-Fiducial (Blind Window) ---
            segments = self.segment_blind(ecg_clean, fs, window_s, stride_s)

        else:
            raise ValueError(f"Unknown preprocessing mode: {mode}. Use 'beat' or 'blind'.")

        # If segmentation produced nothing (too short signal / no peaks)
        if not segments:
            return np.empty((0, 0))

        # -----------------------------------------------------------
        # STEP 3: Normalization & Stacking
        # -----------------------------------------------------------
        final_segments = []
        for seg in segments:
            if norm_method:
                seg = self.normalize_signal(seg, norm_method)
            final_segments.append(seg)

        # Convert list of arrays to a single 2D Numpy array
        # Shape: (N_Samples, Sample_Length)
        return np.vstack(final_segments).astype(np.float32)