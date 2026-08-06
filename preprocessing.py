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

    # Detected R-peaks are not always the R-peak. Several widely used QRS
    # detectors report the maximum of a smoothed detection curve rather than
    # the fiducial point itself, which lags the true peak by roughly half the
    # smoothing window, commonly 40-70 ms. That offset is harmless for interval
    # measurements, where it cancels, but it displaces every window cut around
    # the peak. The alignment search below therefore has to be wide enough to
    # reach back past that lag; a window narrower than the offset can settle on
    # a neighbouring wave and end up worse than no alignment at all.
    #
    # The width is taken as a fraction of the median R-R interval so it tracks
    # heart rate and sampling rate, and is clamped so it stays wide enough to
    # cover the largest reported detector lag and narrow enough to exclude the
    # T-wave.
    ALIGN_RR_FRACTION = 0.12
    ALIGN_MIN_S = 0.10
    ALIGN_MAX_S = 0.20

    def __init__(self):
        self.filtering = Filtering()

    def alignment_window(self, r_locs, fs, pre_s, post_s, align_window_s=None):
        """
        Choose the half-width of the R-peak alignment search, in samples.

        Args:
            r_locs (np.ndarray): Detected R-peak indices.
            fs (float): Sampling frequency.
            pre_s (float): Seconds kept before the peak.
            post_s (float): Seconds kept after the peak.
            align_window_s (float or None): Fixed half-width in seconds. When
                None, the width is derived from the median R-R interval.

        Returns:
            int: Half-width in samples, at least one.
        """
        fs = float(fs)

        if align_window_s is not None:
            samples = int(round(float(align_window_s) * fs))
        else:
            r_locs = np.asarray(r_locs)
            if r_locs.size >= 2:
                # Peak indices are already in samples, so the fraction of the
                # median interval needs no conversion.
                median_rr = float(np.median(np.diff(r_locs)))
                samples = int(round(self.ALIGN_RR_FRACTION * median_rr))
            else:
                samples = int(round(self.ALIGN_MIN_S * fs))

            # The bounds are applied in samples rather than seconds so the
            # returned width honours them exactly, instead of overshooting by
            # up to half a sample at low rates.
            samples = min(
                max(samples, int(round(self.ALIGN_MIN_S * fs))),
                int(self.ALIGN_MAX_S * fs),
            )

        # The search cannot reach outside the cut window without changing which
        # samples the beat contains.
        limit = int(min(float(pre_s), float(post_s)) * fs)

        return max(1, min(samples, limit))

    def peak_polarity(self, ecg, r_locs, fs, window):
        """
        Decide whether R-peaks point up or down in this recording.

        Selecting the largest absolute deviation per beat handles inverted
        leads, but lets a deep S-wave outrank a modest R-peak once the search
        widens. Deciding the polarity once for the whole recording avoids that,
        because the choice is then driven by the many beats where the R-peak
        clearly dominates rather than by one ambiguous beat.

        Args:
            ecg (np.ndarray): The signal being segmented.
            r_locs (np.ndarray): Detected R-peak indices.
            fs (float): Sampling frequency.
            window (int): Alignment half-width in samples.

        Returns:
            int: +1 when upward deflections dominate, -1 when inverted.
        """
        upward = 0.0
        downward = 0.0

        for r in np.asarray(r_locs, dtype=int):
            low = max(0, r - window)
            high = min(len(ecg), r + window)
            if high - low < 2:
                continue
            segment = ecg[low:high]
            baseline = float(np.median(segment))
            upward += float(segment.max()) - baseline
            downward += baseline - float(segment.min())

        return -1 if downward > upward else 1

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

    def cut_beats(self, ecg: np.ndarray, r_locs: np.ndarray, fs: int, pre_s: float, post_s: float, align_peak: bool = True, align_window_s: float = None):
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
            align_window_s (float or None): Half-width of that search, in seconds.
                               None derives it from the median R-R interval, which
                               is the recommended setting; see alignment_window.

        Returns:
            list[np.ndarray]: List of individual beat segments.
        """
        pre = int(round(pre_s * fs))
        post = int(round(post_s * fs))
        beats = []
        N = len(ecg)

        if align_peak:
            search_window = self.alignment_window(
                r_locs, fs, pre_s, post_s, align_window_s
            )
            polarity = self.peak_polarity(ecg, r_locs, fs, search_window)

        for r in r_locs:
            s, e = r - pre, r + post

            # Boundary checks: skip beats at the very start/end if they get cut off
            if s < 0 or e > N:
                continue

            beat = ecg[s:e]
            
            if align_peak:
                # --- Peak Alignment Logic ---
                # The detector reports where a QRS complex is, not necessarily
                # where its peak lies, so the beat is re-centred on the largest
                # deflection nearby. See alignment_window for how wide the
                # search has to be and why.
                center_idx = pre

                # The start is clamped to the array, and the same clamped value
                # is used to convert back, so a beat whose search would run off
                # the front is not silently shifted.
                window_start = max(0, center_idx - search_window)
                window_end = min(len(beat), center_idx + search_window + 1)
                local_window = beat[window_start:window_end]

                if len(local_window) == 0: continue

                # Polarity is decided once per recording rather than per beat,
                # so an inverted lead is handled without letting a deep S-wave
                # win on an individual beat.
                if polarity >= 0:
                    peak_offset = int(np.argmax(local_window))
                else:
                    peak_offset = int(np.argmin(local_window))

                # Convert local offset back to global signal indices
                actual_peak_idx = window_start + peak_offset
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

    def preprocess_ecg(
        self,
        ecg: np.ndarray,
        fs: int,
        mode: str = "beat",
        pre_s: float = 0.2,
        post_s: float = 0.4,
        resample_len: int = None,
        window_s: float = 5.0,
        stride_s: float = 1.0,
        rpeak_method: str = "pantompkins",
        align_peak: bool = True,
        align_window_s: float = None,
        filter_method: str = "butter",
        filter_kwargs: dict = None,
        norm_method: str = "zscore",
    ) -> np.ndarray:
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
            
            [Detection and Alignment Params]
            rpeak_method (str): NeuroKit2 R-peak detector name.
            align_window_s (float or None): Half-width of the alignment search
                in seconds. None derives it from the median R-R interval, which
                is the recommended setting.
            align_peak (bool): Refine each detected R-peak within a local
                +/-50 ms neighbourhood before segmentation.

            [Filter and Normalization Params]
            filter_method (str or None): 'butter', 'fir', 'notch',
                'savgol', or None.
            filter_kwargs (dict or None): Method-specific filter settings.
            norm_method (str or None): 'zscore', 'minmax', or None.
            
        Returns:
            np.ndarray: 2D array of shape (Num_Segments, Segment_Length).
                        Returns empty array if processing fails.
        """
        ecg = np.asarray(ecg)

        if ecg.ndim != 1:
            raise ValueError("ecg must be a one-dimensional signal.")

        if isinstance(fs, bool) or not np.isscalar(fs) or not np.isfinite(fs):
            raise ValueError("fs must be a finite positive number.")

        fs = float(fs)

        if fs <= 0.0:
            raise ValueError("fs must be a finite positive number.")

        if mode not in {"beat", "blind"}:
            raise ValueError(
                f"Unknown preprocessing mode: {mode}. Use 'beat' or 'blind'."
            )

        for parameter_name, value in {
            "pre_s": pre_s,
            "post_s": post_s,
            "window_s": window_s,
            "stride_s": stride_s,
        }.items():
            if (
                isinstance(value, bool)
                or not np.isscalar(value)
                or not np.isfinite(value)
                or float(value) <= 0.0
            ):
                raise ValueError(
                    f"{parameter_name} must be a finite positive number."
                )

        if resample_len is not None:
            if (
                isinstance(resample_len, bool)
                or not isinstance(resample_len, (int, np.integer))
                or int(resample_len) < 1
            ):
                raise ValueError(
                    "resample_len must be a positive integer or None."
                )
            resample_len = int(resample_len)

        if not isinstance(rpeak_method, str) or not rpeak_method.strip():
            raise ValueError("rpeak_method must be a non-empty string.")

        if not isinstance(align_peak, (bool, np.bool_)):
            raise ValueError("align_peak must be Boolean.")

        if align_window_s is not None:
            if isinstance(align_window_s, bool) or not np.isscalar(align_window_s):
                raise ValueError(
                    "align_window_s must be a positive number or None."
                )
            align_window_s = float(align_window_s)
            if not np.isfinite(align_window_s) or align_window_s <= 0.0:
                raise ValueError(
                    "align_window_s must be a positive number or None."
                )

        supported_filter_methods = {
            None,
            "butter",
            "fir",
            "notch",
            "savgol",
        }

        if filter_method not in supported_filter_methods:
            raise ValueError(
                "filter_method must be one of: butter, fir, notch, "
                "savgol, or None."
            )

        if filter_kwargs is None:
            filter_kwargs = {}
        elif not isinstance(filter_kwargs, dict):
            raise ValueError("filter_kwargs must be a dictionary or None.")
        else:
            filter_kwargs = dict(filter_kwargs)

        if norm_method not in {None, "zscore", "minmax"}:
            raise ValueError(
                "norm_method must be 'zscore', 'minmax', or None."
            )

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
            r_locs = self.detect_r_peaks(
                ecg_clean,
                fs,
                method=rpeak_method,
            )
            
            # Need at least 2 R-peaks to be sure of rhythm
            if len(r_locs) < 2:
                return np.empty((0, 0))
            
            raw_beats = self.cut_beats(
                ecg_clean,
                r_locs,
                fs,
                pre_s,
                post_s,
                align_peak=bool(align_peak),
                align_window_s=align_window_s,
            )
            
            for b in raw_beats:
                # Optional resampling for beats to ensure strict input size
                if resample_len is not None:
                    b = self.resample_signal(b, resample_len)
                segments.append(b)

        elif mode == "blind":
            # --- Mode B: Non-Fiducial (Blind Window) ---
            segments = self.segment_blind(ecg_clean, fs, window_s, stride_s)

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