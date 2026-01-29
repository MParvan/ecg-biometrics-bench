# data_augmentation.py
import numpy as np
from scipy.signal import stft, istft, butter, lfilter

class ECGAugmentation:
    """
    Robust ECG signal augmentation pipeline.
    Expects input shape: (num_beats, length)
    """

    def __init__(self):
        pass

    def gaussian(self, beats: np.ndarray, std: float = 0.01, relative: bool = True) -> np.ndarray:
        """Add zero-mean Gaussian noise (simulates thermal/sensor noise)."""
        beats = np.asarray(beats, dtype=np.float32)
        if beats.size == 0: return beats

        if relative:
            sig_std = beats.std(axis=1, keepdims=True) + 1e-8
            noise = np.random.randn(*beats.shape).astype(np.float32) * (std * sig_std)
        else:
            noise = np.random.randn(*beats.shape).astype(np.float32) * std

        return beats + noise

    def amplitude(self, beats: np.ndarray, scale_range: tuple = (0.9, 1.1)) -> np.ndarray:
        """Global amplitude scaling (simulates gain differences)."""
        beats = np.asarray(beats, dtype=np.float32)
        if beats.size == 0: return beats
        
        low, high = scale_range
        factors = np.random.uniform(low, high, size=(beats.shape[0], 1)).astype(np.float32)
        return beats * factors

    def timeshift(self, beats: np.ndarray, max_shift: int = 10) -> np.ndarray:
        """
        Circular time shift. 
        Note: For segmented beats, ensure padding exists or shift is small 
        to avoid moving the R-peak to the edge.
        """
        beats = np.asarray(beats, dtype=np.float32)
        if beats.size == 0: return beats

        N, L = beats.shape
        out = np.empty_like(beats)
        for i in range(N):
            shift = np.random.randint(-max_shift, max_shift + 1)
            out[i] = np.roll(beats[i], shift)
        return out

    def baseline_wander(self, beats, freq=0.3, amp=0.1, fs=250):
        """Add low-frequency sinusoidal drift (simulates breathing)."""
        beats = np.asarray(beats, dtype=np.float32)
        if beats.size == 0: return beats
        
        t = np.arange(beats.shape[1]) / fs
        # Randomize phase for each beat so they don't all drift identically
        phase = np.random.uniform(0, 2*np.pi, size=(beats.shape[0], 1))
        drift = amp * np.sin(2 * np.pi * freq * t + phase)
        return beats + drift

    # --- FIXED FUNCTION ---
    def time_warp(self, beats, sigma=0.2, num_knots=4):
        """
        Elastic deformation (Time Warping) using cubic spline interpolation.
        Simulates Heart Rate Variability within a single beat.
        """
        beats = np.asarray(beats, dtype=np.float32)
        if beats.size == 0: return beats
        
        N, L = beats.shape
        out = np.empty_like(beats)
        x = np.arange(L)
        
        for i in range(N):
            # Generate random control points (knots) along the time axis
            # We perturb these knots to stretch/squeeze time
            knots_x = np.linspace(0, L, num_knots+2)
            knots_y = knots_x + np.random.normal(0, sigma * (L/num_knots), size=num_knots+2)
            
            # Anchor endpoints to prevent shifting the whole beat out of frame
            knots_y[0] = 0
            knots_y[-1] = L
            
            # Interpolate to find the new time indices
            # (We use linear here for speed, cubic is better but slower)
            warped_x = np.interp(x, knots_x, knots_y)
            
            # Map original signal to warped time base
            out[i] = np.interp(x, warped_x, beats[i])
            
        return out

    # --- NEW FUNCTION: CUTOUT ---
    def cutout(self, beats: np.ndarray, num_holes: int = 1, length: int = 20) -> np.ndarray:
        """
        Randomly zeroes out continuous sections (Masking).
        Forces model to learn context rather than local features.
        """
        beats = np.asarray(beats, dtype=np.float32)
        if beats.size == 0: return beats
        
        N, L = beats.shape
        out = beats.copy()
        
        for i in range(N):
            for _ in range(num_holes):
                start = np.random.randint(0, L - length)
                out[i, start:start+length] = 0.0
        return out

    # --- NEW FUNCTION: EMG NOISE ---
    def emg_noise(self, beats, fs=250, std=0.05):
        """
        Simulates Muscle Artifact (EMG).
        High-frequency noise (e.g., >20Hz).
        """
        beats = np.asarray(beats, dtype=np.float32)
        if beats.size == 0: return beats
        
        N, L = beats.shape
        noise = np.random.randn(N, L).astype(np.float32)
        
        # High-pass filter the noise to simulate EMG
        b, a = butter(4, 20, btype='high', fs=fs)
        colored_noise = lfilter(b, a, noise, axis=1)
        
        # Scale noise
        sig_std = beats.std(axis=1, keepdims=True) + 1e-8
        colored_noise = colored_noise * (std * sig_std / (colored_noise.std(axis=1, keepdims=True) + 1e-8))
        
        return beats + colored_noise

    def _ensure_2d(self, x):
        if x.ndim == 1: return x[np.newaxis, :]
        return x

    def istft_augment(self, x, fs, window="hann", nperseg=128, noverlap=64, noise_std=0.05, log_power=True):
        """
        Spectrogram-based augmentation (Advanced).
        Adds noise in the Frequency domain while preserving Phase.
        """
        x = self._ensure_2d(x)
        augmented = []
        
        for seg in x:
            # 1. STFT
            f, t, Z = stft(seg, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
            
            mag = np.abs(Z)
            phase = np.angle(Z)
            
            # 2. Add Noise to Magnitude
            if log_power:
                mag_log = np.log(mag + 1e-8)
                noise = np.random.normal(0, noise_std, mag_log.shape)
                mag_aug = np.exp(mag_log + noise)
            else:
                noise = np.random.normal(0, noise_std * mag.std(), mag.shape)
                mag_aug = np.clip(mag + noise, 0, None)
            
            # 3. Inverse STFT
            Z_aug = mag_aug * np.exp(1j * phase)
            _, x_rec = istft(Z_aug, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
            
            # 4. Fix Length
            if len(x_rec) > len(seg):
                x_rec = x_rec[:len(seg)]
            else:
                x_rec = np.pad(x_rec, (0, len(seg) - len(x_rec)))
                
            augmented.append(x_rec)
            
        return np.stack(augmented)