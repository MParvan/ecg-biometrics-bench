import numpy as np
from scipy.signal import stft
import pywt

class Representation:
    """
    Time-frequency (Spectrogram/Scalogram) and Time-Space (GAF/RP) 
    representations for ECG signals.

    Converts 1D ECG signals into 2D images suitable for CNNs.
    Input shape: (N_segments, T)
    """

    def __init__(self):
        pass

    def _ensure_2d(self, x):
        if x.ndim == 1:
            return x[np.newaxis, :]
        return x

    def _normalize_minmax(self, x):
        """Normalize to [0, 1] per sample (required for GAF/RP)."""
        x_min = x.min(axis=1, keepdims=True)
        x_max = x.max(axis=1, keepdims=True)
        return (x - x_min) / (x_max - x_min + 1e-8)

    # ------------------------------------------------------------
    # 1. Frequency Domain (Spectrograms)
    # ------------------------------------------------------------

    def stft(self, x, fs, window="hann", nperseg=128, noverlap=64, log_power=True):
        """Standard Short-Time Fourier Transform."""
        x = self._ensure_2d(x)
        specs = []

        for seg in x:
            _, _, Z = stft(seg, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
            S = np.abs(Z) ** 2
            if log_power:
                S = np.log(S + 1e-8)
            specs.append(S)

        return np.stack(specs)

    def cwt(self, x, wavelet="morl", scales=None, log_power=True):
        """Continuous Wavelet Transform (Scalogram)."""
        x = self._ensure_2d(x)
        if scales is None:
            # Typical scales for ECG: 1 to 64 or 128
            scales = np.arange(1, 64)

        scalograms = []
        for seg in x:
            coef, _ = pywt.cwt(seg, scales, wavelet)
            power = np.abs(coef) ** 2
            if log_power:
                power = np.log(power + 1e-8)
            scalograms.append(power)

        return np.stack(scalograms)

    def mel_spectrogram(self, x, fs, n_mels=64, n_fft=256, hop_length=64):
        """
        Mel-Spectrogram.
        Focuses resolution on lower frequencies (where ECG energy lives).
        Requires: pip install librosa
        """
        import librosa
        x = self._ensure_2d(x)
        mels = []

        for seg in x:
            # librosa expects float32
            S = librosa.feature.melspectrogram(
                y=seg.astype(np.float32), 
                sr=fs, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                n_mels=n_mels
            )
            S_db = librosa.power_to_db(S, ref=np.max)
            mels.append(S_db)
        
        return np.stack(mels)

    # ------------------------------------------------------------
    # 2. Time-Space Domain (Computer Vision Encodings)
    # ------------------------------------------------------------

    def gaf(self, x, method='summation', image_size=64):
        """
        Gramian Angular Field (GASF / GADF).
        Encodes time-series correlations into polar coordinates.
        
        Parameters:
            method: 'summation' (GASF) or 'difference' (GADF)
            image_size: Output image dimension (NxN). 
                        If input len != image_size, it uses PAA to resize.
        Requires: pip install pyts
        """
        from pyts.image import GramianAngularField
        x = self._ensure_2d(x)
        
        # GAF requires inputs in [-1, 1] or [0, 1]. pyts handles this usually,
        # but explicit scaling is safer.
        gaf_transformer = GramianAngularField(image_size=image_size, method=method)
        return gaf_transformer.transform(x)

    def recurrence_plot(self, x, dimension=1, time_delay=1, threshold=None, image_size=64):
        """
        Recurrence Plot (RP).
        Visualizes the recurrence of states in phase space.
        
        Requires: pip install pyts
        """
        from pyts.image import RecurrencePlot
        x = self._ensure_2d(x)
        
        # If threshold is None, 'point' or 'distance' modes are used automatically
        rp_transformer = RecurrencePlot(
            dimension=dimension, 
            time_delay=time_delay, 
            threshold=threshold
        )
        
        # Resize logic (RP output size = input length)
        # If you need fixed size, you often need to resample the signal first
        # or resize the resulting image.
        imgs = rp_transformer.transform(x)
        
        # Optional: Resize images to fixed size using scipy or cv2 if needed
        # Here we return the raw RP
        return imgs

    # ------------------------------------------------------------
    # 3. Advanced / Niche (Wrapped for safety)
    # ------------------------------------------------------------

    def stransform(self, x):
        """S-transform (Stockwell)."""
        try:
            from stockwell import st
        except ImportError:
            print("[WARN] 'stockwell' library not found. Skipping S-transform.")
            return None

        x = self._ensure_2d(x)
        S_all = []
        for seg in x:
            S = st.st(seg)
            S_all.append(np.abs(S))
        return np.stack(S_all)

    def wvd(self, x):
        """Wigner-Ville Distribution."""
        try:
            from tftb.processing import WignerVilleDistribution
        except ImportError:
            print("[WARN] 'tftb' library not found. Skipping WVD.")
            return None

        x = self._ensure_2d(x)
        W_all = []
        for seg in x:
            wvd_inst = WignerVilleDistribution(seg)
            W, _, _ = wvd_inst.run()
            W_all.append(np.abs(W))
        return np.stack(W_all)