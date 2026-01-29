import numpy as np
from scipy.signal import butter, filtfilt, firwin, medfilt, iirnotch, savgol_filter

class Filtering:
    """
    Collection of signal filtering methods commonly used in biomedical
    signal processing (e.g., ECG, PPG, EEG).

    This class is designed to be used as a sub-module of a preprocessing
    pipeline.

    All methods assume a 1D NumPy array as input.
    """

    def __init__(self):
        """
        Constructor for the Filtering class.
        Stateless to allow safe reuse across datasets.
        """
        pass

    def butter(self, x, fs, low=0.5, high=40.0, order=4):
        """
        Apply a Butterworth band-pass IIR filter using zero-phase filtering.

        This is the standard and most commonly used ECG band-pass filter.
        Zero-phase filtering (via filtfilt) ensures no phase distortion.

        Parameters
        ----------
        x : np.ndarray
            Input signal (1D array).
        fs : int or float
            Sampling frequency in Hz.
        low : float
            Low cutoff frequency (default: 0.5 Hz).
        high : float
            High cutoff frequency (default: 40.0 Hz).
        order : int
            Filter order (default: 4).

        Returns
        -------
        np.ndarray
            Filtered signal.
        """
        x = x.astype(np.float64)
        # Handle cases where user might want low-pass or high-pass only
        if low is None and high is None:
            return x
        elif low is None:
            b, a = butter(order, high, btype='low', fs=fs)
        elif high is None:
            b, a = butter(order, low, btype='high', fs=fs)
        else:
            b, a = butter(order, [low, high], btype='band', fs=fs)
            
        return filtfilt(b, a, x)

    def notch(self, x, fs, freq=50.0, quality=30.0):
        """
        Apply a Notch filter to remove powerline interference (50Hz/60Hz).

        Useful when bandwidth > 50Hz is required (e.g., high-res ECG).

        Parameters
        ----------
        x : np.ndarray
            Input signal.
        fs : float
            Sampling frequency.
        freq : float
            Frequency to remove (usually 50.0 or 60.0).
        quality : float
            Quality factor (Q). Higher Q = narrower notch (removes less signal).
            Typical values are 30.0 - 60.0.
        """
        b, a = iirnotch(freq, quality, fs=fs)
        return filtfilt(b, a, x)

    def savgol(self, x, window_length=11, polyorder=2):
        """
        Apply a Savitzky-Golay filter.

        Superior to moving average for ECG because it preserves the 
        amplitude of high-frequency features (like the R-peak) while 
        removing noise.

        Parameters
        ----------
        x : np.ndarray
            Input signal.
        window_length : int
            Length of the filter window (must be odd).
        polyorder : int
            Order of the polynomial used to fit the samples.
        """
        return savgol_filter(x, window_length, polyorder)

    def movingaverage(self, x, window=5):
        """
        Apply a simple moving average (boxcar) filter.
        Good for trends, but blurs QRS complexes.
        """
        window = int(window)
        if window < 1: raise ValueError("Window size must be >= 1")
        return np.convolve(x, np.ones(window) / window, mode='same')

    def median(self, x, kernel=5):
        """
        Apply a median filter for impulse noise (spikes) removal.
        """
        kernel = int(kernel)
        if kernel % 2 == 0: kernel += 1
        return medfilt(x, kernel)

    def fir(self, x, fs, low=0.5, high=40.0, numtaps=101):
        """
        Apply an FIR band-pass filter.
        Unconditionally stable, but computationally heavier than IIR.
        """
        b = firwin(numtaps, [low, high], fs=fs, pass_zero=False)
        return filtfilt(b, [1.0], x)

    def dwt(self, x, wavelet='db6', level=4):
        """
        Apply DWT-based baseline removal.
        Removes the approximation coefficients (low freq) at the given level.
        """
        import pywt
        # Ensure minimal length for decomposition
        max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
        if level > max_level:
            level = max_level
            
        coeffs = pywt.wavedec(x, wavelet, level=level)
        coeffs[0] *= 0  # Zero out the lowest frequency approximation
        return pywt.waverec(coeffs, wavelet)

    def polynomial(self, x, order=3):
        """
        Remove baseline wander using polynomial subtraction.
        """
        t = np.arange(len(x))
        coeffs = np.polyfit(t, x, order)
        baseline = np.polyval(coeffs, t)
        return x - baseline