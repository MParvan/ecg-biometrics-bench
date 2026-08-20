import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import utils
from load_dataset import DEFAULT_PREPROCESSING_CONFIG, _normalize_preprocessing_config
from preprocessing import Preprocessing


def synthetic_ecg(fs=500.0, beats=12, bpm=60.0, s_wave_depth=0.0, inverted=False):
    """
    Build an ECG-shaped signal whose R-peak positions are known exactly.

    Each beat carries a narrow R deflection, an optional deeper S trough just
    after it, and a broad T wave later in the cycle. Knowing the true peak
    positions is what lets alignment be measured rather than eyeballed.

    Returns:
        tuple: (signal, sampling rate, true R-peak indices).
    """
    interval = int(round(60.0 / bpm * fs))
    total = interval * beats
    time = np.arange(total)
    signal = np.zeros(total, dtype=float)
    peaks = []

    for index in range(beats):
        centre = index * interval + interval // 2
        peaks.append(centre)

        # R wave: narrow and tall.
        width = max(2.0, 0.012 * fs)
        signal += 1.0 * np.exp(-0.5 * ((time - centre) / width) ** 2)

        if s_wave_depth:
            trough = centre + int(0.045 * fs)
            signal -= s_wave_depth * np.exp(
                -0.5 * ((time - trough) / (width * 1.4)) ** 2
            )

        # T wave: broad, low, and well after the R peak.
        t_centre = centre + int(0.30 * fs)
        signal += 0.22 * np.exp(-0.5 * ((time - t_centre) / (0.055 * fs)) ** 2)

    if inverted:
        signal = -signal

    return signal, fs, np.array(peaks, dtype=int)


class AlignmentWindowTests(unittest.TestCase):
    """
    The search has to be wide enough to reach past a detector's reporting lag.
    Its width is a duration, not a function of heart rate: the lag comes from
    each detector's smoothing window, which is defined in seconds.
    """

    def setUp(self):
        self.prep = Preprocessing()

    def test_the_width_is_the_requested_duration(self):
        for fs in (128.0, 250.0, 360.0, 500.0, 1000.0):
            samples = self.prep.alignment_window(fs, 0.10)
            self.assertAlmostEqual(samples / fs, 0.10, places=2)

    def test_an_explicit_width_is_honoured(self):
        self.assertAlmostEqual(
            self.prep.alignment_window(500.0, 0.15) / 500.0, 0.15, places=5
        )

    def test_the_default_covers_the_largest_reported_detector_lag(self):
        # Ported detectors report the peak of a smoothing window of up to
        # 120 ms, lagging by about half of it. A search narrower than that
        # cannot reach the true peak.
        default = self.prep.DEFAULT_ALIGN_WINDOW_S
        self.assertGreaterEqual(default * 1000.0, 70.0)

    def test_the_beat_extent_does_not_limit_the_search(self):
        # pre_s and post_s describe the beat that is cut once the peak is
        # known. The search runs on the recording, so a short pre_s must not
        # shrink it.
        fs = 500.0
        signal, _, peaks = synthetic_ecg(fs=fs)
        lagged = peaks + int(0.08 * fs)

        beats = self.prep.cut_beats(
            signal, lagged, int(fs), 0.05, 0.4,
            align_peak=True, align_window_s=0.10,
        )

        centre = int(round(0.05 * fs))
        errors = [abs(int(np.argmax(beat)) - centre) for beat in beats]
        self.assertTrue(errors)
        self.assertLessEqual(max(errors), 2)

    def test_a_very_small_width_still_yields_one_sample(self):
        self.assertGreaterEqual(self.prep.alignment_window(128.0, 0.0001), 1)


class PolarityTests(unittest.TestCase):
    """
    Choosing the largest absolute deviation per beat lets a deep S-wave win
    once the search widens, so polarity is decided once per recording.
    """

    def setUp(self):
        self.prep = Preprocessing()

    def test_upright_recording_is_positive(self):
        signal, fs, peaks = synthetic_ecg()
        window = self.prep.alignment_window(fs)
        self.assertEqual(self.prep.peak_polarity(signal, peaks, fs, window), 1)

    def test_inverted_recording_is_negative(self):
        signal, fs, peaks = synthetic_ecg(inverted=True)
        window = self.prep.alignment_window(fs)
        self.assertEqual(self.prep.peak_polarity(signal, peaks, fs, window), -1)

    def test_a_deep_s_wave_does_not_flip_an_upright_recording(self):
        signal, fs, peaks = synthetic_ecg(s_wave_depth=0.85)
        window = self.prep.alignment_window(fs)
        self.assertEqual(self.prep.peak_polarity(signal, peaks, fs, window), 1)


class BeatCentringTests(unittest.TestCase):
    """
    A beat is only usable if the fiducial point sits where the model expects it.
    """

    def setUp(self):
        self.prep = Preprocessing()
        self.pre_s = 0.2
        self.post_s = 0.4

    def _centring_error_ms(self, signal, fs, detections, **kwargs):
        beats = self.prep.cut_beats(
            signal, detections, int(fs), self.pre_s, self.post_s,
            align_peak=True, **kwargs,
        )
        self.assertTrue(beats, "no beats were produced")
        centre = int(round(self.pre_s * fs))
        errors = [abs(int(np.argmax(beat)) - centre) for beat in beats]
        return float(np.median(errors)) / fs * 1000.0

    def test_a_lagged_detector_is_recovered(self):
        # Reproduces what the ported detectors report: every peak marked late
        # by half of a 120 ms smoothing window.
        signal, fs, peaks = synthetic_ecg()
        lag = int(round(0.060 * fs))
        lagged = peaks + lag

        self.assertLess(self._centring_error_ms(signal, fs, lagged), 5.0)

    def test_the_previous_fixed_window_could_not_recover_that_lag(self):
        # Guards the reason the default changed: a 50 ms search cannot reach a
        # 60 ms lag, so the beat stays off-centre.
        signal, fs, peaks = synthetic_ecg()
        lagged = peaks + int(round(0.060 * fs))

        error = self._centring_error_ms(signal, fs, lagged, align_window_s=0.05)
        self.assertGreater(error, 5.0)

    def test_an_inverted_recording_is_centred_on_its_trough(self):
        signal, fs, peaks = synthetic_ecg(inverted=True)
        lagged = peaks + int(round(0.040 * fs))

        beats = self.prep.cut_beats(
            signal, lagged, int(fs), self.pre_s, self.post_s, align_peak=True,
        )
        centre = int(round(self.pre_s * fs))
        errors = [abs(int(np.argmin(beat)) - centre) for beat in beats]
        self.assertLess(float(np.median(errors)) / fs * 1000.0, 5.0)

    def test_a_deep_s_wave_does_not_capture_the_alignment(self):
        signal, fs, peaks = synthetic_ecg(s_wave_depth=0.85)
        lagged = peaks + int(round(0.050 * fs))

        self.assertLess(self._centring_error_ms(signal, fs, lagged), 5.0)

    def test_alignment_can_be_disabled(self):
        signal, fs, peaks = synthetic_ecg()
        lagged = peaks + int(round(0.060 * fs))

        beats = self.prep.cut_beats(
            signal, lagged, int(fs), self.pre_s, self.post_s, align_peak=False,
        )
        centre = int(round(self.pre_s * fs))
        errors = [abs(int(np.argmax(beat)) - centre) for beat in beats]
        self.assertGreater(float(np.median(errors)) / fs * 1000.0, 50.0)

    def test_a_window_wider_than_the_requested_beat_still_centres(self):
        # The search runs on the recording, so a half-width larger than the
        # segment being cut is honoured and the peak still lands at the centre.
        signal, fs, peaks = synthetic_ecg()

        beats = self.prep.cut_beats(
            signal, peaks, int(fs), 0.06, 0.4,
            align_peak=True, align_window_s=0.20,
        )
        centre = int(round(0.06 * fs))
        errors = [abs(int(np.argmax(beat)) - centre) for beat in beats]
        self.assertLess(float(np.median(errors)) / fs * 1000.0, 5.0)

    def test_detections_refining_to_one_sample_yield_one_beat(self):
        # Several detections around a single QRS describe one fiducial
        # location, so one beat is cut rather than near-identical copies.
        signal, fs, peaks = synthetic_ecg()
        crowded = np.sort(
            np.concatenate([peaks, peaks + 4, peaks - 5])
        )

        beats = self.prep.cut_beats(
            signal, crowded, int(fs), self.pre_s, self.post_s,
            align_peak=True, align_window_s=0.10,
        )

        self.assertEqual(len(beats), len(peaks))

    def test_beats_keep_their_requested_length(self):
        signal, fs, peaks = synthetic_ecg()
        expected = int(round(self.pre_s * fs)) + int(round(self.post_s * fs))

        beats = self.prep.cut_beats(
            signal, peaks + 20, int(fs), self.pre_s, self.post_s, align_peak=True,
        )
        for beat in beats:
            self.assertEqual(len(beat), expected)


class ConfigurationTests(unittest.TestCase):
    """
    The window has to be part of the recorded configuration, because a change
    that leaves the cache identity untouched would be reused silently.
    """

    def test_the_key_is_part_of_the_canonical_configuration(self):
        self.assertIn("align_window_s", DEFAULT_PREPROCESSING_CONFIG)
        self.assertAlmostEqual(
            DEFAULT_PREPROCESSING_CONFIG["align_window_s"], 0.10
        )

    def test_the_key_survives_normalisation(self):
        config = _normalize_preprocessing_config({"align_window_s": 0.12})
        self.assertAlmostEqual(config["align_window_s"], 0.12)

    def test_an_absent_key_takes_the_documented_default(self):
        config = _normalize_preprocessing_config({})
        self.assertAlmostEqual(config["align_window_s"], 0.10)

    def test_an_explicit_null_is_rejected(self):
        # Omitting the key and writing null are different requests. The
        # width is a duration, so null is reported rather than quietly
        # replaced by the default.
        with self.assertRaises(ValueError):
            _normalize_preprocessing_config({"align_window_s": None})

    def test_invalid_windows_are_rejected(self):
        for value in (0, -0.1, "wide", True, float("nan"), None):
            with self.assertRaises(ValueError):
                _normalize_preprocessing_config({"align_window_s": value})

    def test_the_window_reaches_the_cache_identity(self):
        class Loader:
            cfg = {"preprocessing": {"align_window_s": 0.12}}
            prep_params = {}

        identity = utils._build_loader_cache_identity(Loader())
        self.assertIn("align_window_s", identity["preprocessing"])
        self.assertAlmostEqual(identity["preprocessing"]["align_window_s"], 0.12)

    def test_a_changed_window_changes_the_identity(self):
        class Loader:
            def __init__(self, window):
                self.cfg = {"preprocessing": {"align_window_s": window}}
                self.prep_params = {}

        first = utils._build_loader_cache_identity(Loader(0.05))
        second = utils._build_loader_cache_identity(Loader(0.12))
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
