"""
Pin the alignment search where it meets the ends of a recording.

The search refines each detection to the largest deflection within
``align_window_s`` of it, and it runs on the recording rather than inside the
segment that is cut afterwards. The ends of the recording are therefore the
only thing that limits how far it may look, and a detection close to either end
has to be refined from the samples that exist rather than from a window that
runs off the array.

The beat itself is cut around the refined position, so a detection near the
start can still be dropped: the cut needs ``pre_s`` seconds of signal ahead of
the peak whether or not the search succeeded.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import Preprocessing


FS = 200
PRE_S, POST_S = 0.2, 0.4
WINDOW_S = 0.10


def spiked_signal(length, peak_positions, amplitude=5.0):
    signal = 0.01 * np.sin(np.arange(length) / 7.0)
    for position in peak_positions:
        signal[position] = amplitude
    return signal


class SearchMeetsTheRecordingEdges(unittest.TestCase):
    def setUp(self):
        self.preprocessor = Preprocessing()

    def cut(self, signal, peaks, pre_s=PRE_S, post_s=POST_S):
        return self.preprocessor.cut_beats(
            signal,
            np.asarray(peaks),
            FS,
            pre_s,
            post_s,
            align_peak=True,
            align_window_s=WINDOW_S,
        )

    def test_search_that_reaches_before_the_first_sample(self):
        # The detection sits closer to the start than the search half-width,
        # so the window has to begin at sample zero and the refined position
        # must still be the true peak.
        window = int(WINDOW_S * FS)
        true_peak = window // 2
        signal = spiked_signal(2000, [true_peak])

        refined = self.preprocessor.alignment_window(FS, WINDOW_S)
        self.assertEqual(refined, window)

        low = max(0, (true_peak + 3) - refined)
        high = min(len(signal), (true_peak + 3) + refined + 1)
        self.assertEqual(low, 0)
        self.assertEqual(
            low + int(np.argmax(signal[low:high])),
            true_peak,
            "the refined position should be the peak itself",
        )

    def test_search_that_reaches_past_the_last_sample(self):
        length = 1000
        window = int(WINDOW_S * FS)
        true_peak = length - window // 2
        signal = spiked_signal(length, [true_peak])

        refined = self.preprocessor.alignment_window(FS, WINDOW_S)
        low = max(0, (true_peak - 3) - refined)
        high = min(length, (true_peak - 3) + refined + 1)
        self.assertEqual(high, length)
        self.assertEqual(low + int(np.argmax(signal[low:high])), true_peak)

    def test_a_beat_too_close_to_the_start_is_not_cut(self):
        # Refinement can succeed while the cut still cannot: the segment needs
        # pre_s seconds of signal ahead of the peak.
        pre = int(round(PRE_S * FS))
        signal = spiked_signal(2000, [pre // 2])

        beats = self.cut(signal, [pre // 2])

        self.assertEqual(beats, [])

    def test_interior_beats_are_centred(self):
        pre = int(round(PRE_S * FS))
        peaks = [400, 700, 1000, 1300]
        signal = spiked_signal(2000, peaks)

        beats = self.cut(signal, [position - 8 for position in peaks])

        self.assertEqual(len(beats), len(peaks))
        for beat in beats:
            self.assertEqual(int(np.argmax(beat)), pre)

    def test_every_beat_keeps_the_requested_length(self):
        expected = int(round(PRE_S * FS)) + int(round(POST_S * FS))
        peaks = [400, 700, 1000, 1300]
        signal = spiked_signal(2000, peaks)

        beats = self.cut(signal, [position - 8 for position in peaks])

        self.assertTrue(beats)
        for beat in beats:
            self.assertEqual(len(beat), expected)

    def test_a_short_pre_window_does_not_shorten_the_search(self):
        # pre_s describes the segment that is cut, not how far the search may
        # look, so a detection lagging by more than pre_s is still recovered.
        pre_s = 0.03
        pre = int(round(pre_s * FS))
        true_peak = 600
        signal = spiked_signal(2000, [true_peak])

        beats = self.cut(signal, [true_peak + 12], pre_s=pre_s)

        self.assertEqual(len(beats), 1)
        self.assertEqual(int(np.argmax(beats[0])), pre)


class RepeatedRefinement(unittest.TestCase):
    """
    Detections that refine onto one sample describe a single fiducial location,
    so one beat is cut rather than several copies of the same segment.
    """

    def setUp(self):
        self.preprocessor = Preprocessing()

    def test_detections_on_one_peak_yield_one_beat(self):
        true_peak = 800
        signal = spiked_signal(2000, [true_peak])

        beats = self.preprocessor.cut_beats(
            signal,
            np.array([true_peak - 6, true_peak, true_peak + 5]),
            FS,
            PRE_S,
            POST_S,
            align_peak=True,
            align_window_s=WINDOW_S,
        )

        self.assertEqual(len(beats), 1)
        self.assertEqual(int(np.argmax(beats[0])), int(round(PRE_S * FS)))

    def test_distinct_peaks_are_all_kept(self):
        peaks = [400, 700, 1000]
        signal = spiked_signal(2000, peaks)

        beats = self.preprocessor.cut_beats(
            signal,
            np.asarray([position - 4 for position in peaks]),
            FS,
            PRE_S,
            POST_S,
            align_peak=True,
            align_window_s=WINDOW_S,
        )

        self.assertEqual(len(beats), len(peaks))


if __name__ == "__main__":
    unittest.main()
