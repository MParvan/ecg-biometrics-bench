import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from load_dataset import load_mitbih_dataset


def write_record(directory, name, minutes, fs=360):
    """
    Write a two-channel WFDB record carrying a slow ramp.

    A monotonic signal makes a misaligned read obvious: any sample says where
    in the recording it came from. The amplitude is kept small because WFDB
    stores integers with a gain, and a large range would be quantised on the
    way to disk.
    """
    samples = int(minutes * 60 * fs)
    ramp = np.linspace(-1.0, 1.0, samples, dtype=np.float64)
    signal = np.stack([ramp, -ramp], axis=1)

    wfdb.wrsamp(
        record_name=name,
        fs=fs,
        units=["mV", "mV"],
        sig_name=["MLII", "V5"],
        p_signal=signal,
        write_dir=str(directory),
    )
    return signal


class RangeReadEquivalenceTests(unittest.TestCase):
    """
    Reading only the requested minutes has to produce exactly what reading the
    whole recording and slicing it produced, including at the boundaries.
    """

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.fs = 360
        write_record(self.directory, "100", minutes=3, fs=self.fs)

        self.loader = load_mitbih_dataset(data_split_mode="all-available")
        self.loader.dataset_root = self.directory

        # The reference is the recording as the loader itself would read it,
        # through the same lead selection and after the same integer round
        # trip, so the comparison isolates the range handling rather than
        # re-testing WFDB's storage format.
        header = wfdb.rdheader(str(self.directory / "100"))
        leads = self.loader._get_lead_indices(header.sig_name)
        self.whole, _ = wfdb.rdsamp(str(self.directory / "100"), channels=leads)

    def _both_paths(self, min_ranges):
        produced = self.loader.load_raw_data(min_ranges=min_ranges)
        reference = self.loader._slice_signal(self.whole, self.fs, min_ranges)
        return produced, reference

    def test_a_single_range_matches(self):
        produced, reference = self._both_paths([(0, 1)])
        np.testing.assert_array_equal(produced["100"]["signal"], reference)

    def test_an_interior_range_matches(self):
        produced, reference = self._both_paths([(1, 2)])
        np.testing.assert_array_equal(produced["100"]["signal"], reference)

    def test_disjoint_ranges_are_concatenated_in_order(self):
        produced, reference = self._both_paths([(0, 1), (2, 3)])
        np.testing.assert_array_equal(produced["100"]["signal"], reference)

    def test_a_range_past_the_end_is_clamped(self):
        # A protocol may ask for minutes the recording does not reach; both
        # paths have to stop at the same sample.
        produced, reference = self._both_paths([(2, 10)])
        np.testing.assert_array_equal(produced["100"]["signal"], reference)

    def test_a_range_entirely_past_the_end_yields_no_record(self):
        produced = self.loader.load_raw_data(min_ranges=[(30, 40)])
        self.assertNotIn("100", produced)

    def test_reading_without_ranges_returns_the_whole_recording(self):
        produced = self.loader.load_raw_data()
        np.testing.assert_array_equal(produced["100"]["signal"], self.whole)

    def test_only_the_requested_span_is_read_from_disk(self):
        # The point of the change: a protocol using two minutes of a half-hour
        # recording should not pay to read the other twenty-eight.
        with mock.patch("load_dataset.wfdb.rdsamp", wraps=wfdb.rdsamp) as reader:
            self.loader.load_raw_data(min_ranges=[(0, 1)])

        self.assertEqual(reader.call_count, 1)
        kwargs = reader.call_args.kwargs
        self.assertEqual(kwargs["sampfrom"], 0)
        self.assertEqual(kwargs["sampto"], 60 * self.fs)


class SessionRoutingTests(unittest.TestCase):
    """Each session must receive its own minutes and nothing else."""

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.fs = 360
        write_record(self.directory, "100", minutes=3, fs=self.fs)

    def test_sessions_read_disjoint_spans(self):
        loader = load_mitbih_dataset(
            data_split_mode="custom-split",
            train_parts=[(0, 1)],
            enrol_parts=[(1, 2)],
            test_parts=[(2, 3)],
        )
        loader.dataset_root = self.directory
        # Reporting the first and last sample of whatever it is given makes the
        # span each session received visible in the returned array.
        loader._process_signal = lambda signal, fs: np.asarray(
            [[signal[0, 0], signal[-1, 0]]], dtype=np.float64
        )

        header = wfdb.rdheader(str(self.directory / "100"))
        leads = loader._get_lead_indices(header.sig_name)
        whole, _ = wfdb.rdsamp(str(self.directory / "100"), channels=leads)

        train, _ = loader.load_session("train")
        enrol, _ = loader.load_session("enrollment")
        probe, _ = loader.load_session("probe")

        span = 60 * self.fs
        for produced, start in ((train, 0), (enrol, span), (probe, 2 * span)):
            np.testing.assert_array_equal(
                produced,
                [[whole[start, 0], whole[start + span - 1, 0]]],
            )

        # The three spans must not overlap, which is the property the custom
        # split exists to provide.
        self.assertNotEqual(train[0][0], enrol[0][0])
        self.assertNotEqual(enrol[0][0], probe[0][0])


if __name__ == "__main__":
    unittest.main()
