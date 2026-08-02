import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import load_dataset
from load_dataset import (
    _beat_merge_start_indices,
    _normalize_beat_merge_stride,
)

LOADER_CLASS_NAMES = (
    "load_ecgid_dataset",
    "load_heartprint_dataset",
    "load_ptb_dataset",
    "load_cybhi_dataset",
    "load_mitbih_dataset",
    "load_nsrdb_dataset",
    "load_ptbxl_dataset",
)


class StrideNormalizationTests(unittest.TestCase):
    """
    The stride must be a whole number that never skips beats.
    """

    def test_default_stride_is_one(self):
        self.assertEqual(
            _normalize_beat_merge_stride(None, 3),
            1,
        )

    def test_stride_may_equal_the_merge_width(self):
        self.assertEqual(
            _normalize_beat_merge_stride(3, 3),
            3,
        )

    def test_stride_larger_than_merge_width_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            _normalize_beat_merge_stride(4, 3)

        self.assertIn(
            "cannot exceed num_beats_to_merge",
            str(raised.exception),
        )

    def test_zero_stride_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            _normalize_beat_merge_stride(0, 3)

        self.assertIn(
            "at least 1",
            str(raised.exception),
        )

    def test_fractional_stride_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            _normalize_beat_merge_stride(1.5, 3)

        self.assertIn(
            "whole number",
            str(raised.exception),
        )

    def test_boolean_stride_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            _normalize_beat_merge_stride(True, 3)

        self.assertIn(
            "not a Boolean",
            str(raised.exception),
        )


class MergeWindowTests(unittest.TestCase):
    """
    Window start indices define whether merged samples share beats.
    """

    def test_default_stride_reproduces_sliding_windows(self):
        self.assertEqual(
            list(_beat_merge_start_indices(10, 3, 1)),
            list(range(0, 8)),
        )

    def test_matching_stride_produces_disjoint_windows(self):
        starts = list(
            _beat_merge_start_indices(10, 3, 3)
        )

        self.assertEqual(starts, [0, 3, 6])

        for previous, current in zip(starts, starts[1:]):
            self.assertGreaterEqual(
                current,
                previous + 3,
            )

    def test_too_few_beats_produce_no_windows(self):
        self.assertEqual(
            list(_beat_merge_start_indices(2, 3, 1)),
            [],
        )

    def test_single_beat_merging_is_unaffected(self):
        self.assertEqual(
            list(_beat_merge_start_indices(4, 1, 1)),
            [0, 1, 2, 3],
        )


class SyntheticMergeBehaviourTests(unittest.TestCase):
    """
    Verify merge behaviour on the real loader code path with synthetic beats.
    """

    def _merge(self, beats, num_beats, stride):
        loader = load_dataset.load_ecgid_dataset.__new__(
            load_dataset.load_ecgid_dataset
        )
        loader.num_beats = num_beats
        loader.merge_strategy = "average"
        loader.beat_merge_stride = stride

        merged = []
        starts = _beat_merge_start_indices(
            len(beats),
            num_beats,
            stride,
        )

        for start in starts:
            merged.append(
                np.mean(
                    beats[start:start + num_beats],
                    axis=0,
                )
            )

        return np.asarray(merged)

    def test_default_stride_output_is_unchanged(self):
        beats = np.arange(24, dtype=float).reshape(8, 3)

        legacy = np.asarray(
            [
                np.mean(beats[index:index + 3], axis=0)
                for index in range(0, len(beats) - 3 + 1)
            ]
        )

        np.testing.assert_array_equal(
            self._merge(beats, 3, 1),
            legacy,
        )

    def test_matching_stride_reduces_sample_count(self):
        beats = np.arange(27, dtype=float).reshape(9, 3)

        self.assertEqual(
            len(self._merge(beats, 3, 1)),
            7,
        )
        self.assertEqual(
            len(self._merge(beats, 3, 3)),
            3,
        )


class LoaderInterfaceTests(unittest.TestCase):
    """
    Every dataset loader exposes the same stride option with the same default.
    """

    def test_all_loaders_accept_beat_merge_stride(self):
        for class_name in LOADER_CLASS_NAMES:
            with self.subTest(loader=class_name):
                loader_class = getattr(
                    load_dataset,
                    class_name,
                )

                signature = inspect.signature(
                    loader_class.__init__
                )

                self.assertIn(
                    "beat_merge_stride",
                    signature.parameters,
                )
                self.assertEqual(
                    signature.parameters[
                        "beat_merge_stride"
                    ].default,
                    1,
                )

    def test_loaders_store_the_normalized_stride(self):
        loader = load_dataset.load_ecgid_dataset(
            num_beats_to_merge=4,
            beat_merge_stride=4,
        )

        self.assertEqual(
            loader.beat_merge_stride,
            4,
        )

    def test_invalid_stride_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            load_dataset.load_ecgid_dataset(
                num_beats_to_merge=2,
                beat_merge_stride=5,
            )


if __name__ == "__main__":
    unittest.main()
