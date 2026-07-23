import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import _apply_outlier_filter


class OutlierFilterTests(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(24, dtype=np.float32).reshape(6, 4)
        self.y = np.array(["A", "A", "A", "B", "B", "B"])

        # Subject A: 0.2, 0.9, 0.8
        # Subject B: 0.7, 0.1, 0.6
        self.sqi = np.array([0.2, 0.9, 0.8, 0.7, 0.1, 0.6])

    def test_enrollment_mode_can_keep_top_fraction_per_subject(self):
        filtered_x, filtered_y = _apply_outlier_filter(
            self.x,
            self.y,
            self.sqi,
            absolute_threshold=0.0,
            keep_percentage=0.5,
            apply_subject_ranking=True,
        )

        # int(3 * 0.5) = 1 sample retained for each subject.
        self.assertEqual(len(filtered_x), 2)
        self.assertEqual(filtered_y.tolist(), ["A", "B"])

        # Highest-quality samples are index 1 for A and index 3 for B.
        np.testing.assert_array_equal(filtered_x[0], self.x[1])
        np.testing.assert_array_equal(filtered_x[1], self.x[3])

    def test_probe_mode_uses_only_absolute_threshold(self):
        filtered_x, filtered_y = _apply_outlier_filter(
            self.x,
            self.y,
            self.sqi,
            absolute_threshold=0.5,
            keep_percentage=0.1,
            apply_subject_ranking=False,
        )

        # All samples meeting the threshold are retained, regardless of identity.
        expected_indices = [1, 2, 3, 5]

        np.testing.assert_array_equal(
            filtered_x,
            self.x[expected_indices],
        )
        np.testing.assert_array_equal(
            filtered_y,
            self.y[expected_indices],
        )

    def test_probe_filter_result_does_not_depend_on_subject_labels(self):
        alternative_labels = np.array(
            ["X", "Y", "X", "Y", "X", "Y"]
        )

        filtered_x_original, _ = _apply_outlier_filter(
            self.x,
            self.y,
            self.sqi,
            absolute_threshold=0.5,
            apply_subject_ranking=False,
        )

        filtered_x_alternative, _ = _apply_outlier_filter(
            self.x,
            alternative_labels,
            self.sqi,
            absolute_threshold=0.5,
            apply_subject_ranking=False,
        )

        np.testing.assert_array_equal(
            filtered_x_original,
            filtered_x_alternative,
        )


if __name__ == "__main__":
    unittest.main()