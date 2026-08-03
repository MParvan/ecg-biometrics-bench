import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from run import _get_verification_pair_statistics
from utils import _summarize_verification_pairs


class VerificationPairStatisticsTests(unittest.TestCase):
    def test_pair_counts_and_far_resolution(self):
        labels = np.array(
            [1] * 100 + [0] * 2000,
            dtype=int,
        )

        summary = _summarize_verification_pairs(
            labels,
            target_far=0.001,
        )

        self.assertEqual(
            summary["Total Verification Pairs"],
            2100,
        )
        self.assertEqual(summary["Genuine Pairs"], 100)
        self.assertEqual(summary["Impostor Pairs"], 2000)
        self.assertAlmostEqual(
            summary["Minimum Non-Zero Empirical FAR"],
            0.0005,
        )
        self.assertTrue(
            summary["Target FAR Empirically Resolvable"]
        )

    def test_target_far_is_not_resolvable_with_too_few_impostors(self):
        labels = np.array(
            [1] * 50 + [0] * 500,
            dtype=int,
        )

        summary = _summarize_verification_pairs(
            labels,
            target_far=0.001,
        )

        self.assertAlmostEqual(
            summary["Minimum Non-Zero Empirical FAR"],
            0.002,
        )
        self.assertFalse(
            summary["Target FAR Empirically Resolvable"]
        )

    def test_warning_is_printed_for_insufficient_far_resolution(self):
        labels = np.array(
            [1] * 10 + [0] * 100,
            dtype=int,
        )

        captured_output = StringIO()

        with redirect_stdout(captured_output):
            summary = _get_verification_pair_statistics(
                labels,
                target_far=0.001,
            )

        self.assertFalse(
            summary["Target FAR Empirically Resolvable"]
        )
        self.assertIn(
            "below the empirical FAR resolution",
            captured_output.getvalue(),
        )

    def test_nonbinary_pair_labels_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "binary labels",
        ):
            _summarize_verification_pairs(
                np.array([0, 1, 2]),
            )

    def test_empty_pair_labels_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "cannot be empty",
        ):
            _summarize_verification_pairs(
                np.array([]),
            )


if __name__ == "__main__":
    unittest.main()