import json
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from utils import (
    _build_identification_curve_artifacts,
    _build_verification_curve_artifacts,
    _compute_metrics_identification,
)


class VerificationCurveArtifactTests(
    unittest.TestCase
):
    def setUp(self):
        self.scores = np.asarray(
            [
                0.95,
                0.75,
                0.55,
                0.35,
                0.85,
                0.65,
                0.45,
                0.25,
                0.15,
                0.05,
                -0.05,
                -0.15,
                -0.25,
                -0.35,
            ],
            dtype=float,
        )

        self.labels = np.asarray(
            [
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=int,
        )

    def test_standard_far_targets_and_curves_are_returned(self):
        artifacts = (
            _build_verification_curve_artifacts(
                self.scores,
                self.labels,
            )
        )

        self.assertEqual(
            [
                point["target_far"]
                for point in artifacts[
                    "operating_points"
                ]
            ],
            [
                0.1,
                0.01,
                0.001,
                0.0001,
            ],
        )

        self.assertEqual(
            artifacts[
                "comparison_counts"
            ]["genuine"],
            4,
        )
        self.assertEqual(
            artifacts[
                "comparison_counts"
            ]["impostor"],
            10,
        )

        self.assertEqual(
            len(
                artifacts[
                    "roc_curve"
                ][
                    "false_accept_rates"
                ]
            ),
            len(
                artifacts[
                    "roc_curve"
                ][
                    "true_accept_rates"
                ]
            ),
        )

        self.assertEqual(
            len(
                artifacts[
                    "det_curve"
                ][
                    "false_accept_rates"
                ]
            ),
            len(
                artifacts[
                    "det_curve"
                ][
                    "false_reject_rates"
                ]
            ),
        )

    def test_operating_points_reproduce_threshold_rates(self):
        artifacts = (
            _build_verification_curve_artifacts(
                self.scores,
                self.labels,
            )
        )

        genuine_scores = self.scores[
            self.labels == 1
        ]
        impostor_scores = self.scores[
            self.labels == 0
        ]

        for point in artifacts[
            "operating_points"
        ]:
            threshold = point[
                "threshold"
            ]

            self.assertIsNotNone(
                threshold
            )

            observed_far = np.mean(
                impostor_scores
                >= threshold
            )
            observed_tar = np.mean(
                genuine_scores
                >= threshold
            )

            self.assertAlmostEqual(
                point["observed_far"],
                observed_far,
            )
            self.assertAlmostEqual(
                point["tar"],
                observed_tar,
            )
            self.assertLessEqual(
                point["observed_far"],
                (
                    point["target_far"]
                    + 1e-12
                ),
            )

    def test_far_resolution_uses_impostor_count(self):
        artifacts = (
            _build_verification_curve_artifacts(
                self.scores,
                self.labels,
            )
        )

        self.assertAlmostEqual(
            artifacts[
                "comparison_counts"
            ][
                "minimum_nonzero_far"
            ],
            0.1,
        )

        resolution = {
            point["target_far"]: (
                point[
                    "empirically_resolvable"
                ]
            )
            for point in artifacts[
                "operating_points"
            ]
        }

        self.assertTrue(
            resolution[0.1]
        )
        self.assertFalse(
            resolution[0.01]
        )
        self.assertFalse(
            resolution[0.001]
        )
        self.assertFalse(
            resolution[0.0001]
        )

    def test_both_pair_classes_are_required(self):
        with self.assertRaisesRegex(
            ValueError,
            "both genuine and impostor",
        ):
            _build_verification_curve_artifacts(
                [
                    0.9,
                    0.8,
                ],
                [
                    1,
                    1,
                ],
            )


class IdentificationCurveArtifactTests(
    unittest.TestCase
):
    def setUp(self):
        self.scores = np.asarray(
            [
                [
                    0.90,
                    0.10,
                    0.00,
                ],
                [
                    0.80,
                    0.90,
                    0.10,
                ],
                [
                    0.30,
                    0.20,
                    0.10,
                ],
            ],
            dtype=float,
        )

        self.labels = np.asarray(
            [
                0,
                0,
                2,
            ],
            dtype=int,
        )

    def test_complete_cmc_curve_is_exact(self):
        artifacts = (
            _build_identification_curve_artifacts(
                self.scores,
                self.labels,
            )
        )

        self.assertEqual(
            artifacts[
                "correct_match_ranks"
            ],
            [
                1,
                2,
                3,
            ],
        )
        self.assertEqual(
            artifacts[
                "cmc_curve"
            ]["ranks"],
            [
                1,
                2,
                3,
            ],
        )

        np.testing.assert_allclose(
            artifacts[
                "cmc_curve"
            ][
                "identification_rates"
            ],
            [
                1.0 / 3.0,
                2.0 / 3.0,
                1.0,
            ],
        )

        self.assertAlmostEqual(
            artifacts[
                "rank_1_accuracy"
            ],
            1.0 / 3.0,
        )
        self.assertEqual(
            artifacts[
                "rank_5_accuracy"
            ],
            1.0,
        )
        self.assertEqual(
            artifacts[
                "effective_rank_5"
            ],
            3,
        )

    def test_existing_metrics_are_derived_from_cmc(self):
        artifacts = (
            _build_identification_curve_artifacts(
                self.scores,
                self.labels,
            )
        )

        rank1, rank5 = (
            _compute_metrics_identification(
                self.scores,
                self.labels,
            )
        )

        self.assertEqual(
            rank1,
            artifacts[
                "rank_1_accuracy"
            ],
        )
        self.assertEqual(
            rank5,
            artifacts[
                "rank_5_accuracy"
            ],
        )

    def test_out_of_range_gallery_labels_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "valid gallery columns",
        ):
            _build_identification_curve_artifacts(
                self.scores,
                [
                    0,
                    1,
                    3,
                ],
            )


class StrictSerializationTests(
    unittest.TestCase
):
    def test_curve_artifacts_are_strict_json_serializable(self):
        verification = (
            _build_verification_curve_artifacts(
                [
                    0.9,
                    0.7,
                    0.8,
                    0.6,
                ],
                [
                    1,
                    1,
                    0,
                    0,
                ],
            )
        )

        identification = (
            _build_identification_curve_artifacts(
                [
                    [
                        0.9,
                        0.1,
                    ],
                    [
                        0.2,
                        0.8,
                    ],
                ],
                [
                    0,
                    1,
                ],
            )
        )

        verification_json = json.dumps(
            verification,
            allow_nan=False,
        )
        identification_json = json.dumps(
            identification,
            allow_nan=False,
        )

        self.assertIsInstance(
            verification_json,
            str,
        )
        self.assertIsInstance(
            identification_json,
            str,
        )


if __name__ == "__main__":
    unittest.main()
