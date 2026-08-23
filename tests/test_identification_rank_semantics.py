import unittest

import numpy as np

import run
from utils import (
    _build_identification_curve_artifacts,
    _compute_metrics_identification,
)


class IdentificationRankSemanticsTests(
    unittest.TestCase
):
    def test_rank_5_is_unavailable_below_five_gallery_identities(
        self,
    ):
        scores = np.asarray(
            [
                [
                    0.9,
                    0.1,
                    0.0,
                    -0.1,
                ],
                [
                    0.8,
                    0.7,
                    0.6,
                    0.5,
                ],
            ],
            dtype=float,
        )
        labels = np.asarray(
            [
                0,
                3,
            ],
            dtype=int,
        )

        artifacts = (
            _build_identification_curve_artifacts(
                scores,
                labels,
            )
        )

        self.assertIsNone(
            artifacts[
                "rank_5_accuracy"
            ]
        )
        self.assertIsNone(
            artifacts[
                "effective_rank_5"
            ]
        )
        self.assertFalse(
            artifacts[
                "rank_5_defined"
            ]
        )
        self.assertFalse(
            artifacts[
                "rank_5_reportable"
            ]
        )
        self.assertEqual(
            artifacts[
                "rank_5_reportability_reason"
            ],
            "gallery_size_below_5",
        )

        _, rank_5 = (
            _compute_metrics_identification(
                scores,
                labels,
            )
        )

        self.assertIsNone(
            rank_5
        )

    def test_terminal_rank_5_is_defined_but_not_reportable(
        self,
    ):
        scores = np.asarray(
            [
                [
                    0.9,
                    0.8,
                    0.7,
                    0.6,
                    0.5,
                ],
                [
                    0.9,
                    0.8,
                    0.7,
                    0.6,
                    0.1,
                ],
            ],
            dtype=float,
        )
        labels = np.asarray(
            [
                0,
                4,
            ],
            dtype=int,
        )

        artifacts = (
            _build_identification_curve_artifacts(
                scores,
                labels,
            )
        )

        self.assertEqual(
            artifacts[
                "gallery_size"
            ],
            5,
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
            5,
        )
        self.assertTrue(
            artifacts[
                "rank_5_defined"
            ]
        )
        self.assertFalse(
            artifacts[
                "rank_5_reportable"
            ]
        )
        self.assertEqual(
            artifacts[
                "rank_5_reportability_reason"
            ],
            "terminal_cmc_rank",
        )

        _, returned_rank_5 = (
            _compute_metrics_identification(
                scores,
                labels,
            )
        )

        self.assertIsNone(
            returned_rank_5
        )

    def test_rank_5_is_reportable_above_five_gallery_identities(
        self,
    ):
        scores = np.asarray(
            [
                [
                    0.2,
                    0.9,
                    0.8,
                    0.7,
                    0.6,
                    0.1,
                ],
                [
                    0.9,
                    0.8,
                    0.7,
                    0.6,
                    0.5,
                    0.1,
                ],
            ],
            dtype=float,
        )
        labels = np.asarray(
            [
                0,
                5,
            ],
            dtype=int,
        )

        artifacts = (
            _build_identification_curve_artifacts(
                scores,
                labels,
            )
        )

        self.assertEqual(
            artifacts[
                "gallery_size"
            ],
            6,
        )
        self.assertAlmostEqual(
            artifacts[
                "rank_5_accuracy"
            ],
            0.5,
        )
        self.assertEqual(
            artifacts[
                "effective_rank_5"
            ],
            5,
        )
        self.assertTrue(
            artifacts[
                "rank_5_defined"
            ]
        )
        self.assertTrue(
            artifacts[
                "rank_5_reportable"
            ]
        )
        self.assertIsNone(
            artifacts[
                "rank_5_reportability_reason"
            ]
        )
        self.assertEqual(
            artifacts[
                "cmc_curve"
            ][
                "identification_rates"
            ][-1],
            1.0,
        )

        _, returned_rank_5 = (
            _compute_metrics_identification(
                scores,
                labels,
            )
        )

        self.assertAlmostEqual(
            returned_rank_5,
            0.5,
        )

    def test_nonreportable_seed_masks_rank_5_before_aggregation(
        self,
    ):
        results = [
            (
                0.20,
                1.00,
            ),
            (
                0.40,
                0.75,
            ),
        ]
        artifacts = [
            {
                "type": "identification",
                "rank_5_reportable": False,
            },
            {
                "type": "identification",
                "rank_5_reportable": True,
            },
        ]

        reportable_results = (
            run._apply_identification_metric_reportability(
                results,
                artifacts,
            )
        )

        self.assertEqual(
            reportable_results,
            [
                (
                    0.20,
                    None,
                ),
                (
                    0.40,
                    0.75,
                ),
            ],
        )

        aggregates = (
            run._aggregate_multi_run_metrics(
                reportable_results
            )
        )

        self.assertAlmostEqual(
            aggregates[0][0],
            0.30,
        )
        self.assertEqual(
            aggregates[1],
            (
                None,
                None,
            ),
        )

    def test_reportable_rank_5_aggregates_across_complete_runs(
        self,
    ):
        reportable_results = (
            run._apply_identification_metric_reportability(
                [
                    (
                        0.20,
                        0.70,
                    ),
                    (
                        0.40,
                        0.80,
                    ),
                ],
                [
                    {
                        "type": "identification",
                        "rank_5_reportable": True,
                    },
                    {
                        "type": "identification",
                        "rank_5_reportable": True,
                    },
                ],
            )
        )

        aggregates = (
            run._aggregate_multi_run_metrics(
                reportable_results
            )
        )

        self.assertAlmostEqual(
            aggregates[0][0],
            0.30,
        )
        self.assertAlmostEqual(
            aggregates[1][0],
            0.75,
        )

    def test_result_and_artifact_counts_must_match(
        self,
    ):
        with self.assertRaisesRegex(
            ValueError,
            "same number of runs",
        ):
            run._apply_identification_metric_reportability(
                [
                    (
                        0.20,
                        0.70,
                    ),
                ],
                [],
            )


if __name__ == "__main__":
    unittest.main()
