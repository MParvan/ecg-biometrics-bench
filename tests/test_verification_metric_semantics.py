import sys
import inspect
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

import run

import utils

from utils import (
    _build_verification_curve_artifacts,
    _compute_metrics_verification,
    _interpolate_equal_error_rate,
)


class EqualErrorRateTests(
    unittest.TestCase
):
    def test_linear_interpolation_uses_the_empirical_crossing(self):
        eer = (
            _interpolate_equal_error_rate(
                [
                    0.0,
                    0.2,
                    0.6,
                ],
                [
                    0.8,
                    0.4,
                    0.1,
                ],
            )
        )

        self.assertAlmostEqual(
            eer,
            11.0 / 35.0,
        )


class SubjectDisjointValidationEerTests(
    unittest.TestCase
):
    def test_training_validation_uses_the_empirical_eer_rule(self):
        source = inspect.getsource(
            utils._run_train_loop_unseen_subjects
        )

        self.assertIn(
            "_interpolate_equal_error_rate(",
            source,
        )

        self.assertIn(
            "_validate_verification_curve_inputs(",
            source,
        )

        self.assertNotIn(
            "brentq",
            source,
        )

        self.assertNotIn(
            "interp1d",
            source,
        )

        self.assertNotIn(
            "except: pass",
            source,
        )


class VerificationOperatingPointTests(
    unittest.TestCase
):
    def test_unresolvable_far_has_no_numeric_operating_point(self):
        scores = np.asarray(
            [
                0.95,
                0.85,
                0.75,
                0.65,
                0.55,
                0.45,
                0.35,
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

        labels = np.asarray(
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

        artifacts = (
            _build_verification_curve_artifacts(
                scores,
                labels,
                target_fars=[
                    0.001,
                ],
            )
        )

        point = artifacts[
            "operating_points"
        ][0]

        self.assertFalse(
            point[
                "empirically_resolvable"
            ]
        )

        for field in (
            "observed_far",
            "tar",
            "frr",
            "threshold",
        ):
            self.assertIsNone(
                point[field]
            )

        metrics = (
            _compute_metrics_verification(
                scores,
                labels,
            )
        )

        self.assertIsNone(
            metrics[3]
        )

    def test_scalar_eer_and_tar_match_the_artifact_semantics(self):
        generator = (
            np.random.default_rng(
                7
            )
        )

        genuine = generator.normal(
            0.7,
            0.2,
            40,
        )

        impostor = generator.normal(
            0.2,
            0.2,
            2000,
        )

        scores = np.concatenate(
            [
                genuine,
                impostor,
            ]
        )

        labels = np.concatenate(
            [
                np.ones(
                    len(genuine),
                    dtype=int,
                ),
                np.zeros(
                    len(impostor),
                    dtype=int,
                ),
            ]
        )

        artifacts = (
            _build_verification_curve_artifacts(
                scores,
                labels,
                target_fars=[
                    0.001,
                ],
            )
        )

        metrics = (
            _compute_metrics_verification(
                scores,
                labels,
            )
        )

        point = artifacts[
            "operating_points"
        ][0]

        self.assertTrue(
            point[
                "empirically_resolvable"
            ]
        )

        self.assertEqual(
            metrics[0],
            artifacts["eer"],
        )

        self.assertEqual(
            metrics[1],
            artifacts["roc_auc"],
        )

        self.assertEqual(
            metrics[3],
            point["tar"],
        )




class MissingAwareAggregationTests(
    unittest.TestCase
):
    def test_unavailable_run_makes_the_aggregate_unavailable(self):
        aggregates = (
            run._aggregate_multi_run_metrics(
                [
                    (
                        0.10,
                        0.90,
                        1.20,
                        0.40,
                    ),
                    (
                        0.20,
                        0.80,
                        1.00,
                        None,
                    ),
                ]
            )
        )

        self.assertAlmostEqual(
            aggregates[0][0],
            0.15,
        )

        self.assertAlmostEqual(
            aggregates[1][0],
            0.85,
        )

        self.assertAlmostEqual(
            aggregates[2][0],
            1.10,
        )

        self.assertEqual(
            aggregates[3],
            (
                None,
                None,
            ),
        )

        self.assertIsNone(
            run._format_multi_run_metric(
                aggregates[3]
            )
        )

    def test_uncertainty_is_unavailable_when_any_run_is_missing(self):
        self.assertIsNone(
            run._summarize_metric_uncertainty(
                [
                    0.1,
                    None,
                    0.2,
                ]
            )
        )

    def test_nonfinite_values_serialize_as_none(self):
        converted = (
            run._to_json_compatible(
                {
                    "nan": float("nan"),
                    "positive_infinity": float(
                        "inf"
                    ),
                }
            )
        )

        self.assertIsNone(
            converted["nan"]
        )

        self.assertIsNone(
            converted[
                "positive_infinity"
            ]
        )


if __name__ == "__main__":
    unittest.main()
