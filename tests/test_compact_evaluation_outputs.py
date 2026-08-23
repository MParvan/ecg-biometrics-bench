import json
import sys
import tempfile
import unittest
from datetime import datetime
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


class CompactVerificationOutputTests(
    unittest.TestCase
):
    def test_verification_curves_are_externalized(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_dir = Path(
                temporary_directory
            )

            artifact_items = []

            for run_index, seed in enumerate(
                [42, 43],
                start=1,
            ):
                artifact_items.append(
                    {
                        "run_index": run_index,
                        "seed": seed,
                        "artifact": {
                            "type": "verification",
                            "score_direction": (
                                "higher_is_more_genuine"
                            ),
                            "comparison_counts": {
                                "total": 4,
                                "genuine": 2,
                                "impostor": 2,
                                "minimum_nonzero_far": 0.5,
                            },
                            "roc_auc": 0.75,
                            "operating_points": [
                                {
                                    "name": "TAR@10%FAR",
                                    "target_far": 0.1,
                                    "observed_far": 0.0,
                                    "tar": 0.5,
                                    "frr": 0.5,
                                    "threshold": 0.9,
                                    (
                                        "empirically_"
                                        "resolvable"
                                    ): False,
                                }
                            ],
                            "roc_curve": {
                                "false_accept_rates": [
                                    0.0,
                                    0.0,
                                    1.0,
                                ],
                                "true_accept_rates": [
                                    0.0,
                                    1.0,
                                    1.0,
                                ],
                                "false_reject_rates": [
                                    1.0,
                                    0.0,
                                    0.0,
                                ],
                                "thresholds": [
                                    None,
                                    0.9,
                                    0.1,
                                ],
                            },
                            "det_curve": {
                                "false_accept_rates": [
                                    0.0,
                                    1.0,
                                ],
                                "false_reject_rates": [
                                    1.0,
                                    0.0,
                                ],
                                "thresholds": [
                                    0.9,
                                    0.1,
                                ],
                            },
                        },
                    }
                )

            per_run_results = [
                {
                    "run_index": 1,
                    "seed": 42,
                    "metrics": {
                        "EER": 0.2,
                        "AUC": 0.8,
                    },
                },
                {
                    "run_index": 2,
                    "seed": 43,
                    "metrics": {
                        "EER": 0.3,
                        "AUC": 0.7,
                    },
                },
            ]

            single_artifact, per_run_artifacts = (
                run._save_compact_evaluation_outputs(
                    results_dir=results_dir,
                    safe_task_name=(
                        "Synthetic_Verification"
                    ),
                    experiment_time=datetime(
                        2026,
                        7,
                        27,
                        12,
                        0,
                        0,
                    ),
                    per_run_results=(
                        per_run_results
                    ),
                    per_run_evaluation_artifacts=(
                        artifact_items
                    ),
                )
            )

            self.assertIsNone(
                single_artifact
            )
            self.assertEqual(
                len(per_run_artifacts),
                2,
            )

            compact_artifact = (
                per_run_artifacts[0][
                    "artifact"
                ]
            )

            self.assertNotIn(
                "roc_curve",
                compact_artifact,
            )
            self.assertNotIn(
                "det_curve",
                compact_artifact,
            )
            self.assertEqual(
                compact_artifact[
                    "operating_points"
                ][0]["tar"],
                0.5,
            )

            curve_path = (
                results_dir
                / compact_artifact[
                    "curve_storage"
                ]["file"]
            )

            self.assertTrue(
                curve_path.exists()
            )

            with np.load(
                curve_path
            ) as curve_file:
                self.assertIn(
                    (
                        "seed_42_roc_"
                        "false_accept_rates"
                    ),
                    curve_file.files,
                )
                self.assertIn(
                    (
                        "seed_43_det_"
                        "false_reject_rates"
                    ),
                    curve_file.files,
                )

            output_directory = (
                curve_path.parent
            )

            for filename in [
                "per_seed_metrics.csv",
                (
                    "verification_"
                    "operating_points.csv"
                ),
                "roc.png",
                "det.png",
                "tar_at_far.png",
            ]:
                self.assertTrue(
                    (
                        output_directory
                        / filename
                    ).exists(),
                    filename,
                )

            json.dumps(
                per_run_artifacts,
                allow_nan=False,
            )


class CompactIdentificationOutputTests(
    unittest.TestCase
):
    def test_cmc_summary_remains_compact(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_dir = Path(
                temporary_directory
            )

            artifact = {
                "type": "identification",
                "probe_count": 3,
                "gallery_size": 3,
                "maximum_meaningful_rank": 3,
                "rank_1_accuracy": (
                    1.0 / 3.0
                ),
                "rank_5_accuracy": None,
                "effective_rank_5": None,
                "rank_5_defined": False,
                "rank_5_reportable": False,
                "rank_5_reportability_reason": (
                    "gallery_size_below_5"
                ),
                "correct_match_ranks": [
                    1,
                    2,
                    3,
                ],
                "cmc_curve": {
                    "ranks": [
                        1,
                        2,
                        3,
                    ],
                    (
                        "identification_rates"
                    ): [
                        1.0 / 3.0,
                        2.0 / 3.0,
                        1.0,
                    ],
                },
            }

            compact_artifact, per_run = (
                run._save_compact_evaluation_outputs(
                    results_dir=results_dir,
                    safe_task_name=(
                        "Synthetic_Identification"
                    ),
                    experiment_time=datetime(
                        2026,
                        7,
                        27,
                        12,
                        0,
                        0,
                    ),
                    evaluation_artifacts=(
                        artifact
                    ),
                )
            )

            self.assertEqual(
                per_run,
                [],
            )
            self.assertNotIn(
                "correct_match_ranks",
                compact_artifact,
            )
            self.assertEqual(
                compact_artifact[
                    "cmc_curve"
                ][
                    "identification_rates"
                ][-1],
                1.0,
            )

            curve_path = (
                results_dir
                / compact_artifact[
                    "curve_storage"
                ]["file"]
            )

            with np.load(
                curve_path
            ) as curve_file:
                np.testing.assert_array_equal(
                    curve_file[
                        "run_1_correct_match_ranks"
                    ],
                    [
                        1,
                        2,
                        3,
                    ],
                )

            self.assertTrue(
                (
                    curve_path.parent
                    / "identification_cmc.csv"
                ).exists()
            )
            self.assertTrue(
                (
                    curve_path.parent
                    / "cmc.png"
                ).exists()
            )


class NoArtifactTests(
    unittest.TestCase
):
    def test_missing_artifacts_do_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_dir = Path(
                temporary_directory
            )

            single_artifact, per_run = (
                run._save_compact_evaluation_outputs(
                    results_dir=results_dir,
                    safe_task_name="Synthetic",
                    experiment_time=datetime(
                        2026,
                        7,
                        27,
                        12,
                        0,
                        0,
                    ),
                )
            )

            self.assertIsNone(
                single_artifact
            )
            self.assertEqual(
                per_run,
                [],
            )
            self.assertEqual(
                list(
                    results_dir.iterdir()
                ),
                [],
            )


if __name__ == "__main__":
    unittest.main()
