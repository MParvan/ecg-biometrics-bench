import csv
import json
import math
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from scripts import statistical_comparisons as sc


def make_record(
    values_by_seed,
    metric_name="EER",
    dataset="synthetic",
    task="Synthetic Verification",
):
    return {
        "experiment_time": (
            "2026-07-27T12:00:00"
        ),
        "dataset": dataset,
        "task": task,
        "per_run_results": [
            {
                "run_index": run_index,
                "seed": seed,
                "metrics": {
                    metric_name: value,
                },
            }
            for run_index, (
                seed,
                value,
            ) in enumerate(
                values_by_seed,
                start=1,
            )
        ],
    }


def write_jsonl(
    path,
    records,
):
    Path(path).write_text(
        "".join(
            json.dumps(record)
            + "\n"
            for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )


class StatisticalComparisonTests(
    unittest.TestCase
):
    def test_jsonl_reader_supports_negative_record_index(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = (
                Path(temporary_directory)
                / "results.jsonl"
            )

            write_jsonl(
                path,
                [
                    {
                        "task": "first",
                    },
                    {
                        "task": "latest",
                    },
                ],
            )

            record = sc.read_jsonl_record(
                path,
                record_index=-1,
            )

            self.assertEqual(
                record["task"],
                "latest",
            )

    def test_duplicate_seeds_are_rejected(self):
        record = {
            "per_run_results": [
                {
                    "seed": 42,
                    "metrics": {
                        "EER": 0.1,
                    },
                },
                {
                    "seed": 42,
                    "metrics": {
                        "EER": 0.2,
                    },
                },
            ],
        }

        with self.assertRaisesRegex(
            ValueError,
            "Duplicate seed 42",
        ):
            sc.extract_seed_metrics(
                record
            )

    def test_mismatched_seed_sets_are_rejected(self):
        reference = sc.extract_seed_metrics(
            make_record(
                [
                    (
                        42,
                        0.10,
                    ),
                    (
                        43,
                        0.12,
                    ),
                ]
            )
        )
        comparison = sc.extract_seed_metrics(
            make_record(
                [
                    (
                        42,
                        0.09,
                    ),
                    (
                        44,
                        0.11,
                    ),
                ]
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            "identical seed sets",
        ):
            sc.align_metric_values(
                reference,
                comparison,
                "EER",
            )

    def test_alignment_uses_seed_not_input_order(self):
        reference = sc.extract_seed_metrics(
            make_record(
                [
                    (
                        43,
                        0.20,
                    ),
                    (
                        42,
                        0.10,
                    ),
                ]
            )
        )
        comparison = sc.extract_seed_metrics(
            make_record(
                [
                    (
                        42,
                        0.15,
                    ),
                    (
                        43,
                        0.25,
                    ),
                ]
            )
        )

        (
            seeds,
            reference_values,
            comparison_values,
        ) = sc.align_metric_values(
            reference,
            comparison,
            "EER",
        )

        self.assertEqual(
            seeds,
            [
                42,
                43,
            ],
        )
        np.testing.assert_allclose(
            reference_values,
            [
                0.10,
                0.20,
            ],
        )
        np.testing.assert_allclose(
            comparison_values,
            [
                0.15,
                0.25,
            ],
        )

    def test_paired_statistics_match_scipy(self):
        reference = np.asarray(
            [
                0.20,
                0.25,
                0.18,
                0.23,
                0.21,
            ]
        )
        comparison = np.asarray(
            [
                0.18,
                0.22,
                0.17,
                0.20,
                0.19,
            ]
        )

        result = (
            sc.calculate_paired_statistics(
                reference,
                comparison,
                confidence_level=0.95,
            )
        )
        expected_t = stats.ttest_rel(
            comparison,
            reference,
        )

        self.assertEqual(
            result["status"],
            "ok",
        )
        self.assertAlmostEqual(
            result[
                "mean_difference"
            ],
            float(
                np.mean(
                    comparison
                    - reference
                )
            ),
        )
        self.assertAlmostEqual(
            result[
                "paired_t_statistic"
            ],
            float(
                expected_t.statistic
            ),
        )
        self.assertAlmostEqual(
            result[
                "paired_t_p_value"
            ],
            float(
                expected_t.pvalue
            ),
        )
        self.assertLess(
            result[
                "mean_difference_ci_lower"
            ],
            result[
                "mean_difference_ci_upper"
            ],
        )

    def test_constant_nonzero_differences_avoid_precision_warning(self):
        reference = np.asarray(
            [
                0.20,
                0.22,
                0.19,
            ]
        )
        comparison = np.asarray(
            [
                0.18,
                0.20,
                0.17,
            ]
        )

        with warnings.catch_warnings(
            record=True
        ) as captured_warnings:
            warnings.simplefilter(
                "always"
            )

            result = (
                sc.calculate_paired_statistics(
                    reference,
                    comparison,
                )
            )

        runtime_warnings = [
            warning
            for warning in captured_warnings
            if issubclass(
                warning.category,
                RuntimeWarning,
            )
        ]

        self.assertEqual(
            runtime_warnings,
            [],
        )
        self.assertEqual(
            result[
                (
                    "difference_standard_"
                    "deviation_sample"
                )
            ],
            0.0,
        )
        self.assertTrue(
            math.isinf(
                result[
                    "paired_t_statistic"
                ]
            )
        )
        self.assertLess(
            result[
                "paired_t_statistic"
            ],
            0.0,
        )
        self.assertEqual(
            result[
                "paired_t_p_value"
            ],
            0.0,
        )
        self.assertTrue(
            math.isinf(
                result["cohens_dz"]
            )
        )
        self.assertLess(
            result["cohens_dz"],
            0.0,
        )
        self.assertAlmostEqual(
            result[
                "mean_difference_ci_lower"
            ],
            -0.02,
        )
        self.assertAlmostEqual(
            result[
                "mean_difference_ci_upper"
            ],
            -0.02,
        )

    def test_all_zero_differences_are_handled(self):
        values = np.asarray(
            [
                0.1,
                0.2,
                0.3,
            ]
        )

        result = (
            sc.calculate_paired_statistics(
                values,
                values,
            )
        )

        self.assertEqual(
            result[
                "paired_t_p_value"
            ],
            1.0,
        )
        self.assertEqual(
            result[
                "wilcoxon_p_value"
            ],
            1.0,
        )
        self.assertEqual(
            result["cohens_dz"],
            0.0,
        )
        self.assertEqual(
            result[
                "mean_difference_ci_lower"
            ],
            0.0,
        )
        self.assertEqual(
            result[
                "mean_difference_ci_upper"
            ],
            0.0,
        )

    def test_single_pair_reports_insufficient_runs(self):
        result = (
            sc.calculate_paired_statistics(
                [
                    0.2,
                ],
                [
                    0.1,
                ],
            )
        )

        self.assertEqual(
            result["status"],
            "insufficient_runs",
        )
        self.assertEqual(
            result["n_pairs"],
            1,
        )
        self.assertIsNone(
            result[
                "paired_t_p_value"
            ]
        )
        self.assertIsNone(
            result[
                "wilcoxon_p_value"
            ]
        )
        self.assertIsNone(
            result["cohens_dz"]
        )

    def test_holm_adjustment_is_monotonic(self):
        adjusted = sc.holm_adjust(
            [
                0.01,
                0.04,
                0.03,
                None,
            ]
        )

        self.assertAlmostEqual(
            adjusted[0],
            0.03,
        )
        self.assertAlmostEqual(
            adjusted[1],
            0.06,
        )
        self.assertAlmostEqual(
            adjusted[2],
            0.06,
        )
        self.assertIsNone(
            adjusted[3]
        )

    def test_manifest_analysis_writes_json_and_csv(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(
                temporary_directory
            )

            reference_path = (
                directory
                / "reference.jsonl"
            )
            comparison_path = (
                directory
                / "comparison.jsonl"
            )
            manifest_path = (
                directory
                / "comparison.yaml"
            )
            json_output = (
                directory
                / "statistics.json"
            )
            csv_output = (
                directory
                / "statistics.csv"
            )

            reference_record = {
                "experiment_time": (
                    "2026-07-27T12:00:00"
                ),
                "dataset": "synthetic",
                "task": "Verification",
                "per_run_results": [
                    {
                        "run_index": index,
                        "seed": seed,
                        "metrics": {
                            "EER": eer,
                            "AUC": auc,
                        },
                    }
                    for index, (
                        seed,
                        eer,
                        auc,
                    ) in enumerate(
                        [
                            (
                                42,
                                0.20,
                                0.80,
                            ),
                            (
                                43,
                                0.22,
                                0.82,
                            ),
                            (
                                44,
                                0.19,
                                0.81,
                            ),
                        ],
                        start=1,
                    )
                ],
            }

            comparison_record = {
                "experiment_time": (
                    "2026-07-27T13:00:00"
                ),
                "dataset": "synthetic",
                "task": "Verification",
                "per_run_results": [
                    {
                        "run_index": index,
                        "seed": seed,
                        "metrics": {
                            "EER": eer,
                            "AUC": auc,
                        },
                    }
                    for index, (
                        seed,
                        eer,
                        auc,
                    ) in enumerate(
                        [
                            (
                                42,
                                0.18,
                                0.84,
                            ),
                            (
                                43,
                                0.20,
                                0.85,
                            ),
                            (
                                44,
                                0.17,
                                0.86,
                            ),
                        ],
                        start=1,
                    )
                ],
            }

            write_jsonl(
                reference_path,
                [
                    reference_record,
                ],
            )
            write_jsonl(
                comparison_path,
                [
                    comparison_record,
                ],
            )

            manifest_path.write_text(
                """
reference:
  path: reference.jsonl
  record_index: -1
  label: baseline

comparisons:
  - path: comparison.jsonl
    record_index: -1
    label: augmented

metrics:
  - EER
  - AUC

confidence_level: 0.95
""".lstrip(),
                encoding="utf-8",
                newline="\n",
            )

            exit_code = sc.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--output-json",
                    str(json_output),
                    "--output-csv",
                    str(csv_output),
                ]
            )

            self.assertEqual(
                exit_code,
                0,
            )
            self.assertTrue(
                json_output.is_file()
            )
            self.assertTrue(
                csv_output.is_file()
            )

            analysis = json.loads(
                json_output.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                analysis[
                    "analysis_direction"
                ],
                "comparison - reference",
            )
            self.assertEqual(
                len(
                    analysis[
                        "comparisons"
                    ][0]["metrics"]
                ),
                2,
            )

            eer_result = analysis[
                "comparisons"
            ][0]["metrics"][0]

            self.assertEqual(
                eer_result["metric"],
                "EER",
            )
            self.assertEqual(
                eer_result[
                    "paired_seeds"
                ],
                [
                    42,
                    43,
                    44,
                ],
            )
            self.assertLess(
                eer_result[
                    "mean_difference"
                ],
                0.0,
            )
            self.assertIn(
                "paired_t_p_value_holm",
                eer_result,
            )
            self.assertIn(
                "wilcoxon_p_value_holm",
                eer_result,
            )

            serialized = json.dumps(
                analysis,
                allow_nan=False,
            )

            self.assertIsInstance(
                serialized,
                str,
            )

            with csv_output.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as csv_file:
                rows = list(
                    csv.DictReader(
                        csv_file
                    )
                )

            self.assertEqual(
                len(rows),
                2,
            )
            self.assertEqual(
                rows[0][
                    "comparison_label"
                ],
                "augmented",
            )
            self.assertEqual(
                rows[0]["metric"],
                "EER",
            )


if __name__ == "__main__":
    unittest.main()
