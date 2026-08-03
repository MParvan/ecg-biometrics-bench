import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

import run


class DummyLoader:
    def __init__(self, results_dir):
        self.cfg = {
            "root_dir": "synthetic_dataset",
            "preprocessing": {
                "mode": "beat",
                "filter_method": "butter",
            },
        }
        self.prep_params = {
            "mode": "window",
            "window_s": 5.0,
        }
        self.results_dir = Path(
            results_dir
        )
        self.signal_type = "filtered"
        self.target_leads = (
            "I",
            "II",
        )


class UnsupportedMetadata:
    def __str__(self):
        return "synthetic-object"


class StructuredOutputLoggingTests(
    unittest.TestCase
):
    def _log_experiment(
        self,
        results_dir,
        accuracy=0.95,
    ):
        with patch(
            "run._collect_software_environment",
            return_value={
                "Python": "3.test",
                "CUDA Available": False,
            },
        ), patch(
            "run._collect_source_revision",
            return_value={
                "Git Commit": "abc123",
                "Git Branch": "revision/test",
                "Git Working Tree Dirty": False,
                "Python Invocation": (
                    "main.py --dataset ecgid "
                    "--task 1"
                ),
            },
        ), patch(
            "run._collect_runtime_profile",
            return_value={
                (
                    "Total Wall-Clock Time "
                    "(seconds)"
                ): np.float64(12.5),
            },
        ):
            run._log_experiment_results(
                task_name="Synthetic Task",
                metrics_dict={
                    "Accuracy": np.float32(
                        accuracy
                    ),
                    "Aggregate Metric": (
                        "0.9000 ± 0.0200"
                    ),
                },
                data_stats={
                    "Samples": np.int64(10),
                },
                hyperparams={
                    "epochs": 1,
                    "run_seeds": np.asarray(
                        [
                            42,
                            43,
                        ],
                        dtype=np.int64,
                    ),
                    "device": torch.device(
                        "cpu"
                    ),
                    "output_path": Path(
                        "artifacts/model.pt"
                    ),
                },
                loader=DummyLoader(
                    results_dir
                ),
            )

    def test_logger_writes_text_and_jsonl_companions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            self._log_experiment(
                temporary_directory
            )

            result_directory = (
                Path(temporary_directory)
                / "synthetic_dataset"
            )

            text_path = (
                result_directory
                / "Synthetic_Task.txt"
            )
            jsonl_path = (
                result_directory
                / "Synthetic_Task.jsonl"
            )

            self.assertTrue(
                text_path.is_file()
            )
            self.assertTrue(
                jsonl_path.is_file()
            )

            text_content = text_path.read_text(
                encoding="utf-8"
            )

            self.assertIn(
                "[DATA STATISTICS]",
                text_content,
            )
            self.assertIn(
                "[MODEL HYPERPARAMETERS]",
                text_content,
            )
            self.assertIn(
                "[RESULTS]",
                text_content,
            )

    def test_structured_record_contains_all_sections(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            self._log_experiment(
                temporary_directory
            )

            jsonl_path = (
                Path(temporary_directory)
                / "synthetic_dataset"
                / "Synthetic_Task.jsonl"
            )

            lines = [
                line
                for line in jsonl_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

            self.assertEqual(
                len(lines),
                1,
            )

            record = json.loads(
                lines[0]
            )

            self.assertEqual(
                record["task"],
                "Synthetic Task",
            )
            self.assertEqual(
                record["dataset"],
                "synthetic_dataset",
            )
            self.assertEqual(
                record[
                    "data_statistics"
                ]["Samples"],
                10,
            )

            configuration = record[
                (
                    "effective_experiment_"
                    "configuration"
                )
            ]

            self.assertEqual(
                configuration[
                    "model_hyperparameters"
                ]["run_seeds"],
                [
                    42,
                    43,
                ],
            )
            self.assertEqual(
                configuration[
                    "model_hyperparameters"
                ]["device"],
                "cpu",
            )
            self.assertEqual(
                configuration[
                    (
                        "dataset_and_preprocessing_"
                        "settings"
                    )
                ]["mode"],
                "window",
            )
            self.assertEqual(
                configuration[
                    (
                        "dataset_and_preprocessing_"
                        "settings"
                    )
                ]["target_leads"],
                [
                    "I",
                    "II",
                ],
            )

            self.assertAlmostEqual(
                record[
                    "computational_profile"
                ][
                    (
                        "Total Wall-Clock Time "
                        "(seconds)"
                    )
                ],
                12.5,
            )
            self.assertAlmostEqual(
                record[
                    "results"
                ]["Accuracy"],
                0.95,
                places=6,
            )
            self.assertEqual(
                record[
                    "results"
                ][
                    "Aggregate Metric"
                ]["mean"],
                0.9,
            )
            self.assertEqual(
                record[
                    "results"
                ][
                    "Aggregate Metric"
                ]["std"],
                0.02,
            )

            serialized_record = json.dumps(
                record
            )

            self.assertNotIn(
                "schema_version",
                serialized_record,
            )
            self.assertNotIn(
                "cache_version",
                serialized_record,
            )

    def test_jsonl_appends_one_record_per_experiment(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            self._log_experiment(
                temporary_directory,
                accuracy=0.91,
            )
            self._log_experiment(
                temporary_directory,
                accuracy=0.93,
            )

            jsonl_path = (
                Path(temporary_directory)
                / "synthetic_dataset"
                / "Synthetic_Task.jsonl"
            )

            records = [
                json.loads(line)
                for line in jsonl_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

            self.assertEqual(
                len(records),
                2,
            )
            self.assertAlmostEqual(
                records[0][
                    "results"
                ]["Accuracy"],
                0.91,
                places=6,
            )
            self.assertAlmostEqual(
                records[1][
                    "results"
                ]["Accuracy"],
                0.93,
                places=6,
            )

    def test_serializer_handles_nonfinite_and_unknown_values(self):
        converted = (
            run._to_json_compatible(
                {
                    "nan": float("nan"),
                    "positive_infinity": (
                        np.float32(
                            np.inf
                        )
                    ),
                    "unknown": (
                        UnsupportedMetadata()
                    ),
                }
            )
        )

        self.assertEqual(
            converted["nan"],
            "nan",
        )
        self.assertEqual(
            converted[
                "positive_infinity"
            ],
            "inf",
        )
        self.assertEqual(
            converted["unknown"],
            "synthetic-object",
        )

        serialized = json.dumps(
            converted,
            allow_nan=False,
        )

        self.assertIsInstance(
            serialized,
            str,
        )


if __name__ == "__main__":
    unittest.main()
