import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run


class DummyLoader:
    def __init__(self):
        self.cfg = {
            "root_dir": "synthetic_dataset",
            "preprocessing": {},
        }
        self.prep_params = {}


class ComputationalProfileTests(unittest.TestCase):
    def test_model_parameter_counts_are_correct(self):
        model = nn.Sequential(
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
        )

        summary = run._summarize_model_complexity(
            model
        )

        # Linear(4, 3): 12 weights + 3 biases
        # Linear(3, 2):  6 weights + 2 biases
        expected_parameters = 23

        self.assertEqual(
            summary["Total Model Parameters"],
            expected_parameters,
        )
        self.assertEqual(
            summary["Trainable Model Parameters"],
            expected_parameters,
        )
        self.assertGreater(
            summary["Model State Size (MiB)"],
            0.0,
        )

    def test_frozen_parameters_are_not_trainable(self):
        model = nn.Linear(4, 2)

        model.weight.requires_grad = False

        summary = run._summarize_model_complexity(
            model
        )

        self.assertEqual(
            summary["Total Model Parameters"],
            10,
        )
        self.assertEqual(
            summary["Trainable Model Parameters"],
            2,
        )

    def test_runtime_profile_uses_started_timer(self):
        with patch(
            "run.time.perf_counter",
            side_effect=[100.0, 112.5],
        ), patch(
            "run.torch.cuda.is_available",
            return_value=False,
        ):
            run.start_experiment_timer()
            profile = run._collect_runtime_profile()

        self.assertAlmostEqual(
            profile["Total Wall-Clock Time (seconds)"],
            12.5,
        )

    def test_logger_writes_computational_profile(self):
        original_directory = os.getcwd()

        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)

                with patch(
                    "run._collect_software_environment",
                    return_value={
                        "Python": "test",
                    },
                ), patch(
                    "run._collect_runtime_profile",
                    return_value={
                        "Total Wall-Clock Time (seconds)": 12.5,
                        "Peak CUDA Memory (MiB)": 256.0,
                    },
                ):
                    run._log_experiment_results(
                        task_name="Synthetic Task",
                        metrics_dict={
                            "Accuracy": 0.95,
                        },
                        data_stats={
                            "Samples": 10,
                        },
                        hyperparams={
                            "Total Model Parameters": 100,
                            "Trainable Model Parameters": 100,
                            "Model State Size (MiB)": 0.5,
                        },
                        loader=DummyLoader(),
                    )

                log_path = (
                    Path("results")
                    / "synthetic_dataset"
                    / "Synthetic_Task.txt"
                )

                content = log_path.read_text(
                    encoding="utf-8"
                )

                self.assertIn(
                    "[COMPUTATIONAL PROFILE]",
                    content,
                )
                self.assertIn(
                    "Total Wall-Clock Time",
                    content,
                )
                self.assertIn(
                    "12.5000",
                    content,
                )
                self.assertIn(
                    "Peak CUDA Memory",
                    content,
                )

            finally:
                os.chdir(original_directory)


if __name__ == "__main__":
    unittest.main()