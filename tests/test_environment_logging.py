import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run


class DummyLoader:
    def __init__(self, results_dir=None):
        self.cfg = {
            "root_dir": "synthetic_dataset",
            "preprocessing": {
                "filter_method": "butter",
            },
        }

        self.prep_params = {}
        self.results_dir = results_dir


class EnvironmentLoggingTests(unittest.TestCase):
    def test_environment_contains_required_fields(self):
        environment = run._collect_software_environment()

        required_fields = {
            "Python",
            "Operating System",
            "PyTorch",
            "CUDA Available",
            "CUDA Runtime",
            "NumPy",
            "SciPy",
            "scikit-learn",
            "pandas",
            "NeuroKit2",
            "WFDB",
            "PyYAML",
        }

        self.assertTrue(
            required_fields.issubset(environment.keys())
        )

    def test_missing_optional_package_is_reported(self):
        with patch(
            "run.metadata.version",
            side_effect=run.metadata.PackageNotFoundError,
        ):
            version = run._get_installed_package_version(
                "missing-package"
            )

        self.assertEqual(version, "not installed")

    def test_experiment_log_contains_environment_section(self):
        synthetic_environment = {
            "Python": "3.test",
            "Operating System": "Synthetic OS",
            "PyTorch": "2.test",
            "CUDA Available": False,
        }

        original_directory = os.getcwd()

        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)

                with patch(
                    "run._collect_software_environment",
                    return_value=synthetic_environment,
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
                            "epochs": 1,
                        },
                        loader=DummyLoader(
                            results_dir=temporary_directory,
                        ),
                    )

                log_path = (
                    Path(temporary_directory)
                    / "synthetic_dataset"
                    / "Synthetic_Task.txt"
                )

                self.assertTrue(log_path.exists())

                log_content = log_path.read_text(
                    encoding="utf-8"
                )

                self.assertIn(
                    "[SOFTWARE & HARDWARE ENVIRONMENT]",
                    log_content,
                )
                self.assertIn("Python", log_content)
                self.assertIn("3.test", log_content)
                self.assertIn("Synthetic OS", log_content)
                self.assertIn("[RESULTS]", log_content)

            finally:
                # Restore the working directory before Windows attempts
                # to delete the temporary directory.
                os.chdir(original_directory)


if __name__ == "__main__":
    unittest.main()