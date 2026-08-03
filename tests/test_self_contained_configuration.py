import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

import main


class SelfContainedConfigurationTests(unittest.TestCase):
    def _write_config(self, directory, content):
        path = Path(directory) / "experiment.yaml"
        path.write_text(
            yaml.safe_dump(
                content,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return path

    def test_explicit_cli_ranges_replace_yaml_append_defaults(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = self._write_config(
                temporary_directory,
                {
                    "dataset": "mitbih",
                    "task": 6,
                    "data_split_mode": "custom-split",
                    "train_parts": [[0, 5], [10, 15]],
                    "test_parts": [[25, 30]],
                },
            )

            args, _ = main.parse_experiment_arguments(
                [
                    "--config",
                    str(config_path),
                    "--train_parts",
                    "30",
                    "35",
                    "--test_parts",
                    "40",
                    "45",
                ]
            )

        self.assertEqual(
            args.train_parts,
            [(30.0, 35.0)],
        )
        self.assertEqual(
            args.test_parts,
            [(40.0, 45.0)],
        )

    def test_unknown_yaml_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = self._write_config(
                temporary_directory,
                {
                    "dataset": "ecgid",
                    "task": 1,
                    "unknown_setting": 123,
                },
            )

            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                main.parse_experiment_arguments(
                    [
                        "--config",
                        str(config_path),
                    ]
                )

        self.assertIn(
            "Unknown configuration key(s): unknown_setting",
            stderr.getvalue(),
        )

    def test_dataset_and_task_are_required_after_merging(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = self._write_config(
                temporary_directory,
                {
                    "epochs": 2,
                },
            )

            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                main.parse_experiment_arguments(
                    [
                        "--config",
                        str(config_path),
                    ]
                )

        self.assertIn(
            "final experiment configuration must define: dataset, task",
            stderr.getvalue(),
        )

    def test_yaml_boolean_values_must_be_boolean(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = self._write_config(
                temporary_directory,
                {
                    "dataset": "ecgid",
                    "task": 1,
                    "visualize": "false",
                },
            )

            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                main.parse_experiment_arguments(
                    [
                        "--config",
                        str(config_path),
                    ]
                )

        self.assertIn(
            "'visualize' must be Boolean",
            stderr.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
