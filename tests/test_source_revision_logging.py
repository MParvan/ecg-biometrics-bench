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
    def __init__(self):
        self.cfg = {
            "root_dir": "synthetic_dataset",
            "preprocessing": {},
        }
        self.prep_params = {}


class SourceRevisionLoggingTests(unittest.TestCase):
    def test_revision_metadata_contains_commit_branch_and_command(self):
        git_outputs = {
            ("rev-parse", "HEAD"): "abc123def456",
            (
                "rev-parse",
                "--abbrev-ref",
                "HEAD",
            ): "revision/test-branch",
            ("status", "--porcelain"): "",
        }

        def fake_git_command(*arguments):
            return git_outputs.get(arguments)

        with patch(
            "run._run_git_command",
            side_effect=fake_git_command,
        ), patch.object(
            sys,
            "argv",
            [
                "main.py",
                "--dataset",
                "ecgid",
                "--task",
                "1",
            ],
        ):
            metadata = run._collect_source_revision()

        self.assertEqual(
            metadata["Git Commit"],
            "abc123def456",
        )
        self.assertEqual(
            metadata["Git Branch"],
            "revision/test-branch",
        )
        self.assertFalse(
            metadata["Git Working Tree Dirty"]
        )
        self.assertIn(
            "--dataset ecgid",
            metadata["Python Invocation"],
        )

    def test_dirty_working_tree_is_reported(self):
        git_outputs = {
            ("rev-parse", "HEAD"): "abc123",
            (
                "rev-parse",
                "--abbrev-ref",
                "HEAD",
            ): "revision/test",
            ("status", "--porcelain"): " M run.py",
        }

        def fake_git_command(*arguments):
            return git_outputs.get(arguments)

        with patch(
            "run._run_git_command",
            side_effect=fake_git_command,
        ):
            metadata = run._collect_source_revision()

        self.assertTrue(
            metadata["Git Working Tree Dirty"]
        )

    def test_missing_git_information_is_handled(self):
        with patch(
            "run._run_git_command",
            return_value=None,
        ):
            metadata = run._collect_source_revision()

        self.assertEqual(
            metadata["Git Commit"],
            "unavailable",
        )
        self.assertEqual(
            metadata["Git Branch"],
            "unavailable",
        )
        self.assertEqual(
            metadata["Git Working Tree Dirty"],
            "unavailable",
        )

    def test_logger_writes_source_revision_section(self):
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
                    "run._collect_source_revision",
                    return_value={
                        "Git Commit": "abc123",
                        "Git Branch": "revision/test",
                        "Git Working Tree Dirty": False,
                        "Python Invocation": (
                            "main.py --dataset ecgid --task 1"
                        ),
                    },
                ), patch(
                    "run._collect_runtime_profile",
                    return_value={},
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
                    "[SOURCE REVISION]",
                    content,
                )
                self.assertIn("abc123", content)
                self.assertIn(
                    "revision/test",
                    content,
                )
                self.assertIn(
                    "main.py --dataset ecgid --task 1",
                    content,
                )

            finally:
                # Restore the directory before Windows removes
                # the temporary folder.
                os.chdir(original_directory)


if __name__ == "__main__":
    unittest.main()