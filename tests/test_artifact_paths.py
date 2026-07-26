import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run
import utils
from main import parse_experiment_arguments
from utils import (
    CacheManager,
    DEFAULT_CACHE_DIR,
    DEFAULT_RESULTS_DIR,
    PROJECT_ROOT,
    resolve_artifact_path,
)


class DummyLoader:
    def __init__(
        self,
        cache_dir,
        results_dir,
    ):
        self.cfg = {
            "root_dir": "synthetic",
            "preprocessing": {},
        }
        self.prep_params = {}
        self.cache_dir = cache_dir
        self.results_dir = results_dir


class ArtifactPathTests(unittest.TestCase):
    def test_portable_defaults_are_exposed_by_cli(self):
        args, _ = parse_experiment_arguments(
            [
                "--dataset",
                "ecgid",
                "--task",
                "1",
            ]
        )

        self.assertEqual(
            args.cache_dir,
            DEFAULT_CACHE_DIR,
        )
        self.assertEqual(
            args.results_dir,
            DEFAULT_RESULTS_DIR,
        )

    def test_relative_paths_are_resolved_from_repository(self):
        resolved = Path(
            resolve_artifact_path(
                "../ecg-biometrics-artifacts/cache"
            )
        )

        expected = (
            Path(PROJECT_ROOT)
            / ".."
            / "ecg-biometrics-artifacts"
            / "cache"
        ).resolve()

        self.assertEqual(
            resolved,
            expected,
        )

    def test_default_cache_location_is_external(self):
        default_value = inspect.signature(
            CacheManager
        ).parameters[
            "base_dir"
        ].default

        self.assertEqual(
            default_value,
            DEFAULT_CACHE_DIR,
        )

        resolved = Path(
            resolve_artifact_path(
                default_value
            )
        )

        repository_root = Path(
            PROJECT_ROOT
        ).resolve()

        self.assertNotIn(
            repository_root,
            resolved.parents,
        )

    def test_explicit_cache_directory_is_respected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manager = CacheManager(
                base_dir=temporary_directory
            )

            self.assertEqual(
                Path(manager.base_dir),
                Path(temporary_directory).resolve(),
            )
            self.assertEqual(
                Path(manager.data_dir),
                Path(temporary_directory).resolve()
                / "data",
            )
            self.assertEqual(
                Path(manager.weight_dir),
                Path(temporary_directory).resolve()
                / "weights",
            )

    def test_logger_uses_loader_results_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            loader = DummyLoader(
                cache_dir=temporary_directory,
                results_dir=temporary_directory,
            )

            with patch.object(
                run,
                "_collect_software_environment",
                return_value={},
            ), patch.object(
                run,
                "_collect_source_revision",
                return_value={},
            ), patch.object(
                run,
                "_collect_runtime_profile",
                return_value={},
            ):
                run._log_experiment_results(
                    "Synthetic Task",
                    {"Metric": 1.0},
                    {"Samples": 2},
                    {"epochs": 1},
                    loader,
                )

            expected_log = (
                Path(temporary_directory)
                / "synthetic"
                / "Synthetic_Task.txt"
            )

            self.assertTrue(
                expected_log.is_file()
            )


if __name__ == "__main__":
    unittest.main()
