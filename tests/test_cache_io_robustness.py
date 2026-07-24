import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import (
    CacheManager,
    _atomic_write_file,
)


class CacheIORobustnessTests(unittest.TestCase):
    def test_atomic_write_preserves_existing_file_on_failure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            final_path = (
                Path(temporary_directory)
                / "cache.json"
            )

            final_path.write_text(
                "original",
                encoding="utf-8",
            )

            def failing_writer(temporary_path):
                Path(temporary_path).write_text(
                    "incomplete replacement",
                    encoding="utf-8",
                )
                raise RuntimeError(
                    "simulated interruption"
                )

            with self.assertRaisesRegex(
                RuntimeError,
                "simulated interruption",
            ):
                _atomic_write_file(
                    final_path,
                    failing_writer,
                )

            self.assertEqual(
                final_path.read_text(
                    encoding="utf-8"
                ),
                "original",
            )

            temporary_files = [
                path
                for path in Path(
                    temporary_directory
                ).iterdir()
                if path.name.startswith(
                    ".cache.json."
                )
            ]

            self.assertEqual(
                temporary_files,
                [],
            )

    def test_data_cache_round_trip_closes_npz_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache = CacheManager(
                base_dir=temporary_directory
            )

            configuration = {
                "dataset": "synthetic",
                "split": "train",
            }

            arrays = {
                "x": np.arange(12).reshape(4, 3),
                "y": np.array([0, 0, 1, 1]),
            }

            _, uid = cache.get_data_cache(
                configuration
            )

            cache.save_data_cache(
                arrays,
                configuration,
                uid,
            )

            loaded, loaded_uid = (
                cache.get_data_cache(
                    configuration
                )
            )

            self.assertEqual(
                loaded_uid,
                uid,
            )

            np.testing.assert_array_equal(
                loaded["x"],
                arrays["x"],
            )
            np.testing.assert_array_equal(
                loaded["y"],
                arrays["y"],
            )

            # This deletion fails on Windows if np.load() left the
            # underlying archive open.
            data_path = (
                Path(temporary_directory)
                / "data"
                / f"{uid}.npz"
            )

            data_path.unlink()

            self.assertFalse(
                data_path.exists()
            )

    def test_corrupted_data_cache_is_removed_and_recomputed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache = CacheManager(
                base_dir=temporary_directory
            )

            configuration = {
                "dataset": "synthetic",
                "split": "test",
            }

            _, uid = cache.get_data_cache(
                configuration
            )

            data_path = (
                Path(temporary_directory)
                / "data"
                / f"{uid}.npz"
            )
            metadata_path = (
                Path(temporary_directory)
                / "data"
                / f"{uid}.json"
            )

            data_path.write_bytes(
                b"not a valid npz archive"
            )
            metadata_path.write_text(
                json.dumps(configuration),
                encoding="utf-8",
            )

            loaded, loaded_uid = (
                cache.get_data_cache(
                    configuration
                )
            )

            self.assertIsNone(loaded)
            self.assertEqual(
                loaded_uid,
                uid,
            )
            self.assertFalse(
                data_path.exists()
            )
            self.assertFalse(
                metadata_path.exists()
            )

    def test_corrupted_weight_cache_does_not_modify_model(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache = CacheManager(
                base_dir=temporary_directory
            )

            configuration = {
                "training_regime": "synthetic",
                "model": "Linear",
                "epochs": 10,
            }

            model = nn.Linear(
                3,
                2,
            )

            original_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

            _, uid = cache.get_weight_cache(
                configuration,
                model,
                device="cpu",
            )

            weight_path = (
                Path(temporary_directory)
                / "weights"
                / f"{uid}.pth"
            )
            metadata_path = (
                Path(temporary_directory)
                / "weights"
                / f"{uid}.json"
            )

            weight_path.write_bytes(
                b"not a valid PyTorch checkpoint"
            )
            metadata_path.write_text(
                json.dumps(configuration),
                encoding="utf-8",
            )

            loaded_model, loaded_uid = (
                cache.get_weight_cache(
                    configuration,
                    model,
                    device="cpu",
                )
            )

            self.assertIsNone(
                loaded_model
            )
            self.assertEqual(
                loaded_uid,
                uid,
            )

            for key, value in model.state_dict().items():
                torch.testing.assert_close(
                    value,
                    original_state[key],
                )

            self.assertFalse(
                weight_path.exists()
            )
            self.assertFalse(
                metadata_path.exists()
            )


if __name__ == "__main__":
    unittest.main()