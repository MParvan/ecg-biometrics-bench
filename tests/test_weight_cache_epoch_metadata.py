import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run
from utils import CacheManager


class WeightCacheEpochMetadataTests(
    unittest.TestCase
):
    def test_cache_restores_actual_training_epochs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache = CacheManager(
                base_dir=temporary_directory
            )

            config = {
                "training_regime": "test",
                "model": "Linear",
                "epochs": 100,
                "seed": 42,
            }

            original_model = nn.Linear(
                4,
                2,
            )
            original_model.actual_epochs = 37

            _, uid = cache.get_weight_cache(
                config,
                nn.Linear(4, 2),
                device="cpu",
            )

            cache.save_weight_cache(
                original_model,
                config,
                uid,
            )

            restored_model = nn.Linear(
                4,
                2,
            )

            restored_model, restored_uid = (
                cache.get_weight_cache(
                    config,
                    restored_model,
                    device="cpu",
                )
            )

            self.assertEqual(
                restored_uid,
                uid,
            )
            self.assertEqual(
                restored_model.actual_epochs,
                37,
            )

            for original_parameter, restored_parameter in zip(
                original_model.parameters(),
                restored_model.parameters(),
            ):
                torch.testing.assert_close(
                    original_parameter,
                    restored_parameter,
                )

    def test_saved_metadata_does_not_modify_config(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache = CacheManager(
                base_dir=temporary_directory
            )

            config = {
                "training_regime": "test",
                "epochs": 50,
                "seed": 7,
            }

            original_config = dict(
                config
            )

            model = nn.Linear(
                3,
                2,
            )
            model.actual_epochs = 12

            _, uid = cache.get_weight_cache(
                config,
                nn.Linear(3, 2),
                device="cpu",
            )

            cache.save_weight_cache(
                model,
                config,
                uid,
            )

            self.assertEqual(
                config,
                original_config,
            )

            metadata_path = (
                Path(temporary_directory)
                / "weights"
                / f"{uid}.json"
            )

            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                metadata["actual_epochs"],
                12,
            )
            self.assertEqual(
                metadata["epochs"],
                50,
            )

    def test_legacy_metadata_uses_configured_epochs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache = CacheManager(
                base_dir=temporary_directory
            )

            config = {
                "training_regime": "legacy-test",
                "epochs": 25,
                "seed": 42,
            }

            model = nn.Linear(
                3,
                2,
            )
            model.actual_epochs = 9

            _, uid = cache.get_weight_cache(
                config,
                nn.Linear(3, 2),
                device="cpu",
            )

            cache.save_weight_cache(
                model,
                config,
                uid,
            )

            metadata_path = (
                Path(temporary_directory)
                / "weights"
                / f"{uid}.json"
            )

            # Simulate the old JSON format, which contained
            # configuration only.
            metadata_path.write_text(
                json.dumps(
                    config,
                    indent=4,
                ),
                encoding="utf-8",
            )

            restored_model, _ = (
                cache.get_weight_cache(
                    config,
                    nn.Linear(3, 2),
                    device="cpu",
                )
            )

            self.assertEqual(
                restored_model.actual_epochs,
                25,
            )

    def test_runners_do_not_overwrite_cached_epoch_metadata(self):
        run_source = Path(
            run.__file__
        ).read_text(
            encoding="utf-8"
        )

        self.assertNotIn(
            "model.actual_epochs = epochs",
            run_source,
        )


if __name__ == "__main__":
    unittest.main()