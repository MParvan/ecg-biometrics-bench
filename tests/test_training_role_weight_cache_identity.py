import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run


class TrainingRoleWeightCacheIdentityTests(
    unittest.TestCase
):
    def setUp(self):
        self.training_samples = np.arange(
            24,
            dtype=np.float32,
        ).reshape(6, 4)

        self.training_labels = np.asarray(
            [0, 0, 0, 1, 1, 1]
        )

        self.training_config = {
            "model": "example",
            "epochs": 1,
        }

        self.loader_identity = {
            "loader_class": "ExampleLoader",
            "root_dir": "example",
            "preprocessing": {
                "mode": "beat",
            },
            "settings": {
                "data_split_mode": "custom-record-split",
                "train_sessions": ["train_a"],
                "enroll_sessions": ["enroll_a"],
                "probe_sessions": ["probe_a"],
                "required_cross_sessions": [
                    "train_a",
                    "enroll_a",
                    "probe_a",
                ],
                "train_parts": [(0, 5)],
                "enrol_parts": [(10, 15)],
                "test_parts": [(20, 25)],
                "train_record_indices": (0,),
                "enroll_record_indices": (1,),
                "probe_record_indices": (2,),
                "signal_type": "raw",
            },
        }

    def build(
        self,
        training_role_only,
    ):
        with patch.object(
            run,
            "_build_loader_cache_identity",
            return_value=copy.deepcopy(
                self.loader_identity
            ),
        ):
            return run._build_weight_cache_config(
                loader=object(),
                training_config=self.training_config,
                training_samples=self.training_samples,
                training_labels=self.training_labels,
                training_role_only_loader_identity=(
                    training_role_only
                ),
            )

    def test_default_identity_remains_complete(
        self,
    ):
        config = self.build(
            training_role_only=False
        )

        self.assertEqual(
            config["loader_identity"],
            self.loader_identity,
        )

    def test_training_role_identity_keeps_training_selectors(
        self,
    ):
        config = self.build(
            training_role_only=True
        )

        settings = config[
            "loader_identity"
        ]["settings"]

        self.assertEqual(
            settings["train_sessions"],
            ["train_a"],
        )
        self.assertEqual(
            settings["train_parts"],
            [(0, 5)],
        )
        self.assertEqual(
            settings["train_record_indices"],
            (0,),
        )
        self.assertEqual(
            settings["signal_type"],
            "raw",
        )

    def test_training_role_identity_removes_evaluation_selectors(
        self,
    ):
        config = self.build(
            training_role_only=True
        )

        settings = config[
            "loader_identity"
        ]["settings"]

        excluded = {
            "enroll_sessions",
            "enrol_sessions",
            "probe_sessions",
            "required_cross_sessions",
            "enrol_parts",
            "enroll_parts",
            "test_parts",
            "enroll_record_indices",
            "probe_record_indices",
        }

        self.assertTrue(
            excluded.isdisjoint(
                settings
            )
        )

    def test_training_role_reduction_does_not_mutate_source_identity(
        self,
    ):
        preserved = copy.deepcopy(
            self.loader_identity
        )

        self.build(
            training_role_only=True
        )

        self.assertEqual(
            self.loader_identity,
            preserved,
        )


if __name__ == "__main__":
    unittest.main()
