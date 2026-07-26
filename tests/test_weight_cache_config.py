import sys
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run
from run import _build_weight_cache_config
from utils import _generate_config_hash


class DummyLoader:
    def __init__(
        self,
        root_dir="dataset_a",
        signal_type="filtered",
        train_sessions=None,
    ):
        self.cfg = {
            "root_dir": root_dir,
            "preprocessing": {
                "filter_method": "butter",
                "resample_len": 256,
                "norm_method": "zscore",
            },
        }

        # Explicit loader overrides must replace configured defaults.
        self.prep_params = {
            "resample_len": 128,
        }

        self.data_split_mode = "cross-session"
        self.num_beats = 2
        self.merge_strategy = "average"
        self.signal_type = signal_type
        self.train_sessions = train_sessions or ["session_1"]
        self.enroll_sessions = ["session_2"]
        self.probe_sessions = ["session_3"]
        self.target_leads = ["MLII"]
        self.only_healthy = False
        self.resolution = 100
        self.limit_records = None


def make_training_config():
    return {
        "training_regime": "cross_session_closed_set",
        "model": "DeepECG",
        "epochs": 150,
        "batch_size": 256,
        "lr": 0.001,
        "val_split": 0.0,
        "seed": 42,
        "classes": 20,
        "data_shape": (1000, 256),
    }


class WeightCacheConfigurationTests(unittest.TestCase):
    def test_loader_identity_is_added(self):
        config = _build_weight_cache_config(
            DummyLoader(),
            make_training_config(),
        )

        identity = config["loader_identity"]

        self.assertEqual(
            identity["loader_class"],
            "DummyLoader",
        )
        self.assertEqual(
            identity["root_dir"],
            "dataset_a",
        )
        self.assertEqual(
            identity["settings"]["signal_type"],
            "filtered",
        )
        self.assertEqual(
            identity["settings"]["train_sessions"],
            ["session_1"],
        )

    def test_preprocessing_overrides_are_applied(self):
        config = _build_weight_cache_config(
            DummyLoader(),
            make_training_config(),
        )

        preprocessing = config["loader_identity"]["preprocessing"]

        self.assertEqual(
            preprocessing["filter_method"],
            "butter",
        )
        self.assertEqual(
            preprocessing["norm_method"],
            "zscore",
        )
        self.assertEqual(
            preprocessing["resample_len"],
            128,
        )

    def test_dataset_identity_changes_weight_hash(self):
        base_training_config = make_training_config()

        config_1 = _build_weight_cache_config(
            DummyLoader(root_dir="dataset_a"),
            base_training_config,
        )
        config_2 = _build_weight_cache_config(
            DummyLoader(root_dir="dataset_b"),
            base_training_config,
        )

        self.assertNotEqual(
            _generate_config_hash(config_1),
            _generate_config_hash(config_2),
        )

    def test_session_selection_changes_weight_hash(self):
        base_training_config = make_training_config()

        config_1 = _build_weight_cache_config(
            DummyLoader(train_sessions=["session_1"]),
            base_training_config,
        )
        config_2 = _build_weight_cache_config(
            DummyLoader(train_sessions=["session_4"]),
            base_training_config,
        )

        self.assertNotEqual(
            _generate_config_hash(config_1),
            _generate_config_hash(config_2),
        )

    def test_signal_type_changes_weight_hash(self):
        base_training_config = make_training_config()

        config_1 = _build_weight_cache_config(
            DummyLoader(signal_type="filtered"),
            base_training_config,
        )
        config_2 = _build_weight_cache_config(
            DummyLoader(signal_type="raw"),
            base_training_config,
        )

        self.assertNotEqual(
            _generate_config_hash(config_1),
            _generate_config_hash(config_2),
        )

    def test_input_training_config_is_not_modified(self):
        original = make_training_config()
        preserved = deepcopy(original)

        _build_weight_cache_config(
            DummyLoader(),
            original,
        )

        self.assertEqual(original, preserved)

    def test_none_loader_is_supported(self):
        config = _build_weight_cache_config(
            None,
            make_training_config(),
        )

        self.assertIsNone(config["loader_identity"])

    def test_effective_loader_settings_are_added(self):
        config = _build_weight_cache_config(
            DummyLoader(),
            make_training_config(),
        )

        settings = config[
            "loader_identity"
        ]["settings"]

        self.assertEqual(
            settings["num_beats"],
            2,
        )
        self.assertEqual(
            settings["merge_strategy"],
            "average",
        )
        self.assertEqual(
            settings["target_leads"],
            ["MLII"],
        )
        self.assertFalse(
            settings["only_healthy"]
        )
        self.assertEqual(
            settings["resolution"],
            100,
        )
        self.assertIsNone(
            settings["limit_records"]
        )

    def test_training_partition_fingerprint_is_deterministic(self):
        samples = np.arange(
            24,
            dtype=np.float32,
        ).reshape(4, 6)
        labels = np.asarray(
            [
                "subject_1",
                "subject_1",
                "subject_2",
                "subject_2",
            ],
            dtype=object,
        )

        config_1 = _build_weight_cache_config(
            DummyLoader(),
            make_training_config(),
            training_samples=samples,
            training_labels=labels,
        )
        config_2 = _build_weight_cache_config(
            DummyLoader(),
            make_training_config(),
            training_samples=samples.copy(),
            training_labels=labels.copy(),
        )

        self.assertEqual(
            config_1["training_partition"],
            config_2["training_partition"],
        )
        self.assertEqual(
            _generate_config_hash(config_1),
            _generate_config_hash(config_2),
        )

    def test_training_sample_content_changes_weight_hash(self):
        samples_1 = np.arange(
            24,
            dtype=np.float32,
        ).reshape(4, 6)
        samples_2 = samples_1.copy()
        samples_2[0, 0] += 1.0

        labels = np.asarray(
            [0, 0, 1, 1],
            dtype=np.int64,
        )

        config_1 = _build_weight_cache_config(
            DummyLoader(),
            make_training_config(),
            training_samples=samples_1,
            training_labels=labels,
        )
        config_2 = _build_weight_cache_config(
            DummyLoader(),
            make_training_config(),
            training_samples=samples_2,
            training_labels=labels,
        )

        self.assertNotEqual(
            _generate_config_hash(config_1),
            _generate_config_hash(config_2),
        )

    def test_training_label_content_changes_weight_hash(self):
        samples = np.arange(
            24,
            dtype=np.float32,
        ).reshape(4, 6)

        labels_1 = np.asarray(
            [0, 0, 1, 1],
            dtype=np.int64,
        )
        labels_2 = np.asarray(
            [0, 1, 0, 1],
            dtype=np.int64,
        )

        config_1 = _build_weight_cache_config(
            DummyLoader(),
            make_training_config(),
            training_samples=samples,
            training_labels=labels_1,
        )
        config_2 = _build_weight_cache_config(
            DummyLoader(),
            make_training_config(),
            training_samples=samples,
            training_labels=labels_2,
        )

        self.assertNotEqual(
            _generate_config_hash(config_1),
            _generate_config_hash(config_2),
        )

    def test_partial_training_partition_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "must be supplied together",
        ):
            _build_weight_cache_config(
                DummyLoader(),
                make_training_config(),
                training_samples=np.zeros(
                    (2, 4),
                    dtype=np.float32,
                ),
            )

    def test_all_runners_fingerprint_the_training_partition(self):
        run_source = Path(
            run.__file__
        ).read_text(
            encoding="utf-8"
        )

        self.assertEqual(
            run_source.count(
                "training_samples=X_tr"
            ),
            8,
        )
        self.assertEqual(
            run_source.count(
                "training_labels=y_tr"
            ),
            8,
        )


if __name__ == "__main__":
    unittest.main()
