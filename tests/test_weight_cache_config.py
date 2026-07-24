import sys
import unittest
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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
        self.num_beats_to_merge = 2
        self.beat_merge_method = "average"
        self.signal_type = signal_type
        self.train_sessions = train_sessions or ["session_1"]
        self.enroll_sessions = ["session_2"]
        self.probe_sessions = ["session_3"]
        self.leads = ["MLII"]


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


if __name__ == "__main__":
    unittest.main()