import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import (
    build_data_cache_config
)
from utils import _generate_config_hash


class DummyLoader:
    def __init__(self):
        self.cfg = {
            "preprocessing": {
                "filter_method": "butter",
                "resample_len": 256,
                "norm_method": "zscore",
            }
        }

        # User overrides must replace configured defaults.
        self.prep_params = {
            "resample_len": 128,
        }

        self.leads = ["MLII"]
        self.beat_merge_method = "average"
        self.single_segment_range = (0, 5)
        self.train_parts = [(0, 5)]
        self.enrol_parts = [(5, 10)]
        self.test_parts = [(10, 15)]


def make_args():
    return SimpleNamespace(
        dataset="ecgid",
        data_split_mode="cross-session",
        num_beats_to_merge=2,
        signal_type="filtered",
        train_sessions=["session_1"],
        enroll_sessions=["session_2"],
        probe_sessions=["session_3"],
        session_for_single_session_evaluation=["session_1"],
    )


class DataCacheConfigurationTests(unittest.TestCase):
    def test_cache_configuration_contains_all_routing_fields(self):
        config = build_data_cache_config(
            make_args(),
            DummyLoader(),
            task_type="cross_session",
        )


        self.assertEqual(config["dataset"], "ecgid")
        self.assertEqual(config["task_type"], "cross_session")
        self.assertEqual(config["signal_type"], "filtered")
        self.assertEqual(
            config["train_sessions"],
            ["session_1"],
        )
        self.assertEqual(
            config["enroll_sessions"],
            ["session_2"],
        )
        self.assertEqual(
            config["probe_sessions"],
            ["session_3"],
        )
        self.assertEqual(
            config["session_for_single_session_evaluation"],
            ["session_1"],
        )

    def test_preprocessing_overrides_replace_configured_defaults(self):
        config = build_data_cache_config(
            make_args(),
            DummyLoader(),
            task_type="cross_session",
        )

        self.assertEqual(
            config["preprocessing"]["filter_method"],
            "butter",
        )
        self.assertEqual(
            config["preprocessing"]["norm_method"],
            "zscore",
        )
        self.assertEqual(
            config["preprocessing"]["resample_len"],
            128,
        )

    def test_enrollment_session_changes_the_cache_hash(self):
        args_1 = make_args()
        args_2 = deepcopy(args_1)
        args_2.enroll_sessions = ["different_session"]

        config_1 = build_data_cache_config(
            args_1,
            DummyLoader(),
            task_type="cross_session",
        )
        config_2 = build_data_cache_config(
            args_2,
            DummyLoader(),
            task_type="cross_session",
        )

        self.assertNotEqual(
            _generate_config_hash(config_1),
            _generate_config_hash(config_2),
        )

    def test_signal_type_changes_the_cache_hash(self):
        args_1 = make_args()
        args_2 = deepcopy(args_1)
        args_2.signal_type = "raw"

        config_1 = build_data_cache_config(
            args_1,
            DummyLoader(),
            task_type="cross_session",
        )
        config_2 = build_data_cache_config(
            args_2,
            DummyLoader(),
            task_type="cross_session",
        )

        self.assertNotEqual(
            _generate_config_hash(config_1),
            _generate_config_hash(config_2),
        )


if __name__ == "__main__":
    unittest.main()