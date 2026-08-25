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
    def __init__(self, signal_type="filtered", electrode_unit=None):
        # The channel is resolved by the loader, which may take it from the
        # dataset configuration rather than from the command line, so the cache
        # identity reads it back from here.
        self.signal_type = signal_type

        # The CYBHi acquiring unit is resolved the same way: it may be left
        # unset on the command line and taken from the dataset configuration.
        self.electrode_unit = electrode_unit

        self.cfg = {
            "root_dir": "synthetic_dataset",
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

        self.data_split_mode = "cross-session"
        self.num_beats = 2
        self.merge_strategy = "average"
        self.target_leads = ["MLII"]
        self.only_healthy = False
        self.resolution = 100
        self.limit_records = None
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
    def test_cache_configuration_contains_data_compatibility_identity(self):
        config = build_data_cache_config(
            make_args(),
            DummyLoader(),
            task_type="cross_session",
        )

        implementation = config["implementation_identity"]
        dependencies = config["dependency_identity"]
        self.assertEqual(
            set(implementation["components"]),
            {"load_dataset", "preprocessing", "filtering"},
        )
        self.assertEqual(len(implementation["aggregate_sha256"]), 64)
        self.assertEqual(
            set(dependencies["distributions"]),
            {"numpy", "scipy", "neurokit2", "wfdb", "pandas"},
        )
        self.assertEqual(len(dependencies["aggregate_sha256"]), 64)

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
        config_1 = build_data_cache_config(
            make_args(),
            DummyLoader(signal_type="filtered"),
            task_type="cross_session",
        )
        config_2 = build_data_cache_config(
            make_args(),
            DummyLoader(signal_type="raw"),
            task_type="cross_session",
        )

        self.assertNotEqual(
            _generate_config_hash(config_1),
            _generate_config_hash(config_2),
        )

    def test_cache_records_the_channel_the_loader_resolved(self):
        """
        A run that leaves the channel unset on the command line still reads a
        specific channel, so the cache identity must name that channel rather
        than the absent argument.
        """
        args = make_args()
        args.signal_type = None

        config = build_data_cache_config(
            args,
            DummyLoader(signal_type="raw"),
            task_type="cross_session",
        )

        self.assertEqual(
            config["signal_type"],
            "raw",
        )

    def test_electrode_unit_changes_the_cache_hash(self):
        config_1 = build_data_cache_config(
            make_args(),
            DummyLoader(electrode_unit="8B"),
            task_type="cross_session",
        )
        config_2 = build_data_cache_config(
            make_args(),
            DummyLoader(electrode_unit="85"),
            task_type="cross_session",
        )

        self.assertNotEqual(
            _generate_config_hash(config_1),
            _generate_config_hash(config_2),
        )

    def test_cache_records_the_unit_the_loader_resolved(self):
        """
        The CYBHi acquiring unit may be left unset on the command line and
        resolved from config.yaml, so the cache identity must name the unit the
        loader actually read rather than the absent argument.
        """
        args = make_args()
        args.electrode_unit = None

        config = build_data_cache_config(
            args,
            DummyLoader(electrode_unit="85"),
            task_type="cross_session",
        )

        self.assertEqual(
            config["electrode_unit"],
            "85",
        )

    def test_effective_loader_settings_are_recorded(self):
        config = build_data_cache_config(
            make_args(),
            DummyLoader(),
            task_type="cross_session",
        )

        settings = config["loader_settings"]

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
        self.assertEqual(
            config["dataset_config"]["root_dir"],
            "synthetic_dataset",
        )

    def test_effective_loader_settings_change_data_hash(self):
        changed_values = {
            "num_beats": 4,
            "merge_strategy": "concatenate",
            "target_leads": ["V1"],
            "only_healthy": True,
            "resolution": 500,
            "limit_records": 50,
        }

        for attribute_name, changed_value in changed_values.items():
            with self.subTest(
                attribute=attribute_name
            ):
                loader_1 = DummyLoader()
                loader_2 = DummyLoader()

                setattr(
                    loader_2,
                    attribute_name,
                    changed_value,
                )

                config_1 = build_data_cache_config(
                    make_args(),
                    loader_1,
                    task_type="cross_session",
                )
                config_2 = build_data_cache_config(
                    make_args(),
                    loader_2,
                    task_type="cross_session",
                )

                self.assertNotEqual(
                    _generate_config_hash(
                        config_1
                    ),
                    _generate_config_hash(
                        config_2
                    ),
                )


if __name__ == "__main__":
    unittest.main()
