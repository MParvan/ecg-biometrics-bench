import copy
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import utils
from load_dataset import (
    load_ecgid_dataset,
    load_mitbih_dataset,
    load_nsrdb_dataset,
)


def build_legacy_loader_identity(loader):
    """
    Reconstruct the loader identity as it was before default-elided options.

    Cached preprocessed arrays and trained weights are keyed by this
    structure. A loader option added later must leave it untouched while it
    carries its neutral default, otherwise every existing cache entry
    silently becomes unreachable.
    """
    loader_cfg = getattr(loader, "cfg", {})

    dataset_config = {}
    effective_preprocessing = {}

    if isinstance(loader_cfg, dict):
        dataset_config = copy.deepcopy(loader_cfg)

        configured_preprocessing = dataset_config.pop(
            "preprocessing",
            {},
        )

        if isinstance(configured_preprocessing, dict):
            effective_preprocessing.update(
                configured_preprocessing
            )

    preprocessing_overrides = getattr(
        loader,
        "prep_params",
        {},
    )

    if isinstance(preprocessing_overrides, dict):
        effective_preprocessing.update(
            preprocessing_overrides
        )

    loader_settings = {}

    for attribute_name in utils._CACHE_RELEVANT_LOADER_ATTRIBUTES:
        if hasattr(loader, attribute_name):
            loader_settings[attribute_name] = copy.deepcopy(
                getattr(
                    loader,
                    attribute_name,
                )
            )

    return {
        "loader_class": type(loader).__name__,
        "root_dir": dataset_config.get("root_dir"),
        "dataset_config": dataset_config,
        "preprocessing": effective_preprocessing,
        "settings": loader_settings,
    }


class DefaultCacheIdentityTests(unittest.TestCase):
    """
    Default-configured loaders keep the cache identity they already had.
    """

    def assert_identity_unchanged(self, loader):
        legacy_hash = utils._generate_config_hash(
            build_legacy_loader_identity(loader)
        )
        current_hash = utils._generate_config_hash(
            utils._build_loader_cache_identity(loader)
        )

        self.assertEqual(
            current_hash,
            legacy_hash,
            "A loader option changed the cache identity of a "
            "default-configured loader, which would invalidate every "
            "previously computed cache entry.",
        )

    def test_ecgid_default_identity_is_stable(self):
        self.assert_identity_unchanged(
            load_ecgid_dataset(
                data_split_mode="all-available"
            )
        )

    def test_submitted_mitbih_protocol_identity_is_stable(self):
        self.assert_identity_unchanged(
            load_mitbih_dataset(
                data_split_mode="custom-split",
                train_parts=[(0, 5), (12.5, 17.5)],
                enrol_parts=[(0, 5), (12.5, 17.5)],
                test_parts=[(25, 30)],
            )
        )

    def test_submitted_nsrdb_protocol_identity_is_stable(self):
        self.assert_identity_unchanged(
            load_nsrdb_dataset(
                data_split_mode="custom-split",
                train_parts=[
                    (0, 5),
                    (120, 125),
                    (360, 365),
                    (720, 725),
                ],
                enrol_parts=[
                    (0, 5),
                    (120, 125),
                    (360, 365),
                    (720, 725),
                ],
                test_parts=[(1430, 1435)],
            )
        )

    def test_default_options_are_absent_from_the_identity(self):
        loader = load_ecgid_dataset(
            data_split_mode="all-available"
        )

        settings = utils._build_loader_cache_identity(
            loader
        )["settings"]

        for attribute_name in (
            utils._DEFAULT_ELIDED_LOADER_ATTRIBUTES
        ):
            self.assertNotIn(
                attribute_name,
                settings,
            )


class NonDefaultCacheIdentityTests(unittest.TestCase):
    """
    A deliberately changed option must still invalidate the cache.
    """

    def test_non_default_stride_changes_the_identity(self):
        default_loader = load_ecgid_dataset(
            num_beats_to_merge=3,
        )
        strided_loader = load_ecgid_dataset(
            num_beats_to_merge=3,
            beat_merge_stride=3,
        )

        self.assertNotEqual(
            utils._generate_config_hash(
                utils._build_loader_cache_identity(
                    default_loader
                )
            ),
            utils._generate_config_hash(
                utils._build_loader_cache_identity(
                    strided_loader
                )
            ),
        )

    def test_non_default_guard_changes_the_identity(self):
        default_loader = load_mitbih_dataset(
            data_split_mode="custom-split",
            train_parts=[(0, 5)],
            test_parts=[(25, 30)],
        )
        guarded_loader = load_mitbih_dataset(
            data_split_mode="custom-split",
            train_parts=[(0, 5)],
            test_parts=[(25, 30)],
            temporal_guard_minutes=10.0,
        )

        self.assertNotEqual(
            utils._generate_config_hash(
                utils._build_loader_cache_identity(
                    default_loader
                )
            ),
            utils._generate_config_hash(
                utils._build_loader_cache_identity(
                    guarded_loader
                )
            ),
        )

    def test_non_default_stride_appears_in_the_identity(self):
        loader = load_ecgid_dataset(
            num_beats_to_merge=3,
            beat_merge_stride=3,
        )

        settings = utils._build_loader_cache_identity(
            loader
        )["settings"]

        self.assertEqual(
            settings["beat_merge_stride"],
            3,
        )


class NeutralDefaultComparisonTests(unittest.TestCase):
    """
    Boolean values must not be mistaken for their numeric equivalents.
    """

    def test_boolean_is_not_a_neutral_integer_default(self):
        self.assertFalse(
            utils._is_neutral_default(True, 1)
        )

    def test_integer_matches_float_default(self):
        self.assertTrue(
            utils._is_neutral_default(0, 0.0)
        )

    def test_unequal_values_are_not_neutral(self):
        self.assertFalse(
            utils._is_neutral_default(2, 1)
        )


if __name__ == "__main__":
    unittest.main()
