"""
Pin the settings the reported numbers were produced under.

The reproduction pack is the record of how each row of the paper was obtained,
so a setting that affects a result belongs in the file rather than in a
dataset-wide default that a later edit could move. These checks read every
paper configuration and assert the values the pack is meant to carry.
"""

import sys
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PAPER_CONFIGS = PROJECT_ROOT / "configs" / "paper_reproduction"

# The shortest NSRDB recording runs to 1388 minutes, so a window ending after
# that reaches only the subjects whose recording extends further.
NSRDB_RECORDING_MINUTES = 1388

IDENTIFICATION_TASKS = (1, 3, 5, 7)
VERIFICATION_TASKS = (2, 4, 6, 8)
SUBJECT_DISJOINT_TASKS = (3, 4)
CROSS_SESSION_TASKS = (5, 6, 7, 8)

# Settings that describe verification-pair sampling and therefore do not
# belong in an identification configuration.
FORBIDDEN_ON_IDENTIFICATION = (
    "pair_sampling_mode",
    "max_impostor_pairs",
    "pair_sampling_seed",
    "num_pairs",
    "sampling_mode",
)


def paper_configurations():
    if not PAPER_CONFIGS.exists():
        return []

    return sorted(PAPER_CONFIGS.rglob("*.yaml"))


def load(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _cybhi_uses_short_term_acquisition(config):
    """
    Return True when a CYBHi configuration reads from the short-term
    collection.

    CYBHi session identifiers begin with ``short-term_`` or ``long-term_`` in
    every configured role, so the acquisition regime is decided by an existing
    explicit configuration field rather than by string-matching the whole
    configuration object.
    """
    session_fields = (
        "train_sessions",
        "enroll_sessions",
        "probe_sessions",
        "session_for_single_session_evaluation",
    )

    for field in session_fields:
        sessions = config.get(field)
        if not sessions:
            continue
        for session_id in sessions:
            if str(session_id).startswith("short-term_"):
                return True
            if str(session_id).startswith("long-term_"):
                return False

    return False


class PaperConfigurationInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configurations = [
            (path, load(path))
            for path in paper_configurations()
        ]

        if not cls.configurations:
            raise unittest.SkipTest(
                "paper reproduction pack is not present"
            )

    def test_pack_is_the_expected_size(self):
        self.assertEqual(len(self.configurations), 150)

    def test_band_pass_is_fourth_order_half_to_forty(self):
        for path, config in self.configurations:
            preprocessing = config.get("preprocessing_parameters", {})
            self.assertEqual(
                preprocessing.get("filter_method"),
                "butter",
                path.name,
            )

            parameters = preprocessing.get("filter_parameters", {})
            self.assertEqual(parameters.get("low"), 0.5, path.name)
            self.assertEqual(parameters.get("high"), 40.0, path.name)
            self.assertEqual(parameters.get("order"), 4, path.name)

    def test_beat_window_and_detector_are_fixed(self):
        for path, config in self.configurations:
            preprocessing = config.get("preprocessing_parameters", {})
            self.assertEqual(preprocessing.get("mode"), "beat", path.name)
            self.assertEqual(preprocessing.get("pre_s"), 0.2, path.name)
            self.assertEqual(preprocessing.get("post_s"), 0.4, path.name)
            self.assertEqual(
                preprocessing.get("rpeak_method"),
                "pantompkins",
                path.name,
            )
            self.assertIsNone(
                preprocessing.get("resample_len"),
                path.name,
            )
            self.assertTrue(preprocessing.get("align_peak"), path.name)
            self.assertAlmostEqual(
                preprocessing.get("align_window_s"),
                0.10,
                msg=f"{path.name} does not state the alignment search width",
            )

    def test_matching_and_fusion_are_fixed(self):
        for path, config in self.configurations:
            self.assertEqual(config.get("probe_fusion_size"), 1, path.name)
            self.assertEqual(
                config.get("template_fusion_method"),
                "mean",
                path.name,
            )
            self.assertEqual(
                config.get("matching_method"),
                "cosine",
                path.name,
            )

    def test_baseline_leaves_augmentation_and_gating_off(self):
        for path, config in self.configurations:
            self.assertFalse(config.get("use_augmentation"), path.name)
            self.assertFalse(
                config.get("outlier_filtering_on_train"),
                path.name,
            )
            self.assertFalse(
                config.get("outlier_filtering_on_test"),
                path.name,
            )

    def test_repeated_run_seeds_and_validation_are_fixed(self):
        for path, config in self.configurations:
            self.assertEqual(config.get("n_runs"), 5, path.name)
            self.assertEqual(config.get("seed"), 42, path.name)
            self.assertEqual(config.get("val_split"), 0.0, path.name)

    def test_split_seed_follows_the_training_seed_schedule(self):
        # The paper pack deliberately leaves ``split_seed`` unset so the
        # randomized data-role allocation follows the per-run training seed.
        # Accept either an absent key or an explicit YAML null; reject an
        # accidental integer that would freeze the split across runs.
        for path, config in self.configurations:
            self.assertIsNone(
                config.get("split_seed"),
                f"{path.name} pins split_seed and would decouple the "
                "data-role split from the per-run training seed schedule",
            )

    def test_data_and_weight_caches_are_enabled(self):
        for path, config in self.configurations:
            self.assertTrue(
                config.get("intelligent_data_loading"),
                path.name,
            )
            self.assertTrue(
                config.get("intelligent_weight_loading"),
                path.name,
            )

    def test_verification_uses_canonical_pair_sampling(self):
        checked = 0

        for path, config in self.configurations:
            if config.get("task") not in VERIFICATION_TASKS:
                continue

            checked += 1
            self.assertEqual(
                config.get("pair_sampling_mode"),
                "all_genuine",
                path.name,
            )
            self.assertEqual(
                config.get("max_impostor_pairs"),
                1000000,
                path.name,
            )
            self.assertEqual(
                config.get("pair_sampling_seed"),
                42,
                path.name,
            )

        self.assertEqual(checked, 82)

    def test_identification_carries_no_pair_sampling_settings(self):
        checked = 0

        for path, config in self.configurations:
            if config.get("task") not in IDENTIFICATION_TASKS:
                continue

            checked += 1
            for key in FORBIDDEN_ON_IDENTIFICATION:
                self.assertNotIn(
                    key,
                    config,
                    f"{path.name} carries verification-only key {key!r}",
                )

        self.assertEqual(checked, 68)

    def test_ptbxl_is_verification_only(self):
        checked = 0

        for path, config in self.configurations:
            if config.get("dataset") != "ptbxl":
                continue

            checked += 1
            self.assertIn(
                config.get("task"),
                VERIFICATION_TASKS,
                f"{path.name} uses PTB-XL for a non-verification task",
            )

        self.assertGreater(checked, 0)

    def test_ecgid_reads_the_raw_channel(self):
        checked = 0

        for path, config in self.configurations:
            if config.get("dataset") != "ecgid":
                continue

            checked += 1
            self.assertEqual(
                config.get("signal_type"),
                "raw",
                f"{path.name} does not state the ECG-ID channel",
            )

        self.assertGreater(checked, 0)

    def test_cybhi_short_term_states_its_acquiring_unit(self):
        short_term_checked = 0
        long_term_checked = 0

        for path, config in self.configurations:
            if config.get("dataset") != "cybhi":
                continue

            if _cybhi_uses_short_term_acquisition(config):
                short_term_checked += 1
                self.assertEqual(
                    config.get("electrode_unit"),
                    "8B",
                    f"{path.name} does not state the acquiring unit",
                )
            else:
                # The long-term collection was acquired by a single unit, so
                # there is no choice to record.
                long_term_checked += 1
                self.assertIsNone(
                    config.get("electrode_unit"),
                    path.name,
                )

        self.assertGreater(short_term_checked, 0)
        self.assertGreater(long_term_checked, 0)

    def test_every_nsrdb_window_fits_the_shortest_recording(self):
        checked = 0

        for path, config in self.configurations:
            if config.get("dataset") != "nsrdb":
                continue

            for key in (
                "single_segment_range",
                "train_parts",
                "enrol_parts",
                "enroll_parts",
                "test_parts",
            ):
                ranges = config.get(key)
                if not ranges:
                    continue

                if key == "single_segment_range":
                    ranges = [ranges]

                for window in ranges:
                    checked += 1
                    self.assertLessEqual(
                        float(window[1]),
                        NSRDB_RECORDING_MINUTES,
                        f"{path.name}: window {window} runs past the "
                        "shortest recording",
                    )

        self.assertGreater(checked, 0)

    def test_enrollment_budget_matches_the_task_family(self):
        subject_disjoint = 0

        for path, config in self.configurations:
            task = config.get("task")

            if task in SUBJECT_DISJOINT_TASKS:
                subject_disjoint += 1
                self.assertEqual(
                    config.get("template_size"),
                    1,
                    f"{path.name} leaves the enrollment budget unstated",
                )
            elif task in CROSS_SESSION_TASKS:
                self.assertIsNone(
                    config.get("template_size"),
                    f"{path.name} should enroll on its whole partition",
                )

        self.assertEqual(subject_disjoint, 20)


if __name__ == "__main__":
    unittest.main()
