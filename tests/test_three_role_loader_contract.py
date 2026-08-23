import datetime
import unittest

import numpy as np

import utils
from load_dataset import (
    CUSTOM_RECORD_SPLIT_MODE,
    RECORD_ORDER_SPLIT_MODES,
    _record_partition_assignment,
    load_cybhi_dataset,
    load_ecgid_dataset,
    load_heartprint_dataset,
    load_mitbih_dataset,
    load_nsrdb_dataset,
    load_ptb_dataset,
    load_ptbxl_dataset,
    select_record_order_partition,
    select_record_role_partition,
    summarize_partition_log,
)


RECORD_LOADERS = (
    load_ecgid_dataset,
    load_ptb_dataset,
    load_ptbxl_dataset,
)


def make_records(count):
    base = datetime.date(2024, 1, 1)

    return [
        {
            "signal": np.asarray(
                [[float(index)]],
                dtype=np.float32,
            ),
            "fs": 250,
            "date": (
                base
                + datetime.timedelta(days=index)
            ),
            "filename": f"rec_{index}.hea",
        }
        for index in range(count)
    ]


def make_legacy_records():
    day_1 = datetime.date(2024, 1, 1)

    dates = [
        day_1,
        day_1,
        datetime.date(2024, 2, 1),
        datetime.date(2024, 3, 1),
    ]

    return [
        {
            "signal": np.asarray(
                [[float(index)]],
                dtype=np.float32,
            ),
            "fs": 250,
            "date": date,
            "filename": f"rec_{index}.hea",
        }
        for index, date in enumerate(dates)
    ]


class CustomRecordRoleTests(unittest.TestCase):
    def test_train_and_enrollment_may_share_a_record(self):
        for loader_class in RECORD_LOADERS:
            with self.subTest(loader=loader_class.__name__):
                loader = loader_class(
                    data_split_mode=CUSTOM_RECORD_SPLIT_MODE,
                    train_record_indices=[0],
                    enroll_record_indices=[0],
                    probe_record_indices=[1],
                )

                self.assertEqual(
                    loader.train_record_indices,
                    (0,),
                )
                self.assertEqual(
                    loader.enroll_record_indices,
                    (0,),
                )
                self.assertEqual(
                    loader.probe_record_indices,
                    (1,),
                )

    def test_omitted_enrollment_records_inherit_training_records(self):
        for loader_class in RECORD_LOADERS:
            with self.subTest(loader=loader_class.__name__):
                loader = loader_class(
                    data_split_mode=CUSTOM_RECORD_SPLIT_MODE,
                    train_record_indices=[1, 0],
                    probe_record_indices=[2],
                )

                self.assertEqual(
                    loader.train_record_indices,
                    (0, 1),
                )
                self.assertEqual(
                    loader.enroll_record_indices,
                    (0, 1),
                )

    def test_probe_overlap_is_rejected(self):
        for loader_class in RECORD_LOADERS:
            with self.subTest(loader=loader_class.__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    "Probe recordings must be disjoint",
                ):
                    loader_class(
                        data_split_mode=CUSTOM_RECORD_SPLIT_MODE,
                        train_record_indices=[0],
                        enroll_record_indices=[1],
                        probe_record_indices=[1],
                    )

    def test_record_selectors_require_custom_mode(self):
        for loader_class in RECORD_LOADERS:
            with self.subTest(loader=loader_class.__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    "custom-record-split",
                ):
                    loader_class(
                        data_split_mode="single-cross-session",
                        train_record_indices=[0],
                        probe_record_indices=[1],
                    )

    def test_invalid_record_indices_are_rejected(self):
        invalid_values = (
            [-1],
            [0, 0],
            [True],
            [0.5],
            [],
        )

        for invalid in invalid_values:
            with self.subTest(indices=invalid):
                with self.assertRaises(ValueError):
                    load_ecgid_dataset(
                        data_split_mode=CUSTOM_RECORD_SPLIT_MODE,
                        train_record_indices=invalid,
                        probe_record_indices=[2],
                    )

    def test_subject_missing_any_required_record_is_ineligible_for_every_role(self):
        records = make_records(2)

        for role in (
            "train",
            "enrollment",
            "probe",
        ):
            with self.subTest(role=role):
                selected, eligible = (
                    select_record_role_partition(
                        records,
                        CUSTOM_RECORD_SPLIT_MODE,
                        role,
                        train_record_indices=[0],
                        enroll_record_indices=[1],
                        probe_record_indices=[2],
                    )
                )

                self.assertFalse(eligible)
                self.assertEqual(selected, [])

    def test_custom_role_selection_uses_deterministic_source_order(self):
        records = make_records(4)

        selected, eligible = (
            select_record_role_partition(
                records,
                CUSTOM_RECORD_SPLIT_MODE,
                "train",
                train_record_indices=[2, 0],
                enroll_record_indices=[1],
                probe_record_indices=[3],
            )
        )

        self.assertTrue(eligible)
        self.assertEqual(
            [
                record["filename"]
                for record in selected
            ],
            [
                "rec_0.hea",
                "rec_2.hea",
            ],
        )

    def test_legacy_record_order_modes_are_unchanged(self):
        records = make_legacy_records()

        role_flags = (
            ("train", True),
            ("enrollment", True),
            ("probe", False),
        )

        for mode in RECORD_ORDER_SPLIT_MODES:
            for role, is_enrollment in role_flags:
                with self.subTest(
                    mode=mode,
                    role=role,
                ):
                    legacy_records, legacy_eligible = (
                        select_record_order_partition(
                            records,
                            mode,
                            is_enrollment,
                        )
                    )

                    role_records, role_eligible = (
                        select_record_role_partition(
                            records,
                            mode,
                            role,
                        )
                    )

                    self.assertEqual(
                        role_eligible,
                        legacy_eligible,
                    )
                    self.assertEqual(
                        [
                            record["filename"]
                            for record in role_records
                        ],
                        [
                            record["filename"]
                            for record in legacy_records
                        ],
                    )

    def test_record_loaders_drop_subjects_missing_any_required_role(self):
        for loader_class in RECORD_LOADERS:
            with self.subTest(loader=loader_class.__name__):
                loader = loader_class(
                    data_split_mode=CUSTOM_RECORD_SPLIT_MODE,
                    train_record_indices=[0],
                    enroll_record_indices=[1],
                    probe_record_indices=[2],
                )

                loader.load_raw_data = lambda: {
                    "complete": make_records(3),
                    "incomplete": make_records(2),
                }

                loader._process_signal = (
                    lambda signal, sampling_rate:
                    np.full(
                        (1, 4),
                        float(
                            np.asarray(signal)
                            .reshape(-1)[0]
                        ),
                        dtype=np.float32,
                    )
                )

                expected_markers = {
                    "train": 0.0,
                    "enrollment": 1.0,
                    "probe": 2.0,
                }

                for role, marker in expected_markers.items():
                    x, y = loader.load_session(role)

                    self.assertEqual(
                        set(y.tolist()),
                        {"complete"},
                    )
                    self.assertTrue(
                        np.all(x == marker)
                    )


class ExistingThreeRoleLoaderFallbackTests(unittest.TestCase):
    def test_session_enrollment_defaults_to_training_selection(self):
        heartprint = load_heartprint_dataset(
            train_sessions=["session2"],
            enroll_sessions=None,
            probe_sessions=["session1"],
        )

        self.assertEqual(
            heartprint.enroll_sessions,
            heartprint.train_sessions,
        )

        cybhi = load_cybhi_dataset(
            train_sessions=["long-term_S2"],
            enroll_sessions=None,
            probe_sessions=["long-term_S1"],
        )

        self.assertEqual(
            cybhi.enroll_sessions,
            cybhi.train_sessions,
        )

    def test_continuous_enrollment_defaults_to_training_ranges(self):
        for loader_class in (
            load_mitbih_dataset,
            load_nsrdb_dataset,
        ):
            with self.subTest(loader=loader_class.__name__):
                loader = loader_class(
                    data_split_mode="custom-split",
                    train_parts=[(0, 5)],
                    enrol_parts=None,
                    test_parts=[(10, 15)],
                )

                self.assertEqual(
                    loader.enrol_parts,
                    [(0, 5)],
                )


class ThreeRolePartitionLogTests(unittest.TestCase):
    class Loader:
        data_split_mode = CUSTOM_RECORD_SPLIT_MODE

    def test_training_and_enrollment_overlap_is_allowed(self):
        loader = self.Loader()
        records = make_records(2)

        _record_partition_assignment(
            loader,
            "train",
            "subject_a",
            [records[0]],
        )
        _record_partition_assignment(
            loader,
            "enrollment",
            "subject_a",
            [records[0]],
        )
        _record_partition_assignment(
            loader,
            "probe",
            "subject_a",
            [records[1]],
        )

        report = summarize_partition_log(loader)

        self.assertEqual(
            report["violations"],
            [],
        )
        self.assertTrue(
            report["enrollment_precedes_probe"]
        )

    def test_training_probe_overlap_is_reported(self):
        loader = self.Loader()
        records = make_records(1)

        _record_partition_assignment(
            loader,
            "train",
            "subject_a",
            records,
        )
        _record_partition_assignment(
            loader,
            "probe",
            "subject_a",
            records,
        )

        report = summarize_partition_log(loader)

        self.assertFalse(
            report["enrollment_precedes_probe"]
        )
        self.assertIn(
            "share recording",
            report["violations"][0]["reasons"][0],
        )


class RecordSelectorCacheIdentityTests(unittest.TestCase):
    def test_custom_record_assignment_reaches_loader_identity(self):
        first = load_ecgid_dataset(
            data_split_mode=CUSTOM_RECORD_SPLIT_MODE,
            train_record_indices=[0],
            enroll_record_indices=[1],
            probe_record_indices=[2],
        )

        second = load_ecgid_dataset(
            data_split_mode=CUSTOM_RECORD_SPLIT_MODE,
            train_record_indices=[0],
            enroll_record_indices=[0],
            probe_record_indices=[2],
        )

        first_settings = (
            utils._build_loader_cache_identity(
                first
            )["settings"]
        )

        second_settings = (
            utils._build_loader_cache_identity(
                second
            )["settings"]
        )

        for key in (
            "train_record_indices",
            "enroll_record_indices",
            "probe_record_indices",
        ):
            self.assertIn(
                key,
                first_settings,
            )

        self.assertNotEqual(
            first_settings,
            second_settings,
        )

    def test_legacy_loader_does_not_gain_unused_record_selector_settings(self):
        loader = load_ecgid_dataset()

        settings = (
            utils._build_loader_cache_identity(
                loader
            )["settings"]
        )

        self.assertNotIn(
            "train_record_indices",
            settings,
        )
        self.assertNotIn(
            "enroll_record_indices",
            settings,
        )
        self.assertNotIn(
            "probe_record_indices",
            settings,
        )


if __name__ == "__main__":
    unittest.main()
