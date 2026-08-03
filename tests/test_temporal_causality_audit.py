import datetime
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from load_dataset import (
    RECORD_ORDER_SPLIT_MODES,
    audit_record_order_causality,
    select_record_order_partition,
    summarize_partition_log,
)


def make_records(date_offsets):
    """
    Build subject recordings sorted by (date, record order).
    """
    base_date = datetime.date(2024, 1, 1)

    return [
        {
            "date": base_date + datetime.timedelta(days=offset),
            "filename": f"rec_{index}.hea",
        }
        for index, offset in enumerate(date_offsets)
    ]


def legacy_select(records, data_split_mode, is_enrollment):
    """
    Reference implementation of the regime selection rules.

    The shared helper must reproduce this exactly. Any divergence would
    silently change which recordings are enrolled and which are probed.
    """
    if not records:
        return [], False

    unique_dates = sorted({record["date"] for record in records})
    day1_date = unique_dates[0]
    day1_recs = [
        record
        for record in records
        if record["date"] == day1_date
    ]

    if data_split_mode == "single-cross-session":
        if len(records) < 2:
            return [], False
        return (
            [records[0]] if is_enrollment else [records[1]],
            True,
        )

    if data_split_mode == "single-shot-short-term":
        if len(day1_recs) < 2:
            return [], False
        return (
            [day1_recs[0]] if is_enrollment else day1_recs[1:],
            True,
        )

    if data_split_mode == "leave-last-out-short-term":
        if len(day1_recs) < 2:
            return [], False
        return (
            day1_recs[:-1] if is_enrollment else [day1_recs[-1]],
            True,
        )

    if data_split_mode == "single-shot-long-term":
        if len(unique_dates) < 2:
            return [], False
        return (
            day1_recs
            if is_enrollment
            else [
                record
                for record in records
                if record["date"] > day1_date
            ],
            True,
        )

    if data_split_mode == "leave-last-out-long-term":
        if len(unique_dates) < 2:
            return [], False
        last_date = unique_dates[-1]
        return (
            [
                record
                for record in records
                if record["date"] < last_date
            ]
            if is_enrollment
            else [
                record
                for record in records
                if record["date"] == last_date
            ],
            True,
        )

    raise AssertionError(data_split_mode)


SUBJECT_SHAPES = {
    "single_record": [0],
    "two_same_day": [0, 0],
    "three_same_day": [0, 0, 0],
    "two_days_one_each": [0, 5],
    "two_days_multi_first": [0, 0, 5],
    "three_days": [0, 3, 3, 90],
    "many_same_day_then_later": [0, 0, 0, 30, 30],
}


class RegimeExtractionEquivalenceTests(unittest.TestCase):
    """
    The extracted selector must be numerically identical to the original code.
    """

    def test_selection_matches_the_reference_implementation(self):
        for mode in RECORD_ORDER_SPLIT_MODES:
            for shape_name, offsets in SUBJECT_SHAPES.items():
                records = make_records(offsets)

                for is_enrollment in (True, False):
                    with self.subTest(
                        mode=mode,
                        shape=shape_name,
                        enrollment=is_enrollment,
                    ):
                        expected_records, expected_ok = legacy_select(
                            records,
                            mode,
                            is_enrollment,
                        )
                        actual_records, actual_ok = (
                            select_record_order_partition(
                                records,
                                mode,
                                is_enrollment,
                            )
                        )

                        self.assertEqual(actual_ok, expected_ok)
                        self.assertEqual(
                            [
                                record["filename"]
                                for record in actual_records
                            ],
                            [
                                record["filename"]
                                for record in expected_records
                            ],
                        )

    def test_unsupported_mode_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            select_record_order_partition(
                make_records([0, 1]),
                "all-available",
                True,
            )

        self.assertIn(
            "Unsupported record-order split mode",
            str(raised.exception),
        )

    def test_empty_subject_is_ineligible(self):
        self.assertEqual(
            select_record_order_partition(
                [],
                "single-cross-session",
                True,
            ),
            ([], False),
        )


class CausalityAuditTests(unittest.TestCase):
    """
    Every regime must build templates only from recordings prior to the probe.
    """

    def _records_by_subject(self):
        return {
            subject_id: make_records(offsets)
            for subject_id, offsets in SUBJECT_SHAPES.items()
        }

    def test_every_regime_is_causal(self):
        for mode in RECORD_ORDER_SPLIT_MODES:
            with self.subTest(mode=mode):
                report = audit_record_order_causality(
                    self._records_by_subject(),
                    mode,
                )

                self.assertTrue(
                    report["enrollment_precedes_probe"],
                    f"{mode} reported violations: "
                    f"{report['violations']}",
                )
                self.assertEqual(report["violations"], [])

    def test_leave_last_out_long_term_enrolls_only_past_days(self):
        report = audit_record_order_causality(
            {"subject_a": make_records([0, 3, 3, 90])},
            "leave-last-out-long-term",
        )

        subject_report = report["subject_reports"][0]

        self.assertEqual(
            subject_report["enrollment_records"],
            ["rec_0.hea", "rec_1.hea", "rec_2.hea"],
        )
        self.assertEqual(
            subject_report["probe_records"],
            ["rec_3.hea"],
        )
        self.assertLess(
            subject_report["latest_enrollment_date"],
            subject_report["earliest_probe_date"],
        )

    def test_single_shot_long_term_probes_only_future_days(self):
        report = audit_record_order_causality(
            {"subject_a": make_records([0, 0, 5, 90])},
            "single-shot-long-term",
        )

        subject_report = report["subject_reports"][0]

        self.assertEqual(
            subject_report["enrollment_records"],
            ["rec_0.hea", "rec_1.hea"],
        )
        self.assertEqual(
            subject_report["probe_records"],
            ["rec_2.hea", "rec_3.hea"],
        )

    def test_ineligible_subjects_are_excluded_not_flagged(self):
        report = audit_record_order_causality(
            {"subject_a": make_records([0])},
            "leave-last-out-long-term",
        )

        self.assertEqual(report["subjects_supplied"], 1)
        self.assertEqual(report["subjects_eligible"], 0)
        self.assertEqual(report["violations"], [])

    def test_audit_detects_a_non_causal_assignment(self):
        # A deliberately broken protocol must not pass silently.
        records = make_records([0, 5])

        def broken_select(subject_records, mode, is_enrollment):
            return (
                [subject_records[1]]
                if is_enrollment
                else [subject_records[0]],
                True,
            )

        import load_dataset

        original = load_dataset.select_record_order_partition
        load_dataset.select_record_order_partition = broken_select

        try:
            report = load_dataset.audit_record_order_causality(
                {"subject_a": records},
                "single-cross-session",
            )
        finally:
            load_dataset.select_record_order_partition = original

        self.assertFalse(report["enrollment_precedes_probe"])
        self.assertEqual(len(report["violations"]), 1)


class PartitionLogTests(unittest.TestCase):
    """
    A live run records its assignment so it can be verified afterwards.
    """

    class FakeLoader:
        def __init__(self):
            self.data_split_mode = "leave-last-out-long-term"

    def test_missing_log_returns_none(self):
        self.assertIsNone(
            summarize_partition_log(self.FakeLoader())
        )

    def test_causal_log_passes(self):
        loader = self.FakeLoader()
        loader.partition_assignment_log = {
            "enrollment": {
                "subject_a": [
                    {
                        "filename": "rec_0.hea",
                        "date": "2024-01-01",
                    }
                ]
            },
            "probe": {
                "subject_a": [
                    {
                        "filename": "rec_1.hea",
                        "date": "2024-03-01",
                    }
                ]
            },
        }

        report = summarize_partition_log(loader)

        self.assertTrue(report["enrollment_precedes_probe"])
        self.assertEqual(report["subjects_audited"], 1)

    def test_shared_recording_is_reported(self):
        loader = self.FakeLoader()
        loader.partition_assignment_log = {
            "enrollment": {
                "subject_a": [
                    {
                        "filename": "rec_0.hea",
                        "date": "2024-01-01",
                    }
                ]
            },
            "probe": {
                "subject_a": [
                    {
                        "filename": "rec_0.hea",
                        "date": "2024-01-01",
                    }
                ]
            },
        }

        report = summarize_partition_log(loader)

        self.assertFalse(report["enrollment_precedes_probe"])
        self.assertIn(
            "share recording",
            report["violations"][0]["reasons"][0],
        )

    def test_future_enrollment_is_reported(self):
        loader = self.FakeLoader()
        loader.partition_assignment_log = {
            "enrollment": {
                "subject_a": [
                    {
                        "filename": "rec_1.hea",
                        "date": "2024-06-01",
                    }
                ]
            },
            "probe": {
                "subject_a": [
                    {
                        "filename": "rec_0.hea",
                        "date": "2024-01-01",
                    }
                ]
            },
        }

        report = summarize_partition_log(loader)

        self.assertFalse(report["enrollment_precedes_probe"])
        self.assertIn(
            "dated after",
            report["violations"][0]["reasons"][0],
        )


if __name__ == "__main__":
    unittest.main()


class SessionDisjointnessValidationTests(unittest.TestCase):
    """
    Cross-session regimes must not probe a session used for enrollment.
    """

    def _parse(self, extra_arguments):
        import contextlib
        import io as string_io

        import main

        base_arguments = [
            "--dataset",
            "heartprint",
            "--data_split_mode",
            "cross-session",
            "--use_template",
        ]

        buffer = string_io.StringIO()

        with contextlib.redirect_stdout(
            buffer
        ), contextlib.redirect_stderr(buffer):
            return main.parse_experiment_arguments(
                base_arguments + extra_arguments
            )

    def test_shared_probe_session_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse(
                [
                    "--task",
                    "6",
                    "--train_sessions",
                    "session1",
                    "--probe_sessions",
                    "session1",
                ]
            )

    def test_shared_enrollment_session_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse(
                [
                    "--task",
                    "6",
                    "--train_sessions",
                    "session1",
                    "--enroll_sessions",
                    "session2",
                    "--probe_sessions",
                    "session2",
                ]
            )

    def test_disjoint_sessions_are_accepted(self):
        args, _ = self._parse(
            [
                "--task",
                "6",
                "--train_sessions",
                "session1",
                "session2",
                "--probe_sessions",
                "session3r",
            ]
        )

        self.assertEqual(
            args.probe_sessions,
            ["session3r"],
        )

    def test_reverse_protocol_remains_valid(self):
        # Enrolling on a later session and probing an earlier one measures
        # directional drift and must not be mistaken for leakage.
        args, _ = self._parse(
            [
                "--task",
                "6",
                "--train_sessions",
                "session2",
                "--probe_sessions",
                "session1",
            ]
        )

        self.assertEqual(
            args.train_sessions,
            ["session2"],
        )

    def test_single_session_tasks_are_unaffected(self):
        args, _ = self._parse(
            [
                "--task",
                "2",
                "--data_split_mode",
                "single-session",
                "--train_sessions",
                "session1",
                "--probe_sessions",
                "session1",
            ]
        )

        self.assertEqual(args.task, 2)
