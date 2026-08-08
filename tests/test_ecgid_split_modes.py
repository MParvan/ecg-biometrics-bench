"""
Pin the cohort each ECG-ID split mode evaluates.

The reported cohort size is part of every ECG-ID result: Rank-1 chance level is
1/20 under the long-term regimes and 1/89 under the short-term ones, so a
change in these counts changes how a published number should be read. The
record and date structure below is a fixture reproducing the shape of the real
database, which lets the partition rules be checked without the download.
"""

import datetime
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from load_dataset import (
    audit_record_order_causality,
    select_record_order_partition,
)


DAY_1 = datetime.date(2004, 12, 7)
DAY_2 = datetime.date(2004, 12, 19)
DAY_3 = datetime.date(2005, 5, 12)


def records(*specification):
    """
    Build one subject's recordings from ``(date, record number)`` pairs.
    """
    return [
        {
            "date": date,
            "filename": f"rec_{number}.hea",
            "fs": 500,
        }
        for date, number in specification
    ]


SINGLE_RECORD = records((DAY_3, 1))
TWO_SAME_DAY = records((DAY_1, 1), (DAY_1, 2))
FOUR_SAME_DAY = records(
    (DAY_1, 1), (DAY_1, 2), (DAY_1, 3), (DAY_1, 4)
)
ACROSS_THREE_DAYS = records(
    (DAY_1, 1), (DAY_1, 2), (DAY_2, 3), (DAY_3, 4), (DAY_3, 5)
)


class ECGIDPartitionEligibilityTests(unittest.TestCase):
    def assert_eligibility(self, mode, subject, expected):
        for is_enrollment in (True, False):
            _, eligible = select_record_order_partition(
                subject,
                mode,
                is_enrollment,
            )
            self.assertEqual(
                eligible,
                expected,
                f"{mode} enrollment={is_enrollment}",
            )

    def test_a_single_record_is_eligible_for_no_regime(self):
        for mode in (
            "single-cross-session",
            "single-shot-short-term",
            "leave-last-out-short-term",
            "single-shot-long-term",
            "leave-last-out-long-term",
        ):
            self.assert_eligibility(mode, SINGLE_RECORD, False)

    def test_one_day_only_is_eligible_for_short_term_regimes_alone(self):
        for mode in (
            "single-cross-session",
            "single-shot-short-term",
            "leave-last-out-short-term",
        ):
            self.assert_eligibility(mode, TWO_SAME_DAY, True)

        for mode in (
            "single-shot-long-term",
            "leave-last-out-long-term",
        ):
            self.assert_eligibility(mode, TWO_SAME_DAY, False)

    def test_multiple_days_are_eligible_everywhere(self):
        for mode in (
            "single-cross-session",
            "single-shot-short-term",
            "leave-last-out-short-term",
            "single-shot-long-term",
            "leave-last-out-long-term",
        ):
            self.assert_eligibility(mode, ACROSS_THREE_DAYS, True)


class ECGIDPartitionContentTests(unittest.TestCase):
    def partitions(self, mode, subject):
        enrol, _ = select_record_order_partition(subject, mode, True)
        probe, _ = select_record_order_partition(subject, mode, False)
        return (
            [record["filename"] for record in enrol],
            [record["filename"] for record in probe],
        )

    def test_single_shot_short_term_probes_the_rest_of_day_one(self):
        enrol, probe = self.partitions(
            "single-shot-short-term",
            FOUR_SAME_DAY,
        )

        self.assertEqual(enrol, ["rec_1.hea"])
        self.assertEqual(
            probe,
            ["rec_2.hea", "rec_3.hea", "rec_4.hea"],
        )

    def test_leave_last_out_short_term_probes_only_the_last_of_day_one(self):
        enrol, probe = self.partitions(
            "leave-last-out-short-term",
            FOUR_SAME_DAY,
        )

        self.assertEqual(
            enrol,
            ["rec_1.hea", "rec_2.hea", "rec_3.hea"],
        )
        self.assertEqual(probe, ["rec_4.hea"])

    def test_short_term_regimes_never_leave_day_one(self):
        """
        A short-term regime that reached a later day would silently become a
        long-term one, so both partitions must stay inside the first day.
        """
        for mode in (
            "single-shot-short-term",
            "leave-last-out-short-term",
        ):
            for is_enrollment in (True, False):
                selected, _ = select_record_order_partition(
                    ACROSS_THREE_DAYS,
                    mode,
                    is_enrollment,
                )
                self.assertTrue(selected, mode)
                self.assertEqual(
                    {record["date"] for record in selected},
                    {DAY_1},
                    f"{mode} enrollment={is_enrollment}",
                )

    def test_single_shot_long_term_enrols_day_one_and_probes_later_days(self):
        enrol, probe = self.partitions(
            "single-shot-long-term",
            ACROSS_THREE_DAYS,
        )

        self.assertEqual(enrol, ["rec_1.hea", "rec_2.hea"])
        self.assertEqual(
            probe,
            ["rec_3.hea", "rec_4.hea", "rec_5.hea"],
        )

    def test_leave_last_out_long_term_probes_the_final_day(self):
        enrol, probe = self.partitions(
            "leave-last-out-long-term",
            ACROSS_THREE_DAYS,
        )

        self.assertEqual(
            enrol,
            ["rec_1.hea", "rec_2.hea", "rec_3.hea"],
        )
        self.assertEqual(probe, ["rec_4.hea", "rec_5.hea"])

    def test_every_regime_keeps_the_partitions_disjoint(self):
        for mode in (
            "single-cross-session",
            "single-shot-short-term",
            "leave-last-out-short-term",
            "single-shot-long-term",
            "leave-last-out-long-term",
        ):
            enrol, probe = self.partitions(mode, ACROSS_THREE_DAYS)

            self.assertFalse(
                set(enrol) & set(probe),
                f"{mode} reuses a recording on both sides",
            )

    def test_long_term_regimes_separate_the_partitions_in_time(self):
        """
        Distinguishes a genuine across-day comparison from one that merely
        splits records: no enrollment recording may share a day with a probe.
        """
        for mode in (
            "single-shot-long-term",
            "leave-last-out-long-term",
        ):
            enrol, _ = select_record_order_partition(
                ACROSS_THREE_DAYS,
                mode,
                True,
            )
            probe, _ = select_record_order_partition(
                ACROSS_THREE_DAYS,
                mode,
                False,
            )

            self.assertFalse(
                {record["date"] for record in enrol}
                & {record["date"] for record in probe},
                f"{mode} places enrollment and probe on the same day",
            )

    def test_single_cross_session_compares_the_first_two_records(self):
        """
        The regime pairs consecutive recordings. On ECG-ID those fall on one
        day for every eligible subject, so it compares records within a day.
        """
        enrol, probe = self.partitions(
            "single-cross-session",
            ACROSS_THREE_DAYS,
        )

        self.assertEqual(enrol, ["rec_1.hea"])
        self.assertEqual(probe, ["rec_2.hea"])

        self.assertEqual(
            ACROSS_THREE_DAYS[0]["date"],
            ACROSS_THREE_DAYS[1]["date"],
        )


if __name__ == "__main__":
    unittest.main()


class TemporalSeparationReportingTests(unittest.TestCase):
    """
    Check the gap the audit reports between enrollment and probe recordings.
    """

    def audit(self, mode, subjects):
        return audit_record_order_causality(subjects, mode)

    def test_a_same_day_regime_reports_no_day_separation(self):
        report = self.audit(
            "single-cross-session",
            {"01": ACROSS_THREE_DAYS, "02": TWO_SAME_DAY},
        )

        separation = report["temporal_separation"]

        self.assertEqual(separation["subjects_measured"], 2)
        self.assertEqual(separation["subjects_same_day"], 2)
        self.assertEqual(separation["subjects_different_day"], 0)
        self.assertEqual(separation["max_days"], 0)
        self.assertFalse(separation["separates_days"])

    def test_a_long_term_regime_reports_a_measured_gap(self):
        report = self.audit(
            "single-shot-long-term",
            {"01": ACROSS_THREE_DAYS},
        )

        separation = report["temporal_separation"]

        self.assertEqual(separation["subjects_measured"], 1)
        self.assertEqual(separation["subjects_same_day"], 0)
        self.assertTrue(separation["separates_days"])
        self.assertEqual(
            separation["min_days"],
            (DAY_2 - DAY_1).days,
        )

    def test_short_term_regimes_never_report_a_gap(self):
        for mode in (
            "single-shot-short-term",
            "leave-last-out-short-term",
        ):
            report = self.audit(mode, {"01": ACROSS_THREE_DAYS})

            self.assertEqual(
                report["temporal_separation"]["max_days"],
                0,
                mode,
            )

    def test_missing_dates_are_reported_as_unknown_not_as_zero(self):
        """
        A dataset without acquisition dates must report an unknown gap rather
        than a zero one.
        """
        undated = [
            {
                "date": datetime.date.min,
                "filename": f"rec_{number}.hea",
                "fs": 500,
            }
            for number in (1, 2)
        ]

        report = self.audit("single-cross-session", {"01": undated})

        self.assertIsNone(report["temporal_separation"])

    def test_the_separation_reaches_each_subject_report(self):
        report = self.audit(
            "leave-last-out-long-term",
            {"01": ACROSS_THREE_DAYS},
        )

        self.assertEqual(
            report["subject_reports"][0]["separation_days"],
            (DAY_3 - DAY_2).days,
        )
