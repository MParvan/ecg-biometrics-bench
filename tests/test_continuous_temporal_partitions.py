import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from load_dataset import (
    audit_continuous_temporal_partitions,
    load_mitbih_dataset,
    load_nsrdb_dataset,
)


class MinuteRangeValidationTests(unittest.TestCase):
    """
    Malformed continuous-recording windows must be rejected explicitly.
    """

    def test_reversed_window_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            audit_continuous_temporal_partitions(
                train_parts=[(5, 0)],
                test_parts=[(25, 30)],
            )

        self.assertIn(
            "start_minute < end_minute",
            str(raised.exception),
        )

    def test_negative_window_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            audit_continuous_temporal_partitions(
                train_parts=[(-1, 5)],
                test_parts=[(25, 30)],
            )

        self.assertIn(
            "cannot start before minute 0",
            str(raised.exception),
        )

    def test_window_needs_exactly_two_boundaries(self):
        with self.assertRaises(ValueError) as raised:
            audit_continuous_temporal_partitions(
                train_parts=[(0, 5, 10)],
                test_parts=[(25, 30)],
            )

        self.assertIn(
            "exactly two",
            str(raised.exception),
        )

    def test_non_numeric_boundary_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            audit_continuous_temporal_partitions(
                train_parts=[("start", 5)],
                test_parts=[(25, 30)],
            )

        self.assertIn(
            "must be numeric",
            str(raised.exception),
        )

    def test_negative_guard_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            audit_continuous_temporal_partitions(
                train_parts=[(0, 5)],
                test_parts=[(25, 30)],
                temporal_guard_minutes=-1.0,
            )

        self.assertIn(
            "cannot be negative",
            str(raised.exception),
        )


class EnrollmentProbeLeakageTests(unittest.TestCase):
    """
    Enrollment and probe windows must never share physical samples.
    """

    def test_train_and_probe_overlap_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            audit_continuous_temporal_partitions(
                train_parts=[(0, 10)],
                test_parts=[(5, 15)],
            )

        message = str(raised.exception)
        self.assertIn("train vs test overlap", message)
        self.assertIn("[5, 10) minutes", message)

    def test_enrol_and_probe_overlap_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            audit_continuous_temporal_partitions(
                train_parts=[(0, 5)],
                enrol_parts=[(20, 30)],
                test_parts=[(25, 35)],
            )

        message = str(raised.exception)
        self.assertIn("enrol vs test overlap", message)
        self.assertIn("[25, 30) minutes", message)

    def test_fully_contained_probe_is_rejected(self):
        with self.assertRaises(ValueError):
            audit_continuous_temporal_partitions(
                train_parts=[(0, 30)],
                test_parts=[(10, 15)],
            )

    def test_identical_train_and_enrol_windows_are_allowed(self):
        # The framework uses the training partition as the gallery partition,
        # so multi-shot protocols routinely set them to the same windows.
        audit = audit_continuous_temporal_partitions(
            train_parts=[(0, 5), (12.5, 17.5)],
            enrol_parts=[(0, 5), (12.5, 17.5)],
            test_parts=[(25, 30)],
        )

        self.assertTrue(audit["overlap_free"])
        self.assertEqual(
            audit["train_enrol_shared_coverage"],
            [[0.0, 5.0], [12.5, 17.5]],
        )

    def test_adjacent_windows_are_allowed_by_default(self):
        audit = audit_continuous_temporal_partitions(
            train_parts=[(0, 5)],
            test_parts=[(5, 10)],
        )

        self.assertEqual(
            audit["achieved_separation_minutes"],
            0.0,
        )


class TemporalGuardBandTests(unittest.TestCase):
    """
    An explicit guard band enforces a minimum enrollment/probe separation.
    """

    def test_guard_band_rejects_insufficient_separation(self):
        with self.assertRaises(ValueError) as raised:
            audit_continuous_temporal_partitions(
                train_parts=[(0, 5)],
                test_parts=[(5, 10)],
                temporal_guard_minutes=2.0,
            )

        message = str(raised.exception)
        self.assertIn("separated by only 0 minutes", message)
        self.assertIn("guard band of 2 minutes", message)

    def test_guard_band_accepts_sufficient_separation(self):
        audit = audit_continuous_temporal_partitions(
            train_parts=[(0, 5)],
            test_parts=[(25, 30)],
            temporal_guard_minutes=10.0,
        )

        self.assertEqual(
            audit["achieved_separation_minutes"],
            20.0,
        )
        self.assertEqual(
            audit["temporal_guard_minutes"],
            10.0,
        )

    def test_separation_uses_the_nearest_enrollment_window(self):
        audit = audit_continuous_temporal_partitions(
            train_parts=[(0, 5), (12.5, 17.5)],
            test_parts=[(25, 30)],
        )

        self.assertEqual(
            audit["achieved_separation_minutes"],
            7.5,
        )


class AuditReportContentTests(unittest.TestCase):
    """
    The audit reports the realized windows so they can be cited directly.
    """

    def test_report_records_coverage_and_durations(self):
        audit = audit_continuous_temporal_partitions(
            train_parts=[(0, 5), (120, 125), (360, 365), (720, 725)],
            enrol_parts=[(0, 5), (120, 125), (360, 365), (720, 725)],
            test_parts=[(1430, 1435)],
        )

        self.assertEqual(
            audit["enrollment_coverage"],
            [
                [0.0, 5.0],
                [120.0, 125.0],
                [360.0, 365.0],
                [720.0, 725.0],
            ],
        )
        self.assertEqual(
            audit["probe_coverage"],
            [[1430.0, 1435.0]],
        )
        self.assertEqual(
            audit["covered_minutes"]["enrollment"],
            20.0,
        )
        self.assertEqual(
            audit["covered_minutes"]["probe"],
            5.0,
        )
        self.assertEqual(
            audit["achieved_separation_minutes"],
            705.0,
        )

    def test_touching_enrollment_windows_are_merged(self):
        audit = audit_continuous_temporal_partitions(
            train_parts=[(0, 5), (5, 10)],
            test_parts=[(25, 30)],
        )

        self.assertEqual(
            audit["enrollment_coverage"],
            [[0.0, 10.0]],
        )


class ContinuousLoaderIntegrationTests(unittest.TestCase):
    """
    Both continuous loaders run the audit while constructing custom splits.
    """

    def test_mitbih_rejects_overlapping_custom_split(self):
        with self.assertRaises(ValueError) as raised:
            load_mitbih_dataset(
                data_split_mode="custom-split",
                train_parts=[(0, 10)],
                test_parts=[(5, 15)],
            )

        self.assertIn(
            "share samples between",
            str(raised.exception),
        )

    def test_nsrdb_rejects_overlapping_custom_split(self):
        with self.assertRaises(ValueError) as raised:
            load_nsrdb_dataset(
                data_split_mode="custom-split",
                train_parts=[(0, 120)],
                enrol_parts=[(0, 120)],
                test_parts=[(60, 180)],
            )

        self.assertIn(
            "share samples between",
            str(raised.exception),
        )

    def test_submitted_mitbih_protocol_is_accepted(self):
        loader = load_mitbih_dataset(
            data_split_mode="custom-split",
            train_parts=[(0, 5), (12.5, 17.5)],
            enrol_parts=[(0, 5), (12.5, 17.5)],
            test_parts=[(25, 30)],
        )

        self.assertIsNotNone(loader.temporal_partition_audit)
        self.assertEqual(
            loader.temporal_partition_audit[
                "achieved_separation_minutes"
            ],
            7.5,
        )

    def test_submitted_nsrdb_protocol_is_accepted(self):
        loader = load_nsrdb_dataset(
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

        self.assertIsNotNone(loader.temporal_partition_audit)
        self.assertEqual(
            loader.temporal_partition_audit[
                "covered_minutes"
            ]["enrollment"],
            20.0,
        )

    def test_non_custom_split_modes_skip_the_audit(self):
        loader = load_mitbih_dataset(
            data_split_mode="single-segment",
            single_segment_range=(0, 5),
        )

        self.assertIsNone(loader.temporal_partition_audit)

    def test_loader_guard_band_is_enforced(self):
        with self.assertRaises(ValueError) as raised:
            load_mitbih_dataset(
                data_split_mode="custom-split",
                train_parts=[(0, 5)],
                test_parts=[(5, 10)],
                temporal_guard_minutes=1.0,
            )

        self.assertIn(
            "guard band",
            str(raised.exception),
        )


if __name__ == "__main__":
    unittest.main()
