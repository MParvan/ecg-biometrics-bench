"""
Pin the treatment of recordings that state no acquisition date.

The record-order regimes read meaning into the order of the dates: which
recording came first, which day is the last one, how far apart the partitions
sit. A recording without a stated date carries none of that evidence, so it
takes no part in those regimes and is not given a stand-in date that would let
it stand as proof of elapsed time. It remains available to "all-available",
which draws on every recording without ordering them.
"""

import datetime
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from load_dataset import (
    RECORD_ORDER_SPLIT_MODES,
    TEMPORAL_DATE_POLICY,
    load_ecgid_dataset,
    select_record_order_partition,
)
from utils import _CACHE_RELEVANT_LOADER_ATTRIBUTES, _build_loader_cache_identity


def record(name, day=None):
    return {
        "filename": name,
        "date": (
            None
            if day is None
            else datetime.datetime(1997, 1, day)
        ),
    }


class UndatedRecordsLeaveTemporalRegimes(unittest.TestCase):
    def test_no_regime_ever_selects_an_undated_record(self):
        # Two recordings on day one and a third on a later day, so that every
        # regime — short-term and long-term alike — finds the structure it
        # needs and the undated recording is the only thing left to reject.
        records = [
            record("a.hea", 1),
            record("b.hea", 1),
            record("c.hea", 3),
            record("undated.hea"),
        ]

        for mode in RECORD_ORDER_SPLIT_MODES:
            for is_enrollment in (True, False):
                selected, eligible = select_record_order_partition(
                    records,
                    mode,
                    is_enrollment,
                )

                self.assertTrue(
                    eligible,
                    f"{mode} should keep a subject with three dated records",
                )

                for chosen in selected:
                    self.assertIsNotNone(
                        chosen["date"],
                        f"{mode} selected a record without a date",
                    )

    def test_undated_record_does_not_become_the_last_day(self):
        # An undated recording sorted last must not be mistaken for the most
        # recent one, which is what a stand-in date placed in the future would
        # have caused.
        records = [
            record("first.hea", 1),
            record("last.hea", 5),
            record("undated.hea"),
        ]

        probe, eligible = select_record_order_partition(
            records,
            "leave-last-out-long-term",
            is_enrollment=False,
        )

        self.assertTrue(eligible)
        self.assertEqual(
            [entry["filename"] for entry in probe],
            ["last.hea"],
        )

    def test_undated_record_does_not_become_the_first_day(self):
        # A stand-in date placed in the past would have anchored day one and
        # taken the enrollment slot.
        records = [
            record("undated.hea"),
            record("first.hea", 1),
            record("second.hea", 2),
        ]

        enrolment, eligible = select_record_order_partition(
            records,
            "single-cross-session",
            is_enrollment=True,
        )

        self.assertTrue(eligible)
        self.assertEqual(
            [entry["filename"] for entry in enrolment],
            ["first.hea"],
        )

    def test_subject_with_only_undated_records_is_not_eligible(self):
        records = [record("one.hea"), record("two.hea")]

        for mode in RECORD_ORDER_SPLIT_MODES:
            selected, eligible = select_record_order_partition(
                records,
                mode,
                is_enrollment=True,
            )

            self.assertFalse(eligible, mode)
            self.assertEqual(selected, [], mode)

    def test_one_dated_record_is_not_enough_for_a_temporal_regime(self):
        records = [record("dated.hea", 1), record("undated.hea")]

        for mode in RECORD_ORDER_SPLIT_MODES:
            _, eligible = select_record_order_partition(
                records,
                mode,
                is_enrollment=True,
            )

            self.assertFalse(
                eligible,
                f"{mode} accepted a subject holding one dated record",
            )


class PolicyReachesTheCacheIdentity(unittest.TestCase):
    def test_attribute_is_cache_relevant(self):
        self.assertIn(
            "temporal_date_policy",
            _CACHE_RELEVANT_LOADER_ATTRIBUTES,
        )

    def test_changing_the_policy_changes_the_identity(self):
        class Loader:
            def __init__(self, policy):
                self.cfg = {"fs": 1000}
                self.prep_params = {}
                self.data_split_mode = "single-cross-session"
                self.temporal_date_policy = policy

        current = _build_loader_cache_identity(Loader(TEMPORAL_DATE_POLICY))
        other = _build_loader_cache_identity(Loader("all_records_ordered"))

        self.assertNotEqual(current, other)

    def test_record_order_loaders_declare_the_policy(self):
        import load_dataset

        for name in (
            "load_ecgid_dataset",
            "load_ptb_dataset",
            "load_ptbxl_dataset",
        ):
            source = Path(load_dataset.__file__).read_text(encoding="utf-8")
            self.assertIn(
                "self.temporal_date_policy = TEMPORAL_DATE_POLICY",
                source,
                name,
            )


class TemporalExcludesButAllAvailableKeepsUndated(unittest.TestCase):
    """The revised protocol semantics: date-ordered regimes use only genuinely
    dated records, while all-available draws on every otherwise-valid record.

    Both sides are exercised on the real selection paths — the shared
    ``select_record_order_partition`` for the temporal regimes and the loader's
    own ``load_all_data`` for all-available — with only I/O and signal
    processing stubbed, so the test pins behaviour rather than source text."""

    def _signal_record(self, name, day=None):
        entry = dict(record(name, day))
        entry["signal"] = np.zeros(8, dtype=float)
        entry["fs"] = 500
        return entry

    def test_temporal_regime_uses_only_dated_records_without_fabricating(self):
        records = [
            self._signal_record("d1.hea", 1),
            self._signal_record("d2.hea", 2),
            self._signal_record("undated.hea"),
        ]

        selected, eligible = select_record_order_partition(
            records, "single-cross-session", is_enrollment=True
        )

        self.assertTrue(eligible)
        names = [entry["filename"] for entry in selected]
        # The dated record is retained; the undated one is excluded because its
        # chronological position cannot be established.
        self.assertIn("d1.hea", names)
        self.assertNotIn("undated.hea", names)
        for entry in selected:
            self.assertIsNotNone(entry["date"])
        # No stand-in date was written onto the undated record.
        self.assertIsNone(records[-1]["date"])

    def test_all_available_keeps_both_dated_and_undated(self):
        loader = load_ecgid_dataset(data_split_mode="all-available")
        subject = [
            self._signal_record("dated.hea", 1),
            self._signal_record("undated.hea"),
        ]
        # Stub only the two I/O boundaries; the all-available selection branch
        # inside load_all_data is the code under test.
        loader.load_raw_data = lambda *a, **k: {"S1": [dict(r) for r in subject]}
        loader._process_signal = lambda sig, fs: np.ones((1, 4), dtype=float)

        x, y = loader.load_all_data()

        # Both records contribute a segment: the undated one is not discarded
        # just because it lacks a date, and no date was fabricated for it.
        self.assertEqual(len(y), 2)
        self.assertIsNone(subject[1]["date"])


if __name__ == "__main__":
    unittest.main()
