"""
Provenance-aware probe fusion and first-N enrollment.

Fused probe groups stay within one subject, session/acquisition, physical record
and source segment, are formed in truthful source order at a fixed depth, and
drop the incomplete remainder. Finite first-N enrollment selects the genuinely
first N beats by source order rather than by incidental array order.
"""

import datetime
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from load_dataset import BeatProvenance
from utils import (
    _apply_score_fusion,
    _create_templates,
    _source_order_indices,
)


def prov(records, seg_ids, seg_orders, beat_ords,
         sessions=None, times=None, acq_orders=None, rpeaks=None):
    n = len(records)
    sessions = sessions if sessions is not None else list(records)
    times = times if times is not None else [None] * n
    acq_orders = acq_orders if acq_orders is not None else [0] * n
    rpeaks = rpeaks if rpeaks is not None else [-1] * n
    return BeatProvenance({
        "record_id": np.array(records, dtype=object),
        "session_id": np.array(sessions, dtype=object),
        "acquisition_time": np.array(times, dtype=object),
        "acquisition_order": np.array(acq_orders, dtype=np.int64),
        "source_segment_id": np.array(seg_ids, dtype=object),
        "source_segment_order": np.array(seg_orders, dtype=np.float64),
        "beat_ordinal": np.array(beat_ords, dtype=np.int64),
        "rpeak_index": np.array(rpeaks, dtype=np.int64),
    })


def col(values):
    return np.asarray(values, dtype=float).reshape(-1, 1)


class SourceOrdering(unittest.TestCase):
    def test_chronological_when_all_dated(self):
        # Array order is 1998, 1996, 1997; source order must be by genuine time.
        p = prov(
            ["r", "r", "r"], ["r#0", "r#0", "r#0"], [0.0, 0.0, 0.0], [0, 1, 2],
            times=[datetime.date(1998, 1, 1), datetime.date(1996, 1, 1),
                   datetime.date(1997, 1, 1)],
        )
        self.assertEqual(_source_order_indices(p).tolist(), [1, 2, 0])

    def test_same_date_records_break_ties_on_acquisition_order(self):
        # Two records share a date; array order is [order 2, order 0, order 1].
        # The stable acquisition_order must decide, never the array position.
        p = prov(
            ["c", "a", "b"], ["c#0", "a#0", "b#0"], [0.0, 0.0, 0.0], [0, 0, 0],
            times=[datetime.date(2000, 1, 1)] * 3,
            acq_orders=[2, 0, 1],
        )
        self.assertEqual(_source_order_indices(p).tolist(), [1, 2, 0])

    def test_acquisition_order_when_dates_incomplete(self):
        # One undated record means the whole set uses acquisition_order.
        p = prov(
            ["a", "b", "c"], ["a#0", "b#0", "c#0"], [0.0, 0.0, 0.0], [0, 0, 0],
            times=[datetime.date(1996, 1, 1), None, datetime.date(1997, 1, 1)],
            acq_orders=[2, 0, 1],
        )
        self.assertEqual(_source_order_indices(p).tolist(), [1, 2, 0])


class FusionSizeOne(unittest.TestCase):
    def test_exact_passthrough(self):
        scores = col([1, 2, 3, 4])
        labels = np.array(["a", "a", "b", "b"])
        fused_scores, fused_labels = _apply_score_fusion(scores, labels, 1)
        np.testing.assert_array_equal(fused_scores, scores)
        np.testing.assert_array_equal(fused_labels, labels)

    def test_diagnostics_report_no_drops(self):
        scores = col([1, 2])
        labels = np.array(["a", "a"])
        _, _, diag = _apply_score_fusion(
            scores, labels, 1, return_diagnostics=True
        )
        self.assertEqual(diag["fused_probe_decisions"], 2)
        self.assertEqual(diag["dropped_remainder_observations"], 0)


class FusionRequiresProvenance(unittest.TestCase):
    def test_missing_provenance_is_an_error_for_k_above_one(self):
        with self.assertRaisesRegex(ValueError, "requires per-beat provenance"):
            _apply_score_fusion(col([1, 2, 3, 4]), np.array(["a"] * 4), 2)


class FusionBoundaries(unittest.TestCase):
    def test_groups_never_cross_records(self):
        # One subject, two physical records, two beats each.
        scores = col([10, 11, 20, 21])
        labels = np.array(["s", "s", "s", "s"])
        p = prov(
            ["A", "A", "B", "B"],
            ["A#0", "A#0", "B#0", "B#0"],
            [0.0, 0.0, 0.0, 0.0],
            [0, 1, 0, 1],
        )
        fused, lab = _apply_score_fusion(scores, labels, 2, provenance=p)
        self.assertEqual(len(lab), 2)
        # Each fused score is a within-record mean, never a cross-record mix.
        self.assertEqual(sorted(fused[:, 0].tolist()), [10.5, 20.5])

    def test_groups_never_cross_non_contiguous_segments(self):
        # One subject, one record, two non-contiguous ranges.
        scores = col([1, 2, 100, 200])
        labels = np.array(["s", "s", "s", "s"])
        p = prov(
            ["A", "A", "A", "A"],
            ["A#0", "A#0", "A#1380", "A#1380"],
            [0.0, 0.0, 1380.0, 1380.0],
            [0, 1, 0, 1],
        )
        fused, lab = _apply_score_fusion(scores, labels, 2, provenance=p)
        self.assertEqual(len(lab), 2)
        self.assertEqual(sorted(fused[:, 0].tolist()), [1.5, 150.0])

    def test_groups_never_cross_sessions(self):
        # One subject, identical record and segment ids, two sessions A and B
        # interleaved in the array. Session-bounded fusion averages within each
        # session; crossing sessions would mix them into a different result.
        scores = col([1, 2, 3, 4])
        labels = np.array(["s", "s", "s", "s"])
        p = prov(
            ["r", "r", "r", "r"],
            ["r#0", "r#0", "r#0", "r#0"],
            [0.0, 0.0, 0.0, 0.0],
            [0, 0, 1, 1],
            sessions=["A", "B", "A", "B"],
        )
        fused, lab = _apply_score_fusion(scores, labels, 2, provenance=p)
        self.assertEqual(len(lab), 2)
        # Correct session-bounded means: A={1,3}->2.0, B={2,4}->3.0.
        self.assertEqual(sorted(fused[:, 0].tolist()), [2.0, 3.0])

    def test_source_order_makes_result_independent_of_array_order(self):
        # Same beats and provenance, two array permutations -> same fused result.
        def run(order):
            scores = col([[1, 2, 3, 4][i] for i in order])
            beat_ord = [[0, 1, 2, 3][i] for i in order]
            labels = np.array(["s"] * 4)
            p = prov(["A"] * 4, ["A#0"] * 4, [0.0] * 4, beat_ord)
            return _apply_score_fusion(scores, labels, 2, provenance=p)[0][:, 0].tolist()

        self.assertEqual(run([0, 1, 2, 3]), run([2, 0, 3, 1]))
        self.assertEqual(run([0, 1, 2, 3]), [1.5, 3.5])


class FusionRemainder(unittest.TestCase):
    def test_complete_groups_and_dropped_remainder(self):
        # n = 5, k = 2 -> 2 groups, 1 dropped.
        scores = col([1, 2, 3, 4, 5])
        labels = np.array(["s"] * 5)
        p = prov(["A"] * 5, ["A#0"] * 5, [0.0] * 5, [0, 1, 2, 3, 4])
        fused, lab, diag = _apply_score_fusion(
            scores, labels, 2, provenance=p, return_diagnostics=True
        )
        self.assertEqual(len(lab), 2)
        self.assertEqual(diag["fused_probe_decisions"], 2)
        self.assertEqual(diag["dropped_remainder_observations"], 1)
        self.assertEqual(fused[:, 0].tolist(), [1.5, 3.5])

    def test_block_smaller_than_k_yields_no_group(self):
        # A subject whose only source block has one probe -> zero decisions.
        scores = col([1, 2, 3])
        labels = np.array(["big", "big", "small"])
        p = prov(
            ["A", "A", "B"], ["A#0", "A#0", "B#0"], [0.0, 0.0, 0.0], [0, 1, 0],
        )
        fused, lab, diag = _apply_score_fusion(
            scores, labels, 2, provenance=p, return_diagnostics=True
        )
        self.assertEqual(lab.tolist(), ["big"])
        self.assertEqual(diag["source_blocks_below_fusion_size"], 1)
        self.assertEqual(diag["identities_without_a_fused_decision"], 1)
        self.assertEqual(diag["dropped_remainder_observations"], 1)


class FirstNEnrollment(unittest.TestCase):
    def _embeddings(self, n):
        return np.arange(n * 2, dtype=float).reshape(n, 2)

    def test_first_n_uses_source_order_not_array_order(self):
        # Enrollment beats presented out of source order; the budget of 2 must
        # take the genuinely first two by source order.
        embeddings = np.array([[3.0, 3.0], [1.0, 1.0], [2.0, 2.0]])
        labels = np.array(["s", "s", "s"])
        p = prov(
            ["A", "A", "A"], ["A#0", "A#0", "A#0"], [0.0, 0.0, 0.0], [2, 0, 1],
        )
        templates, tlabels = _create_templates(
            embeddings, labels, method="mean",
            max_enrollment_samples=2, provenance=p
        )
        # Source order is beats with beat_ordinal 0 and 1 -> [1,1] and [2,2].
        np.testing.assert_array_equal(templates[0], np.array([1.5, 1.5]))

    def test_no_fusion_returns_source_first_observations_individually(self):
        embeddings = np.array([[3.0, 3.0], [1.0, 1.0], [2.0, 2.0]])
        labels = np.array(["s", "s", "s"])
        p = prov(
            ["A", "A", "A"],
            ["A#0", "A#0", "A#0"],
            [0.0, 0.0, 0.0],
            [2, 0, 1],
        )

        templates, template_labels = _create_templates(
            embeddings,
            labels,
            method="none",
            max_enrollment_samples=2,
            provenance=p,
        )

        np.testing.assert_array_equal(
            templates,
            np.array([[1.0, 1.0], [2.0, 2.0]]),
        )
        np.testing.assert_array_equal(
            template_labels,
            np.array(["s", "s"]),
        )

    def test_misaligned_provenance_is_rejected_for_a_fusing_method(self):
        embeddings = np.zeros((3, 2))
        labels = np.array(["s", "s", "s"])
        mismatched = prov(["s", "s"], ["s#0", "s#0"], [0.0, 0.0], [0, 1])
        with self.assertRaises(ValueError):
            _create_templates(
                embeddings, labels, method="mean", max_enrollment_samples=1,
                provenance=mismatched,
            )

    def test_default_without_provenance_keeps_array_order(self):
        embeddings = np.array([[1.0, 1.0], [2.0, 2.0], [9.0, 9.0]])
        labels = np.array(["s", "s", "s"])
        templates, _ = _create_templates(
            embeddings, labels, method="mean", max_enrollment_samples=2
        )
        np.testing.assert_array_equal(templates[0], np.array([1.5, 1.5]))


class EnrollmentProbeSplit(unittest.TestCase):
    def _run(self, embeddings, labels, provenance=None, template_size=2):
        from run import _split_enrollment_probe_embeddings

        return _split_enrollment_probe_embeddings(
            np.asarray(embeddings, dtype=float),
            np.asarray(labels),
            np.unique(np.asarray(labels)),
            template_size,
            provenance=provenance,
        )

    def test_provenance_none_preserves_input_order(self):
        embeddings = [[0.0], [1.0], [2.0], [3.0]]
        labels = ["s", "s", "s", "s"]
        result = self._run(embeddings, labels)
        # First two in input order enrol; the rest probe in input order.
        self.assertEqual(result["enrollment"][0][:, 0].tolist(), [0.0, 1.0])
        self.assertEqual(result["probe"][0][:, 0].tolist(), [2.0, 3.0])
        self.assertEqual(result["indices"]["enrollment"].tolist(), [0, 1])
        self.assertEqual(result["indices"]["probe"].tolist(), [2, 3])

    def test_provenance_enrolls_source_first_and_keeps_probe_input_order(self):
        # Array order is scrambled; source order is by beat_ordinal.
        embeddings = [[10.0], [11.0], [12.0], [13.0]]
        labels = ["s", "s", "s", "s"]
        # beat_ordinal: array positions [2, 0, 3, 1] in source terms.
        p = prov(["s"] * 4, ["s#0"] * 4, [0.0] * 4, [2, 0, 3, 1])
        result = self._run(embeddings, labels, provenance=p)
        # Source-first two beats are beat_ordinal 0 and 1 -> array positions 1,3.
        self.assertEqual(
            sorted(result["indices"]["enrollment"].tolist()), [1, 3]
        )
        # Probe is the complement in original input order: positions 0, 2.
        self.assertEqual(result["indices"]["probe"].tolist(), [0, 2])
        # Returned probe embeddings match those indices in that order.
        self.assertEqual(result["probe"][0][:, 0].tolist(), [10.0, 12.0])

    def test_shuffled_input_does_not_change_enrolled_beats(self):
        def enrolled(order):
            emb = [[float(i)] for i in order]
            labels = ["s"] * 4
            beat_ord = [order[i] for i in range(4)]
            p = prov(["s"] * 4, ["s#0"] * 4, [0.0] * 4, beat_ord)
            res = self._run(emb, labels, provenance=p)
            # Identify enrolled beats by their embedding value (== beat_ordinal).
            return sorted(res["enrollment"][0][:, 0].tolist())

        self.assertEqual(enrolled([0, 1, 2, 3]), enrolled([2, 0, 3, 1]))
        self.assertEqual(enrolled([0, 1, 2, 3]), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
