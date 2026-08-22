"""
Task-1 (closed-set identification) provenance threading.

Provenance stays aligned with X/y through the randomized train/test split and
the quality filters, first-N enrollment follows source order, and probe fusion
is source-bounded and fixed-depth. The randomized partition itself remains
controlled only by the split seed.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run
from load_dataset import BeatProvenance
from utils import _apply_outlier_filter
from test_all_task_smoke import TinyECGModel, make_synthetic_ecg_dataset


def provenance_for(labels):
    """
    Build provenance index-aligned to the supplied label order.

    Each row corresponds to the label at the same array index. One synthetic
    record/source segment per subject; ``beat_ordinal`` increases in the order a
    subject appears in the array. Labels/samples are never reordered.
    """
    labels = np.asarray(labels)
    count = len(labels)
    per_subject_counter = {}
    beat_ordinal = np.empty(count, dtype=np.int64)
    for index, label in enumerate(labels.tolist()):
        beat_ordinal[index] = per_subject_counter.get(label, 0)
        per_subject_counter[label] = per_subject_counter.get(label, 0) + 1
    return BeatProvenance({
        "record_id": np.array([f"{label}_r" for label in labels], dtype=object),
        "session_id": np.array([f"{label}_r" for label in labels], dtype=object),
        "acquisition_time": np.array([None] * count, dtype=object),
        "acquisition_order": np.zeros(count, dtype=np.int64),
        "source_segment_id": np.array(
            [f"{label}_r#0" for label in labels], dtype=object
        ),
        "source_segment_order": np.zeros(count, dtype=np.float64),
        "beat_ordinal": beat_ordinal,
        "rpeak_index": np.full(count, -1, dtype=np.int64),
    })


COMMON = {
    "model_class": TinyECGModel,
    "epochs": 1,
    "batch_size": 16,
    "lr": 1e-3,
    "val_split": 0.0,
    "seed": 42,
    "device": "cpu",
    "visualize": False,
    "save_results_and_settings": False,
    "loader": None,
    "n_runs": 1,
    "_return_stats": True,
    "intelligent_weight_loading": False,
}


class ProvenanceHelperAlignment(unittest.TestCase):
    def test_rows_align_with_interleaved_label_order(self):
        labels = np.array(["a", "b", "a", "b"])
        provenance = provenance_for(labels)
        # Each row carries the record of the label at the same index.
        self.assertEqual(
            provenance.columns["record_id"].tolist(),
            ["a_r", "b_r", "a_r", "b_r"],
        )
        # beat_ordinal increases per subject in appearance order.
        self.assertEqual(provenance.columns["beat_ordinal"].tolist(), [0, 0, 1, 1])
        provenance.validate(len(labels))


class Task1Provenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = make_synthetic_ecg_dataset(
            number_of_subjects=6, samples_per_subject=12
        )
        cls.provenance = provenance_for(cls.y)

    def test_fusion_size_two_works_end_to_end_with_provenance(self):
        (rank1, rank5), data_stats, _ = run.run_closed_set_identification(
            self.x, self.y,
            test_split=0.5,
            use_template=False,
            probe_fusion_size=2,
            provenance=self.provenance,
            **COMMON,
        )
        self.assertTrue(0.0 <= rank1 <= 1.0)
        diagnostics = data_stats["Probe Fusion"]
        self.assertEqual(diagnostics["fusion_size"], 2)
        self.assertGreater(diagnostics["fused_probe_decisions"], 0)
        # Every raw probe observation is accounted for.
        self.assertEqual(
            diagnostics["raw_probe_observations"],
            2 * diagnostics["fused_probe_decisions"]
            + diagnostics["dropped_remainder_observations"],
        )

    def test_fusion_size_two_without_provenance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires per-beat provenance"):
            run.run_closed_set_identification(
                self.x, self.y,
                test_split=0.5,
                use_template=False,
                probe_fusion_size=2,
                provenance=None,
                **COMMON,
            )

    def test_fusion_size_one_needs_no_provenance(self):
        (rank1, _), data_stats, _ = run.run_closed_set_identification(
            self.x, self.y,
            test_split=0.5,
            use_template=False,
            probe_fusion_size=1,
            provenance=None,
            **COMMON,
        )
        self.assertTrue(0.0 <= rank1 <= 1.0)
        self.assertEqual(data_stats["Probe Fusion"]["fusion_size"], 1)
        self.assertEqual(
            data_stats["Probe Fusion"]["dropped_remainder_observations"], 0
        )


class Task3Provenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = make_synthetic_ecg_dataset(
            number_of_subjects=8, samples_per_subject=12
        )
        cls.provenance = provenance_for(cls.y)

    def _run(self, probe_fusion_size, provenance):
        return run.run_subject_disjoint_identification(
            self.x, self.y,
            test_split=0.4,
            use_template=True,
            template_fusion_method="mean",
            template_size=2,
            probe_fusion_size=probe_fusion_size,
            provenance=provenance,
            **COMMON,
        )

    def test_fusion_two_end_to_end(self):
        (rank1, _), data_stats, _ = self._run(2, self.provenance)
        self.assertTrue(0.0 <= rank1 <= 1.0)
        self.assertEqual(data_stats["Probe Fusion"]["fusion_size"], 2)
        self.assertGreater(data_stats["Probe Fusion"]["fused_probe_decisions"], 0)

    def test_fusion_two_requires_provenance(self):
        with self.assertRaisesRegex(ValueError, "requires per-beat provenance"):
            self._run(2, None)

    def test_fusion_one_ordinary(self):
        (_, _), data_stats, _ = self._run(1, self.provenance)
        self.assertEqual(data_stats["Probe Fusion"]["fusion_size"], 1)
        self.assertEqual(
            data_stats["Probe Fusion"]["dropped_remainder_observations"], 0
        )


class CrossSessionProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s1_x, cls.s1_y = make_synthetic_ecg_dataset(
            number_of_subjects=8, samples_per_subject=10, seed=456
        )
        cls.s2_x, cls.s2_y = make_synthetic_ecg_dataset(
            number_of_subjects=8, samples_per_subject=8, session_shift=0.1, seed=789
        )
        cls.p1 = provenance_for(cls.s1_y)
        cls.p2 = provenance_for(cls.s2_y)

    def test_task5_fusion_two_end_to_end(self):
        (rank1, _), data_stats, _ = run.run_cross_session_identification(
            self.s1_x, self.s1_y, self.s2_x, self.s2_y,
            use_template=True, template_fusion_method="mean", template_size=2,
            probe_fusion_size=2,
            provenance_s1=self.p1, provenance_s2=self.p2,
            **COMMON,
        )
        self.assertTrue(0.0 <= rank1 <= 1.0)
        self.assertEqual(data_stats["Probe Fusion"]["fusion_size"], 2)
        self.assertGreater(data_stats["Probe Fusion"]["fused_probe_decisions"], 0)

    def test_task5_fusion_two_requires_provenance(self):
        with self.assertRaisesRegex(ValueError, "requires per-beat provenance"):
            run.run_cross_session_identification(
                self.s1_x, self.s1_y, self.s2_x, self.s2_y,
                use_template=False, probe_fusion_size=2,
                provenance_s1=None, provenance_s2=None,
                **COMMON,
            )

    def test_task7_fusion_two_end_to_end(self):
        (rank1, _), data_stats, _ = (
            run.run_subject_disjoint_cross_session_identification(
                self.s1_x, self.s1_y, self.s2_x, self.s2_y,
                test_split=0.4, use_template=True,
                template_fusion_method="mean", template_size=2,
                probe_fusion_size=2,
                provenance_s1=self.p1, provenance_s2=self.p2,
                **COMMON,
            )
        )
        self.assertTrue(0.0 <= rank1 <= 1.0)
        self.assertEqual(data_stats["Probe Fusion"]["fusion_size"], 2)
        self.assertGreater(data_stats["Probe Fusion"]["fused_probe_decisions"], 0)

    def test_task7_fusion_two_requires_provenance(self):
        with self.assertRaisesRegex(ValueError, "requires per-beat provenance"):
            run.run_subject_disjoint_cross_session_identification(
                self.s1_x, self.s1_y, self.s2_x, self.s2_y,
                test_split=0.4, use_template=True,
                template_fusion_method="mean", template_size=2,
                probe_fusion_size=2,
                provenance_s1=None, provenance_s2=None,
                **COMMON,
            )

    def test_task7_fusion_one_ordinary(self):
        (_, _), data_stats, _ = (
            run.run_subject_disjoint_cross_session_identification(
                self.s1_x, self.s1_y, self.s2_x, self.s2_y,
                test_split=0.4, use_template=True,
                template_fusion_method="mean", template_size=2,
                probe_fusion_size=1,
                provenance_s1=self.p1, provenance_s2=self.p2,
                **COMMON,
            )
        )
        self.assertEqual(data_stats["Probe Fusion"]["fusion_size"], 1)


class OutlierFilterIndices(unittest.TestCase):
    def test_return_indices_reproduces_filtered_arrays(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(20, 8)).astype(np.float32)
        y = np.array([f"s{i // 5}" for i in range(20)])
        sqi = rng.uniform(size=20)

        filtered_x, filtered_y = _apply_outlier_filter(
            x, y, sqi, absolute_threshold=0.3, keep_percentage=0.6
        )
        again_x, again_y, indices = _apply_outlier_filter(
            x, y, sqi, absolute_threshold=0.3, keep_percentage=0.6,
            return_indices=True,
        )
        np.testing.assert_array_equal(filtered_x, again_x)
        np.testing.assert_array_equal(filtered_y, again_y)
        # The returned indices reproduce exactly the filtered arrays.
        np.testing.assert_array_equal(x[indices], filtered_x)
        np.testing.assert_array_equal(y[indices], filtered_y)


if __name__ == "__main__":
    unittest.main()
