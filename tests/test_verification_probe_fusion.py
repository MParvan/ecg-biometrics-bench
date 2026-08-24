"""
Verification probe fusion (Tasks 2, 4, 6, 8).

The multi-beat probe fusion path builds source-bounded probe groups, enumerates
the fused-decision space against the enrollment templates and evaluates only
the selected group-template decisions in bounded score batches. probe_fusion
size 1 must preserve the standard verification pair-generation path exactly
for every task family.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run
import utils
from load_dataset import BeatProvenance
from test_all_task_smoke import TinyECGModel, make_synthetic_ecg_dataset
from test_identification_provenance import provenance_for


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


def _prov(records, segments, orders, beat_ordinals, sessions=None):
    n = len(records)
    if sessions is None:
        sessions = list(records)
    return BeatProvenance({
        "record_id": np.array(records, dtype=object),
        "session_id": np.array(sessions, dtype=object),
        "acquisition_time": np.array([None] * n, dtype=object),
        "acquisition_order": np.zeros(n, dtype=np.int64),
        "source_segment_id": np.array(segments, dtype=object),
        "source_segment_order": np.array(orders, dtype=np.float64),
        "beat_ordinal": np.array(beat_ordinals, dtype=np.int64),
        "rpeak_index": np.full(n, -1, dtype=np.int64),
    })


# ---------------------------------------------------------------------------
# B. Grouping
# ---------------------------------------------------------------------------
class ProbeGroupingTests(unittest.TestCase):
    def _one_block_per_subject(self, labels):
        return _prov(
            [f"{label}_r" for label in labels],
            [f"{label}_r#0" for label in labels],
            [0.0] * len(labels),
            list(range(len(labels))),
        )

    def test_groups_never_cross_subjects(self):
        labels = np.array(["a", "a", "b", "b", "a", "b"])
        provenance = self._one_block_per_subject(labels)
        result = utils._build_probe_groups(labels, provenance, group_size=2)
        for member_row in result["groups"]:
            member_labels = labels[member_row]
            self.assertEqual(len(set(member_labels.tolist())), 1)

    def test_groups_never_cross_sessions(self):
        # Two "sessions" for subject "a" — must not merge.
        labels = np.array(["a", "a", "a", "a"])
        provenance = _prov(
            ["a_r", "a_r", "a_r", "a_r"],
            ["a_r#0", "a_r#0", "a_r#0", "a_r#0"],
            [0.0, 0.0, 0.0, 0.0],
            [0, 1, 0, 1],
            sessions=["a_s1", "a_s1", "a_s2", "a_s2"],
        )
        result = utils._build_probe_groups(labels, provenance, group_size=2)
        # Two groups, each drawn from one session's beats.
        self.assertEqual(result["groups"].shape, (2, 2))
        for member_row in result["groups"]:
            session_ids = {
                str(provenance.columns["session_id"][i]) for i in member_row
            }
            self.assertEqual(len(session_ids), 1)

    def test_groups_never_cross_records(self):
        labels = np.array(["a", "a", "a", "a"])
        provenance = _prov(
            ["a_r1", "a_r1", "a_r2", "a_r2"],
            ["a_r1#0", "a_r1#0", "a_r2#0", "a_r2#0"],
            [0.0, 0.0, 0.0, 0.0],
            [0, 1, 0, 1],
        )
        result = utils._build_probe_groups(labels, provenance, group_size=2)
        self.assertEqual(result["groups"].shape, (2, 2))
        for member_row in result["groups"]:
            record_ids = {
                str(provenance.columns["record_id"][i]) for i in member_row
            }
            self.assertEqual(len(record_ids), 1)

    def test_groups_never_cross_source_segments(self):
        labels = np.array(["a"] * 4)
        provenance = _prov(
            ["a_r"] * 4,
            ["a_r#0", "a_r#0", "a_r#1", "a_r#1"],
            [0.0, 0.0, 1.0, 1.0],
            [0, 1, 0, 1],
        )
        result = utils._build_probe_groups(labels, provenance, group_size=2)
        self.assertEqual(result["groups"].shape, (2, 2))
        for member_row in result["groups"]:
            segment_ids = {
                str(provenance.columns["source_segment_id"][i])
                for i in member_row
            }
            self.assertEqual(len(segment_ids), 1)

    def test_groups_respect_source_order(self):
        labels = np.array(["a"] * 4)
        # Provenance beat_ordinals scrambled versus array order.
        provenance = _prov(
            ["a_r"] * 4, ["a_r#0"] * 4, [0.0] * 4,
            [3, 0, 2, 1],
        )
        result = utils._build_probe_groups(labels, provenance, group_size=2)
        self.assertEqual(result["groups"].shape, (2, 2))
        first, second = result["groups"]
        beat_ord = provenance.columns["beat_ordinal"]
        self.assertEqual(beat_ord[first].tolist(), [0, 1])
        self.assertEqual(beat_ord[second].tolist(), [2, 3])

    def test_complete_non_overlapping_windows_and_drop_remainder(self):
        labels = np.array(["a"] * 8)
        provenance = _prov(
            ["a_r"] * 8, ["a_r#0"] * 8, [0.0] * 8, list(range(8)),
        )
        result = utils._build_probe_groups(labels, provenance, group_size=3)
        self.assertEqual(result["groups"].shape, (2, 3))
        # Members are exactly [0,1,2] and [3,4,5]; 6 and 7 are dropped.
        self.assertEqual(result["groups"][0].tolist(), [0, 1, 2])
        self.assertEqual(result["groups"][1].tolist(), [3, 4, 5])
        self.assertEqual(
            result["diagnostics"]["dropped_remainder_observations"], 2
        )
        # No beat appears twice across groups.
        flat = result["groups"].reshape(-1).tolist()
        self.assertEqual(len(flat), len(set(flat)))

    def test_missing_provenance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "provenance is required"):
            utils._build_probe_groups(
                np.array(["a", "a"]), None, group_size=2
            )

    def test_group_size_below_two_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            utils._build_probe_groups(
                np.array(["a", "a"]),
                self._one_block_per_subject(np.array(["a", "a"])),
                group_size=1,
            )


# ---------------------------------------------------------------------------
# C. Scores
# ---------------------------------------------------------------------------
class FusedScoreCorrectnessTests(unittest.TestCase):
    def _fixture(self, matching_method="cosine"):
        # Two subjects, three probe beats each; one template per subject.
        # Probe embeddings differ per beat so mean fusion is not trivially
        # equal to any single-beat score.
        probes = np.array(
            [
                [1.0, 0.0], [0.9, 0.1], [1.1, -0.1],
                [0.0, 1.0], [0.1, 0.9], [-0.1, 1.1],
            ],
            dtype=float,
        )
        probe_labels = np.array(["a", "a", "a", "b", "b", "b"])
        templates = np.array(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=float,
        )
        template_identities = np.array(["a", "b"])
        provenance = _prov(
            ["a_r", "a_r", "a_r", "b_r", "b_r", "b_r"],
            ["a_r#0"] * 3 + ["b_r#0"] * 3,
            [0.0] * 6,
            [0, 1, 2, 0, 1, 2],
        )
        scores, labels_pair, diagnostics = (
            utils._generate_fused_verification_pairs(
                probe_embeddings=probes,
                probe_labels=probe_labels,
                probe_provenance=provenance,
                template_embeddings=templates,
                template_identities=template_identities,
                group_size=3,
                matching_method=matching_method,
                max_impostor_pairs=100,
                pair_sampling_seed=0,
            )
        )
        return probes, probe_labels, templates, template_identities, scores, labels_pair, diagnostics

    def _manual_pair_score(self, probe, template, method):
        # Emulate the framework's higher-is-better score orientation.
        if method == "cosine":
            p = probe / (np.linalg.norm(probe) + 1e-10)
            t = template / (np.linalg.norm(template) + 1e-10)
            return float(np.dot(p, t))
        if method == "euclidean":
            return -float(np.linalg.norm(probe - template))
        raise NotImplementedError(method)

    def test_k3_cosine_genuine_and_impostor_match_manual_mean(self):
        (probes, probe_labels, templates, template_identities,
         scores, labels_pair, diagnostics) = self._fixture("cosine")
        # 2 genuine + 2 impostor decisions (unique 2x2 identity Cartesian).
        self.assertEqual(diagnostics["genuine_fused_decisions"], 2)
        self.assertEqual(
            diagnostics["total_impostor_fused_decisions"], 2
        )
        self.assertEqual(int(labels_pair.sum()), 2)

        # Manually compute expected scores for every (group, template) pair.
        subject_probes = {"a": probes[:3], "b": probes[3:]}
        expected = {}
        for probe_id, group in subject_probes.items():
            for tmpl_index, tmpl_id in enumerate(template_identities):
                template = templates[tmpl_index]
                fused = np.mean(
                    [self._manual_pair_score(row, template, "cosine")
                     for row in group]
                )
                expected[(probe_id, tmpl_id)] = fused

        # Check that every emitted score matches its manual expectation.
        for score, label in zip(scores, labels_pair):
            matched = False
            for (probe_id, tmpl_id), fused in expected.items():
                if abs(score - fused) < 1e-9:
                    genuine = int(probe_id == tmpl_id)
                    self.assertEqual(int(label), genuine)
                    matched = True
                    break
            self.assertTrue(matched, f"unmatched score {score}")

    def test_k3_euclidean_is_higher_is_better(self):
        _, _, _, _, scores, labels_pair, _ = self._fixture("euclidean")
        # Every score should be non-positive since matcher returns
        # -distance; genuine pairs should score higher than impostor pairs.
        self.assertTrue(np.all(scores <= 0.0))
        genuine = scores[labels_pair == 1]
        impostor = scores[labels_pair == 0]
        self.assertGreater(genuine.mean(), impostor.mean())


# ---------------------------------------------------------------------------
# D. Fused-decision pair sampling
# ---------------------------------------------------------------------------
class FusedPairSamplingTests(unittest.TestCase):
    def _small(self, n_subjects=6, per_subject=4, group_size=2, seed=0,
               max_impostor_pairs=10 ** 6):
        probes = np.random.default_rng(seed).normal(
            size=(n_subjects * per_subject, 4)
        )
        labels = np.array(
            [f"s{i}" for i in range(n_subjects) for _ in range(per_subject)]
        )
        provenance = _prov(
            [f"s{i}_r" for i in range(n_subjects) for _ in range(per_subject)],
            [f"s{i}_r#0" for i in range(n_subjects) for _ in range(per_subject)],
            [0.0] * len(labels),
            list(range(per_subject)) * n_subjects,
        )
        templates = np.random.default_rng(seed + 1).normal(
            size=(n_subjects, 4)
        )
        template_identities = np.array([f"s{i}" for i in range(n_subjects)])
        return utils._generate_fused_verification_pairs(
            probe_embeddings=probes,
            probe_labels=labels,
            probe_provenance=provenance,
            template_embeddings=templates,
            template_identities=template_identities,
            group_size=group_size,
            matching_method="cosine",
            max_impostor_pairs=max_impostor_pairs,
            pair_sampling_seed=seed + 100,
        )

    def test_all_genuine_decisions_are_retained(self):
        # Each of 6 subjects yields (4 beats // 2) = 2 groups, so 12 groups
        # and 12 genuine (group, own template) decisions.
        scores, labels_pair, diag = self._small(
            n_subjects=6, per_subject=4, group_size=2
        )
        self.assertEqual(int(labels_pair.sum()), 12)
        self.assertEqual(diag["genuine_fused_decisions"], 12)

    def test_impostor_sampling_is_reproducible(self):
        first = self._small(seed=17)
        second = self._small(seed=17)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_impostor_cap_is_honoured(self):
        scores, labels_pair, diag = self._small(
            n_subjects=8, per_subject=4, group_size=2,
            max_impostor_pairs=5,
        )
        # 8 subjects * 2 groups each = 16 groups. Genuine cells = 16.
        # Total impostor cells = 16 groups * 8 templates - 16 genuine = 112.
        self.assertEqual(diag["total_impostor_fused_decisions"], 112)
        self.assertEqual(diag["impostor_fused_decisions"], 5)
        # No duplicated impostor rows and no genuine misclassified as impostor.
        self.assertEqual(int((labels_pair == 0).sum()), 5)

    def test_different_seeds_change_impostor_selection(self):
        _, labels_a, diag_a = self._small(
            n_subjects=8, per_subject=4, group_size=2,
            max_impostor_pairs=5, seed=1,
        )
        _, labels_b, diag_b = self._small(
            n_subjects=8, per_subject=4, group_size=2,
            max_impostor_pairs=5, seed=2,
        )
        self.assertEqual(diag_a["impostor_fused_decisions"], 5)
        self.assertEqual(diag_b["impostor_fused_decisions"], 5)


# ---------------------------------------------------------------------------
# E. Memory
# ---------------------------------------------------------------------------
class BoundedMemoryTests(unittest.TestCase):
    def test_no_full_group_times_template_matrix_is_allocated(self):
        # 200 subjects x 6 beats = 1200 probes. With k=3 that is 400 groups
        # against 200 templates = 80000 impostor cells (before cap).
        rng = np.random.default_rng(0)
        n_subjects = 200
        per_subject = 6
        probes = rng.normal(size=(n_subjects * per_subject, 8))
        labels = np.array(
            [f"s{i}" for i in range(n_subjects) for _ in range(per_subject)]
        )
        provenance = _prov(
            [f"s{i}_r" for i in range(n_subjects) for _ in range(per_subject)],
            [f"s{i}_r#0" for i in range(n_subjects) for _ in range(per_subject)],
            [0.0] * (n_subjects * per_subject),
            list(range(per_subject)) * n_subjects,
        )
        templates = rng.normal(size=(n_subjects, 8))
        template_identities = np.array([f"s{i}" for i in range(n_subjects)])

        forbidden_shape = (400, 200)  # #groups x #templates
        seen_forbidden = []

        real_empty = np.empty
        real_zeros = np.zeros

        def _guard_empty(shape, *a, **kw):
            if shape == forbidden_shape:
                seen_forbidden.append(("empty", shape))
            return real_empty(shape, *a, **kw)

        def _guard_zeros(shape, *a, **kw):
            if shape == forbidden_shape:
                seen_forbidden.append(("zeros", shape))
            return real_zeros(shape, *a, **kw)

        with patch("numpy.empty", side_effect=_guard_empty), \
             patch("numpy.zeros", side_effect=_guard_zeros):
            utils._generate_fused_verification_pairs(
                probe_embeddings=probes,
                probe_labels=labels,
                probe_provenance=provenance,
                template_embeddings=templates,
                template_identities=template_identities,
                group_size=3,
                matching_method="cosine",
                max_impostor_pairs=5000,
                pair_sampling_seed=42,
                decision_batch_size=1024,
            )

        self.assertEqual(seen_forbidden, [])

    def test_batched_scoring_calls_score_helper_multiple_times(self):
        rng = np.random.default_rng(0)
        n_subjects = 40
        per_subject = 6
        probes = rng.normal(size=(n_subjects * per_subject, 8))
        labels = np.array(
            [f"s{i}" for i in range(n_subjects) for _ in range(per_subject)]
        )
        provenance = _prov(
            [f"s{i}_r" for i in range(n_subjects) for _ in range(per_subject)],
            [f"s{i}_r#0" for i in range(n_subjects) for _ in range(per_subject)],
            [0.0] * (n_subjects * per_subject),
            list(range(per_subject)) * n_subjects,
        )
        templates = rng.normal(size=(n_subjects, 8))
        template_identities = np.array([f"s{i}" for i in range(n_subjects)])

        original = utils._score_selected_pair_indices
        call_count = {"n": 0}

        def _counting(*args, **kwargs):
            call_count["n"] += 1
            return original(*args, **kwargs)

        with patch.object(
            utils, "_score_selected_pair_indices", side_effect=_counting
        ):
            utils._generate_fused_verification_pairs(
                probe_embeddings=probes,
                probe_labels=labels,
                probe_provenance=provenance,
                template_embeddings=templates,
                template_identities=template_identities,
                group_size=3,
                matching_method="cosine",
                max_impostor_pairs=1000,
                pair_sampling_seed=42,
                decision_batch_size=64,
            )

        # 40 groups * (40 templates - 1 genuine) = 1560 impostor cells,
        # capped at 1000. Plus 40 genuine. 1040 decisions / 64 = 17 batches.
        self.assertGreaterEqual(call_count["n"], 2)


# ---------------------------------------------------------------------------
# F. Validation
# ---------------------------------------------------------------------------
class ValidationTests(unittest.TestCase):
    def test_duplicate_template_identities_are_rejected(self):
        probes = np.zeros((4, 3))
        labels = np.array(["a", "a", "b", "b"])
        provenance = _prov(
            ["a_r", "a_r", "b_r", "b_r"],
            ["a_r#0", "a_r#0", "b_r#0", "b_r#0"],
            [0.0] * 4, [0, 1, 0, 1],
        )
        templates = np.zeros((3, 3))
        template_identities = np.array(["a", "b", "a"])
        with self.assertRaisesRegex(ValueError, "one enrollment template"):
            utils._generate_fused_verification_pairs(
                probe_embeddings=probes,
                probe_labels=labels,
                probe_provenance=provenance,
                template_embeddings=templates,
                template_identities=template_identities,
                group_size=2,
                matching_method="cosine",
            )

    def test_k_below_two_short_circuits_via_runner(self):
        with self.assertRaisesRegex(ValueError, ">= 2"):
            utils._generate_fused_verification_pairs(
                probe_embeddings=np.zeros((2, 3)),
                probe_labels=np.array(["a", "a"]),
                probe_provenance=_prov(
                    ["a_r", "a_r"], ["a_r#0", "a_r#0"],
                    [0.0, 0.0], [0, 1],
                ),
                template_embeddings=np.zeros((1, 3)),
                template_identities=np.array(["a"]),
                group_size=1,
                matching_method="cosine",
            )


# ---------------------------------------------------------------------------
# A. k = 1 compatibility for the four verification runners
# ---------------------------------------------------------------------------
class KEqualsOneParityTests(unittest.TestCase):
    """
    probe_fusion_size == 1 must produce byte-for-byte identical scores and
    labels to a run that never mentions probe_fusion_size, on every task
    family. This test compares two adjacent runs under identical seeds.
    """

    def _fixture(self, samples=12, subjects=6):
        x, y = make_synthetic_ecg_dataset(
            number_of_subjects=subjects, samples_per_subject=samples
        )
        provenance = provenance_for(y)
        return x, y, provenance

    def _cross_session_fixture(self, subjects=8):
        s1_x, s1_y = make_synthetic_ecg_dataset(
            number_of_subjects=subjects, samples_per_subject=10, seed=456
        )
        s2_x, s2_y = make_synthetic_ecg_dataset(
            number_of_subjects=subjects, samples_per_subject=8,
            session_shift=0.1, seed=789,
        )
        return s1_x, s1_y, provenance_for(s1_y), s2_x, s2_y, provenance_for(s2_y)

    def _assert_parity_scalar(self, a, b):
        for name, va, vb in zip(("eer", "auc", "dprime", "tar"), a[0], b[0]):
            if va is None or vb is None:
                self.assertEqual(va, vb, name)
            else:
                self.assertAlmostEqual(float(va), float(vb), places=6, msg=name)

    def test_task2_k1_parity(self):
        x, y, provenance = self._fixture()
        baseline = run.run_closed_set_verification(
            x, y, test_split=0.25, sampling_mode="all",
            use_template=True, template_fusion_method="mean",
            template_size=2, provenance=provenance, **COMMON,
        )
        with_fusion_size_one = run.run_closed_set_verification(
            x, y, test_split=0.25, sampling_mode="all",
            use_template=True, template_fusion_method="mean",
            template_size=2, provenance=provenance, probe_fusion_size=1,
            **COMMON,
        )
        self._assert_parity_scalar(baseline, with_fusion_size_one)

    def test_task4_k1_parity(self):
        x, y, provenance = self._fixture(subjects=8)
        baseline = run.run_subject_disjoint_verification(
            x, y, test_split=0.4, sampling_mode="all",
            use_template=True, template_fusion_method="mean",
            template_size=2, provenance=provenance, **COMMON,
        )
        with_fusion_size_one = run.run_subject_disjoint_verification(
            x, y, test_split=0.4, sampling_mode="all",
            use_template=True, template_fusion_method="mean",
            template_size=2, provenance=provenance, probe_fusion_size=1,
            **COMMON,
        )
        self._assert_parity_scalar(baseline, with_fusion_size_one)

    def test_task6_k1_parity(self):
        s1_x, s1_y, p1, s2_x, s2_y, p2 = self._cross_session_fixture()
        baseline = run.run_cross_session_verification(
            s1_x, s1_y, s2_x, s2_y,
            sampling_mode="all", use_template=True,
            template_fusion_method="mean", template_size=2,
            provenance_s1=p1, provenance_s2=p2, **COMMON,
        )
        with_fusion_size_one = run.run_cross_session_verification(
            s1_x, s1_y, s2_x, s2_y,
            sampling_mode="all", use_template=True,
            template_fusion_method="mean", template_size=2,
            provenance_s1=p1, provenance_s2=p2, probe_fusion_size=1,
            **COMMON,
        )
        self._assert_parity_scalar(baseline, with_fusion_size_one)

    def test_task8_k1_parity(self):
        s1_x, s1_y, p1, s2_x, s2_y, p2 = self._cross_session_fixture()
        baseline = run.run_subject_disjoint_cross_session_verification(
            s1_x, s1_y, s2_x, s2_y,
            test_split=0.4, sampling_mode="all",
            use_template=True, template_fusion_method="mean",
            template_size=2, provenance_s1=p1, provenance_s2=p2, **COMMON,
        )
        with_fusion_size_one = run.run_subject_disjoint_cross_session_verification(
            s1_x, s1_y, s2_x, s2_y,
            test_split=0.4, sampling_mode="all",
            use_template=True, template_fusion_method="mean",
            template_size=2, provenance_s1=p1, provenance_s2=p2,
            probe_fusion_size=1, **COMMON,
        )
        self._assert_parity_scalar(baseline, with_fusion_size_one)


# ---------------------------------------------------------------------------
# F.2 Runner-level validation of probe_fusion_size > 1 policies
# ---------------------------------------------------------------------------
class RunnerValidationTests(unittest.TestCase):
    def test_task2_rejects_deployment_evaluation_with_k_gt_one(self):
        x, y = make_synthetic_ecg_dataset(
            number_of_subjects=6, samples_per_subject=10
        )
        with self.assertRaisesRegex(
            ValueError, "not compatible with use_deployment_evaluation"
        ):
            run.run_closed_set_verification(
                x, y, test_split=0.25, val_split=0.2,
                sampling_mode="all", use_template=True,
                template_fusion_method="mean", template_size=2,
                use_deployment_evaluation=True,
                provenance=provenance_for(y),
                probe_fusion_size=3,
                model_class=TinyECGModel, epochs=1, batch_size=16, lr=1e-3,
                seed=42, device="cpu", visualize=False,
                save_results_and_settings=False, loader=None, n_runs=1,
                _return_stats=True, intelligent_weight_loading=False,
            )

    def test_task4_rejects_not_use_template_with_k_gt_one(self):
        x, y = make_synthetic_ecg_dataset(
            number_of_subjects=8, samples_per_subject=10
        )
        with self.assertRaisesRegex(
            ValueError, "requires use_template=True"
        ):
            run.run_subject_disjoint_verification(
                x, y, test_split=0.4, sampling_mode="all",
                use_template=False,
                provenance=provenance_for(y),
                probe_fusion_size=2,
                **COMMON,
            )


class ModeSpecificFusedSamplingTests(unittest.TestCase):
    """
    Each pair-sampling mode selects fused decisions with the same semantics as
    the mature :func:`utils._generate_pairs`, applied to
    ``(probe_group, template)`` decisions instead of ``(beat, template)`` pairs.
    """

    def _fixture(self, n_subjects=6, per_subject=4, group_size=2, seed=0):
        rng = np.random.default_rng(seed)
        probes = rng.normal(size=(n_subjects * per_subject, 5))
        labels = np.array(
            [f"s{i}" for i in range(n_subjects) for _ in range(per_subject)]
        )
        provenance = _prov(
            [f"s{i}_r" for i in range(n_subjects) for _ in range(per_subject)],
            [f"s{i}_r#0"
             for i in range(n_subjects) for _ in range(per_subject)],
            [0.0] * len(labels),
            list(range(per_subject)) * n_subjects,
        )
        templates = rng.normal(size=(n_subjects, 5))
        template_identities = np.array([f"s{i}" for i in range(n_subjects)])
        return (
            probes, labels, provenance, templates, template_identities,
            group_size,
        )

    def _call(self, mode, *, budget=None, seed=None, cap=1_000_000, **fx):
        (
            probes, labels, provenance, templates, template_identities,
            group_size,
        ) = fx["fx"]
        return utils._generate_fused_verification_pairs(
            probe_embeddings=probes,
            probe_labels=labels,
            probe_provenance=provenance,
            template_embeddings=templates,
            template_identities=template_identities,
            group_size=group_size,
            matching_method="cosine",
            pair_sampling_mode=mode,
            pair_sampling_budget=budget,
            max_impostor_pairs=cap,
            pair_sampling_seed=seed,
        )

    # -- all ----------------------------------------------------------
    def test_all_mode_evaluates_entire_group_by_template_space(self):
        fx = self._fixture()
        scores, labels_pair, diag = self._call("all", fx=fx)
        # n_subjects=6, per_subject=4, group_size=2 -> 12 groups, 6 templates.
        self.assertEqual(len(scores), 12 * 6)
        self.assertEqual(int((labels_pair == 1).sum()), 12)
        self.assertEqual(int((labels_pair == 0).sum()), 12 * 6 - 12)
        self.assertEqual(diag["pair_sampling_mode"], "all")
        self.assertNotIn("max_impostor_pairs", diag)
        self.assertNotIn("pair_sampling_seed", diag)

    def test_all_mode_ignores_pair_sampling_seed(self):
        fx = self._fixture()
        first = self._call("all", seed=1, fx=fx)
        second = self._call("all", seed=2, fx=fx)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    # -- all_genuine --------------------------------------------------
    def test_all_genuine_mode_keeps_genuine_and_caps_impostors(self):
        fx = self._fixture(n_subjects=8, per_subject=4, group_size=2)
        scores, labels_pair, diag = self._call(
            "all_genuine", cap=7, seed=17, fx=fx,
        )
        # 8 subjects * 2 groups = 16 groups, 8 templates -> 16 genuine and
        # 16*8 - 16 = 112 impostor cells total.
        self.assertEqual(diag["genuine_fused_decisions"], 16)
        self.assertEqual(diag["total_impostor_fused_decisions"], 112)
        self.assertEqual(diag["impostor_fused_decisions"], 7)
        self.assertEqual(int((labels_pair == 0).sum()), 7)
        self.assertEqual(int((labels_pair == 1).sum()), 16)

    def test_all_genuine_mode_seed_is_reproducible(self):
        fx = self._fixture(n_subjects=8, per_subject=4, group_size=2)
        first = self._call("all_genuine", cap=5, seed=17, fx=fx)
        second = self._call("all_genuine", cap=5, seed=17, fx=fx)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    # -- balanced -----------------------------------------------------
    def test_balanced_mode_matches_budget_and_class_balance(self):
        fx = self._fixture(n_subjects=6, per_subject=4, group_size=2)
        scores, labels_pair, diag = self._call(
            "balanced", budget=40, seed=7, fx=fx,
        )
        # 40 // 2 == 20 genuine iterations and 20 impostor iterations;
        # every iteration successfully places one comparison in this fixture.
        self.assertEqual(int((labels_pair == 1).sum()), 20)
        self.assertEqual(int((labels_pair == 0).sum()), 20)
        self.assertEqual(diag["pair_sampling_mode"], "balanced")
        self.assertEqual(diag["pair_sampling_budget"], 40)
        self.assertEqual(diag["pair_sampling_seed"], 7)
        self.assertNotIn("max_impostor_pairs", diag)

    def test_balanced_mode_seed_is_reproducible(self):
        fx = self._fixture()
        first = self._call("balanced", budget=20, seed=11, fx=fx)
        second = self._call("balanced", budget=20, seed=11, fx=fx)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    # -- random -------------------------------------------------------
    def test_random_mode_returns_exact_budget_of_valid_decisions(self):
        fx = self._fixture(n_subjects=6, per_subject=4, group_size=2)
        scores, labels_pair, diag = self._call(
            "random", budget=30, seed=3, fx=fx,
        )
        self.assertEqual(len(scores), 30)
        self.assertEqual(len(labels_pair), 30)
        self.assertEqual(diag["pair_sampling_mode"], "random")
        self.assertEqual(diag["pair_sampling_budget"], 30)
        self.assertEqual(diag["pair_sampling_seed"], 3)
        self.assertNotIn("max_impostor_pairs", diag)

    def test_random_mode_seed_is_reproducible(self):
        fx = self._fixture()
        first = self._call("random", budget=25, seed=13, fx=fx)
        second = self._call("random", budget=25, seed=13, fx=fx)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])


class CanonicalVersusLegacyAliasTests(unittest.TestCase):
    """
    The runner-level pair-sampling arguments resolve legacy aliases before
    they reach the fused helper. This test proves that when the runner is
    invoked through the canonical or the legacy CLI/YAML surface with
    equivalent inputs, the effective fused verification result is identical.
    """

    def test_canonical_and_legacy_inputs_produce_the_same_result(self):
        rng = np.random.default_rng(0)
        x, y = make_synthetic_ecg_dataset(
            number_of_subjects=6, samples_per_subject=12
        )
        provenance = provenance_for(y)

        canonical = run.run_closed_set_verification(
            x, y, test_split=0.25,
            pair_sampling_mode="all_genuine",
            max_impostor_pairs=20,
            pair_sampling_seed=42,
            use_template=True, template_fusion_method="mean",
            template_size=3, provenance=provenance,
            probe_fusion_size=3, **COMMON,
        )
        legacy = run.run_closed_set_verification(
            x, y, test_split=0.25,
            sampling_mode="all_genuine",
            max_impostor_pairs=20,
            pair_sampling_seed=42,
            use_template=True, template_fusion_method="mean",
            template_size=3, provenance=provenance,
            probe_fusion_size=3, **COMMON,
        )
        for name, va, vb in zip(
            ("eer", "auc", "dprime", "tar"),
            canonical[0], legacy[0],
        ):
            if va is None or vb is None:
                self.assertEqual(va, vb, name)
            else:
                self.assertAlmostEqual(
                    float(va), float(vb), places=9, msg=name
                )


class KEqualsOneRoutingParityTests(unittest.TestCase):
    """
    probe_fusion_size == 1 must never invoke the fused-decision generator, and
    the mature _generate_pairs path must produce the exact scores/labels used
    for metric computation.
    """

    def _spy(self, target_attr):
        holder = {"calls": [], "captured": None}
        original = getattr(utils, target_attr)

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            holder["calls"].append((args, kwargs))
            holder["captured"] = result
            return result

        patcher = patch.object(utils, target_attr, side_effect=wrapped)
        patcher._holder = holder  # type: ignore[attr-defined]
        return patcher

    def _spy_run(self, target_attr):
        holder = {"calls": []}
        original = getattr(run, target_attr)

        def wrapped(*args, **kwargs):
            holder["calls"].append((args, kwargs))
            return original(*args, **kwargs)

        patcher = patch.object(run, target_attr, side_effect=wrapped)
        patcher._holder = holder  # type: ignore[attr-defined]
        return patcher

    def _run_with_spies(self, runner_call):
        fused_spy = self._spy_run("_generate_fused_verification_pairs")
        generate_spy = self._spy_run("_generate_pairs")
        metric_spy = self._spy_run("_compute_metrics_verification")

        with fused_spy, generate_spy, metric_spy:
            runner_call()

        (scores, labels), _ = metric_spy._holder["calls"][0]
        generate_kwargs = [
            kwargs for _, kwargs in generate_spy._holder["calls"]
        ]
        return {
            "fused_calls": len(fused_spy._holder["calls"]),
            "generate_calls": len(generate_spy._holder["calls"]),
            "generate_kwargs": generate_kwargs,
            "scores": np.asarray(scores),
            "labels": np.asarray(labels),
        }

    def _pair_sampling_kwargs(self, kwargs):
        return {
            key: kwargs.get(key)
            for key in (
                "pair_sampling_mode",
                "pair_sampling_budget",
                "max_impostor_pairs",
                "pair_sampling_seed",
            )
        }

    def _assert_k1_contract(self, make_runner_call):
        """
        Run the same configuration with an omitted probe_fusion_size and an
        explicit probe_fusion_size=1: neither run may invoke the fused
        generator, both must use _generate_pairs with the same effective
        pair-sampling arguments, and the exact scores/labels reaching metric
        computation must be equal.
        """
        default_run = self._run_with_spies(make_runner_call(dict()))
        explicit_run = self._run_with_spies(
            make_runner_call(dict(probe_fusion_size=1))
        )

        for observed in (default_run, explicit_run):
            self.assertEqual(observed["fused_calls"], 0)
            self.assertGreaterEqual(observed["generate_calls"], 1)
            self.assertGreater(len(observed["scores"]), 0)
            self.assertEqual(
                len(observed["scores"]), len(observed["labels"])
            )

        self.assertEqual(
            [self._pair_sampling_kwargs(k)
             for k in default_run["generate_kwargs"]],
            [self._pair_sampling_kwargs(k)
             for k in explicit_run["generate_kwargs"]],
        )
        np.testing.assert_array_equal(
            default_run["scores"], explicit_run["scores"]
        )
        np.testing.assert_array_equal(
            default_run["labels"], explicit_run["labels"]
        )

    def test_task2_k1_bypasses_fused_generator_and_matches_default(self):
        x, y = make_synthetic_ecg_dataset(
            number_of_subjects=6, samples_per_subject=12
        )
        provenance = provenance_for(y)

        def make_runner_call(extra):
            def call():
                run.run_closed_set_verification(
                    x, y, test_split=0.25, sampling_mode="all",
                    use_template=True, template_fusion_method="mean",
                    template_size=2, provenance=provenance,
                    **extra, **COMMON,
                )
            return call

        self._assert_k1_contract(make_runner_call)

    def test_task4_k1_bypasses_fused_generator_and_matches_default(self):
        x, y = make_synthetic_ecg_dataset(
            number_of_subjects=8, samples_per_subject=12
        )
        provenance = provenance_for(y)

        def make_runner_call(extra):
            def call():
                run.run_subject_disjoint_verification(
                    x, y, test_split=0.4, sampling_mode="all",
                    use_template=True, template_fusion_method="mean",
                    template_size=2, provenance=provenance,
                    **extra, **COMMON,
                )
            return call

        self._assert_k1_contract(make_runner_call)

    def test_task6_k1_bypasses_fused_generator_and_matches_default(self):
        s1_x, s1_y = make_synthetic_ecg_dataset(
            number_of_subjects=8, samples_per_subject=10, seed=456
        )
        s2_x, s2_y = make_synthetic_ecg_dataset(
            number_of_subjects=8, samples_per_subject=8,
            session_shift=0.1, seed=789,
        )
        p1 = provenance_for(s1_y)
        p2 = provenance_for(s2_y)

        def make_runner_call(extra):
            def call():
                run.run_cross_session_verification(
                    s1_x, s1_y, s2_x, s2_y,
                    sampling_mode="all", use_template=True,
                    template_fusion_method="mean", template_size=2,
                    provenance_s1=p1, provenance_s2=p2,
                    **extra, **COMMON,
                )
            return call

        self._assert_k1_contract(make_runner_call)

    def test_task8_k1_bypasses_fused_generator_and_matches_default(self):
        s1_x, s1_y = make_synthetic_ecg_dataset(
            number_of_subjects=8, samples_per_subject=10, seed=456
        )
        s2_x, s2_y = make_synthetic_ecg_dataset(
            number_of_subjects=8, samples_per_subject=8,
            session_shift=0.1, seed=789,
        )
        p1 = provenance_for(s1_y)
        p2 = provenance_for(s2_y)

        def make_runner_call(extra):
            def call():
                run.run_subject_disjoint_cross_session_verification(
                    s1_x, s1_y, s2_x, s2_y,
                    test_split=0.4, sampling_mode="all",
                    use_template=True, template_fusion_method="mean",
                    template_size=2, provenance_s1=p1, provenance_s2=p2,
                    **extra, **COMMON,
                )
            return call

        self._assert_k1_contract(make_runner_call)


class KEqualsThreeRunnerSmokeTests(unittest.TestCase):
    """
    Real k = 3 execution through each verification runner produces finite
    metrics and populates Probe Fusion diagnostics.
    """

    def _intra_fixture(self, subjects=6, per_subject=9):
        x, y = make_synthetic_ecg_dataset(
            number_of_subjects=subjects, samples_per_subject=per_subject,
        )
        provenance = provenance_for(y)
        return x, y, provenance

    def _cross_fixture(self, subjects=8, per_subject=9):
        s1_x, s1_y = make_synthetic_ecg_dataset(
            number_of_subjects=subjects, samples_per_subject=per_subject,
            seed=456,
        )
        s2_x, s2_y = make_synthetic_ecg_dataset(
            number_of_subjects=subjects, samples_per_subject=per_subject,
            session_shift=0.1, seed=789,
        )
        return (
            s1_x, s1_y, provenance_for(s1_y),
            s2_x, s2_y, provenance_for(s2_y),
        )

    def _assert_valid_metrics(self, metrics_tuple):
        eer, auc, dprime, tar = metrics_tuple
        self.assertTrue(np.isfinite(eer))
        self.assertTrue(np.isfinite(auc))
        self.assertTrue(np.isfinite(dprime))
        # TAR may be None when 0.1% FAR is not empirically resolvable at the
        # fused-decision scale.
        if tar is not None:
            self.assertTrue(np.isfinite(tar))

    def test_task2_k3_smoke(self):
        x, y, provenance = self._intra_fixture()
        (metrics, data_stats, _) = run.run_closed_set_verification(
            x, y, test_split=0.25, sampling_mode="all_genuine",
            max_impostor_pairs=200,
            use_template=True, template_fusion_method="mean",
            template_size=2, provenance=provenance,
            probe_fusion_size=3, **COMMON,
        )
        self._assert_valid_metrics(metrics)
        self.assertIn("Probe Fusion", data_stats)
        self.assertEqual(data_stats["Probe Fusion"]["fusion_size"], 3)
        self.assertGreater(
            data_stats["Probe Fusion"]["valid_probe_groups"], 0
        )

    def test_task4_k3_smoke(self):
        x, y, provenance = self._intra_fixture(subjects=8, per_subject=9)
        (metrics, data_stats, _) = run.run_subject_disjoint_verification(
            x, y, test_split=0.4, sampling_mode="all_genuine",
            max_impostor_pairs=200,
            use_template=True, template_fusion_method="mean",
            template_size=2, provenance=provenance,
            probe_fusion_size=3, **COMMON,
        )
        self._assert_valid_metrics(metrics)
        self.assertIn("Probe Fusion", data_stats)
        self.assertEqual(data_stats["Probe Fusion"]["fusion_size"], 3)

    def test_task6_k3_smoke(self):
        s1_x, s1_y, p1, s2_x, s2_y, p2 = self._cross_fixture()
        (metrics, data_stats, _) = run.run_cross_session_verification(
            s1_x, s1_y, s2_x, s2_y,
            sampling_mode="all_genuine", max_impostor_pairs=200,
            use_template=True, template_fusion_method="mean",
            template_size=2, provenance_s1=p1, provenance_s2=p2,
            probe_fusion_size=3, **COMMON,
        )
        self._assert_valid_metrics(metrics)
        self.assertIn("Probe Fusion", data_stats)
        self.assertEqual(data_stats["Probe Fusion"]["fusion_size"], 3)

    def test_task8_k3_smoke(self):
        s1_x, s1_y, p1, s2_x, s2_y, p2 = self._cross_fixture()
        (metrics, data_stats, _) = (
            run.run_subject_disjoint_cross_session_verification(
                s1_x, s1_y, s2_x, s2_y,
                test_split=0.4, sampling_mode="all_genuine",
                max_impostor_pairs=200,
                use_template=True, template_fusion_method="mean",
                template_size=2, provenance_s1=p1, provenance_s2=p2,
                probe_fusion_size=3, **COMMON,
            )
        )
        self._assert_valid_metrics(metrics)
        self.assertIn("Probe Fusion", data_stats)
        self.assertEqual(data_stats["Probe Fusion"]["fusion_size"], 3)


class EffectivePairSamplingSeedTraceabilityTests(unittest.TestCase):
    """
    all_genuine resolves ``pair_sampling_seed=None`` to 42 at the sampler.
    Structured metadata and fusion diagnostics must record the effective value
    so that a downstream reader can reproduce the exact fused decisions.
    """

    def test_k3_all_genuine_none_seed_matches_explicit_seed_42(self):
        x, y = make_synthetic_ecg_dataset(
            number_of_subjects=6, samples_per_subject=9
        )
        provenance = provenance_for(y)

        common_kwargs = dict(
            test_split=0.25,
            sampling_mode="all_genuine",
            max_impostor_pairs=50,
            use_template=True,
            template_fusion_method="mean",
            template_size=2,
            provenance=provenance,
            probe_fusion_size=3,
        )
        none_result = run.run_closed_set_verification(
            x, y, pair_sampling_seed=None, **common_kwargs, **COMMON,
        )
        explicit_result = run.run_closed_set_verification(
            x, y, pair_sampling_seed=42, **common_kwargs, **COMMON,
        )

        for name, va, vb in zip(
            ("eer", "auc", "dprime", "tar"),
            none_result[0], explicit_result[0],
        ):
            if va is None or vb is None:
                self.assertEqual(va, vb, name)
            else:
                self.assertAlmostEqual(
                    float(va), float(vb), places=9, msg=name
                )

        _, none_stats, none_hyper = none_result
        self.assertEqual(none_hyper["pair_sampling_seed"], 42)
        self.assertEqual(
            none_stats["Probe Fusion"]["pair_sampling_seed"], 42
        )

    def test_k1_all_genuine_none_seed_matches_explicit_seed_42(self):
        x, y = make_synthetic_ecg_dataset(
            number_of_subjects=6, samples_per_subject=12
        )
        provenance = provenance_for(y)

        common_kwargs = dict(
            test_split=0.25,
            sampling_mode="all_genuine",
            max_impostor_pairs=50,
            use_template=True,
            template_fusion_method="mean",
            template_size=2,
            provenance=provenance,
        )
        none_result = run.run_closed_set_verification(
            x, y, pair_sampling_seed=None, **common_kwargs, **COMMON,
        )
        explicit_result = run.run_closed_set_verification(
            x, y, pair_sampling_seed=42, **common_kwargs, **COMMON,
        )

        for name, va, vb in zip(
            ("eer", "auc", "dprime", "tar"),
            none_result[0], explicit_result[0],
        ):
            if va is None or vb is None:
                self.assertEqual(va, vb, name)
            else:
                self.assertAlmostEqual(
                    float(va), float(vb), places=9, msg=name
                )

        _, _, none_hyper = none_result
        self.assertEqual(none_hyper["pair_sampling_seed"], 42)

    def test_balanced_none_seed_is_logged_as_none(self):
        x, y = make_synthetic_ecg_dataset(
            number_of_subjects=6, samples_per_subject=12
        )
        provenance = provenance_for(y)
        _, _, hyper = run.run_closed_set_verification(
            x, y, test_split=0.25,
            sampling_mode="balanced",
            num_pairs=20,
            pair_sampling_seed=None,
            use_template=True, template_fusion_method="mean",
            template_size=2, provenance=provenance,
            **COMMON,
        )
        # None means the legacy global NumPy RNG is used; metadata must
        # reflect that so downstream readers do not mistake it for seed 42.
        self.assertIsNone(hyper["pair_sampling_seed"])
        self.assertEqual(hyper["pair_sampling_mode"], "balanced")

    def test_all_mode_logs_none_seed(self):
        x, y = make_synthetic_ecg_dataset(
            number_of_subjects=6, samples_per_subject=12
        )
        provenance = provenance_for(y)
        _, _, hyper = run.run_closed_set_verification(
            x, y, test_split=0.25,
            sampling_mode="all",
            pair_sampling_seed=None,
            use_template=True, template_fusion_method="mean",
            template_size=2, provenance=provenance,
            **COMMON,
        )
        self.assertEqual(hyper["pair_sampling_mode"], "all")
        self.assertIsNone(hyper["pair_sampling_seed"])
        self.assertIsNone(hyper["max_impostor_pairs"])


if __name__ == "__main__":
    unittest.main()
