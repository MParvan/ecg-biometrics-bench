"""
Multi-template enrollment: selection, composition, sampling, and runners.

``enrollment_template_mode='multi_template'`` stores a fixed number of
representative enrollment observations per identity instead of fusing them
into one template. These tests pin the selection algorithm, the
identity-level score composition (probe fusion before the per-identity
maximum), the verification decision space, the four pair-sampling modes,
and — critically — that the default ``fusion`` mode is byte-for-byte the
behavior the framework had before this option existed.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run
import utils
from load_dataset import BeatProvenance
from utils import (
    _apply_score_fusion,
    _generate_multi_template_verification_pairs,
    _reduce_template_scores_to_identities,
    _select_multi_templates,
)
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


def _custom_provenance(record_ids, beat_ordinals):
    """Build provenance with explicit per-row source ordering."""
    count = len(record_ids)
    return BeatProvenance({
        "record_id": np.array(record_ids, dtype=object),
        "session_id": np.array(record_ids, dtype=object),
        "acquisition_time": np.array([None] * count, dtype=object),
        "acquisition_order": np.zeros(count, dtype=np.int64),
        "source_segment_id": np.array(
            [f"{record}#0" for record in record_ids], dtype=object
        ),
        "source_segment_order": np.zeros(count, dtype=np.float64),
        "beat_ordinal": np.asarray(beat_ordinals, dtype=np.int64),
        "rpeak_index": np.full(count, -1, dtype=np.int64),
    })


# =========================================================================
# Argument validation shared by every runner
# =========================================================================
class ModeArgumentValidation(unittest.TestCase):
    def _validate(self, **overrides):
        arguments = {
            "enrollment_template_mode": "fusion",
            "template_fusion_method": run._UNSET,
            "num_templates_per_identity": None,
            "template_selection_method": None,
            "template_score_aggregation": None,
            "use_template": True,
            "use_deployment_evaluation": False,
            "task_name": "Test Task",
        }
        arguments.update(overrides)
        return run._validate_multi_template_arguments(
            arguments["enrollment_template_mode"],
            arguments["template_fusion_method"],
            arguments["num_templates_per_identity"],
            arguments["template_selection_method"],
            arguments["template_score_aggregation"],
            arguments["use_template"],
            arguments["use_deployment_evaluation"],
            arguments["task_name"],
        )

    def test_fusion_defaults_pass_and_resolve_to_none(self):
        resolved = self._validate()
        self.assertEqual(resolved, ("fusion", None, None, None))

    def test_fusion_accepts_any_explicit_fusion_method(self):
        for method in ("mean", "median", "none", None):
            resolved = self._validate(template_fusion_method=method)
            self.assertEqual(resolved[0], "fusion")

    def test_fusion_rejects_each_multi_template_argument(self):
        for name, value in (
            ("num_templates_per_identity", 3),
            ("template_selection_method", "farthest_first_cosine"),
            ("template_score_aggregation", "max"),
        ):
            with self.assertRaisesRegex(ValueError, name):
                self._validate(**{name: value})

    def test_unknown_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, "enrollment_template_mode"):
            self._validate(enrollment_template_mode="multi")

    def test_multi_template_requires_k(self):
        with self.assertRaisesRegex(
            ValueError, "num_templates_per_identity"
        ):
            self._validate(enrollment_template_mode="multi_template")

    def test_multi_template_rejects_invalid_k(self):
        for bad_k in (0, -1, "three", True, 2.7, 3.0, np.float64(3.0)):
            with self.subTest(bad_k=bad_k):
                with self.assertRaisesRegex(
                    ValueError, "num_templates_per_identity"
                ):
                    self._validate(
                        enrollment_template_mode="multi_template",
                        num_templates_per_identity=bad_k,
                    )

    def test_multi_template_accepts_numpy_integer_k(self):
        # Genuine integral values -- Python int or numpy integer -- are the
        # only accepted representations; no coercion from floats or strings.
        resolved = self._validate(
            enrollment_template_mode="multi_template",
            num_templates_per_identity=np.int64(3),
        )
        self.assertEqual(resolved[1], 3)

    def test_multi_template_rejects_explicit_fusion_method(self):
        # Any explicit value counts, including the legacy no-fusion None.
        for method in ("mean", "none", None):
            with self.assertRaisesRegex(
                ValueError, "template_fusion_method"
            ):
                self._validate(
                    enrollment_template_mode="multi_template",
                    num_templates_per_identity=3,
                    template_fusion_method=method,
                )

    def test_multi_template_requires_use_template(self):
        with self.assertRaisesRegex(ValueError, "use_template=True"):
            self._validate(
                enrollment_template_mode="multi_template",
                num_templates_per_identity=3,
                use_template=False,
            )

    def test_multi_template_rejects_deployment_evaluation(self):
        with self.assertRaisesRegex(
            ValueError, "use_deployment_evaluation"
        ):
            self._validate(
                enrollment_template_mode="multi_template",
                num_templates_per_identity=3,
                use_deployment_evaluation=True,
            )

    def test_multi_template_rejects_unknown_method_and_aggregation(self):
        with self.assertRaisesRegex(
            ValueError, "template_selection_method"
        ):
            self._validate(
                enrollment_template_mode="multi_template",
                num_templates_per_identity=3,
                template_selection_method="k_medoids",
            )
        with self.assertRaisesRegex(
            ValueError, "template_score_aggregation"
        ):
            self._validate(
                enrollment_template_mode="multi_template",
                num_templates_per_identity=3,
                template_score_aggregation="mean",
            )

    def test_multi_template_resolves_defaults(self):
        resolved = self._validate(
            enrollment_template_mode="multi_template",
            num_templates_per_identity=3,
        )
        self.assertEqual(
            resolved,
            ("multi_template", 3, "farthest_first_cosine", "max"),
        )


# =========================================================================
# Template selection
# =========================================================================
class TemplateSelection(unittest.TestCase):
    @staticmethod
    def _clustered_embeddings(counts, feature_dim=6, seed=7):
        rng = np.random.default_rng(seed)
        embeddings, labels = [], []
        for identity, count in counts.items():
            center = rng.normal(size=feature_dim) * 3.0
            for _ in range(count):
                embeddings.append(center + rng.normal(scale=0.1, size=feature_dim))
                labels.append(identity)
        return np.asarray(embeddings), np.asarray(labels)

    def test_output_shapes_and_ordering(self):
        embeddings, labels = self._clustered_embeddings({"b": 5, "a": 6, "c": 4})
        templates, identities, sources, diagnostics = _select_multi_templates(
            embeddings, labels, provenance=None, num_templates_per_identity=3
        )
        self.assertEqual(templates.shape, (9, embeddings.shape[1]))
        self.assertEqual(identities.tolist(), ["a"] * 3 + ["b"] * 3 + ["c"] * 3)
        self.assertEqual(len(sources), 9)
        self.assertEqual(diagnostics["templates_selected_total"], 9)
        self.assertEqual(diagnostics["enrolled_identities"], 3)
        self.assertTrue(diagnostics["deterministic"])

    def test_templates_are_actual_rows_at_global_indices(self):
        embeddings, labels = self._clustered_embeddings({"a": 5, "b": 5})
        templates, identities, sources, _ = _select_multi_templates(
            embeddings, labels, provenance=None, num_templates_per_identity=2
        )
        for row, source_index, identity in zip(templates, sources, identities):
            np.testing.assert_array_equal(row, embeddings[source_index])
            self.assertEqual(labels[source_index], identity)

    def test_selection_is_deterministic(self):
        embeddings, labels = self._clustered_embeddings({"a": 8, "b": 8})
        first = _select_multi_templates(
            embeddings, labels, provenance=None, num_templates_per_identity=4
        )
        second = _select_multi_templates(
            embeddings, labels, provenance=None, num_templates_per_identity=4
        )
        np.testing.assert_array_equal(first[2], second[2])
        self.assertEqual(
            first[3]["selected_index_fingerprint"],
            second[3]["selected_index_fingerprint"],
        )

    def test_hand_computed_farthest_first_geometry(self):
        # Three planar unit vectors: 0, 10, and 170 degrees. The candidate
        # closest to the aggregate direction is the 10-degree vector; the
        # farthest-first step then picks the 170-degree vector, and finally
        # the 0-degree vector.
        angles = np.deg2rad([0.0, 10.0, 170.0])
        embeddings = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        labels = np.array(["a", "a", "a"])

        _, _, sources, _ = _select_multi_templates(
            embeddings, labels, provenance=None, num_templates_per_identity=3
        )
        self.assertEqual(sources.tolist(), [1, 2, 0])

    def test_ties_break_toward_earliest_source_position(self):
        embeddings = np.tile(np.array([[1.0, 2.0]]), (4, 1))
        labels = np.array(["a"] * 4)
        _, _, sources, _ = _select_multi_templates(
            embeddings, labels, provenance=None, num_templates_per_identity=2
        )
        self.assertEqual(sources.tolist(), [0, 1])

    def test_tie_break_follows_source_order_regardless_of_truncation(self):
        # Four indistinguishable (tied) embeddings; provenance places
        # canonical source order as the exact reverse of input/array order.
        # Canonical order must govern tie-breaking whether max_beats leaves
        # the pool untouched (None or >= pool size) or actually truncates it.
        embeddings = np.tile(np.array([[1.0, 2.0]]), (4, 1))
        labels = np.array(["a"] * 4)
        provenance = _custom_provenance(["a_r"] * 4, [3, 2, 1, 0])

        for max_beats in (None, 4, 2):
            with self.subTest(max_beats=max_beats):
                _, _, sources, _ = _select_multi_templates(
                    embeddings, labels, provenance=provenance,
                    num_templates_per_identity=2, max_beats=max_beats,
                )
                self.assertEqual(sources.tolist(), [3, 2])

    def test_max_beats_restricts_pool_in_input_order(self):
        embeddings, labels = self._clustered_embeddings({"a": 6})
        _, _, sources, diagnostics = _select_multi_templates(
            embeddings, labels, provenance=None,
            num_templates_per_identity=2, max_beats=3,
        )
        self.assertTrue(set(sources.tolist()) <= {0, 1, 2})
        self.assertEqual(diagnostics["candidate_count_max"], 3)

    def test_max_beats_follows_source_order_with_provenance(self):
        embeddings, labels = self._clustered_embeddings({"a": 4})
        # Source order is the reverse of input order.
        provenance = _custom_provenance(["a_r"] * 4, [3, 2, 1, 0])
        _, _, sources, _ = _select_multi_templates(
            embeddings, labels, provenance=provenance,
            num_templates_per_identity=2, max_beats=2,
        )
        self.assertEqual(set(sources.tolist()), {2, 3})

    def test_source_indices_are_global_and_provenance_aligned(self):
        embeddings, labels = self._clustered_embeddings({"a": 5, "b": 5})
        provenance = provenance_for(labels)
        _, identities, sources, _ = _select_multi_templates(
            embeddings, labels, provenance=provenance,
            num_templates_per_identity=2,
        )
        template_provenance = provenance.subset(sources)
        template_provenance.validate(len(sources))
        np.testing.assert_array_equal(
            template_provenance.columns["record_id"],
            np.array([f"{identity}_r" for identity in identities], dtype=object),
        )

    def test_selector_rejects_fractional_k(self):
        embeddings, labels = self._clustered_embeddings({"a": 6})
        for bad_k in (2.7, 3.0, "3", True):
            with self.subTest(bad_k=bad_k):
                with self.assertRaisesRegex(
                    ValueError, "num_templates_per_identity"
                ):
                    _select_multi_templates(
                        embeddings, labels, provenance=None,
                        num_templates_per_identity=bad_k,
                    )

    def test_fingerprint_is_a_complete_sha256_digest(self):
        embeddings, labels = self._clustered_embeddings({"a": 5, "b": 5})
        _, _, _, diagnostics = _select_multi_templates(
            embeddings, labels, provenance=None, num_templates_per_identity=2
        )
        fingerprint = diagnostics["selected_index_fingerprint"]
        self.assertRegex(fingerprint, r"^[0-9a-f]{64}$")

    def test_fingerprint_is_stable_across_repeated_calls(self):
        embeddings, labels = self._clustered_embeddings({"a": 5, "b": 5})
        first = _select_multi_templates(
            embeddings, labels, provenance=None, num_templates_per_identity=2
        )[3]["selected_index_fingerprint"]
        second = _select_multi_templates(
            embeddings, labels, provenance=None, num_templates_per_identity=2
        )[3]["selected_index_fingerprint"]
        self.assertEqual(first, second)

    def test_insufficient_candidates_fail_before_scoring(self):
        embeddings, labels = self._clustered_embeddings({"a": 2, "b": 6})
        with self.assertRaisesRegex(
            ValueError, r"num_templates_per_identity=4.*'a': 2"
        ):
            _select_multi_templates(
                embeddings, labels, provenance=None,
                num_templates_per_identity=4,
            )

    def test_non_finite_candidate_rejected(self):
        embeddings, labels = self._clustered_embeddings({"a": 4})
        embeddings[2, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _select_multi_templates(
                embeddings, labels, provenance=None,
                num_templates_per_identity=2,
            )

    def test_near_zero_norm_candidate_rejected(self):
        embeddings, labels = self._clustered_embeddings({"a": 4})
        embeddings[1] = 0.0
        with self.assertRaisesRegex(ValueError, "near-zero norm"):
            _select_multi_templates(
                embeddings, labels, provenance=None,
                num_templates_per_identity=2,
            )

    def test_bad_embedding_error_reports_global_source_index(self):
        # Provenance reverses input order, so the local candidate position
        # (2) and its global row index into the input embedding array (1)
        # differ; the error must name both.
        embeddings, labels = self._clustered_embeddings({"a": 4})
        embeddings[1] = np.nan
        provenance = _custom_provenance(["a_r"] * 4, [3, 2, 1, 0])
        with self.assertRaisesRegex(
            ValueError, r"candidate position 2 \(source index 1\)"
        ):
            _select_multi_templates(
                embeddings, labels, provenance=provenance,
                num_templates_per_identity=2,
            )

    def test_unknown_selection_method_rejected(self):
        embeddings, labels = self._clustered_embeddings({"a": 4})
        with self.assertRaisesRegex(ValueError, "template_selection_method"):
            _select_multi_templates(
                embeddings, labels, provenance=None,
                num_templates_per_identity=2,
                template_selection_method="k_medoids",
            )

    def test_large_pool_runs_without_pairwise_matrix(self):
        # 20000 candidates: a pairwise similarity matrix would need
        # 400 million entries. The traversal touches only O(N * K) scores.
        rng = np.random.default_rng(0)
        embeddings = rng.normal(size=(20000, 4))
        labels = np.zeros(20000, dtype=int)
        _, _, sources, _ = _select_multi_templates(
            embeddings, labels, provenance=None, num_templates_per_identity=2
        )
        self.assertEqual(len(sources), 2)


# =========================================================================
# Identity-score reduction and composition order
# =========================================================================
class ScoreComposition(unittest.TestCase):
    def test_reduction_takes_maximum_per_identity(self):
        scores = np.array([
            [0.1, 0.9, 0.5, 0.2],
            [0.7, 0.3, 0.4, 0.8],
        ])
        template_identities = np.array([0, 0, 1, 1])
        reduced = _reduce_template_scores_to_identities(
            scores, template_identities, np.array([0, 1])
        )
        np.testing.assert_allclose(reduced, [[0.9, 0.5], [0.7, 0.8]])

    def test_missing_identity_raises_instead_of_negative_infinity(self):
        # identity_order names identity 1 as part of the required canonical
        # identity space, but no template row belongs to it. A required
        # identity with no stored template is a structural enrollment gap
        # and must fail loudly rather than silently score -inf for everyone.
        scores = np.array([[0.5, 0.6]])
        with self.assertRaisesRegex(ValueError, r"Missing identities.*1"):
            _reduce_template_scores_to_identities(
                scores, np.array([0, 0]), np.array([0, 1])
            )

    def test_all_missing_identities_are_named(self):
        scores = np.array([[0.5, 0.6]])
        with self.assertRaisesRegex(
            ValueError, r"Missing identities.*(1.*2|2.*1)"
        ):
            _reduce_template_scores_to_identities(
                scores, np.array([0, 0]), np.array([0, 1, 2])
            )

    def test_reduction_succeeds_when_every_identity_has_a_template(self):
        scores = np.array([[0.1, 0.9, 0.5, 0.2]])
        template_identities = np.array([0, 0, 1, 1])
        reduced = _reduce_template_scores_to_identities(
            scores, template_identities, np.array([0, 1])
        )
        np.testing.assert_allclose(reduced, [[0.9, 0.5]])

    def test_unknown_aggregation_rejected_by_reducer(self):
        scores = np.array([[0.1, 0.9]])
        with self.assertRaisesRegex(
            ValueError, "template_score_aggregation"
        ):
            _reduce_template_scores_to_identities(
                scores, np.array([0, 0]), np.array([0]),
                template_score_aggregation="mean",
            )

    def test_identification_composes_fusion_before_identity_maximum(self):
        # One probe group of two beats, one identity with two templates.
        # Per-beat scores are chosen so the two composition orders differ:
        #   mean-then-max: max(0.5, 0.45) = 0.5
        #   max-then-mean: mean(1.0, 0.9) = 0.95
        raw_scores = np.array([
            [1.0, 0.0],
            [0.0, 0.9],
        ])
        probe_labels = np.array(["a", "a"])
        provenance = provenance_for(probe_labels)

        fused, fused_labels = _apply_score_fusion(
            raw_scores, probe_labels, fusion_size=2, provenance=provenance
        )
        reduced = _reduce_template_scores_to_identities(
            fused, np.array(["a", "a"]), np.array(["a"])
        )
        self.assertEqual(reduced.shape, (1, 1))
        self.assertAlmostEqual(reduced[0, 0], 0.5)
        self.assertNotAlmostEqual(reduced[0, 0], 0.95)

    def test_single_beat_groups_reduce_to_plain_maximum(self):
        raw_scores = np.array([[0.2, 0.8, 0.4]])
        reduced = _reduce_template_scores_to_identities(
            raw_scores, np.array([0, 0, 0]), np.array([0])
        )
        self.assertAlmostEqual(reduced[0, 0], 0.8)


# =========================================================================
# Multi-template verification decisions
# =========================================================================
class VerificationDecisions(unittest.TestCase):
    @staticmethod
    def _scalar_setup():
        # Scalar embeddings with negative-Euclidean scoring make every
        # elementary score hand-computable: score(p, t) = -|p - t|.
        template_embeddings = np.array(
            [[0.0], [100.0], [1000.0], [10000.0]]
        )
        template_identities = np.array(["a", "a", "b", "b"])
        return template_embeddings, template_identities

    def test_composition_is_mean_then_max_not_max_then_mean(self):
        template_embeddings, template_identities = self._scalar_setup()
        probes = np.array([[0.0], [100.0]])
        probe_labels = np.array(["a", "a"])
        provenance = provenance_for(probe_labels)

        scores, labels_pair, _ = _generate_multi_template_verification_pairs(
            probe_embeddings=probes,
            probe_labels=probe_labels,
            probe_provenance=provenance,
            template_embeddings=template_embeddings,
            template_identities=template_identities,
            num_templates_per_identity=2,
            probe_fusion_size=2,
            matching_method="euclidean",
            pair_sampling_mode="all",
        )
        genuine = scores[labels_pair == 1]
        self.assertEqual(len(genuine), 1)
        # mean-then-max: both templates average to -50, so the score is -50.
        # max-then-mean would give mean(0, 0) = 0 instead.
        self.assertAlmostEqual(genuine[0], -50.0)

    def test_batch_scorer_maps_each_score_to_its_target_template(self):
        template_embeddings, template_identities = self._scalar_setup()
        probes = np.array([[0.0], [60.0], [1000.0], [9000.0]])
        probe_labels = np.array(["a", "a", "b", "b"])
        provenance = provenance_for(probe_labels)

        scores, labels_pair, _ = _generate_multi_template_verification_pairs(
            probe_embeddings=probes,
            probe_labels=probe_labels,
            probe_provenance=provenance,
            template_embeddings=template_embeddings,
            template_identities=template_identities,
            num_templates_per_identity=2,
            probe_fusion_size=2,
            matching_method="euclidean",
            pair_sampling_mode="all",
        )
        # Decision order for mode "all" is group-major then identity:
        # (Ga, a), (Ga, b), (Gb, a), (Gb, b).
        expected = np.array([
            max(np.mean([-0.0, -60.0]), np.mean([-100.0, -40.0])),
            max(np.mean([-1000.0, -940.0]), np.mean([-10000.0, -9940.0])),
            max(np.mean([-1000.0, -9000.0]), np.mean([-900.0, -8900.0])),
            max(np.mean([-1000.0, -9000.0]), np.mean([-8000.0, -0.0])),
        ])
        np.testing.assert_allclose(scores, expected)
        np.testing.assert_array_equal(labels_pair, [1, 0, 0, 1])

    def test_single_beat_probes_need_no_provenance(self):
        template_embeddings, template_identities = self._scalar_setup()
        probes = np.array([[1.0], [999.0]])
        probe_labels = np.array(["a", "b"])
        scores, labels_pair, diagnostics = (
            _generate_multi_template_verification_pairs(
                probe_embeddings=probes,
                probe_labels=probe_labels,
                probe_provenance=None,
                template_embeddings=template_embeddings,
                template_identities=template_identities,
                num_templates_per_identity=2,
                probe_fusion_size=1,
                matching_method="euclidean",
                pair_sampling_mode="all",
            )
        )
        self.assertEqual(len(scores), 4)
        self.assertEqual(diagnostics["fusion_size"], 1)
        # Genuine score for probe 1.0 against identity a: max(-1, -99) = -1.
        self.assertAlmostEqual(scores[0], -1.0)

    def test_genuine_count_is_per_group_not_per_template(self):
        rng = np.random.default_rng(3)
        probes = rng.normal(size=(12, 4))
        probe_labels = np.repeat(["a", "b", "c"], 4)
        for k in (1, 3):
            templates = rng.normal(size=(3 * k, 4))
            identities = np.repeat(["a", "b", "c"], k)
            _, labels_pair, diagnostics = (
                _generate_multi_template_verification_pairs(
                    probe_embeddings=probes,
                    probe_labels=probe_labels,
                    probe_provenance=None,
                    template_embeddings=templates,
                    template_identities=identities,
                    num_templates_per_identity=k,
                    probe_fusion_size=1,
                    matching_method="cosine",
                    pair_sampling_mode="all_genuine",
                )
            )
            self.assertEqual(diagnostics["genuine_fused_decisions"], 12)
            self.assertEqual(int(labels_pair.sum()), 12)

    def test_unequal_template_counts_rejected(self):
        templates = np.random.default_rng(0).normal(size=(3, 4))
        identities = np.array(["a", "a", "b"])
        with self.assertRaisesRegex(ValueError, "same number"):
            _generate_multi_template_verification_pairs(
                probe_embeddings=templates,
                probe_labels=np.array(["a", "a", "b"]),
                probe_provenance=None,
                template_embeddings=templates,
                template_identities=identities,
                num_templates_per_identity=1,
                probe_fusion_size=1,
                matching_method="cosine",
                pair_sampling_mode="all",
            )

    def test_probe_identity_without_enrolled_target_rejected(self):
        rng = np.random.default_rng(0)
        templates = rng.normal(size=(4, 4))
        identities = np.array(["a", "a", "b", "b"])
        with self.assertRaisesRegex(ValueError, "no.*enrolled target"):
            _generate_multi_template_verification_pairs(
                probe_embeddings=rng.normal(size=(2, 4)),
                probe_labels=np.array(["a", "z"]),
                probe_provenance=None,
                template_embeddings=templates,
                template_identities=identities,
                num_templates_per_identity=2,
                probe_fusion_size=1,
                matching_method="cosine",
                pair_sampling_mode="all",
            )

    def test_unknown_aggregation_rejected(self):
        with self.assertRaisesRegex(ValueError, "template_score_aggregation"):
            _generate_multi_template_verification_pairs(
                probe_embeddings=np.ones((2, 3)),
                probe_labels=np.array(["a", "b"]),
                probe_provenance=None,
                template_embeddings=np.ones((2, 3)),
                template_identities=np.array(["a", "b"]),
                num_templates_per_identity=1,
                probe_fusion_size=1,
                matching_method="cosine",
                pair_sampling_mode="all",
                template_score_aggregation="mean",
            )


class RequestedKEnforcement(unittest.TestCase):
    """
    ``_generate_multi_template_verification_pairs`` enforces the configured
    ``num_templates_per_identity`` explicitly, rather than only inferring a
    per-identity template count from whatever the supplied gallery happens to
    contain. The standard runner path already guarantees this through the
    selector, so this is defense-in-depth against a structurally
    inconsistent direct call, not evidence that the runner path is wrong.
    """

    @staticmethod
    def _gallery(k, n_identities=3):
        rng = np.random.default_rng(0)
        identities = np.array([f"s{i}" for i in range(n_identities)])
        templates = rng.normal(size=(n_identities * k, 4))
        template_identities = np.repeat(identities, k)
        probes = rng.normal(size=(n_identities * 2, 4))
        probe_labels = np.repeat(identities, 2)
        return templates, template_identities, probes, probe_labels

    def test_valid_exact_k_is_accepted(self):
        templates, template_identities, probes, probe_labels = self._gallery(3)
        scores, labels_pair, _ = _generate_multi_template_verification_pairs(
            probe_embeddings=probes,
            probe_labels=probe_labels,
            probe_provenance=None,
            template_embeddings=templates,
            template_identities=template_identities,
            num_templates_per_identity=3,
            probe_fusion_size=1,
            matching_method="cosine",
            pair_sampling_mode="all",
        )
        self.assertGreater(len(scores), 0)
        self.assertEqual(len(scores), len(labels_pair))

    def test_equal_but_mismatched_k_is_rejected(self):
        # Every identity consistently has 2 rows, but the caller configured
        # K=3: a structural mismatch the row-count-equality check alone
        # would not catch.
        templates, template_identities, probes, probe_labels = self._gallery(2)
        with self.assertRaisesRegex(
            ValueError, r"num_templates_per_identity=3.*\bhas 2\b"
        ):
            _generate_multi_template_verification_pairs(
                probe_embeddings=probes,
                probe_labels=probe_labels,
                probe_provenance=None,
                template_embeddings=templates,
                template_identities=template_identities,
                num_templates_per_identity=3,
                probe_fusion_size=1,
                matching_method="cosine",
                pair_sampling_mode="all",
            )

    def test_unequal_counts_still_rejected_before_the_k_check(self):
        # Existing unequal-row-count guard must keep firing regardless of
        # the requested K.
        templates = np.random.default_rng(0).normal(size=(3, 4))
        identities = np.array(["a", "a", "b"])
        with self.assertRaisesRegex(ValueError, "same number"):
            _generate_multi_template_verification_pairs(
                probe_embeddings=templates,
                probe_labels=np.array(["a", "a", "b"]),
                probe_provenance=None,
                template_embeddings=templates,
                template_identities=identities,
                num_templates_per_identity=2,
                probe_fusion_size=1,
                matching_method="cosine",
                pair_sampling_mode="all",
            )

    def test_invalid_helper_k_rejected(self):
        templates, template_identities, probes, probe_labels = self._gallery(2)
        for bad_k in (2.7, 3.0, True, 0, -1, "2"):
            with self.subTest(bad_k=bad_k):
                with self.assertRaisesRegex(
                    ValueError, "num_templates_per_identity"
                ):
                    _generate_multi_template_verification_pairs(
                        probe_embeddings=probes,
                        probe_labels=probe_labels,
                        probe_provenance=None,
                        template_embeddings=templates,
                        template_identities=template_identities,
                        num_templates_per_identity=bad_k,
                        probe_fusion_size=1,
                        matching_method="cosine",
                        pair_sampling_mode="all",
                    )


class VerificationSamplingModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(11)
        cls.probes = rng.normal(size=(20, 4))
        cls.probe_labels = np.repeat(["a", "b", "c", "d"], 5)
        cls.templates = rng.normal(size=(8, 4))
        cls.template_identities = np.repeat(["a", "b", "c", "d"], 2)

    def _run(self, mode, **kwargs):
        return _generate_multi_template_verification_pairs(
            probe_embeddings=self.probes,
            probe_labels=self.probe_labels,
            probe_provenance=None,
            template_embeddings=self.templates,
            template_identities=self.template_identities,
            num_templates_per_identity=2,
            probe_fusion_size=1,
            matching_method="cosine",
            pair_sampling_mode=mode,
            **kwargs,
        )

    def test_all_mode_covers_every_group_identity_decision(self):
        scores, labels_pair, diagnostics = self._run("all")
        self.assertEqual(len(scores), 20 * 4)
        self.assertEqual(int(labels_pair.sum()), 20)
        self.assertEqual(diagnostics["impostor_fused_decisions"], 60)

    def test_all_genuine_caps_impostors_and_defaults_seed(self):
        scores, labels_pair, diagnostics = self._run(
            "all_genuine", max_impostor_pairs=10
        )
        self.assertEqual(diagnostics["genuine_fused_decisions"], 20)
        self.assertEqual(diagnostics["impostor_fused_decisions"], 10)
        self.assertEqual(diagnostics["total_impostor_fused_decisions"], 60)
        self.assertEqual(diagnostics["pair_sampling_seed"], 42)
        self.assertEqual(len(scores), 30)

    def test_all_genuine_is_reproducible_for_a_seed(self):
        first = self._run("all_genuine", max_impostor_pairs=10,
                          pair_sampling_seed=7)
        second = self._run("all_genuine", max_impostor_pairs=10,
                           pair_sampling_seed=7)
        np.testing.assert_array_equal(first[0], second[0])

    def test_balanced_mode_honors_budget_and_seed(self):
        scores, labels_pair, diagnostics = self._run(
            "balanced", pair_sampling_budget=40, pair_sampling_seed=5
        )
        self.assertEqual(diagnostics["genuine_fused_decisions"], 20)
        self.assertEqual(diagnostics["impostor_fused_decisions"], 20)
        repeat = self._run(
            "balanced", pair_sampling_budget=40, pair_sampling_seed=5
        )
        np.testing.assert_array_equal(scores, repeat[0])

    def test_balanced_mode_requires_budget(self):
        with self.assertRaisesRegex(ValueError, "pair_sampling_budget"):
            self._run("balanced")

    def test_random_mode_draws_exactly_the_budget(self):
        scores, labels_pair, _ = self._run(
            "random", pair_sampling_budget=25, pair_sampling_seed=3
        )
        self.assertEqual(len(scores), 25)
        for score, label in zip(scores, labels_pair):
            self.assertIn(label, (0, 1))

    def test_sampled_modes_stay_bounded_on_large_decision_spaces(self):
        # 1000 probe groups x 100 identities is a 100000-decision space;
        # all_genuine scores only the genuine decisions plus the impostor cap.
        rng = np.random.default_rng(2)
        identities = np.array([f"s{i}" for i in range(100)])
        probes = rng.normal(size=(1000, 4))
        probe_labels = np.repeat(identities, 10)
        templates = rng.normal(size=(200, 4))
        template_identities = np.repeat(identities, 2)
        scores, _, diagnostics = _generate_multi_template_verification_pairs(
            probe_embeddings=probes,
            probe_labels=probe_labels,
            probe_provenance=None,
            template_embeddings=templates,
            template_identities=template_identities,
            num_templates_per_identity=2,
            probe_fusion_size=1,
            matching_method="cosine",
            pair_sampling_mode="all_genuine",
            max_impostor_pairs=500,
        )
        self.assertEqual(len(scores), 1000 + 500)
        self.assertEqual(diagnostics["impostor_fused_decisions"], 500)


# =========================================================================
# Fusion mode is unchanged (backward compatibility)
# =========================================================================
class FusionModeBackwardCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = make_synthetic_ecg_dataset(
            number_of_subjects=8, samples_per_subject=12
        )
        cls.provenance = provenance_for(cls.y)
        cls.s1_x, cls.s1_y = make_synthetic_ecg_dataset(
            number_of_subjects=6, samples_per_subject=10, seed=456
        )
        cls.s2_x, cls.s2_y = make_synthetic_ecg_dataset(
            number_of_subjects=6, samples_per_subject=8,
            session_shift=0.1, seed=789
        )
        cls.p1 = provenance_for(cls.s1_y)
        cls.p2 = provenance_for(cls.s2_y)

    def _single_session_calls(self):
        return [
            (run.run_closed_set_identification,
             dict(test_split=0.5, use_template=True,
                  provenance=self.provenance)),
            (run.run_closed_set_verification,
             dict(test_split=0.5, use_template=True,
                  provenance=self.provenance)),
            (run.run_subject_disjoint_identification,
             dict(test_split=0.4, use_template=True, template_size=3,
                  provenance=self.provenance)),
            (run.run_subject_disjoint_verification,
             dict(test_split=0.4, use_template=True, template_size=3,
                  provenance=self.provenance)),
        ]

    def _cross_session_calls(self):
        return [
            (run.run_cross_session_identification,
             dict(use_template=True, template_size=4,
                  provenance_s1=self.p1, provenance_s2=self.p2)),
            (run.run_cross_session_verification,
             dict(use_template=True, template_size=4,
                  provenance_s1=self.p1, provenance_s2=self.p2)),
            (run.run_subject_disjoint_cross_session_identification,
             dict(test_split=0.4, use_template=True, template_size=4,
                  provenance_s1=self.p1, provenance_s2=self.p2)),
            (run.run_subject_disjoint_cross_session_verification,
             dict(test_split=0.4, use_template=True, template_size=4,
                  provenance_s1=self.p1, provenance_s2=self.p2)),
        ]

    def _invoke(self, runner, kwargs, **extra):
        arguments = dict(kwargs)
        arguments.update(COMMON)
        arguments.update(extra)
        if runner in (
            run.run_closed_set_identification,
            run.run_closed_set_verification,
            run.run_subject_disjoint_identification,
            run.run_subject_disjoint_verification,
        ):
            return runner(self.x, self.y, **arguments)
        return runner(
            self.s1_x, self.s1_y, self.s2_x, self.s2_y, **arguments
        )

    def test_omitted_mode_equals_explicit_fusion_for_every_runner(self):
        for runner, kwargs in (
            self._single_session_calls() + self._cross_session_calls()
        ):
            with self.subTest(runner=runner.__name__):
                metrics_omitted, stats_omitted, hyper_omitted = self._invoke(
                    runner, kwargs
                )
                metrics_fusion, stats_fusion, hyper_fusion = self._invoke(
                    runner, kwargs, enrollment_template_mode="fusion"
                )
                np.testing.assert_array_equal(
                    np.asarray(metrics_omitted, dtype=float),
                    np.asarray(metrics_fusion, dtype=float),
                )
                self.assertEqual(stats_omitted, stats_fusion)
                self.assertEqual(
                    hyper_omitted["enrollment_template_mode"], "fusion"
                )
                self.assertEqual(
                    hyper_omitted["template_fusion_method"], "mean"
                )
                self.assertNotIn("Enrollment Templates", stats_omitted)
                self.assertNotIn(
                    "num_templates_per_identity", hyper_omitted
                )

    def test_fusion_mode_never_touches_multi_template_helpers(self):
        def _forbidden(*args, **kwargs):
            raise AssertionError(
                "multi-template helper invoked in fusion mode"
            )

        with mock.patch.object(
            run, "_select_multi_templates", _forbidden
        ), mock.patch.object(
            run, "_generate_multi_template_verification_pairs", _forbidden
        ):
            for runner, kwargs in (
                self._single_session_calls() + self._cross_session_calls()
            ):
                with self.subTest(runner=runner.__name__):
                    metrics, _, _ = self._invoke(runner, kwargs)
                    self.assertTrue(
                        np.all(np.isfinite(
                            [m for m in np.ravel(
                                np.asarray(metrics, dtype=object)
                            ) if isinstance(m, float)]
                        ))
                    )

    def test_legacy_none_fusion_string_and_python_none_still_match(self):
        metrics_string, stats_string, _ = self._invoke(
            run.run_closed_set_identification,
            dict(test_split=0.5, use_template=True,
                 provenance=self.provenance),
            template_fusion_method="none",
        )
        metrics_none, stats_none, _ = self._invoke(
            run.run_closed_set_identification,
            dict(test_split=0.5, use_template=True,
                 provenance=self.provenance),
            template_fusion_method=None,
        )
        np.testing.assert_array_equal(
            np.asarray(metrics_string, dtype=float),
            np.asarray(metrics_none, dtype=float),
        )
        self.assertEqual(stats_string, stats_none)

    def test_fusion_mode_forwards_the_default_fusion_method(self):
        captured = {}
        original = utils._create_templates

        def _capture(embeddings, labels, method="mean", **kwargs):
            captured["method"] = method
            return original(embeddings, labels, method=method, **kwargs)

        with mock.patch.object(run, "_create_templates", _capture):
            self._invoke(
                run.run_closed_set_identification,
                dict(test_split=0.5, use_template=True,
                     provenance=self.provenance),
            )
        self.assertEqual(captured["method"], "mean")


class VerificationRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = make_synthetic_ecg_dataset(
            number_of_subjects=6, samples_per_subject=12
        )
        cls.provenance = provenance_for(cls.y)

    def _run_task2(self, patches, **extra):
        arguments = dict(
            test_split=0.5, use_template=True, provenance=self.provenance
        )
        arguments.update(COMMON)
        arguments.update(extra)
        calls = {name: 0 for name in patches}
        stack = []
        try:
            for name in patches:
                original = getattr(run, name)

                def _wrapper(*args, _name=name, _original=original, **kwargs):
                    calls[_name] += 1
                    return _original(*args, **kwargs)

                patcher = mock.patch.object(run, name, _wrapper)
                patcher.start()
                stack.append(patcher)
            run.run_closed_set_verification(self.x, self.y, **arguments)
        finally:
            for patcher in stack:
                patcher.stop()
        return calls

    def test_fusion_single_beat_routes_to_generate_pairs(self):
        calls = self._run_task2(
            [
                "_generate_pairs",
                "_generate_fused_verification_pairs",
                "_generate_multi_template_verification_pairs",
            ],
        )
        self.assertGreater(calls["_generate_pairs"], 0)
        self.assertEqual(calls["_generate_fused_verification_pairs"], 0)
        self.assertEqual(
            calls["_generate_multi_template_verification_pairs"], 0
        )

    def test_fusion_fused_probes_route_to_fused_pair_generator(self):
        calls = self._run_task2(
            [
                "_generate_fused_verification_pairs",
                "_generate_multi_template_verification_pairs",
            ],
            probe_fusion_size=2,
        )
        self.assertGreater(calls["_generate_fused_verification_pairs"], 0)
        self.assertEqual(
            calls["_generate_multi_template_verification_pairs"], 0
        )

    def test_multi_template_routes_to_the_dedicated_helper(self):
        calls = self._run_task2(
            [
                "_generate_pairs",
                "_generate_fused_verification_pairs",
                "_generate_multi_template_verification_pairs",
            ],
            enrollment_template_mode="multi_template",
            num_templates_per_identity=3,
        )
        self.assertEqual(
            calls["_generate_multi_template_verification_pairs"], 1
        )
        self.assertEqual(calls["_generate_pairs"], 0)
        self.assertEqual(calls["_generate_fused_verification_pairs"], 0)


# =========================================================================
# Multi-template end-to-end runner coverage
# =========================================================================
class MultiTemplateRunnerSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = make_synthetic_ecg_dataset(
            number_of_subjects=8, samples_per_subject=14
        )
        cls.provenance = provenance_for(cls.y)
        cls.s1_x, cls.s1_y = make_synthetic_ecg_dataset(
            number_of_subjects=6, samples_per_subject=10, seed=456
        )
        cls.s2_x, cls.s2_y = make_synthetic_ecg_dataset(
            number_of_subjects=6, samples_per_subject=8,
            session_shift=0.1, seed=789
        )
        cls.p1 = provenance_for(cls.s1_y)
        cls.p2 = provenance_for(cls.s2_y)

    def _assert_diagnostics(self, data_stats, hyperparams):
        diagnostics = data_stats["Enrollment Templates"]
        self.assertEqual(
            diagnostics["enrollment_template_mode"], "multi_template"
        )
        self.assertEqual(diagnostics["num_templates_per_identity"], 3)
        self.assertEqual(
            diagnostics["templates_selected_total"],
            diagnostics["enrolled_identities"] * 3,
        )
        self.assertTrue(diagnostics["deterministic"])
        self.assertRegex(
            diagnostics["selected_index_fingerprint"], r"^[0-9a-f]{64}$"
        )
        self.assertEqual(
            hyperparams["enrollment_template_mode"], "multi_template"
        )
        self.assertEqual(hyperparams["num_templates_per_identity"], 3)
        self.assertEqual(
            hyperparams["template_selection_method"],
            "farthest_first_cosine",
        )
        self.assertEqual(hyperparams["template_score_aggregation"], "max")
        self.assertIsNone(hyperparams["template_fusion_method"])

    def _assert_identification(self, result):
        (rank1, _rank5), data_stats, hyperparams = result
        self.assertTrue(0.0 <= rank1 <= 1.0)
        self._assert_diagnostics(data_stats, hyperparams)

    def _assert_verification(self, result):
        (eer, auc_value, _dprime, _tar), data_stats, hyperparams = result
        self.assertTrue(0.0 <= eer <= 1.0)
        self.assertTrue(0.0 <= auc_value <= 1.0)
        self._assert_diagnostics(data_stats, hyperparams)

    def test_task1_closed_set_identification(self):
        self._assert_identification(run.run_closed_set_identification(
            self.x, self.y, test_split=0.5, use_template=True,
            provenance=self.provenance,
            enrollment_template_mode="multi_template",
            num_templates_per_identity=3,
            **COMMON,
        ))

    def test_task2_closed_set_verification(self):
        self._assert_verification(run.run_closed_set_verification(
            self.x, self.y, test_split=0.5, use_template=True,
            provenance=self.provenance,
            enrollment_template_mode="multi_template",
            num_templates_per_identity=3,
            **COMMON,
        ))

    def test_task3_subject_disjoint_identification(self):
        self._assert_identification(
            run.run_subject_disjoint_identification(
                self.x, self.y, test_split=0.4, use_template=True,
                template_size=3, provenance=self.provenance,
                enrollment_template_mode="multi_template",
                num_templates_per_identity=3,
                **COMMON,
            )
        )

    def test_task4_subject_disjoint_verification(self):
        self._assert_verification(run.run_subject_disjoint_verification(
            self.x, self.y, test_split=0.4, use_template=True,
            template_size=3, provenance=self.provenance,
            enrollment_template_mode="multi_template",
            num_templates_per_identity=3,
            **COMMON,
        ))

    def test_task5_cross_session_identification(self):
        self._assert_identification(run.run_cross_session_identification(
            self.s1_x, self.s1_y, self.s2_x, self.s2_y,
            use_template=True, template_size=4,
            provenance_s1=self.p1, provenance_s2=self.p2,
            enrollment_template_mode="multi_template",
            num_templates_per_identity=3,
            **COMMON,
        ))

    def test_task6_cross_session_verification(self):
        self._assert_verification(run.run_cross_session_verification(
            self.s1_x, self.s1_y, self.s2_x, self.s2_y,
            use_template=True, template_size=4,
            provenance_s1=self.p1, provenance_s2=self.p2,
            enrollment_template_mode="multi_template",
            num_templates_per_identity=3,
            **COMMON,
        ))

    def test_task7_subject_disjoint_cross_session_identification(self):
        self._assert_identification(
            run.run_subject_disjoint_cross_session_identification(
                self.s1_x, self.s1_y, self.s2_x, self.s2_y,
                test_split=0.4, use_template=True, template_size=4,
                provenance_s1=self.p1, provenance_s2=self.p2,
                enrollment_template_mode="multi_template",
                num_templates_per_identity=3,
                **COMMON,
            )
        )

    def test_task8_subject_disjoint_cross_session_verification(self):
        self._assert_verification(
            run.run_subject_disjoint_cross_session_verification(
                self.s1_x, self.s1_y, self.s2_x, self.s2_y,
                test_split=0.4, use_template=True, template_size=4,
                provenance_s1=self.p1, provenance_s2=self.p2,
                enrollment_template_mode="multi_template",
                num_templates_per_identity=3,
                **COMMON,
            )
        )

    def test_probe_fusion_composes_with_multi_template_end_to_end(self):
        (rank1, _), data_stats, _ = run.run_closed_set_identification(
            self.x, self.y, test_split=0.5, use_template=True,
            provenance=self.provenance,
            enrollment_template_mode="multi_template",
            num_templates_per_identity=3,
            probe_fusion_size=2,
            **COMMON,
        )
        self.assertTrue(0.0 <= rank1 <= 1.0)
        self.assertEqual(data_stats["Probe Fusion"]["fusion_size"], 2)

    def test_deployment_evaluation_is_rejected_before_training(self):
        with self.assertRaisesRegex(
            ValueError, "use_deployment_evaluation"
        ):
            run.run_closed_set_verification(
                self.x, self.y, test_split=0.5, use_template=True,
                provenance=self.provenance,
                enrollment_template_mode="multi_template",
                num_templates_per_identity=3,
                use_deployment_evaluation=True,
                **COMMON,
            )

    def test_strict_k_failure_names_the_identities(self):
        with self.assertRaisesRegex(
            ValueError, "num_templates_per_identity=99"
        ):
            run.run_closed_set_identification(
                self.x, self.y, test_split=0.5, use_template=True,
                provenance=self.provenance,
                enrollment_template_mode="multi_template",
                num_templates_per_identity=99,
                **COMMON,
            )


# =========================================================================
# End-to-end proof that template_score_aggregation is actually routed to
# the reduction/verification helpers, and that identification runners pass
# the full canonical identity space rather than only the identities that
# happen to survive template selection.
# =========================================================================
class AggregationAndIdentitySpaceRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = make_synthetic_ecg_dataset(
            number_of_subjects=8, samples_per_subject=14
        )
        cls.provenance = provenance_for(cls.y)
        cls.s1_x, cls.s1_y = make_synthetic_ecg_dataset(
            number_of_subjects=6, samples_per_subject=10, seed=456
        )
        cls.s2_x, cls.s2_y = make_synthetic_ecg_dataset(
            number_of_subjects=6, samples_per_subject=8,
            session_shift=0.1, seed=789
        )
        cls.p1 = provenance_for(cls.s1_y)
        cls.p2 = provenance_for(cls.s2_y)

    @staticmethod
    def _spy(target, name):
        original = getattr(target, name)
        calls = []

        def _wrapper(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        return mock.patch.object(target, name, _wrapper), calls

    def test_task1_routes_aggregation_and_canonical_identity_space(self):
        reduce_patch, reduce_calls = self._spy(
            run, "_reduce_template_scores_to_identities"
        )
        encode_patch, encode_calls = self._spy(run, "_encode_labels")
        with reduce_patch, encode_patch:
            run.run_closed_set_identification(
                self.x, self.y, test_split=0.5, use_template=True,
                provenance=self.provenance,
                enrollment_template_mode="multi_template",
                num_templates_per_identity=3,
                **COMMON,
            )
        self.assertEqual(len(reduce_calls), 1)
        args, kwargs = reduce_calls[0]
        self.assertEqual(kwargs["template_score_aggregation"], "max")
        identity_order = args[2]
        classes = encode_calls[0][1]  # (y_train, classes) return, captured via args
        # _encode_labels(y_train) -> (encoded, classes); grab the actual
        # returned classes array from the wrapped call's return value.
        returned_classes = run._encode_labels(*encode_calls[0][0])[1]
        np.testing.assert_array_equal(
            identity_order, np.arange(len(returned_classes))
        )

    def test_task5_routes_aggregation_and_canonical_identity_space(self):
        reduce_patch, reduce_calls = self._spy(
            run, "_reduce_template_scores_to_identities"
        )
        encode_patch, encode_calls = self._spy(run, "_encode_labels")
        with reduce_patch, encode_patch:
            run.run_cross_session_identification(
                self.s1_x, self.s1_y, self.s2_x, self.s2_y,
                use_template=True, template_size=4,
                provenance_s1=self.p1, provenance_s2=self.p2,
                enrollment_template_mode="multi_template",
                num_templates_per_identity=3,
                **COMMON,
            )
        self.assertEqual(len(reduce_calls), 1)
        args, kwargs = reduce_calls[0]
        self.assertEqual(kwargs["template_score_aggregation"], "max")
        identity_order = args[2]
        returned_classes = run._encode_labels(*encode_calls[0][0])[1]
        np.testing.assert_array_equal(
            identity_order, np.arange(len(returned_classes))
        )

    def test_task7_routes_aggregation_and_canonical_identity_space(self):
        reduce_patch, reduce_calls = self._spy(
            run, "_reduce_template_scores_to_identities"
        )
        cohort_patch, cohort_calls = self._spy(run, "_split_subject_cohorts")
        with reduce_patch, cohort_patch:
            run.run_subject_disjoint_cross_session_identification(
                self.s1_x, self.s1_y, self.s2_x, self.s2_y,
                test_split=0.4, use_template=True, template_size=4,
                provenance_s1=self.p1, provenance_s2=self.p2,
                enrollment_template_mode="multi_template",
                num_templates_per_identity=3,
                **COMMON,
            )
        self.assertEqual(len(reduce_calls), 1)
        args, kwargs = reduce_calls[0]
        self.assertEqual(kwargs["template_score_aggregation"], "max")
        identity_order = args[2]
        _, _, returned_test_subs = run._split_subject_cohorts(
            *cohort_calls[0][0], **cohort_calls[0][1]
        )
        np.testing.assert_array_equal(
            identity_order, np.arange(len(returned_test_subs))
        )

    def test_task3_routes_aggregation_with_its_existing_canonical_order(self):
        reduce_patch, reduce_calls = self._spy(
            run, "_reduce_template_scores_to_identities"
        )
        with reduce_patch:
            (_, _), data_stats, _ = run.run_subject_disjoint_identification(
                self.x, self.y, test_split=0.4, use_template=True,
                template_size=3, provenance=self.provenance,
                enrollment_template_mode="multi_template",
                num_templates_per_identity=3,
                **COMMON,
            )
        self.assertEqual(len(reduce_calls), 1)
        args, kwargs = reduce_calls[0]
        self.assertEqual(kwargs["template_score_aggregation"], "max")
        identity_order = args[2]
        self.assertEqual(len(identity_order), data_stats["Test Subjects"])

    def test_all_four_verification_multi_template_branches_route_aggregation(
        self,
    ):
        verify_patch, verify_calls = self._spy(
            run, "_generate_multi_template_verification_pairs"
        )
        with verify_patch:
            run.run_closed_set_verification(
                self.x, self.y, test_split=0.5, use_template=True,
                provenance=self.provenance,
                enrollment_template_mode="multi_template",
                num_templates_per_identity=3,
                **COMMON,
            )
            run.run_subject_disjoint_verification(
                self.x, self.y, test_split=0.4, use_template=True,
                template_size=3, provenance=self.provenance,
                enrollment_template_mode="multi_template",
                num_templates_per_identity=3,
                **COMMON,
            )
            run.run_cross_session_verification(
                self.s1_x, self.s1_y, self.s2_x, self.s2_y,
                use_template=True, template_size=4,
                provenance_s1=self.p1, provenance_s2=self.p2,
                enrollment_template_mode="multi_template",
                num_templates_per_identity=3,
                **COMMON,
            )
            run.run_subject_disjoint_cross_session_verification(
                self.s1_x, self.s1_y, self.s2_x, self.s2_y,
                test_split=0.4, use_template=True, template_size=4,
                provenance_s1=self.p1, provenance_s2=self.p2,
                enrollment_template_mode="multi_template",
                num_templates_per_identity=3,
                **COMMON,
            )
        self.assertEqual(len(verify_calls), 4)
        for _, kwargs in verify_calls:
            self.assertEqual(kwargs["template_score_aggregation"], "max")
            self.assertEqual(kwargs["num_templates_per_identity"], 3)


# =========================================================================
# n_runs > 1 recursion: the _UNSET sentinel for template_fusion_method must
# survive the recursive single-seed re-invocation unresolved, and an
# explicit template_fusion_method=None in multi_template mode must stay
# rejected regardless of n_runs.
# =========================================================================
class NRunsSentinelRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = make_synthetic_ecg_dataset(
            number_of_subjects=8, samples_per_subject=14
        )
        cls.provenance = provenance_for(cls.y)

    def _spy_validator(self):
        original = run._validate_multi_template_arguments
        calls = []

        def _wrapper(*args, **kwargs):
            calls.append(args if args else kwargs)
            return original(*args, **kwargs)

        return mock.patch.object(
            run, "_validate_multi_template_arguments", _wrapper
        ), calls

    def test_identification_omitted_fusion_survives_n_runs_recursion(self):
        # Task 1, n_runs=2: the validator runs once per recursion layer (the
        # outer n_runs=2 dispatch plus each of the two recursive single-seed
        # calls), and template_fusion_method must still be the _UNSET
        # sentinel every single time -- never coerced into an explicit
        # legacy value that the multi_template validator would reject.
        patch_ctx, calls = self._spy_validator()
        common = dict(COMMON)
        common["n_runs"] = 2
        with patch_ctx:
            result = run.run_closed_set_identification(
                self.x, self.y, test_split=0.5, use_template=True,
                provenance=self.provenance,
                enrollment_template_mode="multi_template",
                num_templates_per_identity=3,
                **common,
            )
        self.assertEqual(len(calls), 3)
        for call_args in calls:
            observed_template_fusion_method = call_args[1]
            self.assertIs(observed_template_fusion_method, run._UNSET)
        # The aggregated multi-run rank metrics come back without raising --
        # if the sentinel had been coerced, the recursive single-seed calls
        # would have raised before any metric was produced.
        self.assertEqual(len(result), 2)

    def test_verification_omitted_fusion_survives_n_runs_recursion(self):
        patch_ctx, calls = self._spy_validator()
        common = dict(COMMON)
        common["n_runs"] = 2
        with patch_ctx:
            result = run.run_closed_set_verification(
                self.x, self.y, test_split=0.5, use_template=True,
                provenance=self.provenance,
                enrollment_template_mode="multi_template",
                num_templates_per_identity=3,
                **common,
            )
        self.assertEqual(len(calls), 3)
        for call_args in calls:
            observed_template_fusion_method = call_args[1]
            self.assertIs(observed_template_fusion_method, run._UNSET)
        self.assertEqual(len(result), 4)

    def test_explicit_none_rejected_before_any_recursive_execution(self):
        # Contrast case: an explicit template_fusion_method=None in
        # multi_template mode is a distinct, rejected state -- omitted and
        # explicit None must never collapse into the same behavior. The
        # rejection must happen at the outermost call, before entering the
        # n_runs recursion loop at all.
        patch_ctx, calls = self._spy_validator()
        common = dict(COMMON)
        common["n_runs"] = 2
        with patch_ctx:
            with self.assertRaisesRegex(
                ValueError, "template_fusion_method"
            ):
                run.run_closed_set_identification(
                    self.x, self.y, test_split=0.5, use_template=True,
                    provenance=self.provenance,
                    enrollment_template_mode="multi_template",
                    num_templates_per_identity=3,
                    template_fusion_method=None,
                    **common,
                )
        # Rejected on the very first (outermost) validation call -- no
        # recursive single-seed call, and therefore no training, was ever
        # attempted.
        self.assertEqual(len(calls), 1)

    def test_all_eight_runners_validate_before_recursion_capture(self):
        # Structural regression: in every runner, argument validation must
        # precede the n_runs>1 recursive-capture point, which must in turn
        # precede _UNSET resolution. Reassigning validated values before
        # capture is safe (the existing probe_fusion_size/pair_sampling
        # pattern already does this); resolving the sentinel into a new
        # value must happen strictly after, since only the deepest
        # single-run execution should perform that resolution.
        runner_names = [
            "run_closed_set_identification",
            "run_closed_set_verification",
            "run_subject_disjoint_identification",
            "run_subject_disjoint_verification",
            "run_cross_session_identification",
            "run_cross_session_verification",
            "run_subject_disjoint_cross_session_identification",
            "run_subject_disjoint_cross_session_verification",
        ]
        source = Path(run.__file__).read_text(encoding="utf-8")
        positions = sorted(
            (
                (name, source.find(f"def {name}("))
                for name in runner_names
            ),
            key=lambda pair: pair[1],
        )
        for index, (name, start) in enumerate(positions):
            end = (
                positions[index + 1][1]
                if index + 1 < len(positions)
                else len(source)
            )
            segment = source[start:end]
            with self.subTest(runner=name):
                validate_pos = segment.find(
                    "_validate_multi_template_arguments("
                )
                recursion_pos = segment.find("if n_runs > 1:")
                resolve_pos = segment.find(
                    "if template_fusion_method is _UNSET:"
                )
                self.assertNotEqual(validate_pos, -1)
                self.assertNotEqual(recursion_pos, -1)
                self.assertNotEqual(resolve_pos, -1)
                self.assertLess(validate_pos, recursion_pos)
                self.assertLess(recursion_pos, resolve_pos)


if __name__ == "__main__":
    unittest.main()
