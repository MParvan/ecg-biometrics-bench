"""
Identification tie policy: exact pessimistic ranks.

The rank of the correct identity is the number of gallery identities whose
score is greater than or equal to the correct identity's score. Ties are
therefore resolved against the correct identity, scores are compared exactly
as represented, and the rank cannot depend on gallery ordering or on which
column the correct identity occupies.

These tests exercise the production helper and the production Rank-k/CMC path
rather than a re-derivation of the formula.
"""

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import utils
from load_dataset import BeatProvenance
from utils import (
    _apply_score_fusion,
    _build_identification_curve_artifacts,
    _compute_metrics_identification,
    _compute_pessimistic_identification_ranks,
    _reduce_template_scores_to_identities,
)


def _ranks(scores, labels):
    """One-based pessimistic ranks straight from the production helper."""
    return _compute_pessimistic_identification_ranks(
        np.asarray(scores, dtype=float),
        np.asarray(labels),
    )


def _artifact_ranks(scores, labels):
    """Ranks as recorded by the production CMC/Rank-k artifact builder."""
    artifacts = _build_identification_curve_artifacts(
        np.asarray(scores, dtype=float),
        np.asarray(labels),
    )
    return artifacts["correct_match_ranks"], artifacts


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
# Enumerated tie and ordering edge cases
# =========================================================================
class PessimisticTieEdgeCases(unittest.TestCase):
    def test_unique_true_maximum_ranks_first(self):
        self.assertEqual(_ranks([[0.9, 0.8, 0.7]], [0]).tolist(), [1])

    def test_two_way_top_tie_ranks_second(self):
        scores = [[0.9, 0.9, 0.7]]
        # Pessimistic for either tied column, never rank 1.
        self.assertEqual(_ranks(scores, [0]).tolist(), [2])
        self.assertEqual(_ranks(scores, [1]).tolist(), [2])

    def test_three_way_tie_counts_every_tied_identity(self):
        scores = [[0.4, 0.4, 0.4, 0.1]]
        for true_column in (0, 1, 2):
            self.assertEqual(
                _ranks(scores, [true_column]).tolist(),
                [3],
                msg=f"true column {true_column}",
            )

    def test_one_strictly_higher_plus_tie(self):
        scores = [[0.95, 0.9, 0.9, 0.2]]
        self.assertEqual(_ranks(scores, [1]).tolist(), [3])
        self.assertEqual(_ranks(scores, [2]).tolist(), [3])

    def test_all_identities_tied_ranks_gallery_size(self):
        gallery_size = 5
        scores = [[0.5] * gallery_size]
        for true_column in range(gallery_size):
            self.assertEqual(
                _ranks(scores, [true_column]).tolist(),
                [gallery_size],
                msg=f"true column {true_column}",
            )

    def test_true_score_strictly_lowest_ranks_last(self):
        scores = [[0.9, 0.8, 0.7, 0.6]]
        self.assertEqual(_ranks(scores, [3]).tolist(), [4])

    def test_near_equal_lower_competitor_is_not_a_tie(self):
        lower = np.nextafter(0.8, -np.inf)
        self.assertLess(lower, 0.8)
        scores = [[0.8, lower, lower]]
        # Only the correct identity itself reaches its own score.
        self.assertEqual(_ranks(scores, [0]).tolist(), [1])

    def test_near_equal_higher_competitor_outranks_true_identity(self):
        higher = np.nextafter(0.8, np.inf)
        self.assertGreater(higher, 0.8)
        scores = [[0.8, higher]]
        self.assertEqual(_ranks(scores, [0]).tolist(), [2])

    def test_single_representable_step_separates_all_three_cases(self):
        lower = np.nextafter(0.8, -np.inf)
        higher = np.nextafter(0.8, np.inf)
        scores = [[0.8, lower, higher]]
        # Exactly one competitor is above; the near-equal lower one is not tied.
        self.assertEqual(_ranks(scores, [0]).tolist(), [2])

    def test_negative_scores_compare_exactly(self):
        scores = [[-0.5, -0.5, -0.9, -0.1]]
        self.assertEqual(_ranks(scores, [0]).tolist(), [3])
        self.assertEqual(_ranks(scores, [1]).tolist(), [3])
        self.assertEqual(_ranks(scores, [2]).tolist(), [4])
        self.assertEqual(_ranks(scores, [3]).tolist(), [1])

    def test_duplicate_scores_among_other_identities_stay_pessimistic(self):
        # Two other identities share a score below the correct identity.
        scores = [[0.9, 0.4, 0.4]]
        self.assertEqual(_ranks(scores, [0]).tolist(), [1])
        # ... and the same duplicate value at the correct identity ties.
        self.assertEqual(_ranks(scores, [1]).tolist(), [3])
        self.assertEqual(_ranks(scores, [2]).tolist(), [3])

    def test_rank_is_bounded_by_one_and_gallery_size(self):
        rng = np.random.RandomState(11)
        for _ in range(50):
            gallery_size = int(rng.randint(1, 9))
            probes = int(rng.randint(1, 7))
            scores = rng.randint(0, 3, size=(probes, gallery_size)).astype(float)
            labels = rng.randint(0, gallery_size, size=probes)
            observed = _ranks(scores, labels)
            self.assertTrue(np.all(observed >= 1))
            self.assertTrue(np.all(observed <= gallery_size))

    def test_single_identity_gallery_always_ranks_first(self):
        self.assertEqual(_ranks([[0.3], [-2.0]], [0, 0]).tolist(), [1, 1])


# =========================================================================
# Gallery ordering must not influence a tied rank
# =========================================================================
class GalleryOrderIndependence(unittest.TestCase):
    def test_permuting_gallery_columns_preserves_rank(self):
        rng = np.random.RandomState(2026)
        for _ in range(60):
            gallery_size = int(rng.randint(2, 9))
            # Coarse quantisation guarantees exact duplicate float values.
            row = rng.randint(0, 3, size=gallery_size).astype(float)
            true_column = int(rng.randint(0, gallery_size))
            baseline = _ranks([row], [true_column])[0]

            for _ in range(5):
                permutation = rng.permutation(gallery_size)
                permuted_row = row[permutation]
                moved_column = int(np.flatnonzero(permutation == true_column)[0])
                self.assertEqual(
                    _ranks([permuted_row], [moved_column])[0],
                    baseline,
                    msg=f"permutation {permutation.tolist()}",
                )

    def test_correct_identity_first_or_last_among_ties_scores_the_same(self):
        # The correct identity is the first tied column ...
        first = _ranks([[0.7, 0.7, 0.7, 0.1]], [0])[0]
        # ... and the last tied column.
        last = _ranks([[0.7, 0.7, 0.7, 0.1]], [2])[0]
        self.assertEqual(first, last)
        self.assertEqual(first, 3)

    def test_reversing_the_gallery_preserves_every_rank(self):
        scores = np.asarray(
            [
                [0.9, 0.9, 0.5, 0.2],
                [0.1, 0.3, 0.3, 0.3],
                [-1.0, -1.0, -1.0, -1.0],
            ],
            dtype=float,
        )
        labels = np.asarray([0, 2, 3])
        gallery_size = scores.shape[1]

        forward = _ranks(scores, labels)
        reversed_scores = scores[:, ::-1]
        reversed_labels = gallery_size - 1 - labels
        backward = _ranks(reversed_scores, reversed_labels)

        self.assertEqual(forward.tolist(), backward.tolist())


# =========================================================================
# Batch behaviour
# =========================================================================
class BatchRanking(unittest.TestCase):
    def test_each_probe_is_ranked_independently(self):
        scores = np.asarray(
            [
                [0.9, 0.8, 0.7],   # unique max at column 0
                [0.5, 0.5, 0.1],   # two-way tie
                [0.2, 0.2, 0.2],   # three-way tie
                [0.1, 0.4, 0.9],   # correct identity lowest
            ],
            dtype=float,
        )
        labels = np.asarray([0, 1, 0, 0])
        self.assertEqual(_ranks(scores, labels).tolist(), [1, 2, 3, 3])

    def test_batch_matches_per_row_evaluation(self):
        rng = np.random.RandomState(707)
        scores = rng.randint(0, 3, size=(25, 6)).astype(float)
        labels = rng.randint(0, 6, size=25)

        batched = _ranks(scores, labels)
        per_row = [
            _ranks(scores[index][None, :], [labels[index]])[0]
            for index in range(len(labels))
        ]
        self.assertEqual(batched.tolist(), per_row)


# =========================================================================
# Gallery-column contract for the correct-identity lookup
# =========================================================================
class TrueIdentityLookupContract(unittest.TestCase):
    def test_labels_outside_the_gallery_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "valid gallery columns"):
            _ranks([[0.5, 0.4]], [2])
        with self.assertRaisesRegex(ValueError, "valid gallery columns"):
            _ranks([[0.5, 0.4]], [-1])

    def test_probe_and_label_counts_must_agree(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            _ranks([[0.5, 0.4], [0.3, 0.2]], [0])

    def test_score_matrix_must_be_two_dimensional(self):
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            _ranks([0.5, 0.4], [0])

    def test_labels_must_be_one_dimensional(self):
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            _ranks([[0.5, 0.4]], [[0]])

    def test_noncontiguous_subject_identifiers_map_through_gallery_columns(self):
        """
        Runners hold arbitrary subject identifiers and build the score matrix
        with one column per identity, in a fixed identity order. The rank
        helper consumes those column indices, never the identifier values.
        """
        gallery_subjects = np.asarray([704, 12, 91], dtype=np.int64)
        # Column j scores identity gallery_subjects[j].
        scores = np.asarray(
            [
                [0.9, 0.9, 0.2],
                [0.1, 0.3, 0.3],
            ],
            dtype=float,
        )
        probe_subjects = np.asarray([704, 91], dtype=np.int64)

        # Runner-style remapping from identifier to gallery column.
        mapped = np.asarray(
            [
                int(np.flatnonzero(gallery_subjects == subject)[0])
                for subject in probe_subjects
            ]
        )
        self.assertEqual(mapped.tolist(), [0, 2])

        # Both probes tie at their correct identity, so both rank 2.
        self.assertEqual(_ranks(scores, mapped).tolist(), [2, 2])


# =========================================================================
# Rank-1, Rank-5 and CMC all derive from one rank vector
# =========================================================================
class RankKAndCmcConsistency(unittest.TestCase):
    def test_artifact_ranks_equal_the_canonical_helper(self):
        rng = np.random.RandomState(3131)
        for _ in range(40):
            probes = int(rng.randint(1, 10))
            gallery_size = int(rng.randint(1, 10))
            scores = rng.randint(0, 4, size=(probes, gallery_size)).astype(float)
            labels = rng.randint(0, gallery_size, size=probes)

            recorded, _ = _artifact_ranks(scores, labels)
            self.assertEqual(
                recorded,
                _ranks(scores, labels).tolist(),
            )

    def test_cmc_curve_is_exactly_the_mean_of_rank_at_most_k(self):
        scores = np.asarray(
            [
                [0.9, 0.9, 0.5, 0.5, 0.1, 0.0],
                [0.2, 0.4, 0.4, 0.4, 0.9, 0.3],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            ],
            dtype=float,
        )
        labels = np.asarray([0, 1, 4])

        recorded, artifacts = _artifact_ranks(scores, labels)
        recorded = np.asarray(recorded)

        curve = artifacts["cmc_curve"]
        for position, k in enumerate(curve["ranks"]):
            self.assertEqual(
                curve["identification_rates"][position],
                float(np.mean(recorded <= k)),
                msg=f"CMC rank {k}",
            )

    def test_rank_1_is_the_mean_of_rank_equal_to_one(self):
        scores = np.asarray(
            [
                [0.9, 0.2, 0.1],   # rank 1
                [0.5, 0.5, 0.1],   # tie -> rank 2, no longer a Rank-1 hit
                [0.7, 0.1, 0.0],   # rank 1
                [0.3, 0.3, 0.3],   # rank 3
            ],
            dtype=float,
        )
        labels = np.asarray([0, 0, 0, 0])

        recorded, artifacts = _artifact_ranks(scores, labels)
        self.assertEqual(recorded, [1, 2, 1, 3])
        self.assertEqual(artifacts["rank_1_accuracy"], 0.5)
        self.assertEqual(
            artifacts["rank_1_accuracy"],
            float(np.mean(np.asarray(recorded) <= 1)),
        )

    def test_rank_5_is_the_mean_of_rank_at_most_five(self):
        scores = np.asarray(
            [
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],   # correct identity top
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],   # all tied -> rank 6
            ],
            dtype=float,
        )
        labels = np.asarray([0, 0])

        recorded, artifacts = _artifact_ranks(scores, labels)
        self.assertEqual(recorded, [1, 6])
        self.assertTrue(artifacts["rank_5_reportable"])
        self.assertEqual(artifacts["rank_5_accuracy"], 0.5)
        self.assertEqual(
            artifacts["rank_5_accuracy"],
            float(np.mean(np.asarray(recorded) <= 5)),
        )

    def test_public_metric_wrapper_reports_the_pessimistic_rank_1(self):
        # Six identities so Rank-5 stays reportable.
        scores = np.asarray(
            [
                [0.5, 0.5, 0.1, 0.1, 0.1, 0.1],
                [0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
            ],
            dtype=float,
        )
        labels = np.asarray([0, 0])

        rank1, rank5 = _compute_metrics_identification(scores, labels)

        # The tied probe is not a Rank-1 hit under the pessimistic policy.
        self.assertEqual(rank1, 0.5)
        self.assertEqual(rank5, 1.0)

    def test_terminal_cmc_rate_is_one_for_every_probe(self):
        scores = np.asarray(
            [
                [0.4, 0.4, 0.4],
                [0.1, 0.9, 0.2],
            ],
            dtype=float,
        )
        labels = np.asarray([0, 0])
        _, artifacts = _artifact_ranks(scores, labels)
        self.assertEqual(
            artifacts["cmc_curve"]["identification_rates"][-1],
            1.0,
        )


# =========================================================================
# Small-gallery Rank-5 policy is untouched by the tie fix
# =========================================================================
class SmallGalleryRank5PolicyUnchanged(unittest.TestCase):
    def test_below_five_identities_rank_5_stays_unavailable(self):
        scores = np.asarray([[0.4, 0.4, 0.4, 0.1]], dtype=float)
        _, artifacts = _artifact_ranks(scores, [0])
        self.assertIsNone(artifacts["rank_5_accuracy"])
        self.assertFalse(artifacts["rank_5_defined"])
        self.assertFalse(artifacts["rank_5_reportable"])
        self.assertEqual(
            artifacts["rank_5_reportability_reason"],
            "gallery_size_below_5",
        )

    def test_exactly_five_identities_stays_defined_but_not_reportable(self):
        scores = np.asarray([[0.4, 0.4, 0.4, 0.4, 0.4]], dtype=float)
        _, artifacts = _artifact_ranks(scores, [0])
        self.assertTrue(artifacts["rank_5_defined"])
        self.assertFalse(artifacts["rank_5_reportable"])
        self.assertEqual(
            artifacts["rank_5_reportability_reason"],
            "terminal_cmc_rank",
        )


# =========================================================================
# Multi-template: identity-level reduction happens before ranking
# =========================================================================
class MultiTemplateIdentityLevelRanking(unittest.TestCase):
    def test_templates_are_not_ranked_as_separate_gallery_identities(self):
        """
        Three identities hold two templates each. After the per-identity
        maximum the correct identity ties one competitor, so the pessimistic
        rank must be 2 -- an identity count, not a template count.
        """
        template_identities = np.asarray([0, 0, 1, 1, 2, 2])
        identity_order = np.arange(3)

        # Per-template scores for one probe row.
        template_scores = np.asarray(
            [[0.10, 0.80, 0.80, 0.20, 0.30, 0.05]],
            dtype=float,
        )

        identity_scores = _reduce_template_scores_to_identities(
            template_scores,
            template_identities,
            identity_order,
        )
        # max over each identity's templates
        self.assertEqual(identity_scores.tolist(), [[0.80, 0.80, 0.30]])
        self.assertEqual(identity_scores.shape[1], len(identity_order))

        ranks = _ranks(identity_scores, [0])
        self.assertEqual(ranks.tolist(), [2])
        # Six templates exist, but the rank can never exceed identity count.
        self.assertLessEqual(int(ranks[0]), len(identity_order))

    def test_extra_templates_that_do_not_change_the_maximum_do_not_change_rank(
        self,
    ):
        identity_order = np.arange(3)

        few_identities = np.asarray([0, 1, 2])
        few_scores = np.asarray([[0.8, 0.8, 0.3]], dtype=float)

        # Identity 0 gains two more templates, all scoring below its maximum.
        many_identities = np.asarray([0, 0, 0, 1, 2])
        many_scores = np.asarray([[0.8, 0.1, 0.4, 0.8, 0.3]], dtype=float)

        few_reduced = _reduce_template_scores_to_identities(
            few_scores, few_identities, identity_order
        )
        many_reduced = _reduce_template_scores_to_identities(
            many_scores, many_identities, identity_order
        )

        self.assertEqual(few_reduced.tolist(), many_reduced.tolist())
        self.assertEqual(
            _ranks(few_reduced, [0]).tolist(),
            _ranks(many_reduced, [0]).tolist(),
        )
        self.assertEqual(_ranks(many_reduced, [0]).tolist(), [2])

    def test_reordering_templates_within_an_identity_does_not_change_rank(self):
        identity_order = np.arange(3)
        identities = np.asarray([0, 0, 1, 2])

        scores = np.asarray([[0.2, 0.8, 0.8, 0.3]], dtype=float)
        swapped = np.asarray([[0.8, 0.2, 0.8, 0.3]], dtype=float)

        reduced = _reduce_template_scores_to_identities(
            scores, identities, identity_order
        )
        reduced_swapped = _reduce_template_scores_to_identities(
            swapped, identities, identity_order
        )

        self.assertEqual(reduced.tolist(), reduced_swapped.tolist())
        self.assertEqual(
            _ranks(reduced, [0]).tolist(),
            _ranks(reduced_swapped, [0]).tolist(),
        )

    def test_multi_template_without_ties_is_unaffected(self):
        identity_order = np.arange(3)
        identities = np.asarray([0, 0, 1, 1, 2, 2])
        template_scores = np.asarray(
            [[0.10, 0.90, 0.50, 0.20, 0.30, 0.05]],
            dtype=float,
        )
        reduced = _reduce_template_scores_to_identities(
            template_scores, identities, identity_order
        )
        self.assertEqual(reduced.tolist(), [[0.90, 0.50, 0.30]])
        self.assertEqual(_ranks(reduced, [0]).tolist(), [1])
        self.assertEqual(_ranks(reduced, [1]).tolist(), [2])
        self.assertEqual(_ranks(reduced, [2]).tolist(), [3])


# =========================================================================
# Probe fusion feeds the same canonical rank
# =========================================================================
class ProbeFusionTieRanking(unittest.TestCase):
    def test_fused_score_vector_with_a_tie_ranks_pessimistically(self):
        # Two probe beats of subject 0, fused in one group of two.
        scores = np.asarray(
            [
                [0.6, 0.4, 0.2],
                [0.4, 0.6, 0.2],
            ],
            dtype=float,
        )
        labels = np.asarray([0, 0])
        provenance = _custom_provenance(["r0", "r0"], [0, 1])

        fused_scores, fused_labels = _apply_score_fusion(
            scores,
            labels,
            fusion_size=2,
            provenance=provenance,
        )

        # Averaging produces an exact tie between identities 0 and 1.
        self.assertEqual(fused_scores.shape, (1, 3))
        self.assertEqual(fused_labels.tolist(), [0])
        self.assertEqual(fused_scores[0, 0], fused_scores[0, 1])

        ranks = _ranks(fused_scores, fused_labels)
        self.assertEqual(ranks.tolist(), [2])

        recorded, artifacts = _artifact_ranks(fused_scores, fused_labels)
        self.assertEqual(recorded, [2])
        self.assertEqual(artifacts["rank_1_accuracy"], 0.0)

    def test_fusion_grouping_is_unchanged_by_the_tie_policy(self):
        scores = np.asarray(
            [
                [0.9, 0.1, 0.0],
                [0.7, 0.3, 0.0],
                [0.5, 0.5, 0.0],
            ],
            dtype=float,
        )
        labels = np.asarray([0, 0, 0])
        provenance = _custom_provenance(["r0", "r0", "r0"], [0, 1, 2])

        fused_scores, fused_labels = _apply_score_fusion(
            scores,
            labels,
            fusion_size=2,
            provenance=provenance,
        )

        # The trailing incomplete remainder is still dropped.
        self.assertEqual(len(fused_labels), 1)
        self.assertEqual(fused_scores[0].tolist(), [0.8, 0.2, 0.0])
        self.assertEqual(_ranks(fused_scores, fused_labels).tolist(), [1])

    def test_unfused_scoring_is_unchanged(self):
        scores = np.asarray([[0.9, 0.5], [0.4, 0.4]], dtype=float)
        labels = np.asarray([0, 0])

        fused_scores, fused_labels = _apply_score_fusion(
            scores,
            labels,
            fusion_size=1,
        )
        self.assertEqual(fused_scores.tolist(), scores.tolist())
        self.assertEqual(fused_labels.tolist(), labels.tolist())
        self.assertEqual(_ranks(fused_scores, fused_labels).tolist(), [1, 2])


# =========================================================================
# Structural guarantees about the tie policy itself
# =========================================================================
class TiePolicyStructure(unittest.TestCase):
    def test_rank_path_uses_no_tolerance_based_comparison(self):
        source = inspect.getsource(_compute_pessimistic_identification_ranks)
        for forbidden in ("isclose", "allclose", "atol", "rtol", "epsilon"):
            self.assertNotIn(forbidden, source, msg=forbidden)

    def test_rank_path_does_not_sort_the_gallery(self):
        source = inspect.getsource(_compute_pessimistic_identification_ranks)
        for forbidden in ("argsort", "np.sort", ".sort("):
            self.assertNotIn(forbidden, source, msg=forbidden)

    def test_artifact_builder_delegates_to_the_canonical_helper(self):
        source = inspect.getsource(_build_identification_curve_artifacts)
        self.assertIn(
            "_compute_pessimistic_identification_ranks",
            source,
        )
        # The previous sorting-based rank derivation is gone.
        self.assertNotIn("argsort", source)

    def test_rank_1_and_rank_5_share_the_artifact_rank_vector(self):
        source = inspect.getsource(_compute_metrics_identification)
        self.assertIn("_build_identification_curve_artifacts", source)
        self.assertNotIn("argsort", source)

    def test_only_one_identification_rank_definition_exists(self):
        """No other production helper derives an identification rank."""
        module_source = inspect.getsource(utils)
        self.assertEqual(
            module_source.count("def _compute_pessimistic_identification_ranks"),
            1,
        )


if __name__ == "__main__":
    unittest.main()
