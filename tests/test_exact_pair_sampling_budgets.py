import itertools
import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main
import run
import utils
from load_dataset import BeatProvenance
from test_all_task_smoke import TinyECGModel, make_synthetic_ecg_dataset
from test_identification_provenance import provenance_for


def pair_tuples(rows, columns):
    return list(zip(rows.tolist(), columns.tolist()))


def fused_provenance(labels):
    labels = np.asarray(labels)
    return BeatProvenance(
        {
            "record_id": np.asarray(
                [f"{label}_record" for label in labels],
                dtype=object,
            ),
            "session_id": np.asarray(
                [f"{label}_session" for label in labels],
                dtype=object,
            ),
            "acquisition_time": np.asarray(
                [None] * len(labels),
                dtype=object,
            ),
            "acquisition_order": np.zeros(len(labels), dtype=np.int64),
            "source_segment_id": np.asarray(
                [f"{label}_segment" for label in labels],
                dtype=object,
            ),
            "source_segment_order": np.zeros(len(labels), dtype=float),
            "beat_ordinal": np.tile(
                np.arange(2, dtype=np.int64),
                len(labels) // 2,
            ),
            "rpeak_index": np.full(len(labels), -1, dtype=np.int64),
        }
    )


class PairBudgetValidationTests(unittest.TestCase):
    def test_balanced_budget_must_be_even(self):
        with self.assertRaisesRegex(ValueError, "must be even"):
            utils._resolve_pair_sampling_arguments(
                pair_sampling_mode="balanced",
                pair_sampling_budget=3,
            )

    def test_cli_effective_configuration_rejects_odd_balanced_budget(self):
        captured_stderr = StringIO()
        with redirect_stderr(captured_stderr):
            with self.assertRaises(SystemExit) as raised:
                main.parse_experiment_arguments(
                    [
                        "--dataset",
                        "ecgid",
                        "--task",
                        "2",
                        "--pair_sampling_mode",
                        "balanced",
                        "--pair_sampling_budget",
                        "3",
                    ]
                )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("must be even", captured_stderr.getvalue())

    def test_positive_integer_policy_is_preserved(self):
        for value in (0, -1, True, 2.0, 2.5, "2"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    utils._resolve_pair_sampling_arguments(
                        pair_sampling_mode="random",
                        pair_sampling_budget=value,
                    )

        mode, budget = utils._resolve_pair_sampling_arguments(
            pair_sampling_mode="balanced",
            pair_sampling_budget=np.int64(4),
        )
        self.assertEqual((mode, budget), ("balanced", 4))

    def test_omitted_stochastic_budget_keeps_existing_default(self):
        self.assertEqual(
            utils._resolve_pair_sampling_arguments(
                pair_sampling_mode="random"
            ),
            ("random", 10000),
        )
        self.assertEqual(
            utils._resolve_pair_sampling_arguments(
                pair_sampling_mode="balanced"
            ),
            ("balanced", 10000),
        )


class ConceptualRankMappingTests(unittest.TestCase):
    def test_upper_triangle_boundaries_and_middle_ranks(self):
        expected = list(itertools.combinations(range(5), 2))
        observed = [
            utils._upper_triangle_rank_to_pair(rank, 5)
            for rank in range(len(expected))
        ]
        self.assertEqual(observed, expected)
        self.assertEqual(observed[0], (0, 1))
        self.assertEqual(observed[-1], (3, 4))
        self.assertEqual(observed[len(observed) // 2], expected[5])

    def test_upper_triangle_repeated_ranks_map_identically(self):
        rows, columns = utils._upper_triangle_ranks_to_pairs(
            np.asarray([0, 5, 9, 5], dtype=np.int64),
            5,
        )
        self.assertEqual(
            pair_tuples(rows, columns),
            [(0, 1), (1, 3), (3, 4), (1, 3)],
        )

    def test_cartesian_quotient_remainder_boundaries(self):
        rows, columns = utils._cartesian_ranks_to_pairs(
            np.asarray([0, 2, 3, 11], dtype=np.int64),
            3,
        )
        self.assertEqual(
            pair_tuples(rows, columns),
            [(0, 0), (0, 2), (1, 0), (3, 2)],
        )


class ExactRandomIndexSamplingTests(unittest.TestCase):
    def sample_same_set(self, item_count, budget, seed=7):
        return utils._sample_random_pair_indices(
            item_count,
            item_count,
            budget,
            np.random.default_rng(seed),
            match_two_sets=False,
        )

    def sample_two_set(self, rows, columns, budget, seed=7):
        return utils._sample_random_pair_indices(
            rows,
            columns,
            budget,
            np.random.default_rng(seed),
            match_two_sets=True,
        )

    def test_same_set_smaller_budget_is_distinct_and_valid(self):
        rows, columns, replacement = self.sample_same_set(6, 7)
        pairs = pair_tuples(rows, columns)
        self.assertEqual(len(pairs), 7)
        self.assertEqual(len(set(pairs)), 7)
        self.assertTrue(all(row < column for row, column in pairs))
        self.assertFalse(replacement)

    def test_same_set_full_universe_contains_every_pair_once(self):
        rows, columns, replacement = self.sample_same_set(5, 10)
        self.assertEqual(
            pair_tuples(rows, columns),
            list(itertools.combinations(range(5), 2)),
        )
        self.assertFalse(replacement)

    def test_same_set_over_budget_uses_replacement_and_stays_exact(self):
        rows, columns, replacement = self.sample_same_set(4, 20)
        pairs = pair_tuples(rows, columns)
        self.assertEqual(len(pairs), 20)
        self.assertLess(len(set(pairs)), 20)
        self.assertTrue(all(row < column for row, column in pairs))
        self.assertTrue(replacement)

    def test_two_sample_universe_repeats_the_only_valid_pair(self):
        rows, columns, replacement = self.sample_same_set(2, 5)
        self.assertEqual(pair_tuples(rows, columns), [(0, 1)] * 5)
        self.assertTrue(replacement)

    def test_empty_same_set_universe_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "universe is empty"):
            self.sample_same_set(1, 3)

    def test_same_seed_repeats_and_different_seed_can_differ(self):
        first = self.sample_same_set(20, 12, seed=19)
        repeat = self.sample_same_set(20, 12, seed=19)
        different = self.sample_same_set(20, 12, seed=20)
        np.testing.assert_array_equal(first[0], repeat[0])
        np.testing.assert_array_equal(first[1], repeat[1])
        self.assertFalse(
            np.array_equal(first[0], different[0])
            and np.array_equal(first[1], different[1])
        )

    def test_two_set_budget_relations(self):
        smaller = self.sample_two_set(3, 4, 5)
        equal = self.sample_two_set(3, 4, 12)
        larger = self.sample_two_set(3, 4, 20)

        self.assertEqual(len(smaller[0]), 5)
        self.assertEqual(len(set(pair_tuples(smaller[0], smaller[1]))), 5)
        self.assertFalse(smaller[2])
        self.assertEqual(
            set(pair_tuples(equal[0], equal[1])),
            set(itertools.product(range(3), range(4))),
        )
        self.assertFalse(equal[2])
        self.assertEqual(len(larger[0]), 20)
        self.assertLess(len(set(pair_tuples(larger[0], larger[1]))), 20)
        self.assertTrue(larger[2])

    def test_two_set_seed_repeatability_and_variability(self):
        first = self.sample_two_set(10, 8, 12, seed=71)
        repeat = self.sample_two_set(10, 8, 12, seed=71)
        different = self.sample_two_set(10, 8, 12, seed=72)
        np.testing.assert_array_equal(first[0], repeat[0])
        np.testing.assert_array_equal(first[1], repeat[1])
        self.assertFalse(
            np.array_equal(first[0], different[0])
            and np.array_equal(first[1], different[1])
        )

    def test_empty_two_set_universe_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "universe is empty"):
            self.sample_two_set(0, 4, 3)

    def test_large_same_set_small_budget_stays_lightweight(self):
        rows, columns, replacement = self.sample_same_set(
            1_000_000,
            8,
            seed=31,
        )
        self.assertEqual(rows.shape, (8,))
        self.assertTrue(np.all(rows < columns))
        self.assertFalse(replacement)


class ExactBalancedIndexSamplingTests(unittest.TestCase):
    def sample(self, labels, budget, seed=5):
        labels = np.asarray(labels)
        return utils._sample_balanced_pair_indices(
            labels,
            labels,
            budget,
            np.random.default_rng(seed),
            match_two_sets=False,
        )

    def test_sufficient_pools_use_no_replacement(self):
        rows, columns, classes, genuine_replacement, impostor_replacement = (
            self.sample([0, 0, 1, 1, 2, 2], 4)
        )
        pairs = pair_tuples(rows, columns)
        self.assertEqual(classes.tolist(), [1, 1, 0, 0])
        self.assertEqual(len(set(pairs[:2])), 2)
        self.assertEqual(len(set(pairs[2:])), 2)
        self.assertFalse(genuine_replacement)
        self.assertFalse(impostor_replacement)

    def test_only_small_genuine_pool_uses_replacement(self):
        rows, columns, classes, genuine_replacement, impostor_replacement = (
            self.sample([0, 0, 1], 4)
        )
        self.assertEqual(classes.tolist(), [1, 1, 0, 0])
        self.assertEqual(len(set(pair_tuples(rows[:2], columns[:2]))), 1)
        self.assertTrue(genuine_replacement)
        self.assertFalse(impostor_replacement)

    def test_only_small_impostor_pool_uses_replacement(self):
        result = self.sample([0, 0, 0, 0, 1], 10)
        rows, columns, classes, genuine_replacement, impostor_replacement = result
        self.assertEqual(int(classes.sum()), 5)
        self.assertEqual(int((classes == 0).sum()), 5)
        self.assertFalse(genuine_replacement)
        self.assertTrue(impostor_replacement)

    def test_both_small_pools_use_replacement(self):
        rows, columns, classes, genuine_replacement, impostor_replacement = (
            self.sample([0, 0, 1], 8)
        )
        self.assertEqual(classes.tolist(), [1] * 4 + [0] * 4)
        self.assertTrue(genuine_replacement)
        self.assertTrue(impostor_replacement)

    def test_missing_candidate_class_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "genuine comparison"):
            self.sample([0, 1, 2], 2)

        with self.assertRaisesRegex(ValueError, "impostor comparison"):
            self.sample([0, 0, 0], 2)

    def test_balanced_sampling_is_seed_deterministic(self):
        labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        first = self.sample(labels, 8, seed=41)
        repeat = self.sample(labels, 8, seed=41)
        different = self.sample(labels, 8, seed=42)
        np.testing.assert_array_equal(first[0], repeat[0])
        np.testing.assert_array_equal(first[1], repeat[1])
        self.assertFalse(
            np.array_equal(first[0], different[0])
            and np.array_equal(first[1], different[1])
        )

    def test_two_set_balanced_sampling_is_exact(self):
        labels1 = np.asarray([0, 0, 1, 1])
        labels2 = np.asarray([0, 1])
        result = utils._sample_balanced_pair_indices(
            labels1,
            labels2,
            6,
            np.random.default_rng(8),
            match_two_sets=True,
        )
        rows, columns, classes = result[:3]
        self.assertEqual(len(rows), 6)
        self.assertEqual(classes.tolist(), [1, 1, 1, 0, 0, 0])
        observed = (labels1[rows] == labels2[columns]).astype(int)
        np.testing.assert_array_equal(observed, classes)

    def test_same_set_impostor_rank_space_is_exhaustive(self):
        labels = np.asarray([0, 1, 0, 2])
        rows, columns, replacement = (
            utils._sample_impostor_pair_indices_exact(
                labels,
                labels,
                5,
                np.random.default_rng(3),
                match_two_sets=False,
            )
        )
        expected = {
            pair
            for pair in itertools.combinations(range(4), 2)
            if labels[pair[0]] != labels[pair[1]]
        }
        self.assertEqual(set(pair_tuples(rows, columns)), expected)
        self.assertFalse(replacement)

    def test_two_set_impostor_rank_space_is_exhaustive(self):
        labels1 = np.asarray([0, 1, 0])
        labels2 = np.asarray([1, 0])
        rows, columns, replacement = (
            utils._sample_impostor_pair_indices_exact(
                labels1,
                labels2,
                3,
                np.random.default_rng(4),
                match_two_sets=True,
            )
        )
        expected = {
            (row, column)
            for row in range(len(labels1))
            for column in range(len(labels2))
            if labels1[row] != labels2[column]
        }
        self.assertEqual(set(pair_tuples(rows, columns)), expected)
        self.assertFalse(replacement)


class ProductionPairGeneratorExactBudgetTests(unittest.TestCase):
    def setUp(self):
        self.embeddings = np.arange(24, dtype=float).reshape(8, 3)
        self.labels = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])

    def generate(self, mode, budget, seed=17, **kwargs):
        return utils._generate_pairs(
            self.embeddings,
            self.labels,
            pair_sampling_mode=mode,
            pair_sampling_budget=budget,
            pair_sampling_seed=seed,
            matching_method="euclidean",
            **kwargs,
        )

    def test_same_set_random_over_universe_is_exact(self):
        scores, labels_pair = self.generate("random", 50)
        self.assertEqual(len(scores), 50)
        self.assertEqual(len(labels_pair), 50)

    def test_two_set_random_over_universe_is_exact(self):
        scores, labels_pair = self.generate(
            "random",
            80,
            embeddings2=self.embeddings[:3],
            labels2=self.labels[:3],
        )
        self.assertEqual(len(scores), 80)
        self.assertEqual(len(labels_pair), 80)

    def test_balanced_generator_is_exact_and_equal(self):
        scores, labels_pair = self.generate("balanced", 40)
        self.assertEqual(len(scores), 40)
        self.assertEqual(int((labels_pair == 1).sum()), 20)
        self.assertEqual(int((labels_pair == 0).sum()), 20)

    def test_explicit_pair_seed_isolated_from_global_rng(self):
        np.random.seed(1)
        first = self.generate("random", 12, seed=91)
        np.random.seed(999)
        repeat = self.generate("random", 12, seed=91)
        different = self.generate("random", 12, seed=92)
        np.testing.assert_array_equal(first[0], repeat[0])
        np.testing.assert_array_equal(first[1], repeat[1])
        self.assertFalse(np.array_equal(first[0], different[0]))

    def test_all_and_all_genuine_frozen_counts(self):
        all_scores, all_labels = self.generate("all", 2)
        self.assertEqual(len(all_scores), 8 * 7 // 2)
        self.assertEqual(int((all_labels == 1).sum()), 4)

        capped_scores, capped_labels = utils._generate_pairs(
            self.embeddings,
            self.labels,
            pair_sampling_mode="all_genuine",
            max_impostor_pairs=3,
            pair_sampling_seed=17,
            matching_method="euclidean",
        )
        self.assertEqual(int((capped_labels == 1).sum()), 4)
        self.assertEqual(int((capped_labels == 0).sum()), 3)
        self.assertEqual(len(capped_scores), 7)


class FusedAndMultiTemplateExactBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = np.random.default_rng(101)
        cls.probe_labels = np.asarray(["a", "a", "b", "b"])
        cls.probes = generator.normal(size=(4, 5))
        cls.provenance = fused_provenance(cls.probe_labels)
        cls.single_templates = generator.normal(size=(2, 5))
        cls.single_identities = np.asarray(["a", "b"])
        cls.multi_templates = generator.normal(size=(4, 5))
        cls.multi_identities = np.asarray(["a", "a", "b", "b"])

    def fused(self, mode, budget):
        return utils._generate_fused_verification_pairs(
            self.probes,
            self.probe_labels,
            self.provenance,
            self.single_templates,
            self.single_identities,
            2,
            "cosine",
            pair_sampling_mode=mode,
            pair_sampling_budget=budget,
            pair_sampling_seed=23,
        )

    def multi(self, mode, budget):
        return utils._generate_multi_template_verification_pairs(
            self.probes,
            self.probe_labels,
            None,
            self.multi_templates,
            self.multi_identities,
            2,
            1,
            "cosine",
            pair_sampling_mode=mode,
            pair_sampling_budget=budget,
            pair_sampling_seed=23,
        )

    def test_probe_fused_random_and_balanced_are_exact(self):
        random_scores, random_labels, _ = self.fused("random", 9)
        balanced_scores, balanced_labels, diagnostics = self.fused(
            "balanced",
            8,
        )
        self.assertEqual(len(random_scores), 9)
        self.assertEqual(len(random_labels), 9)
        self.assertEqual(len(balanced_scores), 8)
        self.assertEqual(int((balanced_labels == 1).sum()), 4)
        self.assertEqual(int((balanced_labels == 0).sum()), 4)
        self.assertEqual(diagnostics["genuine_fused_decisions"], 4)
        self.assertEqual(diagnostics["impostor_fused_decisions"], 4)

    def test_multi_template_random_and_balanced_are_identity_exact(self):
        random_scores, random_labels, _ = self.multi("random", 12)
        balanced_scores, balanced_labels, diagnostics = self.multi(
            "balanced",
            12,
        )
        self.assertEqual(len(random_scores), 12)
        self.assertEqual(len(random_labels), 12)
        self.assertEqual(len(balanced_scores), 12)
        self.assertEqual(int((balanced_labels == 1).sum()), 6)
        self.assertEqual(int((balanced_labels == 0).sum()), 6)
        self.assertEqual(diagnostics["genuine_fused_decisions"], 6)
        self.assertEqual(diagnostics["impostor_fused_decisions"], 6)

    def test_odd_balanced_budget_rejected_by_both_paths(self):
        with self.assertRaisesRegex(ValueError, "must be even"):
            self.fused("balanced", 3)
        with self.assertRaisesRegex(ValueError, "must be even"):
            self.multi("balanced", 3)


class VerificationRunnerExactBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.x, cls.y = make_synthetic_ecg_dataset(
            number_of_subjects=6,
            samples_per_subject=12,
            signal_length=64,
            seed=515,
        )
        cls.provenance = provenance_for(cls.y)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def run_task2(self, **kwargs):
        return run.run_closed_set_verification(
            self.x,
            self.y,
            model_class=TinyECGModel,
            epochs=1,
            batch_size=16,
            lr=1e-3,
            test_split=0.25,
            val_split=0.0,
            seed=42,
            device="cpu",
            visualize=False,
            save_results_and_settings=False,
            loader=None,
            n_runs=1,
            _return_stats=True,
            intelligent_weight_loading=False,
            provenance=self.provenance,
            pair_sampling_seed=37,
            **kwargs,
        )

    def assert_total(self, result, expected):
        _, data_statistics, _ = result
        self.assertEqual(
            data_statistics["Total Verification Pairs"],
            expected,
        )

    def test_ordinary_same_set_and_two_set_paths_are_exact(self):
        self.assert_total(
            self.run_task2(
                use_template=False,
                pair_sampling_mode="random",
                pair_sampling_budget=200,
            ),
            200,
        )
        self.assert_total(
            self.run_task2(
                use_template=True,
                template_fusion_method="mean",
                template_size=2,
                pair_sampling_mode="balanced",
                pair_sampling_budget=200,
            ),
            200,
        )

    def test_probe_fused_path_is_exact(self):
        self.assert_total(
            self.run_task2(
                use_template=True,
                template_fusion_method="mean",
                template_size=2,
                probe_fusion_size=3,
                pair_sampling_mode="random",
                pair_sampling_budget=200,
            ),
            200,
        )

    def test_multi_template_identity_path_is_exact(self):
        self.assert_total(
            self.run_task2(
                use_template=True,
                enrollment_template_mode="multi_template",
                num_templates_per_identity=2,
                template_selection_method="farthest_first_cosine",
                template_score_aggregation="max",
                pair_sampling_mode="random",
                pair_sampling_budget=200,
            ),
            200,
        )


if __name__ == "__main__":
    unittest.main()
