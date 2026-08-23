import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

import utils


def brute_force_intra_impostors(labels):
    labels = np.asarray(labels)

    return {
        (left, right)
        for left in range(len(labels))
        for right in range(
            left + 1,
            len(labels),
        )
        if labels[left] != labels[right]
    }


def brute_force_two_set_impostors(
    labels1,
    labels2,
):
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)

    return {
        (row, column)
        for row in range(len(labels1))
        for column in range(len(labels2))
        if labels1[row] != labels2[column]
    }


def sampled_pairs(rows, columns):
    return set(
        zip(
            rows.tolist(),
            columns.tolist(),
        )
    )


class ExhaustiveMappingTests(
    unittest.TestCase
):
    def test_intra_set_mapping_matches_brute_force(self):
        label_cases = [
            [0, 0, 1, 1],
            [0, 1, 0, 2, 1],
            [0, 0, 0, 1, 2, 2],
            [0, 1, 2, 3, 3, 3, 3, 3],
            ["a", "b", "a", "c", "b"],
        ]

        for labels in label_cases:
            with self.subTest(
                labels=labels
            ):
                labels = np.asarray(labels)

                expected = (
                    brute_force_intra_impostors(
                        labels
                    )
                )

                rows, columns, total = (
                    utils._sample_impostor_pair_indices(
                        labels,
                        max_impostor_pairs=100000,
                        pair_sampling_seed=42,
                    )
                )

                self.assertEqual(
                    total,
                    len(expected),
                )

                self.assertEqual(
                    sampled_pairs(
                        rows,
                        columns,
                    ),
                    expected,
                )

    def test_two_set_mapping_matches_brute_force(self):
        cases = [
            (
                [0, 0, 1],
                [0, 1, 1, 2],
            ),
            (
                [0, 2, 2, 3],
                [1, 2, 4],
            ),
            (
                ["a", "b", "a"],
                ["b", "c", "a", "c"],
            ),
        ]

        for labels1, labels2 in cases:
            with self.subTest(
                labels1=labels1,
                labels2=labels2,
            ):
                labels1 = np.asarray(
                    labels1
                )

                labels2 = np.asarray(
                    labels2
                )

                expected = (
                    brute_force_two_set_impostors(
                        labels1,
                        labels2,
                    )
                )

                rows, columns, total = (
                    utils._sample_impostor_pair_indices(
                        labels1,
                        labels2,
                        max_impostor_pairs=100000,
                        pair_sampling_seed=42,
                    )
                )

                self.assertEqual(
                    total,
                    len(expected),
                )

                self.assertEqual(
                    sampled_pairs(
                        rows,
                        columns,
                    ),
                    expected,
                )


class FilteredOffsetMappingTests(
    unittest.TestCase
):
    def test_mapping_skips_forbidden_positions(self):
        result = (
            utils._map_offsets_excluding_indices(
                offsets=np.asarray(
                    [0, 1]
                ),
                forbidden_indices=np.asarray(
                    [0, 2]
                ),
                lower_bound=0,
                upper_bound=5,
            )
        )

        np.testing.assert_array_equal(
            result,
            np.asarray(
                [1, 3]
            ),
        )

    def test_mapping_handles_boundary_and_outside_forbidden_indices(self):
        result = (
            utils._map_offsets_excluding_indices(
                offsets=np.asarray(
                    [0, 3]
                ),
                forbidden_indices=np.asarray(
                    [0, 3, 7, 9]
                ),
                lower_bound=2,
                upper_bound=8,
            )
        )

        # Active range: [2, 8)
        # Forbidden inside it: 3 and 7
        # Allowed: 2, 4, 5, 6
        np.testing.assert_array_equal(
            result,
            np.asarray(
                [2, 6]
            ),
        )

    def test_mapping_with_no_forbidden_indices_is_direct(self):
        result = (
            utils._map_offsets_excluding_indices(
                offsets=np.asarray(
                    [0, 2, 4]
                ),
                forbidden_indices=np.asarray(
                    [],
                    dtype=np.int64,
                ),
                lower_bound=5,
                upper_bound=10,
            )
        )

        np.testing.assert_array_equal(
            result,
            np.asarray(
                [5, 7, 9]
            ),
        )


class CappedSamplingTests(
    unittest.TestCase
):
    def test_intra_set_sample_is_unique_and_valid(self):
        labels = np.asarray(
            [0] * 9
            + [1] * 3
            + [2] * 2
            + [3]
        )

        expected = (
            brute_force_intra_impostors(
                labels
            )
        )

        rows, columns, total = (
            utils._sample_impostor_pair_indices(
                labels,
                max_impostor_pairs=17,
                pair_sampling_seed=17,
            )
        )

        observed = sampled_pairs(
            rows,
            columns,
        )

        self.assertEqual(
            total,
            len(expected),
        )

        self.assertEqual(
            len(observed),
            17,
        )

        self.assertTrue(
            observed.issubset(
                expected
            )
        )

        self.assertTrue(
            np.all(
                rows < columns
            )
        )

    def test_two_set_sample_is_unique_and_valid(self):
        labels1 = np.asarray(
            [0] * 8
            + [1] * 2
            + [2]
        )

        labels2 = np.asarray(
            [0, 1, 1, 2, 3, 3]
        )

        expected = (
            brute_force_two_set_impostors(
                labels1,
                labels2,
            )
        )

        rows, columns, total = (
            utils._sample_impostor_pair_indices(
                labels1,
                labels2,
                max_impostor_pairs=13,
                pair_sampling_seed=91,
            )
        )

        observed = sampled_pairs(
            rows,
            columns,
        )

        self.assertEqual(
            total,
            len(expected),
        )

        self.assertEqual(
            len(observed),
            13,
        )

        self.assertTrue(
            observed.issubset(
                expected
            )
        )

    def test_cap_equal_to_population_returns_every_impostor(self):
        labels = np.asarray(
            [
                0,
                0,
                1,
                2,
            ]
        )

        expected = (
            brute_force_intra_impostors(
                labels
            )
        )

        rows, columns, total = (
            utils._sample_impostor_pair_indices(
                labels,
                max_impostor_pairs=len(
                    expected
                ),
                pair_sampling_seed=42,
            )
        )

        self.assertEqual(
            total,
            len(expected),
        )

        self.assertEqual(
            sampled_pairs(
                rows,
                columns,
            ),
            expected,
        )

    def test_all_same_identity_has_no_impostors(self):
        labels = np.zeros(
            12,
            dtype=int,
        )

        rows, columns, total = (
            utils._sample_impostor_pair_indices(
                labels,
                max_impostor_pairs=10,
                pair_sampling_seed=42,
            )
        )

        self.assertEqual(
            total,
            0,
        )

        self.assertEqual(
            len(rows),
            0,
        )

        self.assertEqual(
            len(columns),
            0,
        )


class PairSamplingSeedTests(
    unittest.TestCase
):
    def test_same_pair_seed_is_reproducible(self):
        labels = np.repeat(
            np.arange(8),
            4,
        )

        first = (
            utils._sample_impostor_pair_indices(
                labels,
                max_impostor_pairs=30,
                pair_sampling_seed=123,
            )
        )

        second = (
            utils._sample_impostor_pair_indices(
                labels,
                max_impostor_pairs=30,
                pair_sampling_seed=123,
            )
        )

        np.testing.assert_array_equal(
            first[0],
            second[0],
        )

        np.testing.assert_array_equal(
            first[1],
            second[1],
        )

        self.assertEqual(
            first[2],
            second[2],
        )

    def test_pair_seed_is_independent_of_global_numpy_state(self):
        labels = np.repeat(
            np.arange(8),
            4,
        )

        np.random.seed(1)

        first = (
            utils._sample_impostor_pair_indices(
                labels,
                max_impostor_pairs=30,
                pair_sampling_seed=123,
            )
        )

        np.random.seed(999)

        _ = np.random.random(
            1000
        )

        second = (
            utils._sample_impostor_pair_indices(
                labels,
                max_impostor_pairs=30,
                pair_sampling_seed=123,
            )
        )

        np.testing.assert_array_equal(
            first[0],
            second[0],
        )

        np.testing.assert_array_equal(
            first[1],
            second[1],
        )

    def test_different_pair_seeds_change_a_capped_sample(self):
        labels = np.repeat(
            np.arange(10),
            4,
        )

        first = (
            utils._sample_impostor_pair_indices(
                labels,
                max_impostor_pairs=25,
                pair_sampling_seed=1,
            )
        )

        second = (
            utils._sample_impostor_pair_indices(
                labels,
                max_impostor_pairs=25,
                pair_sampling_seed=2,
            )
        )

        self.assertNotEqual(
            sampled_pairs(
                first[0],
                first[1],
            ),
            sampled_pairs(
                second[0],
                second[1],
            ),
        )


class ImplicitLargeSpaceTests(
    unittest.TestCase
):
    def test_billion_scale_pair_population_is_not_materialized(self):
        # Unique identities make every unordered pair an impostor.
        # 50,000 observations imply 1,249,975,000 candidate pairs.
        labels = np.arange(
            50000,
            dtype=np.int64,
        )

        rows, columns, total = (
            utils._sample_impostor_pair_indices(
                labels,
                max_impostor_pairs=128,
                pair_sampling_seed=42,
            )
        )

        self.assertEqual(
            total,
            (
                len(labels)
                * (
                    len(labels) - 1
                )
                // 2
            ),
        )

        self.assertGreater(
            total,
            1_000_000_000,
        )

        self.assertEqual(
            len(rows),
            128,
        )

        self.assertEqual(
            len(columns),
            128,
        )

        self.assertEqual(
            len(
                sampled_pairs(
                    rows,
                    columns,
                )
            ),
            128,
        )

        self.assertTrue(
            np.all(
                rows < columns
            )
        )


class UniqueRankSamplingTests(
    unittest.TestCase
):
    def test_floyd_sampler_returns_valid_unique_ranks(self):
        rng = np.random.default_rng(
            42
        )

        ranks = (
            utils._sample_unique_integer_ranks(
                population_size=1000000000,
                sample_size=200,
                rng=rng,
            )
        )

        self.assertEqual(
            len(ranks),
            200,
        )

        self.assertEqual(
            len(
                np.unique(ranks)
            ),
            200,
        )

        self.assertTrue(
            np.all(
                ranks >= 0
            )
        )

        self.assertTrue(
            np.all(
                ranks < 1000000000
            )
        )

    def test_sampling_entire_population_returns_all_ranks(self):
        rng = np.random.default_rng(
            42
        )

        ranks = (
            utils._sample_unique_integer_ranks(
                population_size=7,
                sample_size=7,
                rng=rng,
            )
        )

        np.testing.assert_array_equal(
            ranks,
            np.arange(7),
        )


class ParameterValidationTests(
    unittest.TestCase
):
    def test_invalid_impostor_cap_is_rejected(self):
        for invalid in (
            0,
            -1,
            1.5,
            True,
        ):
            with self.subTest(
                invalid=invalid
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "positive integer",
                ):
                    utils._sample_impostor_pair_indices(
                        np.asarray(
                            [0, 1]
                        ),
                        max_impostor_pairs=invalid,
                    )

    def test_invalid_pair_seed_is_rejected(self):
        for invalid in (
            -1,
            1.5,
            True,
        ):
            with self.subTest(
                invalid=invalid
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "non-negative integer",
                ):
                    utils._sample_impostor_pair_indices(
                        np.asarray(
                            [0, 1]
                        ),
                        pair_sampling_seed=invalid,
                    )

    def test_non_vector_labels_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "one-dimensional",
        ):
            utils._sample_impostor_pair_indices(
                np.asarray(
                    [[0, 1]]
                )
            )


class ActivatedAllGenuineModeTests(
    unittest.TestCase
):
    @staticmethod
    def make_embeddings(count):
        values = np.arange(
            count * 4,
            dtype=float,
        ).reshape(
            count,
            4,
        )

        return (
            values
            + 1.0
        )

    def test_intra_set_retains_all_genuine_and_caps_impostors(self):
        labels = np.asarray(
            [
                0,
                0,
                1,
                1,
                1,
                2,
            ]
        )

        scores, pair_labels = (
            utils._generate_pairs(
                self.make_embeddings(
                    len(labels)
                ),
                labels,
                pair_sampling_mode=(
                    "all_genuine"
                ),
                max_impostor_pairs=5,
                pair_sampling_seed=42,
            )
        )

        # Genuine count:
        # C(2,2) + C(3,2) + C(1,2) = 4.
        self.assertEqual(
            int(
                np.sum(
                    pair_labels == 1
                )
            ),
            4,
        )

        self.assertEqual(
            int(
                np.sum(
                    pair_labels == 0
                )
            ),
            5,
        )

        self.assertEqual(
            len(scores),
            9,
        )

        self.assertTrue(
            np.isfinite(scores).all()
        )

    def test_two_set_retains_every_genuine_cartesian_cell(self):
        probe_labels = np.asarray(
            [
                0,
                0,
                1,
                2,
            ]
        )

        enrollment_labels = np.asarray(
            [
                0,
                1,
                1,
                3,
            ]
        )

        scores, pair_labels = (
            utils._generate_pairs(
                self.make_embeddings(
                    len(probe_labels)
                ),
                probe_labels,
                embeddings2=(
                    self.make_embeddings(
                        len(
                            enrollment_labels
                        )
                    )
                ),
                labels2=(
                    enrollment_labels
                ),
                pair_sampling_mode=(
                    "all_genuine"
                ),
                max_impostor_pairs=3,
                pair_sampling_seed=42,
            )
        )

        # Genuine cells:
        # probe label 0: 2 * 1
        # probe label 1: 1 * 2
        self.assertEqual(
            int(
                np.sum(
                    pair_labels == 1
                )
            ),
            4,
        )

        self.assertEqual(
            int(
                np.sum(
                    pair_labels == 0
                )
            ),
            3,
        )

        self.assertEqual(
            len(scores),
            7,
        )

    def test_all_genuine_does_not_materialize_full_score_matrix(self):
        labels = np.asarray(
            [
                0,
                0,
                1,
                1,
                2,
                2,
            ]
        )

        with patch(
            "utils._compute_score_matrix",
            side_effect=AssertionError(
                "full comparison matrix was materialized"
            ),
        ):
            scores, pair_labels = (
                utils._generate_pairs(
                    self.make_embeddings(
                        len(labels)
                    ),
                    labels,
                    pair_sampling_mode=(
                        "all_genuine"
                    ),
                    max_impostor_pairs=5,
                    pair_sampling_seed=42,
                )
            )

        self.assertEqual(
            len(scores),
            len(pair_labels),
        )

        self.assertGreater(
            len(scores),
            0,
        )


class AllGenuineExhaustiveEquivalenceTests(
    unittest.TestCase
):
    @staticmethod
    def assert_score_populations_equal(
        exhaustive_scores,
        exhaustive_labels,
        bounded_scores,
        bounded_labels,
    ):
        for pair_class in (
            0,
            1,
        ):
            expected = np.sort(
                exhaustive_scores[
                    exhaustive_labels
                    == pair_class
                ]
            )

            observed = np.sort(
                bounded_scores[
                    bounded_labels
                    == pair_class
                ]
            )

            np.testing.assert_allclose(
                observed,
                expected,
                rtol=1e-9,
                atol=1e-10,
                equal_nan=True,
            )

    def test_intra_set_matches_exhaustive_all_when_cap_covers_population(self):
        embeddings = np.asarray(
            [
                [1.0, 0.2, 0.4, 0.8],
                [0.9, 0.3, 0.5, 0.7],
                [0.1, 0.9, 0.3, 0.6],
                [0.2, 1.0, 0.4, 0.5],
                [0.7, 0.2, 1.1, 0.4],
                [0.6, 0.3, 0.9, 0.2],
            ]
        )

        labels = np.asarray(
            [
                0,
                0,
                1,
                1,
                2,
                2,
            ]
        )

        for method in (
            "cosine",
            "euclidean",
            "manhattan",
            "correlation",
        ):
            with self.subTest(
                method=method
            ):
                exhaustive_scores, exhaustive_labels = (
                    utils._generate_pairs(
                        embeddings,
                        labels,
                        pair_sampling_mode="all",
                        matching_method=method,
                    )
                )

                bounded_scores, bounded_labels = (
                    utils._generate_pairs(
                        embeddings,
                        labels,
                        pair_sampling_mode="all_genuine",
                        max_impostor_pairs=1000,
                        pair_sampling_seed=42,
                        matching_method=method,
                    )
                )

                self.assert_score_populations_equal(
                    exhaustive_scores,
                    exhaustive_labels,
                    bounded_scores,
                    bounded_labels,
                )

    def test_two_set_matches_exhaustive_all_when_cap_covers_population(self):
        probe_embeddings = np.asarray(
            [
                [1.0, 0.2, 0.4, 0.8],
                [0.9, 0.3, 0.5, 0.7],
                [0.1, 0.9, 0.3, 0.6],
                [0.7, 0.2, 1.1, 0.4],
            ]
        )

        probe_labels = np.asarray(
            [
                0,
                0,
                1,
                2,
            ]
        )

        enrollment_embeddings = np.asarray(
            [
                [0.8, 0.1, 0.5, 0.7],
                [0.2, 1.0, 0.4, 0.5],
                [0.3, 0.8, 0.6, 0.4],
                [0.6, 0.3, 0.9, 0.2],
            ]
        )

        enrollment_labels = np.asarray(
            [
                0,
                1,
                1,
                3,
            ]
        )

        for method in (
            "cosine",
            "euclidean",
            "manhattan",
            "correlation",
        ):
            with self.subTest(
                method=method
            ):
                exhaustive_scores, exhaustive_labels = (
                    utils._generate_pairs(
                        probe_embeddings,
                        probe_labels,
                        embeddings2=(
                            enrollment_embeddings
                        ),
                        labels2=(
                            enrollment_labels
                        ),
                        pair_sampling_mode="all",
                        matching_method=method,
                    )
                )

                bounded_scores, bounded_labels = (
                    utils._generate_pairs(
                        probe_embeddings,
                        probe_labels,
                        embeddings2=(
                            enrollment_embeddings
                        ),
                        labels2=(
                            enrollment_labels
                        ),
                        pair_sampling_mode="all_genuine",
                        max_impostor_pairs=1000,
                        pair_sampling_seed=42,
                        matching_method=method,
                    )
                )

                self.assert_score_populations_equal(
                    exhaustive_scores,
                    exhaustive_labels,
                    bounded_scores,
                    bounded_labels,
                )



class PairSamplingApiTests(
    unittest.TestCase
):
    @staticmethod
    def make_data():
        embeddings = np.asarray(
            [
                [1.0, 0.0, 0.2],
                [0.9, 0.1, 0.3],
                [0.0, 1.0, 0.2],
                [0.1, 0.9, 0.3],
                [0.2, 0.1, 1.0],
                [0.3, 0.2, 0.9],
            ]
        )

        labels = np.asarray(
            [
                0,
                0,
                1,
                1,
                2,
                2,
            ]
        )

        return embeddings, labels

    def test_legacy_aliases_match_canonical_arguments(self):
        embeddings, labels = (
            self.make_data()
        )

        canonical = (
            utils._generate_pairs(
                embeddings,
                labels,
                pair_sampling_mode=(
                    "balanced"
                ),
                pair_sampling_budget=30,
                pair_sampling_seed=77,
            )
        )

        legacy = (
            utils._generate_pairs(
                embeddings,
                labels,
                sampling_mode=(
                    "balanced"
                ),
                num_pairs=30,
                pair_sampling_seed=77,
            )
        )

        np.testing.assert_array_equal(
            canonical[0],
            legacy[0],
        )

        np.testing.assert_array_equal(
            canonical[1],
            legacy[1],
        )

    def test_conflicting_mode_aliases_are_rejected(self):
        embeddings, labels = (
            self.make_data()
        )

        with self.assertRaisesRegex(
            ValueError,
            "Conflicting pair sampling modes",
        ):
            utils._generate_pairs(
                embeddings,
                labels,
                pair_sampling_mode=(
                    "all_genuine"
                ),
                sampling_mode="all",
            )

    def test_conflicting_budget_aliases_are_rejected(self):
        embeddings, labels = (
            self.make_data()
        )

        with self.assertRaisesRegex(
            ValueError,
            "Conflicting pair sampling budgets",
        ):
            utils._generate_pairs(
                embeddings,
                labels,
                pair_sampling_mode=(
                    "balanced"
                ),
                pair_sampling_budget=20,
                num_pairs=30,
            )

    def test_balanced_seed_is_independent_of_global_numpy_rng(self):
        embeddings, labels = (
            self.make_data()
        )

        np.random.seed(1)

        first = (
            utils._generate_pairs(
                embeddings,
                labels,
                pair_sampling_mode=(
                    "balanced"
                ),
                pair_sampling_budget=40,
                pair_sampling_seed=123,
            )
        )

        np.random.seed(999)

        _ = np.random.random(
            1000
        )

        second = (
            utils._generate_pairs(
                embeddings,
                labels,
                pair_sampling_mode=(
                    "balanced"
                ),
                pair_sampling_budget=40,
                pair_sampling_seed=123,
            )
        )

        np.testing.assert_array_equal(
            first[0],
            second[0],
        )

        np.testing.assert_array_equal(
            first[1],
            second[1],
        )

    def test_random_seed_is_independent_of_global_numpy_rng(self):
        embeddings, labels = (
            self.make_data()
        )

        np.random.seed(3)

        first = (
            utils._generate_pairs(
                embeddings,
                labels,
                pair_sampling_mode="random",
                pair_sampling_budget=40,
                pair_sampling_seed=321,
            )
        )

        np.random.seed(700)

        _ = np.random.random(
            1000
        )

        second = (
            utils._generate_pairs(
                embeddings,
                labels,
                pair_sampling_mode="random",
                pair_sampling_budget=40,
                pair_sampling_seed=321,
            )
        )

        np.testing.assert_array_equal(
            first[0],
            second[0],
        )

        np.testing.assert_array_equal(
            first[1],
            second[1],
        )


class AlignedPairScoreTests(
    unittest.TestCase
):
    def test_vectorized_pair_scores_match_scalar_scoring(self):
        embeddings1 = np.asarray(
            [
                [1.0, 0.2, 0.4, 0.8],
                [0.1, 0.9, 0.3, 0.6],
                [0.7, 0.2, 1.1, 0.4],
            ]
        )

        embeddings2 = np.asarray(
            [
                [0.8, 0.1, 0.5, 0.7],
                [0.2, 1.0, 0.4, 0.5],
                [0.6, 0.3, 0.9, 0.2],
            ]
        )

        for method in (
            "cosine",
            "euclidean",
            "manhattan",
            "correlation",
        ):
            with self.subTest(
                method=method
            ):
                vectorized = (
                    utils._compute_aligned_pair_scores(
                        embeddings1,
                        embeddings2,
                        method=method,
                    )
                )

                scalar = np.asarray(
                    [
                        utils._compute_pair_score(
                            first,
                            second,
                            method=method,
                        )
                        for first, second in zip(
                            embeddings1,
                            embeddings2,
                        )
                    ]
                )

                np.testing.assert_allclose(
                    vectorized,
                    scalar,
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=True,
                )


if __name__ == "__main__":
    unittest.main()
