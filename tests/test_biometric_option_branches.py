import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import (
    _apply_score_fusion,
    _compute_pair_score,
    _compute_score_matrix,
    _create_templates,
    _generate_pairs,
    _set_seed,
)


TEMPLATE_METHODS = [
    "mean",
    "median",
    "trimmed_mean",
    "representative",
    "soft_centrality",
    "geometric_median",
    "none",
]

MATCHING_METHODS = [
    "cosine",
    "euclidean",
    "manhattan",
    "correlation",
]

SAMPLING_MODES = [
    "all",
    "balanced",
    "random",
]


def make_embeddings():
    """
    Return finite, non-constant embeddings for three subjects.

    The vectors avoid undefined correlation scores caused by constant
    feature vectors.
    """
    embeddings = np.asarray(
        [
            [1.00, 0.20, 0.10, 0.40],
            [0.90, 0.25, 0.15, 0.35],
            [1.10, 0.15, 0.05, 0.45],
            [0.95, 0.22, 0.08, 0.42],

            [0.10, 1.00, 0.30, 0.20],
            [0.15, 0.90, 0.35, 0.25],
            [0.05, 1.10, 0.25, 0.15],
            [0.12, 0.95, 0.32, 0.22],

            [0.25, 0.15, 1.00, 0.30],
            [0.30, 0.10, 0.90, 0.35],
            [0.20, 0.20, 1.10, 0.25],
            [0.28, 0.12, 0.95, 0.32],
        ],
        dtype=np.float64,
    )

    labels = np.repeat(
        np.asarray([0, 1, 2]),
        4,
    )

    return embeddings, labels


class TemplateFusionTests(unittest.TestCase):
    def setUp(self):
        self.embeddings, self.labels = (
            make_embeddings()
        )

    def test_all_template_methods_return_valid_outputs(self):
        for method in TEMPLATE_METHODS:
            with self.subTest(method=method):
                templates, template_labels = (
                    _create_templates(
                        self.embeddings,
                        self.labels,
                        method=method,
                        max_beats=None,
                    )
                )

                self.assertEqual(
                    templates.ndim,
                    2,
                )
                self.assertEqual(
                    template_labels.ndim,
                    1,
                )
                self.assertEqual(
                    templates.shape[0],
                    template_labels.shape[0],
                )
                self.assertEqual(
                    templates.shape[1],
                    self.embeddings.shape[1],
                )

                self.assertTrue(
                    np.isfinite(templates).all()
                )

                if method == "none":
                    np.testing.assert_array_equal(
                        templates,
                        self.embeddings,
                    )
                    np.testing.assert_array_equal(
                        template_labels,
                        self.labels,
                    )
                else:
                    self.assertEqual(
                        templates.shape,
                        (3, 4),
                    )
                    np.testing.assert_array_equal(
                        template_labels,
                        np.asarray([0, 1, 2]),
                    )

    def test_mean_template_has_expected_value(self):
        templates, template_labels = (
            _create_templates(
                self.embeddings,
                self.labels,
                method="mean",
            )
        )

        subject_zero = self.embeddings[
            self.labels == 0
        ]

        np.testing.assert_allclose(
            templates[
                np.where(
                    template_labels == 0
                )[0][0]
            ],
            np.mean(
                subject_zero,
                axis=0,
            ),
        )

    def test_max_beats_uses_first_requested_embeddings(self):
        templates, template_labels = (
            _create_templates(
                self.embeddings,
                self.labels,
                method="mean",
                max_beats=2,
            )
        )

        for subject in np.unique(
            self.labels
        ):
            subject_embeddings = (
                self.embeddings[
                    self.labels == subject
                ]
            )

            expected = np.mean(
                subject_embeddings[:2],
                axis=0,
            )

            result_index = np.where(
                template_labels == subject
            )[0][0]

            np.testing.assert_allclose(
                templates[result_index],
                expected,
            )

    def test_unknown_template_method_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unknown template method",
        ):
            _create_templates(
                self.embeddings,
                self.labels,
                method="unsupported",
            )


class MatchingMethodTests(unittest.TestCase):
    def setUp(self):
        self.embeddings, self.labels = (
            make_embeddings()
        )

        self.probes = self.embeddings[
            [0, 4, 8]
        ]

        self.gallery = self.embeddings[
            [1, 5, 9]
        ]

    def test_all_matching_methods_return_finite_score_matrix(
        self,
    ):
        for method in MATCHING_METHODS:
            with self.subTest(method=method):
                scores = _compute_score_matrix(
                    self.probes,
                    self.gallery,
                    method=method,
                )

                self.assertEqual(
                    scores.shape,
                    (3, 3),
                )
                self.assertTrue(
                    np.isfinite(scores).all()
                )

                # Each probe is closest to the corresponding
                # subject gallery vector.
                np.testing.assert_array_equal(
                    np.argmax(
                        scores,
                        axis=1,
                    ),
                    np.asarray([0, 1, 2]),
                )

    def test_pair_scores_rank_identical_vectors_higher(
        self,
    ):
        reference = np.asarray(
            [1.0, 0.2, 0.4, 0.7]
        )

        similar = np.asarray(
            [0.95, 0.25, 0.38, 0.72]
        )

        different = np.asarray(
            [0.2, 1.0, 0.8, 0.1]
        )

        for method in MATCHING_METHODS:
            with self.subTest(method=method):
                similar_score = (
                    _compute_pair_score(
                        reference,
                        similar,
                        method=method,
                    )
                )

                different_score = (
                    _compute_pair_score(
                        reference,
                        different,
                        method=method,
                    )
                )

                self.assertTrue(
                    np.isfinite(similar_score)
                )
                self.assertTrue(
                    np.isfinite(different_score)
                )

                self.assertGreater(
                    similar_score,
                    different_score,
                )

    def test_unknown_matching_method_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unknown matching method",
        ):
            _compute_score_matrix(
                self.probes,
                self.gallery,
                method="unsupported",
            )


class PairGenerationTests(unittest.TestCase):
    def setUp(self):
        self.embeddings, self.labels = (
            make_embeddings()
        )

    def assert_pair_output(
        self,
        scores,
        pair_labels,
    ):
        self.assertIsInstance(
            scores,
            np.ndarray,
        )
        self.assertIsInstance(
            pair_labels,
            np.ndarray,
        )

        self.assertEqual(
            scores.ndim,
            1,
        )
        self.assertEqual(
            pair_labels.ndim,
            1,
        )
        self.assertEqual(
            len(scores),
            len(pair_labels),
        )

        self.assertGreater(
            len(scores),
            0,
        )
        self.assertTrue(
            np.isfinite(scores).all()
        )

        self.assertTrue(
            set(
                np.unique(pair_labels)
            ).issubset({0, 1})
        )

    def test_all_sampling_generates_every_non_self_pair(
        self,
    ):
        scores, pair_labels = (
            _generate_pairs(
                embeddings1=self.embeddings,
                labels1=self.labels,
                embeddings2=None,
                labels2=None,
                num_pairs=20,
                sampling_mode="all",
                matching_method="cosine",
            )
        )

        self.assert_pair_output(
            scores,
            pair_labels,
        )

        expected_pair_count = (
            len(self.labels)
            * (len(self.labels) - 1)
            // 2
        )

        self.assertEqual(
            len(scores),
            expected_pair_count,
        )

        # Three subjects, each with four samples:
        # 3 * C(4, 2) = 18 genuine comparisons.
        self.assertEqual(
            int(np.sum(pair_labels == 1)),
            18,
        )

        self.assertEqual(
            int(np.sum(pair_labels == 0)),
            expected_pair_count - 18,
        )

    def test_balanced_sampling_has_equal_classes(self):
        _set_seed(42)

        scores, pair_labels = (
            _generate_pairs(
                embeddings1=self.embeddings,
                labels1=self.labels,
                embeddings2=None,
                labels2=None,
                num_pairs=40,
                sampling_mode="balanced",
                matching_method="euclidean",
            )
        )

        self.assert_pair_output(
            scores,
            pair_labels,
        )

        self.assertEqual(
            len(scores),
            40,
        )
        self.assertEqual(
            int(np.sum(pair_labels == 1)),
            20,
        )
        self.assertEqual(
            int(np.sum(pair_labels == 0)),
            20,
        )

    def test_random_sampling_is_reproducible_after_seed_reset(
        self,
    ):
        _set_seed(123)

        scores_first, labels_first = (
            _generate_pairs(
                embeddings1=self.embeddings,
                labels1=self.labels,
                embeddings2=None,
                labels2=None,
                num_pairs=50,
                sampling_mode="random",
                matching_method="manhattan",
            )
        )

        _set_seed(123)

        scores_second, labels_second = (
            _generate_pairs(
                embeddings1=self.embeddings,
                labels1=self.labels,
                embeddings2=None,
                labels2=None,
                num_pairs=50,
                sampling_mode="random",
                matching_method="manhattan",
            )
        )

        self.assert_pair_output(
            scores_first,
            labels_first,
        )

        np.testing.assert_array_equal(
            scores_first,
            scores_second,
        )

        np.testing.assert_array_equal(
            labels_first,
            labels_second,
        )

        # Random intra-set generation skips sampled self-pairs,
        # so the final count may be smaller than num_pairs.
        self.assertLessEqual(
            len(scores_first),
            50,
        )

    def test_cross_set_all_sampling_uses_full_matrix(
        self,
    ):
        enrollment_embeddings = (
            self.embeddings[
                [0, 4, 8]
            ]
        )

        enrollment_labels = np.asarray(
            [0, 1, 2]
        )

        scores, pair_labels = (
            _generate_pairs(
                embeddings1=self.embeddings,
                labels1=self.labels,
                embeddings2=(
                    enrollment_embeddings
                ),
                labels2=enrollment_labels,
                sampling_mode="all",
                matching_method="correlation",
            )
        )

        self.assert_pair_output(
            scores,
            pair_labels,
        )

        self.assertEqual(
            len(scores),
            (
                len(self.embeddings)
                * len(enrollment_embeddings)
            ),
        )

        # Every probe has exactly one same-identity template.
        self.assertEqual(
            int(np.sum(pair_labels == 1)),
            len(self.embeddings),
        )


class ProbeFusionTests(unittest.TestCase):
    def test_fusion_size_one_preserves_inputs(self):
        scores = np.asarray(
            [
                [0.9, 0.1],
                [0.8, 0.2],
            ]
        )
        labels = np.asarray(
            [0, 0]
        )

        fused_scores, fused_labels = (
            _apply_score_fusion(
                scores,
                labels,
                fusion_size=1,
            )
        )

        np.testing.assert_array_equal(
            fused_scores,
            scores,
        )
        np.testing.assert_array_equal(
            fused_labels,
            labels,
        )

    def test_score_fusion_averages_subject_blocks(
        self,
    ):
        scores = np.asarray(
            [
                [0.9, 0.1],
                [0.7, 0.3],
                [0.2, 0.8],
                [0.4, 0.6],
            ],
            dtype=np.float64,
        )

        labels = np.asarray(
            [0, 0, 1, 1]
        )

        fused_scores, fused_labels = (
            _apply_score_fusion(
                scores,
                labels,
                fusion_size=2,
            )
        )

        np.testing.assert_allclose(
            fused_scores,
            np.asarray(
                [
                    [0.8, 0.2],
                    [0.3, 0.7],
                ]
            ),
        )

        np.testing.assert_array_equal(
            fused_labels,
            np.asarray([0, 1]),
        )

    def test_subject_with_fewer_probes_is_still_retained(
        self,
    ):
        scores = np.asarray(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.2, 0.8],
            ],
            dtype=np.float64,
        )

        labels = np.asarray(
            [0, 0, 1]
        )

        fused_scores, fused_labels = (
            _apply_score_fusion(
                scores,
                labels,
                fusion_size=2,
            )
        )

        self.assertEqual(
            fused_scores.shape,
            (2, 2),
        )

        np.testing.assert_array_equal(
            fused_labels,
            np.asarray([0, 1]),
        )

        np.testing.assert_allclose(
            fused_scores[1],
            scores[2],
        )


if __name__ == "__main__":
    unittest.main()