import ast
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

import run


CLOSED_SET_RUNNERS = [
    "run_closed_set_identification",
    "run_closed_set_verification",
]


def make_closed_set_data(
    number_of_subjects=5,
    samples_per_subject=10,
):
    labels = np.repeat(
        np.arange(
            number_of_subjects
        ),
        samples_per_subject,
    )

    sample_indices = np.arange(
        len(labels),
        dtype=np.float32,
    )

    samples = np.column_stack(
        [
            sample_indices,
            sample_indices + 1000.0,
        ]
    ).astype(
        np.float32
    )

    sqi_scores = (
        sample_indices.astype(
            np.float64
        )
        + 0.25
    )

    return (
        samples,
        labels,
        sqi_scores,
    )


class ClosedSetPartitionIntegrityTests(
    unittest.TestCase
):
    def test_three_way_split_has_no_sample_overlap(
        self,
    ):
        (
            samples,
            labels,
            _,
        ) = make_closed_set_data()

        train_test_partitions = (
            run._split_closed_set_samples(
                samples,
                labels,
                holdout_split=0.20,
                seed=42,
            )
        )

        (
            training_pool,
            training_pool_labels,
            _,
        ) = train_test_partitions[
            "retained"
        ]

        (
            test_samples,
            test_labels,
            _,
        ) = train_test_partitions[
            "holdout"
        ]

        validation_partitions = (
            run._split_closed_set_samples(
                training_pool,
                training_pool_labels,
                holdout_split=0.25,
                seed=42,
            )
        )

        (
            training_samples,
            training_labels,
            _,
        ) = validation_partitions[
            "retained"
        ]

        (
            validation_samples,
            validation_labels,
            _,
        ) = validation_partitions[
            "holdout"
        ]

        training_pool_source_indices = (
            train_test_partitions[
                "indices"
            ][
                "retained"
            ]
        )

        training_source_indices = (
            training_pool_source_indices[
                validation_partitions[
                    "indices"
                ][
                    "retained"
                ]
            ]
        )

        validation_source_indices = (
            training_pool_source_indices[
                validation_partitions[
                    "indices"
                ][
                    "holdout"
                ]
            ]
        )

        test_source_indices = (
            train_test_partitions[
                "indices"
            ][
                "holdout"
            ]
        )

        training_index_set = set(
            training_source_indices.tolist()
        )

        validation_index_set = set(
            validation_source_indices.tolist()
        )

        test_index_set = set(
            test_source_indices.tolist()
        )

        self.assertFalse(
            training_index_set
            & validation_index_set
        )

        self.assertFalse(
            training_index_set
            & test_index_set
        )

        self.assertFalse(
            validation_index_set
            & test_index_set
        )

        assigned_indices = (
            training_index_set
            | validation_index_set
            | test_index_set
        )

        self.assertEqual(
            assigned_indices,
            set(
                range(
                    len(samples)
                )
            ),
        )

        all_subjects = set(
            np.unique(
                labels
            ).tolist()
        )

        self.assertEqual(
            set(
                np.unique(
                    training_labels
                ).tolist()
            ),
            all_subjects,
        )

        self.assertEqual(
            set(
                np.unique(
                    validation_labels
                ).tolist()
            ),
            all_subjects,
        )

        self.assertEqual(
            set(
                np.unique(
                    test_labels
                ).tolist()
            ),
            all_subjects,
        )

        self.assertEqual(
            len(training_samples)
            + len(validation_samples)
            + len(test_samples),
            len(samples),
        )

    def test_aligned_sqi_values_follow_sample_indices(
        self,
    ):
        (
            samples,
            labels,
            sqi_scores,
        ) = make_closed_set_data()

        partitions = (
            run._split_closed_set_samples(
                samples,
                labels,
                holdout_split=0.20,
                seed=7,
                aligned_values=sqi_scores,
            )
        )

        for partition_name in [
            "retained",
            "holdout",
        ]:
            with self.subTest(
                partition=partition_name
            ):
                (
                    partition_samples,
                    partition_labels,
                    partition_sqi,
                ) = partitions[
                    partition_name
                ]

                partition_indices = (
                    partitions[
                        "indices"
                    ][
                        partition_name
                    ]
                )

                np.testing.assert_array_equal(
                    partition_samples,
                    samples[
                        partition_indices
                    ],
                )

                np.testing.assert_array_equal(
                    partition_labels,
                    labels[
                        partition_indices
                    ],
                )

                np.testing.assert_array_equal(
                    partition_sqi,
                    sqi_scores[
                        partition_indices
                    ],
                )

    def test_zero_holdout_returns_complete_retained_partition(
        self,
    ):
        (
            samples,
            labels,
            sqi_scores,
        ) = make_closed_set_data()

        partitions = (
            run._split_closed_set_samples(
                samples,
                labels,
                holdout_split=0.0,
                seed=42,
                aligned_values=sqi_scores,
            )
        )

        (
            retained_samples,
            retained_labels,
            retained_sqi,
        ) = partitions[
            "retained"
        ]

        np.testing.assert_array_equal(
            retained_samples,
            samples,
        )

        np.testing.assert_array_equal(
            retained_labels,
            labels,
        )

        np.testing.assert_array_equal(
            retained_sqi,
            sqi_scores,
        )

        self.assertEqual(
            partitions["holdout"],
            (
                None,
                None,
                None,
            ),
        )

        np.testing.assert_array_equal(
            partitions[
                "indices"
            ][
                "retained"
            ],
            np.arange(
                len(samples)
            ),
        )

        self.assertEqual(
            len(
                partitions[
                    "indices"
                ][
                    "holdout"
                ]
            ),
            0,
        )

    def test_fixed_seed_is_deterministic(
        self,
    ):
        (
            samples,
            labels,
            _,
        ) = make_closed_set_data()

        first = (
            run._split_closed_set_samples(
                samples,
                labels,
                holdout_split=0.20,
                seed=123,
            )
        )

        second = (
            run._split_closed_set_samples(
                samples,
                labels,
                holdout_split=0.20,
                seed=123,
            )
        )

        np.testing.assert_array_equal(
            first[
                "indices"
            ][
                "retained"
            ],
            second[
                "indices"
            ][
                "retained"
            ],
        )

        np.testing.assert_array_equal(
            first[
                "indices"
            ][
                "holdout"
            ],
            second[
                "indices"
            ][
                "holdout"
            ],
        )

    def test_misaligned_inputs_are_rejected(
        self,
    ):
        (
            samples,
            labels,
            sqi_scores,
        ) = make_closed_set_data()

        with self.assertRaisesRegex(
            ValueError,
            "same number of samples",
        ):
            run._split_closed_set_samples(
                samples[:-1],
                labels,
                holdout_split=0.20,
                seed=42,
            )

        with self.assertRaisesRegex(
            ValueError,
            "one-dimensional",
        ):
            run._split_closed_set_samples(
                samples,
                labels.reshape(
                    -1,
                    1,
                ),
                holdout_split=0.20,
                seed=42,
            )

        with self.assertRaisesRegex(
            ValueError,
            "one value per sample",
        ):
            run._split_closed_set_samples(
                samples,
                labels,
                holdout_split=0.20,
                seed=42,
                aligned_values=sqi_scores[:-1],
            )

    def test_invalid_holdout_fractions_are_rejected(
        self,
    ):
        (
            samples,
            labels,
            _,
        ) = make_closed_set_data()

        invalid_values = [
            -0.1,
            1.0,
            1.5,
            float("nan"),
            True,
            "invalid",
        ]

        for holdout_split in invalid_values:
            with self.subTest(
                holdout_split=holdout_split
            ):
                with self.assertRaises(
                    ValueError
                ):
                    run._split_closed_set_samples(
                        samples,
                        labels,
                        holdout_split=holdout_split,
                        seed=42,
                    )

    def test_identity_with_one_sample_is_rejected(
        self,
    ):
        samples = np.arange(
            10,
            dtype=np.float32,
        ).reshape(
            5,
            2,
        )

        labels = np.asarray(
            [
                0,
                0,
                1,
                1,
                2,
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "at least two samples",
        ):
            run._split_closed_set_samples(
                samples,
                labels,
                holdout_split=0.40,
                seed=42,
            )

    def test_tasks_1_and_2_use_shared_helper_twice(
        self,
    ):
        syntax_tree = ast.parse(
            Path(
                run.__file__
            ).read_text(
                encoding="utf-8"
            )
        )

        functions = {
            node.name: node
            for node in syntax_tree.body
            if isinstance(
                node,
                ast.FunctionDef,
            )
        }

        for runner_name in (
            CLOSED_SET_RUNNERS
        ):
            with self.subTest(
                runner=runner_name
            ):
                self.assertIn(
                    runner_name,
                    functions,
                )

                helper_calls = [
                    node
                    for node in ast.walk(
                        functions[
                            runner_name
                        ]
                    )
                    if (
                        isinstance(
                            node,
                            ast.Call,
                        )
                        and isinstance(
                            node.func,
                            ast.Name,
                        )
                        and node.func.id
                        == (
                            "_split_closed_set_samples"
                        )
                    )
                ]

                direct_split_calls = [
                    node
                    for node in ast.walk(
                        functions[
                            runner_name
                        ]
                    )
                    if (
                        isinstance(
                            node,
                            ast.Call,
                        )
                        and isinstance(
                            node.func,
                            ast.Name,
                        )
                        and node.func.id
                        == "train_test_split"
                    )
                ]

                self.assertEqual(
                    len(helper_calls),
                    2,
                    (
                        f"{runner_name} must use the "
                        "closed-set split helper exactly twice."
                    ),
                )

                self.assertEqual(
                    len(direct_split_calls),
                    0,
                    (
                        f"{runner_name} must not call "
                        "train_test_split directly."
                    ),
                )


if __name__ == "__main__":
    unittest.main()