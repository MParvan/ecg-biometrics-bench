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


CROSS_SESSION_CLOSED_SET_RUNNERS = [
    "run_cross_session_identification",
    "run_cross_session_verification",
]


class ClosedSetCrossSessionPartitionTests(
    unittest.TestCase
):
    def test_retains_only_shared_eligible_subjects_and_preserves_order(
        self,
    ):
        x_session_1 = np.asarray(
            [
                [10.0],
                [20.0],
                [11.0],
                [30.0],
                [21.0],
                [31.0],
                [40.0],
            ],
            dtype=np.float32,
        )

        y_session_1 = np.asarray(
            [
                "a",
                "b",
                "a",
                "c",
                "b",
                "c",
                "session_1_only",
            ]
        )

        x_session_2 = np.asarray(
            [
                [200.0],
                [100.0],
                [300.0],
                [101.0],
                [201.0],
                [500.0],
            ],
            dtype=np.float32,
        )

        y_session_2 = np.asarray(
            [
                "b",
                "a",
                "c",
                "a",
                "b",
                "session_2_only",
            ]
        )

        partitions = (
            run._partition_closed_set_cross_session_samples(
                x_session_1,
                y_session_1,
                x_session_2,
                y_session_2,
                minimum_session_1_samples=2,
                minimum_session_2_samples=1,
            )
        )

        np.testing.assert_array_equal(
            partitions["subjects"],
            np.asarray(
                [
                    "a",
                    "b",
                    "c",
                ]
            ),
        )

        np.testing.assert_array_equal(
            partitions["session_1"][0],
            x_session_1[:-1],
        )

        np.testing.assert_array_equal(
            partitions["session_1"][1],
            y_session_1[:-1],
        )

        np.testing.assert_array_equal(
            partitions["session_2"][0],
            x_session_2[:-1],
        )

        np.testing.assert_array_equal(
            partitions["session_2"][1],
            y_session_2[:-1],
        )

        np.testing.assert_array_equal(
            partitions[
                "dropped_subjects"
            ][
                "session_1_only"
            ],
            np.asarray(
                [
                    "session_1_only",
                ]
            ),
        )

        np.testing.assert_array_equal(
            partitions[
                "dropped_subjects"
            ][
                "session_2_only"
            ],
            np.asarray(
                [
                    "session_2_only",
                ]
            ),
        )

    def test_minimum_counts_remove_subjects_from_both_sessions(
        self,
    ):
        x_session_1 = np.arange(
            14,
            dtype=np.float32,
        ).reshape(
            7,
            2,
        )

        y_session_1 = np.asarray(
            [
                "a",
                "a",
                "b",
                "c",
                "c",
                "d",
                "d",
            ]
        )

        x_session_2 = np.arange(
            16,
            dtype=np.float32,
        ).reshape(
            8,
            2,
        )

        y_session_2 = np.asarray(
            [
                "a",
                "a",
                "b",
                "b",
                "c",
                "d",
                "d",
                "c",
            ]
        )

        partitions = (
            run._partition_closed_set_cross_session_samples(
                x_session_1,
                y_session_1,
                x_session_2,
                y_session_2,
                minimum_session_1_samples=2,
                minimum_session_2_samples=2,
            )
        )

        np.testing.assert_array_equal(
            partitions["subjects"],
            np.asarray(
                [
                    "a",
                    "c",
                    "d",
                ]
            ),
        )

        np.testing.assert_array_equal(
            partitions[
                "dropped_subjects"
            ][
                "insufficient_session_1"
            ],
            np.asarray(
                [
                    "b",
                ]
            ),
        )

        self.assertNotIn(
            "b",
            partitions["session_1"][1],
        )

        self.assertNotIn(
            "b",
            partitions["session_2"][1],
        )

    def test_session_2_minimum_is_enforced(
        self,
    ):
        x_session_1 = np.arange(
            12,
            dtype=np.float32,
        ).reshape(
            6,
            2,
        )

        y_session_1 = np.asarray(
            [
                "a",
                "a",
                "b",
                "b",
                "c",
                "c",
            ]
        )

        x_session_2 = np.arange(
            10,
            dtype=np.float32,
        ).reshape(
            5,
            2,
        )

        y_session_2 = np.asarray(
            [
                "a",
                "a",
                "b",
                "c",
                "c",
            ]
        )

        partitions = (
            run._partition_closed_set_cross_session_samples(
                x_session_1,
                y_session_1,
                x_session_2,
                y_session_2,
                minimum_session_1_samples=2,
                minimum_session_2_samples=2,
            )
        )

        np.testing.assert_array_equal(
            partitions["subjects"],
            np.asarray(
                [
                    "a",
                    "c",
                ]
            ),
        )

        np.testing.assert_array_equal(
            partitions[
                "dropped_subjects"
            ][
                "insufficient_session_2"
            ],
            np.asarray(
                [
                    "b",
                ]
            ),
        )

    def test_requires_at_least_two_eligible_identities(
        self,
    ):
        x_session_1 = np.arange(
            8,
            dtype=np.float32,
        ).reshape(
            4,
            2,
        )

        y_session_1 = np.asarray(
            [
                "a",
                "a",
                "b",
                "b",
            ]
        )

        x_session_2 = np.arange(
            6,
            dtype=np.float32,
        ).reshape(
            3,
            2,
        )

        y_session_2 = np.asarray(
            [
                "a",
                "a",
                "b",
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "At least two shared identities",
        ):
            run._partition_closed_set_cross_session_samples(
                x_session_1,
                y_session_1,
                x_session_2,
                y_session_2,
                minimum_session_1_samples=2,
                minimum_session_2_samples=2,
            )

    def test_misaligned_inputs_are_rejected(
        self,
    ):
        x_session_1 = np.zeros(
            (
                4,
                8,
            ),
            dtype=np.float32,
        )

        y_session_1 = np.asarray(
            [
                0,
                0,
                1,
                1,
            ]
        )

        x_session_2 = np.zeros(
            (
                4,
                8,
            ),
            dtype=np.float32,
        )

        y_session_2 = np.asarray(
            [
                0,
                0,
                1,
                1,
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "Session 1 samples and labels are misaligned",
        ):
            run._partition_closed_set_cross_session_samples(
                x_session_1[:-1],
                y_session_1,
                x_session_2,
                y_session_2,
            )

        with self.assertRaisesRegex(
            ValueError,
            "Session 2 samples and labels are misaligned",
        ):
            run._partition_closed_set_cross_session_samples(
                x_session_1,
                y_session_1,
                x_session_2[:-1],
                y_session_2,
            )

        with self.assertRaisesRegex(
            ValueError,
            "one-dimensional",
        ):
            run._partition_closed_set_cross_session_samples(
                x_session_1,
                y_session_1.reshape(
                    -1,
                    1,
                ),
                x_session_2,
                y_session_2,
            )

    def test_invalid_minimum_counts_are_rejected(
        self,
    ):
        x_session_1 = np.zeros(
            (
                4,
                8,
            ),
            dtype=np.float32,
        )

        y_session_1 = np.asarray(
            [
                0,
                0,
                1,
                1,
            ]
        )

        x_session_2 = x_session_1.copy()
        y_session_2 = y_session_1.copy()

        invalid_values = [
            0,
            -1,
            1.5,
            True,
            "2",
            None,
        ]

        for invalid_value in invalid_values:
            with self.subTest(
                invalid_value=invalid_value
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "positive integer",
                ):
                    run._partition_closed_set_cross_session_samples(
                        x_session_1,
                        y_session_1,
                        x_session_2,
                        y_session_2,
                        minimum_session_1_samples=(
                            invalid_value
                        ),
                    )

    def test_tasks_5_and_6_use_shared_helper_once(
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
            CROSS_SESSION_CLOSED_SET_RUNNERS
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
                            "_partition_closed_set_"
                            "cross_session_samples"
                        )
                    )
                ]

                self.assertEqual(
                    len(helper_calls),
                    1,
                    (
                        f"{runner_name} must call the "
                        "shared cross-session partition "
                        "helper exactly once."
                    ),
                )


if __name__ == "__main__":
    unittest.main()