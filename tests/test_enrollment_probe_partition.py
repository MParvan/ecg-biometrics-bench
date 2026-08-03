import ast
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run


RUNNERS_USING_ENROLLMENT_SPLIT = [
    "run_subject_disjoint_identification",
    "run_subject_disjoint_verification",
]


def make_ordered_embeddings():
    """
    Create interleaved embeddings for three subjects.

    Values encode the subject and chronological sample position.
    """
    embeddings = np.asarray(
        [
            [10.0, 0.0],
            [20.0, 0.0],
            [10.0, 1.0],
            [30.0, 0.0],
            [20.0, 1.0],
            [10.0, 2.0],
            [30.0, 1.0],
            [20.0, 2.0],
            [10.0, 3.0],
            [30.0, 2.0],
            [20.0, 3.0],
            [30.0, 3.0],
        ],
        dtype=np.float64,
    )

    labels = np.asarray(
        [
            1,
            2,
            1,
            3,
            2,
            1,
            3,
            2,
            1,
            3,
            2,
            3,
        ]
    )

    subjects = np.asarray(
        [
            1,
            2,
            3,
        ]
    )

    return (
        embeddings,
        labels,
        subjects,
    )


class EnrollmentProbePartitionTests(
    unittest.TestCase
):
    def test_first_samples_become_enrollment_and_rest_become_probes(
        self,
    ):
        (
            embeddings,
            labels,
            subjects,
        ) = make_ordered_embeddings()

        partitions = (
            run._split_enrollment_probe_embeddings(
                embeddings,
                labels,
                subjects,
                template_size=2,
            )
        )

        (
            enrollment_embeddings,
            enrollment_labels,
        ) = partitions["enrollment"]

        (
            probe_embeddings,
            probe_labels,
        ) = partitions["probe"]

        np.testing.assert_array_equal(
            enrollment_embeddings,
            np.asarray(
                [
                    [10.0, 0.0],
                    [10.0, 1.0],
                    [20.0, 0.0],
                    [20.0, 1.0],
                    [30.0, 0.0],
                    [30.0, 1.0],
                ]
            ),
        )

        np.testing.assert_array_equal(
            enrollment_labels,
            np.asarray(
                [
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                ]
            ),
        )

        np.testing.assert_array_equal(
            probe_embeddings,
            np.asarray(
                [
                    [10.0, 2.0],
                    [10.0, 3.0],
                    [20.0, 2.0],
                    [20.0, 3.0],
                    [30.0, 2.0],
                    [30.0, 3.0],
                ]
            ),
        )

        np.testing.assert_array_equal(
            probe_labels,
            np.asarray(
                [
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                ]
            ),
        )

    def test_subject_order_controls_output_block_order(
        self,
    ):
        (
            embeddings,
            labels,
            _,
        ) = make_ordered_embeddings()

        partitions = (
            run._split_enrollment_probe_embeddings(
                embeddings,
                labels,
                subjects=np.asarray(
                    [
                        3,
                        1,
                        2,
                    ]
                ),
                template_size=1,
            )
        )

        enrollment_labels = partitions[
            "enrollment"
        ][1]

        probe_labels = partitions[
            "probe"
        ][1]

        np.testing.assert_array_equal(
            enrollment_labels,
            np.asarray(
                [
                    3,
                    1,
                    2,
                ]
            ),
        )

        np.testing.assert_array_equal(
            probe_labels,
            np.asarray(
                [
                    3,
                    3,
                    3,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                ]
            ),
        )

    def test_outputs_are_subject_disjoint_by_sample_position(
        self,
    ):
        (
            embeddings,
            labels,
            subjects,
        ) = make_ordered_embeddings()

        partitions = (
            run._split_enrollment_probe_embeddings(
                embeddings,
                labels,
                subjects,
                template_size=3,
            )
        )

        enrollment_embeddings = partitions[
            "enrollment"
        ][0]

        probe_embeddings = partitions[
            "probe"
        ][0]

        enrollment_rows = {
            tuple(row)
            for row in enrollment_embeddings
        }

        probe_rows = {
            tuple(row)
            for row in probe_embeddings
        }

        self.assertFalse(
            enrollment_rows
            & probe_rows
        )

        self.assertEqual(
            len(enrollment_embeddings),
            9,
        )

        self.assertEqual(
            len(probe_embeddings),
            3,
        )

    def test_invalid_template_sizes_are_rejected(
        self,
    ):
        (
            embeddings,
            labels,
            subjects,
        ) = make_ordered_embeddings()

        invalid_values = [
            0,
            -1,
            1.5,
            "2",
            True,
            None,
        ]

        for template_size in invalid_values:
            with self.subTest(
                template_size=template_size
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "positive integer",
                ):
                    run._split_enrollment_probe_embeddings(
                        embeddings,
                        labels,
                        subjects,
                        template_size=template_size,
                    )

    def test_subject_without_probe_samples_is_rejected(
        self,
    ):
        (
            embeddings,
            labels,
            subjects,
        ) = make_ordered_embeddings()

        with self.assertRaisesRegex(
            ValueError,
            "at least one additional probe sample",
        ):
            run._split_enrollment_probe_embeddings(
                embeddings,
                labels,
                subjects,
                template_size=4,
            )

    def test_subject_coverage_must_match_labels(
        self,
    ):
        (
            embeddings,
            labels,
            _,
        ) = make_ordered_embeddings()

        with self.assertRaisesRegex(
            ValueError,
            "match exactly",
        ):
            run._split_enrollment_probe_embeddings(
                embeddings,
                labels,
                subjects=np.asarray(
                    [
                        1,
                        2,
                    ]
                ),
                template_size=1,
            )

        with self.assertRaisesRegex(
            ValueError,
            "match exactly",
        ):
            run._split_enrollment_probe_embeddings(
                embeddings,
                labels,
                subjects=np.asarray(
                    [
                        1,
                        2,
                        3,
                        4,
                    ]
                ),
                template_size=1,
            )

    def test_duplicate_subjects_are_rejected(
        self,
    ):
        (
            embeddings,
            labels,
            _,
        ) = make_ordered_embeddings()

        with self.assertRaisesRegex(
            ValueError,
            "duplicate",
        ):
            run._split_enrollment_probe_embeddings(
                embeddings,
                labels,
                subjects=np.asarray(
                    [
                        1,
                        2,
                        2,
                        3,
                    ]
                ),
                template_size=1,
            )

    def test_misaligned_and_invalid_shapes_are_rejected(
        self,
    ):
        (
            embeddings,
            labels,
            subjects,
        ) = make_ordered_embeddings()

        with self.assertRaisesRegex(
            ValueError,
            "same number of samples",
        ):
            run._split_enrollment_probe_embeddings(
                embeddings[:-1],
                labels,
                subjects,
                template_size=1,
            )

        with self.assertRaisesRegex(
            ValueError,
            "two-dimensional",
        ):
            run._split_enrollment_probe_embeddings(
                embeddings.reshape(-1),
                labels,
                subjects,
                template_size=1,
            )

        with self.assertRaisesRegex(
            ValueError,
            "one-dimensional",
        ):
            run._split_enrollment_probe_embeddings(
                embeddings,
                labels.reshape(-1, 1),
                subjects,
                template_size=1,
            )

    def test_tasks_3_and_4_use_shared_helper(
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
            RUNNERS_USING_ENROLLMENT_SPLIT
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
                            "_split_enrollment_"
                            "probe_embeddings"
                        )
                    )
                ]

                self.assertEqual(
                    len(helper_calls),
                    1,
                    (
                        f"{runner_name} must call "
                        "_split_enrollment_probe_embeddings "
                        "exactly once."
                    ),
                )


if __name__ == "__main__":
    unittest.main()