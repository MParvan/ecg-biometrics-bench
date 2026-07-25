import ast
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run


INTRA_SESSION_RUNNERS = [
    "run_subject_disjoint_identification",
    "run_subject_disjoint_verification",
]

CROSS_SESSION_RUNNERS = [
    (
        "run_subject_disjoint_"
        "cross_session_identification"
    ),
    (
        "run_subject_disjoint_"
        "cross_session_verification"
    ),
]


def make_intra_session_data():
    labels = np.repeat(
        np.asarray(
            [
                "subject_0",
                "subject_1",
                "subject_2",
                "subject_3",
            ]
        ),
        3,
    )

    samples = np.arange(
        len(labels),
        dtype=np.float32,
    ).reshape(-1, 1)

    sqi_scores = np.linspace(
        0.10,
        0.90,
        len(labels),
    )

    return (
        samples,
        labels,
        sqi_scores,
    )


def make_session_data(
    session_marker,
    subjects,
    samples_per_subject=3,
):
    samples = []
    labels = []

    for subject_index, subject in enumerate(
        subjects
    ):
        for sample_index in range(
            samples_per_subject
        ):
            marker = (
                session_marker
                + subject_index * 10
                + sample_index
            )

            samples.append(
                [float(marker)]
            )

            labels.append(subject)

    return (
        np.asarray(
            samples,
            dtype=np.float32,
        ),
        np.asarray(labels),
    )


class IntraSessionPartitionTests(
    unittest.TestCase
):
    def test_samples_are_assigned_by_subject(
        self,
    ):
        x, y, sqi_scores = (
            make_intra_session_data()
        )

        partitions = (
            run._partition_subject_disjoint_samples(
                x,
                y,
                train_subjects=np.asarray(
                    [
                        "subject_0",
                        "subject_1",
                    ]
                ),
                validation_subjects=np.asarray(
                    ["subject_2"]
                ),
                test_subjects=np.asarray(
                    ["subject_3"]
                ),
                sqi_scores=sqi_scores,
            )
        )

        train_x, train_y, train_sqi = (
            partitions["train"]
        )

        validation_x, validation_y, validation_sqi = (
            partitions["validation"]
        )

        test_x, test_y, test_sqi = (
            partitions["test"]
        )

        self.assertEqual(
            set(np.unique(train_y)),
            {
                "subject_0",
                "subject_1",
            },
        )

        self.assertEqual(
            set(np.unique(validation_y)),
            {"subject_2"},
        )

        self.assertEqual(
            set(np.unique(test_y)),
            {"subject_3"},
        )

        self.assertEqual(
            len(train_x),
            6,
        )

        self.assertEqual(
            len(validation_x),
            3,
        )

        self.assertEqual(
            len(test_x),
            3,
        )

        np.testing.assert_array_equal(
            train_sqi,
            sqi_scores[
                np.isin(
                    y,
                    [
                        "subject_0",
                        "subject_1",
                    ],
                )
            ],
        )

        np.testing.assert_array_equal(
            validation_sqi,
            sqi_scores[
                y == "subject_2"
            ],
        )

        np.testing.assert_array_equal(
            test_sqi,
            sqi_scores[
                y == "subject_3"
            ],
        )

    def test_disabled_validation_returns_none_partition(
        self,
    ):
        x, y, sqi_scores = (
            make_intra_session_data()
        )

        partitions = (
            run._partition_subject_disjoint_samples(
                x,
                y,
                train_subjects=np.asarray(
                    [
                        "subject_0",
                        "subject_1",
                        "subject_2",
                    ]
                ),
                validation_subjects=np.asarray(
                    [],
                    dtype=y.dtype,
                ),
                test_subjects=np.asarray(
                    ["subject_3"]
                ),
                sqi_scores=sqi_scores,
            )
        )

        self.assertEqual(
            partitions["validation"],
            (
                None,
                None,
                None,
            ),
        )

    def test_overlapping_cohorts_are_rejected(
        self,
    ):
        x, y, _ = make_intra_session_data()

        with self.assertRaisesRegex(
            ValueError,
            "overlap",
        ):
            run._partition_subject_disjoint_samples(
                x,
                y,
                train_subjects=np.asarray(
                    [
                        "subject_0",
                        "subject_1",
                    ]
                ),
                validation_subjects=np.asarray(
                    [
                        "subject_1",
                        "subject_2",
                    ]
                ),
                test_subjects=np.asarray(
                    ["subject_3"]
                ),
            )

    def test_incomplete_subject_coverage_is_rejected(
        self,
    ):
        x, y, _ = make_intra_session_data()

        with self.assertRaisesRegex(
            ValueError,
            "cover exactly all subjects",
        ):
            run._partition_subject_disjoint_samples(
                x,
                y,
                train_subjects=np.asarray(
                    ["subject_0"]
                ),
                validation_subjects=np.asarray(
                    ["subject_1"]
                ),
                test_subjects=np.asarray(
                    ["subject_3"]
                ),
            )


class CrossSessionPartitionTests(
    unittest.TestCase
):
    def setUp(self):
        common_subjects = [
            "subject_0",
            "subject_1",
            "subject_2",
            "subject_3",
            "subject_4",
        ]

        self.x_s1, self.y_s1 = (
            make_session_data(
                session_marker=1000,
                subjects=(
                    common_subjects
                    + ["session_1_only"]
                ),
            )
        )

        self.x_s2, self.y_s2 = (
            make_session_data(
                session_marker=2000,
                subjects=(
                    common_subjects
                    + ["session_2_only"]
                ),
            )
        )

    def test_temporal_and_subject_isolation(
        self,
    ):
        partitions = (
            run._partition_subject_disjoint_cross_session_samples(
                self.x_s1,
                self.y_s1,
                self.x_s2,
                self.y_s2,
                train_subjects=np.asarray(
                    [
                        "subject_0",
                        "subject_1",
                    ]
                ),
                validation_subjects=np.asarray(
                    ["subject_2"]
                ),
                test_subjects=np.asarray(
                    [
                        "subject_3",
                        "subject_4",
                    ]
                ),
            )
        )

        train_x, train_y = partitions[
            "train"
        ]

        validation_x, validation_y = (
            partitions["validation"]
        )

        enrollment_x, enrollment_y = (
            partitions["enrollment"]
        )

        probe_x, probe_y = partitions[
            "probe"
        ]

        self.assertEqual(
            set(np.unique(train_y)),
            {
                "subject_0",
                "subject_1",
            },
        )

        self.assertEqual(
            set(np.unique(validation_y)),
            {"subject_2"},
        )

        self.assertEqual(
            set(np.unique(enrollment_y)),
            {
                "subject_3",
                "subject_4",
            },
        )

        self.assertEqual(
            set(np.unique(probe_y)),
            {
                "subject_3",
                "subject_4",
            },
        )

        # Session 1 markers start at 1000.
        self.assertTrue(
            np.all(train_x < 2000)
        )

        self.assertTrue(
            np.all(validation_x < 2000)
        )

        self.assertTrue(
            np.all(enrollment_x < 2000)
        )

        # Session 2 markers start at 2000.
        self.assertTrue(
            np.all(probe_x >= 2000)
        )

        self.assertNotIn(
            "session_1_only",
            train_y,
        )

        self.assertNotIn(
            "session_2_only",
            probe_y,
        )

    def test_disabled_validation_returns_none_partition(
        self,
    ):
        partitions = (
            run._partition_subject_disjoint_cross_session_samples(
                self.x_s1,
                self.y_s1,
                self.x_s2,
                self.y_s2,
                train_subjects=np.asarray(
                    [
                        "subject_0",
                        "subject_1",
                        "subject_2",
                    ]
                ),
                validation_subjects=np.asarray(
                    [],
                    dtype=self.y_s1.dtype,
                ),
                test_subjects=np.asarray(
                    [
                        "subject_3",
                        "subject_4",
                    ]
                ),
            )
        )

        self.assertEqual(
            partitions["validation"],
            (
                None,
                None,
            ),
        )

    def test_non_common_subject_cannot_enter_cohorts(
        self,
    ):
        with self.assertRaisesRegex(
            ValueError,
            "shared by Session 1 and Session 2",
        ):
            run._partition_subject_disjoint_cross_session_samples(
                self.x_s1,
                self.y_s1,
                self.x_s2,
                self.y_s2,
                train_subjects=np.asarray(
                    [
                        "subject_0",
                        "subject_1",
                        "session_1_only",
                    ]
                ),
                validation_subjects=np.asarray(
                    ["subject_2"]
                ),
                test_subjects=np.asarray(
                    [
                        "subject_3",
                        "subject_4",
                    ]
                ),
            )


class RunnerPartitionIntegrationTests(
    unittest.TestCase
):
    def test_runners_use_shared_partition_helpers(
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
            INTRA_SESSION_RUNNERS
        ):
            with self.subTest(
                runner=runner_name
            ):
                calls = [
                    node
                    for node in ast.walk(
                        functions[runner_name]
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
                            "_partition_subject_"
                            "disjoint_samples"
                        )
                    )
                ]

                self.assertEqual(
                    len(calls),
                    1,
                )

        for runner_name in (
            CROSS_SESSION_RUNNERS
        ):
            with self.subTest(
                runner=runner_name
            ):
                calls = [
                    node
                    for node in ast.walk(
                        functions[runner_name]
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
                            "_partition_subject_"
                            "disjoint_cross_session_"
                            "samples"
                        )
                    )
                ]

                self.assertEqual(
                    len(calls),
                    1,
                )


if __name__ == "__main__":
    unittest.main()