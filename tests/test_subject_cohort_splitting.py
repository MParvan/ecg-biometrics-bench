import ast
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run


SUBJECT_DISJOINT_RUNNERS = [
    "run_subject_disjoint_identification",
    "run_subject_disjoint_verification",
    (
        "run_subject_disjoint_"
        "cross_session_identification"
    ),
    (
        "run_subject_disjoint_"
        "cross_session_verification"
    ),
]


class SubjectCohortSplitTests(
    unittest.TestCase
):
    def assert_disjoint_and_exhaustive(
        self,
        original_subjects,
        train_subjects,
        validation_subjects,
        test_subjects,
    ):
        original_set = set(
            np.unique(
                original_subjects
            ).tolist()
        )

        train_set = set(
            train_subjects.tolist()
        )

        validation_set = set(
            validation_subjects.tolist()
        )

        test_set = set(
            test_subjects.tolist()
        )

        self.assertFalse(
            train_set & validation_set
        )

        self.assertFalse(
            train_set & test_set
        )

        self.assertFalse(
            validation_set & test_set
        )

        self.assertEqual(
            train_set
            | validation_set
            | test_set,
            original_set,
        )

    def test_split_is_disjoint_and_exhaustive(
        self,
    ):
        subjects = np.asarray(
            [
                f"subject_{index:02d}"
                for index in range(20)
            ]
        )

        (
            train_subjects,
            validation_subjects,
            test_subjects,
        ) = run._split_subject_cohorts(
            subjects,
            test_split=0.25,
            val_split=0.20,
            seed=42,
        )

        self.assertEqual(
            len(test_subjects),
            5,
        )

        # Fifteen subjects remain after the test split.
        # Twenty percent of those fifteen become validation subjects.
        self.assertEqual(
            len(validation_subjects),
            3,
        )

        self.assertEqual(
            len(train_subjects),
            12,
        )

        self.assert_disjoint_and_exhaustive(
            subjects,
            train_subjects,
            validation_subjects,
            test_subjects,
        )

    def test_duplicate_sample_labels_are_deduplicated(
        self,
    ):
        unique_subjects = np.asarray(
            [
                f"subject_{index:02d}"
                for index in range(12)
            ]
        )

        repeated_labels = np.repeat(
            unique_subjects,
            7,
        )

        (
            train_subjects,
            validation_subjects,
            test_subjects,
        ) = run._split_subject_cohorts(
            repeated_labels,
            test_split=0.25,
            val_split=0.25,
            seed=9,
        )

        assigned_subjects = np.concatenate(
            [
                train_subjects,
                validation_subjects,
                test_subjects,
            ]
        )

        self.assertEqual(
            len(assigned_subjects),
            len(unique_subjects),
        )

        self.assertEqual(
            len(
                np.unique(
                    assigned_subjects
                )
            ),
            len(unique_subjects),
        )

        self.assert_disjoint_and_exhaustive(
            repeated_labels,
            train_subjects,
            validation_subjects,
            test_subjects,
        )

    def test_fixed_seed_is_reproducible(
        self,
    ):
        subjects = np.arange(
            30
        )

        first_split = (
            run._split_subject_cohorts(
                subjects,
                test_split=0.20,
                val_split=0.25,
                seed=123,
            )
        )

        second_split = (
            run._split_subject_cohorts(
                subjects,
                test_split=0.20,
                val_split=0.25,
                seed=123,
            )
        )

        for first_cohort, second_cohort in zip(
            first_split,
            second_split,
        ):
            np.testing.assert_array_equal(
                first_cohort,
                second_cohort,
            )

    def test_different_seeds_change_assignment(
        self,
    ):
        subjects = np.arange(
            40
        )

        first_split = (
            run._split_subject_cohorts(
                subjects,
                test_split=0.25,
                val_split=0.20,
                seed=10,
            )
        )

        second_split = (
            run._split_subject_cohorts(
                subjects,
                test_split=0.25,
                val_split=0.20,
                seed=11,
            )
        )

        identical_cohorts = [
            np.array_equal(
                first_cohort,
                second_cohort,
            )
            for (
                first_cohort,
                second_cohort,
            ) in zip(
                first_split,
                second_split,
            )
        ]

        self.assertFalse(
            all(
                identical_cohorts
            )
        )

    def test_zero_validation_split_returns_empty_cohort(
        self,
    ):
        subjects = np.arange(
            10
        )

        (
            train_subjects,
            validation_subjects,
            test_subjects,
        ) = run._split_subject_cohorts(
            subjects,
            test_split=0.20,
            val_split=0.0,
            seed=42,
        )

        self.assertEqual(
            validation_subjects.shape,
            (0,),
        )

        self.assertEqual(
            len(train_subjects),
            8,
        )

        self.assertEqual(
            len(test_subjects),
            2,
        )

        self.assert_disjoint_and_exhaustive(
            subjects,
            train_subjects,
            validation_subjects,
            test_subjects,
        )

    def test_invalid_split_values_are_rejected(
        self,
    ):
        subjects = np.arange(
            20
        )

        invalid_cases = [
            {
                "test_split": 0.0,
                "val_split": 0.0,
            },
            {
                "test_split": 1.0,
                "val_split": 0.0,
            },
            {
                "test_split": -0.1,
                "val_split": 0.0,
            },
            {
                "test_split": 0.2,
                "val_split": -0.1,
            },
            {
                "test_split": 0.2,
                "val_split": 1.0,
            },
            {
                "test_split": np.nan,
                "val_split": 0.0,
            },
            {
                "test_split": 0.2,
                "val_split": np.inf,
            },
            {
                "test_split": True,
                "val_split": 0.0,
            },
        ]

        for case in invalid_cases:
            with self.subTest(
                **case
            ):
                with self.assertRaises(
                    ValueError
                ):
                    run._split_subject_cohorts(
                        subjects,
                        seed=42,
                        **case,
                    )

    def test_insufficient_subjects_are_rejected(
        self,
    ):
        with self.assertRaisesRegex(
            ValueError,
            "At least two unique subjects",
        ):
            run._split_subject_cohorts(
                np.asarray(
                    [
                        "only_subject",
                        "only_subject",
                    ]
                ),
                test_split=0.20,
                val_split=0.0,
                seed=42,
            )

    def test_all_subject_disjoint_runners_use_shared_helper(
        self,
    ):
        source_path = Path(
            run.__file__
        )

        syntax_tree = ast.parse(
            source_path.read_text(
                encoding="utf-8"
            )
        )

        function_nodes = {
            node.name: node
            for node in syntax_tree.body
            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
        }

        for runner_name in (
            SUBJECT_DISJOINT_RUNNERS
        ):
            with self.subTest(
                runner=runner_name
            ):
                self.assertIn(
                    runner_name,
                    function_nodes,
                )

                runner_node = function_nodes[
                    runner_name
                ]

                helper_calls = [
                    node
                    for node in ast.walk(
                        runner_node
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
                        == "_split_subject_cohorts"
                    )
                ]

                self.assertEqual(
                    len(helper_calls),
                    1,
                    (
                        f"{runner_name} must call "
                        "_split_subject_cohorts exactly once."
                    ),
                )


if __name__ == "__main__":
    unittest.main()