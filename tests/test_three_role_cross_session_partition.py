import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run


def samples(values):
    return np.asarray(
        values,
        dtype=np.float32,
    ).reshape(-1, 1)


class ClosedSetThreeRolePartitionTests(
    unittest.TestCase
):
    def test_explicit_enrollment_uses_three_way_identity_intersection(
        self,
    ):
        x_train = samples(
            [10, 11, 20, 21, 30, 31]
        )
        y_train = np.asarray(
            ["a", "a", "b", "b", "c", "c"]
        )

        x_enroll = samples(
            [100, 200]
        )
        y_enroll = np.asarray(
            ["a", "b"]
        )

        x_probe = samples(
            [1000, 2000, 3000]
        )
        y_probe = np.asarray(
            ["a", "b", "c"]
        )

        result = (
            run._partition_closed_set_cross_session_samples(
                x_train,
                y_train,
                x_probe,
                y_probe,
                x_enrollment=x_enroll,
                y_enrollment=y_enroll,
            )
        )

        np.testing.assert_array_equal(
            result["subjects"],
            np.asarray(["a", "b"]),
        )

        np.testing.assert_array_equal(
            result["train"][0].reshape(-1),
            np.asarray(
                [10, 11, 20, 21],
                dtype=np.float32,
            ),
        )

        np.testing.assert_array_equal(
            result["enrollment"][0].reshape(-1),
            np.asarray(
                [100, 200],
                dtype=np.float32,
            ),
        )

        np.testing.assert_array_equal(
            result["probe"][0].reshape(-1),
            np.asarray(
                [1000, 2000],
                dtype=np.float32,
            ),
        )

        self.assertIn(
            "c",
            result[
                "dropped_subjects"
            ][
                "missing_enrollment"
            ].tolist(),
        )

    def test_training_and_enrollment_may_be_the_same_source(
        self,
    ):
        x_train = samples(
            [1, 2, 3, 4]
        )
        y_train = np.asarray(
            ["a", "a", "b", "b"]
        )

        x_probe = samples(
            [10, 20]
        )
        y_probe = np.asarray(
            ["a", "b"]
        )

        result = (
            run._partition_closed_set_cross_session_samples(
                x_train,
                y_train,
                x_probe,
                y_probe,
            )
        )

        self.assertTrue(
            result[
                "enrollment_reuses_training"
            ]
        )

        np.testing.assert_array_equal(
            result["train"][0],
            result["enrollment"][0],
        )

        np.testing.assert_array_equal(
            result["train"][1],
            result["enrollment"][1],
        )

    def test_explicit_train_equal_enrollment_matches_fallback(
        self,
    ):
        x_train = samples(
            [1, 2, 3, 4]
        )
        y_train = np.asarray(
            ["a", "a", "b", "b"]
        )

        x_probe = samples(
            [10, 20]
        )
        y_probe = np.asarray(
            ["a", "b"]
        )

        fallback = (
            run._partition_closed_set_cross_session_samples(
                x_train,
                y_train,
                x_probe,
                y_probe,
            )
        )

        explicit = (
            run._partition_closed_set_cross_session_samples(
                x_train,
                y_train,
                x_probe,
                y_probe,
                x_enrollment=x_train,
                y_enrollment=y_train,
            )
        )

        for role in (
            "train",
            "enrollment",
            "probe",
        ):
            np.testing.assert_array_equal(
                fallback[role][0],
                explicit[role][0],
            )
            np.testing.assert_array_equal(
                fallback[role][1],
                explicit[role][1],
            )

        np.testing.assert_array_equal(
            fallback["session_1"][0],
            fallback["train"][0],
        )

        np.testing.assert_array_equal(
            fallback["session_2"][0],
            fallback["probe"][0],
        )

    def test_incomplete_enrollment_arguments_are_rejected(
        self,
    ):
        x_train = samples(
            [1, 2, 3, 4]
        )
        y_train = np.asarray(
            ["a", "a", "b", "b"]
        )

        x_probe = samples(
            [10, 20]
        )
        y_probe = np.asarray(
            ["a", "b"]
        )

        with self.assertRaisesRegex(
            ValueError,
            "both be provided or both be omitted",
        ):
            run._partition_closed_set_cross_session_samples(
                x_train,
                y_train,
                x_probe,
                y_probe,
                x_enrollment=samples(
                    [100, 200]
                ),
                y_enrollment=None,
            )


class SubjectDisjointThreeRolePartitionTests(
    unittest.TestCase
):
    def setUp(self):
        self.subjects = np.asarray(
            ["a", "b", "c", "d"]
        )

        self.x_train = samples(
            [10, 20, 30, 40]
        )
        self.y_train = (
            self.subjects.copy()
        )

        self.x_enroll = samples(
            [100, 200, 300, 400]
        )
        self.y_enroll = (
            self.subjects.copy()
        )

        self.x_probe = samples(
            [1000, 2000, 3000, 4000]
        )
        self.y_probe = (
            self.subjects.copy()
        )

    def test_each_role_uses_its_own_physical_array(
        self,
    ):
        result = (
            run._partition_subject_disjoint_cross_session_samples(
                self.x_train,
                self.y_train,
                self.x_probe,
                self.y_probe,
                train_subjects=np.asarray(
                    ["a", "b"]
                ),
                validation_subjects=np.asarray(
                    ["c"]
                ),
                test_subjects=np.asarray(
                    ["d"]
                ),
                x_enrollment=self.x_enroll,
                y_enrollment=self.y_enroll,
            )
        )

        np.testing.assert_array_equal(
            result["train"][0].reshape(-1),
            np.asarray(
                [10, 20],
                dtype=np.float32,
            ),
        )

        np.testing.assert_array_equal(
            result[
                "validation"
            ][0].reshape(-1),
            np.asarray(
                [30],
                dtype=np.float32,
            ),
        )

        np.testing.assert_array_equal(
            result[
                "enrollment"
            ][0].reshape(-1),
            np.asarray(
                [400],
                dtype=np.float32,
            ),
        )

        np.testing.assert_array_equal(
            result["probe"][0].reshape(-1),
            np.asarray(
                [4000],
                dtype=np.float32,
            ),
        )

    def test_subject_missing_enrollment_cannot_enter_matched_cohort(
        self,
    ):
        with self.assertRaisesRegex(
            ValueError,
            "cover exactly",
        ):
            run._partition_subject_disjoint_cross_session_samples(
                self.x_train,
                self.y_train,
                self.x_probe,
                self.y_probe,
                train_subjects=np.asarray(
                    ["a", "b"]
                ),
                validation_subjects=np.asarray(
                    ["c"]
                ),
                test_subjects=np.asarray(
                    ["d"]
                ),
                x_enrollment=self.x_enroll[:3],
                y_enrollment=self.y_enroll[:3],
            )

    def test_fallback_enrollment_uses_training_source_for_test_subjects(
        self,
    ):
        result = (
            run._partition_subject_disjoint_cross_session_samples(
                self.x_train,
                self.y_train,
                self.x_probe,
                self.y_probe,
                train_subjects=np.asarray(
                    ["a", "b"]
                ),
                validation_subjects=np.asarray(
                    ["c"]
                ),
                test_subjects=np.asarray(
                    ["d"]
                ),
            )
        )

        np.testing.assert_array_equal(
            result[
                "enrollment"
            ][0].reshape(-1),
            np.asarray(
                [40],
                dtype=np.float32,
            ),
        )

        np.testing.assert_array_equal(
            result["probe"][0].reshape(-1),
            np.asarray(
                [4000],
                dtype=np.float32,
            ),
        )

    def test_incomplete_enrollment_arguments_are_rejected(
        self,
    ):
        with self.assertRaisesRegex(
            ValueError,
            "both be provided or both be omitted",
        ):
            run._partition_subject_disjoint_cross_session_samples(
                self.x_train,
                self.y_train,
                self.x_probe,
                self.y_probe,
                train_subjects=np.asarray(
                    ["a", "b"]
                ),
                validation_subjects=np.asarray(
                    ["c"]
                ),
                test_subjects=np.asarray(
                    ["d"]
                ),
                x_enrollment=self.x_enroll,
                y_enrollment=None,
            )


if __name__ == "__main__":
    unittest.main()
