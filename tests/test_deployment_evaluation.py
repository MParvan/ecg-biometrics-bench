import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from run import (
    _validate_deployment_evaluation,
    run_closed_set_verification,
    run_subject_disjoint_verification,
    run_cross_session_verification,
    run_subject_disjoint_cross_session_verification,
)


class DeploymentEvaluationTests(unittest.TestCase):
    def test_disabled_deployment_allows_zero_validation_split(self):
        _validate_deployment_evaluation(
            use_deployment_evaluation=False,
            val_split=0.0,
            task_name="Test Task",
        )

    def test_enabled_deployment_requires_positive_validation_split(self):
        with self.assertRaisesRegex(
            ValueError,
            "requires 0 < val_split < 1",
        ):
            _validate_deployment_evaluation(
                use_deployment_evaluation=True,
                val_split=0.0,
                task_name="Test Task",
            )

    def test_enabled_deployment_rejects_complete_validation_split(self):
        with self.assertRaisesRegex(
            ValueError,
            "requires 0 < val_split < 1",
        ):
            _validate_deployment_evaluation(
                use_deployment_evaluation=True,
                val_split=1.0,
                task_name="Test Task",
            )

    def test_all_verification_tasks_validate_before_processing_data(self):
        empty_x = np.empty((0, 1), dtype=np.float32)
        empty_y = np.array([], dtype=int)

        task_calls = [
            (
                "Closed-Set Verification",
                lambda: run_closed_set_verification(
                    empty_x,
                    empty_y,
                    model_class=None,
                    val_split=0.0,
                    use_deployment_evaluation=True,
                ),
            ),
            (
                "Subject-Disjoint Verification",
                lambda: run_subject_disjoint_verification(
                    empty_x,
                    empty_y,
                    model_class=None,
                    val_split=0.0,
                    use_deployment_evaluation=True,
                ),
            ),
            (
                "Cross-Session Verification",
                lambda: run_cross_session_verification(
                    empty_x,
                    empty_y,
                    empty_x,
                    empty_y,
                    model_class=None,
                    val_split=0.0,
                    use_deployment_evaluation=True,
                ),
            ),
            (
                "Subject-Disjoint Cross-Session Verification",
                lambda: run_subject_disjoint_cross_session_verification(
                    empty_x,
                    empty_y,
                    empty_x,
                    empty_y,
                    model_class=None,
                    val_split=0.0,
                    use_deployment_evaluation=True,
                ),
            ),
        ]

        for task_name, task_call in task_calls:
            with self.subTest(task=task_name):
                with self.assertRaisesRegex(
                    ValueError,
                    "requires 0 < val_split < 1",
                ):
                    task_call()


if __name__ == "__main__":
    unittest.main()