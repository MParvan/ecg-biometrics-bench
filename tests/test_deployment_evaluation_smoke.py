import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run


class TinyECGModel(nn.Module):
    """
    Lightweight model implementing the production include_top interface.
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        include_top=True,
    ):
        super().__init__()

        self.include_top = include_top

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels,
                8,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                8,
                12,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.embedding_layer = nn.Linear(
            12,
            16,
        )

        self.classifier = nn.Linear(
            16,
            num_classes,
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x).squeeze(-1)

        embedding = self.embedding_layer(x)

        if self.include_top:
            return self.classifier(
                embedding
            )

        return embedding


def make_synthetic_ecg_dataset(
    number_of_subjects=12,
    samples_per_subject=12,
    signal_length=64,
    session_shift=0.0,
    seed=123,
):
    """
    Create deterministic subject-specific ECG-like waveforms.
    """
    random_generator = np.random.default_rng(
        seed
    )

    time_axis = np.linspace(
        0.0,
        2.0 * np.pi,
        signal_length,
        endpoint=False,
    )

    samples = []
    labels = []

    for subject_index in range(
        number_of_subjects
    ):
        subject_waveform = (
            np.sin(
                (
                    1.0
                    + 0.12 * subject_index
                )
                * time_axis
            )
            + 0.25
            * np.cos(
                (
                    2.0
                    + 0.08 * subject_index
                )
                * time_axis
            )
            + 0.04 * subject_index
            + session_shift
        )

        for _ in range(samples_per_subject):
            noise = random_generator.normal(
                loc=0.0,
                scale=0.04,
                size=signal_length,
            )

            samples.append(
                (
                    subject_waveform + noise
                ).astype(np.float32)
            )

            labels.append(
                f"subject_{subject_index}"
            )

    return (
        np.stack(samples),
        np.asarray(labels),
    )


class DeploymentEvaluationSmokeTests(
    unittest.TestCase
):
    @classmethod
    def setUpClass(cls):
        cls.previous_thread_count = (
            torch.get_num_threads()
        )

        torch.set_num_threads(1)

        cls.intra_x, cls.intra_y = (
            make_synthetic_ecg_dataset(
                number_of_subjects=12,
                samples_per_subject=12,
                signal_length=64,
                session_shift=0.0,
                seed=100,
            )
        )

        cls.session_1_x, cls.session_1_y = (
            make_synthetic_ecg_dataset(
                number_of_subjects=12,
                samples_per_subject=10,
                signal_length=64,
                session_shift=0.0,
                seed=200,
            )
        )

        cls.session_2_x, cls.session_2_y = (
            make_synthetic_ecg_dataset(
                number_of_subjects=12,
                samples_per_subject=8,
                signal_length=64,
                session_shift=0.10,
                seed=300,
            )
        )

        cls.common_arguments = {
            "model_class": TinyECGModel,
            "epochs": 1,
            "batch_size": 16,
            "lr": 1e-3,
            "val_split": 0.25,
            "seed": 42,
            "device": "cpu",
            "visualize": False,
            "save_results_and_settings": False,
            "loader": None,
            "n_runs": 1,
            "_return_stats": True,
            "intelligent_weight_loading": False,
        }

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(
            cls.previous_thread_count
        )

    def assert_verification_result(
        self,
        result,
    ):
        metrics, data_statistics, hyperparameters = (
            result
        )

        self.assertEqual(
            len(metrics),
            4,
        )

        eer = float(metrics[0])
        auc_value = float(metrics[1])
        d_prime = float(metrics[2])

        tar_at_far = (
            None
            if metrics[3] is None
            else float(metrics[3])
        )

        for metric in [
            eer,
            auc_value,
            d_prime,
        ]:
            self.assertTrue(
                np.isfinite(metric)
            )

        if tar_at_far is not None:
            self.assertTrue(
                np.isfinite(
                    tar_at_far
                )
            )

        self.assertGreaterEqual(
            eer,
            0.0,
        )
        self.assertLessEqual(
            eer,
            1.0,
        )

        self.assertGreaterEqual(
            auc_value,
            0.0,
        )
        self.assertLessEqual(
            auc_value,
            1.0,
        )

        self.assertGreaterEqual(
            d_prime,
            0.0,
        )

        if tar_at_far is not None:
            self.assertGreaterEqual(
                tar_at_far,
                0.0,
            )

            self.assertLessEqual(
                tar_at_far,
                1.0,
            )

        self.assertGreater(
            data_statistics[
                "Total Verification Pairs"
            ],
            0,
        )

        self.assertGreater(
            data_statistics[
                "Genuine Pairs"
            ],
            0,
        )

        self.assertGreater(
            data_statistics[
                "Impostor Pairs"
            ],
            0,
        )

        validation_values = []

        for key, value in data_statistics.items():
            if "Validation" not in str(key):
                continue

            if isinstance(
                value,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            ):
                validation_values.append(
                    float(value)
                )

        self.assertTrue(
            validation_values,
            (
                "The runner did not report any validation "
                "statistics."
            ),
        )

        self.assertTrue(
            any(
                value > 0
                for value in validation_values
            ),
            (
                "The runner reported no non-empty validation "
                "partition."
            ),
        )

        self.assertEqual(
            hyperparameters["base_seed"],
            42,
        )

        self.assertEqual(
            hyperparameters["n_runs"],
            1,
        )

        self.assertEqual(
            hyperparameters["run_seeds"],
            [42],
        )

        self.assertGreater(
            hyperparameters[
                "Total Model Parameters"
            ],
            0,
        )

    def execute_and_assert_calibration(
        self,
        experiment,
    ):
        with patch.object(
            run,
            "_find_optimal_threshold",
            wraps=run._find_optimal_threshold,
        ) as threshold_mock, patch.object(
            run,
            "_evaluate_with_global_threshold",
            wraps=run._evaluate_with_global_threshold,
        ) as evaluation_mock:
            result = experiment()

        threshold_mock.assert_called_once()
        evaluation_mock.assert_called_once()

        calibration_scores = (
            threshold_mock.call_args.args[0]
        )

        calibration_labels = (
            threshold_mock.call_args.args[1]
        )

        self.assertGreater(
            len(calibration_scores),
            0,
        )

        self.assertEqual(
            len(calibration_scores),
            len(calibration_labels),
        )

        self.assertIn(
            0,
            np.unique(calibration_labels),
        )

        self.assertIn(
            1,
            np.unique(calibration_labels),
        )

        applied_test_scores = (
            evaluation_mock.call_args.args[0]
        )

        applied_test_labels = (
            evaluation_mock.call_args.args[1]
        )

        applied_threshold = (
            evaluation_mock.call_args.args[2]
        )

        self.assertGreater(
            len(applied_test_scores),
            0,
        )

        self.assertEqual(
            len(applied_test_scores),
            len(applied_test_labels),
        )

        self.assertTrue(
            np.isfinite(
                applied_threshold
            )
        )

        self.assert_verification_result(
            result
        )

    def test_task_2_closed_set_deployment_evaluation(
        self,
    ):
        self.execute_and_assert_calibration(
            lambda: run.run_closed_set_verification(
                self.intra_x,
                self.intra_y,
                test_split=0.25,
                num_pairs=60,
                sampling_mode="balanced",
                use_template=True,
                template_fusion_method="mean",
                template_size=2,
                matching_method="cosine",
                use_deployment_evaluation=True,
                **self.common_arguments,
            )
        )

    def test_task_4_subject_disjoint_deployment_evaluation(
        self,
    ):
        self.execute_and_assert_calibration(
            lambda: (
                run.run_subject_disjoint_verification(
                    self.intra_x,
                    self.intra_y,
                    test_split=0.25,
                    num_pairs=60,
                    sampling_mode="balanced",
                    use_template=True,
                    template_fusion_method="mean",
                    template_size=2,
                    matching_method="cosine",
                    use_deployment_evaluation=True,
                    **self.common_arguments,
                )
            )
        )

    def test_task_6_cross_session_deployment_evaluation(
        self,
    ):
        self.execute_and_assert_calibration(
            lambda: run.run_cross_session_verification(
                self.session_1_x,
                self.session_1_y,
                self.session_2_x,
                self.session_2_y,
                num_pairs=60,
                sampling_mode="balanced",
                use_template=True,
                template_fusion_method="mean",
                template_size=2,
                matching_method="cosine",
                use_deployment_evaluation=True,
                **self.common_arguments,
            )
        )

    def test_task_8_subject_disjoint_cross_session_deployment_evaluation(
        self,
    ):
        self.execute_and_assert_calibration(
            lambda: (
                run.run_subject_disjoint_cross_session_verification(
                    self.session_1_x,
                    self.session_1_y,
                    self.session_2_x,
                    self.session_2_y,
                    test_split=0.25,
                    num_pairs=60,
                    sampling_mode="balanced",
                    use_template=True,
                    template_fusion_method="mean",
                    template_size=2,
                    matching_method="cosine",
                    use_deployment_evaluation=True,
                    **self.common_arguments,
                )
            )
        )


if __name__ == "__main__":
    unittest.main()