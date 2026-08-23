import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run


class TinyECGModel(nn.Module):
    """
    Lightweight model used to exercise the complete task pipelines.

    It implements the same include_top interface expected from the framework's
    production models, while remaining fast enough for routine test execution.
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        include_top=True,
    ):
        super().__init__()

        self.include_top = include_top

        self.feature_extractor = nn.Conv1d(
            in_channels,
            8,
            kernel_size=5,
            padding=2,
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.embedding_layer = nn.Linear(
            8,
            12,
        )

        self.classifier = nn.Linear(
            12,
            num_classes,
        )

    def forward(self, x):
        x = F.relu(
            self.feature_extractor(x)
        )

        x = self.pool(x).squeeze(-1)

        embedding = self.embedding_layer(x)

        if self.include_top:
            return self.classifier(
                embedding
            )

        return embedding


def make_synthetic_ecg_dataset(
    number_of_subjects=8,
    samples_per_subject=12,
    signal_length=64,
    session_shift=0.0,
    seed=123,
):
    """
    Build a small deterministic ECG-like dataset.

    Each subject receives a distinct waveform. Random noise provides
    within-subject variation, while session_shift simulates a change between
    acquisition sessions.
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
        subject_frequency = (
            1.0 + 0.15 * subject_index
        )

        secondary_frequency = (
            2.0 + 0.10 * subject_index
        )

        subject_waveform = (
            np.sin(
                subject_frequency * time_axis
            )
            + 0.20
            * np.cos(
                secondary_frequency * time_axis
            )
            + session_shift
        )

        for _ in range(samples_per_subject):
            noise = random_generator.normal(
                loc=0.0,
                scale=0.05,
                size=signal_length,
            )

            sample = (
                subject_waveform + noise
            ).astype(np.float32)

            samples.append(sample)
            labels.append(
                f"subject_{subject_index}"
            )

    return (
        np.stack(samples),
        np.asarray(labels),
    )


class AllTaskSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_thread_count = (
            torch.get_num_threads()
        )

        # These tests contain very small tensors. A single thread avoids
        # unnecessary thread-management overhead in CI and local runs.
        torch.set_num_threads(1)

        cls.intra_x, cls.intra_y = (
            make_synthetic_ecg_dataset(
                number_of_subjects=8,
                samples_per_subject=12,
                signal_length=64,
                session_shift=0.0,
                seed=123,
            )
        )

        cls.session_1_x, cls.session_1_y = (
            make_synthetic_ecg_dataset(
                number_of_subjects=8,
                samples_per_subject=10,
                signal_length=64,
                session_shift=0.0,
                seed=456,
            )
        )

        cls.session_2_x, cls.session_2_y = (
            make_synthetic_ecg_dataset(
                number_of_subjects=8,
                samples_per_subject=8,
                signal_length=64,
                session_shift=0.10,
                seed=789,
            )
        )

        cls.common_arguments = {
            "model_class": TinyECGModel,
            "epochs": 1,
            "batch_size": 16,
            "lr": 1e-3,
            "val_split": 0.0,
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

    def assert_common_runner_metadata(
        self,
        data_statistics,
        hyperparameters,
    ):
        self.assertIsInstance(
            data_statistics,
            dict,
        )
        self.assertTrue(
            data_statistics,
        )

        self.assertIsInstance(
            hyperparameters,
            dict,
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
        self.assertGreater(
            hyperparameters[
                "Trainable Model Parameters"
            ],
            0,
        )
        self.assertGreater(
            hyperparameters[
                "Model State Size (MiB)"
            ],
            0.0,
        )


    def assert_identification_result(
        self,
        runner_result,
    ):
        metrics, data_statistics, hyperparameters = (
            runner_result
        )

        self.assertEqual(
            len(metrics),
            2,
        )

        rank_1 = float(
            metrics[0]
        )

        self.assertTrue(
            np.isfinite(
                rank_1
            )
        )
        self.assertGreaterEqual(
            rank_1,
            0.0,
        )
        self.assertLessEqual(
            rank_1,
            1.0,
        )

        rank_5 = metrics[1]

        if rank_5 is not None:
            rank_5 = float(
                rank_5
            )

            self.assertTrue(
                np.isfinite(
                    rank_5
                )
            )
            self.assertGreaterEqual(
                rank_5,
                0.0,
            )
            self.assertLessEqual(
                rank_5,
                1.0,
            )
            self.assertGreaterEqual(
                rank_5,
                rank_1,
            )

        self.assert_common_runner_metadata(
            data_statistics,
            hyperparameters,
        )


    def assert_verification_result(
        self,
        runner_result,
    ):
        metrics, data_statistics, hyperparameters = (
            runner_result
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

        self.assert_common_runner_metadata(
            data_statistics,
            hyperparameters,
        )

    def test_task_1_closed_set_identification(self):
        result = run.run_closed_set_identification(
            self.intra_x,
            self.intra_y,
            test_split=0.25,
            use_template=False,
            probe_fusion_size=1,
            **self.common_arguments,
        )

        self.assert_identification_result(
            result
        )

    def test_task_2_closed_set_verification(self):
        result = run.run_closed_set_verification(
            self.intra_x,
            self.intra_y,
            test_split=0.25,
            num_pairs=40,
            sampling_mode="all",
            use_template=False,
            use_deployment_evaluation=False,
            **self.common_arguments,
        )

        self.assert_verification_result(
            result
        )

    def test_task_3_subject_disjoint_identification(self):
        result = (
            run.run_subject_disjoint_identification(
                self.intra_x,
                self.intra_y,
                test_split=0.25,
                use_template=True,
                template_fusion_method="mean",
                template_size=2,
                probe_fusion_size=1,
                **self.common_arguments,
            )
        )

        self.assert_identification_result(
            result
        )

        data_statistics = result[1]

        self.assertGreater(
            data_statistics["Train Subjects"],
            0,
        )
        self.assertGreaterEqual(
            data_statistics["Test Subjects"],
            2,
        )
        self.assertGreater(
            data_statistics["Gallery Size"],
            0,
        )
        self.assertGreater(
            data_statistics["Probe Samples"],
            0,
        )

    def test_task_4_subject_disjoint_verification(self):
        result = (
            run.run_subject_disjoint_verification(
                self.intra_x,
                self.intra_y,
                test_split=0.25,
                num_pairs=40,
                sampling_mode="all",
                use_template=True,
                template_fusion_method="mean",
                template_size=2,
                use_deployment_evaluation=False,
                **self.common_arguments,
            )
        )

        self.assert_verification_result(
            result
        )

        data_statistics = result[1]

        self.assertGreater(
            data_statistics["Train Subjects"],
            0,
        )
        self.assertGreaterEqual(
            data_statistics["Test Subjects"],
            2,
        )

    def test_task_5_cross_session_identification(self):
        result = run.run_cross_session_identification(
            self.session_1_x,
            self.session_1_y,
            self.session_2_x,
            self.session_2_y,
            use_template=False,
            probe_fusion_size=1,
            **self.common_arguments,
        )

        self.assert_identification_result(
            result
        )

        data_statistics = result[1]

        self.assertEqual(
            data_statistics[
                "Total Cross-Session Subjects"
            ],
            8,
        )
        self.assertEqual(
            data_statistics[
                "Enrollment Samples"
            ],
            0,
        )
        self.assertGreater(
            data_statistics[
                "Probe Samples"
            ],
            0,
        )

    def test_task_6_cross_session_verification(self):
        result = run.run_cross_session_verification(
            self.session_1_x,
            self.session_1_y,
            self.session_2_x,
            self.session_2_y,
            num_pairs=40,
            sampling_mode="all",
            use_template=False,
            use_deployment_evaluation=False,
            **self.common_arguments,
        )

        self.assert_verification_result(
            result
        )

        self.assertEqual(
            result[1][
                "Total Cross-Session Subjects"
            ],
            8,
        )

    def test_task_7_subject_disjoint_cross_session_identification(
        self,
    ):
        result = (
            run.run_subject_disjoint_cross_session_identification(
                self.session_1_x,
                self.session_1_y,
                self.session_2_x,
                self.session_2_y,
                test_split=0.25,
                use_template=True,
                template_fusion_method="mean",
                template_size=2,
                probe_fusion_size=1,
                **self.common_arguments,
            )
        )

        self.assert_identification_result(
            result
        )

        data_statistics = result[1]

        self.assertGreater(
            data_statistics["Train Subjects"],
            0,
        )
        self.assertGreaterEqual(
            data_statistics["Test Subjects"],
            2,
        )
        self.assertGreater(
            data_statistics[
                "Enrollment Samples"
            ],
            0,
        )
        self.assertGreater(
            data_statistics[
                "Probe Samples"
            ],
            0,
        )

    def test_task_8_subject_disjoint_cross_session_verification(
        self,
    ):
        result = (
            run.run_subject_disjoint_cross_session_verification(
                self.session_1_x,
                self.session_1_y,
                self.session_2_x,
                self.session_2_y,
                test_split=0.25,
                num_pairs=40,
                sampling_mode="all",
                use_template=True,
                template_fusion_method="mean",
                template_size=2,
                use_deployment_evaluation=False,
                **self.common_arguments,
            )
        )

        self.assert_verification_result(
            result
        )

        data_statistics = result[1]

        self.assertGreater(
            data_statistics["Train Subjects"],
            0,
        )
        self.assertGreaterEqual(
            data_statistics["Test Subjects"],
            2,
        )
        self.assertGreater(
            data_statistics[
                "Enrollment Samples"
            ],
            0,
        )
        self.assertGreater(
            data_statistics[
                "Probe Samples"
            ],
            0,
        )


if __name__ == "__main__":
    unittest.main()