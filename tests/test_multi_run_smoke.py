import sys
import unittest
from pathlib import Path
from unittest.mock import call, patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run


class TinyECGModel(nn.Module):
    """
    Lightweight model implementing the framework's include_top interface.
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        include_top=True,
    ):
        super().__init__()

        self.include_top = include_top

        self.features = nn.Sequential(
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
        x = self.features(x)
        x = self.pool(x).squeeze(-1)

        embedding = self.embedding_layer(x)

        if self.include_top:
            return self.classifier(
                embedding
            )

        return embedding


def make_synthetic_ecg_dataset(
    number_of_subjects=10,
    samples_per_subject=10,
    signal_length=64,
    session_shift=0.0,
    seed=123,
):
    """
    Create deterministic subject-specific ECG-like samples.
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
        base_waveform = (
            np.sin(
                (
                    1.0
                    + 0.13 * subject_index
                )
                * time_axis
            )
            + 0.25
            * np.cos(
                (
                    2.0
                    + 0.07 * subject_index
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
                    base_waveform + noise
                ).astype(np.float32)
            )

            labels.append(
                f"subject_{subject_index}"
            )

    return (
        np.stack(samples),
        np.asarray(labels),
    )


class MultiRunSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_thread_count = (
            torch.get_num_threads()
        )

        torch.set_num_threads(1)

        cls.intra_x, cls.intra_y = (
            make_synthetic_ecg_dataset(
                number_of_subjects=10,
                samples_per_subject=10,
                signal_length=64,
                session_shift=0.0,
                seed=100,
            )
        )

        cls.session_1_x, cls.session_1_y = (
            make_synthetic_ecg_dataset(
                number_of_subjects=10,
                samples_per_subject=8,
                signal_length=64,
                session_shift=0.0,
                seed=200,
            )
        )

        cls.session_2_x, cls.session_2_y = (
            make_synthetic_ecg_dataset(
                number_of_subjects=10,
                samples_per_subject=6,
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
            "val_split": 0.0,
            "seed": 40,
            "n_runs": 2,
            "device": "cpu",
            "visualize": False,
            "save_results_and_settings": False,
            "loader": None,
            "intelligent_weight_loading": False,
        }

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(
            cls.previous_thread_count
        )

    def assert_identification_aggregate(
        self,
        result,
    ):
        self.assertEqual(
            len(result),
            2,
        )

        for metric_index, metric_summary in enumerate(
            result
        ):
            self.assertEqual(
                len(metric_summary),
                2,
            )

            mean_value = float(
                metric_summary[0]
            )

            standard_deviation = float(
                metric_summary[1]
            )

            self.assertTrue(
                np.isfinite(mean_value)
            )

            self.assertTrue(
                np.isfinite(
                    standard_deviation
                )
            )

            self.assertGreaterEqual(
                mean_value,
                0.0,
            )

            self.assertLessEqual(
                mean_value,
                1.0,
            )

            self.assertGreaterEqual(
                standard_deviation,
                0.0,
            )

            self.assertLessEqual(
                standard_deviation,
                1.0,
            )

        rank_1_mean = float(
            result[0][0]
        )

        rank_5_mean = float(
            result[1][0]
        )

        self.assertGreaterEqual(
            rank_5_mean,
            rank_1_mean,
        )

    def assert_verification_aggregate(
        self,
        result,
    ):
        self.assertEqual(
            len(result),
            4,
        )

        for metric_summary in result:
            self.assertEqual(
                len(metric_summary),
                2,
            )

            mean_value = float(
                metric_summary[0]
            )

            standard_deviation = float(
                metric_summary[1]
            )

            self.assertTrue(
                np.isfinite(mean_value)
            )

            self.assertTrue(
                np.isfinite(
                    standard_deviation
                )
            )

            self.assertGreaterEqual(
                standard_deviation,
                0.0,
            )

        eer_mean = float(
            result[0][0]
        )

        auc_mean = float(
            result[1][0]
        )

        d_prime_mean = float(
            result[2][0]
        )

        tar_mean = float(
            result[3][0]
        )

        self.assertGreaterEqual(
            eer_mean,
            0.0,
        )
        self.assertLessEqual(
            eer_mean,
            1.0,
        )

        self.assertGreaterEqual(
            auc_mean,
            0.0,
        )
        self.assertLessEqual(
            auc_mean,
            1.0,
        )

        self.assertGreaterEqual(
            d_prime_mean,
            0.0,
        )

        self.assertGreaterEqual(
            tar_mean,
            0.0,
        )
        self.assertLessEqual(
            tar_mean,
            1.0,
        )

    def execute_multi_run(
        self,
        experiment,
        result_type,
    ):
        """
        Execute one two-seed experiment while prohibiting cache and log use.
        """
        with patch.object(
            run,
            "_set_seed",
            wraps=run._set_seed,
        ) as seed_mock, patch(
            "utils.CacheManager",
            side_effect=AssertionError(
                "Weight caching was unexpectedly enabled "
                "inside a recursive multi-run execution."
            ),
        ) as cache_manager_mock, patch.object(
            run,
            "_log_experiment_results",
        ) as logger_mock:
            result = experiment()

        self.assertEqual(
            seed_mock.call_args_list,
            [
                call(40),
                call(41),
            ],
        )

        cache_manager_mock.assert_not_called()
        logger_mock.assert_not_called()

        if result_type == "identification":
            self.assert_identification_aggregate(
                result
            )
        elif result_type == "verification":
            self.assert_verification_aggregate(
                result
            )
        else:
            self.fail(
                f"Unknown result type: {result_type}"
            )

    def test_task_1_closed_set_identification_multi_run(
        self,
    ):
        self.execute_multi_run(
            lambda: run.run_closed_set_identification(
                self.intra_x,
                self.intra_y,
                test_split=0.25,
                use_template=False,
                probe_fusion_size=1,
                **self.common_arguments,
            ),
            result_type="identification",
        )

    def test_task_2_closed_set_verification_multi_run(
        self,
    ):
        self.execute_multi_run(
            lambda: run.run_closed_set_verification(
                self.intra_x,
                self.intra_y,
                test_split=0.25,
                num_pairs=40,
                sampling_mode="all",
                use_template=False,
                use_deployment_evaluation=False,
                **self.common_arguments,
            ),
            result_type="verification",
        )

    def test_task_3_subject_disjoint_identification_multi_run(
        self,
    ):
        self.execute_multi_run(
            lambda: (
                run.run_subject_disjoint_identification(
                    self.intra_x,
                    self.intra_y,
                    test_split=0.30,
                    use_template=True,
                    template_fusion_method="mean",
                    template_size=2,
                    probe_fusion_size=1,
                    **self.common_arguments,
                )
            ),
            result_type="identification",
        )

    def test_task_4_subject_disjoint_verification_multi_run(
        self,
    ):
        self.execute_multi_run(
            lambda: (
                run.run_subject_disjoint_verification(
                    self.intra_x,
                    self.intra_y,
                    test_split=0.30,
                    num_pairs=40,
                    sampling_mode="all",
                    use_template=True,
                    template_fusion_method="mean",
                    template_size=2,
                    use_deployment_evaluation=False,
                    **self.common_arguments,
                )
            ),
            result_type="verification",
        )

    def test_task_5_cross_session_identification_multi_run(
        self,
    ):
        self.execute_multi_run(
            lambda: run.run_cross_session_identification(
                self.session_1_x,
                self.session_1_y,
                self.session_2_x,
                self.session_2_y,
                use_template=False,
                probe_fusion_size=1,
                **self.common_arguments,
            ),
            result_type="identification",
        )

    def test_task_6_cross_session_verification_multi_run(
        self,
    ):
        self.execute_multi_run(
            lambda: run.run_cross_session_verification(
                self.session_1_x,
                self.session_1_y,
                self.session_2_x,
                self.session_2_y,
                num_pairs=40,
                sampling_mode="all",
                use_template=False,
                use_deployment_evaluation=False,
                **self.common_arguments,
            ),
            result_type="verification",
        )

    def test_task_7_subject_disjoint_cross_session_identification_multi_run(
        self,
    ):
        self.execute_multi_run(
            lambda: (
                run.run_subject_disjoint_cross_session_identification(
                    self.session_1_x,
                    self.session_1_y,
                    self.session_2_x,
                    self.session_2_y,
                    test_split=0.30,
                    use_template=True,
                    template_fusion_method="mean",
                    template_size=2,
                    probe_fusion_size=1,
                    **self.common_arguments,
                )
            ),
            result_type="identification",
        )

    def test_task_8_subject_disjoint_cross_session_verification_multi_run(
        self,
    ):
        self.execute_multi_run(
            lambda: (
                run.run_subject_disjoint_cross_session_verification(
                    self.session_1_x,
                    self.session_1_y,
                    self.session_2_x,
                    self.session_2_y,
                    test_split=0.30,
                    num_pairs=40,
                    sampling_mode="all",
                    use_template=True,
                    template_fusion_method="mean",
                    template_size=2,
                    use_deployment_evaluation=False,
                    **self.common_arguments,
                )
            ),
            result_type="verification",
        )


if __name__ == "__main__":
    unittest.main()