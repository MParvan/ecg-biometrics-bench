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
from utils import _apply_outlier_filter


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
            nn.AdaptiveAvgPool1d(1),
        )

        self.embedding_layer = nn.Linear(
            8,
            12,
        )

        self.classifier = nn.Linear(
            12,
            num_classes,
        )

    def forward(self, x):
        x = self.features(x).squeeze(-1)
        embedding = self.embedding_layer(x)

        if self.include_top:
            return self.classifier(
                embedding
            )

        return embedding


def make_synthetic_ecg_dataset(
    number_of_subjects=8,
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
                    + 0.15 * subject_index
                )
                * time_axis
            )
            + 0.20
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


def get_subject_ranking_argument(
    filter_call,
):
    """
    Resolve apply_subject_ranking from a mock call.

    Calls that omit the argument use the helper default of True.
    """
    if (
        "apply_subject_ranking"
        in filter_call.kwargs
    ):
        return filter_call.kwargs[
            "apply_subject_ranking"
        ]

    if len(filter_call.args) >= 6:
        return filter_call.args[5]

    return True


class OutlierFilterContractTests(
    unittest.TestCase
):
    def setUp(self):
        self.samples = np.arange(
            6,
            dtype=np.float32,
        ).reshape(6, 1)

        self.labels = np.asarray(
            [
                "subject_a",
                "subject_a",
                "subject_a",
                "subject_b",
                "subject_b",
                "subject_b",
            ]
        )

        self.sqi_scores = np.asarray(
            [
                0.90,
                0.10,
                0.80,
                0.20,
                0.70,
                0.60,
            ],
            dtype=np.float64,
        )

    def test_identity_independent_filter_uses_only_threshold(
        self,
    ):
        filtered_x, filtered_y = (
            _apply_outlier_filter(
                self.samples,
                self.labels,
                self.sqi_scores,
                absolute_threshold=0.50,
                keep_percentage=0.25,
                apply_subject_ranking=False,
            )
        )

        np.testing.assert_array_equal(
            filtered_x.reshape(-1),
            np.asarray(
                [0.0, 2.0, 4.0, 5.0],
                dtype=np.float32,
            ),
        )

        np.testing.assert_array_equal(
            filtered_y,
            self.labels[
                [0, 2, 4, 5]
            ],
        )

    def test_identity_independent_result_does_not_depend_on_labels(
        self,
    ):
        alternative_labels = np.asarray(
            [
                "x",
                "y",
                "x",
                "y",
                "x",
                "y",
            ]
        )

        first_x, _ = _apply_outlier_filter(
            self.samples,
            self.labels,
            self.sqi_scores,
            absolute_threshold=0.50,
            keep_percentage=0.25,
            apply_subject_ranking=False,
        )

        second_x, _ = _apply_outlier_filter(
            self.samples,
            alternative_labels,
            self.sqi_scores,
            absolute_threshold=0.50,
            keep_percentage=0.25,
            apply_subject_ranking=False,
        )

        np.testing.assert_array_equal(
            first_x,
            second_x,
        )

    def test_training_filter_keeps_best_sample_per_subject(
        self,
    ):
        filtered_x, filtered_y = (
            _apply_outlier_filter(
                self.samples,
                self.labels,
                self.sqi_scores,
                absolute_threshold=0.0,
                keep_percentage=0.50,
                apply_subject_ranking=True,
            )
        )

        # Each subject has three samples. int(3 * 0.5) is 1,
        # so only the highest-SQI sample from each subject remains.
        np.testing.assert_array_equal(
            filtered_x.reshape(-1),
            np.asarray(
                [0.0, 4.0],
                dtype=np.float32,
            ),
        )

        np.testing.assert_array_equal(
            filtered_y,
            np.asarray(
                [
                    "subject_a",
                    "subject_b",
                ]
            ),
        )

    def test_filter_preserves_original_chronological_order(
        self,
    ):
        filtered_x, _ = _apply_outlier_filter(
            self.samples,
            self.labels,
            self.sqi_scores,
            absolute_threshold=0.0,
            keep_percentage=2.0 / 3.0,
            apply_subject_ranking=True,
        )

        selected_indices = (
            filtered_x.reshape(-1)
        )

        np.testing.assert_array_equal(
            selected_indices,
            np.sort(selected_indices),
        )

    def test_mismatched_sample_and_label_lengths_are_rejected(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            _apply_outlier_filter(
                self.samples,
                self.labels[:-1],
                self.sqi_scores,
            )

    def test_mismatched_sqi_length_is_rejected(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            _apply_outlier_filter(
                self.samples,
                self.labels,
                self.sqi_scores[:-1],
            )

    def test_invalid_absolute_threshold_is_rejected(
        self,
    ):
        for threshold in [
            -0.01,
            1.01,
        ]:
            with self.subTest(
                threshold=threshold
            ):
                with self.assertRaises(
                    ValueError
                ):
                    _apply_outlier_filter(
                        self.samples,
                        self.labels,
                        self.sqi_scores,
                        absolute_threshold=threshold,
                    )

    def test_invalid_keep_percentage_is_rejected(
        self,
    ):
        for keep_percentage in [
            0.0,
            -0.1,
            1.1,
        ]:
            with self.subTest(
                keep_percentage=(
                    keep_percentage
                )
            ):
                with self.assertRaises(
                    ValueError
                ):
                    _apply_outlier_filter(
                        self.samples,
                        self.labels,
                        self.sqi_scores,
                        keep_percentage=(
                            keep_percentage
                        ),
                    )


class RunnerSQIIntegrationTests(
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
                number_of_subjects=8,
                samples_per_subject=10,
                signal_length=64,
                session_shift=0.0,
                seed=100,
            )
        )

        cls.session_1_x, cls.session_1_y = (
            make_synthetic_ecg_dataset(
                number_of_subjects=8,
                samples_per_subject=10,
                signal_length=64,
                session_shift=0.0,
                seed=200,
            )
        )

        cls.session_2_x, cls.session_2_y = (
            make_synthetic_ecg_dataset(
                number_of_subjects=8,
                samples_per_subject=8,
                signal_length=64,
                session_shift=0.10,
                seed=300,
            )
        )

        cls.intra_sqi = np.ones(
            len(cls.intra_x),
            dtype=np.float64,
        )

        cls.session_1_sqi = np.ones(
            len(cls.session_1_x),
            dtype=np.float64,
        )

        cls.session_2_sqi = np.ones(
            len(cls.session_2_x),
            dtype=np.float64,
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
            "outlier_filtering_on_train": True,
            "outlier_filtering_on_test": True,
            "sqi_threshold": 0.0,
            "sqi_keep_pct": 1.0,
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

    def assert_finite_runner_result(
        self,
        result,
    ):
        metrics, data_statistics, hyperparameters = (
            result
        )

        if (
            len(metrics) == 4
            and metrics[3] is None
        ):
            metric_values = np.asarray(
                metrics[:3],
                dtype=np.float64,
            )
        else:
            metric_values = np.asarray(
                metrics,
                dtype=np.float64,
            )

        self.assertTrue(
            np.isfinite(
                metric_values
            ).all()
        )

        self.assertIsInstance(
            data_statistics,
            dict,
        )

        self.assertIsInstance(
            hyperparameters,
            dict,
        )

    def execute_and_check_filter_roles(
        self,
        experiment,
    ):
        """
        Verify that controlled data uses subject ranking and probe data does not.
        """
        with patch.object(
            run,
            "_apply_outlier_filter",
            wraps=run._apply_outlier_filter,
        ) as filter_mock:
            result = experiment()

        self.assertEqual(
            len(filter_mock.call_args_list),
            2,
            (
                "Expected one controlled training/enrollment "
                "filter call and one operational probe filter call."
            ),
        )

        controlled_call = (
            filter_mock.call_args_list[0]
        )

        probe_call = (
            filter_mock.call_args_list[1]
        )

        self.assertTrue(
            get_subject_ranking_argument(
                controlled_call
            ),
            (
                "Controlled training/enrollment filtering "
                "should retain per-subject SQI ranking."
            ),
        )

        self.assertFalse(
            get_subject_ranking_argument(
                probe_call
            ),
            (
                "Probe filtering must be identity-independent "
                "and must set apply_subject_ranking=False."
            ),
        )

        self.assert_finite_runner_result(
            result
        )

    def test_all_eight_tasks_use_identity_independent_probe_filtering(
        self,
    ):
        experiments = [
            (
                "task_1",
                lambda: (
                    run.run_closed_set_identification(
                        self.intra_x,
                        self.intra_y,
                        test_split=0.25,
                        use_template=False,
                        sqi_scores=self.intra_sqi,
                        probe_fusion_size=1,
                        **self.common_arguments,
                    )
                ),
            ),
            (
                "task_2",
                lambda: (
                    run.run_closed_set_verification(
                        self.intra_x,
                        self.intra_y,
                        test_split=0.25,
                        num_pairs=40,
                        sampling_mode="all",
                        use_template=False,
                        sqi_scores=self.intra_sqi,
                        use_deployment_evaluation=False,
                        **self.common_arguments,
                    )
                ),
            ),
            (
                "task_3",
                lambda: (
                    run.run_subject_disjoint_identification(
                        self.intra_x,
                        self.intra_y,
                        test_split=0.25,
                        use_template=True,
                        template_fusion_method="mean",
                        template_size=2,
                        sqi_scores=self.intra_sqi,
                        probe_fusion_size=1,
                        **self.common_arguments,
                    )
                ),
            ),
            (
                "task_4",
                lambda: (
                    run.run_subject_disjoint_verification(
                        self.intra_x,
                        self.intra_y,
                        test_split=0.25,
                        num_pairs=40,
                        sampling_mode="all",
                        use_template=True,
                        template_fusion_method="mean",
                        template_size=2,
                        sqi_scores=self.intra_sqi,
                        use_deployment_evaluation=False,
                        **self.common_arguments,
                    )
                ),
            ),
            (
                "task_5",
                lambda: (
                    run.run_cross_session_identification(
                        self.session_1_x,
                        self.session_1_y,
                        self.session_2_x,
                        self.session_2_y,
                        use_template=False,
                        sqi_train=(
                            self.session_1_sqi
                        ),
                        sqi_test=(
                            self.session_2_sqi
                        ),
                        probe_fusion_size=1,
                        **self.common_arguments,
                    )
                ),
            ),
            (
                "task_6",
                lambda: (
                    run.run_cross_session_verification(
                        self.session_1_x,
                        self.session_1_y,
                        self.session_2_x,
                        self.session_2_y,
                        num_pairs=40,
                        sampling_mode="all",
                        use_template=False,
                        sqi_train=(
                            self.session_1_sqi
                        ),
                        sqi_test=(
                            self.session_2_sqi
                        ),
                        use_deployment_evaluation=False,
                        **self.common_arguments,
                    )
                ),
            ),
            (
                "task_7",
                lambda: (
                    run.run_subject_disjoint_cross_session_identification(
                        self.session_1_x,
                        self.session_1_y,
                        self.session_2_x,
                        self.session_2_y,
                        test_split=0.25,
                        use_template=True,
                        template_fusion_method="mean",
                        template_size=2,
                        sqi_s1=(
                            self.session_1_sqi
                        ),
                        sqi_s2=(
                            self.session_2_sqi
                        ),
                        probe_fusion_size=1,
                        **self.common_arguments,
                    )
                ),
            ),
            (
                "task_8",
                lambda: (
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
                        sqi_s1=(
                            self.session_1_sqi
                        ),
                        sqi_s2=(
                            self.session_2_sqi
                        ),
                        use_deployment_evaluation=False,
                        **self.common_arguments,
                    )
                ),
            ),
        ]

        for task_name, experiment in experiments:
            with self.subTest(
                task=task_name
            ):
                self.execute_and_check_filter_roles(
                    experiment
                )

    def test_string_sqi_method_is_computed_and_applied(
        self,
    ):
        computed_scores = np.ones(
            len(self.intra_x),
            dtype=np.float64,
        )

        with patch.object(
            run,
            "_compute_sqi",
            return_value=computed_scores,
        ) as compute_mock:
            result = (
                run.run_closed_set_identification(
                    self.intra_x,
                    self.intra_y,
                    model_class=TinyECGModel,
                    epochs=1,
                    batch_size=16,
                    lr=1e-3,
                    test_split=0.25,
                    val_split=0.0,
                    seed=42,
                    device="cpu",
                    visualize=False,
                    use_template=False,
                    outlier_filtering_on_train=True,
                    outlier_filtering_on_test=True,
                    sqi_scores="kurtosis",
                    sqi_threshold=0.0,
                    sqi_keep_pct=1.0,
                    probe_fusion_size=1,
                    save_results_and_settings=False,
                    loader=None,
                    n_runs=1,
                    _return_stats=True,
                    intelligent_weight_loading=False,
                )
            )

        compute_mock.assert_called_once()

        called_signal = (
            compute_mock.call_args.args[0]
        )

        called_method = (
            compute_mock.call_args.kwargs[
                "method"
            ]
        )

        self.assertIs(
            called_signal,
            self.intra_x,
        )

        self.assertEqual(
            called_method,
            "kurtosis",
        )

        self.assert_finite_runner_result(
            result
        )


if __name__ == "__main__":
    unittest.main()