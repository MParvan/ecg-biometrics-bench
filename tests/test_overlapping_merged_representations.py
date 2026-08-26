import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main
import run
from utils import _validate_merged_representation_partitioning


INTRA_SESSION_RUNNERS = (
    (1, run.run_closed_set_identification),
    (2, run.run_closed_set_verification),
    (3, run.run_subject_disjoint_identification),
    (4, run.run_subject_disjoint_verification),
)

CROSS_SESSION_RUNNERS = (
    (5, run.run_cross_session_identification),
    (6, run.run_cross_session_verification),
    (7, run.run_subject_disjoint_cross_session_identification),
    (8, run.run_subject_disjoint_cross_session_verification),
)


class RecordingConcatModel(nn.Module):
    observed_lengths = []

    def __init__(self, in_channels=1, num_classes=2, include_top=True):
        super().__init__()
        self.include_top = include_top
        self.feature_extractor = nn.Conv1d(
            in_channels,
            6,
            kernel_size=5,
            padding=2,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_layer = nn.Linear(6, 8)
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x):
        type(self).observed_lengths.append(int(x.shape[-1]))
        features = F.relu(self.feature_extractor(x))
        embedding = self.embedding_layer(self.pool(features).squeeze(-1))
        if self.include_top:
            return self.classifier(embedding)
        return embedding


def make_concat_dataset():
    generator = np.random.default_rng(77)
    samples = []
    labels = []
    signal_length = 64 * 3

    for subject_index in range(5):
        base = np.sin(
            np.linspace(
                0.0,
                (2.0 + 0.2 * subject_index) * np.pi,
                signal_length,
                endpoint=False,
            )
        )
        for _ in range(8):
            samples.append(
                (base + generator.normal(0.0, 0.03, signal_length)).astype(
                    np.float32
                )
            )
            labels.append(f"subject_{subject_index}")

    return np.stack(samples), np.asarray(labels)


class OverlappingMergedRepresentationGuardTests(unittest.TestCase):
    def assert_cli_validation_error(self, arguments, expected_message):
        captured_stderr = StringIO()

        with redirect_stderr(captured_stderr):
            with self.assertRaises(SystemExit) as raised:
                main.parse_experiment_arguments(arguments)

        self.assertEqual(raised.exception.code, 2)
        self.assertIn(expected_message, captured_stderr.getvalue())

    def test_tasks_one_through_four_reject_overlap(self):
        for task in range(1, 5):
            with self.subTest(task=task):
                with self.assertRaisesRegex(
                    ValueError,
                    "share raw beats across evaluation partitions",
                ):
                    _validate_merged_representation_partitioning(task, 3, 1)

    def test_single_beats_and_disjoint_windows_are_accepted(self):
        for task in range(1, 5):
            with self.subTest(task=task, case="single"):
                _validate_merged_representation_partitioning(task, 1, 1)
            with self.subTest(task=task, case="disjoint"):
                _validate_merged_representation_partitioning(task, 3, 3)

    def test_cli_accepts_single_beats_and_disjoint_windows(self):
        for merge_width, stride in ((1, 1), (3, 3)):
            with self.subTest(merge_width=merge_width, stride=stride):
                arguments, _ = main.parse_experiment_arguments(
                    [
                        "--dataset",
                        "ecgid",
                        "--task",
                        "1",
                        "--num_beats_to_merge",
                        str(merge_width),
                        "--beat_merge_stride",
                        str(stride),
                    ]
                )
                self.assertEqual(arguments.num_beats_to_merge, merge_width)
                self.assertEqual(arguments.beat_merge_stride, stride)

    def test_tasks_five_through_eight_do_not_gain_overlap_rejection(self):
        for task in range(5, 9):
            with self.subTest(task=task):
                _validate_merged_representation_partitioning(task, 3, 1)

    def test_cli_effective_configuration_rejects_overlap(self):
        for task in range(1, 5):
            with self.subTest(task=task):
                self.assert_cli_validation_error(
                    [
                        "--dataset",
                        "ecgid",
                        "--task",
                        str(task),
                        "--num_beats_to_merge",
                        "3",
                        "--beat_merge_stride",
                        "1",
                    ],
                    f"Task {task} cannot use overlapping merged representations",
                )

    def test_yaml_effective_configuration_rejects_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "overlap.yaml"
            config_path.write_text(
                "dataset: ecgid\n"
                "task: 3\n"
                "num_beats_to_merge: 3\n"
                "beat_merge_stride: 1\n",
                encoding="utf-8",
            )

            self.assert_cli_validation_error(
                ["--config", str(config_path)],
                "Task 3 cannot use overlapping merged representations",
            )

    def test_existing_larger_than_width_rule_is_preserved(self):
        self.assert_cli_validation_error(
            [
                "--dataset",
                "ecgid",
                "--task",
                "1",
                "--num_beats_to_merge",
                "3",
                "--beat_merge_stride",
                "4",
            ],
            "cannot exceed 'num_beats_to_merge'",
        )

    def test_direct_intra_session_runners_fail_before_runtime_setup(self):
        invalid_loader = SimpleNamespace(
            num_beats=3,
            beat_merge_stride=1,
        )

        with patch.object(
            run,
            "_activate_top_level_runtime_profile",
            side_effect=AssertionError("runtime setup must not be reached"),
        ):
            for task, runner in INTRA_SESSION_RUNNERS:
                with self.subTest(task=task):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"Task {task} cannot use overlapping",
                    ):
                        runner(
                            np.empty((0, 64), dtype=np.float32),
                            np.empty((0,), dtype=str),
                            None,
                            loader=invalid_loader,
                        )

    def test_direct_cross_session_runners_reach_runtime_setup(self):
        legacy_overlap_loader = SimpleNamespace(
            num_beats=3,
            beat_merge_stride=1,
        )
        sentinel = RuntimeError("runtime setup reached")

        with patch.object(
            run,
            "_activate_top_level_runtime_profile",
            side_effect=sentinel,
        ):
            for task, runner in CROSS_SESSION_RUNNERS:
                with self.subTest(task=task):
                    with self.assertRaisesRegex(RuntimeError, str(sentinel)):
                        runner(
                            np.empty((0, 64), dtype=np.float32),
                            np.empty((0,), dtype=str),
                            np.empty((0, 64), dtype=np.float32),
                            np.empty((0,), dtype=str),
                            None,
                            loader=legacy_overlap_loader,
                        )


class SafeConcatenatedRunnerSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_thread_count = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_thread_count)

    def test_disjoint_concat_reaches_real_runner_model_and_evaluation(self):
        x, y = make_concat_dataset()
        RecordingConcatModel.observed_lengths = []
        loader = SimpleNamespace(num_beats=3, beat_merge_stride=3)

        metrics, data_statistics, hyperparameters = (
            run.run_closed_set_identification(
                x,
                y,
                model_class=RecordingConcatModel,
                epochs=1,
                batch_size=10,
                lr=1e-3,
                test_split=0.25,
                val_split=0.0,
                seed=42,
                device="cpu",
                visualize=False,
                use_template=True,
                template_fusion_method="mean",
                template_size=2,
                matching_method="cosine",
                outlier_filtering_on_train=False,
                outlier_filtering_on_test=False,
                probe_fusion_size=1,
                save_results_and_settings=False,
                loader=loader,
                n_runs=1,
                _return_stats=True,
                intelligent_weight_loading=False,
            )
        )

        self.assertIn(64 * 3, RecordingConcatModel.observed_lengths)
        self.assertTrue(np.isfinite(float(metrics[0])))
        self.assertGreater(data_statistics["Train Samples"], 0)
        self.assertGreater(hyperparameters["Total Model Parameters"], 0)


if __name__ == "__main__":
    unittest.main()
