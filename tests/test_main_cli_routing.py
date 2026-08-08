import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch, call

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main
from models import DeepECG, ResNet1D


RUNNER_BY_TASK = {
    1: "run_closed_set_identification",
    2: "run_closed_set_verification",
    3: "run_subject_disjoint_identification",
    4: "run_subject_disjoint_verification",
    5: "run_cross_session_identification",
    6: "run_cross_session_verification",
    7: "run_subject_disjoint_cross_session_identification",
    8: "run_subject_disjoint_cross_session_verification",
}

INTRA_SESSION_TASKS = {
    1,
    2,
    3,
    4,
}


class SyntheticLoader:
    """
    Minimal loader used to test main.py without downloading real datasets.
    """

    def __init__(self):
        self.cfg = {
            "root_dir": "synthetic_dataset",
            "preprocessing": {},
        }
        self.prep_params = {}

        self.intra_x = np.zeros(
            (24, 64),
            dtype=np.float32,
        )
        self.intra_y = np.asarray(
            [
                f"subject_{index // 4}"
                for index in range(24)
            ]
        )

        self.session_1_x = np.zeros(
            (18, 64),
            dtype=np.float32,
        )
        self.session_1_y = np.asarray(
            [
                f"subject_{index // 3}"
                for index in range(18)
            ]
        )

        self.session_2_x = np.ones(
            (12, 64),
            dtype=np.float32,
        )
        self.session_2_y = np.asarray(
            [
                f"subject_{index // 2}"
                for index in range(12)
            ]
        )

        self.load_all_data = Mock(
            return_value=(
                self.intra_x,
                self.intra_y,
            )
        )

        self.load_session = Mock(
            side_effect=self._load_session
        )

    def _load_session(self, session_name):
        if session_name == "train":
            return (
                self.session_1_x,
                self.session_1_y,
            )

        if session_name == "test":
            return (
                self.session_2_x,
                self.session_2_y,
            )

        raise ValueError(
            f"Unexpected synthetic session: {session_name}"
        )


def build_cli_arguments(task):
    arguments = [
        "main.py",
        "--dataset",
        "ecgid",
        "--task",
        str(task),
        "--model",
        "deepecg",
        "--data_split_mode",
        "all-available",
        "--epochs",
        "2",
        "--batch_size",
        "8",
        "--lr",
        "0.002",
        "--test_split",
        "0.25",
        "--val_split",
        "0.0",
        "--seed",
        "17",
        "--n_runs",
        "1",
        "--template_fusion_method",
        "median",
        "--template_size",
        "2",
        "--matching_method",
        "euclidean",
        "--num_pairs",
        "40",
        "--sampling_mode",
        "balanced",
        "--probe_fusion_size",
        "2",
        "--sqi_method",
        "kurtosis",
        "--sqi_threshold",
        "0.10",
        "--sqi_keep_pct",
        "0.75",
        "--device",
        "cpu",
        "--intelligent_weight_loading",
    ]

    # main.py correctly requires a gallery for subject-disjoint
    # identification tasks.
    if task in {3, 7}:
        arguments.append(
            "--use_template"
        )

    return arguments


class MainCLIRoutingTests(unittest.TestCase):
    def run_main_with_mocked_runners(
        self,
        task,
        cli_arguments=None,
    ):
        loader = SyntheticLoader()

        if cli_arguments is None:
            cli_arguments = build_cli_arguments(
                task
            )

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    sys,
                    "argv",
                    cli_arguments,
                )
            )

            loader_constructor = stack.enter_context(
                patch.object(
                    main,
                    "load_ecgid_dataset",
                    return_value=loader,
                )
            )

            timer_mock = stack.enter_context(
                patch.object(
                    main.run,
                    "start_experiment_timer",
                )
            )

            runner_mocks = {}

            for runner_name in RUNNER_BY_TASK.values():
                runner_mocks[runner_name] = (
                    stack.enter_context(
                        patch.object(
                            main,
                            runner_name,
                        )
                    )
                )

            main.main()

        return {
            "loader": loader,
            "loader_constructor": loader_constructor,
            "timer_mock": timer_mock,
            "runner_mocks": runner_mocks,
        }

    def assert_common_arguments(
        self,
        keyword_arguments,
        loader,
        use_template,
    ):
        self.assertIs(
            keyword_arguments["model_class"],
            DeepECG,
        )

        self.assertEqual(
            keyword_arguments["epochs"],
            2,
        )
        self.assertEqual(
            keyword_arguments["batch_size"],
            8,
        )
        self.assertEqual(
            keyword_arguments["lr"],
            0.002,
        )
        self.assertEqual(
            keyword_arguments["val_split"],
            0.0,
        )
        self.assertEqual(
            keyword_arguments["seed"],
            17,
        )
        self.assertEqual(
            keyword_arguments["n_runs"],
            1,
        )
        self.assertEqual(
            keyword_arguments["device"],
            "cpu",
        )

        self.assertFalse(
            keyword_arguments["visualize"]
        )
        self.assertEqual(
            keyword_arguments["use_template"],
            use_template,
        )
        self.assertEqual(
            keyword_arguments[
                "template_fusion_method"
            ],
            "median",
        )
        self.assertEqual(
            keyword_arguments["template_size"],
            2,
        )
        self.assertEqual(
            keyword_arguments["matching_method"],
            "euclidean",
        )

        self.assertFalse(
            keyword_arguments[
                "outlier_filtering_on_train"
            ]
        )
        self.assertFalse(
            keyword_arguments[
                "outlier_filtering_on_test"
            ]
        )

        self.assertEqual(
            keyword_arguments["sqi_threshold"],
            0.10,
        )
        self.assertEqual(
            keyword_arguments["sqi_keep_pct"],
            0.75,
        )

        self.assertFalse(
            keyword_arguments[
                "save_results_and_settings"
            ]
        )
        self.assertIs(
            keyword_arguments["loader"],
            loader,
        )
        self.assertTrue(
            keyword_arguments[
                "intelligent_weight_loading"
            ]
        )

    def assert_task_specific_arguments(
        self,
        task,
        keyword_arguments,
    ):
        if task in {1, 3}:
            self.assertEqual(
                keyword_arguments["test_split"],
                0.25,
            )
            self.assertEqual(
                keyword_arguments[
                    "probe_fusion_size"
                ],
                2,
            )
            self.assertEqual(
                keyword_arguments["sqi_scores"],
                "kurtosis",
            )

        elif task in {2, 4}:
            self.assertEqual(
                keyword_arguments["test_split"],
                0.25,
            )
            self.assertEqual(
                keyword_arguments["num_pairs"],
                40,
            )
            self.assertEqual(
                keyword_arguments["sampling_mode"],
                "balanced",
            )
            self.assertFalse(
                keyword_arguments[
                    "use_deployment_evaluation"
                ]
            )
            self.assertEqual(
                keyword_arguments["sqi_scores"],
                "kurtosis",
            )

        elif task == 5:
            self.assertEqual(
                keyword_arguments[
                    "probe_fusion_size"
                ],
                2,
            )
            self.assertEqual(
                keyword_arguments["sqi_train"],
                "kurtosis",
            )
            self.assertEqual(
                keyword_arguments["sqi_test"],
                "kurtosis",
            )

        elif task == 6:
            self.assertEqual(
                keyword_arguments["num_pairs"],
                40,
            )
            self.assertEqual(
                keyword_arguments["sampling_mode"],
                "balanced",
            )
            self.assertFalse(
                keyword_arguments[
                    "use_deployment_evaluation"
                ]
            )
            self.assertEqual(
                keyword_arguments["sqi_train"],
                "kurtosis",
            )
            self.assertEqual(
                keyword_arguments["sqi_test"],
                "kurtosis",
            )

        elif task == 7:
            self.assertEqual(
                keyword_arguments["test_split"],
                0.25,
            )
            self.assertEqual(
                keyword_arguments[
                    "probe_fusion_size"
                ],
                2,
            )
            self.assertEqual(
                keyword_arguments["sqi_s1"],
                "kurtosis",
            )
            self.assertEqual(
                keyword_arguments["sqi_s2"],
                "kurtosis",
            )

        elif task == 8:
            self.assertEqual(
                keyword_arguments["test_split"],
                0.25,
            )
            self.assertEqual(
                keyword_arguments["num_pairs"],
                40,
            )
            self.assertEqual(
                keyword_arguments["sampling_mode"],
                "balanced",
            )
            self.assertFalse(
                keyword_arguments[
                    "use_deployment_evaluation"
                ]
            )
            self.assertEqual(
                keyword_arguments["sqi_s1"],
                "kurtosis",
            )
            self.assertEqual(
                keyword_arguments["sqi_s2"],
                "kurtosis",
            )

    def test_all_eight_tasks_route_to_correct_runner(self):
        for task in range(1, 9):
            with self.subTest(task=task):
                execution = (
                    self.run_main_with_mocked_runners(
                        task
                    )
                )

                loader = execution["loader"]
                loader_constructor = execution[
                    "loader_constructor"
                ]
                timer_mock = execution[
                    "timer_mock"
                ]
                runner_mocks = execution[
                    "runner_mocks"
                ]

                # An unset --signal_type is forwarded as None so that the loader
                # resolves the channel from the dataset configuration.
                loader_constructor.assert_called_once_with(
                    data_split_mode="all-available",
                    num_beats_to_merge=1,
                    beat_merge_stride=1,
                    preprocessing_config={},
                    signal_type=None,
                )

                self.assertEqual(
                    loader.effective_experiment_configuration[
                        "dataset"
                    ],
                    "ecgid",
                )
                self.assertEqual(
                    loader.effective_experiment_configuration[
                        "task"
                    ],
                    task,
                )

                timer_mock.assert_called_once_with()

                expected_runner_name = (
                    RUNNER_BY_TASK[task]
                )
                expected_runner = runner_mocks[
                    expected_runner_name
                ]

                expected_runner.assert_called_once_with(
                    *expected_runner.call_args.args,
                    **expected_runner.call_args.kwargs,
                )

                for (
                    runner_name,
                    runner_mock,
                ) in runner_mocks.items():
                    if runner_name == expected_runner_name:
                        continue

                    runner_mock.assert_not_called()

                positional_arguments = (
                    expected_runner.call_args.args
                )
                keyword_arguments = (
                    expected_runner.call_args.kwargs
                )

                if task in INTRA_SESSION_TASKS:
                    loader.load_all_data.assert_called_once_with()
                    loader.load_session.assert_not_called()

                    self.assertEqual(
                        len(positional_arguments),
                        2,
                    )
                    self.assertIs(
                        positional_arguments[0],
                        loader.intra_x,
                    )
                    self.assertIs(
                        positional_arguments[1],
                        loader.intra_y,
                    )

                else:
                    loader.load_all_data.assert_not_called()
                    self.assertEqual(
                        loader.load_session.call_args_list,
                        [
                            call(
                                "train"
                            ),
                            call(
                                "test"
                            ),
                        ],
                    )

                    self.assertEqual(
                        len(positional_arguments),
                        4,
                    )
                    self.assertIs(
                        positional_arguments[0],
                        loader.session_1_x,
                    )
                    self.assertIs(
                        positional_arguments[1],
                        loader.session_1_y,
                    )
                    self.assertIs(
                        positional_arguments[2],
                        loader.session_2_x,
                    )
                    self.assertIs(
                        positional_arguments[3],
                        loader.session_2_y,
                    )

                self.assert_common_arguments(
                    keyword_arguments,
                    loader,
                    use_template=task in {3, 7},
                )

                self.assert_task_specific_arguments(
                    task,
                    keyword_arguments,
                )

    def test_yaml_defaults_and_cli_overrides_reach_selected_runner(self):
        loader = SyntheticLoader()

        configuration = {
            "dataset": "ptb",
            "task": 1,
            "model": "resnet1d",
            "data_split_mode": "single-session",
            "epochs": 3,
            "batch_size": 7,
            "lr": 0.004,
            "test_split": 0.30,
            "val_split": 0.20,
            "num_pairs": 24,
            "sampling_mode": "balanced",
            "use_template": True,
            "template_fusion_method": "mean",
            "template_size": 3,
            "matching_method": "cosine",
            "use_deployment_evaluation": True,
            "intelligent_weight_loading": False,
            # Exercise the backward-compatible YAML alias.
            "save_results_and_settings": True,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            configuration_path = (
                Path(temporary_directory)
                / "experiment.yaml"
            )

            configuration_path.write_text(
                yaml.safe_dump(
                    configuration
                ),
                encoding="utf-8",
            )

            cli_arguments = [
                "main.py",
                "--dataset",
                "ecgid",
                "--task",
                "2",
                "--model",
                "deepecg",
                "--data_split_mode",
                "all-available",
                "--epochs",
                "1",
                "--batch_size",
                "4",
                "--sampling_mode",
                "all",
                "--config",
                str(configuration_path),
                "--device",
                "cpu",
            ]

            with patch.object(
                sys,
                "argv",
                cli_arguments,
            ), patch.object(
                main,
                "load_ecgid_dataset",
                return_value=loader,
            ), patch.object(
                main.run,
                "start_experiment_timer",
            ), patch.object(
                main,
                "run_closed_set_verification",
            ) as runner_mock:
                main.main()

        runner_mock.assert_called_once()

        keyword_arguments = (
            runner_mock.call_args.kwargs
        )

        # Explicit command-line values override YAML defaults.
        self.assertIs(
            keyword_arguments["model_class"],
            DeepECG,
        )
        self.assertEqual(
            keyword_arguments["epochs"],
            1,
        )
        self.assertEqual(
            keyword_arguments["batch_size"],
            4,
        )
        self.assertEqual(
            keyword_arguments["sampling_mode"],
            "all",
        )

        # Values not supplied on the CLI still come from YAML.
        self.assertEqual(
            keyword_arguments["lr"],
            0.004,
        )
        self.assertEqual(
            keyword_arguments["test_split"],
            0.30,
        )
        self.assertEqual(
            keyword_arguments["val_split"],
            0.20,
        )
        self.assertEqual(
            keyword_arguments["num_pairs"],
            24,
        )
        self.assertTrue(
            keyword_arguments["use_template"]
        )
        self.assertEqual(
            keyword_arguments[
                "template_fusion_method"
            ],
            "mean",
        )
        self.assertEqual(
            keyword_arguments["template_size"],
            3,
        )
        self.assertEqual(
            keyword_arguments["matching_method"],
            "cosine",
        )
        self.assertTrue(
            keyword_arguments[
                "use_deployment_evaluation"
            ]
        )
        self.assertFalse(
            keyword_arguments[
                "intelligent_weight_loading"
            ]
        )
        self.assertTrue(
            keyword_arguments[
                "save_results_and_settings"
            ]
        )

        effective_configuration = (
            loader.effective_experiment_configuration
        )
        self.assertEqual(
            effective_configuration["dataset"],
            "ecgid",
        )
        self.assertEqual(
            effective_configuration["task"],
            2,
        )
        self.assertEqual(
            effective_configuration["model"],
            "deepecg",
        )
        self.assertEqual(
            effective_configuration["epochs"],
            1,
        )
        self.assertTrue(
            effective_configuration["save_results"],
        )

    def test_config_only_supplies_dataset_and_task(self):
        loader = SyntheticLoader()

        configuration = {
            "dataset": "ecgid",
            "task": 2,
            "model": "resnet1d",
            "data_split_mode": "all-available",
            "epochs": 3,
            "batch_size": 7,
            "val_split": 0.0,
            "device": "cpu",
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            configuration_path = (
                Path(temporary_directory)
                / "self_contained.yaml"
            )
            configuration_path.write_text(
                yaml.safe_dump(
                    configuration
                ),
                encoding="utf-8",
            )

            cli_arguments = [
                "main.py",
                "--config",
                str(configuration_path),
            ]

            with patch.object(
                sys,
                "argv",
                cli_arguments,
            ), patch.object(
                main,
                "load_ecgid_dataset",
                return_value=loader,
            ), patch.object(
                main.run,
                "start_experiment_timer",
            ), patch.object(
                main,
                "run_closed_set_verification",
            ) as runner_mock:
                main.main()

        runner_mock.assert_called_once()
        self.assertIs(
            runner_mock.call_args.kwargs[
                "model_class"
            ],
            ResNet1D,
        )
        self.assertEqual(
            loader.effective_experiment_configuration[
                "dataset"
            ],
            "ecgid",
        )
        self.assertEqual(
            loader.effective_experiment_configuration[
                "task"
            ],
            2,
        )

    def test_data_cache_hit_bypasses_dataset_loading(self):
        loader = SyntheticLoader()

        cached_x = np.full(
            (10, 64),
            fill_value=2.0,
            dtype=np.float32,
        )
        cached_y = np.asarray(
            [
                f"cached_subject_{index // 2}"
                for index in range(10)
            ]
        )

        cli_arguments = [
            "main.py",
            "--dataset",
            "ecgid",
            "--task",
            "1",
            "--epochs",
            "1",
            "--batch_size",
            "4",
            "--device",
            "cpu",
            "--intelligent_data_loading",
        ]

        with patch.object(
            sys,
            "argv",
            cli_arguments,
        ), patch.object(
            main,
            "load_ecgid_dataset",
            return_value=loader,
        ), patch.object(
            main.run,
            "start_experiment_timer",
        ), patch.object(
            main,
            "run_closed_set_identification",
        ) as runner_mock, patch.object(
            main.utils,
            "CacheManager",
        ) as cache_manager_class:
            cache_manager = (
                cache_manager_class.return_value
            )

            cache_manager.get_data_cache.return_value = (
                {
                    "x": cached_x,
                    "y": cached_y,
                },
                "synthetic-cache-id",
            )

            main.main()

        loader.load_all_data.assert_not_called()
        loader.load_session.assert_not_called()

        cache_manager.get_data_cache.assert_called_once()
        cache_manager.save_data_cache.assert_not_called()

        cache_configuration = (
            cache_manager.get_data_cache
            .call_args.args[0]
        )

        self.assertEqual(
            cache_configuration["dataset"],
            "ecgid",
        )
        self.assertEqual(
            cache_configuration["task_type"],
            "intra_session",
        )
        self.assertEqual(
            cache_configuration["split_mode"],
            "all-available",
        )

        positional_arguments = (
            runner_mock.call_args.args
        )

        self.assertIs(
            positional_arguments[0],
            cached_x,
        )
        self.assertIs(
            positional_arguments[1],
            cached_y,
        )


if __name__ == "__main__":
    unittest.main()