import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

import main


def record_role_args(*extra):
    return [
        "--dataset",
        "ecgid",
        "--task",
        "5",
        "--data_split_mode",
        "custom-record-split",
        "--train_record_indices",
        "0",
        "--probe_record_indices",
        "2",
        *extra,
    ]


class FakeRoleLoader:
    def __init__(self, kwargs):
        self.kwargs = dict(
            kwargs
        )
        self.cfg = {
            "root_dir": "synthetic",
            "preprocessing": {},
        }
        self.prep_params = {}
        self.data_split_mode = kwargs[
            "data_split_mode"
        ]
        self.train_record_indices = tuple(
            kwargs.get(
                "train_record_indices",
                [0],
            )
        )
        self.enroll_record_indices = tuple(
            kwargs.get(
                "enroll_record_indices",
                self.train_record_indices,
            )
        )
        self.probe_record_indices = tuple(
            kwargs.get(
                "probe_record_indices",
                [2],
            )
        )
        self.calls = []

    def load_session(
        self,
        role,
        return_provenance=False,
    ):
        self.calls.append(
            role
        )

        values = {
            "train": 1.0,
            "enrollment": 2.0,
            "probe": 3.0,
        }

        x = np.full(
            (4, 8),
            values[role],
            dtype=np.float32,
        )

        y = np.asarray(
            [
                0,
                0,
                1,
                1,
            ],
            dtype=int,
        )

        if return_provenance:
            return x, y, None

        return x, y


class ArgumentRoutingTests(
    unittest.TestCase
):
    def test_record_selectors_reach_loader_when_enrollment_is_used(
        self,
    ):
        args, _ = (
            main.parse_experiment_arguments(
                record_role_args(
                    "--enroll_record_indices",
                    "1",
                    "--use_template",
                )
            )
        )

        kwargs = (
            main._build_dataset_loader_kwargs(
                args
            )
        )

        self.assertEqual(
            kwargs[
                "train_record_indices"
            ],
            [0],
        )
        self.assertEqual(
            kwargs[
                "enroll_record_indices"
            ],
            [1],
        )
        self.assertEqual(
            kwargs[
                "probe_record_indices"
            ],
            [2],
        )

    def test_unused_record_enrollment_selector_is_not_routed(
        self,
    ):
        args, _ = (
            main.parse_experiment_arguments(
                record_role_args(
                    "--enroll_record_indices",
                    "2",
                )
            )
        )

        kwargs = (
            main._build_dataset_loader_kwargs(
                args
            )
        )

        self.assertNotIn(
            "enroll_record_indices",
            kwargs,
        )

    def test_unused_session_enrollment_may_overlap_probe(
        self,
    ):
        args, _ = (
            main.parse_experiment_arguments(
                [
                    "--dataset",
                    "heartprint",
                    "--task",
                    "5",
                    "--data_split_mode",
                    "cross-session",
                    "--train_sessions",
                    "session1",
                    "--enroll_sessions",
                    "session2",
                    "--probe_sessions",
                    "session2",
                ]
            )
        )

        kwargs = (
            main._build_dataset_loader_kwargs(
                args
            )
        )

        self.assertNotIn(
            "enroll_sessions",
            kwargs,
        )

    def test_active_session_enrollment_may_not_overlap_probe(
        self,
    ):
        with self.assertRaises(
            SystemExit
        ):
            main.parse_experiment_arguments(
                [
                    "--dataset",
                    "heartprint",
                    "--task",
                    "5",
                    "--data_split_mode",
                    "cross-session",
                    "--train_sessions",
                    "session1",
                    "--enroll_sessions",
                    "session2",
                    "--probe_sessions",
                    "session2",
                    "--use_template",
                ]
            )

    def test_wrong_selector_families_fail_fast(
        self,
    ):
        with self.assertRaises(
            SystemExit
        ):
            main.parse_experiment_arguments(
                [
                    "--dataset",
                    "ecgid",
                    "--task",
                    "5",
                    "--data_split_mode",
                    "single-cross-session",
                    "--train_sessions",
                    "session1",
                ]
            )

        with self.assertRaises(
            SystemExit
        ):
            main.parse_experiment_arguments(
                [
                    "--dataset",
                    "mitbih",
                    "--task",
                    "5",
                    "--data_split_mode",
                    "custom-split",
                    "--train_parts",
                    "0",
                    "5",
                    "--test_parts",
                    "20",
                    "25",
                    "--train_record_indices",
                    "0",
                ]
            )

    def test_record_selectors_require_custom_record_split(
        self,
    ):
        with self.assertRaises(
            SystemExit
        ):
            main.parse_experiment_arguments(
                [
                    "--dataset",
                    "ecgid",
                    "--task",
                    "5",
                    "--data_split_mode",
                    "single-cross-session",
                    "--train_record_indices",
                    "0",
                    "--probe_record_indices",
                    "1",
                ]
            )

    def test_all_three_selector_families_route_enrollment(
        self,
    ):
        session_args, _ = (
            main.parse_experiment_arguments(
                [
                    "--dataset",
                    "heartprint",
                    "--task",
                    "5",
                    "--data_split_mode",
                    "cross-session",
                    "--train_sessions",
                    "session1",
                    "--enroll_sessions",
                    "session2",
                    "--probe_sessions",
                    "session3r",
                    "--use_template",
                ]
            )
        )

        self.assertEqual(
            main._build_dataset_loader_kwargs(
                session_args
            )[
                "enroll_sessions"
            ],
            ["session2"],
        )

        continuous_args, _ = (
            main.parse_experiment_arguments(
                [
                    "--dataset",
                    "mitbih",
                    "--task",
                    "5",
                    "--data_split_mode",
                    "custom-split",
                    "--train_parts",
                    "0",
                    "5",
                    "--enroll_parts",
                    "10",
                    "15",
                    "--test_parts",
                    "20",
                    "25",
                    "--use_template",
                ]
            )
        )

        self.assertEqual(
            main._build_dataset_loader_kwargs(
                continuous_args
            )[
                "enrol_parts"
            ],
            [(10.0, 15.0)],
        )

        record_args, _ = (
            main.parse_experiment_arguments(
                record_role_args(
                    "--enroll_record_indices",
                    "1",
                    "--use_template",
                )
            )
        )

        self.assertEqual(
            main._build_dataset_loader_kwargs(
                record_args
            )[
                "enroll_record_indices"
            ],
            [1],
        )


class EnrollmentReuseTests(
    unittest.TestCase
):
    def test_session_equality_is_order_sensitive(
        self,
    ):
        same = SimpleNamespace(
            train_sessions=[
                "session1",
                "session2",
            ],
            enroll_sessions=[
                "session1",
                "session2",
            ],
        )

        reordered = SimpleNamespace(
            train_sessions=[
                "session1",
                "session2",
            ],
            enroll_sessions=[
                "session2",
                "session1",
            ],
        )

        self.assertTrue(
            main._resolved_enrollment_reuses_training(
                same,
                "heartprint",
            )
        )

        self.assertFalse(
            main._resolved_enrollment_reuses_training(
                reordered,
                "heartprint",
            )
        )

    def test_continuous_equality_is_order_sensitive(
        self,
    ):
        same = SimpleNamespace(
            train_parts=[
                [0.0, 5.0],
                [10.0, 15.0],
            ],
            enrol_parts=[
                [0.0, 5.0],
                [10.0, 15.0],
            ],
        )

        distinct = SimpleNamespace(
            train_parts=[
                [0.0, 5.0],
            ],
            enrol_parts=[
                [10.0, 15.0],
            ],
        )

        self.assertTrue(
            main._resolved_enrollment_reuses_training(
                same,
                "mitbih",
            )
        )

        self.assertFalse(
            main._resolved_enrollment_reuses_training(
                distinct,
                "mitbih",
            )
        )

    def test_legacy_record_protocol_reuses_training(
        self,
    ):
        loader = SimpleNamespace(
            data_split_mode=(
                "single-cross-session"
            )
        )

        self.assertTrue(
            main._resolved_enrollment_reuses_training(
                loader,
                "ecgid",
            )
        )

    def test_custom_record_equality_controls_reuse(
        self,
    ):
        same = SimpleNamespace(
            data_split_mode=(
                "custom-record-split"
            ),
            train_record_indices=(
                0,
                1,
            ),
            enroll_record_indices=(
                0,
                1,
            ),
        )

        distinct = SimpleNamespace(
            data_split_mode=(
                "custom-record-split"
            ),
            train_record_indices=(
                0,
            ),
            enroll_record_indices=(
                1,
            ),
        )

        self.assertTrue(
            main._resolved_enrollment_reuses_training(
                same,
                "ptb",
            )
        )

        self.assertFalse(
            main._resolved_enrollment_reuses_training(
                distinct,
                "ptb",
            )
        )

    def test_reused_enrollment_is_not_loaded_twice(
        self,
    ):
        loader = FakeRoleLoader(
            {
                "data_split_mode": (
                    "custom-record-split"
                ),
                "train_record_indices": [
                    0
                ],
                "enroll_record_indices": [
                    0
                ],
                "probe_record_indices": [
                    2
                ],
            }
        )

        roles = (
            main._load_cross_session_roles(
                loader,
                use_enrollment=True,
                enrollment_reuses_training=True,
            )
        )

        self.assertEqual(
            loader.calls,
            [
                "train",
                "probe",
            ],
        )

        self.assertIsNone(
            roles[3]
        )

    def test_distinct_enrollment_is_loaded(
        self,
    ):
        loader = FakeRoleLoader(
            {
                "data_split_mode": (
                    "custom-record-split"
                ),
                "train_record_indices": [
                    0
                ],
                "enroll_record_indices": [
                    1
                ],
                "probe_record_indices": [
                    2
                ],
            }
        )

        roles = (
            main._load_cross_session_roles(
                loader,
                use_enrollment=True,
                enrollment_reuses_training=False,
            )
        )

        self.assertEqual(
            loader.calls,
            [
                "train",
                "enrollment",
                "probe",
            ],
        )

        np.testing.assert_array_equal(
            roles[3],
            np.full(
                (4, 8),
                2.0,
                dtype=np.float32,
            ),
        )


class CacheContractTests(
    unittest.TestCase
):
    def test_legacy_two_role_payload_is_a_cache_miss(
        self,
    ):
        legacy = {
            "x_s1": np.ones(
                (2, 4),
                dtype=np.float32,
            ),
            "y_s1": np.asarray(
                [0, 1]
            ),
            "x_s2": np.ones(
                (2, 4),
                dtype=np.float32,
            ),
            "y_s2": np.asarray(
                [0, 1]
            ),
        }

        self.assertIsNone(
            main._restore_cross_session_cache(
                legacy,
                require_distinct_enrollment=False,
            )
        )

    def test_unused_session_enrollment_is_removed_from_cache_identity(
        self,
    ):
        args = SimpleNamespace(
            dataset="heartprint",
            data_split_mode="cross-session",
            num_beats_to_merge=1,
            train_sessions=[
                "session1"
            ],
            enroll_sessions=[
                "session2"
            ],
            probe_sessions=[
                "session3r"
            ],
            session_for_single_session_evaluation=None,
        )

        loader = SimpleNamespace(
            cfg={
                "root_dir": "synthetic",
                "preprocessing": {},
            },
            prep_params={},
            train_sessions=[
                "session1"
            ],
            enroll_sessions=[
                "session1"
            ],
            probe_sessions=[
                "session3r"
            ],
        )

        config = (
            main.build_data_cache_config(
                args,
                loader,
                task_type="cross_session",
                enrollment_consumed=False,
            )
        )

        self.assertIsNone(
            config[
                "enroll_sessions"
            ]
        )


class MainIntegrationTests(
    unittest.TestCase
):
    def execute_task5(
        self,
        use_template,
        enrollment_index,
    ):
        created_loaders = []

        def loader_factory(
            **kwargs,
        ):
            loader = FakeRoleLoader(
                kwargs
            )

            created_loaders.append(
                loader
            )

            return loader

        runner = Mock()

        arguments = record_role_args(
            "--enroll_record_indices",
            str(
                enrollment_index
            ),
            *(
                [
                    "--use_template"
                ]
                if use_template
                else []
            ),
        )

        with patch.object(
            sys,
            "argv",
            [
                "main.py",
                *arguments,
            ],
        ), patch.object(
            main,
            "load_ecgid_dataset",
            side_effect=loader_factory,
        ), patch.object(
            main,
            "run_cross_session_identification",
            runner,
        ):
            main.main()

        self.assertEqual(
            len(
                created_loaders
            ),
            1,
        )

        return (
            created_loaders[0],
            runner,
        )

    def test_distinct_enrollment_reaches_task5(
        self,
    ):
        loader, runner = (
            self.execute_task5(
                use_template=True,
                enrollment_index=1,
            )
        )

        self.assertEqual(
            loader.calls,
            [
                "train",
                "enrollment",
                "probe",
            ],
        )

        self.assertIsNotNone(
            runner.call_args.kwargs[
                "x_enroll"
            ]
        )

        np.testing.assert_array_equal(
            runner.call_args.kwargs[
                "x_enroll"
            ],
            np.full(
                (4, 8),
                2.0,
                dtype=np.float32,
            ),
        )

    def test_unused_enrollment_is_not_loaded(
        self,
    ):
        loader, runner = (
            self.execute_task5(
                use_template=False,
                enrollment_index=2,
            )
        )

        self.assertEqual(
            loader.calls,
            [
                "train",
                "probe",
            ],
        )

        self.assertNotIn(
            "enroll_record_indices",
            loader.kwargs,
        )

        self.assertIsNone(
            runner.call_args.kwargs[
                "x_enroll"
            ]
        )

    def test_train_equal_enroll_is_loaded_once(
        self,
    ):
        loader, runner = (
            self.execute_task5(
                use_template=True,
                enrollment_index=0,
            )
        )

        self.assertEqual(
            loader.calls,
            [
                "train",
                "probe",
            ],
        )

        self.assertIsNone(
            runner.call_args.kwargs[
                "x_enroll"
            ]
        )


class ExistingConfigurationRegressionTests(
    unittest.TestCase
):
    def test_all_existing_cross_session_configs_keep_train_equal_enroll(
        self,
    ):
        roots = [
            PROJECT_ROOT
            / "configs"
            / "paper_reproduction",
            PROJECT_ROOT
            / "configs"
            / "model_comparison",
        ]

        configs = []

        for root in roots:
            for path in root.rglob(
                "*.yaml"
            ):
                with path.open(
                    "r",
                    encoding="utf-8",
                ) as handle:
                    config = (
                        yaml.safe_load(
                            handle
                        )
                        or {}
                    )

                if config.get(
                    "task"
                ) in {
                    5,
                    6,
                    7,
                    8,
                }:
                    configs.append(
                        (
                            path,
                            config,
                        )
                    )

        self.assertEqual(
            len(
                configs
            ),
            194,
        )

        for (
            path,
            config,
        ) in configs:
            with self.subTest(
                config=str(
                    path
                )
            ):
                dataset = (
                    config[
                        "dataset"
                    ]
                )

                if (
                    dataset
                    in main.SESSION_ROUTED_DATASETS
                ):
                    self.assertEqual(
                        config.get(
                            "train_sessions"
                        ),
                        config.get(
                            "enroll_sessions"
                        ),
                    )

                elif (
                    dataset
                    in main.CONTINUOUS_ROUTED_DATASETS
                ):
                    self.assertEqual(
                        config.get(
                            "train_parts"
                        ),
                        config.get(
                            "enrol_parts"
                        ),
                    )

                elif (
                    dataset
                    in main.RECORD_ROUTED_DATASETS
                ):
                    self.assertNotEqual(
                        config.get(
                            "data_split_mode"
                        ),
                        "custom-record-split",
                    )

                else:
                    self.fail(
                        "Unexpected cross-session "
                        f"dataset {dataset!r}."
                    )


if __name__ == "__main__":
    unittest.main()
