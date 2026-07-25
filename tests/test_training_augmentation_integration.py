import ast
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main
import run


RUNNER_NAMES = [
    "run_closed_set_identification",
    "run_closed_set_verification",
    "run_subject_disjoint_identification",
    "run_subject_disjoint_verification",
    "run_cross_session_identification",
    "run_cross_session_verification",
    (
        "run_subject_disjoint_"
        "cross_session_identification"
    ),
    (
        "run_subject_disjoint_"
        "cross_session_verification"
    ),
]


class AugmentationConfigurationTests(
    unittest.TestCase
):
    def test_default_configuration_is_disabled(
        self,
    ):
        config = (
            run._normalize_augmentation_config(
                None
            )
        )

        self.assertEqual(
            config,
            {
                "enabled": False,
                "method": "gaussian",
                "copies": 1,
                "parameters": {},
            },
        )

    def test_alias_and_parameters_are_normalized(
        self,
    ):
        config = (
            run._normalize_augmentation_config(
                {
                    "enabled": True,
                    "method": "time-shift",
                    "copies": np.int64(2),
                    "parameters": {
                        "max_shift": 4,
                    },
                }
            )
        )

        self.assertEqual(
            config["method"],
            "timeshift",
        )

        self.assertEqual(
            config["copies"],
            2,
        )

        self.assertEqual(
            config["parameters"],
            {
                "max_shift": 4,
            },
        )

    def test_invalid_method_parameters_are_rejected(
        self,
    ):
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported parameter",
        ):
            run._normalize_augmentation_config(
                {
                    "enabled": True,
                    "method": "gaussian",
                    "copies": 1,
                    "parameters": {
                        "length": 20,
                    },
                }
            )


class TrainingPartitionAugmentationTests(
    unittest.TestCase
):
    def setUp(self):
        self.x = np.arange(
            32,
            dtype=np.float32,
        ).reshape(4, 8)

        self.y = np.asarray(
            [
                0,
                0,
                1,
                1,
            ]
        )

    def test_disabled_augmentation_preserves_partition(
        self,
    ):
        output_x, output_y = (
            run._augment_training_partition(
                self.x,
                self.y,
                augmentation_config=None,
                seed=42,
            )
        )

        np.testing.assert_array_equal(
            output_x,
            self.x,
        )

        np.testing.assert_array_equal(
            output_y,
            self.y,
        )

    def test_enabled_augmentation_appends_copies_and_labels(
        self,
    ):
        config = {
            "enabled": True,
            "method": "gaussian",
            "copies": 2,
            "parameters": {
                "std": 0.02,
            },
        }

        output_x, output_y = (
            run._augment_training_partition(
                self.x,
                self.y,
                augmentation_config=config,
                seed=42,
            )
        )

        self.assertEqual(
            output_x.shape,
            (
                12,
                8,
            ),
        )

        np.testing.assert_array_equal(
            output_x[:4],
            self.x,
        )

        np.testing.assert_array_equal(
            output_y,
            np.tile(
                self.y,
                3,
            ),
        )

        self.assertFalse(
            np.array_equal(
                output_x[4:8],
                self.x,
            )
        )

    def test_augmentation_is_deterministic_for_fixed_seed(
        self,
    ):
        config = {
            "enabled": True,
            "method": "amplitude",
            "copies": 1,
            "parameters": {
                "scale_range": [
                    0.8,
                    1.2,
                ],
            },
        }

        first_x, first_y = (
            run._augment_training_partition(
                self.x,
                self.y,
                augmentation_config=config,
                seed=17,
            )
        )

        second_x, second_y = (
            run._augment_training_partition(
                self.x,
                self.y,
                augmentation_config=config,
                seed=17,
            )
        )

        np.testing.assert_array_equal(
            first_x,
            second_x,
        )

        np.testing.assert_array_equal(
            first_y,
            second_y,
        )

    def test_multilead_temporal_augmentation_stays_synchronised(
        self,
    ):
        base = np.asarray(
            [
                [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                ],
                [
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                    16.0,
                    17.0,
                ],
            ],
            dtype=np.float32,
        )

        multilead = np.stack(
            [
                base,
                base + 100.0,
            ],
            axis=1,
        )

        labels = np.asarray(
            [
                0,
                1,
            ]
        )

        output_x, _ = (
            run._augment_training_partition(
                multilead,
                labels,
                augmentation_config={
                    "enabled": True,
                    "method": "timeshift",
                    "copies": 1,
                    "parameters": {
                        "max_shift": 3,
                    },
                },
                seed=9,
            )
        )

        augmented = output_x[
            len(multilead):
        ]

        np.testing.assert_allclose(
            augmented[:, 1, :]
            - augmented[:, 0, :],
            100.0,
        )


class RunnerIntegrationTests(
    unittest.TestCase
):
    def setUp(self):
        source = Path(
            run.__file__
        ).read_text(
            encoding="utf-8"
        )

        self.syntax_tree = ast.parse(
            source
        )

        self.functions = {
            node.name: node
            for node in self.syntax_tree.body
            if isinstance(
                node,
                ast.FunctionDef,
            )
        }

    def test_all_runners_augment_only_x_tr_y_tr(
        self,
    ):
        for runner_name in RUNNER_NAMES:
            with self.subTest(
                runner=runner_name
            ):
                function = self.functions[
                    runner_name
                ]

                argument_names = [
                    argument.arg
                    for argument in (
                        function.args.args
                    )
                ]

                self.assertIn(
                    "augmentation_config",
                    argument_names,
                )

                augmentation_calls = [
                    node
                    for node in ast.walk(
                        function
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
                        == (
                            "_augment_training_partition"
                        )
                    )
                ]

                self.assertEqual(
                    len(augmentation_calls),
                    1,
                )

                call = augmentation_calls[0]

                self.assertEqual(
                    call.args[0].id,
                    "X_tr",
                )

                self.assertEqual(
                    call.args[1].id,
                    "y_tr",
                )

                training_loader_calls = [
                    node
                    for node in ast.walk(
                        function
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
                        == "_make_loader"
                        and len(node.args) >= 2
                        and isinstance(
                            node.args[0],
                            ast.Name,
                        )
                        and node.args[0].id
                        == "X_tr"
                        and isinstance(
                            node.args[1],
                            ast.Name,
                        )
                        and node.args[1].id
                        == "y_tr"
                    )
                ]

                self.assertEqual(
                    len(training_loader_calls),
                    1,
                )

                self.assertLess(
                    call.lineno,
                    training_loader_calls[
                        0
                    ].lineno,
                )

    def test_all_weight_cache_configs_include_augmentation(
        self,
    ):
        for runner_name in RUNNER_NAMES:
            with self.subTest(
                runner=runner_name
            ):
                function = self.functions[
                    runner_name
                ]

                train_config_assignments = [
                    node
                    for node in ast.walk(
                        function
                    )
                    if (
                        isinstance(
                            node,
                            ast.Assign,
                        )
                        and any(
                            isinstance(
                                target,
                                ast.Name,
                            )
                            and target.id
                            == "train_config"
                            for target in node.targets
                        )
                        and isinstance(
                            node.value,
                            ast.Dict,
                        )
                    )
                ]

                self.assertEqual(
                    len(
                        train_config_assignments
                    ),
                    1,
                )

                dictionary = (
                    train_config_assignments[
                        0
                    ].value
                )

                keys = {
                    key.value
                    for key in dictionary.keys
                    if isinstance(
                        key,
                        ast.Constant,
                    )
                }

                self.assertIn(
                    "augmentation",
                    keys,
                )


class MainConfigurationTests(
    unittest.TestCase
):
    def test_cli_augmentation_configuration_is_validated(
        self,
    ):
        parser = main.get_parser()

        args = parser.parse_args(
            [
                "--dataset",
                "ecgid",
                "--task",
                "1",
                "--use_augmentation",
                "--augmentation_method",
                "gaussian",
                "--augmentation_copies",
                "2",
                "--augmentation_parameters",
                '{"std": 0.02, "relative": true}',
            ]
        )

        args = (
            main.validate_experiment_arguments(
                args,
                parser,
            )
        )

        self.assertTrue(
            args.use_augmentation
        )

        self.assertEqual(
            args.augmentation_method,
            "gaussian",
        )

        self.assertEqual(
            args.augmentation_copies,
            2,
        )

        self.assertEqual(
            args.augmentation_parameters,
            {
                "std": 0.02,
                "relative": True,
            },
        )

    def test_yaml_exposes_disabled_safe_defaults(
        self,
    ):
        configuration = yaml.safe_load(
            (
                PROJECT_ROOT
                / "experiment_settings.yaml"
            ).read_text(
                encoding="utf-8"
            )
        )

        self.assertFalse(
            configuration[
                "use_augmentation"
            ]
        )

        self.assertEqual(
            configuration[
                "augmentation_method"
            ],
            "gaussian",
        )

        self.assertEqual(
            configuration[
                "augmentation_copies"
            ],
            1,
        )

        self.assertEqual(
            configuration[
                "augmentation_parameters"
            ],
            {},
        )

    def test_main_passes_augmentation_config_to_runners(
        self,
    ):
        syntax_tree = ast.parse(
            Path(
                main.__file__
            ).read_text(
                encoding="utf-8"
            )
        )

        main_function = next(
            node
            for node in syntax_tree.body
            if (
                isinstance(
                    node,
                    ast.FunctionDef,
                )
                and node.name == "main"
            )
        )

        common_args_assignment = next(
            node
            for node in ast.walk(
                main_function
            )
            if (
                isinstance(
                    node,
                    ast.Assign,
                )
                and any(
                    isinstance(
                        target,
                        ast.Name,
                    )
                    and target.id
                    == "common_args"
                    for target in node.targets
                )
                and isinstance(
                    node.value,
                    ast.Dict,
                )
            )
        )

        keys = {
            key.value
            for key in (
                common_args_assignment.value.keys
            )
            if isinstance(
                key,
                ast.Constant,
            )
        }

        self.assertIn(
            "augmentation_config",
            keys,
        )


if __name__ == "__main__":
    unittest.main()
