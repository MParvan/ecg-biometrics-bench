import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main


def parse_basic_arguments(task="1"):
    parser = main.get_parser()

    arguments = parser.parse_args(
        [
            "--dataset",
            "ecgid",
            "--task",
            task,
        ]
    )

    return parser, arguments


class ArgumentValidationTests(unittest.TestCase):
    def assert_validation_error(
        self,
        parser,
        arguments,
        expected_message,
    ):
        captured_stderr = StringIO()

        with redirect_stderr(captured_stderr):
            with self.assertRaises(SystemExit) as context:
                main.validate_experiment_arguments(
                    arguments,
                    parser,
                )

        self.assertEqual(context.exception.code, 2)
        self.assertIn(
            expected_message,
            captured_stderr.getvalue(),
        )

    def test_default_task_one_configuration_is_valid(self):
        parser, arguments = parse_basic_arguments()

        validated = main.validate_experiment_arguments(
            arguments,
            parser,
        )

        self.assertIs(validated, arguments)

    def test_yaml_override_cannot_bypass_parser_choices(self):
        parser, arguments = parse_basic_arguments()
        arguments.sampling_mode = "unsupported"

        self.assert_validation_error(
            parser,
            arguments,
            "Invalid value for 'sampling_mode'",
        )

    def test_positive_integer_parameters_are_enforced(self):
        invalid_values = [
            ("epochs", 0),
            ("batch_size", 0),
            ("n_runs", 0),
            ("num_pairs", 0),
            ("num_beats_to_merge", 0),
            ("probe_fusion_size", 0),
        ]

        for argument_name, invalid_value in invalid_values:
            with self.subTest(argument=argument_name):
                parser, arguments = parse_basic_arguments()
                setattr(
                    arguments,
                    argument_name,
                    invalid_value,
                )

                self.assert_validation_error(
                    parser,
                    arguments,
                    "must be greater than or equal to 1",
                )

    def test_probe_fusion_size_default_is_one(self):
        _, arguments = parse_basic_arguments()
        self.assertEqual(arguments.probe_fusion_size, 1)

    def test_probe_fusion_size_accepts_explicit_multi_beat_values(self):
        parser = main.get_parser()

        for explicit_value in (2, 3, 5):
            with self.subTest(explicit_value=explicit_value):
                arguments = parser.parse_args(
                    [
                        "--dataset",
                        "ecgid",
                        "--task",
                        "1",
                        "--probe_fusion_size",
                        str(explicit_value),
                    ]
                )
                self.assertEqual(
                    arguments.probe_fusion_size,
                    explicit_value,
                )

    def test_template_size_must_be_positive_when_set(self):
        parser, arguments = parse_basic_arguments()
        arguments.template_size = 0

        self.assert_validation_error(
            parser,
            arguments,
            "'template_size' must be greater than or equal to 1",
        )

    def test_test_split_must_be_strictly_between_zero_and_one(self):
        for invalid_value in [0.0, 1.0, -0.1, 1.1]:
            with self.subTest(value=invalid_value):
                parser, arguments = parse_basic_arguments()
                arguments.test_split = invalid_value

                self.assert_validation_error(
                    parser,
                    arguments,
                    "'test_split'",
                )

    def test_zero_validation_split_is_allowed(self):
        parser, arguments = parse_basic_arguments()
        arguments.val_split = 0.0

        validated = main.validate_experiment_arguments(
            arguments,
            parser,
        )

        self.assertEqual(validated.val_split, 0.0)

    def test_validation_split_one_is_rejected(self):
        parser, arguments = parse_basic_arguments()
        arguments.val_split = 1.0

        self.assert_validation_error(
            parser,
            arguments,
            "'val_split' must satisfy 0 <= val_split < 1",
        )

    def test_sqi_ranges_are_enforced(self):
        parser, arguments = parse_basic_arguments()
        arguments.sqi_threshold = 1.1

        self.assert_validation_error(
            parser,
            arguments,
            "'sqi_threshold'",
        )

        parser, arguments = parse_basic_arguments()
        arguments.sqi_keep_pct = 0.0

        self.assert_validation_error(
            parser,
            arguments,
            "'sqi_keep_pct' must satisfy",
        )

    def test_session_configuration_must_be_a_list(self):
        parser, arguments = parse_basic_arguments()

        # Simulate a YAML scalar instead of a YAML list.
        arguments.train_sessions = "session_1"

        self.assert_validation_error(
            parser,
            arguments,
            "'train_sessions' must be a list",
        )

    def test_subject_disjoint_identification_requires_template_mode(self):
        for task in ["3", "7"]:
            with self.subTest(task=task):
                parser, arguments = parse_basic_arguments(
                    task=task
                )

                arguments.use_template = False

                self.assert_validation_error(
                    parser,
                    arguments,
                    "requires 'use_template: true'",
                )

    def test_deployment_evaluation_requires_validation_data(self):
        parser, arguments = parse_basic_arguments(
            task="2"
        )

        arguments.use_deployment_evaluation = True
        arguments.val_split = 0.0

        self.assert_validation_error(
            parser,
            arguments,
            "Deployment evaluation requires val_split > 0",
        )

    def test_valid_subject_disjoint_configuration_passes(self):
        parser, arguments = parse_basic_arguments(
            task="3"
        )

        arguments.use_template = True
        arguments.template_size = 5

        validated = main.validate_experiment_arguments(
            arguments,
            parser,
        )

        self.assertTrue(validated.use_template)
        self.assertEqual(validated.template_size, 5)


class EnrollmentTemplateModeValidation(unittest.TestCase):
    """
    Argument validation for ``--enrollment_template_mode`` and its three
    companion parameters, exercised directly against
    ``validate_experiment_arguments`` and, where presence tracking matters,
    through the full ``parse_experiment_arguments`` pipeline.
    """

    def assert_validation_error(self, parser, arguments, expected_message):
        captured_stderr = StringIO()

        with redirect_stderr(captured_stderr):
            with self.assertRaises(SystemExit) as context:
                main.validate_experiment_arguments(arguments, parser)

        self.assertEqual(context.exception.code, 2)
        self.assertIn(expected_message, captured_stderr.getvalue())

    def test_default_mode_is_fusion(self):
        _, arguments = parse_basic_arguments()
        self.assertEqual(arguments.enrollment_template_mode, "fusion")
        self.assertIsNone(arguments.num_templates_per_identity)

    def test_fusion_mode_rejects_explicit_num_templates_per_identity(self):
        parser, arguments = parse_basic_arguments()
        arguments.num_templates_per_identity = 3

        self.assert_validation_error(
            parser,
            arguments,
            "enrollment_template_mode='fusion' does not use "
            "num_templates_per_identity",
        )

    def test_fusion_mode_rejects_explicit_selection_method(self):
        parser, arguments = parse_basic_arguments()
        arguments.template_selection_method = "farthest_first_cosine"

        self.assert_validation_error(
            parser,
            arguments,
            "enrollment_template_mode='fusion' does not use",
        )

    def test_multi_template_requires_use_template(self):
        parser, arguments = parse_basic_arguments()
        arguments.enrollment_template_mode = "multi_template"
        arguments.num_templates_per_identity = 3
        arguments.use_template = False

        self.assert_validation_error(
            parser,
            arguments,
            "enrollment_template_mode='multi_template' requires "
            "'use_template: true'",
        )

    def test_multi_template_requires_num_templates_per_identity(self):
        parser, arguments = parse_basic_arguments()
        arguments.enrollment_template_mode = "multi_template"
        arguments.use_template = True

        self.assert_validation_error(
            parser,
            arguments,
            "enrollment_template_mode='multi_template' requires "
            "'num_templates_per_identity'",
        )

    def test_multi_template_rejects_non_positive_k(self):
        parser, arguments = parse_basic_arguments()
        arguments.enrollment_template_mode = "multi_template"
        arguments.use_template = True
        arguments.num_templates_per_identity = 0

        self.assert_validation_error(
            parser,
            arguments,
            "'num_templates_per_identity' must be greater than or equal "
            "to 1",
        )

    def test_multi_template_rejects_deployment_evaluation(self):
        parser, arguments = parse_basic_arguments(task="2")
        arguments.enrollment_template_mode = "multi_template"
        arguments.use_template = True
        arguments.num_templates_per_identity = 3
        arguments.use_deployment_evaluation = True
        arguments.val_split = 0.2

        self.assert_validation_error(
            parser,
            arguments,
            "enrollment_template_mode='multi_template' is not compatible "
            "with use_deployment_evaluation",
        )

    def test_valid_multi_template_configuration_passes(self):
        parser, arguments = parse_basic_arguments()
        arguments.enrollment_template_mode = "multi_template"
        arguments.use_template = True
        arguments.num_templates_per_identity = 3

        validated = main.validate_experiment_arguments(arguments, parser)

        self.assertEqual(validated.num_templates_per_identity, 3)

    def test_explicit_template_fusion_method_with_multi_template_is_rejected(
        self,
    ):
        # Presence tracking distinguishes an explicitly configured
        # template_fusion_method from the untouched parser default, which
        # validate_experiment_arguments alone cannot see. This exercises the
        # full parse_experiment_arguments pipeline via a YAML file.
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(
                "dataset: ecgid\n"
                "task: 1\n"
                "template_fusion_method: mean\n",
                encoding="utf-8",
            )

            captured_stderr = StringIO()
            with redirect_stderr(captured_stderr):
                with self.assertRaises(SystemExit) as context:
                    main.parse_experiment_arguments(
                        [
                            "--config",
                            str(config_path),
                            "--use_template",
                            "--enrollment_template_mode",
                            "multi_template",
                            "--num_templates_per_identity",
                            "3",
                        ]
                    )

            self.assertEqual(context.exception.code, 2)
            self.assertIn(
                "does not use 'template_fusion_method'",
                captured_stderr.getvalue(),
            )

    def test_fusion_mode_yaml_default_is_untouched_by_presence_tracking(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(
                "dataset: ecgid\ntask: 1\n",
                encoding="utf-8",
            )

            args, _ = main.parse_experiment_arguments(
                ["--config", str(config_path)]
            )

            self.assertEqual(args.enrollment_template_mode, "fusion")
            self.assertEqual(args.template_fusion_method, "mean")

    def test_effective_configuration_omits_presence_tracking_attribute(self):
        _, arguments = parse_basic_arguments()
        arguments.explicitly_configured_arguments = frozenset(
            {"dataset", "task"}
        )

        effective = main.build_effective_configuration(arguments)

        self.assertNotIn(
            "explicitly_configured_arguments", effective
        )


if __name__ == "__main__":
    unittest.main()