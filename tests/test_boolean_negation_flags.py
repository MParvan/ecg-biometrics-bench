import contextlib
import io as string_io
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main

NEGATABLE_OPTIONS = (
    "save_results",
    "visualize",
    "use_template",
    "use_augmentation",
    "use_deployment_evaluation",
    "outlier_filtering_on_train",
    "outlier_filtering_on_test",
    "intelligent_data_loading",
    "intelligent_weight_loading",
)


class BooleanNegationFlagTests(unittest.TestCase):
    """
    A YAML file can enable any boolean option, so each needs a CLI override.
    """

    CONFIG = str(
        PROJECT_ROOT
        / "configs"
        / "paper_reproduction"
        / "ecgid"
        / "ecgid_all_available_closed_set_task01_identification.yaml"
    )

    def _resolve(self, extra_arguments):
        buffer = string_io.StringIO()

        with contextlib.redirect_stdout(
            buffer
        ), contextlib.redirect_stderr(buffer):
            args, _ = main.parse_experiment_arguments(
                ["--config", self.CONFIG] + extra_arguments
            )

        return args

    def test_every_store_true_option_has_a_negation(self):
        parser = main.get_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }

        for option_name in NEGATABLE_OPTIONS:
            with self.subTest(option=option_name):
                self.assertIn(
                    f"--no_{option_name}",
                    option_strings,
                )

    def test_negation_overrides_the_configuration(self):
        self.assertTrue(
            self._resolve([]).intelligent_weight_loading
        )
        self.assertFalse(
            self._resolve(
                ["--no_intelligent_weight_loading"]
            ).intelligent_weight_loading
        )

    def test_last_flag_wins(self):
        args = self._resolve(
            [
                "--no_intelligent_weight_loading",
                "--intelligent_weight_loading",
            ]
        )

        self.assertTrue(args.intelligent_weight_loading)

    def test_negation_does_not_affect_other_options(self):
        args = self._resolve(
            ["--no_intelligent_weight_loading"]
        )

        self.assertTrue(args.intelligent_data_loading)
        self.assertTrue(args.use_template)

    def test_negations_are_hidden_from_help(self):
        help_text = main.get_parser().format_help()

        for option_name in NEGATABLE_OPTIONS:
            with self.subTest(option=option_name):
                self.assertNotIn(
                    f"--no_{option_name}",
                    help_text,
                )


if __name__ == "__main__":
    unittest.main()
