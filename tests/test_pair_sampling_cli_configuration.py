import tempfile
import unittest
from pathlib import Path

import yaml

import main


class PairSamplingCLIConfigurationTests(
    unittest.TestCase
):
    @staticmethod
    def parse_verification(extra=None):
        arguments = [
            "--dataset",
            "ecgid",
            "--task",
            "2",
        ]

        if extra:
            arguments.extend(
                extra
            )

        return (
            main.parse_experiment_arguments(
                arguments
            )[0]
        )

    def test_default_verification_behavior_remains_exhaustive_all(self):
        arguments = (
            self.parse_verification()
        )

        self.assertEqual(
            arguments.pair_sampling_mode,
            "all",
        )

        self.assertIsNone(
            arguments.pair_sampling_budget
        )

        self.assertIsNone(
            arguments.max_impostor_pairs
        )

        self.assertIsNone(
            arguments.pair_sampling_seed
        )

    def test_legacy_cli_names_resolve_to_canonical_configuration(self):
        arguments = (
            self.parse_verification(
                [
                    "--sampling_mode",
                    "balanced",
                    "--num_pairs",
                    "123",
                ]
            )
        )

        self.assertEqual(
            arguments.pair_sampling_mode,
            "balanced",
        )

        self.assertEqual(
            arguments.pair_sampling_budget,
            123,
        )

        self.assertIsNone(
            arguments.max_impostor_pairs
        )

        self.assertEqual(
            arguments.pair_sampling_seed,
            42,
        )

    def test_all_genuine_uses_cap_and_dedicated_seed(self):
        arguments = (
            self.parse_verification(
                [
                    "--pair_sampling_mode",
                    "all_genuine",
                    "--max_impostor_pairs",
                    "500",
                    "--pair_sampling_seed",
                    "7",
                ]
            )
        )

        self.assertEqual(
            arguments.pair_sampling_mode,
            "all_genuine",
        )

        self.assertIsNone(
            arguments.pair_sampling_budget
        )

        self.assertEqual(
            arguments.max_impostor_pairs,
            500,
        )

        self.assertEqual(
            arguments.pair_sampling_seed,
            7,
        )

    def test_legacy_yaml_names_are_normalized(self):
        configuration = {
            "dataset": "ecgid",
            "task": 2,
            "sampling_mode": "balanced",
            "num_pairs": 321,
        }

        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "legacy.yaml"
            )

            path.write_text(
                yaml.safe_dump(
                    configuration
                ),
                encoding="utf-8",
            )

            arguments, _ = (
                main.parse_experiment_arguments(
                    [
                        "--config",
                        str(path),
                    ]
                )
            )

        self.assertEqual(
            arguments.pair_sampling_mode,
            "balanced",
        )

        self.assertEqual(
            arguments.pair_sampling_budget,
            321,
        )

        effective = (
            main.build_effective_configuration(
                arguments
            )
        )

        self.assertEqual(
            effective[
                "pair_sampling_mode"
            ],
            "balanced",
        )

        self.assertEqual(
            effective[
                "pair_sampling_budget"
            ],
            321,
        )

        self.assertNotIn(
            "sampling_mode",
            effective,
        )

        self.assertNotIn(
            "num_pairs",
            effective,
        )

    def test_legacy_cli_alias_overrides_canonical_yaml_value(self):
        configuration = {
            "dataset": "ecgid",
            "task": 2,
            "pair_sampling_mode": "balanced",
            "pair_sampling_budget": 321,
        }

        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "canonical.yaml"
            )

            path.write_text(
                yaml.safe_dump(
                    configuration
                ),
                encoding="utf-8",
            )

            arguments, _ = (
                main.parse_experiment_arguments(
                    [
                        "--config",
                        str(path),
                        "--sampling_mode",
                        "all",
                    ]
                )
            )

        self.assertEqual(
            arguments.pair_sampling_mode,
            "all",
        )

        self.assertIsNone(
            arguments.pair_sampling_budget
        )

        self.assertIsNone(
            arguments.max_impostor_pairs
        )

        self.assertIsNone(
            arguments.pair_sampling_seed
        )

    def test_matching_legacy_and_canonical_cli_values_are_accepted(self):
        arguments = (
            self.parse_verification(
                [
                    "--sampling_mode",
                    "balanced",
                    "--pair_sampling_mode",
                    "balanced",
                    "--num_pairs",
                    "40",
                    "--pair_sampling_budget",
                    "40",
                ]
            )
        )

        self.assertEqual(
            arguments.pair_sampling_mode,
            "balanced",
        )

        self.assertEqual(
            arguments.pair_sampling_budget,
            40,
        )

    def test_conflicting_cli_aliases_are_rejected(self):
        with self.assertRaises(
            SystemExit
        ):
            self.parse_verification(
                [
                    "--sampling_mode",
                    "balanced",
                    "--pair_sampling_mode",
                    "random",
                ]
            )

        with self.assertRaises(
            SystemExit
        ):
            self.parse_verification(
                [
                    "--num_pairs",
                    "40",
                    "--pair_sampling_budget",
                    "50",
                ]
            )

    def test_invalid_pair_sampling_parameters_are_rejected(self):
        invalid_cases = (
            [
                "--pair_sampling_mode",
                "unknown",
            ],
            [
                "--pair_sampling_mode",
                "balanced",
                "--pair_sampling_budget",
                "0",
            ],
            [
                "--pair_sampling_mode",
                "all_genuine",
                "--max_impostor_pairs",
                "0",
            ],
            [
                "--pair_sampling_mode",
                "random",
                "--pair_sampling_seed",
                "-1",
            ],
        )

        for extra in invalid_cases:
            with self.subTest(
                extra=extra
            ):
                with self.assertRaises(
                    SystemExit
                ):
                    self.parse_verification(
                        list(extra)
                    )

    def test_identification_tasks_do_not_consume_pair_sampling_settings(self):
        arguments, _ = (
            main.parse_experiment_arguments(
                [
                    "--dataset",
                    "ecgid",
                    "--task",
                    "1",
                    "--sampling_mode",
                    "balanced",
                    "--num_pairs",
                    "50",
                ]
            )
        )

        self.assertIsNone(
            arguments.pair_sampling_mode
        )

        self.assertIsNone(
            arguments.pair_sampling_budget
        )

        self.assertIsNone(
            arguments.max_impostor_pairs
        )

        self.assertIsNone(
            arguments.pair_sampling_seed
        )


if __name__ == "__main__":
    unittest.main()
