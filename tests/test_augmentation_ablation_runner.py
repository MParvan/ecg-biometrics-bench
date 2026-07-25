import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_augmentation_ablation as ablation


class AugmentationAblationRunnerTests(
    unittest.TestCase
):
    def test_method_normalization_deduplicates(self):
        methods = ablation._normalize_methods(
            [
                "gaussian",
                "time-warp",
                "gaussian",
            ]
        )

        self.assertEqual(
            methods,
            [
                "gaussian",
                "time_warp",
            ],
        )

    def test_baseline_config_disables_augmentation(self):
        base = {
            "epochs": 3,
            "use_augmentation": True,
        }

        config = ablation._build_variant_config(
            base,
            method=None,
            copies=2,
            n_runs=4,
        )

        self.assertFalse(
            config["use_augmentation"]
        )
        self.assertEqual(
            config["augmentation_parameters"],
            {},
        )
        self.assertEqual(
            config["n_runs"],
            4,
        )
        self.assertFalse(
            config["intelligent_weight_loading"]
        )

    def test_augmented_config_preserves_base_and_parameters(self):
        base = {
            "dataset": "ecgid",
            "task": 2,
            "epochs": 3,
        }

        config = ablation._build_variant_config(
            base,
            method="gaussian",
            copies=2,
            n_runs=None,
            parameters={
                "std": 0.02,
            },
        )

        self.assertEqual(
            config["dataset"],
            "ecgid",
        )
        self.assertEqual(
            config["task"],
            2,
        )
        self.assertEqual(
            config["epochs"],
            3,
        )
        self.assertTrue(
            config["use_augmentation"]
        )
        self.assertEqual(
            config["augmentation_method"],
            "gaussian",
        )
        self.assertEqual(
            config["augmentation_copies"],
            2,
        )
        self.assertEqual(
            config["augmentation_parameters"],
            {
                "std": 0.02,
            },
        )

    def test_sampling_frequency_is_required_for_frequency_methods(self):
        with self.assertRaisesRegex(
            ValueError,
            "sampling-frequency",
        ):
            ablation._get_method_parameters(
                "istft_augment",
                parameter_overrides={},
                sampling_frequency=None,
            )

        parameters = ablation._get_method_parameters(
            "istft_augment",
            parameter_overrides={},
            sampling_frequency=250,
        )

        self.assertEqual(
            parameters["fs"],
            250.0,
        )

    def test_parameter_plan_is_loaded_and_normalized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plan_path = Path(temp_dir) / "plan.yaml"
            plan_path.write_text(
                yaml.safe_dump(
                    {
                        "time-warp": {
                            "sigma": 0.3,
                        },
                    }
                ),
                encoding="utf-8",
            )

            plan = ablation._load_parameter_overrides(
                plan_path
            )

        self.assertEqual(
            plan,
            {
                "time_warp": {
                    "sigma": 0.3,
                },
            },
        )

    def test_last_results_section_is_parsed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "result.txt"
            log_path.write_text(
                "\n".join(
                    [
                        "[RESULTS]",
                        "  Rank-1 Accuracy             : 0.5000",
                        "======================================================================",
                        "[RESULTS]",
                        "  Rank-1 Accuracy             : 0.7500",
                        "  Rank-5 Accuracy             : 1.0000",
                        "======================================================================",
                    ]
                ),
                encoding="utf-8",
            )

            metrics = ablation._parse_last_section(
                log_path,
                "RESULTS",
            )

        self.assertEqual(
            metrics,
            {
                "Rank-1 Accuracy": "0.7500",
                "Rank-5 Accuracy": "1.0000",
            },
        )

    def test_changed_result_log_is_detected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results_root = Path(temp_dir)
            old_log = results_root / "old.txt"
            old_log.write_text(
                "old",
                encoding="utf-8",
            )

            snapshot = ablation._snapshot_result_logs(
                results_root
            )

            new_log = results_root / "new.txt"
            new_log.write_text(
                "new",
                encoding="utf-8",
            )

            detected = ablation._find_changed_result_log(
                snapshot,
                results_root,
            )

        self.assertEqual(
            detected,
            new_log.resolve(),
        )

    def test_target_is_read_from_yaml_and_cli_can_override(self):
        parser = Mock()
        base_config = {
            "dataset": "ptb",
            "task": 1,
        }

        dataset, task = ablation._resolve_experiment_target(
            None,
            None,
            base_config,
            parser,
        )
        self.assertEqual(
            (dataset, task),
            ("ptb", 1),
        )

        dataset, task = ablation._resolve_experiment_target(
            "ecgid",
            2,
            base_config,
            parser,
        )
        self.assertEqual(
            (dataset, task),
            ("ecgid", 2),
        )
        parser.error.assert_not_called()

    def test_main_command_uses_only_self_contained_config(self):
        command = ablation._build_main_command(
            Path("project"),
            Path("variant.yaml"),
        )

        self.assertEqual(
            command[-2:],
            [
                "--config",
                "variant.yaml",
            ],
        )
        self.assertNotIn(
            "--dataset",
            command,
        )
        self.assertNotIn(
            "--task",
            command,
        )

    def test_summary_csv_and_json_are_written(self):
        rows = [
            {
                "scenario": "baseline",
                "use_augmentation": False,
                "augmentation_method": "none",
                "augmentation_copies": 0,
                "n_runs": 2,
                "status": "success",
                "return_code": 0,
                "duration_seconds": 1.5,
                "result_log": "result.txt",
                "console_log": "console.log",
                "metrics": {
                    "EER": "0.1000",
                },
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path, json_path = ablation._write_summary(
                rows,
                temp_dir,
            )

            with csv_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                csv_rows = list(
                    csv.DictReader(file)
                )

            json_text = json_path.read_text(
                encoding="utf-8"
            )

        self.assertEqual(
            csv_rows[0]["EER"],
            "0.1000",
        )
        self.assertIn(
            '"scenario": "baseline"',
            json_text,
        )


if __name__ == "__main__":
    unittest.main()
