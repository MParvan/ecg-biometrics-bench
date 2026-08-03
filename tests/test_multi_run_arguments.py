import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from run import _prepare_multi_run_arguments


class MultiRunArgumentTests(unittest.TestCase):
    def test_weight_loading_false_is_preserved(self):
        arguments = {
            "seed": 42,
            "n_runs": 5,
            "_return_stats": False,
            "save_results_and_settings": True,
            "intelligent_weight_loading": False,
            "data_stats": {"samples": 100},
            "hyperparams": {"epochs": 150},
        }

        prepared = _prepare_multi_run_arguments(arguments)

        self.assertFalse(
            prepared["intelligent_weight_loading"]
        )

    def test_weight_loading_true_is_preserved(self):
        arguments = {
            "seed": 42,
            "n_runs": 5,
            "_return_stats": False,
            "save_results_and_settings": True,
            "intelligent_weight_loading": True,
        }

        prepared = _prepare_multi_run_arguments(arguments)

        self.assertTrue(
            prepared["intelligent_weight_loading"]
        )

    def test_recursive_control_values_are_set(self):
        arguments = {
            "seed": 42,
            "n_runs": 5,
            "_return_stats": False,
            "save_results_and_settings": True,
            "intelligent_weight_loading": False,
        }

        prepared = _prepare_multi_run_arguments(arguments)

        self.assertEqual(prepared["n_runs"], 1)
        self.assertTrue(prepared["_return_stats"])
        self.assertFalse(
            prepared["save_results_and_settings"]
        )

    def test_internal_aggregation_values_are_removed(self):
        arguments = {
            "seed": 42,
            "data_stats": {"samples": 100},
            "hyperparams": {"epochs": 150},
            "call_args": {"stale": True},
        }

        prepared = _prepare_multi_run_arguments(arguments)

        self.assertNotIn("data_stats", prepared)
        self.assertNotIn("hyperparams", prepared)
        self.assertNotIn("call_args", prepared)

    def test_original_dictionary_is_not_modified(self):
        arguments = {
            "seed": 42,
            "n_runs": 5,
            "intelligent_weight_loading": False,
        }

        original = dict(arguments)

        _prepare_multi_run_arguments(arguments)

        self.assertEqual(arguments, original)


if __name__ == "__main__":
    unittest.main()