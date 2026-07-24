import ast
import inspect
import sys
import textwrap
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run


RUNNERS = [
    run.run_closed_set_identification,
    run.run_closed_set_verification,
    run.run_subject_disjoint_identification,
    run.run_subject_disjoint_verification,
    run.run_cross_session_identification,
    run.run_cross_session_verification,
    run.run_subject_disjoint_cross_session_identification,
    run.run_subject_disjoint_cross_session_verification,
]


def count_helper_calls(function, helper_name):
    source = textwrap.dedent(
        inspect.getsource(function)
    )
    tree = ast.parse(source)

    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == helper_name
    )


class SeedMetadataTests(unittest.TestCase):
    def test_single_run_seed_schedule(self):
        original = {
            "epochs": 10,
        }

        updated = run._add_seed_metadata(
            original,
            base_seed=42,
            n_runs=1,
        )

        self.assertEqual(updated["base_seed"], 42)
        self.assertEqual(updated["n_runs"], 1)
        self.assertEqual(updated["run_seeds"], [42])

    def test_multi_run_seed_schedule(self):
        updated = run._add_seed_metadata(
            {},
            base_seed=42,
            n_runs=5,
        )

        self.assertEqual(
            updated["run_seeds"],
            [42, 43, 44, 45, 46],
        )

    def test_input_dictionary_is_not_modified(self):
        original = {
            "epochs": 10,
        }

        run._add_seed_metadata(
            original,
            base_seed=42,
            n_runs=3,
        )

        self.assertEqual(
            original,
            {
                "epochs": 10,
            },
        )

    def test_invalid_run_count_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "greater than or equal to 1",
        ):
            run._add_seed_metadata(
                {},
                base_seed=42,
                n_runs=0,
            )

    def test_each_runner_records_initial_and_aggregate_seeds(self):
        for runner in RUNNERS:
            with self.subTest(runner=runner.__name__):
                self.assertEqual(
                    count_helper_calls(
                        runner,
                        "_add_seed_metadata",
                    ),
                    2,
                )


if __name__ == "__main__":
    unittest.main()