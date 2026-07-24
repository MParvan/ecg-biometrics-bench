import ast
import inspect
import sys
import textwrap
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run


IDENTIFICATION_RUNNERS = [
    run.run_closed_set_identification,
    run.run_subject_disjoint_identification,
    run.run_cross_session_identification,
    run.run_subject_disjoint_cross_session_identification,
]

VERIFICATION_RUNNERS = [
    run.run_closed_set_verification,
    run.run_subject_disjoint_verification,
    run.run_cross_session_verification,
    run.run_subject_disjoint_cross_session_verification,
]


def count_direct_calls(function, helper_name):
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


class MetricCallCountTests(unittest.TestCase):
    def test_identification_metrics_are_computed_once_per_runner(self):
        for runner in IDENTIFICATION_RUNNERS:
            with self.subTest(runner=runner.__name__):
                self.assertEqual(
                    count_direct_calls(
                        runner,
                        "_compute_metrics_identification",
                    ),
                    1,
                )

    def test_verification_metrics_are_computed_once_per_runner(self):
        for runner in VERIFICATION_RUNNERS:
            with self.subTest(runner=runner.__name__):
                self.assertEqual(
                    count_direct_calls(
                        runner,
                        "_compute_metrics_verification",
                    ),
                    1,
                )


if __name__ == "__main__":
    unittest.main()