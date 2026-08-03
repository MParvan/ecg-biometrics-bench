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


def get_function_node(function):
    source = textwrap.dedent(
        inspect.getsource(function)
    )
    tree = ast.parse(source)
    return tree.body[0]


def is_return_stats_guard(node):
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "_return_stats"
    )


def assigns_nonempty_data_stats_before_guard(function):
    function_node = get_function_node(function)

    return_stats_guards = [
        node
        for node in ast.walk(function_node)
        if is_return_stats_guard(node)
    ]

    if len(return_stats_guards) != 1:
        return False

    guard_line = return_stats_guards[0].lineno

    for node in ast.walk(function_node):
        if not isinstance(node, ast.Assign):
            continue

        assigns_data_stats = any(
            isinstance(target, ast.Name)
            and target.id == "data_stats"
            for target in node.targets
        )

        is_nonempty_dictionary = (
            isinstance(node.value, ast.Dict)
            and len(node.value.keys) > 0
        )

        if (
            assigns_data_stats
            and is_nonempty_dictionary
            and node.lineno < guard_line
        ):
            return True

    return False


class MultiRunDataStatisticsTests(unittest.TestCase):
    def test_data_statistics_are_populated_before_internal_return(self):
        for runner in RUNNERS:
            with self.subTest(runner=runner.__name__):
                self.assertTrue(
                    assigns_nonempty_data_stats_before_guard(
                        runner
                    ),
                    (
                        f"{runner.__name__} returns multi-run "
                        "statistics before populating data_stats."
                    ),
                )


if __name__ == "__main__":
    unittest.main()