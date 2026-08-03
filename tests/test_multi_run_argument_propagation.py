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


def get_function_tree(function):
    source = textwrap.dedent(
        inspect.getsource(function)
    )
    return ast.parse(source)


def count_helper_calls(function, helper_name):
    tree = get_function_tree(function)

    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == helper_name
    )


def discarded_literal_arguments(function):
    """
    Return literal argument names placed in list or tuple cleanup loops.

    This detects patterns such as:

        for key in ["...", "intelligent_weight_loading"]:
            call_args.pop(key, None)
    """
    tree = get_function_tree(function)
    discarded_names = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue

        if not isinstance(
            node.iter,
            (ast.List, ast.Tuple),
        ):
            continue

        literal_names = {
            element.value
            for element in node.iter.elts
            if isinstance(element, ast.Constant)
            and isinstance(element.value, str)
        }

        body_calls_pop = any(
            isinstance(body_node, ast.Call)
            and isinstance(body_node.func, ast.Attribute)
            and body_node.func.attr == "pop"
            for statement in node.body
            for body_node in ast.walk(statement)
        )

        if body_calls_pop:
            discarded_names.update(
                literal_names
            )

    return discarded_names


class MultiRunArgumentPropagationTests(
    unittest.TestCase
):
    def test_helper_preserves_disabled_weight_cache_setting(self):
        prepared = run._prepare_multi_run_arguments(
            {
                "seed": 42,
                "n_runs": 5,
                "_return_stats": False,
                "save_results_and_settings": True,
                "intelligent_weight_loading": False,
                "data_stats": {},
                "hyperparams": {},
            }
        )

        self.assertFalse(
            prepared["intelligent_weight_loading"]
        )
        self.assertEqual(
            prepared["n_runs"],
            1,
        )
        self.assertTrue(
            prepared["_return_stats"]
        )
        self.assertFalse(
            prepared["save_results_and_settings"]
        )

    def test_helper_preserves_enabled_weight_cache_setting(self):
        prepared = run._prepare_multi_run_arguments(
            {
                "intelligent_weight_loading": True,
            }
        )

        self.assertTrue(
            prepared["intelligent_weight_loading"]
        )

    def test_every_runner_uses_shared_argument_helper_once(self):
        for runner in RUNNERS:
            with self.subTest(
                runner=runner.__name__
            ):
                self.assertEqual(
                    count_helper_calls(
                        runner,
                        "_prepare_multi_run_arguments",
                    ),
                    1,
                )

    def test_no_runner_discards_weight_cache_setting(self):
        for runner in RUNNERS:
            with self.subTest(
                runner=runner.__name__
            ):
                discarded = (
                    discarded_literal_arguments(
                        runner
                    )
                )

                self.assertNotIn(
                    "intelligent_weight_loading",
                    discarded,
                    (
                        f"{runner.__name__} removes "
                        "intelligent_weight_loading before "
                        "recursive multi-run execution."
                    ),
                )


if __name__ == "__main__":
    unittest.main()