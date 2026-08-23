import ast
import inspect
import sys
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

import run


RUNNERS = (
    run.run_closed_set_verification,
    run.run_subject_disjoint_verification,
    run.run_cross_session_verification,
    run.run_subject_disjoint_cross_session_verification,
)


def get_function_tree(function):
    return ast.parse(
        textwrap.dedent(
            inspect.getsource(
                function
            )
        )
    )


class VerificationPairSamplingRoutingTests(
    unittest.TestCase
):
    def test_runner_interfaces_expose_canonical_and_legacy_names(self):
        for runner in RUNNERS:
            with self.subTest(
                runner=runner.__name__
            ):
                signature = (
                    inspect.signature(
                        runner
                    )
                )

                for name in (
                    "pair_sampling_mode",
                    "pair_sampling_budget",
                    "max_impostor_pairs",
                    "pair_sampling_seed",
                    "sampling_mode",
                    "num_pairs",
                ):
                    self.assertIn(
                        name,
                        signature.parameters,
                    )

                self.assertIsNone(
                    signature.parameters[
                        "sampling_mode"
                    ].default
                )

                self.assertIsNone(
                    signature.parameters[
                        "num_pairs"
                    ].default
                )

                self.assertEqual(
                    signature.parameters[
                        "max_impostor_pairs"
                    ].default,
                    1000000,
                )

                self.assertEqual(
                    signature.parameters[
                        "pair_sampling_seed"
                    ].default,
                    42,
                )

                for name in (
                    "pair_sampling_mode",
                    "pair_sampling_budget",
                    "max_impostor_pairs",
                    "pair_sampling_seed",
                ):
                    self.assertEqual(
                        signature.parameters[
                            name
                        ].kind,
                        inspect.Parameter.KEYWORD_ONLY,
                    )

    def test_each_runner_resolves_aliases_once(self):
        for runner in RUNNERS:
            with self.subTest(
                runner=runner.__name__
            ):
                tree = get_function_tree(
                    runner
                )

                calls = [
                    node
                    for node in ast.walk(
                        tree
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
                        == "_resolve_pair_sampling_arguments"
                    )
                ]

                self.assertEqual(
                    len(calls),
                    1,
                )

    def test_internal_pair_generation_routes_canonical_names(self):
        for runner in RUNNERS:
            with self.subTest(
                runner=runner.__name__
            ):
                tree = get_function_tree(
                    runner
                )

                calls = [
                    node
                    for node in ast.walk(
                        tree
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
                        == "_generate_pairs"
                    )
                ]

                self.assertEqual(
                    len(calls),
                    3,
                )

                for call in calls:
                    keywords = {
                        keyword.arg
                        for keyword
                        in call.keywords
                        if keyword.arg
                        is not None
                    }

                    self.assertTrue(
                        {
                            "pair_sampling_mode",
                            "pair_sampling_budget",
                            "max_impostor_pairs",
                            "pair_sampling_seed",
                            "matching_method",
                        }.issubset(
                            keywords
                        )
                    )

                    self.assertNotIn(
                        "sampling_mode",
                        keywords,
                    )

                    self.assertNotIn(
                        "num_pairs",
                        keywords,
                    )

    def test_logged_metadata_uses_canonical_sampling_names(self):
        for runner in RUNNERS:
            with self.subTest(
                runner=runner.__name__
            ):
                source = textwrap.dedent(
                    inspect.getsource(
                        runner
                    )
                )

                for token in (
                    '"pair_sampling_mode"',
                    '"pair_sampling_budget"',
                    '"max_impostor_pairs"',
                    '"pair_sampling_seed"',
                ):
                    self.assertIn(
                        token,
                        source,
                    )

                self.assertIn(
                    'hyperparams.pop(\n'
                    '        "num_pairs",',
                    source,
                )

                self.assertIn(
                    'hyperparams.pop(\n'
                    '        "sampling_mode",',
                    source,
                )


if __name__ == "__main__":
    unittest.main()
