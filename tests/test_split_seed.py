"""Regression tests for the optional ``split_seed`` (Issue #5).

``split_seed`` governs data-role allocation (train/test/cohort/validation
splits) independently of the training ``seed`` (model initialisation, DataLoader
shuffling, augmentation, validation-EER and verification-pair sampling).

When ``split_seed`` is ``None`` the allocation follows the per-run training seed,
so results are unchanged from the historical single-seed behaviour. A fixed
``split_seed`` holds the partition constant while the training seed varies.
"""

import ast
import inspect
import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main  # noqa: E402
import run  # noqa: E402


TASK_FUNCTIONS = [
    "run_closed_set_identification",
    "run_closed_set_verification",
    "run_subject_disjoint_identification",
    "run_subject_disjoint_verification",
    "run_cross_session_identification",
    "run_cross_session_verification",
    "run_subject_disjoint_cross_session_identification",
    "run_subject_disjoint_cross_session_verification",
]

# Data-role allocation calls that MUST receive the resolved split seed. The
# splitters themselves keep one generic ``seed`` parameter; only their callers
# (inside the task functions) are required to route ``resolved_split_seed``.
ALLOCATION_FUNCS = {
    "_split_closed_set_samples",   # Tasks 1/2 beat splits
    "_split_subject_cohorts",      # Tasks 3/4/7/8 cohort splits
    "train_test_split",            # validation splits (all applicable tasks)
}
ALLOCATION_SEED_KWARGS = {"seed", "random_state"}

_RUN_SOURCE = Path(run.__file__).read_text(encoding="utf-8")
_RUN_TREE = ast.parse(_RUN_SOURCE)
_TASK_NODES = {
    node.name: node
    for node in _RUN_TREE.body
    if isinstance(node, ast.FunctionDef) and node.name in TASK_FUNCTIONS
}


def _allocation_calls(task_name):
    """Yield (callee_name, ast.Call) for allocation calls inside a task fn."""
    for node in ast.walk(_TASK_NODES[task_name]):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ALLOCATION_FUNCS:
                yield node.func.id, node


def _count_calls(task_name, callee):
    return sum(1 for name, _ in _allocation_calls(task_name) if name == callee)


def make_closed_set_data(n_subjects=8, per_subject=20):
    y = np.repeat(np.arange(n_subjects), per_subject)
    x = np.column_stack(
        [np.arange(len(y), dtype=np.float32)] * 4
    )
    return x, y


class SplitSeedResolutionTests(unittest.TestCase):
    """The resolution rule and its effect on the shipped splitters."""

    def test_null_split_seed_reproduces_seed_based_partition(self):
        # resolved_split_seed = seed when split_seed is None, so the partition
        # is identical to the historical single-seed behaviour.
        x, y = make_closed_set_data()
        for seed in (42, 43, 44, 45, 46):
            resolved = seed if None is None else None  # documents the rule
            a = run._split_closed_set_samples(x, y, 0.2, seed)
            b = run._split_closed_set_samples(x, y, 0.2, resolved)
            np.testing.assert_array_equal(
                a["indices"]["holdout"], b["indices"]["holdout"]
            )

    def test_fixed_split_seed_fixes_partition_while_training_seed_varies(self):
        # A fixed split seed (resolved constant) yields the same partition for
        # every training seed; a follow-seed policy (resolved = training seed)
        # yields a different partition per seed.
        x, y = make_closed_set_data()
        fixed = [
            run._split_closed_set_samples(x, y, 0.2, 42)["indices"]["holdout"]
            for _ in range(5)
        ]
        for other in fixed[1:]:
            np.testing.assert_array_equal(fixed[0], other)

        following = [
            run._split_closed_set_samples(x, y, 0.2, s)["indices"]["holdout"]
            for s in (42, 43, 44, 45, 46)
        ]
        # At least one differs from the seed-42 partition.
        self.assertTrue(
            any(
                not np.array_equal(following[0], following[i])
                for i in range(1, 5)
            )
        )

    def test_subject_cohort_fixed_vs_following(self):
        subjects = np.arange(60)
        fixed = [
            run._split_subject_cohorts(subjects, 0.2, 0.0, 42)[2]
            for _ in range(5)
        ]
        for other in fixed[1:]:
            np.testing.assert_array_equal(np.sort(fixed[0]), np.sort(other))
        following = [
            run._split_subject_cohorts(subjects, 0.2, 0.0, s)[2]
            for s in (42, 43, 44, 45, 46)
        ]
        self.assertTrue(
            any(
                set(following[0].tolist()) != set(following[i].tolist())
                for i in range(1, 5)
            )
        )


class SplitSeedIndependenceTests(unittest.TestCase):
    """Changing the split seed must not consume the global training RNG."""

    def test_split_does_not_perturb_global_numpy_stream(self):
        x, y = make_closed_set_data()
        np.random.seed(123)
        before = np.random.random()

        np.random.seed(123)
        run._split_closed_set_samples(x, y, 0.2, 999)  # sklearn int random_state
        after = np.random.random()

        # The splitter uses an integer random_state (a private RandomState in
        # scikit-learn); it must not draw from the global NumPy stream.
        self.assertEqual(before, after)

    def test_changing_training_seed_does_not_change_fixed_partition(self):
        # Modelled at the splitter level: a fixed resolved split seed gives the
        # same partition no matter what the training seed later does.
        x, y = make_closed_set_data()
        p1 = run._split_closed_set_samples(x, y, 0.2, 42)["indices"]["holdout"]
        np.random.seed(7)  # simulate training-seed-driven draws in between
        _ = np.random.random(1000)
        p2 = run._split_closed_set_samples(x, y, 0.2, 42)["indices"]["holdout"]
        np.testing.assert_array_equal(p1, p2)


class SplitSeedRoutingTests(unittest.TestCase):
    """AST-precise: every data-role allocation call routes resolved_split_seed."""

    def test_all_task_functions_accept_split_seed(self):
        for name in TASK_FUNCTIONS:
            fn = getattr(run, name)
            self.assertIn(
                "split_seed",
                inspect.signature(fn).parameters,
                f"{name} is missing the split_seed parameter",
            )

    def test_every_task_resolves_split_seed(self):
        for name in TASK_FUNCTIONS:
            source = inspect.getsource(getattr(run, name))
            self.assertIn(
                "resolved_split_seed = seed if split_seed is None else "
                "split_seed",
                source,
                f"{name} does not resolve split_seed",
            )

    def test_every_allocation_call_uses_resolved_split_seed(self):
        # The strong contract: in every task function, each call to an
        # allocation function passes its seed/random_state as the Name
        # ``resolved_split_seed`` — never bare ``seed``.
        for name in TASK_FUNCTIONS:
            for callee, call in _allocation_calls(name):
                seed_kwargs = [
                    kw for kw in call.keywords
                    if kw.arg in ALLOCATION_SEED_KWARGS
                ]
                self.assertTrue(
                    seed_kwargs,
                    f"{name}: {callee} has no seed/random_state keyword",
                )
                for kw in seed_kwargs:
                    self.assertIsInstance(
                        kw.value, ast.Name,
                        f"{name}: {callee} {kw.arg} is not a bare name",
                    )
                    self.assertEqual(
                        kw.value.id, "resolved_split_seed",
                        f"{name}: {callee} routes {kw.arg}={kw.value.id!r}, "
                        "expected resolved_split_seed",
                    )

    def test_no_task_level_allocation_uses_bare_seed(self):
        # No task-level data-role allocation site may still use seed=seed or
        # random_state=seed after resolution.
        for name in TASK_FUNCTIONS:
            for callee, call in _allocation_calls(name):
                for kw in call.keywords:
                    if kw.arg in ALLOCATION_SEED_KWARGS:
                        is_bare_seed = (
                            isinstance(kw.value, ast.Name)
                            and kw.value.id == "seed"
                        )
                        self.assertFalse(
                            is_bare_seed,
                            f"{name}: {callee} still uses {kw.arg}=seed",
                        )

    # ---- per-task-family structural expectations -----------------------

    def test_tasks_1_2_route_both_closed_set_splits(self):
        for name in (
            "run_closed_set_identification",
            "run_closed_set_verification",
        ):
            # Train/test split + validation split -> exactly two.
            self.assertEqual(
                _count_calls(name, "_split_closed_set_samples"), 2,
                f"{name}: expected 2 _split_closed_set_samples calls",
            )

    def test_tasks_3_4_7_8_route_cohort_and_seen_validation(self):
        for name in (
            "run_subject_disjoint_identification",
            "run_subject_disjoint_verification",
            "run_subject_disjoint_cross_session_identification",
            "run_subject_disjoint_cross_session_verification",
        ):
            self.assertEqual(
                _count_calls(name, "_split_subject_cohorts"), 1,
                f"{name}: expected 1 _split_subject_cohorts call",
            )
            self.assertGreaterEqual(
                _count_calls(name, "train_test_split"), 1,
                f"{name}: expected a validation train_test_split",
            )

    def test_tasks_5_6_route_validation_only_no_partition_split(self):
        for name in (
            "run_cross_session_identification",
            "run_cross_session_verification",
        ):
            # The cross-session evaluation partition is protocol-defined; the
            # only seeded allocation is the (optional) validation split.
            self.assertEqual(
                _count_calls(name, "_split_closed_set_samples"), 0,
                f"{name}: unexpected beat split in a cross-session task",
            )
            self.assertEqual(
                _count_calls(name, "_split_subject_cohorts"), 0,
                f"{name}: unexpected cohort split in a cross-session task",
            )
            self.assertGreaterEqual(
                _count_calls(name, "train_test_split"), 1,
                f"{name}: expected a validation train_test_split",
            )

    def test_subject_cohort_splitter_keeps_single_seed_parameter(self):
        params = list(
            inspect.signature(run._split_subject_cohorts).parameters
        )
        self.assertEqual(params, ["subjects", "test_split", "val_split", "seed"])

    def test_cross_session_docstrings_state_partition_is_fixed(self):
        for name in (
            "run_cross_session_identification",
            "run_cross_session_verification",
        ):
            doc = inspect.getdoc(getattr(run, name)) or ""
            self.assertIn("protocol-defined", doc)


class SplitSeedMetadataTests(unittest.TestCase):
    def test_follow_seed_policy(self):
        m = run._add_seed_metadata({}, base_seed=42, n_runs=5, split_seed=None)
        self.assertEqual(m["run_seeds"], [42, 43, 44, 45, 46])
        self.assertEqual(m["resolved_split_seeds"], [42, 43, 44, 45, 46])
        self.assertIsNone(m["configured_split_seed"])
        self.assertEqual(m["split_seed_policy"], "follow_seed")

    def test_fixed_policy(self):
        m = run._add_seed_metadata({}, base_seed=42, n_runs=5, split_seed=42)
        self.assertEqual(m["resolved_split_seeds"], [42, 42, 42, 42, 42])
        self.assertEqual(m["configured_split_seed"], 42)
        self.assertEqual(m["split_seed_policy"], "fixed")

    def test_metadata_labels_are_neutral(self):
        # No paper "Mode A/B" terminology leaks into generic framework metadata.
        source = inspect.getsource(run._add_seed_metadata)
        self.assertNotIn("mode_b", source.lower())
        self.assertNotIn("mode a", source.lower())


class SplitSeedConfigValidationTests(unittest.TestCase):
    def _args(self, split_seed_value):
        parser = main.get_parser()
        arguments = parser.parse_args(
            ["--dataset", "ecgid", "--task", "1"]
        )
        arguments.split_seed = split_seed_value
        return parser, arguments

    def _expect_error(self, parser, arguments):
        captured = StringIO()
        with redirect_stderr(captured):
            with self.assertRaises(SystemExit) as ctx:
                main.validate_experiment_arguments(arguments, parser)
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("split_seed", captured.getvalue())

    def test_absent_split_seed_defaults_to_none(self):
        parser = main.get_parser()
        arguments = parser.parse_args(
            ["--dataset", "ecgid", "--task", "1"]
        )
        self.assertIsNone(arguments.split_seed)
        main.validate_experiment_arguments(arguments, parser)  # no error

    def test_null_split_seed_is_valid(self):
        parser, arguments = self._args(None)
        main.validate_experiment_arguments(arguments, parser)  # no error

    def test_nonnegative_integer_is_valid(self):
        parser, arguments = self._args(43)
        main.validate_experiment_arguments(arguments, parser)  # no error

    def test_negative_split_seed_is_rejected(self):
        parser, arguments = self._args(-1)
        self._expect_error(parser, arguments)

    def test_boolean_split_seed_is_rejected(self):
        # YAML `split_seed: true` becomes Python True; bool must be rejected
        # even though bool subclasses int.
        parser, arguments = self._args(True)
        self._expect_error(parser, arguments)


class SplitSeedCacheIdentityTests(unittest.TestCase):
    """split_seed must not appear in any cache identity."""

    def test_split_seed_absent_from_data_cache_builder(self):
        source = inspect.getsource(main.build_data_cache_config)
        self.assertNotIn("split_seed", source)

    def test_split_seed_absent_from_weight_cache_builder(self):
        source = inspect.getsource(run._build_weight_cache_config)
        self.assertNotIn("split_seed", source)


if __name__ == "__main__":
    unittest.main()
