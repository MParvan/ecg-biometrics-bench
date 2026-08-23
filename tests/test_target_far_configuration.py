import contextlib
import inspect
import io
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main
import run
import utils

VERIFICATION_RUNNERS = (
    "run_closed_set_verification",
    "run_subject_disjoint_verification",
    "run_cross_session_verification",
    "run_subject_disjoint_cross_session_verification",
)


def build_separable_scores(count=400, seed=0):
    """
    Build genuine and impostor scores that produce a well-formed ROC.
    """
    generator = np.random.default_rng(seed)

    genuine = generator.normal(1.0, 0.25, count)
    impostor = generator.normal(-1.0, 0.25, count)

    scores = np.concatenate([genuine, impostor])
    labels = np.concatenate(
        [
            np.ones(count, dtype=int),
            np.zeros(count, dtype=int),
        ]
    )

    return scores, labels


class DefaultOperatingPointTests(unittest.TestCase):
    """
    The reported operating points must not change unless asked to.
    """

    def test_default_targets_are_unchanged(self):
        self.assertEqual(
            utils._DEFAULT_VERIFICATION_TARGET_FARS,
            (0.1, 0.01, 0.001, 0.0001),
        )

    def test_none_selects_the_default_targets(self):
        scores, labels = build_separable_scores()

        explicit = utils._build_verification_curve_artifacts(
            scores,
            labels,
            target_fars=list(
                utils._DEFAULT_VERIFICATION_TARGET_FARS
            ),
        )
        implicit = utils._build_verification_curve_artifacts(
            scores,
            labels,
            target_fars=None,
        )

        self.assertEqual(
            implicit["operating_points"],
            explicit["operating_points"],
        )

    def test_headline_operating_point_is_present_by_default(self):
        scores, labels = build_separable_scores()

        artifacts = utils._build_verification_curve_artifacts(
            scores,
            labels,
        )

        names = [
            point["name"]
            for point in artifacts["operating_points"]
        ]

        self.assertIn("TAR@0.1%FAR", names)


class CustomOperatingPointTests(unittest.TestCase):
    """
    Requested operating points are reported in the requested order.
    """

    def test_custom_targets_are_reported(self):
        scores, labels = build_separable_scores()

        artifacts = utils._build_verification_curve_artifacts(
            scores,
            labels,
            target_fars=[0.05, 0.005],
        )

        self.assertEqual(
            [
                point["name"]
                for point in artifacts["operating_points"]
            ],
            ["TAR@5%FAR", "TAR@0.5%FAR"],
        )

    def test_requested_order_is_preserved(self):
        scores, labels = build_separable_scores()

        artifacts = utils._build_verification_curve_artifacts(
            scores,
            labels,
            target_fars=[0.001, 0.1],
        )

        self.assertEqual(
            [
                point["target_far"]
                for point in artifacts["operating_points"]
            ],
            [0.001, 0.1],
        )


    def test_observed_far_never_exceeds_the_target(self):
        scores, labels = build_separable_scores()

        artifacts = (
            utils._build_verification_curve_artifacts(
                scores,
                labels,
                target_fars=[
                    0.2,
                    0.02,
                    0.002,
                ],
            )
        )

        for point in artifacts[
            "operating_points"
        ]:
            if not point[
                "empirically_resolvable"
            ]:
                self.assertIsNone(
                    point["observed_far"]
                )
                self.assertIsNone(
                    point["tar"]
                )
                self.assertIsNone(
                    point["frr"]
                )
                self.assertIsNone(
                    point["threshold"]
                )
                continue

            self.assertLessEqual(
                point[
                    "observed_far"
                ],
                (
                    point[
                        "target_far"
                    ]
                    + 1e-12
                ),
            )


class RunnerPlumbingTests(unittest.TestCase):
    """
    Every verification runner accepts and forwards the operating points.
    """

    def test_all_verification_runners_accept_target_fars(self):
        for runner_name in VERIFICATION_RUNNERS:
            with self.subTest(runner=runner_name):
                runner = getattr(run, runner_name)
                signature = inspect.signature(runner)

                self.assertIn(
                    "target_fars",
                    signature.parameters,
                )
                self.assertIsNone(
                    signature.parameters[
                        "target_fars"
                    ].default,
                )

    def test_identification_runners_do_not_take_target_fars(self):
        for runner_name in (
            "run_closed_set_identification",
            "run_subject_disjoint_identification",
            "run_cross_session_identification",
            "run_subject_disjoint_cross_session_identification",
        ):
            with self.subTest(runner=runner_name):
                signature = inspect.signature(
                    getattr(run, runner_name)
                )

                self.assertNotIn(
                    "target_fars",
                    signature.parameters,
                )


class CommandLineValidationTests(unittest.TestCase):
    """
    Invalid operating points must be rejected before any training starts.
    """

    def _parse(self, extra_arguments):
        buffer = io.StringIO()

        with contextlib.redirect_stdout(
            buffer
        ), contextlib.redirect_stderr(buffer):
            return main.parse_experiment_arguments(
                [
                    "--dataset",
                    "ecgid",
                    "--use_template",
                ]
                + extra_arguments
            )

    def test_default_is_none(self):
        args, _ = self._parse(["--task", "2"])

        self.assertIsNone(args.target_fars)

    def test_valid_targets_are_accepted(self):
        # Task 4 enrolls and probes on the same unseen subjects, so it asks for
        # the gallery budget to be stated.
        args, _ = self._parse(
            [
                "--task",
                "4",
                "--template_size",
                "1",
                "--target_fars",
                "0.01",
                "0.001",
            ]
        )

        self.assertEqual(
            args.target_fars,
            [0.01, 0.001],
        )

    def test_far_of_one_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse(
                [
                    "--task",
                    "2",
                    "--target_fars",
                    "1.0",
                ]
            )

    def test_far_of_zero_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse(
                [
                    "--task",
                    "2",
                    "--target_fars",
                    "0.0",
                ]
            )

    def test_duplicate_targets_are_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse(
                [
                    "--task",
                    "2",
                    "--target_fars",
                    "0.01",
                    "0.01",
                ]
            )

    def test_identification_task_rejects_target_fars(self):
        with self.assertRaises(SystemExit):
            self._parse(
                [
                    "--task",
                    "1",
                    "--target_fars",
                    "0.01",
                ]
            )


if __name__ == "__main__":
    unittest.main()
