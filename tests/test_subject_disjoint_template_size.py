"""
Pin the enrollment budget for the subject-disjoint tasks.

Tasks 3 and 4 evaluate subjects the representation never saw, and both the
gallery and the probes come from those same subjects. The number of beats spent
on the gallery therefore decides how many are left to probe with, which is a
property of the protocol rather than something to infer from the partition.
Tasks 5 to 8 draw enrollment and probe from separately defined partitions, so
leaving the budget unset there means "every beat in the enrollment partition"
and stays valid.
"""

import re
import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main


PAPER_CONFIGS = PROJECT_ROOT / "configs" / "paper_reproduction"


def parse_arguments(task, template_size=None):
    parser = main.get_parser()

    argv = [
        "--dataset",
        "ecgid",
        "--task",
        str(task),
        "--use_template",
    ]

    if template_size is not None:
        argv += ["--template_size", str(template_size)]

    return parser, parser.parse_args(argv)


class SubjectDisjointTasksNeedAnExplicitBudget(unittest.TestCase):
    def test_tasks_three_and_four_reject_an_unset_budget(self):
        for task in (3, 4):
            parser, arguments = parse_arguments(task)
            captured = StringIO()

            with redirect_stderr(captured):
                with self.assertRaises(SystemExit) as context:
                    main.validate_experiment_arguments(arguments, parser)

            self.assertEqual(context.exception.code, 2, f"task {task}")
            self.assertIn("template_size", captured.getvalue())

    def test_tasks_three_and_four_accept_an_explicit_budget(self):
        for task in (3, 4):
            parser, arguments = parse_arguments(task, template_size=1)
            main.validate_experiment_arguments(arguments, parser)
            self.assertEqual(arguments.template_size, 1)

    def test_cross_session_tasks_still_accept_an_unset_budget(self):
        for task in (5, 6, 7, 8):
            parser, arguments = parse_arguments(task)
            main.validate_experiment_arguments(arguments, parser)
            self.assertIsNone(arguments.template_size, f"task {task}")


class PaperConfigurationsStateTheBudget(unittest.TestCase):
    def read_task_and_template_size(self, path):
        text = path.read_text(encoding="utf-8")
        task = re.search(r"^task:\s*(\d+)\s*$", text, re.M)
        size = re.search(r"^template_size:\s*(\S+)\s*$", text, re.M)
        return (
            int(task.group(1)) if task else None,
            size.group(1) if size else None,
        )

    def test_every_subject_disjoint_config_states_a_budget(self):
        if not PAPER_CONFIGS.exists():
            self.skipTest("paper reproduction pack is not present")

        checked = 0

        for path in sorted(PAPER_CONFIGS.rglob("*.yaml")):
            task, template_size = self.read_task_and_template_size(path)

            if task in (3, 4):
                checked += 1
                self.assertEqual(
                    template_size,
                    "1",
                    f"{path.name} leaves the enrollment budget unstated",
                )
            elif task in (5, 6, 7, 8):
                self.assertEqual(
                    template_size,
                    "null",
                    f"{path.name} should enroll on its whole partition",
                )

        self.assertGreater(checked, 0, "no task 3 or 4 configuration found")


if __name__ == "__main__":
    unittest.main()
