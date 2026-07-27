import ast
import inspect
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

import run


RUNNERS = [
    run.run_closed_set_identification,
    run.run_closed_set_verification,
    run.run_subject_disjoint_identification,
    run.run_subject_disjoint_verification,
    run.run_cross_session_identification,
    run.run_cross_session_verification,
    (
        run
        .run_subject_disjoint_cross_session_identification
    ),
    (
        run
        .run_subject_disjoint_cross_session_verification
    ),
]


class DummyLoader:
    def __init__(
        self,
        results_dir,
    ):
        self.cfg = {
            "root_dir": "synthetic_dataset",
            "preprocessing": {},
        }
        self.prep_params = {}
        self.results_dir = Path(
            results_dir
        )


def get_function_tree(function):
    source = textwrap.dedent(
        inspect.getsource(function)
    )
    return ast.parse(source)


def named_calls(
    function,
    helper_name,
):
    tree = get_function_tree(function)

    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(
            node.func,
            ast.Name,
        )
        and node.func.id == helper_name
    ]


class PerRunMetricLoggingTests(
    unittest.TestCase
):
    def test_identification_results_are_named_and_seeded(self):
        records = run._build_per_run_results(
            results=[
                (
                    np.float32(0.80),
                    np.float64(0.95),
                ),
                (
                    np.float32(0.90),
                    np.float64(1.00),
                ),
            ],
            seeds=[
                42,
                43,
            ],
        )

        self.assertEqual(
            records,
            [
                {
                    "run_index": 1,
                    "seed": 42,
                    "metrics": {
                        "Rank-1 Accuracy": (
                            float(
                                np.float32(0.80)
                            )
                        ),
                        "Rank-5 Accuracy": 0.95,
                    },
                },
                {
                    "run_index": 2,
                    "seed": 43,
                    "metrics": {
                        "Rank-1 Accuracy": (
                            float(
                                np.float32(0.90)
                            )
                        ),
                        "Rank-5 Accuracy": 1.0,
                    },
                },
            ],
        )

    def test_verification_results_use_all_four_metrics(self):
        records = run._build_per_run_results(
            results=[
                (
                    0.10,
                    0.91,
                    1.50,
                    0.70,
                ),
            ],
            seeds=[
                51,
            ],
        )

        self.assertEqual(
            records[0]["metrics"],
            {
                "EER": 0.10,
                "AUC": 0.91,
                "d-prime": 1.50,
                "TAR@0.1%FAR": 0.70,
            },
        )

    def test_result_and_seed_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "must match",
        ):
            run._build_per_run_results(
                results=[
                    (
                        0.8,
                        0.9,
                    ),
                ],
                seeds=[
                    42,
                    43,
                ],
            )

    def test_metric_arity_mismatch_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "same number of values",
        ):
            run._build_per_run_results(
                results=[
                    (
                        0.8,
                        0.9,
                    ),
                    (
                        0.7,
                    ),
                ],
                seeds=[
                    42,
                    43,
                ],
            )

    def test_structured_record_defaults_to_empty_per_run_list(self):
        record = (
            run._build_structured_experiment_record(
                experiment_time=(
                    __import__(
                        "datetime"
                    ).datetime(
                        2026,
                        7,
                        27,
                        12,
                        0,
                        0,
                    )
                ),
                task_name="Synthetic Task",
                dataset_name="synthetic",
                metrics_dict={
                    "Accuracy": 0.9,
                },
                data_stats={},
                hyperparams={},
                dataset_kwargs={},
                software_environment={},
                source_revision={},
                runtime_profile={},
            )
        )

        self.assertEqual(
            record["per_run_results"],
            [],
        )

    def test_logger_writes_per_run_metrics_only_to_jsonl(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            per_run_results = (
                run._build_per_run_results(
                    results=[
                        (
                            0.80,
                            0.90,
                        ),
                        (
                            0.84,
                            0.94,
                        ),
                    ],
                    seeds=[
                        42,
                        43,
                    ],
                )
            )

            with patch(
                "run._collect_software_environment",
                return_value={
                    "Python": "test",
                },
            ), patch(
                "run._collect_source_revision",
                return_value={
                    "Git Commit": "abc123",
                },
            ), patch(
                "run._collect_runtime_profile",
                return_value={},
            ):
                run._log_experiment_results(
                    task_name="Synthetic Task",
                    metrics_dict={
                        "Rank-1 Accuracy": 0.82,
                        "Rank-5 Accuracy": 0.92,
                    },
                    data_stats={
                        "Samples": 10,
                    },
                    hyperparams={
                        "run_seeds": [
                            42,
                            43,
                        ],
                    },
                    loader=DummyLoader(
                        temporary_directory
                    ),
                    per_run_results=(
                        per_run_results
                    ),
                )

            result_directory = (
                Path(temporary_directory)
                / "synthetic_dataset"
            )

            jsonl_path = (
                result_directory
                / "Synthetic_Task.jsonl"
            )
            text_path = (
                result_directory
                / "Synthetic_Task.txt"
            )

            record = json.loads(
                jsonl_path.read_text(
                    encoding="utf-8"
                ).strip()
            )

            self.assertEqual(
                record["per_run_results"],
                per_run_results,
            )

            self.assertEqual(
                record[
                    "per_run_results"
                ][1]["seed"],
                43,
            )
            self.assertEqual(
                record[
                    "per_run_results"
                ][1]["metrics"][
                    "Rank-1 Accuracy"
                ],
                0.84,
            )

            text_content = text_path.read_text(
                encoding="utf-8"
            )

            self.assertIn(
                "[RESULTS]",
                text_content,
            )
            self.assertNotIn(
                "[PER-RUN RESULTS]",
                text_content,
            )

    def test_every_runner_records_and_logs_per_run_metrics(self):
        for runner in RUNNERS:
            with self.subTest(
                runner=runner.__name__
            ):
                helper_calls = named_calls(
                    runner,
                    "_build_per_run_results",
                )

                self.assertEqual(
                    len(helper_calls),
                    1,
                )

                logger_calls = named_calls(
                    runner,
                    "_log_experiment_results",
                )

                keyword_values = []

                for logger_call in logger_calls:
                    for keyword in logger_call.keywords:
                        if (
                            keyword.arg
                            == "per_run_results"
                        ):
                            keyword_values.append(
                                keyword.value
                            )

                self.assertEqual(
                    len(keyword_values),
                    1,
                )
                self.assertIsInstance(
                    keyword_values[0],
                    ast.Name,
                )
                self.assertEqual(
                    keyword_values[0].id,
                    "per_run_results",
                )


if __name__ == "__main__":
    unittest.main()
