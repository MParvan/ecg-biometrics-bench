import ast
import datetime
import json
import sys
import tempfile
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


RUNNER_KINDS = {
    "run_closed_set_identification": (
        "identification"
    ),
    "run_closed_set_verification": (
        "verification"
    ),
    "run_subject_disjoint_identification": (
        "identification"
    ),
    "run_subject_disjoint_verification": (
        "verification"
    ),
    "run_cross_session_identification": (
        "identification"
    ),
    "run_cross_session_verification": (
        "verification"
    ),
    (
        "run_subject_disjoint_"
        "cross_session_identification"
    ): "identification",
    (
        "run_subject_disjoint_"
        "cross_session_verification"
    ): "verification",
}


class DummyLoader:
    def __init__(
        self,
        results_dir,
    ):
        self.cfg = {
            "root_dir": (
                "synthetic_dataset"
            ),
            "preprocessing": {
                "mode": "beat",
            },
        }
        self.prep_params = {}
        self.results_dir = Path(
            results_dir
        )


class StructuredArtifactTests(
    unittest.TestCase
):
    def build_record(
        self,
        **artifact_arguments,
    ):
        return (
            run._build_structured_experiment_record(
                experiment_time=(
                    datetime.datetime(
                        2026,
                        7,
                        27,
                        12,
                        0,
                        0,
                    )
                ),
                task_name=(
                    "Synthetic Task"
                ),
                dataset_name=(
                    "synthetic_dataset"
                ),
                metrics_dict={
                    "EER": 0.1,
                },
                data_stats={},
                hyperparams={},
                dataset_kwargs={},
                software_environment={},
                source_revision={},
                runtime_profile={},
                **artifact_arguments,
            )
        )

    def test_structured_record_defaults_are_stable(self):
        record = self.build_record()

        self.assertEqual(
            record[
                "evaluation_artifacts"
            ],
            {
                "single_run": None,
                "per_run": [],
            },
        )

    def test_single_run_artifact_is_preserved(self):
        artifact = {
            "type": "verification",
            "roc_curve": {
                "false_accept_rates": [
                    0.0,
                    1.0,
                ],
                "true_accept_rates": [
                    0.0,
                    1.0,
                ],
            },
        }

        record = self.build_record(
            evaluation_artifacts=artifact,
        )

        self.assertEqual(
            record[
                "evaluation_artifacts"
            ]["single_run"],
            artifact,
        )
        self.assertEqual(
            record[
                "evaluation_artifacts"
            ]["per_run"],
            [],
        )


class PerRunArtifactTests(
    unittest.TestCase
):
    def test_artifacts_are_seeded_and_ordered(self):
        artifacts = (
            run._build_per_run_evaluation_artifacts(
                artifacts=[
                    {
                        "type": (
                            "identification"
                        ),
                        "gallery_size": 3,
                    },
                    {
                        "type": (
                            "identification"
                        ),
                        "gallery_size": 4,
                    },
                ],
                seeds=[
                    42,
                    43,
                ],
            )
        )

        self.assertEqual(
            [
                artifact[
                    "run_index"
                ]
                for artifact in artifacts
            ],
            [
                1,
                2,
            ],
        )
        self.assertEqual(
            [
                artifact["seed"]
                for artifact in artifacts
            ],
            [
                42,
                43,
            ],
        )
        self.assertEqual(
            artifacts[0][
                "artifact"
            ]["gallery_size"],
            3,
        )

    def test_artifact_seed_mismatch_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "must match",
        ):
            run._build_per_run_evaluation_artifacts(
                artifacts=[
                    {
                        "type": (
                            "verification"
                        ),
                    },
                ],
                seeds=[
                    42,
                    43,
                ],
            )


    def test_context_sink_records_and_resets(self):
        artifact = {
            "type": "verification",
            "comparison_counts": {
                "genuine": 4,
                "impostor": 8,
            },
        }
        sink = []

        token = (
            run._EVALUATION_ARTIFACT_SINK.set(
                sink
            )
        )

        try:
            run._record_evaluation_artifact(
                artifact
            )
        finally:
            run._EVALUATION_ARTIFACT_SINK.reset(
                token
            )

        self.assertEqual(
            sink,
            [
                artifact,
            ],
        )

        run._record_evaluation_artifact(
            {
                "type": "identification",
            }
        )

        self.assertEqual(
            sink,
            [
                artifact,
            ],
        )


class LoggerArtifactTests(
    unittest.TestCase
):
    def test_logger_writes_artifact_only_to_jsonl(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            loader = DummyLoader(
                temporary_directory
            )

            artifact = {
                "type": "verification",
                "operating_points": [
                    {
                        "name": (
                            "TAR@0.1%FAR"
                        ),
                        "target_far": (
                            0.001
                        ),
                        "tar": 0.8,
                    },
                ],
            }

            run._log_experiment_results(
                task_name=(
                    "Synthetic Task"
                ),
                metrics_dict={
                    "EER": 0.1,
                },
                data_stats={},
                hyperparams={},
                loader=loader,
                evaluation_artifacts=(
                    artifact
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

            records = [
                json.loads(line)
                for line in (
                    jsonl_path
                    .read_text(
                        encoding="utf-8"
                    )
                    .splitlines()
                )
                if line.strip()
            ]

            self.assertEqual(
                len(records),
                1,
            )
            self.assertEqual(
                records[0][
                    "evaluation_artifacts"
                ]["single_run"],
                artifact,
            )

            text_content = (
                text_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertNotIn(
                "[EVALUATION ARTIFACTS]",
                text_content,
            )
            self.assertNotIn(
                "operating_points",
                text_content,
            )


class RunnerWiringTests(
    unittest.TestCase
):
    @classmethod
    def setUpClass(cls):
        cls.source = (
            Path(run.__file__)
            .read_text(
                encoding="utf-8"
            )
        )
        cls.tree = ast.parse(
            cls.source
        )
        cls.functions = {
            node.name: node
            for node in (
                cls.tree.body
            )
            if isinstance(
                node,
                ast.FunctionDef,
            )
        }

    def runner_source(
        self,
        runner_name,
    ):
        node = self.functions[
            runner_name
        ]

        return ast.get_source_segment(
            self.source,
            node,
        )

    def test_all_runners_compute_and_log_artifacts(self):
        for (
            runner_name,
            runner_kind,
        ) in RUNNER_KINDS.items():
            with self.subTest(
                runner=runner_name
            ):
                source = (
                    self.runner_source(
                        runner_name
                    )
                )

                expected_builder = (
                    (
                        "_build_identification_"
                        "curve_artifacts("
                    )
                    if runner_kind
                    == "identification"
                    else (
                        "_build_verification_"
                        "curve_artifacts("
                    )
                )

                self.assertEqual(
                    source.count(
                        expected_builder
                    ),
                    1,
                )
                self.assertEqual(
                    source.count(
                        (
                            "_build_per_run_"
                            "evaluation_artifacts("
                        )
                    ),
                    1,
                )
                self.assertEqual(
                    source.count(
                        (
                            "evaluation_artifacts="
                            "evaluation_artifacts"
                        )
                    ),
                    1,
                )
                self.assertEqual(
                    source.count(
                        (
                            "per_run_evaluation_artifacts="
                            "per_run_evaluation_artifacts"
                        )
                    ),
                    1,
                )
                self.assertEqual(
                    source.count(
                        (
                            "_EVALUATION_ARTIFACT_"
                            "SINK.set("
                        )
                    ),
                    1,
                )
                self.assertEqual(
                    source.count(
                        (
                            "_EVALUATION_ARTIFACT_"
                            "SINK.reset("
                        )
                    ),
                    1,
                )
                self.assertEqual(
                    source.count(
                        (
                            "_record_evaluation_"
                            "artifact("
                        )
                    ),
                    1,
                )
                self.assertIn(
                    (
                        "res, d_stats, "
                        "h_params ="
                    ),
                    source,
                )
                self.assertNotIn(
                    "run_artifacts",
                    source,
                )

    def test_internal_returns_preserve_three_value_contract(self):
        for runner_name in RUNNER_KINDS:
            with self.subTest(
                runner=runner_name
            ):
                source = (
                    self.runner_source(
                        runner_name
                    )
                )

                runner_tree = ast.parse(
                    source
                )
                runner_node = (
                    runner_tree.body[0]
                )

                three_value_returns = []
                four_value_returns = []

                for child in ast.walk(
                    runner_node
                ):
                    if not isinstance(
                        child,
                        ast.Return,
                    ):
                        continue

                    if not isinstance(
                        child.value,
                        ast.Tuple,
                    ):
                        continue

                    elements = (
                        child.value.elts
                    )

                    if (
                        len(elements) == 3
                        and isinstance(
                            elements[1],
                            ast.Name,
                        )
                        and elements[1].id
                        == "data_stats"
                        and isinstance(
                            elements[2],
                            ast.Name,
                        )
                        and elements[2].id
                        == "hyperparams"
                    ):
                        three_value_returns.append(
                            child
                        )

                    if (
                        len(elements) == 4
                        and isinstance(
                            elements[-1],
                            ast.Name,
                        )
                        and elements[-1].id
                        == "evaluation_artifacts"
                    ):
                        four_value_returns.append(
                            child
                        )

                self.assertEqual(
                    len(
                        three_value_returns
                    ),
                    1,
                )
                self.assertEqual(
                    four_value_returns,
                    [],
                )


if __name__ == "__main__":
    unittest.main()
