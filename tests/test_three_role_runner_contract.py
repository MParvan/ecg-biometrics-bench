"""
Independent training, enrollment, and probe routing for cross-session runners.
"""

import ast
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run
from load_dataset import BeatProvenance
from test_all_task_smoke import (
    TinyECGModel,
    make_synthetic_ecg_dataset,
)


COMMON = {
    "model_class": TinyECGModel,
    "epochs": 1,
    "batch_size": 16,
    "lr": 1e-3,
    "val_split": 0.0,
    "seed": 42,
    "device": "cpu",
    "visualize": False,
    "save_results_and_settings": False,
    "loader": None,
    "n_runs": 1,
    "_return_stats": True,
    "intelligent_weight_loading": False,
}


def provenance_for(labels, source_name):
    labels = np.asarray(labels)

    counters = {}
    beat_ordinals = []

    for label in labels.tolist():
        ordinal = counters.get(
            label,
            0,
        )

        beat_ordinals.append(
            ordinal
        )

        counters[label] = ordinal + 1

    record_ids = np.asarray(
        [
            f"{source_name}_{label}.hea"
            for label in labels.tolist()
        ],
        dtype=object,
    )

    return BeatProvenance(
        {
            "record_id": record_ids,
            "session_id": np.asarray(
                [source_name] * len(labels),
                dtype=object,
            ),
            "acquisition_time": np.asarray(
                [None] * len(labels),
                dtype=object,
            ),
            "acquisition_order": np.zeros(
                len(labels),
                dtype=np.int64,
            ),
            "source_segment_id": np.asarray(
                [
                    f"{source_name}_{label}#0"
                    for label in labels.tolist()
                ],
                dtype=object,
            ),
            "source_segment_order": np.zeros(
                len(labels),
                dtype=np.float64,
            ),
            "beat_ordinal": np.asarray(
                beat_ordinals,
                dtype=np.int64,
            ),
            "rpeak_index": np.full(
                len(labels),
                -1,
                dtype=np.int64,
            ),
        }
    )


def loader_received_samples(
    loader_spy,
    expected_samples,
):
    expected_samples = np.asarray(
        expected_samples
    )

    for recorded_call in (
        loader_spy.call_args_list
    ):
        if not recorded_call.args:
            continue

        samples = np.asarray(
            recorded_call.args[0]
        )

        if (
            samples.shape
            == expected_samples.shape
            and np.array_equal(
                samples,
                expected_samples,
            )
        ):
            return True

    return False


class ThreeRoleRunnerContractTests(
    unittest.TestCase
):
    @classmethod
    def setUpClass(cls):
        cls.previous_thread_count = (
            torch.get_num_threads()
        )

        torch.set_num_threads(1)

        (
            cls.train_x,
            cls.train_y,
        ) = make_synthetic_ecg_dataset(
            number_of_subjects=8,
            samples_per_subject=8,
            signal_length=64,
            session_shift=0.0,
            seed=100,
        )

        (
            cls.enroll_x,
            cls.enroll_y,
        ) = make_synthetic_ecg_dataset(
            number_of_subjects=8,
            samples_per_subject=5,
            signal_length=64,
            session_shift=0.35,
            seed=200,
        )

        (
            cls.probe_x,
            cls.probe_y,
        ) = make_synthetic_ecg_dataset(
            number_of_subjects=8,
            samples_per_subject=6,
            signal_length=64,
            session_shift=0.70,
            seed=300,
        )

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(
            cls.previous_thread_count
        )

    def test_task5_template_gallery_uses_explicit_enrollment_samples(
        self,
    ):
        with patch.object(
            run,
            "_make_loader",
            wraps=run._make_loader,
        ) as loader_spy:
            result = (
                run.run_cross_session_identification(
                    self.train_x,
                    self.train_y,
                    self.probe_x,
                    self.probe_y,
                    use_template=True,
                    template_fusion_method="mean",
                    template_size=None,
                    probe_fusion_size=1,
                    x_enroll=self.enroll_x,
                    y_enroll=self.enroll_y,
                    **COMMON,
                )
            )

        self.assertTrue(
            loader_received_samples(
                loader_spy,
                self.enroll_x,
            )
        )

        self.assertEqual(
            result[1]["Enrollment Samples"],
            len(self.enroll_x),
        )

        self.assertEqual(
            result[1]["Training Samples"],
            len(self.train_x),
        )

    def test_task6_template_gallery_uses_explicit_enrollment_samples(
        self,
    ):
        with patch.object(
            run,
            "_make_loader",
            wraps=run._make_loader,
        ) as loader_spy:
            result = (
                run.run_cross_session_verification(
                    self.train_x,
                    self.train_y,
                    self.probe_x,
                    self.probe_y,
                    num_pairs=40,
                    sampling_mode="all",
                    use_template=True,
                    template_fusion_method="mean",
                    template_size=None,
                    use_deployment_evaluation=False,
                    x_enroll=self.enroll_x,
                    y_enroll=self.enroll_y,
                    **COMMON,
                )
            )

        self.assertTrue(
            loader_received_samples(
                loader_spy,
                self.enroll_x,
            )
        )

        self.assertEqual(
            result[1]["Enrollment Samples"],
            len(self.enroll_x),
        )

    def test_task7_gallery_uses_held_out_subjects_from_enrollment_role(
        self,
    ):
        eligible_subjects = sorted(
            set(
                self.train_y.tolist()
            )
            & set(
                self.enroll_y.tolist()
            )
            & set(
                self.probe_y.tolist()
            )
        )

        (
            _,
            _,
            test_subjects,
        ) = run._split_subject_cohorts(
            eligible_subjects,
            test_split=0.4,
            val_split=0.0,
            seed=42,
        )

        expected_enrollment = (
            self.enroll_x[
                np.isin(
                    self.enroll_y,
                    test_subjects,
                )
            ]
        )

        with patch.object(
            run,
            "_make_loader",
            wraps=run._make_loader,
        ) as loader_spy:
            result = (
                run.run_subject_disjoint_cross_session_identification(
                    self.train_x,
                    self.train_y,
                    self.probe_x,
                    self.probe_y,
                    test_split=0.4,
                    use_template=True,
                    template_fusion_method="mean",
                    template_size=None,
                    probe_fusion_size=1,
                    x_enroll=self.enroll_x,
                    y_enroll=self.enroll_y,
                    **COMMON,
                )
            )

        self.assertTrue(
            loader_received_samples(
                loader_spy,
                expected_enrollment,
            )
        )

        self.assertEqual(
            result[1]["Enrollment Samples"],
            len(expected_enrollment),
        )

    def test_task8_gallery_uses_held_out_subjects_from_enrollment_role(
        self,
    ):
        eligible_subjects = sorted(
            set(
                self.train_y.tolist()
            )
            & set(
                self.enroll_y.tolist()
            )
            & set(
                self.probe_y.tolist()
            )
        )

        (
            _,
            _,
            test_subjects,
        ) = run._split_subject_cohorts(
            eligible_subjects,
            test_split=0.4,
            val_split=0.0,
            seed=42,
        )

        expected_enrollment = (
            self.enroll_x[
                np.isin(
                    self.enroll_y,
                    test_subjects,
                )
            ]
        )

        with patch.object(
            run,
            "_make_loader",
            wraps=run._make_loader,
        ) as loader_spy:
            result = (
                run.run_subject_disjoint_cross_session_verification(
                    self.train_x,
                    self.train_y,
                    self.probe_x,
                    self.probe_y,
                    test_split=0.4,
                    num_pairs=40,
                    sampling_mode="all",
                    use_template=True,
                    template_fusion_method="mean",
                    template_size=None,
                    use_deployment_evaluation=False,
                    x_enroll=self.enroll_x,
                    y_enroll=self.enroll_y,
                    **COMMON,
                )
            )

        self.assertTrue(
            loader_received_samples(
                loader_spy,
                expected_enrollment,
            )
        )

        self.assertEqual(
            result[1]["Enrollment Samples"],
            len(expected_enrollment),
        )

    def test_no_template_branches_ignore_unused_enrollment_arguments(
        self,
    ):
        malformed_enrollment = np.zeros(
            (3, self.train_x.shape[-1]),
            dtype=np.float32,
        )

        task5 = (
            run.run_cross_session_identification(
                self.train_x,
                self.train_y,
                self.probe_x,
                self.probe_y,
                use_template=False,
                probe_fusion_size=1,
                x_enroll=malformed_enrollment,
                y_enroll=None,
                **COMMON,
            )
        )

        self.assertEqual(
            task5[1]["Enrollment Samples"],
            0,
        )

        task6 = (
            run.run_cross_session_verification(
                self.train_x,
                self.train_y,
                self.probe_x,
                self.probe_y,
                num_pairs=40,
                sampling_mode="all",
                use_template=False,
                use_deployment_evaluation=False,
                x_enroll=malformed_enrollment,
                y_enroll=None,
                **COMMON,
            )
        )

        self.assertEqual(
            task6[1]["Enrollment Samples"],
            0,
        )

        task8 = (
            run.run_subject_disjoint_cross_session_verification(
                self.train_x,
                self.train_y,
                self.probe_x,
                self.probe_y,
                test_split=0.4,
                num_pairs=40,
                sampling_mode="all",
                use_template=False,
                use_deployment_evaluation=False,
                x_enroll=malformed_enrollment,
                y_enroll=None,
                **COMMON,
            )
        )

        self.assertEqual(
            task8[1]["Enrollment Samples"],
            0,
        )

    def test_task7_rejects_incomplete_explicit_enrollment_arguments(
        self,
    ):
        with self.assertRaisesRegex(
            ValueError,
            "both be provided or both be omitted",
        ):
            run.run_subject_disjoint_cross_session_identification(
                self.train_x,
                self.train_y,
                self.probe_x,
                self.probe_y,
                test_split=0.4,
                use_template=True,
                template_fusion_method="mean",
                template_size=None,
                probe_fusion_size=1,
                x_enroll=self.enroll_x,
                y_enroll=None,
                **COMMON,
            )

    def test_separate_enrollment_provenance_reaches_template_selection(
        self,
    ):
        enrollment_provenance = provenance_for(
            self.enroll_y,
            "enrollment",
        )

        with patch.object(
            run,
            "_create_templates",
            wraps=run._create_templates,
        ) as template_spy:
            run.run_cross_session_identification(
                self.train_x,
                self.train_y,
                self.probe_x,
                self.probe_y,
                use_template=True,
                template_fusion_method="mean",
                template_size=2,
                probe_fusion_size=1,
                x_enroll=self.enroll_x,
                y_enroll=self.enroll_y,
                provenance_enroll=enrollment_provenance,
                **COMMON,
            )

        template_calls = [
            recorded_call
            for recorded_call
            in template_spy.call_args_list
            if (
                recorded_call.kwargs.get(
                    "max_enrollment_samples"
                )
                == 2
            )
        ]

        self.assertEqual(
            len(template_calls),
            1,
        )

        passed_provenance = (
            template_calls[0]
            .kwargs
            .get("provenance")
        )

        self.assertIsNotNone(
            passed_provenance
        )

        record_ids = (
            passed_provenance
            .columns[
                "record_id"
            ]
            .tolist()
        )

        self.assertTrue(
            record_ids
        )

        self.assertTrue(
            all(
                str(record_id).startswith(
                    "enrollment_"
                )
                for record_id in record_ids
            )
        )


    def test_multi_run_preserves_explicit_enrollment_arrays(
        self,
    ):
        with patch.object(
            run,
            "_prepare_cross_session_enrollment_role",
            wraps=(
                run._prepare_cross_session_enrollment_role
            ),
        ) as enrollment_spy:
            run.run_cross_session_identification(
                self.train_x,
                self.train_y,
                self.probe_x,
                self.probe_y,
                use_template=True,
                template_fusion_method="mean",
                template_size=None,
                probe_fusion_size=1,
                x_enroll=self.enroll_x,
                y_enroll=self.enroll_y,
                n_runs=2,
                _return_stats=False,
                model_class=TinyECGModel,
                epochs=1,
                batch_size=16,
                lr=1e-3,
                val_split=0.0,
                seed=42,
                device="cpu",
                visualize=False,
                save_results_and_settings=False,
                loader=None,
                intelligent_weight_loading=False,
            )

        self.assertEqual(
            enrollment_spy.call_count,
            2,
        )

        for recorded_call in (
            enrollment_spy.call_args_list
        ):
            self.assertIs(
                recorded_call.kwargs[
                    "x_enroll"
                ],
                self.enroll_x,
            )

            self.assertIs(
                recorded_call.kwargs[
                    "y_enroll"
                ],
                self.enroll_y,
            )

    def test_all_cross_session_runners_request_training_role_weight_identity(
        self,
    ):
        source = Path(
            run.__file__
        ).read_text(
            encoding="utf-8"
        )

        tree = ast.parse(
            source
        )

        functions = {
            node.name: node
            for node in tree.body
            if isinstance(
                node,
                ast.FunctionDef,
            )
        }

        for runner_name in (
            "run_cross_session_identification",
            "run_cross_session_verification",
            "run_subject_disjoint_cross_session_identification",
            "run_subject_disjoint_cross_session_verification",
        ):
            with self.subTest(
                runner=runner_name
            ):
                runner_node = functions[
                    runner_name
                ]

                calls = [
                    node
                    for node in ast.walk(
                        runner_node
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
                        == "_build_weight_cache_config"
                    )
                ]

                self.assertEqual(
                    len(calls),
                    1,
                )

                keywords = {
                    keyword.arg: keyword.value
                    for keyword
                    in calls[0].keywords
                    if keyword.arg is not None
                }

                self.assertIn(
                    "training_role_only_loader_identity",
                    keywords,
                )

                value = keywords[
                    "training_role_only_loader_identity"
                ]

                self.assertIsInstance(
                    value,
                    ast.Constant,
                )

                self.assertIs(
                    value.value,
                    True,
                )


class EnrollmentPreparationValidationTests(
    unittest.TestCase
):
    def setUp(self):
        self.x_enroll = np.arange(
            24,
            dtype=np.float32,
        ).reshape(
            6,
            4,
        )

        self.y_enroll = np.asarray(
            [
                0,
                0,
                0,
                1,
                1,
                1,
            ]
        )

    def test_explicit_enrollment_is_returned_unchanged(
        self,
    ):
        (
            x_prepared,
            y_prepared,
            provenance_prepared,
            is_explicit,
        ) = run._prepare_cross_session_enrollment_role(
            x_enroll=self.x_enroll,
            y_enroll=self.y_enroll,
            provenance_enroll=None,
            use_enrollment=True,
        )

        self.assertIs(
            x_prepared,
            self.x_enroll,
        )

        self.assertIs(
            y_prepared,
            self.y_enroll,
        )

        self.assertIsNone(
            provenance_prepared
        )

        self.assertTrue(
            is_explicit
        )

    def test_enrollment_provenance_requires_explicit_arrays(
        self,
    ):
        provenance = provenance_for(
            self.y_enroll,
            "enrollment",
        )

        with self.assertRaisesRegex(
            ValueError,
            "provenance_enroll requires explicit",
        ):
            run._prepare_cross_session_enrollment_role(
                x_enroll=None,
                y_enroll=None,
                provenance_enroll=provenance,
                use_enrollment=True,
            )

    def test_enrollment_provenance_must_be_aligned(
        self,
    ):
        provenance = provenance_for(
            self.y_enroll,
            "enrollment",
        ).subset(
            np.arange(
                len(self.y_enroll) - 1
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            "Enrollment provenance is misaligned",
        ):
            run._prepare_cross_session_enrollment_role(
                x_enroll=self.x_enroll,
                y_enroll=self.y_enroll,
                provenance_enroll=provenance,
                use_enrollment=True,
            )

    def test_unused_enrollment_role_ignores_incomplete_arrays(
        self,
    ):
        (
            x_prepared,
            y_prepared,
            provenance_prepared,
            is_explicit,
        ) = run._prepare_cross_session_enrollment_role(
            x_enroll=self.x_enroll,
            y_enroll=None,
            provenance_enroll=None,
            use_enrollment=False,
        )

        self.assertIsNone(
            x_prepared
        )

        self.assertIsNone(
            y_prepared
        )

        self.assertIsNone(
            provenance_prepared
        )

        self.assertFalse(
            is_explicit
        )


if __name__ == "__main__":
    unittest.main()
