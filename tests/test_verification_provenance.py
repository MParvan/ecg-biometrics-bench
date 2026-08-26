"""
Verification-side provenance threading (Tasks 2, 4, 6, 8).

Enrollment ordering for the four verification runners follows per-beat
provenance the same way the identification runners do: the finite template
budget selects source-first beats, and provenance stays aligned through the
initial eligibility mask, the partition indices, and SQI filtering. Runners
called without provenance keep the historical input-order behavior.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run
from test_all_task_smoke import TinyECGModel, make_synthetic_ecg_dataset
from test_identification_provenance import provenance_for


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


def _shuffle_together(x, y, provenance, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    return x[order], y[order], provenance.subset(order)


class Task2ClosedSetVerificationProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = make_synthetic_ecg_dataset(
            number_of_subjects=6, samples_per_subject=12
        )
        cls.provenance = provenance_for(cls.y)

    def test_provenance_flows_into_create_templates(self):
        with patch.object(
            run, "_create_templates", wraps=run._create_templates
        ) as spy:
            run.run_closed_set_verification(
                self.x, self.y,
                test_split=0.25, num_pairs=40, sampling_mode="all",
                use_template=True, template_fusion_method="mean",
                template_size=2,
                use_deployment_evaluation=False,
                provenance=self.provenance,
                **COMMON,
            )
        # The finite-template branch must receive provenance aligned to labels.
        template_calls = [
            call for call in spy.call_args_list
            if call.kwargs.get("max_enrollment_samples") == 2
        ]
        self.assertEqual(len(template_calls), 1)
        provenance_passed = template_calls[0].kwargs.get("provenance")
        self.assertIsNotNone(provenance_passed)
        embeddings_passed, labels_passed = template_calls[0].args[:2]
        self.assertEqual(len(provenance_passed), len(labels_passed))
        self.assertEqual(len(embeddings_passed), len(labels_passed))

    def test_template_size_none_is_unchanged_by_provenance(self):
        # method='mean' with no enrollment limit uses the fully averaged branch,
        # which is unchanged regardless of enrollment ordering.
        with patch.object(
            run, "_create_templates", wraps=run._create_templates
        ) as spy:
            run.run_closed_set_verification(
                self.x, self.y,
                test_split=0.25, num_pairs=40, sampling_mode="all",
                use_template=True, template_fusion_method="mean",
                template_size=None,
                use_deployment_evaluation=False,
                provenance=self.provenance,
                **COMMON,
            )
        template_calls = [
            call for call in spy.call_args_list
            if call.kwargs.get("max_enrollment_samples") is None
        ]
        self.assertEqual(len(template_calls), 1)

    def test_sqi_filtering_keeps_provenance_aligned(self):
        sqi_scores = np.linspace(0.0, 1.0, num=len(self.y))
        # A finite template so the provenance actually flows into templates,
        # combined with SQI filtering on both partitions.
        with patch.object(
            run, "_create_templates", wraps=run._create_templates
        ) as spy:
            run.run_closed_set_verification(
                self.x, self.y,
                test_split=0.25, num_pairs=40, sampling_mode="all",
                use_template=True, template_fusion_method="mean",
                template_size=2,
                sqi_scores=sqi_scores, sqi_threshold=0.1, sqi_keep_pct=0.9,
                outlier_filtering_on_train=True,
                outlier_filtering_on_test=True,
                use_deployment_evaluation=False,
                provenance=self.provenance,
                **COMMON,
            )
        template_calls = [
            call for call in spy.call_args_list
            if call.kwargs.get("max_enrollment_samples") == 2
        ]
        self.assertEqual(len(template_calls), 1)
        # The provenance object accompanying the template call must have the
        # same length as the (post-filter) enrollment arrays.
        provenance_passed = template_calls[0].kwargs["provenance"]
        embeddings_passed, labels_passed = template_calls[0].args[:2]
        self.assertEqual(len(provenance_passed), len(labels_passed))
        self.assertEqual(len(embeddings_passed), len(labels_passed))


class Task4SubjectDisjointVerificationProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = make_synthetic_ecg_dataset(
            number_of_subjects=8, samples_per_subject=12
        )
        cls.provenance = provenance_for(cls.y)

    def _spy_split(self):
        """Return a context manager that records helper calls and results."""
        captured = {}
        original = run._split_enrollment_probe_embeddings

        def _recording(*args, **kwargs):
            result = original(*args, **kwargs)
            captured.setdefault("calls", []).append((args, kwargs, result))
            return result

        patcher = patch.object(
            run, "_split_enrollment_probe_embeddings", side_effect=_recording
        )
        patcher._captured = captured  # type: ignore[attr-defined]
        return patcher

    def test_template_size_one_selects_source_first_when_provenance_given(self):
        # Deliberately shuffle the input so array-order first-N and source-order
        # first-N would disagree. With provenance, the runner must enrol the
        # source-first beat per subject.
        x, y, provenance = _shuffle_together(
            self.x, self.y, self.provenance, seed=17
        )
        patcher = self._spy_split()
        with patcher:
            run.run_subject_disjoint_verification(
                x, y,
                test_split=0.4, num_pairs=40, sampling_mode="all",
                use_template=True, template_fusion_method="mean",
                template_size=1,
                use_deployment_evaluation=False,
                provenance=provenance,
                **COMMON,
            )
        calls = patcher._captured["calls"]
        self.assertEqual(len(calls), 1)
        args, kwargs, result = calls[0]
        provenance_passed = kwargs.get("provenance")
        self.assertIsNotNone(provenance_passed)
        # Per subject the source-first beat has minimum beat_ordinal among the
        # beats that reached the helper.
        labels_at_split = args[1]
        beat_ord = provenance_passed.columns["beat_ordinal"]
        enrol_indices = result["indices"]["enrollment"]
        for subject in np.unique(labels_at_split):
            subject_positions = np.flatnonzero(labels_at_split == subject)
            expected = subject_positions[
                np.argmin(beat_ord[subject_positions])
            ]
            picked = [i for i in enrol_indices if labels_at_split[i] == subject]
            self.assertEqual(len(picked), 1)
            self.assertEqual(picked[0], expected)

    def test_without_provenance_keeps_input_order_enrollment(self):
        x, y, _ = _shuffle_together(
            self.x, self.y, self.provenance, seed=17
        )
        patcher = self._spy_split()
        with patcher:
            run.run_subject_disjoint_verification(
                x, y,
                test_split=0.4, num_pairs=40, sampling_mode="all",
                use_template=True, template_fusion_method="mean",
                template_size=1,
                use_deployment_evaluation=False,
                provenance=None,
                **COMMON,
            )
        calls = patcher._captured["calls"]
        self.assertEqual(len(calls), 1)
        args, kwargs, result = calls[0]
        self.assertIsNone(kwargs.get("provenance"))
        labels_at_split = args[1]
        enrol_indices = result["indices"]["enrollment"]
        for subject in np.unique(labels_at_split):
            subject_positions = np.flatnonzero(labels_at_split == subject)
            expected = subject_positions[0]
            picked = [i for i in enrol_indices if labels_at_split[i] == subject]
            self.assertEqual(len(picked), 1)
            self.assertEqual(picked[0], expected)


class Task6CrossSessionVerificationProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s1_x, cls.s1_y = make_synthetic_ecg_dataset(
            number_of_subjects=8, samples_per_subject=10, seed=456
        )
        cls.s2_x, cls.s2_y = make_synthetic_ecg_dataset(
            number_of_subjects=8, samples_per_subject=8,
            session_shift=0.1, seed=789,
        )
        cls.p1 = provenance_for(cls.s1_y)
        cls.p2 = provenance_for(cls.s2_y)

    def test_finite_template_receives_s1_provenance(self):
        with patch.object(
            run, "_create_templates", wraps=run._create_templates
        ) as spy:
            run.run_cross_session_verification(
                self.s1_x, self.s1_y, self.s2_x, self.s2_y,
                num_pairs=40, sampling_mode="all",
                use_template=True, template_fusion_method="mean",
                template_size=2,
                use_deployment_evaluation=False,
                provenance_s1=self.p1, provenance_s2=self.p2,
                **COMMON,
            )
        template_calls = [
            call for call in spy.call_args_list
            if call.kwargs.get("max_enrollment_samples") == 2
        ]
        self.assertEqual(len(template_calls), 1)
        provenance_passed = template_calls[0].kwargs.get("provenance")
        self.assertIsNotNone(provenance_passed)
        _, labels_passed = template_calls[0].args[:2]
        self.assertEqual(len(provenance_passed), len(labels_passed))


class Task8SubjectDisjointCrossSessionVerificationProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s1_x, cls.s1_y = make_synthetic_ecg_dataset(
            number_of_subjects=8, samples_per_subject=10, seed=456
        )
        cls.s2_x, cls.s2_y = make_synthetic_ecg_dataset(
            number_of_subjects=8, samples_per_subject=8,
            session_shift=0.1, seed=789,
        )
        cls.p1 = provenance_for(cls.s1_y)
        cls.p2 = provenance_for(cls.s2_y)

    def test_finite_template_receives_enrollment_provenance(self):
        with patch.object(
            run, "_create_templates", wraps=run._create_templates
        ) as spy:
            run.run_subject_disjoint_cross_session_verification(
                self.s1_x, self.s1_y, self.s2_x, self.s2_y,
                test_split=0.4, num_pairs=40, sampling_mode="all",
                use_template=True, template_fusion_method="mean",
                template_size=2,
                use_deployment_evaluation=False,
                provenance_s1=self.p1, provenance_s2=self.p2,
                **COMMON,
            )
        template_calls = [
            call for call in spy.call_args_list
            if call.kwargs.get("max_enrollment_samples") == 2
        ]
        self.assertEqual(len(template_calls), 1)
        provenance_passed = template_calls[0].kwargs.get("provenance")
        self.assertIsNotNone(provenance_passed)
        _, labels_passed = template_calls[0].args[:2]
        self.assertEqual(len(provenance_passed), len(labels_passed))


class MultiRunPropagatesVerificationProvenance(unittest.TestCase):
    """
    ``_prepare_multi_run_arguments`` copies the caller's locals dict as-is,
    which means new keyword arguments (``provenance``, ``provenance_s1``,
    ``provenance_s2``) survive the recursive single-seed re-entry without
    additional bookkeeping. Guard that invariant so future refactors of the
    helper do not silently drop them.
    """

    def test_helper_keeps_verification_provenance_keys(self):
        prepared = run._prepare_multi_run_arguments(
            {
                "provenance": "PROV_INTRA",
                "provenance_s1": "PROV_S1",
                "provenance_s2": "PROV_S2",
                "data_stats": {},
                "hyperparams": {},
            }
        )
        self.assertEqual(prepared["provenance"], "PROV_INTRA")
        self.assertEqual(prepared["provenance_s1"], "PROV_S1")
        self.assertEqual(prepared["provenance_s2"], "PROV_S2")

    def test_real_multi_run_recursion_keeps_verification_provenance(self):
        x, y = make_synthetic_ecg_dataset(
            number_of_subjects=6,
            samples_per_subject=12,
        )
        provenance = provenance_for(y)

        kwargs = dict(COMMON)
        kwargs["n_runs"] = 2
        kwargs["_return_stats"] = False

        with patch.object(
            run,
            "_create_templates",
            wraps=run._create_templates,
        ) as spy:
            run.run_closed_set_verification(
                x,
                y,
                test_split=0.25,
                num_pairs=40,
                sampling_mode="all",
                use_template=True,
                template_fusion_method="mean",
                template_size=2,
                use_deployment_evaluation=False,
                provenance=provenance,
                **kwargs,
            )

        template_calls = [
            call
            for call in spy.call_args_list
            if call.kwargs.get("max_enrollment_samples") == 2
        ]

        self.assertEqual(len(template_calls), 2)

        for call in template_calls:
            provenance_passed = call.kwargs.get("provenance")
            self.assertIsNotNone(provenance_passed)

            embeddings_passed, labels_passed = call.args[:2]

            self.assertEqual(
                len(provenance_passed),
                len(labels_passed),
            )
            self.assertEqual(
                len(embeddings_passed),
                len(labels_passed),
            )


if __name__ == "__main__":
    unittest.main()
