import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main
from utils import _create_templates


FUSION_METHODS = [
    "mean",
    "median",
    "trimmed_mean",
    "representative",
    "soft_centrality",
    "geometric_median",
    "none",
]


class TemplateFusionMethodTests(unittest.TestCase):
    def setUp(self):
        self.embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.8, 0.2, 0.1],
                [1.1, 0.0, 0.1],
                [0.0, 1.0, 0.0],
                [0.1, 0.9, 0.0],
                [0.2, 0.8, 0.1],
                [0.0, 1.1, 0.1],
            ],
            dtype=np.float64,
        )

        self.labels = np.array(
            [0, 0, 0, 0, 1, 1, 1, 1],
            dtype=int,
        )

    def test_cli_exposes_every_implemented_method(self):
        parser = main.get_parser()

        action = next(
            parser_action
            for parser_action in parser._actions
            if parser_action.dest == "template_fusion_method"
        )

        self.assertEqual(
            set(action.choices),
            set(FUSION_METHODS),
        )

    def test_all_fusion_methods_produce_valid_outputs(self):
        for method in FUSION_METHODS:
            with self.subTest(method=method):
                templates, template_labels = _create_templates(
                    self.embeddings,
                    self.labels,
                    method=method,
                    max_enrollment_samples=None,
                )

                if method == "none":
                    self.assertEqual(
                        templates.shape,
                        self.embeddings.shape,
                    )
                    np.testing.assert_array_equal(
                        template_labels,
                        self.labels,
                    )
                else:
                    self.assertEqual(
                        templates.shape,
                        (2, 3),
                    )
                    np.testing.assert_array_equal(
                        template_labels,
                        np.array([0, 1]),
                    )
                    self.assertTrue(
                        np.all(np.isfinite(templates))
                    )

    def test_private_enrollment_limit_parameter_has_precise_name(self):
        parameters = inspect.signature(_create_templates).parameters
        self.assertIn("max_enrollment_samples", parameters)
        self.assertNotIn("max_beats", parameters)

    def test_no_fusion_without_limit_retains_every_observation(self):
        templates, template_labels = _create_templates(
            self.embeddings,
            self.labels,
            method="none",
            max_enrollment_samples=None,
        )

        np.testing.assert_array_equal(templates, self.embeddings)
        np.testing.assert_array_equal(template_labels, self.labels)

    def test_no_fusion_limit_retains_first_observations_per_identity(self):
        templates, template_labels = _create_templates(
            self.embeddings,
            self.labels,
            method="none",
            max_enrollment_samples=2,
        )

        expected_indices = np.array([0, 1, 4, 5])
        np.testing.assert_array_equal(
            templates,
            self.embeddings[expected_indices],
        )
        np.testing.assert_array_equal(
            template_labels,
            self.labels[expected_indices],
        )

    def test_no_fusion_limit_is_repeatable(self):
        first = _create_templates(
            self.embeddings,
            self.labels,
            method="none",
            max_enrollment_samples=2,
        )
        second = _create_templates(
            self.embeddings,
            self.labels,
            method="none",
            max_enrollment_samples=2,
        )

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_oversized_no_fusion_limit_retains_original_arrays(self):
        templates, template_labels = _create_templates(
            self.embeddings,
            self.labels,
            method="none",
            max_enrollment_samples=100,
        )

        np.testing.assert_array_equal(templates, self.embeddings)
        np.testing.assert_array_equal(template_labels, self.labels)

    def test_fusing_methods_keep_selection_before_fusion(self):
        selected_indices = np.array([0, 1, 4, 5])

        for method in FUSION_METHODS:
            if method == "none":
                continue

            with self.subTest(method=method):
                limited = _create_templates(
                    self.embeddings,
                    self.labels,
                    method=method,
                    max_enrollment_samples=2,
                )
                manually_selected = _create_templates(
                    self.embeddings[selected_indices],
                    self.labels[selected_indices],
                    method=method,
                    max_enrollment_samples=None,
                )

                np.testing.assert_allclose(
                    limited[0],
                    manually_selected[0],
                )
                np.testing.assert_array_equal(
                    limited[1],
                    manually_selected[1],
                )

    def test_parser_accepts_each_fusion_method(self):
        parser = main.get_parser()

        for method in FUSION_METHODS:
            with self.subTest(method=method):
                arguments = parser.parse_args(
                    [
                        "--dataset",
                        "ecgid",
                        "--task",
                        "1",
                        "--template_fusion_method",
                        method,
                    ]
                )

                self.assertEqual(
                    arguments.template_fusion_method,
                    method,
                )


if __name__ == "__main__":
    unittest.main()
