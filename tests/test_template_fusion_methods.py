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
                    max_beats=None,
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