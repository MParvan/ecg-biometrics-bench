import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import make_figures as figures


class PalettePolicyTests(unittest.TestCase):
    """
    Colour must encode the evaluation setting and carry a second encoding.

    The palette was validated for deuteranopia, tritanopia, and normal-vision
    separation. Hatching and markers duplicate that encoding so the figures
    survive grayscale printing.
    """

    def test_every_setting_has_a_colour(self):
        self.assertEqual(
            set(figures.SETTING_COLORS),
            {"closed_set", "subject_disjoint"},
        )

    def test_colours_are_distinct(self):
        self.assertEqual(
            len(set(figures.SETTING_COLORS.values())),
            len(figures.SETTING_COLORS),
        )

    def test_every_setting_has_a_secondary_encoding(self):
        for setting in figures.SETTING_COLORS:
            with self.subTest(setting=setting):
                self.assertIn(
                    setting,
                    figures.SETTING_HATCHES,
                )
                self.assertIn(
                    setting,
                    figures.SETTING_MARKERS,
                )

    def test_hatches_and_markers_are_distinct(self):
        self.assertEqual(
            len(set(figures.SETTING_HATCHES.values())),
            len(figures.SETTING_HATCHES),
        )
        self.assertEqual(
            len(set(figures.SETTING_MARKERS.values())),
            len(figures.SETTING_MARKERS),
        )


class ConfidenceIntervalTests(unittest.TestCase):
    """
    Error bars must be intervals, not standard deviations.
    """

    def test_half_width_matches_the_closed_form(self):
        values = [0.05, 0.06, 0.055, 0.058, 0.052]

        expected = stats.t.ppf(0.975, df=4) * (
            np.std(values, ddof=1) / np.sqrt(5)
        )

        self.assertAlmostEqual(
            figures.confidence_interval(values),
            float(expected),
        )

    def test_single_value_has_no_width(self):
        self.assertEqual(
            figures.confidence_interval([0.05]),
            0.0,
        )

    def test_identical_values_have_no_width(self):
        self.assertAlmostEqual(
            figures.confidence_interval([0.05] * 5),
            0.0,
        )


class PairedSignificanceTests(unittest.TestCase):
    """
    Paired testing must use the seed pairing, not independent samples.
    """

    def test_statistic_matches_scipy(self):
        left = [0.05, 0.06, 0.055, 0.058, 0.052]
        right = [0.09, 0.10, 0.095, 0.098, 0.092]

        result = figures.paired_significance(left, right)
        expected = stats.ttest_rel(right, left)

        self.assertAlmostEqual(
            result["t_statistic"],
            float(expected.statistic),
        )
        self.assertAlmostEqual(
            result["p_value"],
            float(expected.pvalue),
        )
        self.assertEqual(result["n_pairs"], 5)

    def test_direction_is_right_minus_left(self):
        result = figures.paired_significance(
            [0.05] * 5,
            [0.09, 0.10, 0.095, 0.098, 0.092],
        )

        self.assertGreater(
            result["mean_difference"],
            0.0,
        )

    def test_effect_size_is_reported(self):
        result = figures.paired_significance(
            [0.05, 0.06, 0.055, 0.058, 0.052],
            [0.09, 0.10, 0.095, 0.098, 0.092],
        )

        self.assertIn("cohens_dz", result)
        self.assertGreater(result["cohens_dz"], 0.0)

    def test_nonparametric_test_accompanies_the_t_test(self):
        result = figures.paired_significance(
            [0.05, 0.06, 0.055, 0.058, 0.052],
            [0.09, 0.10, 0.095, 0.098, 0.092],
        )

        self.assertIn("wilcoxon_p_value", result)

    def test_identical_conditions_are_not_significant(self):
        values = [0.05, 0.06, 0.055, 0.058, 0.052]

        result = figures.paired_significance(values, values)

        self.assertEqual(result["p_value"], 1.0)
        self.assertEqual(result["cohens_dz"], 0.0)

    def test_unequal_lengths_are_truncated_to_pairs(self):
        result = figures.paired_significance(
            [0.05, 0.06, 0.055],
            [0.09, 0.10, 0.095, 0.098, 0.092],
        )

        self.assertEqual(result["n_pairs"], 3)

    def test_too_few_pairs_returns_none(self):
        self.assertIsNone(
            figures.paired_significance([0.05], [0.09])
        )


class SignificanceMarkerTests(unittest.TestCase):
    """
    Markers must follow the conventional thresholds.
    """

    def test_thresholds(self):
        cases = [
            (0.0001, "***"),
            (0.005, "**"),
            (0.03, "*"),
            (0.2, "n.s."),
            (None, "n/a"),
        ]

        for p_value, expected in cases:
            with self.subTest(p_value=p_value):
                self.assertEqual(
                    figures.significance_marker(p_value),
                    expected,
                )

    def test_boundary_is_not_significant(self):
        self.assertEqual(
            figures.significance_marker(0.05),
            "n.s.",
        )


class AxisLabelTests(unittest.TestCase):
    """
    The reader must be told which direction is better.
    """

    def test_lower_is_better_is_annotated(self):
        self.assertIn(
            "lower is better",
            figures._metric_label("EER"),
        )

    def test_higher_is_better_is_not_annotated(self):
        self.assertNotIn(
            "lower is better",
            figures._metric_label("AUC"),
        )


class RenderingTests(unittest.TestCase):
    """
    Every figure type must render to disk without manual intervention.
    """

    def setUp(self):
        self.root = Path(
            tempfile.mkdtemp(prefix="make_figures_")
        )
        figures.apply_publication_style(13.0)

        rng = np.random.default_rng(11)
        protocols = [
            "all_available",
            "single_session",
            "single_cross_session",
        ]

        self.protocols = protocols
        self.series = {
            "closed_set": {
                protocol: list(
                    rng.normal(0.06, 0.005, 5)
                )
                for protocol in protocols
            },
            "subject_disjoint": {
                protocol: list(
                    rng.normal(0.18, 0.02, 5)
                )
                for protocol in protocols
            },
        }

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_degradation_figure_renders(self):
        written = figures.plot_degradation(
            "ecgid",
            "EER",
            self.series,
            self.protocols,
            self.root,
            ["png"],
        )

        self.assertEqual(len(written), 1)
        self.assertTrue(written[0].exists())
        self.assertGreater(
            written[0].stat().st_size,
            1000,
        )

    def test_comparison_figure_renders(self):
        written = figures.plot_comparison(
            "ecgid",
            "EER",
            self.series,
            self.protocols,
            self.root,
            ["png", "pdf"],
        )

        self.assertEqual(len(written), 2)
        self.assertTrue(
            all(path.exists() for path in written)
        )

    def test_paired_figure_renders_and_reports_statistics(self):
        written, statistics = figures.plot_paired(
            "ecgid",
            "EER",
            self.series,
            "all_available",
            "single_cross_session",
            self.root,
            ["png"],
        )

        self.assertTrue(written[0].exists())
        self.assertEqual(len(statistics), 2)

        for row in statistics:
            self.assertIn("p_value", row)
            self.assertIn("cohens_dz", row)

    def test_paired_figure_requires_both_conditions(self):
        with self.assertRaises(SystemExit):
            figures.plot_paired(
                "ecgid",
                "EER",
                self.series,
                "all_available",
                "not_a_protocol",
                self.root,
                ["png"],
            )

    def test_statistics_csv_is_written(self):
        _, statistics = figures.plot_paired(
            "ecgid",
            "EER",
            self.series,
            "all_available",
            "single_cross_session",
            self.root,
            ["png"],
        )

        path = figures.write_statistics_csv(
            statistics,
            self.root / "tests.csv",
        )

        self.assertIsNotNone(path)
        self.assertTrue(path.exists())

        content = path.read_text(encoding="utf-8")

        self.assertIn("cohens_dz", content)
        self.assertIn("wilcoxon_p_value", content)

    def test_no_statistics_writes_no_file(self):
        self.assertIsNone(
            figures.write_statistics_csv(
                [],
                self.root / "empty.csv",
            )
        )


class ProtocolOrderingTests(unittest.TestCase):
    """
    Protocols must be plotted in a defined order, not an arbitrary one.
    """

    def test_known_protocols_follow_the_defined_order(self):
        series = {
            "closed_set": {
                "leave_last_out_long_term": [0.1],
                "all_available": [0.05],
                "single_session": [0.06],
            }
        }

        self.assertEqual(
            figures.order_protocols("ecgid", series),
            [
                "all_available",
                "single_session",
                "leave_last_out_long_term",
            ],
        )

    def test_unknown_protocols_are_appended(self):
        series = {
            "closed_set": {
                "zzz_custom": [0.1],
                "all_available": [0.05],
            }
        }

        self.assertEqual(
            figures.order_protocols("ecgid", series),
            ["all_available", "zzz_custom"],
        )


if __name__ == "__main__":
    unittest.main()
