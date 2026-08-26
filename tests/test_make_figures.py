import math
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
        # Differences must actually vary, otherwise the comparison would pin
        # floating-point residue rather than a test statistic.
        left = [0.05, 0.06, 0.055, 0.058, 0.052]
        right = [0.09, 0.11, 0.094, 0.101, 0.088]

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

    def test_constant_difference_reports_an_unbounded_statistic(self):
        """
        A constant shift has no spread, so the standardized separation is
        unbounded. Reporting a finite statistic here would quote floating-point
        residue as though it were evidence.
        """
        left = [0.05, 0.06, 0.055, 0.058, 0.052]
        right = [value + 0.04 for value in left]

        result = figures.paired_significance(left, right)

        self.assertEqual(result["t_statistic"], math.inf)
        self.assertEqual(result["p_value"], 0.0)
        self.assertEqual(result["cohens_dz"], math.inf)

    def test_constant_negative_difference_keeps_the_sign(self):
        left = [0.05, 0.06, 0.055, 0.058, 0.052]
        right = [value - 0.04 for value in left]

        result = figures.paired_significance(left, right)

        self.assertEqual(result["t_statistic"], -math.inf)
        self.assertEqual(result["cohens_dz"], -math.inf)

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

    def test_unequal_lengths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            figures.paired_significance(
                [0.05, 0.06, 0.055],
                [0.09, 0.10, 0.095, 0.098, 0.092],
            )

    def test_non_finite_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            figures.paired_significance(
                [0.05, float("nan"), 0.055, 0.058, 0.052],
                [0.09, 0.10, 0.095, 0.098, 0.092],
            )

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
        self.seeds = [42, 43, 44, 45, 46]
        self.series = {
            "closed_set": {
                protocol: dict(
                    zip(
                        self.seeds,
                        rng.normal(0.06, 0.005, len(self.seeds)),
                    )
                )
                for protocol in protocols
            },
            "subject_disjoint": {
                protocol: dict(
                    zip(
                        self.seeds,
                        rng.normal(0.18, 0.02, len(self.seeds)),
                    )
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


class SeedMetricExtractionTests(unittest.TestCase):
    """
    Missing-strict per-seed metric collection.

    Any missing, None, non-numeric, or non-finite metric value must mark the
    whole configuration as unavailable so figure code cannot silently average
    a subset of runs.
    """

    def _record(self, per_run):
        return {"per_run_results": per_run}

    def test_returns_seed_indexed_values(self):
        record = self._record(
            [
                {"seed": 42, "metrics": {"EER": 0.05}},
                {"seed": 43, "metrics": {"EER": 0.06}},
                {"seed": 44, "metrics": {"EER": 0.055}},
            ]
        )

        self.assertEqual(
            figures._extract_seed_metric(record, "EER"),
            {42: 0.05, 43: 0.06, 44: 0.055},
        )

    def test_missing_metric_omits_configuration(self):
        record = self._record(
            [
                {"seed": 42, "metrics": {"EER": 0.05}},
                {"seed": 43, "metrics": {"EER": None}},
                {"seed": 44, "metrics": {"EER": 0.055}},
            ]
        )

        self.assertIsNone(figures._extract_seed_metric(record, "EER"))

    def test_absent_metric_key_omits_configuration(self):
        record = self._record(
            [
                {"seed": 42, "metrics": {"EER": 0.05}},
                {"seed": 43, "metrics": {"AUC": 0.99}},
            ]
        )

        self.assertIsNone(figures._extract_seed_metric(record, "EER"))

    def test_non_finite_value_omits_configuration(self):
        record = self._record(
            [
                {"seed": 42, "metrics": {"EER": 0.05}},
                {"seed": 43, "metrics": {"EER": float("nan")}},
            ]
        )

        self.assertIsNone(figures._extract_seed_metric(record, "EER"))

    def test_duplicate_seed_omits_configuration(self):
        record = self._record(
            [
                {"seed": 42, "metrics": {"EER": 0.05}},
                {"seed": 42, "metrics": {"EER": 0.06}},
            ]
        )

        self.assertIsNone(figures._extract_seed_metric(record, "EER"))

    def test_missing_seed_omits_configuration(self):
        record = self._record(
            [
                {"metrics": {"EER": 0.05}},
                {"seed": 43, "metrics": {"EER": 0.06}},
            ]
        )

        self.assertIsNone(figures._extract_seed_metric(record, "EER"))

    def test_non_integer_seed_types_are_rejected(self):
        cases = [
            ("float_with_fraction", 42.5),
            ("float_integer_valued", 42.0),
            ("numeric_string", "42"),
            ("boolean_true", True),
            ("boolean_false", False),
        ]

        for label, bad_seed in cases:
            with self.subTest(label=label):
                record = self._record(
                    [
                        {
                            "seed": bad_seed,
                            "metrics": {"EER": 0.05},
                        },
                    ]
                )
                self.assertIsNone(
                    figures._extract_seed_metric(record, "EER")
                )

    def test_numpy_integer_seed_is_accepted(self):
        record = self._record(
            [
                {"seed": np.int64(42), "metrics": {"EER": 0.05}},
                {"seed": np.int32(43), "metrics": {"EER": 0.06}},
            ]
        )

        self.assertEqual(
            figures._extract_seed_metric(record, "EER"),
            {42: 0.05, 43: 0.06},
        )

    def test_empty_record_returns_none(self):
        self.assertIsNone(figures._extract_seed_metric({}, "EER"))
        self.assertIsNone(figures._extract_seed_metric(None, "EER"))
        self.assertIsNone(
            figures._extract_seed_metric(
                {"per_run_results": []}, "EER"
            )
        )


class PairedSeedAlignmentTests(unittest.TestCase):
    """
    Seed-set alignment for paired conditions.
    """

    def test_shuffled_seeds_align_by_identity(self):
        left = {43: 0.06, 42: 0.05, 44: 0.055}
        right = {44: 0.07, 42: 0.09, 43: 0.10}

        seeds, left_arr, right_arr = figures._align_paired_seed_values(
            left, right
        )

        self.assertEqual(seeds, [42, 43, 44])
        np.testing.assert_array_equal(left_arr, [0.05, 0.06, 0.055])
        np.testing.assert_array_equal(right_arr, [0.09, 0.10, 0.07])

    def test_mismatched_seed_sets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "identical seed sets"):
            figures._align_paired_seed_values(
                {42: 0.05, 43: 0.06},
                {42: 0.09, 44: 0.10},
            )

    def test_subset_is_rejected_not_silently_intersected(self):
        with self.assertRaisesRegex(ValueError, "identical seed sets"):
            figures._align_paired_seed_values(
                {42: 0.05, 43: 0.06, 44: 0.055},
                {42: 0.09, 43: 0.10},
            )

    def test_non_finite_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            figures._align_paired_seed_values(
                {42: 0.05, 43: float("inf")},
                {42: 0.09, 43: 0.10},
            )

    def test_non_integer_seed_keys_are_rejected(self):
        cases = [
            ("float_key", {42.5: 0.05, 43: 0.06}),
            ("string_key", {"42": 0.05, 43: 0.06}),
            ("boolean_key", {True: 0.05, 43: 0.06}),
        ]
        good = {42: 0.09, 43: 0.10}

        for label, bad_left in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    ValueError, "non-integer seed"
                ):
                    figures._align_paired_seed_values(bad_left, good)

    def test_positional_lists_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "seed-indexed mappings"
        ):
            figures._align_paired_seed_values(
                [0.05, 0.06],
                [0.09, 0.10],
            )


if __name__ == "__main__":
    unittest.main()
