"""
Shared paired-comparison statistics.

These tests pin the pairing contract, the two paired tests, the paired effect
size, and the Holm correction, including the degenerate cases where a naive
implementation would divide by zero, depend on input order, or quote
floating-point residue as evidence.

The production helpers are exercised directly; the expected values are either
hand-derived or taken from SciPy for inputs that are not degenerate.
"""

import inspect
import math
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import make_figures as figures
from scripts import statistical_comparisons as sc
from scripts import statistical_utils as su

CANONICAL_SEEDS = (42, 43, 44, 45, 46)


# =========================================================================
# Seed pairing
# =========================================================================
class SeedPairingTests(unittest.TestCase):
    def test_same_seed_set_in_order_pairs_correctly(self):
        left = {42: 1.0, 43: 2.0, 44: 3.0}
        right = {42: 1.5, 43: 2.5, 44: 3.5}

        seeds, left_array, right_array = su.align_paired_seed_values(
            left, right
        )

        self.assertEqual(seeds, [42, 43, 44])
        self.assertEqual(left_array.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(right_array.tolist(), [1.5, 2.5, 3.5])

    def test_shuffled_insertion_order_produces_identical_pairs(self):
        ordered_left = {42: 1.0, 43: 2.0, 44: 3.0}
        ordered_right = {42: 1.5, 43: 2.5, 44: 3.5}

        shuffled_left = {44: 3.0, 42: 1.0, 43: 2.0}
        shuffled_right = {43: 2.5, 44: 3.5, 42: 1.5}

        ordered = su.align_paired_seed_values(ordered_left, ordered_right)
        shuffled = su.align_paired_seed_values(shuffled_left, shuffled_right)

        self.assertEqual(ordered[0], shuffled[0])
        self.assertEqual(ordered[1].tolist(), shuffled[1].tolist())
        self.assertEqual(ordered[2].tolist(), shuffled[2].tolist())

    def test_values_are_not_sorted_independently_of_seeds(self):
        # Descending values under ascending seeds: sorting the values instead
        # of indexing by seed would silently reverse one condition.
        left = {42: 9.0, 43: 5.0, 44: 1.0}
        right = {42: 1.0, 43: 5.0, 44: 9.0}

        _, left_array, right_array = su.align_paired_seed_values(left, right)

        self.assertEqual(left_array.tolist(), [9.0, 5.0, 1.0])
        self.assertEqual(right_array.tolist(), [1.0, 5.0, 9.0])

    def test_seed_missing_from_the_right_condition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "identical seed sets"):
            su.align_paired_seed_values(
                {42: 1.0, 43: 2.0},
                {42: 1.0},
            )

    def test_seed_missing_from_the_left_condition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "identical seed sets"):
            su.align_paired_seed_values(
                {42: 1.0},
                {42: 1.0, 43: 2.0},
            )

    def test_disjoint_seed_sets_are_not_silently_intersected(self):
        with self.assertRaisesRegex(ValueError, "identical seed sets"):
            su.align_paired_seed_values(
                {1: 1.0, 2: 2.0},
                {3: 1.0, 4: 2.0},
            )

    def test_seeds_need_not_begin_at_zero_or_be_contiguous(self):
        left = {701: 1.0, 12: 2.0, 9999: 3.0}
        right = {701: 1.1, 12: 2.2, 9999: 3.3}

        seeds, left_array, right_array = su.align_paired_seed_values(
            left, right
        )

        self.assertEqual(seeds, [12, 701, 9999])
        self.assertEqual(left_array.tolist(), [2.0, 1.0, 3.0])
        self.assertEqual(right_array.tolist(), [2.2, 1.1, 3.3])

    def test_canonical_seed_schedule_pairs(self):
        left = {seed: float(index) for index, seed in enumerate(CANONICAL_SEEDS)}
        right = {seed: float(index) + 0.5 for index, seed in enumerate(CANONICAL_SEEDS)}

        seeds, left_array, right_array = su.align_paired_seed_values(
            left, right
        )

        self.assertEqual(seeds, list(CANONICAL_SEEDS))
        self.assertEqual(len(left_array), 5)
        self.assertEqual(
            (right_array - left_array).tolist(),
            [0.5] * 5,
        )

    def test_non_integer_seed_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-integer"):
            su.align_paired_seed_values(
                {"42": 1.0},
                {"42": 1.0},
            )

    def test_boolean_seed_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-integer"):
            su.align_paired_seed_values(
                {True: 1.0},
                {True: 1.0},
            )

    def test_positional_sequences_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "seed-indexed mappings"):
            su.align_paired_seed_values([1.0, 2.0], [1.5, 2.5])

    def test_non_finite_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            su.align_paired_seed_values(
                {42: float("nan"), 43: 1.0},
                {42: 1.0, 43: 1.0},
            )

    def test_empty_conditions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one seed"):
            su.align_paired_seed_values({}, {})

    def test_duplicate_seeds_cannot_survive_record_extraction(self):
        """
        A mapping cannot hold a duplicate key, so duplicates must be rejected
        where per-run records are read, before pairing sees them.
        """
        record = {
            "per_run_results": [
                {"seed": 42, "metrics": {"EER": 0.1}},
                {"seed": 42, "metrics": {"EER": 0.2}},
            ]
        }

        with self.assertRaisesRegex(ValueError, "Duplicate seed"):
            sc.extract_seed_metrics(record)


# =========================================================================
# Paired t-test
# =========================================================================
class PairedTTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.array([0.50, 0.62, 0.55, 0.58, 0.52])
        self.comparison = np.array([0.61, 0.60, 0.66, 0.71, 0.58])

    def test_matches_scipy_ttest_rel(self):
        result = su.paired_t_test(self.reference, self.comparison)
        expected = stats.ttest_rel(self.comparison, self.reference)

        self.assertEqual(result["test"], "paired_t")
        self.assertAlmostEqual(result["statistic"], float(expected.statistic))
        self.assertAlmostEqual(result["raw_p"], float(expected.pvalue))
        self.assertEqual(result["n_pairs"], 5)

    def test_is_not_the_independent_samples_test(self):
        paired = su.paired_t_test(self.reference, self.comparison)
        independent = stats.ttest_ind(self.comparison, self.reference)

        self.assertNotAlmostEqual(
            paired["statistic"],
            float(independent.statistic),
        )

    def test_swapping_conditions_reverses_the_statistic_sign(self):
        forward = su.paired_t_test(self.reference, self.comparison)
        reverse = su.paired_t_test(self.comparison, self.reference)

        self.assertAlmostEqual(forward["statistic"], -reverse["statistic"])
        self.assertAlmostEqual(forward["raw_p"], reverse["raw_p"])

    def test_seed_shuffling_does_not_change_the_result(self):
        left = dict(zip(CANONICAL_SEEDS, self.reference))
        right = dict(zip(CANONICAL_SEEDS, self.comparison))
        shuffled_left = dict(reversed(list(left.items())))
        shuffled_right = dict(reversed(list(right.items())))

        _, ordered_a, ordered_b = su.align_paired_seed_values(left, right)
        _, shuffled_a, shuffled_b = su.align_paired_seed_values(
            shuffled_left, shuffled_right
        )

        self.assertEqual(
            su.paired_t_test(ordered_a, ordered_b),
            su.paired_t_test(shuffled_a, shuffled_b),
        )

    def test_all_zero_differences_report_no_effect(self):
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = su.paired_t_test(values, values)

        self.assertEqual(result["statistic"], 0.0)
        self.assertEqual(result["raw_p"], 1.0)
        self.assertEqual(result["effect_size_dz"], 0.0)

    def test_constant_non_zero_difference_is_unbounded_not_noise(self):
        reference = np.array([0.05, 0.06, 0.055, 0.058, 0.052])
        comparison = reference + 0.04

        result = su.paired_t_test(reference, comparison)

        self.assertEqual(result["statistic"], math.inf)
        self.assertEqual(result["raw_p"], 0.0)

    def test_fewer_than_two_pairs_cannot_be_tested(self):
        result = su.paired_t_test(np.array([0.5]), np.array([0.6]))

        self.assertEqual(result["n_pairs"], 1)
        self.assertIsNone(result["statistic"])
        self.assertIsNone(result["raw_p"])

    def test_non_finite_and_mismatched_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            su.paired_t_test([0.1, float("inf")], [0.2, 0.3])

        with self.assertRaisesRegex(ValueError, "same number"):
            su.paired_t_test([0.1, 0.2, 0.3], [0.2, 0.3])

    def test_n_of_five_is_supported(self):
        result = su.paired_t_test(self.reference, self.comparison)
        self.assertEqual(result["n_pairs"], 5)


# =========================================================================
# Wilcoxon signed-rank
# =========================================================================
class WilcoxonTests(unittest.TestCase):
    def test_matches_scipy_with_the_pinned_parameters(self):
        reference = np.array([0.50, 0.62, 0.55, 0.58, 0.52])
        comparison = np.array([0.61, 0.60, 0.66, 0.71, 0.58])

        result = su.wilcoxon_signed_rank(reference, comparison)
        expected = stats.wilcoxon(
            comparison,
            reference,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )

        self.assertEqual(result["test"], "wilcoxon")
        self.assertAlmostEqual(result["statistic"], float(expected.statistic))
        self.assertAlmostEqual(result["raw_p"], float(expected.pvalue))
        self.assertEqual(result["n_pairs"], 5)

    def test_mixed_sign_differences(self):
        reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        comparison = np.array([1.5, 1.0, 3.8, 3.2, 6.0])

        result = su.wilcoxon_signed_rank(reference, comparison)

        self.assertIsNotNone(result["raw_p"])
        self.assertTrue(0.0 <= result["raw_p"] <= 1.0)

    def test_some_zero_differences_are_handled(self):
        reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        comparison = np.array([1.0, 2.5, 3.0, 4.5, 5.5])

        result = su.wilcoxon_signed_rank(reference, comparison)

        self.assertIsNotNone(result["raw_p"])
        self.assertTrue(0.0 <= result["raw_p"] <= 1.0)

    def test_all_zero_differences_are_not_evidence_of_a_difference(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = su.wilcoxon_signed_rank(values, values)

        self.assertEqual(result["statistic"], 0.0)
        self.assertEqual(result["raw_p"], 1.0)

    def test_all_zero_differences_do_not_warn_or_raise(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            su.wilcoxon_signed_rank(values, values)

        self.assertEqual([str(item.message) for item in caught], [])

    def test_repeated_calls_are_deterministic(self):
        reference = np.array([0.50, 0.62, 0.55, 0.58, 0.52])
        comparison = np.array([0.61, 0.60, 0.66, 0.71, 0.58])

        first = su.wilcoxon_signed_rank(reference, comparison)
        for _ in range(5):
            self.assertEqual(
                su.wilcoxon_signed_rank(reference, comparison),
                first,
            )

    def test_fewer_than_two_pairs_cannot_be_tested(self):
        result = su.wilcoxon_signed_rank(np.array([1.0]), np.array([2.0]))

        self.assertIsNone(result["statistic"])
        self.assertIsNone(result["raw_p"])


# =========================================================================
# Cohen's dz
# =========================================================================
class CohensDzTests(unittest.TestCase):
    def test_hand_calculated_example(self):
        # differences: 1, 2, 3, 4 -> mean 2.5, sample sd = sqrt(5/3)
        reference = np.array([0.0, 0.0, 0.0, 0.0])
        comparison = np.array([1.0, 2.0, 3.0, 4.0])

        expected = 2.5 / math.sqrt(5.0 / 3.0)

        self.assertAlmostEqual(
            su.cohens_dz(reference, comparison),
            expected,
        )

    def test_uses_the_sample_standard_deviation(self):
        reference = np.array([0.0, 0.0, 0.0, 0.0])
        comparison = np.array([1.0, 2.0, 3.0, 4.0])

        differences = comparison - reference
        population = float(np.mean(differences) / np.std(differences, ddof=0))
        sample = float(np.mean(differences) / np.std(differences, ddof=1))

        self.assertAlmostEqual(su.cohens_dz(reference, comparison), sample)
        self.assertNotAlmostEqual(su.cohens_dz(reference, comparison), population)

    def test_swapping_conditions_reverses_the_sign(self):
        reference = np.array([0.50, 0.62, 0.55, 0.58, 0.52])
        comparison = np.array([0.61, 0.60, 0.66, 0.71, 0.58])

        self.assertAlmostEqual(
            su.cohens_dz(reference, comparison),
            -su.cohens_dz(comparison, reference),
        )

    def test_all_zero_differences_have_no_effect(self):
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(su.cohens_dz(values, values), 0.0)

    def test_constant_positive_difference_is_positive_infinity(self):
        reference = np.array([0.05, 0.06, 0.055, 0.058, 0.052])
        self.assertEqual(su.cohens_dz(reference, reference + 0.04), math.inf)

    def test_constant_negative_difference_is_negative_infinity(self):
        reference = np.array([0.05, 0.06, 0.055, 0.058, 0.052])
        self.assertEqual(su.cohens_dz(reference, reference - 0.04), -math.inf)

    def test_no_epsilon_is_added_to_the_denominator(self):
        """
        An epsilon denominator would turn a zero-spread comparison into a very
        large finite number instead of an unbounded one.
        """
        reference = np.array([1.0, 2.0, 3.0])
        value = su.cohens_dz(reference, reference + 1.0)

        self.assertTrue(math.isinf(value))
        self.assertFalse(math.isfinite(value))

    def test_small_sample_is_supported(self):
        self.assertTrue(
            math.isfinite(su.cohens_dz([1.0, 2.0], [1.5, 3.0]))
        )

    def test_degenerate_values_serialize_through_the_json_helper(self):
        """
        The unbounded effect sizes must survive export. Strict JSON has no
        infinity literal, so the analysis writer renders them as text rather
        than emitting a document no conformant parser will read.
        """
        import json

        reference = np.array([0.05, 0.06, 0.055])
        payload = {
            "positive": su.cohens_dz(reference, reference + 0.04),
            "negative": su.cohens_dz(reference, reference - 0.04),
            "zero": su.cohens_dz(reference, reference),
        }

        safe = sc._json_safe(payload)

        self.assertEqual(safe["positive"], "inf")
        self.assertEqual(safe["negative"], "-inf")
        self.assertEqual(safe["zero"], 0.0)

        # Strict JSON refuses raw infinities but accepts the exported form.
        with self.assertRaises(ValueError):
            json.dumps(payload, allow_nan=False)

        self.assertEqual(
            json.loads(json.dumps(safe, allow_nan=False)),
            {"positive": "inf", "negative": "-inf", "zero": 0.0},
        )


# =========================================================================
# Holm correction
# =========================================================================
class HolmTests(unittest.TestCase):
    def test_reference_values(self):
        # sorted 0.01, 0.03, 0.04 with m=3:
        #   0.01 * 3 = 0.03
        #   0.03 * 2 = 0.06
        #   0.04 * 1 = 0.04 -> raised to the running maximum 0.06
        self.assertEqual(
            su.holm_adjust([0.01, 0.04, 0.03]),
            [0.03, 0.06, 0.06],
        )

    def test_adjusted_values_are_monotonic_in_sorted_order(self):
        raw = [0.001, 0.008, 0.02, 0.04, 0.6]
        adjusted = su.holm_adjust(raw)

        ordered = [
            adjusted[index]
            for index in sorted(range(len(raw)), key=lambda i: raw[i])
        ]

        self.assertEqual(ordered, sorted(ordered))

    def test_adjusted_values_stay_within_the_unit_interval(self):
        adjusted = su.holm_adjust([0.4, 0.5, 0.6, 0.9])

        for value in adjusted:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_raw_values_are_never_modified(self):
        raw = [0.01, 0.04, 0.03]
        original = list(raw)

        su.holm_adjust(raw)

        self.assertEqual(raw, original)

    def test_ordering_of_the_input_does_not_change_the_result(self):
        import itertools

        raw = [0.001, 0.02, 0.03, 0.5]
        canonical = su.holm_adjust(raw)

        for permutation in itertools.permutations(range(len(raw))):
            permuted = [raw[index] for index in permutation]
            adjusted = su.holm_adjust(permuted)

            for position, original_index in enumerate(permutation):
                self.assertEqual(
                    adjusted[position],
                    canonical[original_index],
                    msg=f"permutation {permutation}",
                )

    def test_single_hypothesis_family_is_unchanged(self):
        self.assertEqual(su.holm_adjust([0.04]), [0.04])

    def test_equal_raw_values_receive_equal_adjusted_values(self):
        self.assertEqual(
            su.holm_adjust([0.02, 0.02, 0.02]),
            [0.06, 0.06, 0.06],
        )

    def test_untestable_hypotheses_do_not_inflate_the_family(self):
        with_none = su.holm_adjust([0.01, 0.04, 0.03, None])
        without_none = su.holm_adjust([0.01, 0.04, 0.03])

        self.assertEqual(with_none[:3], without_none)
        self.assertIsNone(with_none[3])

    def test_invalid_p_values_are_rejected(self):
        for invalid in (-0.1, 1.5, float("nan")):
            with self.assertRaisesRegex(ValueError, "Invalid p-value"):
                su.holm_adjust([0.01, invalid])

    def test_family_correction_records_decisions(self):
        records = [
            {"raw_p": 0.001},
            {"raw_p": 0.02},
            {"raw_p": 0.30},
        ]

        su.holm_correct_family(records, alpha=0.05)

        self.assertEqual(records[0]["raw_p"], 0.001)
        self.assertAlmostEqual(records[0]["adjusted_p"], 0.003)
        self.assertTrue(records[0]["reject"])
        self.assertAlmostEqual(records[1]["adjusted_p"], 0.04)
        self.assertTrue(records[1]["reject"])
        self.assertAlmostEqual(records[2]["adjusted_p"], 0.30)
        self.assertFalse(records[2]["reject"])

    def test_rejection_uses_the_adjusted_value_not_the_raw_value(self):
        # Both raw values clear 0.05; neither survives the correction.
        records = [{"raw_p": 0.03}, {"raw_p": 0.04}]

        su.holm_correct_family(records, alpha=0.05)

        self.assertLess(records[0]["raw_p"], 0.05)
        self.assertGreaterEqual(records[0]["adjusted_p"], 0.05)
        self.assertFalse(records[0]["reject"])
        self.assertFalse(records[1]["reject"])

    def test_untestable_records_have_no_decision(self):
        records = [{"raw_p": 0.01}, {"raw_p": None}]

        su.holm_correct_family(records)

        self.assertIsNone(records[1]["adjusted_p"])
        self.assertIsNone(records[1]["reject"])

    def test_invalid_alpha_is_rejected(self):
        for invalid in (0.0, 1.0, -0.5):
            with self.assertRaisesRegex(ValueError, "alpha"):
                su.holm_correct_family([{"raw_p": 0.01}], alpha=invalid)

    def test_independent_families_are_corrected_separately(self):
        family_a = [{"raw_p": 0.01}, {"raw_p": 0.04}]
        family_b = [{"raw_p": 0.01}]

        su.holm_correct_family(family_a)
        su.holm_correct_family(family_b)

        # The same raw value is adjusted differently by family size.
        self.assertAlmostEqual(family_a[0]["adjusted_p"], 0.02)
        self.assertAlmostEqual(family_b[0]["adjusted_p"], 0.01)


# =========================================================================
# The two tests are separate families
# =========================================================================
class TestFamilySeparationTests(unittest.TestCase):
    def test_manifest_families_are_distinct_per_test(self):
        rows = [
            {
                "paired_t_p_value": 0.01,
                "wilcoxon_p_value": 0.0625,
            },
            {
                "paired_t_p_value": 0.04,
                "wilcoxon_p_value": 0.0625,
            },
        ]
        comparisons = [{"metrics": rows}]

        sc._apply_holm_corrections(comparisons)

        self.assertNotEqual(
            rows[0]["paired_t_family_id"],
            rows[0]["wilcoxon_family_id"],
        )
        self.assertTrue(rows[0]["paired_t_family_id"].endswith("::paired_t"))
        self.assertTrue(rows[0]["wilcoxon_family_id"].endswith("::wilcoxon"))

    def test_wilcoxon_values_do_not_enter_the_t_test_family(self):
        rows = [
            {"paired_t_p_value": 0.01, "wilcoxon_p_value": 0.5},
            {"paired_t_p_value": 0.04, "wilcoxon_p_value": 0.5},
        ]
        sc._apply_holm_corrections([{"metrics": rows}])

        # Family size two, not four: 0.01 * 2 = 0.02.
        self.assertAlmostEqual(rows[0]["paired_t_p_value_holm"], 0.02)

    def test_raw_values_survive_the_manifest_correction(self):
        rows = [
            {"paired_t_p_value": 0.01, "wilcoxon_p_value": 0.0625},
            {"paired_t_p_value": 0.04, "wilcoxon_p_value": 0.125},
        ]
        sc._apply_holm_corrections([{"metrics": rows}])

        self.assertEqual(rows[0]["paired_t_p_value"], 0.01)
        self.assertEqual(rows[1]["paired_t_p_value"], 0.04)
        self.assertEqual(rows[0]["wilcoxon_p_value"], 0.0625)

    def test_manifest_rows_record_the_decision_and_family(self):
        rows = [{"paired_t_p_value": 0.001, "wilcoxon_p_value": 0.0625}]
        sc._apply_holm_corrections([{"metrics": rows}])

        self.assertTrue(rows[0]["paired_t_reject"])
        self.assertFalse(rows[0]["wilcoxon_reject"])
        self.assertIn("manifest", rows[0]["paired_t_family_id"])


# =========================================================================
# Figure significance markers
# =========================================================================
class FigureMarkerTests(unittest.TestCase):
    def test_marker_thresholds(self):
        self.assertEqual(su.significance_marker(0.0005), "***")
        self.assertEqual(su.significance_marker(0.005), "**")
        self.assertEqual(su.significance_marker(0.03), "*")
        self.assertEqual(su.significance_marker(0.2), "n.s.")
        self.assertEqual(su.significance_marker(None), "n/a")

    def test_threshold_boundaries_are_exclusive(self):
        self.assertEqual(su.significance_marker(0.05), "n.s.")
        self.assertEqual(su.significance_marker(0.01), "*")
        self.assertEqual(su.significance_marker(0.001), "**")

    def test_significant_raw_p_that_does_not_survive_holm_gets_no_star(self):
        """
        The case the correction exists for: two panels each significant on
        their own, neither surviving the family correction.
        """
        statistics = [
            {"p_value": 0.03, "wilcoxon_p_value": 0.0625},
            {"p_value": 0.04, "wilcoxon_p_value": 0.0625},
        ]

        figures.apply_figure_holm_correction(
            statistics,
            dataset="ecgid",
            metric="EER",
            left_protocol="short_term",
            right_protocol="long_term",
        )

        for row in statistics:
            self.assertLess(row["p_value"], 0.05)
            self.assertGreaterEqual(row["p_value_holm"], 0.05)
            self.assertFalse(row["reject"])
            self.assertEqual(su.significance_marker(row["p_value_holm"]), "n.s.")
            # The raw value would have been starred.
            self.assertEqual(su.significance_marker(row["p_value"]), "*")

    def test_strongly_significant_panel_keeps_its_star_after_correction(self):
        statistics = [
            {"p_value": 0.0001, "wilcoxon_p_value": 0.0625},
            {"p_value": 0.40, "wilcoxon_p_value": 0.5},
        ]

        figures.apply_figure_holm_correction(
            statistics,
            dataset="ecgid",
            metric="EER",
            left_protocol="short_term",
            right_protocol="long_term",
        )

        self.assertAlmostEqual(statistics[0]["p_value_holm"], 0.0002)
        self.assertTrue(statistics[0]["reject"])
        self.assertEqual(
            su.significance_marker(statistics[0]["p_value_holm"]),
            "***",
        )
        self.assertFalse(statistics[1]["reject"])

    def test_figure_family_identifier_is_built_from_the_analysis(self):
        family = figures.figure_family_id(
            "ecgid", "EER", "short_term", "long_term", "paired_t"
        )

        self.assertIn("dataset=ecgid", family)
        self.assertIn("metric=EER", family)
        self.assertIn("left=short_term", family)
        self.assertIn("right=long_term", family)
        self.assertTrue(family.endswith("::paired_t"))

    def test_figure_families_differ_per_test_type(self):
        arguments = ("ecgid", "EER", "short_term", "long_term")

        self.assertNotEqual(
            figures.figure_family_id(*arguments, "paired_t"),
            figures.figure_family_id(*arguments, "wilcoxon"),
        )

    def test_wilcoxon_correction_does_not_alter_the_t_test_decision(self):
        statistics = [
            {"p_value": 0.0001, "wilcoxon_p_value": 0.9},
            {"p_value": 0.0002, "wilcoxon_p_value": 0.9},
        ]

        figures.apply_figure_holm_correction(
            statistics,
            dataset="ecgid",
            metric="EER",
            left_protocol="short_term",
            right_protocol="long_term",
        )

        # t-test family of two, corrected independently of the Wilcoxon family.
        self.assertAlmostEqual(statistics[0]["p_value_holm"], 0.0002)
        self.assertTrue(statistics[0]["reject"])
        self.assertFalse(statistics[0]["wilcoxon_reject"])

    def test_figure_annotation_quotes_the_adjusted_value(self):
        source = inspect.getsource(figures._annotate_paired_axis)

        self.assertIn("p_value_holm", source)
        self.assertIn("significance_marker(adjusted_p)", source)

    def test_empty_family_is_handled(self):
        self.assertEqual(
            figures.apply_figure_holm_correction(
                [],
                dataset="ecgid",
                metric="EER",
                left_protocol="a",
                right_protocol="b",
            ),
            [],
        )


# =========================================================================
# One shared implementation
# =========================================================================
class SharedImplementationTests(unittest.TestCase):
    def test_figures_and_comparisons_share_the_statistics(self):
        self.assertIs(figures.paired_t_test, su.paired_t_test)
        self.assertIs(figures.wilcoxon_signed_rank, su.wilcoxon_signed_rank)
        self.assertIs(figures.significance_marker, su.significance_marker)
        self.assertIs(figures.holm_correct_family, su.holm_correct_family)
        self.assertIs(sc.paired_t_test, su.paired_t_test)
        self.assertIs(sc.wilcoxon_signed_rank, su.wilcoxon_signed_rank)

    def test_figure_alignment_delegates_to_the_shared_helper(self):
        left = {42: 1.0, 43: 2.0}
        right = {42: 1.5, 43: 2.5}

        self.assertEqual(
            figures._align_paired_seed_values(left, right)[0],
            su.align_paired_seed_values(left, right)[0],
        )

    def test_manifest_holm_delegates_to_the_shared_implementation(self):
        self.assertEqual(
            sc.holm_adjust([0.01, 0.04, 0.03]),
            su.holm_adjust([0.01, 0.04, 0.03]),
        )

    def test_no_script_calls_an_independent_samples_test(self):
        for module in (figures, sc, su):
            source = inspect.getsource(module)
            self.assertNotIn("ttest_ind", source, msg=module.__name__)

    def test_only_the_shared_module_calls_scipy_paired_tests(self):
        for module in (figures, sc):
            source = inspect.getsource(module)
            self.assertNotIn("stats.ttest_rel", source, msg=module.__name__)
            self.assertNotIn("stats.wilcoxon", source, msg=module.__name__)

    def test_figure_statistics_match_the_manifest_statistics(self):
        reference = np.array([0.50, 0.62, 0.55, 0.58, 0.52])
        comparison = np.array([0.61, 0.60, 0.66, 0.71, 0.58])

        figure_result = figures.paired_significance(reference, comparison)
        manifest_result = sc.calculate_paired_statistics(reference, comparison)

        self.assertEqual(
            figure_result["t_statistic"],
            manifest_result["paired_t_statistic"],
        )
        self.assertEqual(
            figure_result["p_value"],
            manifest_result["paired_t_p_value"],
        )
        self.assertEqual(
            figure_result["cohens_dz"],
            manifest_result["cohens_dz"],
        )
        self.assertEqual(
            figure_result["wilcoxon_p_value"],
            manifest_result["wilcoxon_p_value"],
        )


if __name__ == "__main__":
    unittest.main()
