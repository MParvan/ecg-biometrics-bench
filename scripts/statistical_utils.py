"""
Shared paired-comparison statistics for benchmark result analysis.

This module is the single implementation of the inferential statistics used by
the result-analysis scripts. It operates on per-run metrics that already exist;
it never recomputes a biometric metric.

The repeated-run unit is the run seed. Two conditions are compared by aligning
them on seed identity, never on the order in which records happen to appear in
a file, so a paired comparison always speaks about the same training runs.

Three analyses are provided for a pair of seed-aligned conditions, all defined
in the direction ``comparison - reference``:

* a paired t-test (``scipy.stats.ttest_rel``),
* a Wilcoxon signed-rank test with explicitly pinned parameters, and
* Cohen's dz, the mean paired difference divided by the sample standard
  deviation of those differences.

Families of hypotheses are corrected with the Holm step-down procedure. A
family must be a prespecified set of hypotheses, and the parametric and
non-parametric tests are always corrected as separate families.
"""

import math

import numpy as np
from scipy import stats

# Rejection threshold used when a caller does not supply one. This matches the
# 0.05 convention already used for the reported significance markers.
DEFAULT_ALPHA = 0.05

PAIRED_T_TEST = "paired_t"
WILCOXON_TEST = "wilcoxon"


def _paired_arrays(reference_values, comparison_values):
    """
    Validate and return two paired observation vectors as float arrays.
    """
    reference_values = np.asarray(reference_values, dtype=float)
    comparison_values = np.asarray(comparison_values, dtype=float)

    if reference_values.ndim != 1 or comparison_values.ndim != 1:
        raise ValueError("Paired values must be one-dimensional.")

    if reference_values.shape != comparison_values.shape:
        raise ValueError(
            "Paired conditions must contain the same number of observations "
            "aligned by seed. "
            f"Received {reference_values.size} and {comparison_values.size}."
        )

    if reference_values.size == 0:
        raise ValueError("At least one paired observation is required.")

    if not (
        np.all(np.isfinite(reference_values))
        and np.all(np.isfinite(comparison_values))
    ):
        raise ValueError("Paired observations must be finite.")

    return reference_values, comparison_values


def align_paired_seed_values(left_seed_values, right_seed_values):
    """
    Align two seed-indexed conditions into paired arrays.

    Both inputs must be seed-indexed mappings covering exactly the same seed
    set, with unique integer seeds and finite numeric values. Positional
    truncation and silent intersection are refused so that a paired comparison
    always speaks about the same training seeds. Seeds need not start at zero
    or be contiguous; they are only required to match between the two
    conditions.

    Returns ``(seeds, left_array, right_array)`` ordered by ascending seed.
    """
    if not isinstance(left_seed_values, dict) or not isinstance(
        right_seed_values, dict
    ):
        raise ValueError(
            "Paired conditions must be provided as seed-indexed mappings."
        )

    for side, mapping in (
        ("left", left_seed_values),
        ("right", right_seed_values),
    ):
        for key in mapping:
            if not _is_integer_seed(key):
                raise ValueError(
                    f"Paired {side} condition contains a non-integer "
                    f"seed identifier: {key!r}."
                )

    left_seeds = {int(seed) for seed in left_seed_values}
    right_seeds = {int(seed) for seed in right_seed_values}

    if left_seeds != right_seeds:
        missing_from_right = sorted(left_seeds - right_seeds)
        missing_from_left = sorted(right_seeds - left_seeds)
        raise ValueError(
            "Paired conditions must contain identical seed sets. "
            f"Missing from right: {missing_from_right}; "
            f"missing from left: {missing_from_left}."
        )

    if not left_seeds:
        raise ValueError("Paired conditions must contain at least one seed.")

    seeds = sorted(left_seeds)
    left_array = np.asarray(
        [left_seed_values[seed] for seed in seeds],
        dtype=float,
    )
    right_array = np.asarray(
        [right_seed_values[seed] for seed in seeds],
        dtype=float,
    )

    if not (np.all(np.isfinite(left_array)) and np.all(np.isfinite(right_array))):
        raise ValueError("Paired observations must be finite.")

    return seeds, left_array, right_array


def _is_integer_seed(value):
    """
    Return whether a mapping key is usable as an integer seed identifier.
    """
    if isinstance(value, bool):
        return False

    return isinstance(value, (int, np.integer))


def difference_profile(reference_values, comparison_values):
    """
    Describe the paired differences ``comparison - reference``.

    Constancy is decided with a tolerance scaled to the magnitude of the
    differences, because differences that are mathematically identical can
    differ in their final representable digits after arithmetic. The tolerance
    only classifies the differences; it is never used as a denominator.

    Returns a mapping with the differences, the number of pairs, the mean
    difference, the sample standard deviation (exactly zero when the
    differences are constant), and flags for the constant and all-zero cases.
    """
    reference_values, comparison_values = _paired_arrays(
        reference_values,
        comparison_values,
    )

    differences = comparison_values - reference_values
    number_of_pairs = int(differences.size)
    mean_difference = float(np.mean(differences))

    difference_scale = max(1.0, float(np.max(np.abs(differences))))
    constant_tolerance = np.finfo(float).eps * difference_scale * 32.0

    maximum_difference_deviation = float(
        np.max(np.abs(differences - differences[0]))
    )
    constant_differences = maximum_difference_deviation <= constant_tolerance
    all_zero_differences = bool(
        np.max(np.abs(differences)) <= constant_tolerance
    )

    if constant_differences or number_of_pairs < 2:
        standard_deviation = 0.0
    else:
        standard_deviation = float(np.std(differences, ddof=1))

    return {
        "differences": differences,
        "n_pairs": number_of_pairs,
        "mean_difference": mean_difference,
        "standard_deviation": standard_deviation,
        "constant_differences": bool(constant_differences),
        "all_zero_differences": all_zero_differences,
        "constant_tolerance": float(constant_tolerance),
    }


def cohens_dz(reference_values, comparison_values):
    """
    Return Cohen's dz for the paired differences ``comparison - reference``.

    ``dz = mean(d) / sample_std(d)`` using the sample standard deviation
    (``ddof=1``). Two degenerate cases are given explicit, deterministic
    values rather than a division by zero:

    * all differences exactly zero: there is no effect, and dz is ``0.0``.
    * constant non-zero differences: the differences have no spread, so the
      standardized effect is unbounded and dz is ``+inf`` or ``-inf``
      following the sign of the mean difference.

    No epsilon is added to the denominator, so a finite dz always reflects
    real observed spread.
    """
    profile = difference_profile(reference_values, comparison_values)

    return _dz_from_profile(profile)


def _dz_from_profile(profile):
    """
    Derive Cohen's dz from a computed difference profile.
    """
    if profile["all_zero_differences"]:
        return 0.0

    if profile["standard_deviation"] == 0.0:
        return math.copysign(math.inf, profile["mean_difference"])

    return float(profile["mean_difference"] / profile["standard_deviation"])


def paired_t_test(reference_values, comparison_values):
    """
    Run a paired t-test for ``comparison - reference``.

    Uses ``scipy.stats.ttest_rel``. The two conditions share seeds, so the
    observations are paired and an independent-samples test would be wrong.

    The degenerate cases are resolved before calling SciPy, because floating
    point noise on differences with no real spread otherwise produces an
    arbitrarily large statistic and an arbitrarily small p-value:

    * all differences exactly zero: ``statistic = 0.0``, ``raw_p = 1.0``.
    * constant non-zero differences: the separation is infinite, so
      ``statistic`` is ``+inf`` or ``-inf`` by the sign of the mean
      difference and ``raw_p = 0.0``.

    Fewer than two pairs cannot support a test, and ``statistic`` and
    ``raw_p`` are ``None``.
    """
    profile = difference_profile(reference_values, comparison_values)

    result = {
        "test": PAIRED_T_TEST,
        "statistic": None,
        "raw_p": None,
        "n_pairs": profile["n_pairs"],
        "effect_size_dz": None,
    }

    if profile["n_pairs"] < 2:
        return result

    result["effect_size_dz"] = _dz_from_profile(profile)

    if profile["all_zero_differences"]:
        result["statistic"] = 0.0
        result["raw_p"] = 1.0
        return result

    if profile["standard_deviation"] == 0.0:
        result["statistic"] = math.copysign(
            math.inf,
            profile["mean_difference"],
        )
        result["raw_p"] = 0.0
        return result

    test_result = stats.ttest_rel(
        np.asarray(comparison_values, dtype=float),
        np.asarray(reference_values, dtype=float),
    )

    result["statistic"] = float(test_result.statistic)
    result["raw_p"] = float(test_result.pvalue)

    return result


def wilcoxon_signed_rank(reference_values, comparison_values):
    """
    Run a Wilcoxon signed-rank test for ``comparison - reference``.

    Parameters are pinned explicitly rather than left to release-dependent
    defaults: zero differences are handled with the ``wilcox`` convention
    (discarded before ranking), no continuity correction is applied, the test
    is two-sided, and the exact or normal approximation is selected by SciPy's
    ``auto`` rule. The older positional signature is used as a fallback so the
    analysis behaves identically across the supported SciPy releases.

    When every difference is exactly zero the ranking is empty and SciPy has
    no signed ranks to work with. That case is resolved before calling SciPy
    as ``statistic = 0.0`` and ``raw_p = 1.0``: identical conditions are
    reported as no evidence of a difference, never as evidence of one.

    Fewer than two pairs cannot support a test, and ``statistic`` and
    ``raw_p`` are ``None``.
    """
    profile = difference_profile(reference_values, comparison_values)

    result = {
        "test": WILCOXON_TEST,
        "statistic": None,
        "raw_p": None,
        "n_pairs": profile["n_pairs"],
    }

    if profile["n_pairs"] < 2:
        return result

    if profile["all_zero_differences"]:
        result["statistic"] = 0.0
        result["raw_p"] = 1.0
        return result

    test_result = _call_scipy_wilcoxon(
        np.asarray(comparison_values, dtype=float),
        np.asarray(reference_values, dtype=float),
    )

    result["statistic"] = float(test_result.statistic)
    result["raw_p"] = float(test_result.pvalue)

    return result


def _call_scipy_wilcoxon(comparison_values, reference_values):
    """
    Call SciPy's Wilcoxon test with pinned parameters.
    """
    try:
        return stats.wilcoxon(
            comparison_values,
            reference_values,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
    except TypeError:
        return stats.wilcoxon(
            comparison_values,
            reference_values,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
        )


def holm_adjust(p_values):
    """
    Apply the Holm step-down family-wise error correction.

    ``None`` entries are preserved and excluded from the correction family, so
    a hypothesis that could not be tested does not inflate the family size.

    Adjusted values are produced by sorting the family ascending, scaling each
    p-value by the number of remaining hypotheses, enforcing monotonicity with
    a running maximum, and clipping to one. The result depends only on the
    multiset of p-values, never on the order in which they were supplied.
    """
    adjusted_values = [None for _ in p_values]

    valid_values = []

    for index, p_value in enumerate(p_values):
        if p_value is None:
            continue

        p_value = float(p_value)

        if not math.isfinite(p_value) or p_value < 0.0 or p_value > 1.0:
            raise ValueError(f"Invalid p-value: {p_value!r}.")

        valid_values.append((index, p_value))

    valid_values.sort(key=lambda item: item[1])

    family_size = len(valid_values)
    running_maximum = 0.0

    for rank, (original_index, p_value) in enumerate(valid_values, start=1):
        raw_adjusted = (family_size - rank + 1) * p_value
        running_maximum = max(running_maximum, raw_adjusted)
        adjusted_values[original_index] = min(1.0, running_maximum)

    return adjusted_values


def holm_correct_family(records, raw_p_key="raw_p", adjusted_p_key="adjusted_p",
                        reject_key="reject", alpha=DEFAULT_ALPHA):
    """
    Correct one prespecified family of hypotheses in place.

    Every record in ``records`` must belong to the same family and be measured
    with the same test. The raw p-value is read from ``raw_p_key`` and left
    unchanged; the adjusted p-value and the rejection decision are written to
    ``adjusted_p_key`` and ``reject_key``.

    A hypothesis is rejected when its adjusted p-value is strictly below
    ``alpha``. Records whose raw p-value is ``None`` receive ``None`` for both
    the adjusted value and the decision.
    """
    alpha = float(alpha)

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one.")

    records = list(records)

    adjusted_values = holm_adjust(
        [record[raw_p_key] for record in records]
    )

    for record, adjusted_value in zip(records, adjusted_values):
        record[adjusted_p_key] = adjusted_value
        record[reject_key] = (
            None if adjusted_value is None else bool(adjusted_value < alpha)
        )

    return records


def significance_marker(p_value):
    """
    Render a p-value as a conventional significance marker.

    Callers that correct for multiple comparisons must pass the adjusted
    p-value, so the marker reflects the same evidence as the reported
    decision.
    """
    if p_value is None:
        return "n/a"

    if p_value < 0.001:
        return "***"

    if p_value < 0.01:
        return "**"

    if p_value < 0.05:
        return "*"

    return "n.s."
