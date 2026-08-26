"""
Render publication-quality comparison figures from structured experiment
records.

Figures are built from the per-seed values already stored in the result
records, so a figure can never disagree with the table it accompanies. Type
is sized for print, error bars are confidence intervals rather than standard
deviations, and paired comparisons carry an explicit significance test.

Figure types
------------
degradation
    A protocol-ordered line chart with 95% confidence bands, one series per
    evaluation setting. This is the figure that shows performance collapsing
    as the protocol becomes realistic.
comparison
    A grouped bar chart with 95% confidence intervals and direct value labels,
    for reading exact numbers per protocol.
paired
    A per-seed slope plot between two conditions with a paired significance
    test annotated on the bracket. Use it for same-session versus
    cross-session, or single-shot versus leave-last-out.

Design notes
------------
Colour encodes the evaluation setting and nothing else, drawn in fixed order
from a palette validated for deuteranopia, tritanopia, and normal-vision
separation. Hatching duplicates that encoding so the figures survive grayscale
printing, and every bar carries a direct value label so the figures remain
readable without colour at all.

Usage:

    python -m scripts.make_figures --dataset ecgid --metric EER \
        --figure degradation --output-dir figures

    python -m scripts.make_figures --dataset heartprint \
        --figure paired --metric EER \
        --left short_term --right long_term --output-dir figures
"""

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reproduce_tables import (  # noqa: E402
    DATASET_LABELS,
    PROTOCOL_LABELS,
    PROTOCOL_ORDER,
    SETTING_LABELS,
    collect_configuration_record,
    discover_configurations,
    filter_configurations,
)
from experiment_provenance import (  # noqa: E402
    ResultCollectionError,
    build_experiment_implementation_identity,
)
from scripts.statistical_utils import (  # noqa: E402
    DEFAULT_ALPHA,
    PAIRED_T_TEST,
    WILCOXON_TEST,
    align_paired_seed_values,
    holm_correct_family,
    paired_t_test,
    significance_marker,
    wilcoxon_signed_rank,
)

# Categorical slots assigned by evaluation setting, in fixed order. Validated
# with the palette checker: worst all-pairs CVD dE 9.2, normal-vision dE 24.0.
SETTING_COLORS = OrderedDict(
    [
        ("closed_set", "#2a78d6"),
        ("subject_disjoint", "#eb6834"),
    ]
)

# Secondary encoding so the figures read in grayscale and under forced colours.
SETTING_HATCHES = OrderedDict(
    [
        ("closed_set", ""),
        ("subject_disjoint", "///"),
    ]
)

SETTING_MARKERS = OrderedDict(
    [
        ("closed_set", "o"),
        ("subject_disjoint", "s"),
    ]
)

# Text stays in ink colours; the mark beside it carries identity.
INK_PRIMARY = "#1a1a1a"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8a85"
SURFACE = "#ffffff"

# Metrics where a lower value is better, so the reader is told which way is up.
LOWER_IS_BETTER = {"EER"}

METRIC_LABELS = {
    "EER": "Equal Error Rate",
    "AUC": "Area Under ROC Curve",
    "d-prime": "Decidability index $d'$",
    "TAR@0.1%FAR": "TAR @ 0.1% FAR",
    "Rank-1 Accuracy": "Rank-1 Accuracy",
    "Rank-5 Accuracy": "Rank-5 Accuracy",
}

IDENTIFICATION_METRICS = {
    "Rank-1 Accuracy",
    "Rank-5 Accuracy",
}


def apply_publication_style(base_font_size):
    """
    Apply a print-oriented style with legible type and recessive chrome.
    """
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": base_font_size,
            "axes.titlesize": base_font_size + 3,
            "axes.labelsize": base_font_size + 2,
            "xtick.labelsize": base_font_size,
            "ytick.labelsize": base_font_size,
            "legend.fontsize": base_font_size,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK_PRIMARY,
            "text.color": INK_PRIMARY,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "axes.edgecolor": INK_MUTED,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#e2e2df",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "figure.dpi": 150,
        }
    )


def _is_integer_seed(seed):
    """
    Return True only for genuine integer-like seed identifiers.

    Booleans, floats (even integer-valued ones such as ``42.0``), strings, and
    ``None`` are rejected so that seed identity used for paired statistics can
    never be silently normalized away from a malformed record.
    """
    if isinstance(seed, bool):
        return False

    return isinstance(seed, (int, np.integer))


def _extract_seed_metric(record, metric):
    """
    Return a seed-indexed mapping of one metric, or None if the record is
    unfit for figure aggregation.

    A configuration is considered unavailable and returned as ``None`` when
    any per-run result is missing the metric, carries a non-numeric or
    non-finite value, or lacks a unique integer seed. Refusing to average a
    partial series here prevents figures from silently reproducing subset
    aggregation that runtime metric handling now rejects.
    """
    if not record:
        return None

    per_run_results = record.get("per_run_results")

    if not isinstance(per_run_results, list) or not per_run_results:
        return None

    seed_values = {}

    for run_record in per_run_results:
        if not isinstance(run_record, dict):
            return None

        seed = run_record.get("seed")

        if not _is_integer_seed(seed):
            return None

        seed_int = int(seed)

        if seed_int in seed_values:
            return None

        metrics = run_record.get("metrics")

        if not isinstance(metrics, dict) or metric not in metrics:
            return None

        value = metrics[metric]

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None

        value_float = float(value)

        if not math.isfinite(value_float):
            return None

        seed_values[seed_int] = value_float

    if not seed_values:
        return None

    return seed_values


def collect_series(
    entries,
    metric,
    *,
    campaign_id=None,
    publication_mode=True,
    include_provenance=False,
):
    """
    Collect per-seed metric values for every (protocol, setting) combination.

    Returns a mapping of setting to an ordered mapping of protocol to a
    seed-indexed dictionary of per-run values, plus the list of configurations
    whose metric was unavailable for plotting.

    A configuration is retained only when the metric is present and finite for
    every one of its recorded seeds. Otherwise it is reported as missing and
    excluded, so the figure code cannot silently average or plot a partial
    series behind runtime aggregation.
    """
    task_type = (
        "identification"
        if metric in IDENTIFICATION_METRICS
        else "verification"
    )

    series = OrderedDict(
        (setting, OrderedDict())
        for setting in SETTING_COLORS
    )
    missing = []
    result_provenance = OrderedDict()
    implementation_identity = (
        build_experiment_implementation_identity()
        if publication_mode
        else None
    )

    for entry in entries:
        if entry.task_type != task_type:
            continue

        if entry.setting not in series:
            continue

        try:
            collected = collect_configuration_record(
                entry,
                campaign_id=campaign_id,
                publication_mode=publication_mode,
                implementation_identity=implementation_identity,
            )
        except (ResultCollectionError, FileNotFoundError):
            missing.append(entry.path.name)
            continue

        record = collected["record"]

        seed_values = _extract_seed_metric(record, metric)

        if seed_values is None:
            missing.append(entry.path.name)
            continue

        series[entry.setting][entry.protocol] = seed_values
        result_provenance[
            f"{entry.setting}/{entry.protocol}/{entry.task_type}"
        ] = (
            collected["locator"]
            if collected["locator"] is not None
            else {"legacy_exploratory": True}
        )

    if include_provenance:
        return series, missing, result_provenance
    return series, missing


def _values_from(seed_values):
    """
    Return ordered per-seed metric values as a list.

    Accepts either a seed-indexed mapping produced by ``collect_series`` or an
    already-flat sequence. Mapping order follows ascending seed so downstream
    means and error bars are deterministic regardless of JSONL insertion order.
    """
    if seed_values is None:
        return []

    if isinstance(seed_values, dict):
        return [seed_values[seed] for seed in sorted(seed_values)]

    return list(seed_values)


def _align_paired_seed_values(left_seed_values, right_seed_values):
    """
    Align two seed-indexed conditions into paired numpy arrays.

    Thin alias for the shared implementation so that figures and the
    statistics scripts pair runs by the same rule.
    """
    return align_paired_seed_values(
        left_seed_values,
        right_seed_values,
    )


def order_protocols(dataset, series):
    """
    Order protocols for presentation, keeping only those with results.
    """
    present = set()

    for protocol_values in series.values():
        present.update(protocol_values)

    ordered = [
        protocol
        for protocol in PROTOCOL_ORDER.get(dataset, [])
        if protocol in present
    ]

    ordered.extend(
        sorted(present - set(ordered))
    )

    return ordered


def confidence_interval(values, confidence_level=0.95):
    """
    Return the half-width of the Student-t interval for a mean.
    """
    values = np.asarray(values, dtype=float)

    if values.size < 2:
        return 0.0

    standard_error = float(
        np.std(values, ddof=1) / np.sqrt(values.size)
    )
    critical_value = float(
        stats.t.ppf(
            1.0 - (1.0 - confidence_level) / 2.0,
            df=values.size - 1,
        )
    )

    return critical_value * standard_error


def paired_significance(left_values, right_values):
    """
    Compare two conditions measured under the same seeds.

    A paired t-test is reported because the two conditions share seeds, which
    removes seed-to-seed variance from the comparison. The Wilcoxon
    signed-rank statistic accompanies it because five seeds are too few to
    verify the normality the t-test assumes.

    The tests, the effect size, and their handling of degenerate differences
    come from ``scripts.statistical_utils``, so a figure and a statistics
    table computed from the same runs cannot disagree. ``p_value`` is the raw
    paired-t p-value; the Holm-adjusted value is added later, once the family
    of comparisons drawn in the figure is known.

    Differences are taken as ``right - left``. Fewer than two pairs cannot
    support a test and return ``None``.
    """
    left_values = np.asarray(left_values, dtype=float)
    right_values = np.asarray(right_values, dtype=float)

    t_result = paired_t_test(left_values, right_values)

    if t_result["n_pairs"] < 2:
        return None

    wilcoxon_result = wilcoxon_signed_rank(left_values, right_values)

    return {
        "n_pairs": t_result["n_pairs"],
        "mean_difference": float(np.mean(right_values - left_values)),
        "test": PAIRED_T_TEST,
        "t_statistic": t_result["statistic"],
        "p_value": t_result["raw_p"],
        "cohens_dz": t_result["effect_size_dz"],
        "wilcoxon_test": WILCOXON_TEST,
        "wilcoxon_statistic": wilcoxon_result["statistic"],
        "wilcoxon_p_value": wilcoxon_result["raw_p"],
    }


def figure_family_id(dataset, metric, left_protocol, right_protocol, test):
    """
    Identify the hypothesis family a paired figure panel belongs to.

    One paired figure fixes the dataset, the metric, and the two protocols
    being contrasted, and tests that contrast once per evaluation setting.
    Those panels are the family. The identifier is built from the analysis
    itself, so the same figure always produces the same family regardless of
    how many panels happen to have results.

    The parametric and non-parametric tests are corrected separately and
    therefore carry different identifiers.
    """
    return (
        f"figure:paired|dataset={dataset}|metric={metric}"
        f"|left={left_protocol}|right={right_protocol}::{test}"
    )


def apply_figure_holm_correction(statistics, dataset, metric, left_protocol,
                                 right_protocol, alpha=DEFAULT_ALPHA):
    """
    Add Holm-adjusted p-values to every panel of one paired figure.

    The raw p-values are left untouched. ``p_value_holm`` and
    ``wilcoxon_p_value_holm`` are added alongside them, together with the
    family identifier and the rejection decision at ``alpha``.

    The two tests form two families and are corrected independently, so a
    Wilcoxon result can never change a t-test decision.
    """
    statistics = list(statistics)

    if not statistics:
        return statistics

    for test_name, raw_key, adjusted_key, family_key, reject_key in (
        (
            PAIRED_T_TEST,
            "p_value",
            "p_value_holm",
            "family_id",
            "reject",
        ),
        (
            WILCOXON_TEST,
            "wilcoxon_p_value",
            "wilcoxon_p_value_holm",
            "wilcoxon_family_id",
            "wilcoxon_reject",
        ),
    ):
        holm_correct_family(
            statistics,
            raw_p_key=raw_key,
            adjusted_p_key=adjusted_key,
            reject_key=reject_key,
            alpha=alpha,
        )

        family_id = figure_family_id(
            dataset,
            metric,
            left_protocol,
            right_protocol,
            test_name,
        )

        for row in statistics:
            row[family_key] = family_id
            row["alpha"] = float(alpha)

    return statistics


def _annotate_paired_axis(axis, test_result, left_values, right_values):
    """
    Draw the significance bracket for one paired panel.

    The marker and the quoted p-value are the Holm-adjusted paired-t values,
    so the annotation reflects the same evidence as the reported decision.
    The Wilcoxon result stays in the statistics output and is deliberately not
    drawn, because two competing markers on one bracket cannot be read
    unambiguously.
    """
    combined_values = np.concatenate((left_values, right_values))
    span = max(
        float(combined_values.max() - combined_values.min()),
        1e-9,
    )
    bracket_y = (
        float(max(left_values.max(), right_values.max()))
        + span * 0.18
    )
    tick = span * 0.05

    axis.plot(
        [0, 0, 1, 1],
        [
            bracket_y - tick,
            bracket_y,
            bracket_y,
            bracket_y - tick,
        ],
        color=INK_SECONDARY,
        linewidth=1.4,
        zorder=5,
    )

    adjusted_p = test_result.get("p_value_holm")
    adjusted_text = (
        "n/a" if adjusted_p is None else f"{adjusted_p:.3g}"
    )
    effect_size = test_result.get("cohens_dz")
    effect_text = (
        "n/a"
        if effect_size is None or not math.isfinite(effect_size)
        else f"{effect_size:.2f}"
    )

    axis.text(
        0.5,
        bracket_y + tick * 0.5,
        f"{significance_marker(adjusted_p)}  "
        f"$p_{{\\mathrm{{Holm}}}}$ = {adjusted_text}\n"
        f"paired $t$-test, $n$ = {test_result['n_pairs']} seeds, "
        f"$d_z$ = {effect_text}",
        ha="center",
        va="bottom",
        fontsize=plt.rcParams["font.size"] - 2,
        color=INK_SECONDARY,
        zorder=5,
    )
    axis.margins(y=0.28)


def _metric_label(metric):
    label = METRIC_LABELS.get(metric, metric)

    if metric in LOWER_IS_BETTER:
        return f"{label} (lower is better)"

    return label


def _label_offset(series, protocols, fraction=0.02):
    """
    Return a small vertical offset proportional to the plotted value range.
    """
    values = [
        value
        for protocol_values in series.values()
        for protocol in protocols
        for value in _values_from(protocol_values.get(protocol))
    ]

    if not values:
        return 0.0

    return float(max(values)) * fraction


def _save(figure, output_dir, stem, formats):
    """
    Write one figure in every requested format.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []

    for extension in formats:
        path = output_dir / f"{stem}.{extension}"

        figure.savefig(
            path,
            bbox_inches="tight",
            dpi=600 if extension == "png" else None,
        )
        written.append(path)

    plt.close(figure)

    return written


def plot_degradation(dataset, metric, series, protocols, output_dir,
                     formats, confidence_level=0.95):
    """
    Plot the metric across progressively more realistic protocols.
    """
    figure, axis = plt.subplots(figsize=(9.5, 5.6))

    positions = np.arange(len(protocols))

    for setting, protocol_values in series.items():
        means = []
        errors = []
        present_positions = []

        for index, protocol in enumerate(protocols):
            values = _values_from(protocol_values.get(protocol))

            if not values:
                continue

            present_positions.append(index)
            means.append(float(np.mean(values)))
            errors.append(
                confidence_interval(
                    values,
                    confidence_level,
                )
            )

        if not means:
            continue

        means = np.asarray(means)
        errors = np.asarray(errors)
        present_positions = np.asarray(present_positions)

        axis.plot(
            present_positions,
            means,
            marker=SETTING_MARKERS[setting],
            color=SETTING_COLORS[setting],
            label=SETTING_LABELS[setting],
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=3,
        )
        axis.fill_between(
            present_positions,
            means - errors,
            means + errors,
            color=SETTING_COLORS[setting],
            alpha=0.16,
            linewidth=0,
            zorder=2,
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        [
            PROTOCOL_LABELS.get(
                protocol,
                protocol.replace("_", " "),
            )
            for protocol in protocols
        ],
        rotation=22,
        ha="right",
    )
    axis.set_ylabel(_metric_label(metric))
    axis.set_title(
        f"{DATASET_LABELS.get(dataset, dataset)}: "
        f"{METRIC_LABELS.get(metric, metric)} across evaluation protocols"
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="best")

    percentage = int(round(confidence_level * 100))
    axis.text(
        0.0,
        -0.34,
        f"Bands show {percentage}% confidence intervals of the mean "
        "over independent seeds.",
        transform=axis.transAxes,
        fontsize=plt.rcParams["font.size"] - 1,
        color=INK_SECONDARY,
    )

    return _save(
        figure,
        output_dir,
        f"{dataset}_{_slug(metric)}_degradation",
        formats,
    )


def plot_comparison(dataset, metric, series, protocols, output_dir,
                    formats, confidence_level=0.95):
    """
    Plot per-protocol means as grouped bars with direct value labels.
    """
    figure, axis = plt.subplots(figsize=(10.5, 5.8))

    settings = [
        setting
        for setting in series
        if series[setting]
    ]
    group_width = 0.8
    bar_width = group_width / max(len(settings), 1)
    positions = np.arange(len(protocols))

    # Direct labels are offset above the error-bar cap so they never sit on
    # top of it.
    label_offset = _label_offset(series, protocols)

    for setting_index, setting in enumerate(settings):
        protocol_values = series[setting]

        offsets = (
            positions
            - group_width / 2
            + bar_width * (setting_index + 0.5)
        )

        for index, protocol in enumerate(protocols):
            values = _values_from(protocol_values.get(protocol))

            if not values:
                continue

            mean_value = float(np.mean(values))
            error = confidence_interval(
                values,
                confidence_level,
            )

            axis.bar(
                offsets[index],
                mean_value,
                # A 2px surface gap between adjacent fills.
                width=bar_width * 0.88,
                color=SETTING_COLORS[setting],
                hatch=SETTING_HATCHES[setting],
                edgecolor=SURFACE,
                linewidth=1.2,
                zorder=3,
                label=(
                    SETTING_LABELS[setting]
                    if index == 0
                    else None
                ),
            )
            axis.errorbar(
                offsets[index],
                mean_value,
                yerr=error,
                fmt="none",
                ecolor=INK_SECONDARY,
                elinewidth=1.4,
                capsize=4,
                zorder=4,
            )
            # Direct labels satisfy the relief rule and keep the figure
            # readable in grayscale.
            axis.text(
                offsets[index],
                mean_value + error + label_offset,
                f"{mean_value:.3f}",
                ha="center",
                va="bottom",
                fontsize=plt.rcParams["font.size"] - 2,
                color=INK_SECONDARY,
                rotation=90,
                zorder=5,
            )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        [
            PROTOCOL_LABELS.get(
                protocol,
                protocol.replace("_", " "),
            )
            for protocol in protocols
        ],
        rotation=22,
        ha="right",
    )
    axis.set_ylabel(_metric_label(metric))
    axis.set_title(
        f"{DATASET_LABELS.get(dataset, dataset)}: "
        f"{METRIC_LABELS.get(metric, metric)} by protocol and setting"
    )
    axis.margins(y=0.22)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="best")

    percentage = int(round(confidence_level * 100))
    axis.text(
        0.0,
        -0.34,
        f"Error bars show {percentage}% confidence intervals of the mean "
        "over independent seeds.",
        transform=axis.transAxes,
        fontsize=plt.rcParams["font.size"] - 1,
        color=INK_SECONDARY,
    )

    return _save(
        figure,
        output_dir,
        f"{dataset}_{_slug(metric)}_comparison",
        formats,
    )


def plot_paired(dataset, metric, series, left_protocol, right_protocol,
                output_dir, formats, share_y=True):
    """
    Plot per-seed values for two conditions with a paired significance test.

    Panels share a y-axis by default. Independent axes would let a small
    absolute change in one setting occupy the same visual height as a large
    change in another, which is exactly the comparison a reader makes across
    panels.
    """
    settings = [
        setting
        for setting in series
        if left_protocol in series[setting]
        and right_protocol in series[setting]
    ]

    if not settings:
        raise SystemExit(
            f"No setting has results for both '{left_protocol}' and "
            f"'{right_protocol}'. Run those configurations first."
        )

    figure, axes = plt.subplots(
        1,
        len(settings),
        figsize=(5.6 * len(settings), 5.8),
        squeeze=False,
        sharey=share_y,
    )

    statistics = []
    annotated_axes = []

    for axis, setting in zip(axes[0], settings):
        seeds, left_values, right_values = _align_paired_seed_values(
            series[setting][left_protocol],
            series[setting][right_protocol],
        )

        colour = SETTING_COLORS[setting]

        for left_value, right_value in zip(
            left_values,
            right_values,
        ):
            axis.plot(
                [0, 1],
                [left_value, right_value],
                color=colour,
                alpha=0.45,
                linewidth=1.6,
                marker=SETTING_MARKERS[setting],
                markersize=7,
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
                zorder=3,
            )

        left_mean = float(np.mean(left_values))
        right_mean = float(np.mean(right_values))

        axis.plot(
            [0, 1],
            [left_mean, right_mean],
            color=INK_PRIMARY,
            linewidth=2.8,
            marker="D",
            markersize=9,
            zorder=4,
            label="Mean",
        )

        test_result = paired_significance(
            left_values,
            right_values,
        )

        if test_result is not None:
            test_result.update(
                {
                    "dataset": dataset,
                    "setting": setting,
                    "metric": metric,
                    "left_protocol": left_protocol,
                    "right_protocol": right_protocol,
                    "left_mean": left_mean,
                    "right_mean": right_mean,
                }
            )
            statistics.append(test_result)
            annotated_axes.append((axis, test_result, left_values, right_values))

        axis.set_xticks([0, 1])
        axis.set_xticklabels(
            [
                PROTOCOL_LABELS.get(
                    left_protocol,
                    left_protocol.replace("_", " "),
                ),
                PROTOCOL_LABELS.get(
                    right_protocol,
                    right_protocol.replace("_", " "),
                ),
            ]
        )
        axis.set_xlim(-0.35, 1.35)
        axis.set_title(SETTING_LABELS[setting])
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    # The panels of one figure are a prespecified family: the same metric and
    # the same pair of protocols, tested once per evaluation setting. Holm is
    # applied across that family before anything is annotated, so a marker is
    # never drawn from an uncorrected p-value.
    apply_figure_holm_correction(
        statistics,
        dataset=dataset,
        metric=metric,
        left_protocol=left_protocol,
        right_protocol=right_protocol,
    )

    for axis, test_result, left_values, right_values in annotated_axes:
        _annotate_paired_axis(
            axis,
            test_result,
            left_values,
            right_values,
        )

    axes[0][0].set_ylabel(_metric_label(metric))
    axes[0][0].legend(loc="upper left")

    figure.suptitle(
        f"{DATASET_LABELS.get(dataset, dataset)}: "
        f"{METRIC_LABELS.get(metric, metric)}, paired by seed",
        fontsize=plt.rcParams["axes.titlesize"] + 1,
        fontweight="bold",
    )
    figure.tight_layout()

    written = _save(
        figure,
        output_dir,
        (
            f"{dataset}_{_slug(metric)}_paired_"
            f"{left_protocol}_vs_{right_protocol}"
        ),
        formats,
    )

    return written, statistics


def _slug(text):
    """
    Turn a metric name into a filename-safe slug.
    """
    return (
        str(text)
        .replace("@", "_at_")
        .replace("%", "pct")
        .replace(".", "p")
        .replace(" ", "_")
        .replace("-", "_")
        .lower()
    )


def write_statistics_csv(statistics, output_path):
    """
    Write the significance tests behind the annotated figures.
    """
    if not statistics:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "setting",
        "metric",
        "left_protocol",
        "right_protocol",
        "left_mean",
        "right_mean",
        "mean_difference",
        "n_pairs",
        "test",
        "t_statistic",
        "p_value",
        "p_value_holm",
        "reject",
        "family_id",
        "cohens_dz",
        "wilcoxon_test",
        "wilcoxon_statistic",
        "wilcoxon_p_value",
        "wilcoxon_p_value_holm",
        "wilcoxon_reject",
        "wilcoxon_family_id",
        "alpha",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in statistics:
            writer.writerow(row)

    return output_path


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Render manuscript comparison figures with significance "
            "annotation from structured experiment records."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_LABELS),
        help="Dataset to plot.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="EER",
        choices=list(METRIC_LABELS),
        help="Metric to plot (default: EER).",
    )
    parser.add_argument(
        "--figure",
        type=str,
        default="degradation",
        choices=["degradation", "comparison", "paired"],
        help="Figure type (default: degradation).",
    )
    parser.add_argument(
        "--left",
        type=str,
        default=None,
        help="Left protocol for the paired figure.",
    )
    parser.add_argument(
        "--right",
        type=str,
        default=None,
        help="Right protocol for the paired figure.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory for the rendered figures.",
    )
    parser.add_argument(
        "--format",
        type=str,
        action="append",
        default=None,
        choices=["pdf", "svg", "png"],
        help=(
            "Output format, repeatable. Defaults to pdf and png so the "
            "figure is both vector and directly viewable."
        ),
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=13.0,
        help=(
            "Base font size in points (default: 13). Axis labels and "
            "titles scale from it."
        ),
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for the intervals (default: 0.95).",
    )
    parser.add_argument(
        "--independent-y",
        action="store_true",
        help=(
            "Give each paired-figure panel its own y-axis. Panels share "
            "one axis by default so effect sizes stay comparable across "
            "settings."
        ),
    )
    parser.add_argument(
        "--config-root",
        type=str,
        default=str(
            PROJECT_ROOT / "configs" / "paper_reproduction"
        ),
        help="Directory containing the reproduction configurations.",
    )
    parser.add_argument(
        "--campaign-id",
        type=str,
        default=None,
        help="Require results from this campaign identifier.",
    )
    parser.add_argument(
        "--allow-exploratory-results",
        action="store_true",
        help=(
            "Explicitly permit legacy latest-record inputs. Generated "
            "figures are then not publication-provenance verified."
        ),
    )

    return parser


def main(argv=None):
    parser = build_parser()
    arguments = parser.parse_args(argv)

    formats = arguments.format or ["pdf", "png"]

    if not 0.0 < arguments.confidence_level < 1.0:
        parser.error(
            "'--confidence-level' must satisfy 0 < level < 1."
        )

    apply_publication_style(arguments.font_size)

    entries = filter_configurations(
        discover_configurations(
            Path(arguments.config_root)
        ),
        datasets=[arguments.dataset],
    )

    series, missing, result_provenance = collect_series(
        entries,
        arguments.metric,
        campaign_id=arguments.campaign_id,
        publication_mode=not arguments.allow_exploratory_results,
        include_provenance=True,
    )

    if not any(series.values()):
        raise SystemExit(
            f"No results found for {arguments.dataset} / "
            f"{arguments.metric}. Run the corresponding "
            "configurations first, for example:\n"
            f"  python -m scripts.reproduce_tables "
            f"--dataset {arguments.dataset} --run"
        )

    if missing:
        if not arguments.allow_exploratory_results:
            raise SystemExit(
                f"Refusing to render a publication figure: "
                f"{len(missing)} intended configuration(s) do not have "
                "an exact, provenance-verified result record."
            )
        print(
            f"[WARN] {len(missing)} configuration(s) have no results yet "
            "and are omitted from the figure."
        )

    protocols = order_protocols(
        arguments.dataset,
        series,
    )

    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_metric_name = "".join(
        character if character.isalnum() else "_"
        for character in arguments.metric
    ).strip("_")
    provenance_path = output_dir / (
        f"{arguments.dataset}_{safe_metric_name}_"
        "result_provenance.json"
    )
    provenance_path.write_text(
        json.dumps(
            {
                "campaign_id": arguments.campaign_id,
                "publication_provenance_verified": (
                    not arguments.allow_exploratory_results
                ),
                "records": result_provenance,
            },
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    statistics = []

    if arguments.figure == "degradation":
        written = plot_degradation(
            arguments.dataset,
            arguments.metric,
            series,
            protocols,
            output_dir,
            formats,
            arguments.confidence_level,
        )
    elif arguments.figure == "comparison":
        written = plot_comparison(
            arguments.dataset,
            arguments.metric,
            series,
            protocols,
            output_dir,
            formats,
            arguments.confidence_level,
        )
    else:
        if not (arguments.left and arguments.right):
            parser.error(
                "The paired figure requires --left and --right. "
                f"Available protocols: {protocols}"
            )

        written, statistics = plot_paired(
            arguments.dataset,
            arguments.metric,
            series,
            arguments.left,
            arguments.right,
            output_dir,
            formats,
            share_y=not arguments.independent_y,
        )

    for path in written:
        print(f"[INFO] Figure written to {path}")

    statistics_path = write_statistics_csv(
        statistics,
        output_dir / "figure_significance_tests.csv",
    )

    if statistics_path is not None:
        print(
            f"[INFO] Significance tests written to {statistics_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
