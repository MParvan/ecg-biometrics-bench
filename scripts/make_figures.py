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
    discover_configurations,
    filter_configurations,
    read_latest_record,
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


def collect_series(entries, metric):
    """
    Collect per-seed metric values for every (protocol, setting) combination.

    Returns a mapping of setting to an ordered mapping of protocol to the list
    of per-seed values, plus the dataset the values belong to.
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

    for entry in entries:
        if entry.task_type != task_type:
            continue

        if entry.setting not in series:
            continue

        log_path = entry.structured_log_path()
        record = (
            read_latest_record(log_path) if log_path else None
        )

        if not record:
            missing.append(entry.path.name)
            continue

        values = [
            run_record.get("metrics", {}).get(metric)
            for run_record in record.get("per_run_results", [])
        ]
        values = [
            float(value)
            for value in values
            if isinstance(value, (int, float))
        ]

        if not values:
            missing.append(entry.path.name)
            continue

        series[entry.setting][entry.protocol] = values

    return series, missing


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
    """
    left_values = np.asarray(left_values, dtype=float)
    right_values = np.asarray(right_values, dtype=float)

    paired_count = min(left_values.size, right_values.size)

    if paired_count < 2:
        return None

    left_values = left_values[:paired_count]
    right_values = right_values[:paired_count]

    differences = right_values - left_values

    result = {
        "n_pairs": int(paired_count),
        "mean_difference": float(np.mean(differences)),
        "test": "paired t-test",
    }

    if np.allclose(differences, 0.0):
        # A constant difference of zero has no test statistic.
        result.update(
            {
                "t_statistic": 0.0,
                "p_value": 1.0,
                "cohens_dz": 0.0,
                "wilcoxon_p_value": 1.0,
            }
        )

        return result

    t_statistic, p_value = stats.ttest_rel(
        right_values,
        left_values,
    )

    result["t_statistic"] = float(t_statistic)
    result["p_value"] = float(p_value)
    result["cohens_dz"] = float(
        np.mean(differences) / np.std(differences, ddof=1)
    )

    try:
        _, wilcoxon_p = stats.wilcoxon(
            right_values,
            left_values,
        )
        result["wilcoxon_p_value"] = float(wilcoxon_p)
    except ValueError:
        result["wilcoxon_p_value"] = None

    return result


def significance_marker(p_value):
    """
    Render a p-value as a conventional significance marker.
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
        for value in protocol_values.get(protocol, [])
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
            values = protocol_values.get(protocol)

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
            values = protocol_values.get(protocol)

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

    for axis, setting in zip(axes[0], settings):
        left_values = series[setting][left_protocol]
        right_values = series[setting][right_protocol]

        paired_count = min(
            len(left_values),
            len(right_values),
        )
        left_values = left_values[:paired_count]
        right_values = right_values[:paired_count]

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

            span = max(
                max(left_values + right_values)
                - min(left_values + right_values),
                1e-9,
            )
            bracket_y = (
                max(max(left_values), max(right_values))
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
            axis.text(
                0.5,
                bracket_y + tick * 0.5,
                f"{significance_marker(test_result['p_value'])}  "
                f"$p$ = {test_result['p_value']:.3g}\n"
                f"paired $t$-test, $n$ = {test_result['n_pairs']} seeds, "
                f"$d_z$ = {test_result['cohens_dz']:.2f}",
                ha="center",
                va="bottom",
                fontsize=plt.rcParams["font.size"] - 2,
                color=INK_SECONDARY,
                zorder=5,
            )
            axis.margins(y=0.28)

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
        "cohens_dz",
        "wilcoxon_p_value",
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

    series, missing = collect_series(
        entries,
        arguments.metric,
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
        print(
            f"[WARN] {len(missing)} configuration(s) have no results yet "
            "and are omitted from the figure."
        )

    protocols = order_protocols(
        arguments.dataset,
        series,
    )

    output_dir = Path(arguments.output_dir)
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
