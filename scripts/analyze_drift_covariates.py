"""
Relate per-subject cross-session degradation to the covariates that are
actually measurable in public ECG biometric datasets.

Scope and honest limits
-----------------------
Temporal drift in ECG biometrics is usually attributed to some mixture of
electrode shift, heart-rate change, myocardial variation, and elapsed time.
This script quantifies as much of that as public data permit, and is explicit
about the part that cannot be determined.

What is measurable, per subject, from the datasets this framework integrates:

- **Elapsed time** between enrollment and probe. Available everywhere: real
  acquisition dates for ECG-ID, PTB, PTB-XL and CYBHi, and exact minute
  offsets for the continuous MIT-BIH and NSRDB recordings.
- **Heart-rate change** between enrollment and probe, computed from RR
  intervals of the detected R-peaks. Available everywhere; needs no metadata.
- **Morphology amplitude ratio** between the enrollment and probe templates.
  Available everywhere.
- **Health status**, as a subject-level label. Available only for PTB
  (`_is_healthy`) and PTB-XL (`scp_codes`). ECG-ID, Heartprint, CYBHi and
  NSRDB contain healthy volunteers only, so within those datasets this
  covariate has no variance and cannot explain anything.

What is **not** identifiable, and why:

- **Electrode shift** is not annotated in any of the seven datasets. The
  natural proxy, a change in signal amplitude, is driven jointly by electrode
  placement, physiological state, and recording gain, so it cannot be
  attributed to placement alone.
- **The factors are confounded by construction.** A CYBHi long-term pair
  differs in elapsed time *and* in electrode re-application *and* in
  physiological state simultaneously. Nothing is randomised and no
  intervention is available, so an observational regression estimates
  association, never a causal contribution.

This script therefore reports **associations with explicit uncertainty**, not
a causal decomposition. Treat a coefficient as evidence about what covaries
with drift, and read the collinearity diagnostics before interpreting any
single one.

Usage:

    python -m scripts.analyze_drift_covariates \
        --dataset ecgid \
        --data_split_mode leave-last-out-long-term \
        --output-json drift_ecgid.json --output-csv drift_ecgid.csv
"""

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from load_dataset import (  # noqa: E402
    DEFAULT_PREPROCESSING_CONFIG,
    RECORD_ORDER_SPLIT_MODES,
    load_cybhi_dataset,
    load_ecgid_dataset,
    load_heartprint_dataset,
    load_ptb_dataset,
    load_ptbxl_dataset,
    select_record_order_partition,
)
from preprocessing import Preprocessing  # noqa: E402

RECORD_ORDER_LOADERS = {
    "ecgid": load_ecgid_dataset,
    "ptb": load_ptb_dataset,
    "ptbxl": load_ptbxl_dataset,
}

SESSION_LOADERS = {
    "heartprint": load_heartprint_dataset,
    "cybhi": load_cybhi_dataset,
}

# Datasets in which health status varies between subjects. Elsewhere the
# cohort is healthy by construction and the covariate is constant.
DATASETS_WITH_HEALTH_VARIATION = {"ptb", "ptbxl"}

# Templates are resampled to a common length so datasets recorded at
# different sampling rates remain comparable.
_TEMPLATE_LENGTH = 256

COVARIATE_DESCRIPTIONS = OrderedDict(
    [
        (
            "elapsed_days",
            "Days between the latest enrollment recording and the earliest "
            "probe recording.",
        ),
        (
            "heart_rate_change_bpm",
            "Absolute change in mean heart rate between the enrollment and "
            "probe partitions, from RR intervals.",
        ),
        (
            "amplitude_log_ratio",
            "Log ratio of probe to enrollment mean beat amplitude. Jointly "
            "driven by electrode placement, physiology, and gain, so it "
            "cannot isolate electrode shift.",
        ),
        (
            "beat_count_ratio",
            "Log ratio of probe to enrollment beat counts. A control for "
            "how much signal each partition contributed.",
        ),
    ]
)


def _mean_heart_rate(signal, sampling_rate, preprocessor):
    """
    Return mean heart rate in beats per minute, or None if undetectable.
    """
    try:
        r_peaks = preprocessor.detect_r_peaks(
            np.asarray(signal, dtype=float),
            fs=int(sampling_rate),
        )
    except Exception:
        return None

    if r_peaks is None or len(r_peaks) < 3:
        return None

    rr_intervals = np.diff(np.asarray(r_peaks, dtype=float))
    rr_intervals = rr_intervals[rr_intervals > 0]

    if rr_intervals.size == 0:
        return None

    mean_rr_seconds = float(
        np.median(rr_intervals) / float(sampling_rate)
    )

    if mean_rr_seconds <= 0:
        return None

    heart_rate = 60.0 / mean_rr_seconds

    # Reject physiologically implausible values from failed detection.
    if not 25.0 <= heart_rate <= 220.0:
        return None

    return heart_rate


def _average_beat_template(records, preprocessor, preprocessing_config):
    """
    Build one averaged, length-normalized beat template for a partition.

    The template is model-free on purpose. It measures drift in the signal
    itself rather than drift in one architecture's embedding, so the result
    describes the phenomenon rather than a particular model's response to it.
    """
    beats = []

    for record in records:
        signal = record.get("signal")
        sampling_rate = record.get("fs")

        if signal is None or sampling_rate is None:
            continue

        signal = np.asarray(signal, dtype=float).squeeze()

        if signal.ndim > 1:
            signal = signal[:, 0]

        try:
            segmented = preprocessor.preprocess_ecg(
                ecg=signal,
                fs=int(sampling_rate),
                mode=preprocessing_config["mode"],
                pre_s=preprocessing_config["pre_s"],
                post_s=preprocessing_config["post_s"],
                resample_len=_TEMPLATE_LENGTH,
                window_s=preprocessing_config["window_s"],
                stride_s=preprocessing_config["stride_s"],
                rpeak_method=preprocessing_config["rpeak_method"],
                align_peak=preprocessing_config["align_peak"],
                filter_method=preprocessing_config["filter_method"],
                filter_kwargs=dict(
                    preprocessing_config["filter_parameters"]
                ),
                norm_method=preprocessing_config[
                    "normalization_method"
                ],
            )
        except Exception:
            continue

        segmented = np.asarray(segmented, dtype=float)

        if segmented.ndim != 2 or segmented.shape[0] == 0:
            continue

        beats.append(segmented)

    if not beats:
        return None

    stacked = np.vstack(beats)

    # Median across beats resists the occasional badly segmented beat.
    return np.median(stacked, axis=0)


def _template_correlation(enrollment_template, probe_template):
    """
    Return the Pearson correlation between two beat templates.
    """
    if enrollment_template is None or probe_template is None:
        return None

    if enrollment_template.shape != probe_template.shape:
        return None

    if (
        np.std(enrollment_template) == 0
        or np.std(probe_template) == 0
    ):
        return None

    correlation = float(
        np.corrcoef(
            enrollment_template,
            probe_template,
        )[0, 1]
    )

    if not np.isfinite(correlation):
        return None

    return correlation


def _partition_statistics(records, preprocessor):
    """
    Summarize one partition: heart rate, amplitude, beat count, date span.
    """
    heart_rates = []
    amplitudes = []
    dates = []

    for record in records:
        signal = record.get("signal")
        sampling_rate = record.get("fs")

        if record.get("date") is not None:
            dates.append(record["date"])

        if signal is None or sampling_rate is None:
            continue

        signal = np.asarray(signal, dtype=float).squeeze()

        if signal.ndim > 1:
            signal = signal[:, 0]

        heart_rate = _mean_heart_rate(
            signal,
            sampling_rate,
            preprocessor,
        )

        if heart_rate is not None:
            heart_rates.append(heart_rate)

        # Robust amplitude: interquartile range resists baseline offsets.
        amplitudes.append(
            float(
                np.percentile(signal, 95)
                - np.percentile(signal, 5)
            )
        )

    return {
        "mean_heart_rate": (
            float(np.mean(heart_rates)) if heart_rates else None
        ),
        "mean_amplitude": (
            float(np.mean(amplitudes)) if amplitudes else None
        ),
        "record_count": len(records),
        "earliest_date": min(dates) if dates else None,
        "latest_date": max(dates) if dates else None,
    }


def _elapsed_days(enrollment_stats, probe_stats):
    """
    Return days between the last enrollment and the first probe recording.
    """
    latest_enrollment = enrollment_stats["latest_date"]
    earliest_probe = probe_stats["earliest_date"]

    if latest_enrollment is None or earliest_probe is None:
        return None

    try:
        delta = earliest_probe - latest_enrollment
    except TypeError:
        return None

    return float(getattr(delta, "days", 0))


def collect_subject_covariates(dataset, data_split_mode, limit_subjects=None,
                               only_healthy=False,
                               compute_morphological_drift=True,
                               preprocessing_config=None):
    """
    Build per-subject covariates for one record-order regime.

    The same selection function the pipeline uses assigns records to the
    enrollment and probe partitions, so the covariates describe exactly the
    partitions that were evaluated.
    """
    if dataset not in RECORD_ORDER_LOADERS:
        raise SystemExit(
            f"Covariate extraction currently supports "
            f"{sorted(RECORD_ORDER_LOADERS)}. Session-structured datasets "
            "do not expose per-recording timestamps through the loader, so "
            "elapsed time cannot be computed per subject."
        )

    loader_kwargs = {"data_split_mode": data_split_mode}

    if dataset in {"ptb", "ptbxl"}:
        loader_kwargs["only_healthy"] = only_healthy

    if dataset == "ptbxl" and limit_subjects is not None:
        loader_kwargs["limit_records"] = limit_subjects

    loader = RECORD_ORDER_LOADERS[dataset](**loader_kwargs)
    preprocessor = Preprocessing()

    if preprocessing_config is None:
        preprocessing_config = getattr(
            loader,
            "prep_params",
            DEFAULT_PREPROCESSING_CONFIG,
        )

    records_by_subject = loader.load_raw_data()

    rows = []

    for subject_id in sorted(records_by_subject):
        records = list(records_by_subject[subject_id])

        (
            enrollment_records,
            enrollment_ok,
        ) = select_record_order_partition(
            records,
            data_split_mode,
            is_enrollment=True,
        )
        (
            probe_records,
            probe_ok,
        ) = select_record_order_partition(
            records,
            data_split_mode,
            is_enrollment=False,
        )

        if not (enrollment_ok and probe_ok):
            continue

        enrollment_stats = _partition_statistics(
            enrollment_records,
            preprocessor,
        )
        probe_stats = _partition_statistics(
            probe_records,
            preprocessor,
        )

        heart_rate_change = None

        if (
            enrollment_stats["mean_heart_rate"] is not None
            and probe_stats["mean_heart_rate"] is not None
        ):
            heart_rate_change = abs(
                probe_stats["mean_heart_rate"]
                - enrollment_stats["mean_heart_rate"]
            )

        amplitude_log_ratio = None

        if (
            enrollment_stats["mean_amplitude"]
            and probe_stats["mean_amplitude"]
            and enrollment_stats["mean_amplitude"] > 0
            and probe_stats["mean_amplitude"] > 0
        ):
            amplitude_log_ratio = float(
                np.log(
                    probe_stats["mean_amplitude"]
                    / enrollment_stats["mean_amplitude"]
                )
            )

        morphological_drift = None

        if compute_morphological_drift:
            correlation = _template_correlation(
                _average_beat_template(
                    enrollment_records,
                    preprocessor,
                    preprocessing_config,
                ),
                _average_beat_template(
                    probe_records,
                    preprocessor,
                    preprocessing_config,
                ),
            )

            if correlation is not None:
                # 0 means the probe template is identical to the enrollment
                # template; larger values mean more morphological drift.
                morphological_drift = float(1.0 - correlation)

        rows.append(
            OrderedDict(
                [
                    ("subject", str(subject_id)),
                    (
                        "morphological_drift",
                        morphological_drift,
                    ),
                    (
                        "elapsed_days",
                        _elapsed_days(
                            enrollment_stats,
                            probe_stats,
                        ),
                    ),
                    (
                        "heart_rate_change_bpm",
                        heart_rate_change,
                    ),
                    (
                        "amplitude_log_ratio",
                        amplitude_log_ratio,
                    ),
                    (
                        "beat_count_ratio",
                        float(
                            np.log(
                                max(probe_stats["record_count"], 1)
                                / max(
                                    enrollment_stats["record_count"],
                                    1,
                                )
                            )
                        ),
                    ),
                    (
                        "enrollment_heart_rate_bpm",
                        enrollment_stats["mean_heart_rate"],
                    ),
                    (
                        "probe_heart_rate_bpm",
                        probe_stats["mean_heart_rate"],
                    ),
                    (
                        "enrollment_records",
                        enrollment_stats["record_count"],
                    ),
                    ("probe_records", probe_stats["record_count"]),
                ]
            )
        )

        if limit_subjects is not None and len(rows) >= limit_subjects:
            break

    return rows


def load_subject_scores(path):
    """
    Load per-subject degradation scores from a two-column CSV.

    Expected columns: ``subject`` and one numeric outcome column, typically a
    per-subject EER or the drop in mean genuine similarity.
    """
    path = Path(path)

    if not path.exists():
        raise SystemExit(
            f"Subject score file not found: {path}"
        )

    scores = {}
    outcome_name = None

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None or "subject" not in reader.fieldnames:
            raise SystemExit(
                "The score file must have a 'subject' column and one "
                "numeric outcome column."
            )

        outcome_columns = [
            name
            for name in reader.fieldnames
            if name != "subject"
        ]

        if not outcome_columns:
            raise SystemExit(
                "The score file must contain an outcome column."
            )

        outcome_name = outcome_columns[0]

        for row in reader:
            try:
                scores[str(row["subject"])] = float(
                    row[outcome_name]
                )
            except (TypeError, ValueError):
                continue

    return scores, outcome_name


def _variance_inflation_factors(design_matrix, covariate_names):
    """
    Return the variance inflation factor of each covariate.

    A high value means a covariate is largely predictable from the others, so
    its individual coefficient cannot be interpreted on its own. This is the
    quantitative form of the confounding warning in the module docstring.
    """
    factors = {}

    for index, name in enumerate(covariate_names):
        target = design_matrix[:, index]
        others = np.delete(design_matrix, index, axis=1)

        if others.shape[1] == 0:
            factors[name] = 1.0
            continue

        others = np.column_stack(
            [np.ones(others.shape[0]), others]
        )

        try:
            coefficients, *_ = np.linalg.lstsq(
                others,
                target,
                rcond=None,
            )
        except np.linalg.LinAlgError:
            factors[name] = float("inf")
            continue

        predicted = others @ coefficients
        residual_ss = float(
            np.sum((target - predicted) ** 2)
        )
        total_ss = float(
            np.sum((target - np.mean(target)) ** 2)
        )

        if total_ss <= 0:
            factors[name] = float("inf")
            continue

        r_squared = 1.0 - residual_ss / total_ss

        factors[name] = (
            float("inf")
            if r_squared >= 1.0
            else float(1.0 / (1.0 - r_squared))
        )

    return factors


def analyze(rows, scores, outcome_name):
    """
    Relate the outcome to each covariate, univariately and jointly.
    """
    covariate_names = list(COVARIATE_DESCRIPTIONS)

    paired = []

    for row in rows:
        subject = row["subject"]

        if subject not in scores:
            continue

        values = [row.get(name) for name in covariate_names]

        if any(
            value is None or not np.isfinite(value)
            for value in values
        ):
            continue

        paired.append((subject, values, scores[subject]))

    report = OrderedDict(
        [
            ("outcome", outcome_name),
            ("subjects_with_covariates", len(rows)),
            ("subjects_with_scores", len(scores)),
            ("subjects_analysed", len(paired)),
            (
                "covariate_descriptions",
                dict(COVARIATE_DESCRIPTIONS),
            ),
            (
                "unidentifiable_factors",
                [
                    "Electrode shift is not annotated in any integrated "
                    "dataset. amplitude_log_ratio responds to it but also "
                    "to physiological state and recording gain, so it "
                    "cannot isolate placement.",
                    "Elapsed time, electrode re-application, and "
                    "physiological state change together between sessions. "
                    "No integrated dataset varies them independently, so "
                    "these associations are not causal contributions.",
                ],
            ),
        ]
    )

    if len(paired) < 4:
        report["status"] = (
            "Too few subjects have both covariates and an outcome score for "
            "any association to be estimated."
        )
        report["univariate"] = {}
        report["joint_model"] = None

        return report

    design_matrix = np.asarray(
        [values for _, values, _ in paired],
        dtype=float,
    )
    outcomes = np.asarray(
        [score for _, _, score in paired],
        dtype=float,
    )

    univariate = OrderedDict()

    for index, name in enumerate(covariate_names):
        column = design_matrix[:, index]

        if np.allclose(column, column[0]):
            univariate[name] = {
                "status": (
                    "constant across subjects in this dataset, so it "
                    "cannot explain any variation"
                ),
                "spearman_rho": None,
                "p_value": None,
            }
            continue

        rho, p_value = stats.spearmanr(column, outcomes)
        pearson_r, pearson_p = stats.pearsonr(column, outcomes)

        univariate[name] = {
            "spearman_rho": float(rho),
            "p_value": float(p_value),
            "pearson_r": float(pearson_r),
            "pearson_p_value": float(pearson_p),
            "n": int(len(outcomes)),
        }

    report["univariate"] = univariate

    # Joint least-squares fit on standardized covariates, so coefficients are
    # comparable in magnitude.
    usable_indices = [
        index
        for index in range(design_matrix.shape[1])
        if not np.allclose(
            design_matrix[:, index],
            design_matrix[0, index],
        )
    ]

    if not usable_indices or len(paired) <= len(usable_indices) + 1:
        report["joint_model"] = {
            "status": (
                "Not enough subjects relative to covariates for a joint "
                "fit. Report the univariate associations only."
            )
        }

        return report

    usable_names = [
        covariate_names[index] for index in usable_indices
    ]
    usable_matrix = design_matrix[:, usable_indices]

    standardized = (
        usable_matrix - usable_matrix.mean(axis=0)
    ) / usable_matrix.std(axis=0)

    design = np.column_stack(
        [np.ones(standardized.shape[0]), standardized]
    )

    coefficients, *_ = np.linalg.lstsq(
        design,
        outcomes,
        rcond=None,
    )

    predicted = design @ coefficients
    residuals = outcomes - predicted
    residual_ss = float(np.sum(residuals ** 2))
    total_ss = float(
        np.sum((outcomes - outcomes.mean()) ** 2)
    )

    degrees_of_freedom = len(outcomes) - design.shape[1]

    r_squared = (
        1.0 - residual_ss / total_ss if total_ss > 0 else None
    )

    adjusted_r_squared = None

    if r_squared is not None and degrees_of_freedom > 0:
        adjusted_r_squared = float(
            1.0
            - (1.0 - r_squared)
            * (len(outcomes) - 1)
            / degrees_of_freedom
        )

    report["joint_model"] = {
        "covariates": usable_names,
        "standardized_coefficients": {
            name: float(coefficients[index + 1])
            for index, name in enumerate(usable_names)
        },
        "intercept": float(coefficients[0]),
        "r_squared": (
            float(r_squared) if r_squared is not None else None
        ),
        "adjusted_r_squared": adjusted_r_squared,
        "n": int(len(outcomes)),
        "degrees_of_freedom": int(degrees_of_freedom),
        "variance_inflation_factors": (
            _variance_inflation_factors(
                standardized,
                usable_names,
            )
        ),
        "note": (
            "Coefficients are standardized, so they are comparable in "
            "magnitude but are associations rather than causal effects. A "
            "variance inflation factor above roughly 5 means the covariate "
            "is largely explained by the others and its individual "
            "coefficient should not be read on its own."
        ),
    }

    return report


def render_summary(report, dataset, data_split_mode):
    """
    Render a terminal summary that leads with the limitations.
    """
    lines = [
        "=" * 72,
        f"Drift covariate analysis: {dataset} / {data_split_mode}",
        "=" * 72,
        f"Subjects with covariates : {report['subjects_with_covariates']}",
        f"Subjects with outcomes   : {report['subjects_with_scores']}",
        f"Subjects analysed        : {report['subjects_analysed']}",
        "",
        "Not identifiable from this data:",
    ]

    for limitation in report["unidentifiable_factors"]:
        lines.append(f"  - {limitation}")

    lines.append("")

    if not report.get("univariate"):
        lines.append(report.get("status", "No analysis performed."))
        lines.append("")

        return "\n".join(lines)

    lines.append("Univariate association with the outcome:")
    lines.append(
        f"  {'covariate':<26} {'spearman':>9} {'p':>10}"
    )

    for name, result in report["univariate"].items():
        if result.get("spearman_rho") is None:
            lines.append(
                f"  {name:<26} {'-':>9} {'-':>10}   "
                f"({result.get('status', 'unavailable')})"
            )
            continue

        lines.append(
            f"  {name:<26} {result['spearman_rho']:>9.3f} "
            f"{result['p_value']:>10.4g}"
        )

    joint = report.get("joint_model")

    if joint and "standardized_coefficients" in joint:
        lines.extend(
            [
                "",
                "Joint model (standardized coefficients):",
                f"  {'covariate':<26} {'coef':>9} {'VIF':>8}",
            ]
        )

        for name in joint["covariates"]:
            coefficient = joint["standardized_coefficients"][name]
            vif = joint["variance_inflation_factors"][name]
            lines.append(
                f"  {name:<26} {coefficient:>9.4f} {vif:>8.2f}"
            )

        lines.append(
            f"  R-squared = {joint['r_squared']:.4f}, "
            f"adjusted = {joint['adjusted_r_squared']:.4f}, "
            f"n = {joint['n']}"
        )
    elif joint:
        lines.extend(["", joint.get("status", "")])

    lines.append("")

    return "\n".join(lines)


def write_covariate_csv(rows, output_path):
    """
    Write the per-subject covariate table.
    """
    if not rows:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Relate per-subject cross-session degradation to measurable "
            "covariates, and state what cannot be identified."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(RECORD_ORDER_LOADERS),
        help=(
            "Dataset to analyse. Only record-order datasets expose "
            "per-recording timestamps."
        ),
    )
    parser.add_argument(
        "--data_split_mode",
        type=str,
        default="leave-last-out-long-term",
        choices=list(RECORD_ORDER_SPLIT_MODES),
        help="Cross-session regime to analyse.",
    )
    parser.add_argument(
        "--subject-scores",
        type=str,
        default=None,
        help=(
            "CSV with a 'subject' column and one numeric outcome column, "
            "typically per-subject EER from a trained model. Overrides the "
            "default model-free outcome."
        ),
    )
    parser.add_argument(
        "--outcome",
        type=str,
        default="morphological_drift",
        choices=["morphological_drift", "none"],
        help=(
            "Outcome to explain when --subject-scores is not supplied. "
            "'morphological_drift' is one minus the correlation between a "
            "subject's enrollment and probe beat templates: model-free, so "
            "it characterises drift in the signal rather than in one "
            "architecture's embedding. 'none' emits the covariate table "
            "only."
        ),
    )
    parser.add_argument(
        "--only_healthy",
        action="store_true",
        help="Restrict PTB or PTB-XL to healthy control subjects.",
    )
    parser.add_argument(
        "--limit-subjects",
        type=int,
        default=None,
        help="Analyse only the first N eligible subjects.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write the analysis report to this JSON file.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Write the per-subject covariate table to this CSV file.",
    )

    return parser


def main(argv=None):
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if (
        arguments.dataset not in DATASETS_WITH_HEALTH_VARIATION
        and not arguments.only_healthy
    ):
        print(
            f"[NOTE] {arguments.dataset} contains healthy subjects only, so "
            "health status has no within-dataset variance and is excluded "
            "from the covariate set."
        )

    use_model_free_outcome = (
        arguments.subject_scores is None
        and arguments.outcome == "morphological_drift"
    )

    rows = collect_subject_covariates(
        arguments.dataset,
        arguments.data_split_mode,
        limit_subjects=arguments.limit_subjects,
        only_healthy=arguments.only_healthy,
        compute_morphological_drift=use_model_free_outcome,
    )

    if not rows:
        raise SystemExit(
            "No subject satisfied the structure this regime requires."
        )

    print(
        f"[INFO] Extracted covariates for {len(rows)} subject(s)."
    )

    if arguments.output_csv:
        path = write_covariate_csv(rows, arguments.output_csv)
        print(f"[INFO] Covariate table written to {path}")

    if arguments.subject_scores:
        scores, outcome_name = load_subject_scores(
            arguments.subject_scores
        )
    elif use_model_free_outcome:
        scores = {
            row["subject"]: row["morphological_drift"]
            for row in rows
            if row.get("morphological_drift") is not None
        }
        outcome_name = "morphological_drift"

        if not scores:
            raise SystemExit(
                "No subject produced a usable beat template, so the "
                "model-free outcome could not be computed. Supply "
                "--subject-scores instead."
            )

        print(
            f"[INFO] Computed model-free morphological drift for "
            f"{len(scores)} subject(s)."
        )
    else:
        print(
            "\n[INFO] Outcome set to 'none', so only the covariate table "
            "was produced."
        )

        return 0

    report = analyze(rows, scores, outcome_name)
    report["dataset"] = arguments.dataset
    report["data_split_mode"] = arguments.data_split_mode

    print()
    print(
        render_summary(
            report,
            arguments.dataset,
            arguments.data_split_mode,
        )
    )

    if arguments.output_json:
        output_path = Path(arguments.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                report,
                handle,
                indent=2,
                default=str,
            )

        print(f"[INFO] Report written to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
