"""
Paired statistical comparisons for structured ECG benchmark results.

The module reads experiment records containing ``per_run_results`` and
aligns configurations by seed. Statistical differences are always defined as:

    comparison - reference

Example manifest:

reference:
  path: results/baseline.jsonl
  scientific_configuration_sha256: <configuration digest>
  implementation_source_sha256: <implementation digest>
  label: baseline

comparisons:
  - path: results/augmentation.jsonl
    scientific_configuration_sha256: <configuration digest>
    implementation_source_sha256: <implementation digest>
    label: gaussian_augmentation

metrics:
  - EER
  - AUC
  - d-prime
  - TAR@0.1%FAR

confidence_level: 0.95

Usage:

    python -m scripts.statistical_comparisons \
        --manifest comparison_manifest.yaml \
        --output-json paired_statistics.json \
        --output-csv paired_statistics.csv
"""

import argparse
import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from experiment_provenance import select_exact_result_record
from scripts.statistical_utils import (
    DEFAULT_ALPHA,
    PAIRED_T_TEST,
    WILCOXON_TEST,
    difference_profile,
)
from scripts.statistical_utils import holm_adjust as _shared_holm_adjust
from scripts.statistical_utils import (
    holm_correct_family,
    paired_t_test,
    wilcoxon_signed_rank,
)

# Hypothesis family for a manifest analysis. A manifest names one reference,
# the comparison arms tested against it, and the metrics to report, so the
# set of hypotheses is fixed before any result is read.
MANIFEST_FAMILY_SCOPE = "manifest"


def _json_safe(value):
    """
    Convert analysis values into strict JSON-compatible objects.
    """
    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            bool,
            int,
        ),
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        if math.isfinite(value):
            return value

        return str(value)

    if isinstance(
        value,
        np.generic,
    ):
        return _json_safe(
            value.item()
        )

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            _json_safe(item)
            for item in value
        ]

    return str(value)


def read_jsonl_record(
    path,
    record_index=None,
    *,
    allow_exploratory_index=False,
):
    """
    Read one record from a JSON Lines experiment file.

    Positional selection is retained only for explicitly exploratory use.
    """
    path = Path(path)

    if record_index is None:
        raise ValueError(
            "Exploratory JSONL reads require an explicit record index."
        )
    if int(record_index) < 0 and not allow_exploratory_index:
        raise ValueError(
            "Negative/latest record selection requires explicit "
            "exploratory mode."
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"JSONL result file does not exist: {path}"
        )

    records = []

    for line_number, line in enumerate(
        path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in {path} at "
                f"line {line_number}: {error}"
            ) from error

        if not isinstance(
            record,
            Mapping,
        ):
            raise ValueError(
                f"JSONL record at {path}:{line_number} "
                "must be a JSON object."
            )

        records.append(
            dict(record)
        )

    if not records:
        raise ValueError(
            f"No experiment records were found in {path}."
        )

    try:
        return records[
            int(record_index)
        ]
    except IndexError as error:
        raise IndexError(
            f"Record index {record_index} is out of range "
            f"for {path}, which contains "
            f"{len(records)} record(s)."
        ) from error


def _coerce_finite_metric(
    value,
    context,
):
    """
    Convert a metric to a finite floating-point value.
    """
    if isinstance(
        value,
        bool,
    ):
        raise ValueError(
            f"{context} must be numeric, not boolean."
        )

    try:
        numeric_value = float(value)
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"{context} must be numeric, "
            f"received {value!r}."
        ) from error

    if not math.isfinite(
        numeric_value
    ):
        raise ValueError(
            f"{context} must be finite, "
            f"received {value!r}."
        )

    return numeric_value


def extract_seed_metrics(record):
    """
    Return seed-indexed metric dictionaries from one experiment record.
    """
    if not isinstance(
        record,
        Mapping,
    ):
        raise TypeError(
            "Experiment record must be a mapping."
        )

    per_run_results = record.get(
        "per_run_results"
    )

    if not isinstance(
        per_run_results,
        list,
    ) or not per_run_results:
        raise ValueError(
            "Experiment record does not contain "
            "non-empty per_run_results."
        )

    seed_metrics = {}

    for run_position, run_result in enumerate(
        per_run_results,
        start=1,
    ):
        if not isinstance(
            run_result,
            Mapping,
        ):
            raise ValueError(
                "Every per_run_results entry must "
                "be a mapping."
            )

        if "seed" not in run_result:
            raise ValueError(
                f"Per-run result {run_position} "
                "does not contain a seed."
            )

        try:
            seed = int(
                run_result["seed"]
            )
        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                f"Per-run result {run_position} "
                "contains an invalid seed."
            ) from error

        if seed in seed_metrics:
            raise ValueError(
                f"Duplicate seed {seed} found in "
                "per_run_results."
            )

        metrics = run_result.get(
            "metrics"
        )

        if not isinstance(
            metrics,
            Mapping,
        ) or not metrics:
            raise ValueError(
                f"Per-run result for seed {seed} "
                "does not contain metrics."
            )

        seed_metrics[seed] = {
            str(metric_name): (
                _coerce_finite_metric(
                    metric_value,
                    (
                        f"Metric {metric_name!r} "
                        f"for seed {seed}"
                    ),
                )
            )
            for (
                metric_name,
                metric_value,
            ) in metrics.items()
        }

    return seed_metrics


def _common_metrics_across_seeds(
    seed_metrics,
):
    """
    Find metric names present for every seed, preserving first-run order.
    """
    first_metrics = next(
        iter(
            seed_metrics.values()
        )
    )

    common_metrics = list(
        first_metrics.keys()
    )

    for metrics in seed_metrics.values():
        common_metrics = [
            metric_name
            for metric_name in common_metrics
            if metric_name in metrics
        ]

    return common_metrics


def _validate_matching_seed_sets(
    reference_seed_metrics,
    comparison_seed_metrics,
):
    """
    Require exact reference/comparison seed pairing.
    """
    reference_seeds = set(
        reference_seed_metrics
    )
    comparison_seeds = set(
        comparison_seed_metrics
    )

    if (
        reference_seeds
        == comparison_seeds
    ):
        return

    missing_from_comparison = sorted(
        reference_seeds
        - comparison_seeds
    )
    missing_from_reference = sorted(
        comparison_seeds
        - reference_seeds
    )

    raise ValueError(
        "Reference and comparison experiments "
        "must contain identical seed sets. "
        "Missing from comparison: "
        f"{missing_from_comparison}; "
        "missing from reference: "
        f"{missing_from_reference}."
    )


def resolve_metric_names(
    reference_seed_metrics,
    comparison_seed_metrics,
    requested_metrics=None,
):
    """
    Resolve metrics that are present for every paired seed.
    """
    reference_metrics = (
        _common_metrics_across_seeds(
            reference_seed_metrics
        )
    )
    comparison_metrics = set(
        _common_metrics_across_seeds(
            comparison_seed_metrics
        )
    )

    common_metrics = [
        metric_name
        for metric_name in reference_metrics
        if metric_name in comparison_metrics
    ]

    if requested_metrics is None:
        if not common_metrics:
            raise ValueError(
                "No common per-run metrics were found."
            )

        return common_metrics

    requested_metrics = [
        str(metric_name)
        for metric_name in requested_metrics
    ]

    if not requested_metrics:
        raise ValueError(
            "The requested metric list is empty."
        )

    if len(
        set(requested_metrics)
    ) != len(
        requested_metrics
    ):
        raise ValueError(
            "Requested metric names must be unique."
        )

    missing_metrics = [
        metric_name
        for metric_name in requested_metrics
        if metric_name not in common_metrics
    ]

    if missing_metrics:
        raise ValueError(
            "Requested metrics are not available "
            "for every paired seed: "
            f"{missing_metrics}."
        )

    return requested_metrics


def align_metric_values(
    reference_seed_metrics,
    comparison_seed_metrics,
    metric_name,
):
    """
    Align one metric by ascending seed rather than JSONL list order.
    """
    _validate_matching_seed_sets(
        reference_seed_metrics,
        comparison_seed_metrics,
    )

    seeds = sorted(
        reference_seed_metrics
    )

    reference_values = np.asarray(
        [
            reference_seed_metrics[
                seed
            ][metric_name]
            for seed in seeds
        ],
        dtype=float,
    )
    comparison_values = np.asarray(
        [
            comparison_seed_metrics[
                seed
            ][metric_name]
            for seed in seeds
        ],
        dtype=float,
    )

    return (
        seeds,
        reference_values,
        comparison_values,
    )


def calculate_paired_statistics(
    reference_values,
    comparison_values,
    confidence_level=0.95,
):
    """
    Calculate paired inference for ``comparison - reference``.

    Existing benchmark aggregation remains unchanged. This analysis uses the
    sample standard deviation of paired differences for confidence intervals
    and Cohen's dz, as required for inferential statistics.
    """
    reference_values = np.asarray(
        reference_values,
        dtype=float,
    )
    comparison_values = np.asarray(
        comparison_values,
        dtype=float,
    )

    if (
        reference_values.ndim != 1
        or comparison_values.ndim != 1
    ):
        raise ValueError(
            "Paired values must be one-dimensional."
        )

    if (
        reference_values.shape
        != comparison_values.shape
    ):
        raise ValueError(
            "Reference and comparison values "
            "must have identical shapes."
        )

    if reference_values.size == 0:
        raise ValueError(
            "At least one paired observation is required."
        )

    if not (
        np.all(
            np.isfinite(
                reference_values
            )
        )
        and np.all(
            np.isfinite(
                comparison_values
            )
        )
    ):
        raise ValueError(
            "Paired observations must be finite."
        )

    confidence_level = float(
        confidence_level
    )

    if not (
        0.0
        < confidence_level
        < 1.0
    ):
        raise ValueError(
            "confidence_level must be between "
            "zero and one."
        )

    profile = difference_profile(
        reference_values,
        comparison_values,
    )

    number_of_pairs = profile["n_pairs"]
    mean_difference = profile["mean_difference"]

    result = {
        "status": (
            "ok"
            if number_of_pairs >= 2
            else "insufficient_runs"
        ),
        "n_pairs": number_of_pairs,
        "reference_mean": float(
            np.mean(
                reference_values
            )
        ),
        "comparison_mean": float(
            np.mean(
                comparison_values
            )
        ),
        "mean_difference": (
            mean_difference
        ),
        (
            "difference_standard_"
            "deviation_sample"
        ): None,
        "confidence_level": (
            confidence_level
        ),
        "mean_difference_ci_lower": None,
        "mean_difference_ci_upper": None,
        "paired_t_statistic": None,
        "paired_t_p_value": None,
        "wilcoxon_statistic": None,
        "wilcoxon_p_value": None,
        "cohens_dz": None,
    }

    if number_of_pairs < 2:
        return result

    difference_standard_deviation = profile[
        "standard_deviation"
    ]

    result[
        (
            "difference_standard_"
            "deviation_sample"
        )
    ] = difference_standard_deviation

    if profile["constant_differences"]:
        ci_lower = mean_difference
        ci_upper = mean_difference
    else:
        standard_error = (
            difference_standard_deviation
            / math.sqrt(
                number_of_pairs
            )
        )
        alpha = (
            1.0
            - confidence_level
        )
        critical_value = float(
            stats.t.ppf(
                1.0
                - alpha / 2.0,
                df=(
                    number_of_pairs
                    - 1
                ),
            )
        )
        margin = (
            critical_value
            * standard_error
        )
        ci_lower = (
            mean_difference
            - margin
        )
        ci_upper = (
            mean_difference
            + margin
        )

    result[
        "mean_difference_ci_lower"
    ] = float(ci_lower)
    result[
        "mean_difference_ci_upper"
    ] = float(ci_upper)

    paired_t_result = paired_t_test(
        reference_values,
        comparison_values,
    )
    wilcoxon_result = wilcoxon_signed_rank(
        reference_values,
        comparison_values,
    )

    result[
        "paired_t_statistic"
    ] = paired_t_result[
        "statistic"
    ]
    result[
        "paired_t_p_value"
    ] = paired_t_result[
        "raw_p"
    ]
    result[
        "cohens_dz"
    ] = paired_t_result[
        "effect_size_dz"
    ]
    result[
        "wilcoxon_statistic"
    ] = wilcoxon_result[
        "statistic"
    ]
    result[
        "wilcoxon_p_value"
    ] = wilcoxon_result[
        "raw_p"
    ]

    return result


def holm_adjust(
    p_values,
):
    """
    Apply the Holm step-down family-wise error correction.

    ``None`` entries are preserved and excluded from the correction family.
    The implementation is shared with the other result-analysis scripts.
    """
    return _shared_holm_adjust(
        p_values
    )


def _record_descriptor(
    record,
    path,
    record_index,
    label,
    locator=None,
):
    """
    Return concise provenance for one selected experiment record.
    """
    return {
        "label": str(label),
        "record_locator": (
            locator
            if locator is not None
            else {
                "legacy_exploratory": True,
                "relative_path": Path(path).name,
                "record_index": int(record_index),
            }
        ),
        "record_index": int(record_index),
        "experiment_time": record.get(
            "experiment_time"
        ),
        "dataset": record.get(
            "dataset"
        ),
        "task": record.get(
            "task"
        ),
    }


def compare_records(
    reference_record,
    comparison_record,
    reference_descriptor,
    comparison_descriptor,
    requested_metrics=None,
    confidence_level=0.95,
):
    """
    Compare two experiment records after exact seed alignment.
    """
    reference_seed_metrics = (
        extract_seed_metrics(
            reference_record
        )
    )
    comparison_seed_metrics = (
        extract_seed_metrics(
            comparison_record
        )
    )

    _validate_matching_seed_sets(
        reference_seed_metrics,
        comparison_seed_metrics,
    )

    metric_names = resolve_metric_names(
        reference_seed_metrics,
        comparison_seed_metrics,
        requested_metrics=requested_metrics,
    )

    metric_results = []

    for metric_name in metric_names:
        (
            seeds,
            reference_values,
            comparison_values,
        ) = align_metric_values(
            reference_seed_metrics,
            comparison_seed_metrics,
            metric_name,
        )

        statistics_result = (
            calculate_paired_statistics(
                reference_values,
                comparison_values,
                confidence_level=(
                    confidence_level
                ),
            )
        )

        metric_results.append(
            {
                "metric": metric_name,
                "paired_seeds": seeds,
                **statistics_result,
            }
        )

    return {
        "reference": dict(
            reference_descriptor
        ),
        "comparison": dict(
            comparison_descriptor
        ),
        "metrics": metric_results,
    }


def _apply_holm_corrections(
    comparisons,
    family_scope=MANIFEST_FAMILY_SCOPE,
    alpha=DEFAULT_ALPHA,
):
    """
    Apply separate Holm corrections to the t-test and Wilcoxon families.

    The hypothesis family is the manifest itself. One manifest is a
    prespecified analysis: it names one reference, the comparison arms tested
    against it, and the metrics to report. Every ``(arm, metric)`` hypothesis
    the manifest declares is corrected together, so the family is fixed by the
    analysis definition rather than by whichever rows a particular run
    happens to produce.

    The parametric and non-parametric tests answer the same question with
    different assumptions, so they are corrected as two separate families and
    are never pooled. Each row records the ``family_id`` it was corrected in,
    the ``test`` that produced its p-value, and the resulting decision.
    """
    metric_rows = [
        metric_result
        for comparison in comparisons
        for metric_result in comparison[
            "metrics"
        ]
    ]

    for (
        test_name,
        source_key,
        adjusted_key,
        family_key,
        reject_key,
    ) in (
        (
            PAIRED_T_TEST,
            "paired_t_p_value",
            (
                "paired_t_p_value_"
                "holm"
            ),
            "paired_t_family_id",
            "paired_t_reject",
        ),
        (
            WILCOXON_TEST,
            "wilcoxon_p_value",
            (
                "wilcoxon_p_value_"
                "holm"
            ),
            "wilcoxon_family_id",
            "wilcoxon_reject",
        ),
    ):
        family_id = f"{family_scope}::{test_name}"

        holm_correct_family(
            metric_rows,
            raw_p_key=source_key,
            adjusted_p_key=adjusted_key,
            reject_key=reject_key,
            alpha=alpha,
        )

        for metric_result in metric_rows:
            metric_result[
                family_key
            ] = family_id


def _resolve_experiment_spec(
    spec,
    base_directory,
    default_label,
    publication_mode,
):
    """
    Resolve one manifest experiment specification.
    """
    if not isinstance(
        spec,
        Mapping,
    ):
        raise ValueError(
            "Experiment specifications must "
            "be mappings."
        )

    if "path" not in spec:
        raise ValueError(
            "Experiment specification is "
            "missing 'path'."
        )

    path = Path(
        spec["path"]
    )

    if not path.is_absolute():
        path = (
            base_directory
            / path
        )

    path = path.resolve()

    label = str(
        spec.get(
            "label",
            default_label,
        )
    )

    if publication_mode:
        required_fields = (
            "scientific_configuration_sha256",
            "implementation_source_sha256",
        )
        missing = [field for field in required_fields if field not in spec]
        if missing:
            raise ValueError(
                "Publication experiment specification is missing: "
                + ", ".join(missing)
            )
        selected = select_exact_result_record(
            path,
            result_root=base_directory,
            expected_scientific_sha256=str(
                spec["scientific_configuration_sha256"]
            ),
            expected_implementation={
                "source_sha256": str(
                    spec["implementation_source_sha256"]
                ),
                "git": {"commit": spec.get("git_commit")},
            },
            required_campaign_id=spec.get("campaign_id"),
            publication_mode=True,
            expected_smoke_run=False,
        )
        record = selected["record"]
        locator = selected["locator"]
        record_index = locator["line_number"]
    else:
        record_index = int(spec.get("record_index", -1))
        record = read_jsonl_record(
            path,
            record_index=record_index,
            allow_exploratory_index=True,
        )
        locator = None

    descriptor = _record_descriptor(
        record,
        path,
        record_index,
        label,
        locator=locator,
    )

    return (
        record,
        descriptor,
    )


def analyse_manifest(
    manifest,
    manifest_path=None,
):
    """
    Execute all paired comparisons defined by a loaded manifest.
    """
    if not isinstance(
        manifest,
        Mapping,
    ):
        raise ValueError(
            "Comparison manifest must be a mapping."
        )

    if manifest_path is None:
        base_directory = Path.cwd()
    else:
        base_directory = (
            Path(manifest_path)
            .resolve()
            .parent
        )

    if "reference" not in manifest:
        raise ValueError(
            "Comparison manifest is missing "
            "'reference'."
        )

    comparisons_spec = manifest.get(
        "comparisons"
    )

    if not isinstance(
        comparisons_spec,
        list,
    ) or not comparisons_spec:
        raise ValueError(
            "Comparison manifest must contain "
            "at least one comparison."
        )

    requested_metrics = manifest.get(
        "metrics"
    )

    if requested_metrics is not None:
        if not isinstance(
            requested_metrics,
            list,
        ):
            raise ValueError(
                "'metrics' must be a list."
            )

    confidence_level = float(
        manifest.get(
            "confidence_level",
            0.95,
        )
    )
    publication_mode = not bool(
        manifest.get("allow_exploratory_results", False)
    )

    (
        reference_record,
        reference_descriptor,
    ) = _resolve_experiment_spec(
        manifest["reference"],
        base_directory,
        default_label="reference",
        publication_mode=publication_mode,
    )

    comparison_results = []

    for comparison_index, comparison_spec in enumerate(
        comparisons_spec,
        start=1,
    ):
        (
            comparison_record,
            comparison_descriptor,
        ) = _resolve_experiment_spec(
            comparison_spec,
            base_directory,
            default_label=(
                f"comparison_{comparison_index}"
            ),
            publication_mode=publication_mode,
        )

        comparison_results.append(
            compare_records(
                reference_record=(
                    reference_record
                ),
                comparison_record=(
                    comparison_record
                ),
                reference_descriptor=(
                    reference_descriptor
                ),
                comparison_descriptor=(
                    comparison_descriptor
                ),
                requested_metrics=(
                    requested_metrics
                ),
                confidence_level=(
                    confidence_level
                ),
            )
        )

    _apply_holm_corrections(
        comparison_results
    )

    return {
        "analysis_direction": (
            "comparison - reference"
        ),
        "confidence_level": (
            confidence_level
        ),
        "reference": (
            reference_descriptor
        ),
        "comparisons": (
            comparison_results
        ),
    }


def load_manifest(path):
    """
    Load a JSON or YAML comparison manifest.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest does not exist: {path}"
        )

    suffix = path.suffix.lower()
    text = path.read_text(
        encoding="utf-8"
    )

    if suffix == ".json":
        manifest = json.loads(
            text
        )
    elif suffix in {
        ".yaml",
        ".yml",
    }:
        manifest = yaml.safe_load(
            text
        )
    else:
        raise ValueError(
            "Comparison manifest must use "
            ".json, .yaml, or .yml."
        )

    if not isinstance(
        manifest,
        Mapping,
    ):
        raise ValueError(
            "Comparison manifest must "
            "contain a mapping."
        )

    return dict(manifest)


def write_json_summary(
    path,
    analysis,
):
    """
    Write a strict, human-readable JSON summary.
    """
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            _json_safe(
                analysis
            ),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _flatten_comparison_rows(
    analysis,
):
    """
    Flatten nested comparison output for CSV export.
    """
    rows = []

    for comparison in analysis[
        "comparisons"
    ]:
        reference = comparison[
            "reference"
        ]
        comparison_descriptor = (
            comparison[
                "comparison"
            ]
        )

        for metric_result in comparison[
            "metrics"
        ]:
            rows.append(
                {
                    "reference_label": (
                        reference["label"]
                    ),
                    "comparison_label": (
                        comparison_descriptor[
                            "label"
                        ]
                    ),
                        "reference_path": (
                            reference[
                                "record_locator"
                            ]["relative_path"]
                        ),
                        "comparison_path": (
                            comparison_descriptor[
                                "record_locator"
                            ][
                                "relative_path"
                            ]
                        ),
                    "reference_record_index": (
                        reference[
                            "record_index"
                        ]
                    ),
                    "comparison_record_index": (
                        comparison_descriptor[
                            "record_index"
                        ]
                    ),
                    "reference_dataset": (
                        reference.get(
                            "dataset"
                        )
                    ),
                    "comparison_dataset": (
                        comparison_descriptor.get(
                            "dataset"
                        )
                    ),
                    "reference_task": (
                        reference.get(
                            "task"
                        )
                    ),
                    "comparison_task": (
                        comparison_descriptor.get(
                            "task"
                        )
                    ),
                    "metric": (
                        metric_result[
                            "metric"
                        ]
                    ),
                    "paired_seeds": ";".join(
                        str(seed)
                        for seed in metric_result[
                            "paired_seeds"
                        ]
                    ),
                    "status": (
                        metric_result[
                            "status"
                        ]
                    ),
                    "n_pairs": (
                        metric_result[
                            "n_pairs"
                        ]
                    ),
                    "reference_mean": (
                        metric_result[
                            "reference_mean"
                        ]
                    ),
                    "comparison_mean": (
                        metric_result[
                            "comparison_mean"
                        ]
                    ),
                    "mean_difference": (
                        metric_result[
                            "mean_difference"
                        ]
                    ),
                    (
                        "difference_standard_"
                        "deviation_sample"
                    ): metric_result[
                        (
                            "difference_standard_"
                            "deviation_sample"
                        )
                    ],
                    "confidence_level": (
                        metric_result[
                            "confidence_level"
                        ]
                    ),
                    "mean_difference_ci_lower": (
                        metric_result[
                            "mean_difference_ci_lower"
                        ]
                    ),
                    "mean_difference_ci_upper": (
                        metric_result[
                            "mean_difference_ci_upper"
                        ]
                    ),
                    "paired_t_statistic": (
                        metric_result[
                            "paired_t_statistic"
                        ]
                    ),
                    "paired_t_p_value": (
                        metric_result[
                            "paired_t_p_value"
                        ]
                    ),
                    "paired_t_p_value_holm": (
                        metric_result[
                            (
                                "paired_t_p_value_"
                                "holm"
                            )
                        ]
                    ),
                    "paired_t_reject": (
                        metric_result[
                            "paired_t_reject"
                        ]
                    ),
                    "paired_t_family_id": (
                        metric_result[
                            "paired_t_family_id"
                        ]
                    ),
                    "wilcoxon_statistic": (
                        metric_result[
                            "wilcoxon_statistic"
                        ]
                    ),
                    "wilcoxon_p_value": (
                        metric_result[
                            "wilcoxon_p_value"
                        ]
                    ),
                    "wilcoxon_p_value_holm": (
                        metric_result[
                            (
                                "wilcoxon_p_value_"
                                "holm"
                            )
                        ]
                    ),
                    "wilcoxon_reject": (
                        metric_result[
                            "wilcoxon_reject"
                        ]
                    ),
                    "wilcoxon_family_id": (
                        metric_result[
                            "wilcoxon_family_id"
                        ]
                    ),
                    "cohens_dz": (
                        metric_result[
                            "cohens_dz"
                        ]
                    ),
                    "alpha": DEFAULT_ALPHA,
                }
            )

    return rows


def write_csv_summary(
    path,
    analysis,
):
    """
    Write one CSV row per comparison and metric.
    """
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = _flatten_comparison_rows(
        analysis
    )

    if not rows:
        raise ValueError(
            "Analysis contains no comparison rows."
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                _json_safe(
                    row
                )
            )


def build_argument_parser():
    """
    Build the command-line parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run paired statistical comparisons "
            "over structured ECG experiment "
            "JSONL records."
        )
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help=(
            "JSON or YAML comparison manifest."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help=(
            "Destination JSON summary. Defaults "
            "beside the manifest."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help=(
            "Destination CSV summary. Defaults "
            "beside the manifest."
        ),
    )

    return parser


def main(argv=None):
    """
    Command-line entry point.
    """
    parser = build_argument_parser()
    arguments = parser.parse_args(
        argv
    )

    manifest_path = Path(
        arguments.manifest
    ).resolve()

    manifest = load_manifest(
        manifest_path
    )
    analysis = analyse_manifest(
        manifest,
        manifest_path=manifest_path,
    )

    default_prefix = (
        manifest_path.parent
        / (
            manifest_path.stem
            + "_paired_statistics"
        )
    )

    output_json = (
        Path(arguments.output_json)
        if arguments.output_json
        else default_prefix.with_suffix(
            ".json"
        )
    )
    output_csv = (
        Path(arguments.output_csv)
        if arguments.output_csv
        else default_prefix.with_suffix(
            ".csv"
        )
    )

    write_json_summary(
        output_json,
        analysis,
    )
    write_csv_summary(
        output_csv,
        analysis,
    )

    comparison_count = len(
        analysis["comparisons"]
    )
    metric_count = sum(
        len(
            comparison["metrics"]
        )
        for comparison in analysis[
            "comparisons"
        ]
    )

    print(
        "Paired statistical analysis complete."
    )
    print(
        f"Comparisons: {comparison_count}"
    )
    print(
        f"Metric comparisons: {metric_count}"
    )
    print(
        f"JSON summary: {output_json}"
    )
    print(
        f"CSV summary: {output_csv}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
