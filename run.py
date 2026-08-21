from contextvars import ContextVar
# run.py
# -----------------------------------------------------------------------------
# UNIFIED TRAINING & EVALUATION UTILITY FOR ECG BIOMETRICS
# -----------------------------------------------------------------------------
# This module handles the core deep learning and biometric evaluation logic.
# It features advanced training loops including dynamic Learning Rate rollback,
# strict temporal isolation, and a Composite Validation Metric (CE Loss + EER) 
# to optimize Subject-Disjoint generalization.
#
# SUPPORTED TASKS:
#   1. Closed-Set Identification           (Intra-session, Known Subjects)
#   2. Closed-Set Verification             (Intra-session, Known Subjects)
#   3. Subject-Disjoint Identification     (Intra-session, Unseen Subjects)
#   4. Subject-Disjoint Verification       (Intra-session, Unseen Subjects)
#   5. Cross-Session Identification        (Temporal Robustness, Known Subjects)
#   6. Cross-Session Verification          (Temporal Robustness, Known Subjects)
#   7. Subject-Disjoint Cross-Session ID   (Ultimate Test: Unseen + Temporal)
#   8. Subject-Disjoint Cross-Session Verif(Ultimate Test: Unseen + Temporal)
#
# METRICS:
#   - Identification: Rank-1 Accuracy, Rank-5 Accuracy (with Score-Level Fusion)
#   - Verification:   EER, AUC, d-prime, TAR @ 0.1% FAR
# -----------------------------------------------------------------------------

import numpy as np
import random
import collections
import copy
from functools import wraps
from collections.abc import Mapping
from inspect import signature
from typing import Dict, Any, Optional, Tuple, List, Union

import time

import shlex
import subprocess
import platform
import sys
from importlib import metadata

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy import stats

from utils import (
    _apply_score_fusion, _make_loader, _encode_labels, _get_device, _set_seed,
    _apply_outlier_filter, _compute_sqi, _compute_score_matrix,
    _get_embeddings, _create_templates, _generate_pairs,
    _find_optimal_threshold, _evaluate_with_global_threshold, _summarize_verification_pairs,
    _build_identification_curve_artifacts,
    _build_verification_curve_artifacts,
    _compute_metrics_identification, _compute_metrics_verification,
    _run_training_loop, _run_train_loop_unseen_subjects, _train_epoch, _detect_channels,
    DEFAULT_CACHE_DIR, DEFAULT_RESULTS_DIR, resolve_artifact_path,
    _build_loader_cache_identity, _fingerprint_array_collection,
)

from load_dataset import summarize_partition_log

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import datetime
import json
from pathlib import Path

from visualizations import Visualizer
from data_augmentation import ECGAugmentation

# =============================================================================
# REPRODUCIBILITY ENVIRONMENT
# =============================================================================

def _get_installed_package_version(distribution_name):
    """
    Return the installed version of a Python distribution.

    Missing optional dependencies are reported explicitly rather than
    causing experiment logging to fail.
    """
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _collect_software_environment():
    """
    Collect the software and hardware environment used by an experiment.
    """
    environment = {
        "Python": platform.python_version(),
        "Operating System": platform.platform(),
        "PyTorch": str(torch.__version__),
        "CUDA Available": bool(torch.cuda.is_available()),
        "CUDA Runtime": (
            str(torch.version.cuda)
            if torch.version.cuda is not None
            else "not available"
        ),
        "NumPy": _get_installed_package_version("numpy"),
        "SciPy": _get_installed_package_version("scipy"),
        "scikit-learn": _get_installed_package_version(
            "scikit-learn"
        ),
        "pandas": _get_installed_package_version("pandas"),
        "NeuroKit2": _get_installed_package_version(
            "neurokit2"
        ),
        "WFDB": _get_installed_package_version("wfdb"),
        "PyYAML": _get_installed_package_version("PyYAML"),
    }

    if torch.cuda.is_available():
        try:
            environment["CUDA Device"] = (
                torch.cuda.get_device_name(0)
            )
        except Exception:
            environment["CUDA Device"] = "unavailable"

    return environment

# =============================================================================
# COMPUTATIONAL PROFILE
# =============================================================================

_EXPERIMENT_START_TIME = None

_RUNTIME_STAGE_TOTALS = collections.defaultdict(
    float
)
_RUNTIME_STAGE_COUNTS = collections.defaultdict(
    int
)
_RUNTIME_RUN_TIMES = []

_RUNTIME_STAGE_ORDER = (
    "Data Preparation (inclusive)",
    "Data Cache Read",
    "Data Cache Write",
    "Partition Preparation",
    "Weight Cache Read",
    "Model Training",
    "Weight Cache Write",
    "Embedding Extraction",
    "Template Construction",
    "Similarity Scoring",
    "Pair Generation",
    "Probe Fusion",
    "Metric Computation",
)


def _reset_runtime_profile():
    """
    Clear stage and per-run timing measurements.
    """
    _RUNTIME_STAGE_TOTALS.clear()
    _RUNTIME_STAGE_COUNTS.clear()
    _RUNTIME_RUN_TIMES.clear()


def _synchronize_cuda():
    """
    Synchronize pending CUDA operations before a timing boundary.

    CUDA execution is asynchronous, so synchronization is required for
    meaningful wall-clock measurements. Synchronization failures are ignored
    to ensure profiling cannot interrupt an experiment.
    """
    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.synchronize()
    except Exception:
        pass


def start_experiment_timer():
    """
    Start wall-clock, stage, and peak-memory profiling for one experiment.

    The timer is started by the command-line entry point before dataset
    loading, so total duration covers data preparation, training, evaluation,
    and result generation up to creation of the experiment log.
    """
    global _EXPERIMENT_START_TIME

    _reset_runtime_profile()
    _EXPERIMENT_START_TIME = time.perf_counter()

    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def _start_runtime_stage():
    """
    Return a synchronized high-resolution stage start time.
    """
    _synchronize_cuda()
    return time.perf_counter()


def _finish_runtime_interval(started_at):
    """
    Finish one synchronized timing interval and return elapsed seconds.
    """
    _synchronize_cuda()

    elapsed_seconds = (
        time.perf_counter()
        - started_at
    )

    return max(
        0.0,
        float(elapsed_seconds),
    )


def _record_runtime_stage(
    stage_name,
    started_at,
):
    """
    Add one elapsed interval to a named runtime stage.
    """
    elapsed_seconds = _finish_runtime_interval(
        started_at
    )

    _RUNTIME_STAGE_TOTALS[
        str(stage_name)
    ] += elapsed_seconds

    _RUNTIME_STAGE_COUNTS[
        str(stage_name)
    ] += 1

    return elapsed_seconds


def _timed_runtime_call(
    stage_name,
    function,
    *args,
    **kwargs,
):
    """
    Execute a callable and record its complete wall-clock duration.

    Timing is also recorded when the callable raises, after which the original
    exception is propagated unchanged.
    """
    started_at = _start_runtime_stage()

    try:
        return function(
            *args,
            **kwargs,
        )
    finally:
        _record_runtime_stage(
            stage_name,
            started_at,
        )


def _record_multi_run_time(
    run_index,
    seed,
    started_at,
):
    """
    Record the complete wall-clock duration of one recursive seed run.
    """
    elapsed_seconds = _finish_runtime_interval(
        started_at
    )

    _RUNTIME_RUN_TIMES.append(
        {
            "run_index": int(run_index),
            "seed": int(seed),
            "seconds": elapsed_seconds,
        }
    )

    return elapsed_seconds


def _make_runtime_wrapper(
    stage_name,
    function,
):
    """
    Create a transparent timing wrapper around an imported helper.
    """
    @wraps(function)
    def timed_function(
        *args,
        **kwargs,
    ):
        return _timed_runtime_call(
            stage_name,
            function,
            *args,
            **kwargs,
        )

    return timed_function


# Retain direct references to the numerical implementations. The local names
# used by this module are then replaced with transparent timing wrappers.
_ORIGINAL_RUN_TRAINING_LOOP = (
    _run_training_loop
)
_ORIGINAL_RUN_TRAIN_LOOP_UNSEEN_SUBJECTS = (
    _run_train_loop_unseen_subjects
)
_ORIGINAL_GET_EMBEDDINGS = (
    _get_embeddings
)
_ORIGINAL_CREATE_TEMPLATES = (
    _create_templates
)
_ORIGINAL_COMPUTE_SCORE_MATRIX = (
    _compute_score_matrix
)
_ORIGINAL_GENERATE_PAIRS = (
    _generate_pairs
)
_ORIGINAL_APPLY_SCORE_FUSION = (
    _apply_score_fusion
)
_ORIGINAL_COMPUTE_METRICS_IDENTIFICATION = (
    _compute_metrics_identification
)
_ORIGINAL_COMPUTE_METRICS_VERIFICATION = (
    _compute_metrics_verification
)


_run_training_loop = _make_runtime_wrapper(
    "Model Training",
    _ORIGINAL_RUN_TRAINING_LOOP,
)

_run_train_loop_unseen_subjects = (
    _make_runtime_wrapper(
        "Model Training",
        _ORIGINAL_RUN_TRAIN_LOOP_UNSEEN_SUBJECTS,
    )
)

_get_embeddings = _make_runtime_wrapper(
    "Embedding Extraction",
    _ORIGINAL_GET_EMBEDDINGS,
)

_create_templates = _make_runtime_wrapper(
    "Template Construction",
    _ORIGINAL_CREATE_TEMPLATES,
)

_compute_score_matrix = _make_runtime_wrapper(
    "Similarity Scoring",
    _ORIGINAL_COMPUTE_SCORE_MATRIX,
)

_generate_pairs = _make_runtime_wrapper(
    "Pair Generation",
    _ORIGINAL_GENERATE_PAIRS,
)

_apply_score_fusion = _make_runtime_wrapper(
    "Probe Fusion",
    _ORIGINAL_APPLY_SCORE_FUSION,
)

_compute_metrics_identification = (
    _make_runtime_wrapper(
        "Metric Computation",
        _ORIGINAL_COMPUTE_METRICS_IDENTIFICATION,
    )
)

_compute_metrics_verification = (
    _make_runtime_wrapper(
        "Metric Computation",
        _ORIGINAL_COMPUTE_METRICS_VERIFICATION,
    )
)


def _collect_runtime_profile():
    """
    Return total, stage, multi-run, and optional CUDA-memory statistics.

    An empty dictionary is returned when the complete experiment timer was not
    initialized, such as when a task runner is called directly rather than
    through main.py.
    """
    if _EXPERIMENT_START_TIME is None:
        return {}

    _synchronize_cuda()

    elapsed_seconds = (
        time.perf_counter()
        - _EXPERIMENT_START_TIME
    )

    profile = {
        "Total Wall-Clock Time (seconds)": float(
            elapsed_seconds
        ),
    }

    ordered_stage_names = list(
        _RUNTIME_STAGE_ORDER
    )

    additional_stage_names = sorted(
        set(_RUNTIME_STAGE_TOTALS)
        - set(ordered_stage_names)
    )

    ordered_stage_names.extend(
        additional_stage_names
    )

    for stage_name in ordered_stage_names:
        if stage_name not in _RUNTIME_STAGE_TOTALS:
            continue

        profile[
            f"{stage_name} Time (seconds)"
        ] = float(
            _RUNTIME_STAGE_TOTALS[
                stage_name
            ]
        )

        profile[
            f"{stage_name} Calls"
        ] = int(
            _RUNTIME_STAGE_COUNTS[
                stage_name
            ]
        )

    if _RUNTIME_RUN_TIMES:
        run_durations = np.asarray(
            [
                entry["seconds"]
                for entry in _RUNTIME_RUN_TIMES
            ],
            dtype=float,
        )

        for entry in _RUNTIME_RUN_TIMES:
            run_index = entry[
                "run_index"
            ]

            profile[
                f"Run {run_index} Seed"
            ] = entry["seed"]

            profile[
                (
                    f"Run {run_index} "
                    "Wall-Clock Time (seconds)"
                )
            ] = float(
                entry["seconds"]
            )

        profile[
            "Per-Run Time Mean (seconds)"
        ] = float(
            np.mean(run_durations)
        )

        profile[
            "Per-Run Time Std (seconds)"
        ] = float(
            np.std(run_durations)
        )

        profile[
            "Per-Run Time Min (seconds)"
        ] = float(
            np.min(run_durations)
        )

        profile[
            "Per-Run Time Max (seconds)"
        ] = float(
            np.max(run_durations)
        )

    if torch.cuda.is_available():
        try:
            peak_memory_bytes = (
                torch.cuda.max_memory_allocated()
            )

            profile[
                "Peak CUDA Memory (MiB)"
            ] = (
                peak_memory_bytes
                / (1024 ** 2)
            )
        except Exception:
            profile[
                "Peak CUDA Memory (MiB)"
            ] = "unavailable"

    return profile


def _summarize_model_complexity(model):
    """
    Summarize parameter counts and state size for a PyTorch model.

    The reported size covers model parameters and registered buffers. It does
    not include optimizer state, activations, temporary tensors, or dataset
    memory.
    """
    if model is None:
        raise ValueError(
            "model cannot be None when computing model complexity."
        )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )

    buffer_bytes = sum(
        buffer.numel() * buffer.element_size()
        for buffer in model.buffers()
    )

    model_state_size_mib = (
        parameter_bytes + buffer_bytes
    ) / (1024 ** 2)

    return {
        "Total Model Parameters": int(total_parameters),
        "Trainable Model Parameters": int(
            trainable_parameters
        ),
        "Model State Size (MiB)": float(
            model_state_size_mib
        ),
    }

# =============================================================================
# SOURCE REVISION METADATA
# =============================================================================

def _run_git_command(*arguments):
    """
    Run a read-only Git command from the repository root.

    ``None`` is returned when Git is unavailable, the directory is not a
    repository, or the command cannot complete successfully.
    """
    repository_root = Path(__file__).resolve().parent

    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None

    return result.stdout.strip()


def _collect_source_revision():
    """
    Collect the source-code revision and process invocation.

    The dirty-state field is important because a commit hash alone cannot
    reproduce an experiment when local uncommitted modifications were used.
    """
    commit = _run_git_command(
        "rev-parse",
        "HEAD",
    )

    branch = _run_git_command(
        "rev-parse",
        "--abbrev-ref",
        "HEAD",
    )

    working_tree_status = _run_git_command(
        "status",
        "--porcelain",
    )

    if working_tree_status is None:
        working_tree_dirty = "unavailable"
    else:
        working_tree_dirty = bool(
            working_tree_status
        )

    return {
        "Git Commit": commit or "unavailable",
        "Git Branch": branch or "unavailable",
        "Git Working Tree Dirty": working_tree_dirty,
        "Python Invocation": shlex.join(sys.argv),
    }

# =============================================================================
# STRUCTURED EXPERIMENT OUTPUT
# =============================================================================

def _to_json_compatible(value):
    """
    Recursively convert experiment metadata into JSON-compatible values.

    NumPy scalars and arrays, tensors, paths, tuples, sets, and uncommon
    metadata objects are converted without modifying the caller's objects.
    Non-finite floating-point values are represented as strings because
    strict JSON does not support NaN or infinity.
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
        if np.isfinite(value):
            return value

        return str(value)

    if isinstance(
        value,
        np.generic,
    ):
        return _to_json_compatible(
            value.item()
        )

    if isinstance(
        value,
        np.ndarray,
    ):
        return _to_json_compatible(
            value.tolist()
        )

    if torch.is_tensor(value):
        return _to_json_compatible(
            value.detach().cpu().tolist()
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
            str(key): _to_json_compatible(
                item
            )
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
            _to_json_compatible(item)
            for item in value
        ]

    if isinstance(
        value,
        (
            set,
            frozenset,
        ),
    ):
        return [
            _to_json_compatible(item)
            for item in sorted(
                value,
                key=repr,
            )
        ]

    if isinstance(
        value,
        bytes,
    ):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    if isinstance(
        value,
        type,
    ):
        return (
            f"{value.__module__}."
            f"{value.__qualname__}"
        )

    try:
        json.dumps(
            value,
            allow_nan=False,
        )
    except (
        TypeError,
        ValueError,
    ):
        return str(value)

    return value


# Separators accepted when splitting an aggregate "mean +/- std" string.
# The task runners emit U+00B1; the ASCII spellings are accepted so that
# logs written by a terminal or editor that cannot represent U+00B1 remain
# machine-readable.
_AGGREGATE_METRIC_SEPARATORS = (
    "±",
    "+/-",
    "+-",
)


def _to_structured_result_value(value):
    """
    Preserve numeric metrics and normalize mean-plus-std result strings.

    Multi-run task runners report aggregate metrics as strings such as
    ``0.9500 ± 0.0100``. The structured output exposes their numeric
    mean and standard deviation while retaining the original display value,
    so downstream analysis never has to re-parse formatted text.
    """
    if isinstance(
        value,
        str,
    ):
        for separator in _AGGREGATE_METRIC_SEPARATORS:
            if value.count(separator) != 1:
                continue

            mean_text, std_text = value.split(
                separator,
                maxsplit=1,
            )

            try:
                mean_value = float(
                    mean_text.strip()
                )
                std_value = float(
                    std_text.strip()
                )
            except ValueError:
                continue

            return {
                "mean": mean_value,
                "std": std_value,
                "display": value,
            }

    return _to_json_compatible(
        value
    )


def _build_structured_experiment_record(
    experiment_time,
    task_name,
    dataset_name,
    metrics_dict,
    data_stats,
    hyperparams,
    dataset_kwargs,
    software_environment,
    source_revision,
    runtime_profile,
    per_run_results=None,
    evaluation_artifacts=None,
    per_run_evaluation_artifacts=None,
):
    """
    Build one self-contained machine-readable experiment record.
    """
    return {
        "experiment_time": (
            experiment_time.isoformat(
                timespec="seconds"
            )
        ),
        "task": str(task_name),
        "dataset": str(dataset_name),
        "data_statistics": (
            _to_json_compatible(
                data_stats
            )
        ),
        "effective_experiment_configuration": {
            "model_hyperparameters": (
                _to_json_compatible(
                    hyperparams
                )
            ),
            (
                "dataset_and_preprocessing_"
                "settings"
            ): _to_json_compatible(
                dataset_kwargs
            ),
        },
        "software_and_hardware_environment": (
            _to_json_compatible(
                software_environment
            )
        ),
        "source_revision": (
            _to_json_compatible(
                source_revision
            )
        ),
        "computational_profile": (
            _to_json_compatible(
                runtime_profile
            )
        ),
        "per_run_results": (
            _to_json_compatible(
                (
                    []
                    if per_run_results is None
                    else per_run_results
                )
            )
        ),
        "across_seed_uncertainty": (
            _to_json_compatible(
                _summarize_per_run_uncertainty(
                    per_run_results
                )
            )
        ),
        "evaluation_artifacts": {
            "single_run": (
                _to_json_compatible(
                    evaluation_artifacts
                )
            ),
            "per_run": (
                _to_json_compatible(
                    (
                        []
                        if per_run_evaluation_artifacts
                        is None
                        else per_run_evaluation_artifacts
                    )
                )
            ),
        },
        "results": {
            str(key): (
                _to_structured_result_value(
                    value
                )
            )
            for key, value in metrics_dict.items()
        },
    }


def _append_structured_experiment_record(
    structured_log_file,
    record,
):
    """
    Append one strict JSON record to a JSON Lines result file.
    """
    serialized_record = json.dumps(
        _to_json_compatible(
            record
        ),
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(
            ",",
            ":",
        ),
    )

    with open(
        structured_log_file,
        "a",
        encoding="utf-8",
        newline="\n",
    ) as structured_file:
        structured_file.write(
            serialized_record
        )
        structured_file.write(
            "\n"
        )



def _write_csv_rows(
    output_path,
    rows,
):
    """Write dictionaries to CSV while preserving first-seen field order."""
    rows = list(rows)

    if not rows:
        return None

    import csv

    fieldnames = []

    for row in rows:
        for fieldname in row:
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)

    with open(
        output_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    return Path(output_path)


def _save_compact_evaluation_outputs(
    results_dir,
    safe_task_name,
    experiment_time,
    per_run_results=None,
    evaluation_artifacts=None,
    per_run_evaluation_artifacts=None,
):
    """
    Save curve outputs without embedding large arrays in the JSONL record.

    Scalar metrics, comparison counts, operating points, and compact CMC data
    remain in the structured record. Full ROC/DET arrays and probe-level rank
    arrays are stored in one compressed NumPy file. Plots and CSV summaries are
    generated automatically whenever result logging is enabled.
    """
    if per_run_evaluation_artifacts:
        artifact_items = copy.deepcopy(
            list(per_run_evaluation_artifacts)
        )
        is_multi_run = True
    elif evaluation_artifacts:
        artifact_items = [
            {
                "run_index": 1,
                "seed": None,
                "artifact": copy.deepcopy(
                    evaluation_artifacts
                ),
            }
        ]
        is_multi_run = False
    else:
        return (
            evaluation_artifacts,
            (
                []
                if per_run_evaluation_artifacts is None
                else per_run_evaluation_artifacts
            ),
        )

    experiment_id = experiment_time.strftime(
        "%Y%m%dT%H%M%S_%f"
    )

    output_directory = (
        Path(results_dir)
        / "evaluation_outputs"
        / safe_task_name
        / experiment_id
    )

    curve_arrays = {}
    compact_items = []
    operating_point_rows = []
    cmc_rows = []
    artifact_type = None

    for item in artifact_items:
        run_index = int(
            item.get(
                "run_index",
                1,
            )
        )
        seed = item.get("seed")
        artifact = copy.deepcopy(
            item["artifact"]
        )

        current_type = artifact.get("type")

        if current_type not in {
            "verification",
            "identification",
        }:
            compact_items.append(
                {
                    "run_index": run_index,
                    "seed": seed,
                    "artifact": artifact,
                }
            )
            continue

        if artifact_type is None:
            artifact_type = current_type
        elif artifact_type != current_type:
            raise ValueError(
                "One experiment cannot contain mixed "
                "evaluation artifact types."
            )

        prefix = (
            f"seed_{int(seed)}"
            if seed is not None
            else f"run_{run_index}"
        )

        if current_type == "verification":
            roc_curve_data = artifact.pop(
                "roc_curve",
                None,
            )
            det_curve_data = artifact.pop(
                "det_curve",
                None,
            )

            if roc_curve_data:
                curve_arrays[
                    f"{prefix}_roc_false_accept_rates"
                ] = np.asarray(
                    roc_curve_data[
                        "false_accept_rates"
                    ],
                    dtype=float,
                )
                curve_arrays[
                    f"{prefix}_roc_true_accept_rates"
                ] = np.asarray(
                    roc_curve_data[
                        "true_accept_rates"
                    ],
                    dtype=float,
                )

            if det_curve_data:
                curve_arrays[
                    f"{prefix}_det_false_accept_rates"
                ] = np.asarray(
                    det_curve_data[
                        "false_accept_rates"
                    ],
                    dtype=float,
                )
                curve_arrays[
                    f"{prefix}_det_false_reject_rates"
                ] = np.asarray(
                    det_curve_data[
                        "false_reject_rates"
                    ],
                    dtype=float,
                )

            for operating_point in artifact.get(
                "operating_points",
                [],
            ):
                operating_point_rows.append(
                    {
                        "run_index": run_index,
                        "seed": seed,
                        **operating_point,
                    }
                )

        else:
            correct_match_ranks = artifact.pop(
                "correct_match_ranks",
                None,
            )
            cmc_curve_data = artifact.get(
                "cmc_curve"
            )

            if correct_match_ranks is not None:
                curve_arrays[
                    f"{prefix}_correct_match_ranks"
                ] = np.asarray(
                    correct_match_ranks,
                    dtype=int,
                )

            if cmc_curve_data:
                curve_arrays[
                    f"{prefix}_cmc_ranks"
                ] = np.asarray(
                    cmc_curve_data["ranks"],
                    dtype=int,
                )
                curve_arrays[
                    f"{prefix}_cmc_identification_rates"
                ] = np.asarray(
                    cmc_curve_data[
                        "identification_rates"
                    ],
                    dtype=float,
                )

                for rank, rate in zip(
                    cmc_curve_data["ranks"],
                    cmc_curve_data[
                        "identification_rates"
                    ],
                ):
                    cmc_rows.append(
                        {
                            "run_index": run_index,
                            "seed": seed,
                            "rank": int(rank),
                            "identification_rate": float(
                                rate
                            ),
                        }
                    )

        compact_items.append(
            {
                "run_index": run_index,
                "seed": seed,
                "artifact": artifact,
                "curve_prefix": prefix,
            }
        )

    if not curve_arrays:
        if is_multi_run:
            return (
                None,
                [
                    {
                        "run_index": item[
                            "run_index"
                        ],
                        "seed": item["seed"],
                        "artifact": item[
                            "artifact"
                        ],
                    }
                    for item in compact_items
                ],
            )

        return (
            compact_items[0]["artifact"],
            [],
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    curve_file = output_directory / "curves.npz"

    np.savez_compressed(
        curve_file,
        **curve_arrays,
    )

    relative_curve_file = curve_file.relative_to(
        results_dir
    ).as_posix()

    per_seed_metric_rows = []

    for result in per_run_results or []:
        per_seed_metric_rows.append(
            {
                "run_index": result.get(
                    "run_index"
                ),
                "seed": result.get("seed"),
                **result.get(
                    "metrics",
                    {},
                ),
            }
        )

    _write_csv_rows(
        output_directory
        / "per_seed_metrics.csv",
        per_seed_metric_rows,
    )

    _write_csv_rows(
        output_directory
        / "verification_operating_points.csv",
        operating_point_rows,
    )

    _write_csv_rows(
        output_directory
        / "identification_cmc.csv",
        cmc_rows,
    )

    plot_paths = {}

    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import DetCurveDisplay

        if artifact_type == "verification":
            roc_figure, roc_axis = plt.subplots(
                figsize=(7, 6)
            )
            det_figure, det_axis = plt.subplots(
                figsize=(7, 6)
            )
            tar_figure, tar_axis = plt.subplots(
                figsize=(7, 6)
            )

            for item in artifact_items:
                artifact = item["artifact"]
                seed = item.get("seed")
                run_index = item.get(
                    "run_index",
                    1,
                )
                label = (
                    f"Seed {seed}"
                    if seed is not None
                    else f"Run {run_index}"
                )

                roc_curve_data = artifact.get(
                    "roc_curve"
                )
                det_curve_data = artifact.get(
                    "det_curve"
                )

                if roc_curve_data:
                    roc_axis.plot(
                        roc_curve_data[
                            "false_accept_rates"
                        ],
                        roc_curve_data[
                            "true_accept_rates"
                        ],
                        label=(
                            f"{label} "
                            f"(AUC={artifact['roc_auc']:.3f})"
                        ),
                    )

                if det_curve_data:
                    DetCurveDisplay(
                        fpr=np.asarray(
                            det_curve_data[
                                "false_accept_rates"
                            ],
                            dtype=float,
                        ),
                        fnr=np.asarray(
                            det_curve_data[
                                "false_reject_rates"
                            ],
                            dtype=float,
                        ),
                        estimator_name=label,
                    ).plot(
                        ax=det_axis
                    )

                ordered_points = sorted(
                    artifact.get(
                        "operating_points",
                        [],
                    ),
                    key=lambda point: point[
                        "target_far"
                    ],
                )

                if ordered_points:
                    tar_axis.semilogx(
                        [
                            point["target_far"]
                            for point in ordered_points
                        ],
                        [
                            point["tar"]
                            for point in ordered_points
                        ],
                        marker="o",
                        label=label,
                    )

            roc_axis.plot(
                [0.0, 1.0],
                [0.0, 1.0],
                linestyle="--",
                label="Random",
            )
            roc_axis.set_xlabel(
                "False Accept Rate"
            )
            roc_axis.set_ylabel(
                "True Accept Rate"
            )
            roc_axis.set_title(
                "Verification ROC Curves"
            )
            roc_axis.legend()
            roc_figure.tight_layout()

            roc_path = output_directory / "roc.png"
            roc_figure.savefig(
                roc_path,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(roc_figure)
            plot_paths["roc"] = roc_path

            det_axis.set_title(
                "Verification DET Curves"
            )
            det_figure.tight_layout()

            det_path = output_directory / "det.png"
            det_figure.savefig(
                det_path,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(det_figure)
            plot_paths["det"] = det_path

            tar_axis.set_xlabel(
                "Target FAR"
            )
            tar_axis.set_ylabel(
                "True Accept Rate"
            )
            tar_axis.set_title(
                "TAR at Requested FARs"
            )
            tar_axis.set_ylim(
                0.0,
                1.0,
            )
            tar_axis.grid(
                True,
                which="both",
            )
            tar_axis.legend()
            tar_figure.tight_layout()

            tar_path = (
                output_directory
                / "tar_at_far.png"
            )
            tar_figure.savefig(
                tar_path,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(tar_figure)
            plot_paths["tar_at_far"] = tar_path

        elif artifact_type == "identification":
            cmc_figure, cmc_axis = plt.subplots(
                figsize=(7, 6)
            )

            for item in artifact_items:
                artifact = item["artifact"]
                seed = item.get("seed")
                run_index = item.get(
                    "run_index",
                    1,
                )
                label = (
                    f"Seed {seed}"
                    if seed is not None
                    else f"Run {run_index}"
                )
                cmc_curve_data = artifact.get(
                    "cmc_curve"
                )

                if cmc_curve_data:
                    cmc_axis.plot(
                        cmc_curve_data["ranks"],
                        cmc_curve_data[
                            "identification_rates"
                        ],
                        label=label,
                    )

            cmc_axis.set_xlabel("Rank")
            cmc_axis.set_ylabel(
                "Identification Rate"
            )
            cmc_axis.set_title(
                "Cumulative Match Characteristic"
            )
            cmc_axis.set_ylim(
                0.0,
                1.0,
            )
            cmc_axis.grid(True)
            cmc_axis.legend()
            cmc_figure.tight_layout()

            cmc_path = output_directory / "cmc.png"
            cmc_figure.savefig(
                cmc_path,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(cmc_figure)
            plot_paths["cmc"] = cmc_path

    except (
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            "[WARN] Evaluation curves were saved, "
            "but plot generation failed: "
            f"{error}"
        )

    relative_plot_paths = {
        plot_name: plot_path.relative_to(
            results_dir
        ).as_posix()
        for plot_name, plot_path in (
            plot_paths.items()
        )
    }

    for item in compact_items:
        item["artifact"][
            "curve_storage"
        ] = {
            "file": relative_curve_file,
            "prefix": item.pop(
                "curve_prefix"
            ),
        }

        if relative_plot_paths:
            item["artifact"][
                "plots"
            ] = dict(
                relative_plot_paths
            )

    print(
        "[INFO] Evaluation curves and summaries "
        f"saved to: {output_directory}"
    )

    if is_multi_run:
        return (
            None,
            compact_items,
        )

    return (
        compact_items[0]["artifact"],
        [],
    )


# =============================================================================
# AUTOMATED EXPERIMENT LOGGER
# =============================================================================
def _log_experiment_results(
    task_name,
    metrics_dict,
    data_stats,
    hyperparams,
    loader=None,
    per_run_results=None,
    evaluation_artifacts=None,
    per_run_evaluation_artifacts=None,
):
    """
    Dynamically writes experiment configurations and results to a text file.
    Intelligently extracts parameters directly from the dataset loader object.
    Automatically categorizes and saves the logs into task-specific .txt files.
    """
    dataset_name = "unknown_dataset"
    dataset_kwargs = {}
    
    if loader is not None:
        # 1. Extract Dataset Name from cfg dict
        if hasattr(loader, 'cfg') and 'root_dir' in loader.cfg:
            dataset_name = str(loader.cfg['root_dir']).lower()
            
        # 2. Extract Preprocessing Params (Merge defaults with user overrides)
        default_prep = loader.cfg.get('preprocessing', {}) if hasattr(loader, 'cfg') else {}
        user_prep = getattr(loader, 'prep_params', {})
        
        # This guarantees user-defined params overwrite the defaults
        dataset_kwargs.update(default_prep)
        dataset_kwargs.update(user_prep) 
        
        # 3. Extract other useful loader attributes dynamically
        # We ignore backend objects, large paths, and redundant dictionaries.
        # The raw per-subject partition log is replaced below by its compact
        # summary so that large cohorts do not bloat every experiment record.
        ignore_keys = ['preprocessor', 'cfg', 'data_root', 'dataset_root',
                       'zip_path', 'url', 'prep_params', 'cleanup_zip',
                       'partition_assignment_log']

        for k, v in vars(loader).items():
            if k not in ignore_keys and not k.startswith('_'):
                dataset_kwargs[k] = v

        # 3b. Record the enrollment/probe causality evidence for this run.
        causality_summary = summarize_partition_log(
            loader
        )

        if causality_summary is not None:
            dataset_kwargs[
                "temporal_causality_audit"
            ] = causality_summary

    # 4. Resolve the configured external result directory.
    configured_results_dir = getattr(
        loader,
        "results_dir",
        DEFAULT_RESULTS_DIR,
    )

    results_dir = (
        Path(
            resolve_artifact_path(
                configured_results_dir
            )
        )
        / dataset_name.replace(" ", "_")
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 5. TARGET FILE: Dynamically name the file based on the Task Name
    # (e.g., "Closed-Set Identification" -> "Closed-Set_Identification.txt")
    safe_task_name = str(task_name).replace(" ", "_").replace("/", "_").replace("\\", "_")
    log_file = results_dir / f"{safe_task_name}.txt"
    structured_log_file = (
        results_dir
        / f"{safe_task_name}.jsonl"
    )

    experiment_time = datetime.datetime.now()
    software_environment = _collect_software_environment()
    source_revision = _collect_source_revision()
    runtime_profile = _collect_runtime_profile()

    (
        compact_evaluation_artifacts,
        compact_per_run_evaluation_artifacts,
    ) = _save_compact_evaluation_outputs(
        results_dir=results_dir,
        safe_task_name=safe_task_name,
        experiment_time=experiment_time,
        per_run_results=per_run_results,
        evaluation_artifacts=evaluation_artifacts,
        per_run_evaluation_artifacts=(
            per_run_evaluation_artifacts
        ),
    )
    
    # 6. Format and Append
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*70}\n")
        f.write(
            "EXPERIMENT TIME : "
            f"{experiment_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"TASK            : {task_name}\n")
        f.write(f"DATASET         : {dataset_name}\n")
        f.write(f"{'-'*70}\n")
        
        f.write("[DATA STATISTICS]\n")
        for k, v in data_stats.items():
            f.write(f"  {k:<28}: {v}\n")
        f.write(f"{'-'*70}\n")
        
        f.write("[MODEL HYPERPARAMETERS]\n")
        for k, v in hyperparams.items():
            f.write(f"  {k:<28}: {v}\n")
            
        if dataset_kwargs:
            f.write(f"{'-'*70}\n")
            f.write("[DATASET & PREPROCESSING SETTINGS]\n")
            for k, v in dataset_kwargs.items():
                f.write(f"  {k:<28}: {v}\n")

        f.write(f"{'-'*70}\n")
        f.write("[SOFTWARE & HARDWARE ENVIRONMENT]\n")

        for key, value in software_environment.items():
            f.write(f"  {key:<28}: {value}\n")

        f.write(f"{'-' * 70}\n")
        f.write("[SOURCE REVISION]\n")

        for key, value in source_revision.items():
            f.write(f"  {key:<28}: {value}\n")

        if runtime_profile:
            f.write(f"{'-'*70}\n")
            f.write("[COMPUTATIONAL PROFILE]\n")

            for key, value in runtime_profile.items():
                if isinstance(value, float):
                    f.write(
                        f"  {key:<36}: {value:.4f}\n"
                    )
                else:
                    f.write(
                        f"  {key:<36}: {value}\n"
                    )

        f.write(f"{'-'*70}\n")
        f.write("[RESULTS]\n")
        for k, v in metrics_dict.items():
            if isinstance(v, float):
                f.write(f"  {k:<28}: {v:.4f}\n")
            else:
                f.write(f"  {k:<28}: {v}\n")

        # Across-seed intervals are reported alongside, never instead of, the
        # aggregate values above.
        uncertainty_summary = (
            _summarize_per_run_uncertainty(
                per_run_results
            )
        )

        if uncertainty_summary is not None:
            confidence_percentage = (
                uncertainty_summary["confidence_level"]
                * 100.0
            )

            f.write(f"{'-'*70}\n")
            f.write(
                "[ACROSS-SEED UNCERTAINTY] "
                f"{confidence_percentage:g}% confidence "
                f"over {uncertainty_summary['runs']} run(s)\n"
            )

            for (
                metric_name,
                metric_summary,
            ) in uncertainty_summary["metrics"].items():
                t_interval = metric_summary["t_interval"]
                bootstrap_interval = metric_summary[
                    "bootstrap_interval"
                ]

                if t_interval is None:
                    f.write(
                        f"  {metric_name:<28}: "
                        "single run, no interval\n"
                    )
                    continue

                f.write(
                    f"  {metric_name:<28}: "
                    f"t [{t_interval['lower']:.4f}, "
                    f"{t_interval['upper']:.4f}]  "
                    f"bootstrap [{bootstrap_interval['lower']:.4f}, "
                    f"{bootstrap_interval['upper']:.4f}]\n"
                )

        f.write(f"{'='*70}\n")

    structured_record = (
        _build_structured_experiment_record(
            experiment_time=experiment_time,
            task_name=task_name,
            dataset_name=dataset_name,
            metrics_dict=metrics_dict,
            data_stats=data_stats,
            hyperparams=hyperparams,
            dataset_kwargs=dataset_kwargs,
            software_environment=software_environment,
            source_revision=source_revision,
            runtime_profile=runtime_profile,
            per_run_results=per_run_results,
            evaluation_artifacts=(
                compact_evaluation_artifacts
            ),
            per_run_evaluation_artifacts=(
                compact_per_run_evaluation_artifacts
            ),
        )
    )

    _append_structured_experiment_record(
        structured_log_file,
        structured_record,
    )

    print(
        "\n[INFO] Experiment settings and "
        f"results successfully saved to: {log_file}"
    )
    print(
        "[INFO] Structured experiment record "
        f"appended to: {structured_log_file}"
    )

# =============================================================================
# EVALUATION CONFIGURATION VALIDATION
# =============================================================================
def _validate_deployment_evaluation(
    use_deployment_evaluation,
    val_split,
    task_name,
):
    """
    Require an independent validation partition for threshold calibration.

    Deployment evaluation estimates a decision threshold on validation data
    and then freezes that threshold for final test evaluation. Calibrating on
    training data would produce optimistically biased deployment results.
    """
    if not use_deployment_evaluation:
        return

    try:
        validation_fraction = float(val_split)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{task_name}: val_split must be numeric when "
            "deployment evaluation is enabled."
        ) from error

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            f"{task_name}: deployment evaluation requires "
            "0 < val_split < 1. Training-data threshold calibration "
            "is not permitted."
        )

def _split_subject_cohorts(
    subjects,
    test_split,
    val_split,
    seed,
):
    """
    Split unique subjects into disjoint train, validation, and test cohorts.

    ``test_split`` is applied first to the complete eligible subject set.
    ``val_split`` is then applied to the remaining training-subject pool,
    preserving the framework's existing split semantics.

    Returns:
        tuple:
            (
                train_subjects,
                validation_subjects,
                test_subjects,
            )
    """
    subject_array = np.asarray(
        subjects
    )

    if subject_array.ndim != 1:
        raise ValueError(
            "subjects must be a one-dimensional sequence."
        )

    unique_subjects = np.unique(
        subject_array
    )

    if len(unique_subjects) < 2:
        raise ValueError(
            "At least two unique subjects are required "
            "to create train and test cohorts."
        )

    split_values = {
        "test_split": test_split,
        "val_split": val_split,
    }

    normalized_splits = {}

    for split_name, split_value in (
        split_values.items()
    ):
        if isinstance(
            split_value,
            (
                bool,
                np.bool_,
            ),
        ):
            raise ValueError(
                f"{split_name} must be numeric, not Boolean."
            )

        try:
            normalized_value = float(
                split_value
            )
        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                f"{split_name} must be numeric."
            ) from error

        if not np.isfinite(
            normalized_value
        ):
            raise ValueError(
                f"{split_name} must be finite."
            )

        normalized_splits[
            split_name
        ] = normalized_value

    test_fraction = normalized_splits[
        "test_split"
    ]

    validation_fraction = normalized_splits[
        "val_split"
    ]

    if not 0.0 < test_fraction < 1.0:
        raise ValueError(
            "test_split must satisfy 0 < test_split < 1."
        )

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError(
            "val_split must satisfy 0 <= val_split < 1."
        )

    try:
        train_validation_subjects, test_subjects = (
            train_test_split(
                unique_subjects,
                test_size=test_fraction,
                random_state=seed,
            )
        )
    except ValueError as error:
        raise ValueError(
            "Unable to create train and test subject "
            "cohorts from the available subjects."
        ) from error

    if validation_fraction > 0.0:
        if len(train_validation_subjects) < 2:
            raise ValueError(
                "At least two subjects must remain after "
                "the test split when val_split is positive."
            )

        try:
            (
                train_subjects,
                validation_subjects,
            ) = train_test_split(
                train_validation_subjects,
                test_size=validation_fraction,
                random_state=seed,
            )
        except ValueError as error:
            raise ValueError(
                "Unable to create train and validation "
                "subject cohorts from the remaining subjects."
            ) from error
    else:
        train_subjects = (
            train_validation_subjects
        )

        validation_subjects = np.asarray(
            [],
            dtype=unique_subjects.dtype,
        )

    train_subjects = np.asarray(
        train_subjects
    )

    validation_subjects = np.asarray(
        validation_subjects
    )

    test_subjects = np.asarray(
        test_subjects
    )

    train_set = set(
        train_subjects.tolist()
    )

    validation_set = set(
        validation_subjects.tolist()
    )

    test_set = set(
        test_subjects.tolist()
    )

    all_subjects = set(
        unique_subjects.tolist()
    )

    if train_set & validation_set:
        raise RuntimeError(
            "Train and validation subject cohorts overlap."
        )

    if train_set & test_set:
        raise RuntimeError(
            "Train and test subject cohorts overlap."
        )

    if validation_set & test_set:
        raise RuntimeError(
            "Validation and test subject cohorts overlap."
        )

    assigned_subjects = (
        train_set
        | validation_set
        | test_set
    )

    if assigned_subjects != all_subjects:
        raise RuntimeError(
            "Subject cohort splitting did not preserve "
            "the complete eligible subject set."
        )

    return (
        train_subjects,
        validation_subjects,
        test_subjects,
    )

def _split_closed_set_samples(
    x,
    y,
    holdout_split,
    seed,
    aligned_values=None,
):
    """
    Create a stratified closed-set sample partition.

    The retained and holdout partitions contain the same identities but
    mutually exclusive samples. Optional aligned values, such as SQI scores,
    are partitioned using exactly the same sample indices.

    A zero holdout fraction returns the complete input as the retained
    partition and a disabled holdout represented by ``(None, None, None)``.
    """
    x = np.asarray(
        x
    )

    y = np.asarray(
        y
    )

    if x.ndim < 1:
        raise ValueError(
            "x must contain a sample dimension."
        )

    if y.ndim != 1:
        raise ValueError(
            "y must be a one-dimensional label array."
        )

    if len(x) != len(y):
        raise ValueError(
            "x and y must contain the same number of samples."
        )

    if aligned_values is not None:
        aligned_values = np.asarray(
            aligned_values
        )

        if (
            aligned_values.ndim == 0
            or len(aligned_values) != len(y)
        ):
            raise ValueError(
                "aligned_values must contain one value "
                "per sample."
            )

    if isinstance(
        holdout_split,
        (
            bool,
            np.bool_,
        ),
    ):
        raise ValueError(
            "holdout_split must be numeric, not Boolean."
        )

    try:
        holdout_fraction = float(
            holdout_split
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "holdout_split must be numeric."
        ) from error

    if not np.isfinite(
        holdout_fraction
    ):
        raise ValueError(
            "holdout_split must be finite."
        )

    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError(
            "holdout_split must satisfy "
            "0 <= holdout_split < 1."
        )

    all_indices = np.arange(
        len(y),
        dtype=np.int64,
    )

    if holdout_fraction == 0.0:
        return {
            "retained": (
                x,
                y,
                aligned_values,
            ),
            "holdout": (
                None,
                None,
                None,
            ),
            "indices": {
                "retained": all_indices,
                "holdout": np.asarray(
                    [],
                    dtype=np.int64,
                ),
            },
        }

    identities, identity_counts = np.unique(
        y,
        return_counts=True,
    )

    identities_with_too_few_samples = (
        identities[
            identity_counts < 2
        ]
    )

    if len(
        identities_with_too_few_samples
    ) > 0:
        raise ValueError(
            "Every closed-set identity must contain at "
            "least two samples before splitting. "
            "Insufficient identities: "
            f"{identities_with_too_few_samples.tolist()}."
        )

    try:
        (
            retained_indices,
            holdout_indices,
        ) = train_test_split(
            all_indices,
            test_size=holdout_fraction,
            stratify=y,
            random_state=seed,
        )
    except ValueError as error:
        raise ValueError(
            "Unable to create a stratified closed-set "
            "sample split with the requested fraction."
        ) from error

    retained_indices = np.asarray(
        retained_indices,
        dtype=np.int64,
    )

    holdout_indices = np.asarray(
        holdout_indices,
        dtype=np.int64,
    )

    retained_index_set = set(
        retained_indices.tolist()
    )

    holdout_index_set = set(
        holdout_indices.tolist()
    )

    all_index_set = set(
        all_indices.tolist()
    )

    if retained_index_set & holdout_index_set:
        raise RuntimeError(
            "Closed-set retained and holdout samples overlap."
        )

    assigned_indices = (
        retained_index_set
        | holdout_index_set
    )

    if assigned_indices != all_index_set:
        raise RuntimeError(
            "Closed-set splitting did not preserve exactly "
            "all source samples."
        )

    all_identities = set(
        identities.tolist()
    )

    retained_identities = set(
        np.unique(
            y[retained_indices]
        ).tolist()
    )

    holdout_identities = set(
        np.unique(
            y[holdout_indices]
        ).tolist()
    )

    if retained_identities != all_identities:
        raise RuntimeError(
            "The retained closed-set partition does not "
            "contain every identity."
        )

    if holdout_identities != all_identities:
        raise RuntimeError(
            "The holdout closed-set partition does not "
            "contain every identity."
        )

    retained_aligned_values = (
        aligned_values[
            retained_indices
        ]
        if aligned_values is not None
        else None
    )

    holdout_aligned_values = (
        aligned_values[
            holdout_indices
        ]
        if aligned_values is not None
        else None
    )

    return {
        "retained": (
            x[retained_indices],
            y[retained_indices],
            retained_aligned_values,
        ),
        "holdout": (
            x[holdout_indices],
            y[holdout_indices],
            holdout_aligned_values,
        ),
        "indices": {
            "retained": retained_indices,
            "holdout": holdout_indices,
        },
    }

def _partition_closed_set_cross_session_samples(
    x_session_1,
    y_session_1,
    x_session_2,
    y_session_2,
    minimum_session_1_samples=2,
    minimum_session_2_samples=1,
):
    """
    Synchronise known-subject samples across two sessions.

    Only identities present in both sessions and satisfying the configured
    per-session sample requirements are retained. Original sample order is
    preserved within each session.

    Session 1 normally supplies representation-learning and enrollment
    samples. Session 2 supplies probe samples.
    """
    x_session_1 = np.asarray(
        x_session_1
    )

    y_session_1 = np.asarray(
        y_session_1
    )

    x_session_2 = np.asarray(
        x_session_2
    )

    y_session_2 = np.asarray(
        y_session_2
    )

    if x_session_1.ndim < 1:
        raise ValueError(
            "Session 1 samples must contain a sample dimension."
        )

    if x_session_2.ndim < 1:
        raise ValueError(
            "Session 2 samples must contain a sample dimension."
        )

    if y_session_1.ndim != 1:
        raise ValueError(
            "Session 1 labels must be one-dimensional."
        )

    if y_session_2.ndim != 1:
        raise ValueError(
            "Session 2 labels must be one-dimensional."
        )

    if len(x_session_1) != len(y_session_1):
        raise ValueError(
            "Session 1 samples and labels are misaligned."
        )

    if len(x_session_2) != len(y_session_2):
        raise ValueError(
            "Session 2 samples and labels are misaligned."
        )

    minimum_values = {
        "minimum_session_1_samples": (
            minimum_session_1_samples
        ),
        "minimum_session_2_samples": (
            minimum_session_2_samples
        ),
    }

    normalised_minimums = {}

    for parameter_name, value in (
        minimum_values.items()
    ):
        if isinstance(
            value,
            (
                bool,
                np.bool_,
            ),
        ) or not isinstance(
            value,
            (
                int,
                np.integer,
            ),
        ):
            raise ValueError(
                f"{parameter_name} must be a positive integer."
            )

        value = int(
            value
        )

        if value < 1:
            raise ValueError(
                f"{parameter_name} must be a positive integer."
            )

        normalised_minimums[
            parameter_name
        ] = value

    minimum_session_1_samples = (
        normalised_minimums[
            "minimum_session_1_samples"
        ]
    )

    minimum_session_2_samples = (
        normalised_minimums[
            "minimum_session_2_samples"
        ]
    )

    session_1_subjects = np.unique(
        y_session_1
    )

    session_2_subjects = np.unique(
        y_session_2
    )

    common_subjects = np.intersect1d(
        session_1_subjects,
        session_2_subjects,
    )

    if len(common_subjects) < 2:
        raise ValueError(
            "At least two identities must be shared by "
            "Session 1 and Session 2."
        )

    session_1_counts = {
        subject: int(
            np.sum(
                y_session_1 == subject
            )
        )
        for subject in common_subjects
    }

    session_2_counts = {
        subject: int(
            np.sum(
                y_session_2 == subject
            )
        )
        for subject in common_subjects
    }

    insufficient_session_1 = np.asarray(
        [
            subject
            for subject in common_subjects
            if session_1_counts[subject]
            < minimum_session_1_samples
        ],
        dtype=common_subjects.dtype,
    )

    insufficient_session_2 = np.asarray(
        [
            subject
            for subject in common_subjects
            if session_2_counts[subject]
            < minimum_session_2_samples
        ],
        dtype=common_subjects.dtype,
    )

    eligible_subjects = np.asarray(
        [
            subject
            for subject in common_subjects
            if (
                session_1_counts[subject]
                >= minimum_session_1_samples
                and session_2_counts[subject]
                >= minimum_session_2_samples
            )
        ],
        dtype=common_subjects.dtype,
    )

    if len(eligible_subjects) < 2:
        raise ValueError(
            "At least two shared identities must satisfy "
            "the required Session 1 and Session 2 "
            "sample counts."
        )

    session_1_mask = np.isin(
        y_session_1,
        eligible_subjects,
    )

    session_2_mask = np.isin(
        y_session_2,
        eligible_subjects,
    )

    filtered_x_session_1 = (
        x_session_1[
            session_1_mask
        ]
    )

    filtered_y_session_1 = (
        y_session_1[
            session_1_mask
        ]
    )

    filtered_x_session_2 = (
        x_session_2[
            session_2_mask
        ]
    )

    filtered_y_session_2 = (
        y_session_2[
            session_2_mask
        ]
    )

    expected_subjects = set(
        eligible_subjects.tolist()
    )

    retained_session_1_subjects = set(
        np.unique(
            filtered_y_session_1
        ).tolist()
    )

    retained_session_2_subjects = set(
        np.unique(
            filtered_y_session_2
        ).tolist()
    )

    if (
        retained_session_1_subjects
        != expected_subjects
    ):
        raise RuntimeError(
            "Session 1 partitioning did not preserve "
            "exactly the eligible identities."
        )

    if (
        retained_session_2_subjects
        != expected_subjects
    ):
        raise RuntimeError(
            "Session 2 partitioning did not preserve "
            "exactly the eligible identities."
        )

    return {
        "session_1": (
            filtered_x_session_1,
            filtered_y_session_1,
        ),
        "session_2": (
            filtered_x_session_2,
            filtered_y_session_2,
        ),
        "subjects": eligible_subjects,
        "dropped_subjects": {
            "session_1_only": np.setdiff1d(
                session_1_subjects,
                session_2_subjects,
            ),
            "session_2_only": np.setdiff1d(
                session_2_subjects,
                session_1_subjects,
            ),
            "insufficient_session_1": (
                insufficient_session_1
            ),
            "insufficient_session_2": (
                insufficient_session_2
            ),
        },
    }

def _partition_subject_disjoint_samples(
    x,
    y,
    train_subjects,
    validation_subjects,
    test_subjects,
    sqi_scores=None,
):
    """
    Assign intra-session samples to disjoint subject cohorts.

    Training and validation samples are used for representation learning.
    Test-subject samples are reserved for gallery/enrollment and probe
    construction inside the subject-disjoint task runner.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError(
            "y must be a one-dimensional label array."
        )

    if len(x) != len(y):
        raise ValueError(
            "x and y must contain the same number of samples."
        )

    if sqi_scores is not None:
        sqi_scores = np.asarray(
            sqi_scores
        )

        if len(sqi_scores) != len(y):
            raise ValueError(
                "sqi_scores must contain one value per sample."
            )

    train_subjects = np.asarray(
        train_subjects
    )
    validation_subjects = np.asarray(
        validation_subjects
    )
    test_subjects = np.asarray(
        test_subjects
    )

    for cohort_name, cohort in [
        ("train_subjects", train_subjects),
        (
            "validation_subjects",
            validation_subjects,
        ),
        ("test_subjects", test_subjects),
    ]:
        if cohort.ndim != 1:
            raise ValueError(
                f"{cohort_name} must be one-dimensional."
            )

    train_set = set(
        train_subjects.tolist()
    )

    validation_set = set(
        validation_subjects.tolist()
    )

    test_set = set(
        test_subjects.tolist()
    )

    if train_set & validation_set:
        raise ValueError(
            "Train and validation subjects overlap."
        )

    if train_set & test_set:
        raise ValueError(
            "Train and test subjects overlap."
        )

    if validation_set & test_set:
        raise ValueError(
            "Validation and test subjects overlap."
        )

    available_subjects = set(
        np.unique(y).tolist()
    )

    assigned_subjects = (
        train_set
        | validation_set
        | test_set
    )

    if assigned_subjects != available_subjects:
        raise ValueError(
            "The subject cohorts must cover exactly all "
            "subjects present in y."
        )

    def select_samples(subjects):
        mask = np.isin(
            y,
            subjects,
        )

        selected_sqi = (
            sqi_scores[mask]
            if sqi_scores is not None
            else None
        )

        return (
            x[mask],
            y[mask],
            selected_sqi,
        )

    train_partition = select_samples(
        train_subjects
    )

    if len(validation_subjects) > 0:
        validation_partition = select_samples(
            validation_subjects
        )
    else:
        validation_partition = (
            None,
            None,
            None,
        )

    test_partition = select_samples(
        test_subjects
    )

    return {
        "train": train_partition,
        "validation": validation_partition,
        "test": test_partition,
    }


def _partition_subject_disjoint_cross_session_samples(
    x_s1,
    y_s1,
    x_s2,
    y_s2,
    train_subjects,
    validation_subjects,
    test_subjects,
):
    """
    Assign cross-session samples while enforcing temporal isolation.

    Session 1 supplies:
        - representation-learning samples;
        - unseen-subject validation samples;
        - held-out-subject enrollment samples.

    Session 2 supplies:
        - held-out-subject probe samples.
    """
    x_s1 = np.asarray(x_s1)
    y_s1 = np.asarray(y_s1)
    x_s2 = np.asarray(x_s2)
    y_s2 = np.asarray(y_s2)

    if y_s1.ndim != 1 or y_s2.ndim != 1:
        raise ValueError(
            "Session labels must be one-dimensional."
        )

    if len(x_s1) != len(y_s1):
        raise ValueError(
            "Session 1 samples and labels are misaligned."
        )

    if len(x_s2) != len(y_s2):
        raise ValueError(
            "Session 2 samples and labels are misaligned."
        )

    train_subjects = np.asarray(
        train_subjects
    )
    validation_subjects = np.asarray(
        validation_subjects
    )
    test_subjects = np.asarray(
        test_subjects
    )

    for cohort_name, cohort in [
        ("train_subjects", train_subjects),
        (
            "validation_subjects",
            validation_subjects,
        ),
        ("test_subjects", test_subjects),
    ]:
        if cohort.ndim != 1:
            raise ValueError(
                f"{cohort_name} must be one-dimensional."
            )

    train_set = set(
        train_subjects.tolist()
    )

    validation_set = set(
        validation_subjects.tolist()
    )

    test_set = set(
        test_subjects.tolist()
    )

    if train_set & validation_set:
        raise ValueError(
            "Train and validation subjects overlap."
        )

    if train_set & test_set:
        raise ValueError(
            "Train and test subjects overlap."
        )

    if validation_set & test_set:
        raise ValueError(
            "Validation and test subjects overlap."
        )

    common_subjects = (
        set(
            np.unique(y_s1).tolist()
        )
        & set(
            np.unique(y_s2).tolist()
        )
    )

    assigned_subjects = (
        train_set
        | validation_set
        | test_set
    )

    if assigned_subjects != common_subjects:
        raise ValueError(
            "The subject cohorts must cover exactly the "
            "subjects shared by Session 1 and Session 2."
        )

    train_mask_s1 = np.isin(
        y_s1,
        train_subjects,
    )

    validation_mask_s1 = np.isin(
        y_s1,
        validation_subjects,
    )

    enrollment_mask_s1 = np.isin(
        y_s1,
        test_subjects,
    )

    probe_mask_s2 = np.isin(
        y_s2,
        test_subjects,
    )

    train_partition = (
        x_s1[train_mask_s1],
        y_s1[train_mask_s1],
    )

    if len(validation_subjects) > 0:
        validation_partition = (
            x_s1[validation_mask_s1],
            y_s1[validation_mask_s1],
        )
    else:
        validation_partition = (
            None,
            None,
        )

    enrollment_partition = (
        x_s1[enrollment_mask_s1],
        y_s1[enrollment_mask_s1],
    )

    probe_partition = (
        x_s2[probe_mask_s2],
        y_s2[probe_mask_s2],
    )

    return {
        "train": train_partition,
        "validation": validation_partition,
        "enrollment": enrollment_partition,
        "probe": probe_partition,
    }

def _split_enrollment_probe_embeddings(
    embeddings,
    labels,
    subjects,
    template_size,
):
    """
    Split each subject's ordered embeddings into enrollment and probe sets.

    The first ``template_size`` embeddings of every subject are assigned to
    enrollment. All remaining embeddings from that subject are assigned to
    the probe set. Input order is preserved within every subject.
    """
    embeddings = np.asarray(
        embeddings
    )

    labels = np.asarray(
        labels
    )

    subjects = np.asarray(
        subjects
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings must be a two-dimensional array."
        )

    if labels.ndim != 1:
        raise ValueError(
            "labels must be a one-dimensional array."
        )

    if subjects.ndim != 1:
        raise ValueError(
            "subjects must be a one-dimensional array."
        )

    if len(embeddings) != len(labels):
        raise ValueError(
            "embeddings and labels must contain the "
            "same number of samples."
        )

    if isinstance(
        template_size,
        (
            bool,
            np.bool_,
        ),
    ) or not isinstance(
        template_size,
        (
            int,
            np.integer,
        ),
    ):
        raise ValueError(
            "template_size must be a positive integer."
        )

    template_size = int(
        template_size
    )

    if template_size < 1:
        raise ValueError(
            "template_size must be a positive integer."
        )

    if len(subjects) == 0:
        raise ValueError(
            "At least one subject is required."
        )

    if len(np.unique(subjects)) != len(subjects):
        raise ValueError(
            "subjects must not contain duplicate entries."
        )

    available_subjects = set(
        np.unique(labels).tolist()
    )

    requested_subjects = set(
        subjects.tolist()
    )

    if requested_subjects != available_subjects:
        raise ValueError(
            "subjects must match exactly the subjects "
            "present in labels."
        )

    enrollment_embeddings = []
    enrollment_labels = []

    probe_embeddings = []
    probe_labels = []

    for subject in subjects:
        subject_indices = np.flatnonzero(
            labels == subject
        )

        if len(subject_indices) <= template_size:
            raise ValueError(
                f"Subject {subject!r} has "
                f"{len(subject_indices)} samples, but "
                f"template_size={template_size} requires "
                "at least one additional probe sample."
            )

        enrollment_indices = subject_indices[
            :template_size
        ]

        probe_indices = subject_indices[
            template_size:
        ]

        enrollment_embeddings.append(
            embeddings[
                enrollment_indices
            ]
        )

        enrollment_labels.append(
            labels[
                enrollment_indices
            ]
        )

        probe_embeddings.append(
            embeddings[
                probe_indices
            ]
        )

        probe_labels.append(
            labels[
                probe_indices
            ]
        )

    return {
        "enrollment": (
            np.vstack(
                enrollment_embeddings
            ),
            np.concatenate(
                enrollment_labels
            ),
        ),
        "probe": (
            np.vstack(
                probe_embeddings
            ),
            np.concatenate(
                probe_labels
            ),
        ),
    }


def _normalize_augmentation_config(
    augmentation_config,
):
    """
    Validate and normalize training-only augmentation settings.

    The returned mapping is safe to record in experiment metadata and model
    weight-cache identities.
    """
    normalized = {
        "enabled": False,
        "method": "gaussian",
        "copies": 1,
        "parameters": {},
    }

    if augmentation_config is None:
        return normalized

    if not isinstance(
        augmentation_config,
        Mapping,
    ):
        raise ValueError(
            "augmentation_config must be a mapping or None."
        )

    allowed_keys = set(
        normalized
    )

    unknown_keys = (
        set(augmentation_config)
        - allowed_keys
    )

    if unknown_keys:
        raise ValueError(
            "Unknown augmentation_config key(s): "
            + ", ".join(
                sorted(
                    str(key)
                    for key in unknown_keys
                )
            )
        )

    normalized.update(
        dict(augmentation_config)
    )

    if not isinstance(
        normalized["enabled"],
        (
            bool,
            np.bool_,
        ),
    ):
        raise ValueError(
            "augmentation_config['enabled'] must be Boolean."
        )

    method = normalized["method"]

    if not isinstance(
        method,
        str,
    ) or not method.strip():
        raise ValueError(
            "augmentation_config['method'] must be a "
            "non-empty string."
        )

    method = (
        method
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    method_aliases = {
        "time_shift": "timeshift",
        "baseline": "baseline_wander",
        "warp": "time_warp",
        "emg": "emg_noise",
        "istft": "istft_augment",
    }

    method = method_aliases.get(
        method,
        method,
    )

    if method not in (
        ECGAugmentation.SUPPORTED_METHODS
    ):
        raise ValueError(
            "Unknown augmentation method "
            f"{normalized['method']!r}. Supported methods are: "
            f"{', '.join(ECGAugmentation.SUPPORTED_METHODS)}."
        )

    copies = normalized["copies"]

    if isinstance(
        copies,
        (
            bool,
            np.bool_,
        ),
    ) or not isinstance(
        copies,
        (
            int,
            np.integer,
        ),
    ):
        raise ValueError(
            "augmentation_config['copies'] must be a "
            "positive integer."
        )

    copies = int(
        copies
    )

    if copies < 1:
        raise ValueError(
            "augmentation_config['copies'] must be a "
            "positive integer."
        )

    parameters = normalized["parameters"]

    if parameters is None:
        parameters = {}

    if not isinstance(
        parameters,
        Mapping,
    ):
        raise ValueError(
            "augmentation_config['parameters'] must be "
            "a mapping."
        )

    parameters = dict(
        parameters
    )

    method_parameters = set(
        signature(
            getattr(
                ECGAugmentation,
                method,
            )
        ).parameters
    )

    method_parameters.discard(
        "self"
    )

    method_parameters.discard(
        "beats"
    )

    unknown_parameters = (
        set(parameters)
        - method_parameters
    )

    if unknown_parameters:
        raise ValueError(
            f"Unsupported parameter(s) for augmentation "
            f"method {method!r}: "
            + ", ".join(
                sorted(
                    str(key)
                    for key in unknown_parameters
                )
            )
        )

    return {
        "enabled": bool(
            normalized["enabled"]
        ),
        "method": method,
        "copies": copies,
        "parameters": parameters,
    }


def _augment_training_partition(
    x_train,
    y_train,
    augmentation_config,
    seed,
):
    """
    Append deterministic augmented copies of the optimisation partition.

    Only the array passed as ``x_train`` is augmented. Validation, enrollment,
    gallery, calibration, test, and probe arrays must never be passed to this
    helper.

    Both univariate arrays ``(samples, length)`` and channel-first multilead
    arrays ``(samples, channels, length)`` are supported. Morphological
    transformations are synchronised across leads to preserve lead alignment.
    """
    config = _normalize_augmentation_config(
        augmentation_config
    )

    x_train = np.asarray(
        x_train
    )

    y_train = np.asarray(
        y_train
    )

    if y_train.ndim != 1:
        raise ValueError(
            "Training labels must be one-dimensional."
        )

    if len(x_train) != len(y_train):
        raise ValueError(
            "Training samples and labels must contain the "
            "same number of entries."
        )

    if x_train.ndim not in {
        2,
        3,
    }:
        raise ValueError(
            "Training augmentation expects shape "
            "(samples, length) or "
            "(samples, channels, length)."
        )

    if not config["enabled"]:
        return (
            x_train,
            y_train,
        )

    if len(x_train) == 0:
        raise ValueError(
            "Training augmentation cannot be applied to "
            "an empty training partition."
        )

    if isinstance(
        seed,
        (
            bool,
            np.bool_,
        ),
    ) or not isinstance(
        seed,
        (
            int,
            np.integer,
        ),
    ):
        raise ValueError(
            "seed must be an integer for deterministic "
            "training augmentation."
        )

    seed = int(
        seed
    )

    method = config["method"]
    parameters = config["parameters"]

    synchronised_multilead_methods = {
        "amplitude",
        "timeshift",
        "baseline_wander",
        "time_warp",
        "cutout",
    }

    augmented_batches = []

    for copy_index in range(
        config["copies"]
    ):
        copy_seed = (
            seed + copy_index
        )

        if x_train.ndim == 2:
            augmented_copy = (
                ECGAugmentation(
                    seed=copy_seed
                ).apply(
                    x_train,
                    method,
                    **parameters,
                )
            )
        else:
            augmented_copy = np.empty(
                x_train.shape,
                dtype=np.float32,
            )

            for channel_index in range(
                x_train.shape[1]
            ):
                if (
                    method
                    in synchronised_multilead_methods
                ):
                    channel_seed = copy_seed
                else:
                    channel_seed = (
                        copy_seed
                        + 1000 * channel_index
                    )

                augmented_copy[
                    :,
                    channel_index,
                    :,
                ] = ECGAugmentation(
                    seed=channel_seed
                ).apply(
                    x_train[
                        :,
                        channel_index,
                        :,
                    ],
                    method,
                    **parameters,
                )

        augmented_batches.append(
            augmented_copy
        )

    original_batch = x_train.astype(
        np.float32,
        copy=False,
    )

    augmented_x = np.concatenate(
        [
            original_batch,
            *augmented_batches,
        ],
        axis=0,
    )

    augmented_y = np.concatenate(
        [
            y_train
            for _ in range(
                config["copies"] + 1
            )
        ],
        axis=0,
    )

    print(
        "[INFO] Training-only augmentation applied: "
        f"method={method}, "
        f"copies={config['copies']}, "
        f"samples={len(x_train)} -> "
        f"{len(augmented_x)}"
    )

    return (
        augmented_x,
        augmented_y,
    )


# =============================================================================
# MULTI-RUN ARGUMENT HANDLING
# =============================================================================

def _prepare_multi_run_arguments(local_arguments):
    """
    Prepare arguments for a recursive single-seed experiment run.

    """
    call_args = dict(local_arguments)

    internal_keys = {
        "data_stats",
        "hyperparams",
        "call_args",
    }

    for key in internal_keys:
        call_args.pop(key, None)

    call_args.update(
        {
            "n_runs": 1,
            "_return_stats": True,
            "save_results_and_settings": False,
        }
    )

    return call_args

# =============================================================================
# WEIGHT CACHE CONFIGURATION
# =============================================================================

def _build_weight_cache_config(
    loader,
    training_config,
    training_samples=None,
    training_labels=None,
):
    """
    Build the complete identity for reusable trained model weights.

    When training arrays are supplied, their complete post-partition and
    post-augmentation contents are fingerprinted so equal shapes cannot cause
    weights from a different training population to be reused.
    """
    complete_config = dict(
        training_config
    )

    if (
        training_samples is None
    ) != (
        training_labels is None
    ):
        raise ValueError(
            "training_samples and training_labels "
            "must be supplied together."
        )

    if training_samples is not None:
        complete_config[
            "training_partition"
        ] = _fingerprint_array_collection(
            {
                "samples": training_samples,
                "labels": training_labels,
            }
        )

    complete_config[
        "loader_identity"
    ] = _build_loader_cache_identity(
        loader
    )

    return complete_config

def _get_verification_pair_statistics(
    labels_pair,
    target_far=0.001,
):
    """
    Build pair-count statistics and warn when the requested FAR is below
    the empirical resolution of the available impostor comparisons.
    """
    pair_statistics = _summarize_verification_pairs(
        labels_pair,
        target_far=target_far,
    )

    if not pair_statistics[
        "Target FAR Empirically Resolvable"
    ]:
        impostor_count = pair_statistics["Impostor Pairs"]
        minimum_far = pair_statistics[
            "Minimum Non-Zero Empirical FAR"
        ]

        if minimum_far is None:
            print(
                "[WARN] Verification evaluation contains no "
                "impostor comparisons."
            )
        else:
            print(
                "[WARN] TAR@0.1%FAR is below the empirical FAR "
                f"resolution supported by {impostor_count} impostor "
                f"comparisons. Minimum non-zero FAR={minimum_far:.6f}."
            )

    return pair_statistics

# =============================================================================
# RANDOM-SEED METADATA
# =============================================================================

def _build_per_run_results(
    results,
    seeds,
    metric_names=None,
):
    """
    Build structured seed-level metric records for a multi-run experiment.

    The original metric values returned by each single-seed execution are
    retained. This helper does not recompute metrics or alter the aggregate
    mean and standard-deviation calculations performed by the task runners.
    """
    results = list(results)
    seeds = list(seeds)

    if len(results) != len(seeds):
        raise ValueError(
            "The number of multi-run results must match "
            "the number of run seeds."
        )

    if not results:
        return []

    normalized_results = []

    for result in results:
        if isinstance(
            result,
            np.ndarray,
        ):
            if result.ndim == 0:
                values = [
                    result.item()
                ]
            else:
                values = list(
                    result.tolist()
                )
        elif isinstance(
            result,
            (
                list,
                tuple,
            ),
        ):
            values = list(result)
        else:
            values = [
                result
            ]

        normalized_results.append(
            values
        )

    if metric_names is None:
        default_metric_names = {
            2: (
                "Rank-1 Accuracy",
                "Rank-5 Accuracy",
            ),
            4: (
                "EER",
                "AUC",
                "d-prime",
                "TAR@0.1%FAR",
            ),
        }

        metric_names = (
            default_metric_names.get(
                len(
                    normalized_results[0]
                )
            )
        )

        if metric_names is None:
            raise ValueError(
                "Metric names must be provided when "
                "a run returns an unsupported number "
                "of metric values."
            )

    metric_names = [
        str(metric_name)
        for metric_name in metric_names
    ]

    if len(set(metric_names)) != len(
        metric_names
    ):
        raise ValueError(
            "Per-run metric names must be unique."
        )

    per_run_results = []

    for run_index, (
        seed,
        metric_values,
    ) in enumerate(
        zip(
            seeds,
            normalized_results,
        ),
        start=1,
    ):
        if len(metric_values) != len(
            metric_names
        ):
            raise ValueError(
                "Every multi-run result must contain "
                "the same number of values as the "
                "metric-name list."
            )

        per_run_results.append(
            {
                "run_index": run_index,
                "seed": int(seed),
                "metrics": {
                    metric_name: (
                        _to_json_compatible(
                            metric_value
                        )
                    )
                    for (
                        metric_name,
                        metric_value,
                    ) in zip(
                        metric_names,
                        metric_values,
                    )
                },
            }
        )

    return per_run_results


_DEFAULT_CONFIDENCE_LEVEL = 0.95

# Bootstrap resampling is deterministic so a reported interval can be
# regenerated exactly from the same per-seed values.
_BOOTSTRAP_RESAMPLES = 10000
_BOOTSTRAP_SEED = 12345


def _summarize_metric_uncertainty(
    values,
    confidence_level=_DEFAULT_CONFIDENCE_LEVEL,
):
    """
    Summarize the across-seed uncertainty of one metric.

    Two intervals are reported because neither alone is sufficient with the
    small number of seeds typical of this benchmark:

    - A Student-t interval, which is the conventional choice but assumes the
      per-seed values are approximately normal.
    - A percentile bootstrap interval, which makes no distributional
      assumption and is more honest when five seeds produce a skewed spread.

    Both standard deviations are reported. The aggregate ``mean +/- std``
    strings use the population form; the intervals use the sample form,
    which is the correct estimator for the standard error of a mean.
    """
    values = np.asarray(
        [
            float(value)
            for value in values
        ],
        dtype=float,
    )

    if values.size == 0:
        return None

    if not np.all(np.isfinite(values)):
        return None

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must satisfy 0 < level < 1."
        )

    run_count = int(values.size)
    mean_value = float(np.mean(values))

    summary = {
        "runs": run_count,
        "mean": mean_value,
        "population_std": float(
            np.std(values)
        ),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "confidence_level": float(
            confidence_level
        ),
    }

    if run_count < 2:
        # A single seed carries no information about run-to-run spread.
        summary.update(
            {
                "sample_std": None,
                "standard_error": None,
                "t_interval": None,
                "bootstrap_interval": None,
            }
        )

        return summary

    sample_std = float(
        np.std(values, ddof=1)
    )
    standard_error = sample_std / np.sqrt(run_count)

    summary["sample_std"] = sample_std
    summary["standard_error"] = float(
        standard_error
    )

    tail_probability = 1.0 - (
        1.0 - confidence_level
    ) / 2.0
    critical_value = float(
        stats.t.ppf(
            tail_probability,
            df=run_count - 1,
        )
    )

    margin = critical_value * standard_error

    summary["t_interval"] = {
        "lower": float(mean_value - margin),
        "upper": float(mean_value + margin),
        "critical_value": critical_value,
        "degrees_of_freedom": run_count - 1,
        "margin_of_error": float(margin),
    }

    generator = np.random.default_rng(
        _BOOTSTRAP_SEED
    )
    resampled_means = np.mean(
        generator.choice(
            values,
            size=(
                _BOOTSTRAP_RESAMPLES,
                run_count,
            ),
            replace=True,
        ),
        axis=1,
    )

    lower_percentile = (
        1.0 - confidence_level
    ) / 2.0 * 100.0
    upper_percentile = 100.0 - lower_percentile

    summary["bootstrap_interval"] = {
        "lower": float(
            np.percentile(
                resampled_means,
                lower_percentile,
            )
        ),
        "upper": float(
            np.percentile(
                resampled_means,
                upper_percentile,
            )
        ),
        "resamples": _BOOTSTRAP_RESAMPLES,
        "seed": _BOOTSTRAP_SEED,
        "method": "percentile",
    }

    return summary


def _summarize_per_run_uncertainty(
    per_run_results,
    confidence_level=_DEFAULT_CONFIDENCE_LEVEL,
):
    """
    Build across-seed confidence intervals for every recorded metric.

    This is derived purely from the seed-level values already stored in
    ``per_run_results``. It does not recompute, rescale, or replace the
    aggregate metrics reported elsewhere in the record.
    """
    if not per_run_results:
        return None

    values_by_metric = {}

    for run_record in per_run_results:
        metrics = run_record.get("metrics", {})

        for metric_name, metric_value in metrics.items():
            values_by_metric.setdefault(
                str(metric_name),
                [],
            ).append(metric_value)

    if not values_by_metric:
        return None

    metric_summaries = {}

    for metric_name, values in values_by_metric.items():
        try:
            summary = _summarize_metric_uncertainty(
                values,
                confidence_level=confidence_level,
            )
        except (TypeError, ValueError):
            summary = None

        if summary is not None:
            metric_summaries[metric_name] = summary

    if not metric_summaries:
        return None

    return {
        "confidence_level": float(
            confidence_level
        ),
        "runs": len(per_run_results),
        "note": (
            "Intervals describe the across-seed uncertainty of the mean. "
            "The aggregate 'mean +/- std' values use the population "
            "standard deviation; the intervals use the sample standard "
            "deviation, which is the correct estimator for the standard "
            "error of a mean."
        ),
        "metrics": metric_summaries,
    }


def _build_per_run_evaluation_artifacts(
    artifacts,
    seeds,
):
    """
    Build seed-labelled curve and operating-point artifacts.

    Curves are preserved independently for each run. They are not averaged,
    interpolated across seeds, or treated as aggregate curves.
    """
    artifacts = list(
        artifacts
    )
    seeds = list(
        seeds
    )

    if len(artifacts) != len(
        seeds
    ):
        raise ValueError(
            "The number of evaluation artifacts "
            "must match the number of run seeds."
        )

    per_run_artifacts = []

    for run_index, (
        seed,
        artifact,
    ) in enumerate(
        zip(
            seeds,
            artifacts,
        ),
        start=1,
    ):
        if not isinstance(
            artifact,
            Mapping,
        ):
            raise ValueError(
                "Every per-run evaluation artifact "
                "must be a mapping."
            )

        if not artifact:
            raise ValueError(
                "Per-run evaluation artifacts "
                "cannot be empty."
            )

        per_run_artifacts.append(
            {
                "run_index": int(
                    run_index
                ),
                "seed": int(
                    seed
                ),
                "artifact": (
                    _to_json_compatible(
                        artifact
                    )
                ),
            }
        )

    return per_run_artifacts


_EVALUATION_ARTIFACT_SINK = ContextVar(
    "_EVALUATION_ARTIFACT_SINK",
    default=None,
)


def _record_evaluation_artifact(
    artifact,
):
    """
    Record one task artifact in the active multi-run context.

    Direct single-run calls have no active sink and therefore retain their
    established public return contract without storing additional state.
    """
    sink = (
        _EVALUATION_ARTIFACT_SINK.get()
    )

    if sink is not None:
        sink.append(
            artifact
        )


def _add_seed_metadata(
    hyperparams,
    base_seed,
    n_runs,
):
    """
    Add the complete random-seed schedule to experiment metadata.

    Multi-run experiments use consecutive seeds beginning at ``base_seed``.
    A copied dictionary is returned so the caller's original metadata is not
    modified unexpectedly.
    """
    base_seed = int(base_seed)
    n_runs = int(n_runs)

    if n_runs < 1:
        raise ValueError(
            "n_runs must be greater than or equal to 1."
        )

    updated_hyperparams = dict(hyperparams)

    updated_hyperparams.update(
        {
            "base_seed": base_seed,
            "n_runs": n_runs,
            "run_seeds": [
                base_seed + run_index
                for run_index in range(n_runs)
            ],
        }
    )

    return updated_hyperparams

# =============================================================================
# TASK 1: CLOSED-SET IDENTIFICATION
# =============================================================================
def run_closed_set_identification(x, y, model_class, epochs=150, batch_size=256, 
                                  lr=1e-3, test_split=0.2, val_split=0.0, seed=42, 
                                  device=None, visualize=False, use_template=False, 
                                  template_fusion_method='mean', template_size=None,
                                  matching_method='cosine', outlier_filtering_on_train=False,
                                  outlier_filtering_on_test=False, sqi_scores=None,
                                  sqi_threshold=0.05, sqi_keep_pct=0.8, probe_fusion_size=3,
                                  save_results_and_settings=False, loader=None, 
                                  n_runs=1, _return_stats=False,
                                  intelligent_weight_loading=True,
                                  augmentation_config=None):
    """
    Standard Closed-Set Identification Pipeline (Intra-session).
    Determines "Who is this person?" from a known pool of subjects seen during training.

    Args:
        x (np.ndarray): Input ECG signals or features.
        y (np.ndarray): Subject class labels corresponding to x.
        model_class (nn.Module): The PyTorch model architecture class to instantiate.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per training batch.
        lr (float): Learning rate for the Adam optimizer.
        test_split (float): Fraction of the data to hold out for testing (0.0 to 1.0).
        val_split (float): Fraction of the training data to use for early stopping validation.
        seed (int): Random seed for reproducibility across splits and weights.
        device (str): Computation device ('cuda', 'cpu', or 'auto').
        visualize (bool): If True, displays and optionally saves a Confusion Matrix.
        use_template (bool): 
            - False: Uses standard end-to-end Softmax classification.
            - True: Strips the Softmax layer and uses the network as a feature extractor. 
                    Matches Test probes against Train templates.
        template_fusion_method (str): Logic used to create the subject templates.
            Options: ['mean', 'median', 'trimmed_mean', 'representative',
            'soft_centrality', 'geometric_median', 'none']
        template_size (int, optional): Number of beats used to form the template. None uses all available.
        matching_method (str): Distance/Similarity metric for template matching.
            Options: ['cosine', 'euclidean', 'manhattan', 'correlation']
        outlier_filtering_on_train (bool): If True, filters noisy beats from the Training set.
        outlier_filtering_on_test (bool): If True, filters noisy beats from the Test set.
        sqi_scores (str or np.ndarray): SQI calculation method (e.g., 'kurtosis') or pre-computed array.
        sqi_threshold (float): Absolute minimum SQI score required to keep a beat (0.0 to 1.0).
        sqi_keep_pct (float): Top percentage of beats to keep per subject after filtering.
        probe_fusion_size (int): Number of consecutive probe beats to average before making a decision.
        save_results_and_settings (bool): If True, logs results and parameters to a text file.
        loader (object): Dataset loader instance (used for extracting metadata for logging).
        n_runs (int): Number of independent runs (with varying seeds) for statistical validation.
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.

    Returns:
        tuple: (Rank-1 Accuracy, Rank-5 Accuracy)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for both metrics.
    """

    # ====================================================
    # 0. Capture Hyperparameters for Logger & MULTI-RUN AGGREGATOR
    # ====================================================
    data_stats = {}
    hyperparams = {
        'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr, 
        'test_split': test_split, 'val_split': val_split, 'use_template': use_template,
        'template_fusion_method': template_fusion_method, 'template_size': template_size,
        'matching_method': matching_method, 'probe_fusion_size': probe_fusion_size,
        'outlier_filter_train': outlier_filtering_on_train, 'outlier_filter_test': outlier_filtering_on_test
    }

    hyperparams = _add_seed_metadata(
        hyperparams,
        base_seed=seed,
        n_runs=n_runs,
    )

    # 2. MULTI-RUN AGGREGATOR (Handles statistical validation)
    if n_runs > 1:
        # Capture current arguments to repeat the experiment
        call_args = _prepare_multi_run_arguments(locals())
        base_seed = call_args.get('seed', 42)
        results = []
        evaluation_artifact_runs = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs) for Statistical Validation...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False # Prevent 5 pop-up windows
            
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            
            # Recursive call to execute a single seed
            run_wall_clock_started = _start_runtime_stage()
            artifact_sink_token = (
                _EVALUATION_ARTIFACT_SINK.set(
                    evaluation_artifact_runs
                )
            )
            try:
                res, d_stats, h_params = run_closed_set_identification(
                    **call_args
                )
            finally:
                _EVALUATION_ARTIFACT_SINK.reset(
                    artifact_sink_token
                )
            _record_multi_run_time(
                run_index=i + 1,
                seed=call_args["seed"],
                started_at=run_wall_clock_started,
            )
            
            results.append(res)
            # Preserve metadata from the last successful run for the final log file
            data_stats = d_stats  
            hyperparams = h_params 

        hyperparams = _add_seed_metadata(
            hyperparams,
            base_seed=base_seed,
            n_runs=n_runs,
        )

        per_run_results = (
            _build_per_run_results(
                results=results,
                seeds=hyperparams[
                    "run_seeds"
                ],
            )
        )

        per_run_evaluation_artifacts = (
            _build_per_run_evaluation_artifacts(
                artifacts=evaluation_artifact_runs,
                seeds=hyperparams[
                    "run_seeds"
                ],
            )
        )
                
        # Aggregate metrics across all runs
        r1_vals = [r[0] for r in results]
        r5_vals = [r[1] for r in results]
        r1_mean, r1_std = np.mean(r1_vals), np.std(r1_vals)
        r5_mean, r5_std = np.mean(r5_vals), np.std(r5_vals)
        
        print(f"\n[MULTI-RUN RESULTS | {n_runs} Runs]")
        print(f"Rank-1 Acc: {r1_mean:.4f} ± {r1_std:.4f} | Rank-5 Acc: {r5_mean:.4f} ± {r5_std:.4f}")
        
        if save_results_and_settings:
            metrics_dict = {
                "Rank-1 Accuracy": f"{r1_mean:.4f} ± {r1_std:.4f}", 
                "Rank-5 Accuracy": f"{r5_mean:.4f} ± {r5_std:.4f}"
            }
            _log_experiment_results(
                "Closed-Set Identification",
                metrics_dict,
                data_stats,
                hyperparams,
                loader,
                per_run_results=per_run_results,
                per_run_evaluation_artifacts=per_run_evaluation_artifacts,
            )
        
        # Return the aggregated statistics
        return (r1_mean, r1_std), (r5_mean, r5_std)

    augmentation_config = (
        _normalize_augmentation_config(
            augmentation_config
        )
    )

    hyperparams[
        "augmentation"
    ] = augmentation_config

    _set_seed(seed); device = _get_device(device)
    partition_stage_started = _start_runtime_stage()
    task_title = "Closed-Set Identification"
    mode_str = f"Template ({template_fusion_method}, {matching_method})" if use_template else "Softmax"
    print(f"\n[TASK] {task_title} | Mode: {mode_str} | Device: {device}")

    # ====================================================
    # 1. DYNAMIC SQI CALCULATION
    # ====================================================
    if outlier_filtering_on_train or outlier_filtering_on_test:
        if sqi_scores is None:
            print("[WARN] Filtering requested but sqi_scores is None. Skipping filtering entirely.")
            # Explicitly turn off the flags so the rest of the pipeline safely ignores filtering
            outlier_filtering_on_train = False
            outlier_filtering_on_test = False
        elif isinstance(sqi_scores, str):
            print(f"[INFO] Calculating SQI using method: '{sqi_scores}'")
            sqi_scores = np.array(_compute_sqi(x, method=sqi_scores))
        elif isinstance(sqi_scores, (list, np.ndarray)):
            sqi_scores = np.array(sqi_scores)
        else:
            raise TypeError("[ERROR] sqi_scores must be a string, array, or None.")
    else:
        sqi_scores = None

    # ====================================================
    # 2. PRE-SPLIT CLEANUP (Ensures stratify doesn't crash)
    # ====================================================
    unique_classes, counts = np.unique(y, return_counts=True)
    valid_classes = unique_classes[counts >= 2] # Need at least 2 beats to split Train/Test
    valid_mask = np.isin(y, valid_classes)
    
    x, y = x[valid_mask], y[valid_mask]
    if sqi_scores is not None: 
        sqi_scores = sqi_scores[valid_mask]

    # ====================================================
    # 3. SPLIT DATA & SQI SCORES
    # ====================================================
    train_test_partitions = (
        _split_closed_set_samples(
            x,
            y,
            holdout_split=test_split,
            seed=seed,
            aligned_values=sqi_scores,
        )
    )

    (
        X_train,
        y_train,
        sqi_train,
    ) = train_test_partitions[
        "retained"
    ]

    (
        X_test,
        y_test,
        sqi_test,
    ) = train_test_partitions[
        "holdout"
    ]

    # ====================================================
    # 4. APPLY FILTERS INDEPENDENTLY
    # ====================================================
    if outlier_filtering_on_train and sqi_train is not None:
        print("\n[INFO] Filtering Train Set (Enrollment)...")
        X_train, y_train = _apply_outlier_filter(
            X_train, y_train, sqi_train, absolute_threshold=sqi_threshold, keep_percentage=sqi_keep_pct
        )

    if outlier_filtering_on_test and sqi_test is not None:
        print("\n[INFO] Filtering Test Set (Probes)...")
        X_test, y_test = _apply_outlier_filter(
            X_test, y_test, sqi_test, absolute_threshold=sqi_threshold, keep_percentage=sqi_keep_pct, apply_subject_ranking=False
        )

    # ====================================================
    # 5. CLASS SYNCHRONIZATION & ENCODING
    # ====================================================
    valid_train_classes = np.unique(y_train)
    original_classes = np.unique(y) # 'y' holds the original classes before the split/filter
    
    # Find classes that existed before filtering but are now completely gone
    dropped_classes = np.setdiff1d(original_classes, valid_train_classes)
    
    if len(dropped_classes) > 0:
        print(f"[WARN] Aggressive filtering completely removed {len(dropped_classes)} subjects: {dropped_classes.tolist()}")
    
    # If the filter deleted a class entirely from Train, we MUST drop it from Test
    test_mask = np.isin(y_test, valid_train_classes)
    orphaned_test_beats = len(y_test) - np.sum(test_mask)
    
    if orphaned_test_beats > 0:
        print(f"[WARN] Dropping {orphaned_test_beats} Test beats because their corresponding Train subjects were filtered out.")

    X_test, y_test = X_test[test_mask], y_test[test_mask]

    if len(valid_train_classes) < 2:
        raise ValueError("[ERROR] Data filtering was too aggressive. Not enough subjects left to continue!")

    # Encode labels safely based ONLY on surviving Train classes
    y_train_enc, classes = _encode_labels(y_train)
    label_map = {c: i for i, c in enumerate(classes)}
    y_test_enc = np.array([label_map[l] for l in y_test])

    # ====================================================
    # 6. RESUME STANDARD PIPELINE
    # ====================================================
    validation_partitions = (
        _split_closed_set_samples(
            X_train,
            y_train_enc,
            holdout_split=val_split,
            seed=seed,
        )
    )

    (
        X_tr,
        y_tr,
        _,
    ) = validation_partitions[
        "retained"
    ]

    (
        X_val,
        y_val,
        _,
    ) = validation_partitions[
        "holdout"
    ]

    if X_val is not None:
        val_loader = _make_loader(
            X_val,
            y_val,
            batch_size,
            shuffle=False,
        )
    else:
        val_loader = None

    print(
        "Data Split: "
        f"Train={len(X_tr)}, "
        f"Val={len(X_val) if X_val is not None else 0}, "
        f"Test={len(X_test)}"
    )

    X_tr, y_tr = (
        _augment_training_partition(
            X_tr,
            y_tr,
            augmentation_config,
            seed,
        )
    )

    train_loader = _make_loader(
        X_tr,
        y_tr,
        batch_size,
        shuffle=True,
    )
    test_loader = _make_loader(X_test, y_test_enc, batch_size, shuffle=False)
 
    # 3. Train (Always start with Softmax training)
    _record_runtime_stage(
        "Partition Preparation",
        partition_stage_started,
    )

    model = model_class(in_channels=_detect_channels(x), num_classes=len(classes), include_top=True).to(device)

    hyperparams.update(
        _summarize_model_complexity(model)
    )

    if intelligent_weight_loading:
        from utils import CacheManager
        cache = CacheManager(
            base_dir=getattr(
                loader,
                "cache_dir",
                DEFAULT_CACHE_DIR,
            )
        )
        train_config = {
        "training_regime": "intra_session_closed_set",
        "model": model_class.__name__,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "val_split": val_split,
        "seed": seed,
        "outlier_train": outlier_filtering_on_train,
        "sqi_thresh": sqi_threshold,
        "classes": len(classes),
        "data_shape": X_tr.shape,
        "augmentation": augmentation_config,
        }

        train_config = _build_weight_cache_config(
            loader,
            train_config,
            training_samples=X_tr,
            training_labels=y_tr,
        )
        
        cached_model, uid = _timed_runtime_call(
            "Weight Cache Read",
            cache.get_weight_cache,
            train_config,
            model,
            device,
        )
        if cached_model:
            print(f"\n[INFO] Loaded pre-trained weights (Hash: {uid}). Skipping training!")
            model = cached_model
        else:
            print(f"\n[INFO] Training new model (Hash: {uid})...")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
            model = _run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs)
            _timed_runtime_call(
                "Weight Cache Write",
                cache.save_weight_cache,
                model,
                train_config,
                uid,
            )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
        model = _run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs)

    # 4. Evaluation Logic
    if not use_template:
        # --- PATH A: Standard Softmax ---
        model.eval()
        all_probs, all_trues = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                probs = torch.softmax(model(xb), dim=1)
                all_probs.append(probs.cpu().numpy())
                all_trues.append(yb.cpu().numpy())
        final_scores = np.vstack(all_probs)
        final_labels = np.concatenate(all_trues)
        
    else:
        # --- PATH B: Template Matching ---
        print(f"[INFO] Building Templates using '{template_fusion_method}' (Beats: {template_size or 'All'})...")
        
        # Switch to Feature Extractor
        model.include_top = False 
        
        # Extract Embeddings from TRAIN set (Enrollment)
        train_extract_loader = _make_loader(X_train, y_train_enc, batch_size, shuffle=False)
        train_emb, train_lab = _get_embeddings(model, train_extract_loader, device)
        
        # Create Templates
        templates, temp_labels = _create_templates(
            train_emb, train_lab, method=template_fusion_method, max_beats=template_size
        )
        
        # Extract Embeddings from TEST set (Probe)
        test_emb, test_lab = _get_embeddings(model, test_loader, device)
        
        # Matching
        raw_scores = _compute_score_matrix(test_emb, templates, method=matching_method)
            
        # Required if template_fusion_method='none' leaves multiple templates per subject
        scores = np.full((len(test_emb), len(classes)), -np.inf)
        for class_idx in range(len(classes)):
            gallery_mask = (temp_labels == class_idx)
            if np.any(gallery_mask):
                scores[:, class_idx] = np.max(raw_scores[:, gallery_mask], axis=1)
                
        final_scores = scores
        final_labels = test_lab
        
        # Restore model
        model.include_top = True

    # ====================================================
    # 5. APPLY SCORE-LEVEL FUSION (If requested)
    # ====================================================
    final_scores, final_labels = _apply_score_fusion(
        final_scores, final_labels, fusion_size=probe_fusion_size
    )

    if visualize:
        viz = Visualizer()
        preds = np.argmax(final_scores, axis=1)
        viz.plot_confusion_matrix(final_labels, preds, normalize=True)

    rank1, rank5 = _compute_metrics_identification(final_scores, final_labels)

    evaluation_artifacts = (
        _build_identification_curve_artifacts(
            final_scores,
            final_labels,
        )
    )

    _record_evaluation_artifact(
        evaluation_artifacts
    )

    # Update hyperparams dictionary dynamically
    hyperparams['epochs'] = f"{epochs} (stopped at {model.actual_epochs})" if model.actual_epochs < epochs else epochs

    data_stats = {
        "Total Subjects": len(classes),
        "Train Samples": len(X_tr),
        "Validation Samples": len(X_val) if X_val is not None else 0,
        "Test (Probe) Samples": len(X_test),
    }

    if _return_stats:
        return (rank1, rank5), data_stats, hyperparams

    if save_results_and_settings:
        _log_experiment_results(
            task_title,
            {
                "Rank-1 Accuracy": rank1,
                "Rank-5 Accuracy": rank5,
            },
            data_stats,
            hyperparams,
            loader,
            evaluation_artifacts=evaluation_artifacts,
        )

    return rank1, rank5

# =============================================================================
# TASK 2: CLOSED-SET VERIFICATION
# =============================================================================
def run_closed_set_verification(x, y, model_class, epochs=150, batch_size=256, lr=1e-3,
                                test_split=0.2, val_split=0.0, num_pairs=10000, 
                                sampling_mode="all", seed=42, device=None, visualize=False,
                                use_template=False, template_fusion_method='mean', 
                                template_size=None, matching_method='cosine',
                                outlier_filtering_on_train=False, outlier_filtering_on_test=False, 
                                sqi_scores=None, sqi_threshold=0.05, sqi_keep_pct=0.8, 
                                use_deployment_evaluation=False, target_fars=None,
                                save_results_and_settings=False, 
                                loader=None, n_runs=1, _return_stats=False,
                                intelligent_weight_loading=True,
                                augmentation_config=None):
    """
    Standard Closed-Set Verification Pipeline (Intra-session).
    Determines "Is this person who they claim to be?" (1:1 matching) for subjects known to the model.

    Args:
        x (np.ndarray): Input ECG signals or features.
        y (np.ndarray): Subject class labels corresponding to x.
        model_class (nn.Module): The PyTorch model architecture class to instantiate.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per training batch.
        lr (float): Learning rate for the Adam optimizer.
        test_split (float): Fraction of the data to hold out for testing (0.0 to 1.0).
        val_split (float): Fraction of the training data to use for early stopping.
        num_pairs (int): Total number of Genuine and Impostor pairs to generate for evaluation.
        sampling_mode (str): Logic used to pair beats together.
            Options: ['all', 'balanced', 'random']
        seed (int): Random seed for reproducibility.
        device (str): Computation device ('cuda', 'cpu', or 'auto').
        visualize (bool): If True, generates t-SNE scatter plots of the test embeddings.
        use_template (bool): 
            - False: Evaluates raw feature space (Unseen test beats paired vs other Unseen test beats).
            - True: Simulates real-world authentication (Test probes paired against Train templates).
        template_fusion_method (str): Logic used to create the subject templates.
            Options: ['mean', 'median', 'trimmed_mean', 'representative',
            'soft_centrality', 'geometric_median', 'none']
        template_size (int, optional): Number of beats used to form the template. None uses all available.
        matching_method (str): Distance/Similarity metric used to score the pairs.
            Options: ['cosine', 'euclidean', 'manhattan', 'correlation']
        outlier_filtering_on_train (bool): If True, filters noisy beats from the Training set.
        outlier_filtering_on_test (bool): If True, filters noisy beats from the Test set.
        sqi_scores (str or np.ndarray): SQI calculation method (e.g., 'kurtosis') or pre-computed array.
        sqi_threshold (float): Absolute minimum SQI score required to keep a beat (0.0 to 1.0).
        sqi_keep_pct (float): Top percentage of beats to keep per subject after filtering.
        use_deployment_evaluation (bool): If True, calculates a Global Optimal Threshold on the Validation 
                                          set first, and applies it to the Test set to simulate deployment.
        save_results_and_settings (bool): If True, logs results and parameters to a text file.
        loader (object): Dataset loader instance (used for extracting metadata for logging).
        n_runs (int): Number of independent runs (with varying seeds) for statistical validation.
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.

    Returns:
        tuple: (EER, AUC, d-prime, TAR @ 0.1% FAR)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for all four metrics.
    """

    _validate_deployment_evaluation(
        use_deployment_evaluation,
        val_split,
        "Closed-Set Verification",
    )

    # ====================================================
    # 0. Capture Hyperparameters for Logger & MULTI-RUN AGGREGATOR
    # ====================================================
    data_stats = {}
    hyperparams = {
        'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr, 
        'test_split': test_split, 'val_split': val_split, 'num_pairs': num_pairs,
        'sampling_mode': sampling_mode, 'use_template': use_template, 
        'template_fusion_method': template_fusion_method, 'template_size': template_size,
        'matching_method': matching_method,
        'outlier_filter_train': outlier_filtering_on_train, 
        'outlier_filter_test': outlier_filtering_on_test,
        'use_deployment_eval': use_deployment_evaluation
    }

    hyperparams = _add_seed_metadata(
        hyperparams,
        base_seed=seed,
        n_runs=n_runs,
    )

    # --- MULTI-RUN AGGREGATOR ---
    if n_runs > 1:
        call_args = _prepare_multi_run_arguments(locals())
        base_seed = call_args.get('seed', 42)
        results = []
        evaluation_artifact_runs = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs)...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False 
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            run_wall_clock_started = _start_runtime_stage()
            artifact_sink_token = (
                _EVALUATION_ARTIFACT_SINK.set(
                    evaluation_artifact_runs
                )
            )
            try:
                res, d_stats, h_params = run_closed_set_verification(
                    **call_args
                )
            finally:
                _EVALUATION_ARTIFACT_SINK.reset(
                    artifact_sink_token
                )
            _record_multi_run_time(
                run_index=i + 1,
                seed=call_args["seed"],
                started_at=run_wall_clock_started,
            )
            results.append(res)
            data_stats = d_stats
            hyperparams = h_params

        hyperparams = _add_seed_metadata(
            hyperparams,
            base_seed=base_seed,
            n_runs=n_runs,
        )

        per_run_results = (
            _build_per_run_results(
                results=results,
                seeds=hyperparams[
                    "run_seeds"
                ],
            )
        )

        per_run_evaluation_artifacts = (
            _build_per_run_evaluation_artifacts(
                artifacts=evaluation_artifact_runs,
                seeds=hyperparams[
                    "run_seeds"
                ],
            )
        )
                
        metrics_t = list(zip(*results))
        means, stds = [np.mean(m) for m in metrics_t], [np.std(m) for m in metrics_t]
        
        if save_results_and_settings:
            metrics_dict = {
                "EER": f"{means[0]:.4f} ± {stds[0]:.4f}", "AUC": f"{means[1]:.4f} ± {stds[1]:.4f}", 
                "d-prime": f"{means[2]:.4f} ± {stds[2]:.4f}", "TAR@0.1%FAR": f"{means[3]:.4f} ± {stds[3]:.4f}"
            }
            _log_experiment_results(
                "Closed-Set Verification",
                metrics_dict,
                data_stats,
                hyperparams,
                loader,
                per_run_results=per_run_results,
                per_run_evaluation_artifacts=per_run_evaluation_artifacts,
            )
        return tuple(zip(means, stds))
    # ----------------------------

    augmentation_config = (
        _normalize_augmentation_config(
            augmentation_config
        )
    )

    hyperparams[
        "augmentation"
    ] = augmentation_config

    _set_seed(seed); device = _get_device(device)
    partition_stage_started = _start_runtime_stage()
    task_title = "Closed-Set Verification"
    mode_str = f"Template ({template_fusion_method}, size={template_size})" if use_template else "Cloud Pairs (Test Only)"
    print(f"\n[TASK] {task_title} | Mode: {mode_str} | Match: {matching_method}")

    # ====================================================
    # 1. DYNAMIC SQI CALCULATION
    # ====================================================
    if outlier_filtering_on_train or outlier_filtering_on_test:
        if sqi_scores is None:
            print("[WARN] Filtering requested but sqi_scores is None. Skipping filtering entirely.")
            outlier_filtering_on_train = False
            outlier_filtering_on_test = False
        elif isinstance(sqi_scores, str):
            print(f"[INFO] Calculating SQI using method: '{sqi_scores}'")
            sqi_scores = np.array(_compute_sqi(x, method=sqi_scores))
        elif isinstance(sqi_scores, (list, np.ndarray)):
            sqi_scores = np.array(sqi_scores)
        else:
            raise TypeError("[ERROR] sqi_scores must be a string, array, or None.")
    else:
        sqi_scores = None

    # ====================================================
    # 2. PRE-SPLIT CLEANUP (Ensures stratify doesn't crash)
    # ====================================================
    unique_classes, counts = np.unique(y, return_counts=True)
    valid_classes = unique_classes[counts >= 2] # Need at least 2 beats to split Train/Test
    valid_mask = np.isin(y, valid_classes)
    
    x, y = x[valid_mask], y[valid_mask]
    if sqi_scores is not None: 
        sqi_scores = sqi_scores[valid_mask]

    # ====================================================
    # 3. SPLIT DATA & SQI SCORES
    # ====================================================
    train_test_partitions = (
        _split_closed_set_samples(
            x,
            y,
            holdout_split=test_split,
            seed=seed,
            aligned_values=sqi_scores,
        )
    )

    (
        X_train,
        y_train,
        sqi_train,
    ) = train_test_partitions[
        "retained"
    ]

    (
        X_test,
        y_test,
        sqi_test,
    ) = train_test_partitions[
        "holdout"
    ]

    # ====================================================
    # 4. APPLY FILTERS INDEPENDENTLY
    # ====================================================
    if outlier_filtering_on_train and sqi_train is not None:
        print("\n[INFO] Filtering Train Set (Enrollment)...")
        X_train, y_train = _apply_outlier_filter(
            X_train, y_train, sqi_train, absolute_threshold=sqi_threshold, keep_percentage=sqi_keep_pct
        )

    if outlier_filtering_on_test and sqi_test is not None:
        print("\n[INFO] Filtering Test Set (Probes)...")
        X_test, y_test = _apply_outlier_filter(
            X_test, y_test, sqi_test, absolute_threshold=sqi_threshold, keep_percentage=sqi_keep_pct, apply_subject_ranking=False
        )

    # ====================================================
    # 5. CLASS SYNCHRONIZATION & ENCODING
    # ====================================================
    valid_train_classes = np.unique(y_train)
    original_classes = np.unique(y)
    
    dropped_classes = np.setdiff1d(original_classes, valid_train_classes)
    if len(dropped_classes) > 0:
        print(f"[WARN] Aggressive filtering completely removed {len(dropped_classes)} subjects: {dropped_classes.tolist()}")
    
    test_mask = np.isin(y_test, valid_train_classes)
    orphaned_test_beats = len(y_test) - np.sum(test_mask)
    if orphaned_test_beats > 0:
        print(f"[WARN] Dropping {orphaned_test_beats} Test beats because their corresponding Train subjects were filtered out.")

    X_test, y_test = X_test[test_mask], y_test[test_mask]

    if len(valid_train_classes) < 2:
        raise ValueError("[ERROR] Data filtering was too aggressive. Not enough subjects left to continue!")

    y_train_enc, classes = _encode_labels(y_train)
    label_map = {c: i for i, c in enumerate(classes)}
    y_test_enc = np.array([label_map[l] for l in y_test])

    # ====================================================
    # 6. RESUME STANDARD PIPELINE
    # ====================================================
    validation_partitions = (
        _split_closed_set_samples(
            X_train,
            y_train_enc,
            holdout_split=val_split,
            seed=seed,
        )
    )

    (
        X_tr,
        y_tr,
        _,
    ) = validation_partitions[
        "retained"
    ]

    (
        X_val,
        y_val,
        _,
    ) = validation_partitions[
        "holdout"
    ]

    if X_val is not None:
        val_loader = _make_loader(
            X_val,
            y_val,
            batch_size,
            shuffle=False,
        )
    else:
        val_loader = None

    print(
        "Data Split: "
        f"Train={len(X_tr)}, "
        f"Val={len(X_val) if X_val is not None else 0}, "
        f"Test={len(X_test)}"
    )

    X_tr, y_tr = (
        _augment_training_partition(
            X_tr,
            y_tr,
            augmentation_config,
            seed,
        )
    )

    train_loader = _make_loader(
        X_tr,
        y_tr,
        batch_size,
        shuffle=True,
    )
    test_loader = _make_loader(X_test, y_test_enc, batch_size, shuffle=False)
    
    # 7. Train Model
    _record_runtime_stage(
        "Partition Preparation",
        partition_stage_started,
    )

    model = model_class(in_channels=_detect_channels(x), num_classes=len(classes), include_top=True).to(device)

    hyperparams.update(
        _summarize_model_complexity(model)
    )

    if intelligent_weight_loading:
        from utils import CacheManager
        cache = CacheManager(
            base_dir=getattr(
                loader,
                "cache_dir",
                DEFAULT_CACHE_DIR,
            )
        )
        train_config = {
            "training_regime": "intra_session_closed_set",
            "model": model_class.__name__, "epochs": epochs, "batch_size": batch_size, "lr": lr, 
            "val_split": val_split, "seed": seed, "outlier_train": outlier_filtering_on_train, 
            "sqi_thresh": sqi_threshold, "classes": len(classes), "data_shape": X_tr.shape,
            "augmentation": augmentation_config,
        }

        train_config = _build_weight_cache_config(
            loader,
            train_config,
            training_samples=X_tr,
            training_labels=y_tr,
        )
        
        cached_model, uid = _timed_runtime_call(
            "Weight Cache Read",
            cache.get_weight_cache,
            train_config,
            model,
            device,
        )
        if cached_model:
            print(f"\n[INFO] Loaded pre-trained weights (Hash: {uid}). Skipping training!")
            model = cached_model
        else:
            print(f"\n[INFO] Training new model (Hash: {uid})...")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
            model = _run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs)
            _timed_runtime_call(
                "Weight Cache Write",
                cache.save_weight_cache,
                model,
                train_config,
                uid,
            )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        model = _run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs)
        
    # Switch model to feature extractor
    model.include_top = False

    # ====================================================
    # 8. MODEL CALIBRATION (Optional)
    # ====================================================
    if use_deployment_evaluation:
        print("\n[INFO] --- DEPLOYMENT THRESHOLD CALIBRATION ---")
        calib_loader = val_loader
        calib_name = "Validation"
        
        print(f"[INFO] Extracting features for Calibration ({calib_name} Set)...")
        calib_emb, calib_lab = _get_embeddings(model, calib_loader, device)
        
        print(f"[INFO] Generating Calibration Pairs to find Global Threshold...")
        calib_scores, calib_pair_labels = _generate_pairs(
            embeddings1=calib_emb, labels1=calib_lab, embeddings2=None, labels2=None,
            num_pairs=num_pairs, sampling_mode=sampling_mode, matching_method=matching_method
        )
        global_threshold = _find_optimal_threshold(calib_scores, calib_pair_labels)
        print(f"[INFO] Optimal Global Threshold Found: {global_threshold:.4f}")
    
    # Extract Test Embeddings (Probes)
    test_emb, test_lab = _get_embeddings(model, test_loader, device)

    # ====================================================
    # 9. EVALUATION STRATEGY
    # ====================================================
    if not use_template:
        # STRATEGY A: Test vs Test (Intra-session unseen evaluation)
        print(f"[INFO] Bypassing Templates. Generating pairs exclusively from Test split...")
        scores, labels_pair = _generate_pairs(
            embeddings1=test_emb, 
            labels1=test_lab, 
            embeddings2=None, # None forces Test vs Test matching
            labels2=None, 
            num_pairs=num_pairs, 
            sampling_mode=sampling_mode, 
            matching_method=matching_method
        )
    else:
        # STRATEGY B: Test vs Train Templates (Authentication Simulation)
        print(f"[INFO] Building Enrollment Templates from Train split...")
        # [FIX]: Use y_train_enc to match the neural network encoding output
        train_extract_loader = _make_loader(X_train, y_train_enc, batch_size, shuffle=False)
        train_emb, train_lab = _get_embeddings(model, train_extract_loader, device)
        
        templates, temp_labels = _create_templates(
            train_emb, train_lab, method=template_fusion_method, max_beats=template_size
        )
        
        scores, labels_pair = _generate_pairs(
            embeddings1=test_emb, # Probes
            labels1=test_lab, 
            embeddings2=templates, # Enrollment
            labels2=temp_labels, 
            num_pairs=num_pairs, 
            sampling_mode=sampling_mode, 
            matching_method=matching_method
        )
        
    if visualize:
        viz = Visualizer()
        viz.plot_embeddings(test_emb, test_lab, title="Verification Test Embeddings (T-SNE)")

    if use_deployment_evaluation:
        _evaluate_with_global_threshold(scores, labels_pair, global_threshold)

    eer, auc_val, dprime, tar = _compute_metrics_verification(scores, labels_pair)

    evaluation_artifacts = (
        _build_verification_curve_artifacts(
            scores,
            labels_pair,
            target_fars=target_fars,
        )
    )

    _record_evaluation_artifact(
        evaluation_artifacts
    )

    # Update hyperparams dictionary dynamically
    hyperparams['epochs'] = f"{epochs} (stopped at {model.actual_epochs})" if model.actual_epochs < epochs else epochs

    # ====================================================
    # 10. SAVE RESULTS
    # ====================================================
    data_stats = {
        "Total Subjects": len(classes),
        "Train Samples": len(X_tr),
        "Validation Samples": (
            len(X_val)
            if X_val is not None
            else 0
        ),
        "Test (Probe) Samples": len(X_test),
    }

    data_stats.update(
        _get_verification_pair_statistics(
            labels_pair,
            target_far=0.001,
        )
    )

    if _return_stats:
        return (
            eer,
            auc_val,
            dprime,
            tar,
        ), data_stats, hyperparams

    if save_results_and_settings:
        _log_experiment_results(
            task_title,
            {
                "EER": eer,
                "AUC": auc_val,
                "d-prime": dprime,
                "TAR@0.1%FAR": tar,
            },
            data_stats,
            hyperparams,
            loader,
            evaluation_artifacts=evaluation_artifacts,
        )

    return eer, auc_val, dprime, tar

# =============================================================================
# TASK 3: SUBJECT-DISJOINT IDENTIFICATION (TEMPLATE MATCHING)
# =============================================================================
def run_subject_disjoint_identification(x, y, model_class, epochs=150, batch_size=256, lr=1e-3, 
                                        test_split=0.2, val_split=0.0, seed=42, device=None, 
                                        visualize=False, use_template=True, 
                                        template_fusion_method='mean', template_size=1, 
                                        matching_method='cosine', outlier_filtering_on_train=False, 
                                        outlier_filtering_on_test=False, sqi_scores=None, 
                                        sqi_threshold=0.05, sqi_keep_pct=0.8, probe_fusion_size=3, 
                                        save_results_and_settings=False, loader=None, 
                                        n_runs=1, _return_stats=False,
                                        intelligent_weight_loading=True,
                                        augmentation_config=None):
    """
    Subject-Disjoint Identification Pipeline.
    Evaluates identification performance on subjects entirely UNSEEN during the training phase.
    The model learns generalized feature representations on Subject Group A, and builds a gallery for Subject Group B.

    Args:
        x (np.ndarray): Input ECG signals or features.
        y (np.ndarray): Subject class labels corresponding to x.
        model_class (nn.Module): The PyTorch model architecture class to instantiate.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per training batch.
        lr (float): Learning rate for the Adam optimizer.
        test_split (float): Fraction of unique SUBJECTS to hold out for the disjoint test set.
        val_split (float): Fraction of training subjects to use for early stopping.
        seed (int): Random seed for reproducibility.
        device (str): Computation device ('cuda', 'cpu', or 'auto').
        visualize (bool): If True, generates t-SNE scatter plots of the unseen embeddings.
        use_template (bool): MUST be True for Subject-Disjoint identification (requires a gallery).
        template_fusion_method (str): Logic used to enroll unseen subjects into the gallery.
            Options: ['mean', 'median', 'trimmed_mean', 'representative',
            'soft_centrality', 'geometric_median', 'none']
        template_size (int): Number of chronological beats (e.g., first 5) used to form the gallery template.
        matching_method (str): Distance/Similarity metric used to search the gallery.
            Options: ['cosine', 'euclidean', 'manhattan', 'correlation']
        outlier_filtering_on_train (bool): If True, filters noisy beats from the representation learning phase.
        outlier_filtering_on_test (bool): If True, filters noisy beats before forming gallery/probes.
        sqi_scores (str or np.ndarray): SQI calculation method (e.g., 'kurtosis') or pre-computed array.
        sqi_threshold (float): Absolute minimum SQI score required to keep a beat (0.0 to 1.0).
        sqi_keep_pct (float): Top percentage of beats to keep per subject after filtering.
        probe_fusion_size (int): Number of consecutive probe beats to average before searching the gallery.
        save_results_and_settings (bool): If True, logs results and parameters to a text file.
        loader (object): Dataset loader instance (used for extracting metadata for logging).
        n_runs (int): Number of independent runs (with varying seeds) for statistical validation.
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.

    Returns:
        tuple: (Rank-1 Accuracy, Rank-5 Accuracy)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for both metrics.
    """   

    # --- ENFORCE OUR AGREED TERMINOLOGY ---
    if not use_template:
        raise ValueError(
            "[ERROR] use_template=False is invalid for Identification. "
            "Identification is a 1:N search and MUST have a defined gallery/reference set. "
            "Please set use_template=True. (If you want to evaluate raw beats without averaging, "
            "set use_template=True and template_fusion_method='none')."
        )
    # --------------------------------------
    
    if template_size is None:
        raise ValueError(
            "Subject-disjoint identification needs an explicit template_size. "
            "Enrollment and probe samples come from the same unseen subjects "
            "here, so the gallery budget decides how many beats are left to "
            "probe with and cannot be inferred from the partition. Set "
            "template_size to the number of enrollment beats per subject; the "
            "paper configurations use 1."
        )

    # ====================================================
    # 0. Capture Hyperparameters for Logger & MULTI-RUN AGGREGATOR
    # ====================================================
    data_stats = {}
    hyperparams = {
        'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr, 
        'test_split_subjects': test_split, 'val_split': val_split,
        'template_fusion_method': template_fusion_method, 'template_size': template_size, 
        'matching_method': matching_method, 'probe_fusion_size': probe_fusion_size,
        'outlier_filter_train': outlier_filtering_on_train, 'outlier_filter_test': outlier_filtering_on_test
    }

    hyperparams = _add_seed_metadata(
        hyperparams,
        base_seed=seed,
        n_runs=n_runs,
    )

    if n_runs > 1:
        call_args = _prepare_multi_run_arguments(locals())
        base_seed = call_args.get('seed', 42)
        results = []
        evaluation_artifact_runs = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs)...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False 
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            run_wall_clock_started = _start_runtime_stage()
            artifact_sink_token = (
                _EVALUATION_ARTIFACT_SINK.set(
                    evaluation_artifact_runs
                )
            )
            try:
                res, d_stats, h_params = run_subject_disjoint_identification(
                    **call_args
                )
            finally:
                _EVALUATION_ARTIFACT_SINK.reset(
                    artifact_sink_token
                )
            _record_multi_run_time(
                run_index=i + 1,
                seed=call_args["seed"],
                started_at=run_wall_clock_started,
            )
            results.append(res); data_stats = d_stats; hyperparams = h_params

        hyperparams = _add_seed_metadata(
            hyperparams,
            base_seed=base_seed,
            n_runs=n_runs,
        )

        per_run_results = (
            _build_per_run_results(
                results=results,
                seeds=hyperparams[
                    "run_seeds"
                ],
            )
        )

        per_run_evaluation_artifacts = (
            _build_per_run_evaluation_artifacts(
                artifacts=evaluation_artifact_runs,
                seeds=hyperparams[
                    "run_seeds"
                ],
            )
        )
                
        r1_mean, r1_std = np.mean([r[0] for r in results]), np.std([r[0] for r in results])
        r5_mean, r5_std = np.mean([r[1] for r in results]), np.std([r[1] for r in results])
        
        if save_results_and_settings:
            metrics_dict = {"Rank-1 Accuracy": f"{r1_mean:.4f} ± {r1_std:.4f}", "Rank-5 Accuracy": f"{r5_mean:.4f} ± {r5_std:.4f}"}
            _log_experiment_results(
                "Subject-Disjoint Identification",
                metrics_dict,
                data_stats,
                hyperparams,
                loader,
                per_run_results=per_run_results,
                per_run_evaluation_artifacts=per_run_evaluation_artifacts,
            )
        return (r1_mean, r1_std), (r5_mean, r5_std)
    # ----------------------------

    augmentation_config = (
        _normalize_augmentation_config(
            augmentation_config
        )
    )

    hyperparams[
        "augmentation"
    ] = augmentation_config

    _set_seed(seed); device = _get_device(device)
    partition_stage_started = _start_runtime_stage()
    task_title = "Subject-Disjoint Identification"
    mode_str = f"Gallery: First {template_size} beats | Fusion: {template_fusion_method}"
    print(f"\n[TASK] {task_title} | Mode: {mode_str} | Match: {matching_method}")

    # ====================================================
    # 1. DYNAMIC SQI CALCULATION
    # ====================================================
    if outlier_filtering_on_train or outlier_filtering_on_test:
        if sqi_scores is None:
            print("[WARN] Filtering requested but sqi_scores is None. Skipping filtering entirely.")
            outlier_filtering_on_train = False
            outlier_filtering_on_test = False
        elif isinstance(sqi_scores, str):
            print(f"[INFO] Calculating SQI using method: '{sqi_scores}'")
            sqi_scores = np.array(_compute_sqi(x, method=sqi_scores))
        elif isinstance(sqi_scores, (list, np.ndarray)):
            sqi_scores = np.array(sqi_scores)
        else:
            raise TypeError("[ERROR] sqi_scores must be a string, array, or None.")
    else:
        sqi_scores = None

    # ====================================================
    # 2. PRE-SPLIT CLEANUP 
    # ====================================================
    # Test subjects absolutely MUST have enough beats for Gallery + at least 1 Probe
    min_required = template_size + 1 
    unique_classes, counts = np.unique(y, return_counts=True)
    valid_classes = unique_classes[counts >= min_required]
    
    valid_mask = np.isin(y, valid_classes)
    x, y = x[valid_mask], y[valid_mask]
    if sqi_scores is not None: 
        sqi_scores = sqi_scores[valid_mask]

    # ====================================================
    # 3. SPLIT SUBJECTS (Strictly Disjoint)
    # ====================================================
    y_enc, classes = _encode_labels(
        y
    )

    unique_subjs = np.unique(
        y_enc
    )

    (
        train_subs,
        val_subs,
        test_subs,
    ) = _split_subject_cohorts(
        unique_subjs,
        test_split=test_split,
        val_split=val_split,
        seed=seed,
    )

    partitions = (
        _partition_subject_disjoint_samples(
            x,
            y_enc,
            train_subjects=train_subs,
            validation_subjects=val_subs,
            test_subjects=test_subs,
            sqi_scores=sqi_scores,
        )
    )

    (
        X_train,
        y_train,
        sqi_train,
    ) = partitions["train"]

    (
        X_val,
        y_val,
        sqi_val,
    ) = partitions["validation"]

    (
        X_test,
        y_test,
        sqi_test,
    ) = partitions["test"]

    print(
        "Subject Split: "
        f"Train={len(train_subs)}, "
        f"Val={len(val_subs)}, "
        f"Test={len(test_subs)}"
    )

    # ====================================================
    # 4. APPLY SQI FILTERS
    # ====================================================
    if outlier_filtering_on_train and sqi_scores is not None:
        print("\n[INFO] Filtering Train Set (Representation Learning)...")
        X_train, y_train = _apply_outlier_filter(X_train, y_train, sqi_train, sqi_threshold, sqi_keep_pct)
        # Note: We usually DO NOT filter the Val set to keep early stopping realistic.

    if outlier_filtering_on_test and sqi_scores is not None:
        print("\n[INFO] Filtering Test Set (Gallery & Probes)...")
        X_test, y_test = _apply_outlier_filter(
            X_test,
            y_test,
            sqi_test,
            absolute_threshold=sqi_threshold,
            keep_percentage=sqi_keep_pct,
            apply_subject_ranking=False,
        )

    # ====================================================
    # 5. POST-FILTER SYNCHRONIZATION
    # ====================================================
    # Ensure Test subjects still have enough beats AFTER filtering
    test_subjs_surviving, test_counts = np.unique(y_test, return_counts=True)
    valid_test_subs = test_subjs_surviving[test_counts >= min_required]
    
    dropped_test_subs = len(test_subs) - len(valid_test_subs)
    if dropped_test_subs > 0:
        print(f"[WARN] Dropping {dropped_test_subs} Test subjects who lost too many beats during filtering to form a Gallery+Probe.")
        
    test_survivor_mask = np.isin(y_test, valid_test_subs)
    X_test, y_test = X_test[test_survivor_mask], y_test[test_survivor_mask]
    test_subs_final = valid_test_subs
    
    if len(test_subs_final) < 2:
        raise ValueError("[ERROR] Data filtering was too aggressive. Not enough Test subjects left to evaluate!")

    # Remap Train Labels to 0..N-1 so CrossEntropy is happy
    y_train_remap, train_classes = _encode_labels(y_train)
    num_train_classes = len(train_classes)
    
    if num_train_classes < 2:
        raise ValueError("[ERROR] Too few Train subjects remaining after filtering.")

    # ====================================================
    # 6. LOADERS & CUSTOM TRAINING LOOP
    # ====================================================
    # Create a Validation split from the SEEN Training subjects
    if val_split > 0.0:
        X_tr, X_val_seen, y_tr, y_val_seen = train_test_split(
            X_train, y_train_remap, test_size=val_split, stratify=y_train_remap, random_state=seed
        )
        # Create loader ONLY if we actually made a split
        val_loader_seen = _make_loader(X_val_seen, y_val_seen, batch_size, shuffle=False)
    else:
        X_tr, y_tr = X_train, y_train_remap
        # Gracefully assign None without calling _make_loader
        val_loader_seen = None
    
    X_tr, y_tr = (
        _augment_training_partition(
            X_tr,
            y_tr,
            augmentation_config,
            seed,
        )
    )

    train_loader = _make_loader(
        X_tr,
        y_tr,
        batch_size,
        shuffle=True,
    )
    
    # This remains the UNSEEN Validation subjects loader
    val_loader_unseen = _make_loader(X_val, y_val, batch_size, shuffle=False) if X_val is not None else None
    
    test_loader = _make_loader(X_test, y_test, batch_size, shuffle=False)
    
    _record_runtime_stage(
        "Partition Preparation",
        partition_stage_started,
    )

    model = model_class(in_channels=_detect_channels(x), num_classes=num_train_classes, include_top=True).to(device)

    hyperparams.update(
        _summarize_model_complexity(model)
    )

    if intelligent_weight_loading:
        from utils import CacheManager
        cache = CacheManager(
            base_dir=getattr(
                loader,
                "cache_dir",
                DEFAULT_CACHE_DIR,
            )
        )
        train_config = {
            "training_regime": "intra_session_subject_disjoint",
            "model": model_class.__name__, "epochs": epochs, "batch_size": batch_size, "lr": lr, 
            "val_split": val_split, "seed": seed, "outlier_train": outlier_filtering_on_train, 
            "sqi_thresh": sqi_threshold, "classes": num_train_classes, "data_shape": X_tr.shape,
            "augmentation": augmentation_config,
            "matching_method": matching_method  # Affects early stopping EER!
        }

        train_config = _build_weight_cache_config(
            loader,
            train_config,
            training_samples=X_tr,
            training_labels=y_tr,
        )
        
        cached_model, uid = _timed_runtime_call(
            "Weight Cache Read",
            cache.get_weight_cache,
            train_config,
            model,
            device,
        )
        if cached_model:
            print(f"\n[INFO] Loaded pre-trained weights (Hash: {uid}). Skipping training!")
            model = cached_model
        else:
            print(f"\n[INFO] Training new Subject-Disjoint model (Hash: {uid})...")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
            model = _run_train_loop_unseen_subjects(
                model=model, train_loader=train_loader, val_loader_seen=val_loader_seen, 
                val_loader_unseen=val_loader_unseen, optimizer=optimizer, criterion=criterion, 
                device=device, epochs=epochs, matching_method=matching_method, patience=40, lr_patience=15
            )
            _timed_runtime_call(
                "Weight Cache Write",
                cache.save_weight_cache,
                model,
                train_config,
                uid,
            )
    
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Call the updated custom loop passing both validation loaders of seen and unseen subjects!
        model = _run_train_loop_unseen_subjects(
            model, train_loader, val_loader_seen, val_loader_unseen, optimizer, criterion, device, 
            epochs, matching_method=matching_method, patience=40, lr_patience=15
        )

    # ====================================================
    # 7. FINAL INFERENCE ON UNSEEN TEST SUBJECTS
    # ====================================================
    model.include_top = False
    test_emb, test_lab = _get_embeddings(model, test_loader, device)

    print(
        "[INFO] Splitting Test Data: "
        f"First {template_size} beats = Gallery, "
        "Rest = Probe"
    )

    # Map disjoint test subject IDs to 0..N_test-1
    # for the identification metric array.
    test_sub_map = {
        subject: index
        for index, subject in enumerate(
            test_subs_final
        )
    }

    enrollment_probe_partitions = (
        _split_enrollment_probe_embeddings(
            test_emb,
            test_lab,
            subjects=test_subs_final,
            template_size=template_size,
        )
    )

    (
        emb_enroll,
        lab_enroll,
    ) = enrollment_probe_partitions[
        "enrollment"
    ]

    (
        emb_probe,
        lab_probe,
    ) = enrollment_probe_partitions[
        "probe"
    ]
    
    # 8. Apply Template Fusion Strategy to the Gallery
    gallery_emb, gallery_lab = _create_templates(
        emb_enroll, lab_enroll, method=template_fusion_method, max_beats=None
    )
    
    # 9. Generate Score Matrix for Rank-N Evaluation
    probe_mapped = np.array([test_sub_map[l] for l in lab_probe])
    
    raw_scores = _compute_score_matrix(emb_probe, gallery_emb, method=matching_method)
    scores = np.full((len(emb_probe), len(test_subs_final)), -np.inf)
    
    for class_idx, sub in enumerate(test_subs_final):
        gallery_mask = (gallery_lab == sub)
        if np.any(gallery_mask):
            scores[:, class_idx] = np.max(raw_scores[:, gallery_mask], axis=1)

    # ====================================================
    # 10. APPLY SCORE-LEVEL FUSION
    # ====================================================
    final_scores, final_labels = _apply_score_fusion(
        scores, probe_mapped, fusion_size=probe_fusion_size
    )

    if visualize:
        # Visualizing original un-fused test embeddings
        viz = Visualizer()
        viz.plot_embeddings(test_emb, test_lab, title="Unseen Subject Embeddings (T-SNE)")

    rank1, rank5 = _compute_metrics_identification(final_scores, final_labels)

    evaluation_artifacts = (
        _build_identification_curve_artifacts(
            final_scores,
            final_labels,
        )
    )

    _record_evaluation_artifact(
        evaluation_artifacts
    )

    data_stats = {
        "Train Subjects": len(train_subs),
        "Train Samples": len(X_train),
        "Validation Subjects": (
            len(val_subs)
            if val_subs is not None
            else 0
        ),
        "Validation Samples": (
            len(X_val)
            if X_val is not None
            else 0
        ),
        "Test Subjects": len(test_subs_final),
        "Gallery Size": len(gallery_emb),
        "Probe Samples": len(emb_probe),
    }

    if _return_stats:
        return (rank1, rank5), data_stats, hyperparams

    if save_results_and_settings:
        _log_experiment_results(
            task_title,
            {
                "Rank-1 Accuracy": rank1,
                "Rank-5 Accuracy": rank5,
            },
            data_stats,
            hyperparams,
            loader,
            evaluation_artifacts=evaluation_artifacts,
        )

    # 11. Report Identification Metrics
    return rank1, rank5

# =============================================================================
# TASK 4: SUBJECT-DISJOINT VERIFICATION
# =============================================================================
def run_subject_disjoint_verification(x, y, model_class, epochs=150, batch_size=256, lr=1e-3, 
                                      test_split=0.2, val_split=0.0, num_pairs=10000, 
                                      sampling_mode="all", seed=42, device=None, 
                                      visualize=False, use_template=False, template_fusion_method='mean', 
                                      template_size=1, matching_method='cosine',
                                      outlier_filtering_on_train=False, outlier_filtering_on_test=False, 
                                      sqi_scores=None, sqi_threshold=0.05, sqi_keep_pct=0.8, 
                                      use_deployment_evaluation=False, target_fars=None,
                                      save_results_and_settings=False, 
                                      loader=None, n_runs=1, _return_stats=False,
                                      intelligent_weight_loading=True,
                                      augmentation_config=None):
    """
    Subject-Disjoint Verification Pipeline (Subject-Disjoint 1:1 Matching).
    Tests the system's ability to verify the identity of completely new users.
    The model is trained on Subject Group A and evaluated via pairs constructed from Subject Group B.

    Args:
        x (np.ndarray): Input ECG signals or features.
        y (np.ndarray): Subject class labels corresponding to x.
        model_class (nn.Module): The PyTorch model architecture class to instantiate.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per training batch.
        lr (float): Learning rate for the Adam optimizer.
        test_split (float): Fraction of unique SUBJECTS to hold out for the disjoint test set.
        val_split (float): Fraction of training subjects to use for early stopping.
        num_pairs (int): Total number of Genuine and Impostor pairs to generate.
        sampling_mode (str): Logic used to pair beats together.
            Options: ['all', 'balanced', 'random']
        seed (int): Random seed for reproducibility.
        device (str): Computation device ('cuda', 'cpu', or 'auto').
        visualize (bool): If True, generates t-SNE scatter plots of the unseen embeddings.
        use_template (bool): 
            - False: "Cloud-based" matching (Random pairs formed entirely within the unseen Test group).
            - True: "Authentication" simulation (Unseen subjects' later beats matched against their initial beats).
        template_fusion_method (str): Logic used to create templates if use_template is True.
            Options: ['mean', 'median', 'trimmed_mean', 'representative',
            'soft_centrality', 'geometric_median', 'none']
        template_size (int): Number of initial beats to form the enrollment template for unseen subjects.
        matching_method (str): Distance/Similarity metric used to score the pairs.
            Options: ['cosine', 'euclidean', 'manhattan', 'correlation']
        outlier_filtering_on_train (bool): If True, filters noisy beats from the representation learning phase.
        outlier_filtering_on_test (bool): If True, filters noisy beats from the verification test subjects.
        sqi_scores (str or np.ndarray): SQI calculation method (e.g., 'kurtosis') or pre-computed array.
        sqi_threshold (float): Absolute minimum SQI score required to keep a beat (0.0 to 1.0).
        sqi_keep_pct (float): Top percentage of beats to keep per subject after filtering.
        use_deployment_evaluation (bool): Uses an unseen validation group to find a Global Threshold.
        save_results_and_settings (bool): If True, logs results and parameters to a text file.
        loader (object): Dataset loader instance (used for extracting metadata for logging).
        n_runs (int): Number of independent runs (with varying seeds) for statistical validation.
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.

    Returns:
        tuple: (EER, AUC, d-prime, TAR @ 0.1% FAR)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for all four metrics.
    """

    _validate_deployment_evaluation(
        use_deployment_evaluation,
        val_split,
        "Subject-Disjoint Verification",
    )

    # ====================================================
    # 0. Capture Hyperparameters for Logger & MULTI-RUN AGGREGATOR
    # ====================================================
    data_stats = {}
    hyperparams = {
        'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr, 
        'test_split_subjects': test_split, 'val_split': val_split, 'num_pairs': num_pairs,
        'use_template': use_template, 'template_fusion_method': template_fusion_method, 
        'template_size': template_size, 'matching_method': matching_method, 
        'outlier_filter_train': outlier_filtering_on_train, 'outlier_filter_test': outlier_filtering_on_test
    }

    hyperparams = _add_seed_metadata(
        hyperparams,
        base_seed=seed,
        n_runs=n_runs,
    )

    if n_runs > 1:
        call_args = _prepare_multi_run_arguments(locals())
        base_seed = call_args.get('seed', 42)
        results = []
        evaluation_artifact_runs = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs)...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False 
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            run_wall_clock_started = _start_runtime_stage()
            artifact_sink_token = (
                _EVALUATION_ARTIFACT_SINK.set(
                    evaluation_artifact_runs
                )
            )
            try:
                res, d_stats, h_params = run_subject_disjoint_verification(
                    **call_args
                )
            finally:
                _EVALUATION_ARTIFACT_SINK.reset(
                    artifact_sink_token
                )
            _record_multi_run_time(
                run_index=i + 1,
                seed=call_args["seed"],
                started_at=run_wall_clock_started,
            )
            results.append(res); data_stats = d_stats; hyperparams = h_params

        hyperparams = _add_seed_metadata(
            hyperparams,
            base_seed=base_seed,
            n_runs=n_runs,
        )

        per_run_results = (
            _build_per_run_results(
                results=results,
                seeds=hyperparams[
                    "run_seeds"
                ],
            )
        )

        per_run_evaluation_artifacts = (
            _build_per_run_evaluation_artifacts(
                artifacts=evaluation_artifact_runs,
                seeds=hyperparams[
                    "run_seeds"
                ],
            )
        )
                
        metrics_t = list(zip(*results))
        means, stds = [np.mean(m) for m in metrics_t], [np.std(m) for m in metrics_t]
        
        if save_results_and_settings:
            metrics_dict = {
                "EER": f"{means[0]:.4f} ± {stds[0]:.4f}", "AUC": f"{means[1]:.4f} ± {stds[1]:.4f}", 
                "d-prime": f"{means[2]:.4f} ± {stds[2]:.4f}", "TAR@0.1%FAR": f"{means[3]:.4f} ± {stds[3]:.4f}"
            }
            _log_experiment_results(
                "Subject-Disjoint Verification",
                metrics_dict,
                data_stats,
                hyperparams,
                loader,
                per_run_results=per_run_results,
                per_run_evaluation_artifacts=per_run_evaluation_artifacts,
            )
        return tuple(zip(means, stds))
    # ----------------------------

    if use_template and template_size is None:
        raise ValueError(
            "Subject-disjoint verification needs an explicit template_size. "
            "Enrollment and probe samples come from the same unseen subjects "
            "here, so the gallery budget decides how many beats are left to "
            "probe with and cannot be inferred from the partition. Set "
            "template_size to the number of enrollment beats per subject; the "
            "paper configurations use 1."
        )

    augmentation_config = (
        _normalize_augmentation_config(
            augmentation_config
        )
    )

    hyperparams[
        "augmentation"
    ] = augmentation_config

    _set_seed(seed); device = _get_device(device)
    partition_stage_started = _start_runtime_stage()
    task_title = "Subject-Disjoint Verification"        
    mode_str = f"Template ({template_fusion_method}, First {template_size})" if use_template else "Cloud Pairs (Test Only)"
    print(f"\n[TASK] {task_title} | Mode: {mode_str} | Match: {matching_method}")

    # ====================================================
    # 1. DYNAMIC SQI CALCULATION
    # ====================================================
    if outlier_filtering_on_train or outlier_filtering_on_test:
        if sqi_scores is None:
            print("[WARN] Filtering requested but sqi_scores is None. Skipping filtering entirely.")
            outlier_filtering_on_train = False
            outlier_filtering_on_test = False
        elif isinstance(sqi_scores, str):
            print(f"[INFO] Calculating SQI using method: '{sqi_scores}'")
            sqi_scores = np.array(_compute_sqi(x, method=sqi_scores))
        elif isinstance(sqi_scores, (list, np.ndarray)):
            sqi_scores = np.array(sqi_scores)
        else:
            raise TypeError("[ERROR] sqi_scores must be a string, array, or None.")
    else:
        sqi_scores = None

    # ====================================================
    # 2. PRE-SPLIT CLEANUP 
    # ====================================================
    # If using templates, subjects need enough beats for Gallery + Probes.
    # If not using templates, they just need at least 2 beats to form pairs.
    min_required = (template_size + 1) if use_template else 2
    
    unique_classes, counts = np.unique(y, return_counts=True)
    valid_classes = unique_classes[counts >= min_required]
    
    valid_mask = np.isin(y, valid_classes)
    x, y = x[valid_mask], y[valid_mask]
    if sqi_scores is not None: 
        sqi_scores = sqi_scores[valid_mask]

    # ====================================================
    # 3. SPLIT SUBJECTS (Strictly Disjoint)
    # ====================================================
    y_enc, classes = _encode_labels(
        y
    )

    unique_subjs = np.unique(
        y_enc
    )

    (
        train_subs,
        val_subs,
        test_subs,
    ) = _split_subject_cohorts(
        unique_subjs,
        test_split=test_split,
        val_split=val_split,
        seed=seed,
    )

    partitions = (
        _partition_subject_disjoint_samples(
            x,
            y_enc,
            train_subjects=train_subs,
            validation_subjects=val_subs,
            test_subjects=test_subs,
            sqi_scores=sqi_scores,
        )
    )

    (
        X_train,
        y_train,
        sqi_train,
    ) = partitions["train"]

    (
        X_val,
        y_val,
        sqi_val,
    ) = partitions["validation"]

    (
        X_test,
        y_test,
        sqi_test,
    ) = partitions["test"]

    print(
        "Subject Split: "
        f"Train={len(train_subs)}, "
        f"Val={len(val_subs)}, "
        f"Test={len(test_subs)}"
    )

    # ====================================================
    # 4. APPLY SQI FILTERS
    # ====================================================
    if outlier_filtering_on_train and sqi_scores is not None:
        print("\n[INFO] Filtering Train Set (Representation Learning)...")
        X_train, y_train = _apply_outlier_filter(X_train, y_train, sqi_train, sqi_threshold, sqi_keep_pct)

    if outlier_filtering_on_test and sqi_scores is not None:
        print("\n[INFO] Filtering Test Set (Probes)...")
        X_test, y_test = _apply_outlier_filter(
            X_test,
            y_test,
            sqi_test,
            absolute_threshold=sqi_threshold,
            keep_percentage=sqi_keep_pct,
            apply_subject_ranking=False,
        )

    # ====================================================
    # 5. POST-FILTER SYNCHRONIZATION
    # ====================================================
    test_subjs_surviving, test_counts = np.unique(y_test, return_counts=True)
    valid_test_subs = test_subjs_surviving[test_counts >= min_required]
    
    dropped_test_subs = len(test_subs) - len(valid_test_subs)
    if dropped_test_subs > 0:
        print(f"[WARN] Dropping {dropped_test_subs} Test subjects who lost too many beats during filtering.")
        
    test_survivor_mask = np.isin(y_test, valid_test_subs)
    X_test, y_test = X_test[test_survivor_mask], y_test[test_survivor_mask]
    test_subs_final = valid_test_subs
    
    if len(test_subs_final) < 2:
        raise ValueError("[ERROR] Data filtering was too aggressive. Not enough Test subjects left to evaluate!")

    y_train_remap, train_classes = _encode_labels(y_train)
    num_train_classes = len(train_classes)
    
    if num_train_classes < 2:
        raise ValueError("[ERROR] Too few Train subjects remaining after filtering.")

# ====================================================
    # 6. LOADERS & CUSTOM TRAINING LOOP
    # ====================================================
    # Create a Validation split from the SEEN Training subjects
    # This gives us the smooth Cross-Entropy loss anchor for the composite metric
    if val_split > 0.0:
        X_tr, X_val_seen, y_tr, y_val_seen = train_test_split(
            X_train, y_train_remap, test_size=val_split, stratify=y_train_remap, random_state=seed
        )
        # Create loader ONLY if we actually made a split
        val_loader_seen = _make_loader(X_val_seen, y_val_seen, batch_size, shuffle=False)
    else:
        X_tr, y_tr = X_train, y_train_remap
        # Gracefully assign None without calling _make_loader
        val_loader_seen = None
    
    X_tr, y_tr = (
        _augment_training_partition(
            X_tr,
            y_tr,
            augmentation_config,
            seed,
        )
    )

    train_loader = _make_loader(
        X_tr,
        y_tr,
        batch_size,
        shuffle=True,
    )
    
    # This remains the UNSEEN Validation subjects loader (used for EER)
    val_loader_unseen = _make_loader(X_val, y_val, batch_size, shuffle=False) if X_val is not None else None
    
    test_loader = _make_loader(X_test, y_test, batch_size, shuffle=False)
    
    _record_runtime_stage(
        "Partition Preparation",
        partition_stage_started,
    )

    model = model_class(in_channels=_detect_channels(x), num_classes=num_train_classes, include_top=True).to(device)

    hyperparams.update(
        _summarize_model_complexity(model)
    )

    if intelligent_weight_loading:
        from utils import CacheManager
        cache = CacheManager(
            base_dir=getattr(
                loader,
                "cache_dir",
                DEFAULT_CACHE_DIR,
            )
        )
        train_config = {
            "training_regime": "intra_session_subject_disjoint",
            "model": model_class.__name__, "epochs": epochs, "batch_size": batch_size, "lr": lr, 
            "val_split": val_split, "seed": seed, "outlier_train": outlier_filtering_on_train, 
            "sqi_thresh": sqi_threshold, "classes": num_train_classes, "data_shape": X_tr.shape,
            "augmentation": augmentation_config,
            "matching_method": matching_method  # Affects early stopping EER!
        }

        train_config = _build_weight_cache_config(
            loader,
            train_config,
            training_samples=X_tr,
            training_labels=y_tr,
        )
        
        cached_model, uid = _timed_runtime_call(
            "Weight Cache Read",
            cache.get_weight_cache,
            train_config,
            model,
            device,
        )
        if cached_model:
            print(f"\n[INFO] Loaded pre-trained weights (Hash: {uid}). Skipping training!")
            model = cached_model
        else:
            print(f"\n[INFO] Training new Subject-Disjoint model (Hash: {uid})...")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
            model = _run_train_loop_unseen_subjects(
                model=model, train_loader=train_loader, val_loader_seen=val_loader_seen, 
                val_loader_unseen=val_loader_unseen, optimizer=optimizer, criterion=criterion, 
                device=device, epochs=epochs, matching_method=matching_method, patience=40, lr_patience=15
            )
            _timed_runtime_call(
                "Weight Cache Write",
                cache.save_weight_cache,
                model,
                train_config,
                uid,
            )
    
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Single line execution using the Composite Metric loop!
        model = _run_train_loop_unseen_subjects(
            model=model, 
            train_loader=train_loader, 
            val_loader_seen=val_loader_seen, 
            val_loader_unseen=val_loader_unseen, 
            optimizer=optimizer, 
            criterion=criterion, 
            device=device, 
            epochs=epochs, 
            matching_method=matching_method, 
            patience=40,       # Max epochs to wait for composite score improvement
            lr_patience=15     # Epochs to wait before halving Learning Rate
        )

    # ====================================================
    # 7. MODEL CALIBRATION (Optional)
    # ====================================================
    model.include_top = False
    
    if use_deployment_evaluation:
        print("\n[INFO] --- DEPLOYMENT THRESHOLD CALIBRATION ---")
        calib_loader = val_loader_unseen
        calib_name = "Unseen Validation"
        
        print(f"[INFO] Extracting features for Calibration ({calib_name} Set)...")
        calib_emb, calib_lab = _get_embeddings(model, calib_loader, device)
        
        print(f"[INFO] Generating Calibration Pairs to find Global Threshold...")
        calib_scores, calib_pair_labels = _generate_pairs(
            embeddings1=calib_emb, labels1=calib_lab, embeddings2=None, labels2=None,
            num_pairs=num_pairs, sampling_mode=sampling_mode, matching_method=matching_method
        )
        global_threshold = _find_optimal_threshold(calib_scores, calib_pair_labels)
        print(f"[INFO] Optimal Global Threshold Found: {global_threshold:.4f}")
        
    # ====================================================
    # 8. FINAL INFERENCE ON UNSEEN TEST SUBJECTS
    # ====================================================
    test_emb, test_lab = _get_embeddings(model, test_loader, device)

    # 9. Evaluation Strategy
    if not use_template:
        print(f"[INFO] Bypassing Templates. Generating pairs entirely from Unseen Test Subjects...")
        scores, labels_pair = _generate_pairs(
            embeddings1=test_emb, labels1=test_lab, 
            embeddings2=None, labels2=None, 
            num_pairs=num_pairs, sampling_mode=sampling_mode, matching_method=matching_method
        )
    else:
        print(
            "[INFO] Splitting Test Data: "
            f"First {template_size} beats = Enroll, "
            "Rest = Probe"
        )

        enrollment_probe_partitions = (
            _split_enrollment_probe_embeddings(
                test_emb,
                test_lab,
                subjects=test_subs_final,
                template_size=template_size,
            )
        )

        (
            emb_enroll,
            lab_enroll,
        ) = enrollment_probe_partitions[
            "enrollment"
        ]

        (
            emb_probe,
            lab_probe,
        ) = enrollment_probe_partitions[
            "probe"
        ]
        
        templates, temp_labels = _create_templates(
            emb_enroll, lab_enroll, method=template_fusion_method, max_beats=None
        )
        
        scores, labels_pair = _generate_pairs(
            embeddings1=emb_probe, labels1=lab_probe, 
            embeddings2=templates, labels2=temp_labels, 
            num_pairs=num_pairs, sampling_mode=sampling_mode, matching_method=matching_method
        )
        
    if visualize:
        viz = Visualizer()
        viz.plot_embeddings(test_emb, test_lab, title="Unseen Subject Embeddings (T-SNE)")

    if use_deployment_evaluation:
        _evaluate_with_global_threshold(scores, labels_pair, global_threshold)

    eer, auc_val, dprime, tar = _compute_metrics_verification(scores, labels_pair)

    evaluation_artifacts = (
        _build_verification_curve_artifacts(
            scores,
            labels_pair,
            target_fars=target_fars,
        )
    )

    _record_evaluation_artifact(
        evaluation_artifacts
    )

    # Update hyperparams dictionary dynamically using the local 'ep' variable
    # hyperparams['epochs'] = f"{epochs} (stopped at {ep + 1})" if (ep + 1) < epochs else epochs
    
    data_stats = {
        "Train Subjects": len(train_subs),
        "Train Samples": len(X_train),
        "Validation Subjects": (
            len(val_subs)
            if val_subs is not None
            else 0
        ),
        "Validation Samples": (
            len(X_val)
            if X_val is not None
            else 0
        ),
        "Test Subjects": len(test_subs_final),
        "Test Pairs Evaluated": len(labels_pair),
    }

    data_stats.update(
        _get_verification_pair_statistics(
            labels_pair,
            target_far=0.001,
        )
    )

    if _return_stats:
        return (
            eer,
            auc_val,
            dprime,
            tar,
        ), data_stats, hyperparams

    if save_results_and_settings:
        _log_experiment_results(
            task_title,
            {
                "EER": eer,
                "AUC": auc_val,
                "d-prime": dprime,
                "TAR@0.1%FAR": tar,
            },
            data_stats,
            hyperparams,
            loader,
            evaluation_artifacts=evaluation_artifacts,
        )

    return eer, auc_val, dprime, tar

# =============================================================================
# TASK 5: CROSS-SESSION IDENTIFICATION
# =============================================================================
def run_cross_session_identification(x_train, y_train, x_test, y_test, model_class, epochs=150, 
                                     batch_size=256, lr=1e-3, val_split=0.0, seed=42, device=None, 
                                     visualize=False, use_template=False, template_fusion_method='mean',
                                     template_size=None, matching_method='cosine',
                                     outlier_filtering_on_train=False, outlier_filtering_on_test=False, 
                                     sqi_train=None, sqi_test=None, sqi_threshold=0.05, 
                                     sqi_keep_pct=0.8, probe_fusion_size=3, save_results_and_settings=False, 
                                     loader=None, n_runs=1, _return_stats=False,
                                     intelligent_weight_loading=True,
                                     augmentation_config=None):
    """
    Cross-Session Identification Pipeline (Temporal Robustness).
    Evaluates system robustness against physiological aging and sensor variations over time.
    Trains the model on Session 1 (Enrollment) and identifies subjects using Session 2 (Probes).

    Args:
        x_train (np.ndarray): Input ECG signals from Session 1.
        y_train (np.ndarray): Labels for Session 1.
        x_test (np.ndarray): Input ECG signals from Session 2.
        y_test (np.ndarray): Labels for Session 2.
        model_class (nn.Module): The PyTorch model architecture class to instantiate.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per training batch.
        lr (float): Learning rate for the Adam optimizer.
        val_split (float): Fraction of Session 1 data to use for early stopping.
        seed (int): Random seed for reproducibility.
        device (str): Computation device ('cuda', 'cpu', or 'auto').
        visualize (bool): If True, generates t-SNE scatter plots of the cross-session embeddings.
        use_template (bool): 
            - False: Uses the Session 1 Softmax classification weights to classify Session 2 data.
            - True: Uses Session 1 features to form a gallery, and metric-matches Session 2 probes.
        template_fusion_method (str): Logic used to create the Session 1 gallery template.
            Options: ['mean', 'median', 'trimmed_mean', 'representative',
            'soft_centrality', 'geometric_median', 'none']
        template_size (int, optional): Number of Session 1 beats used for enrollment. None uses all available.
        matching_method (str): Distance/Similarity metric for template matching.
            Options: ['cosine', 'euclidean', 'manhattan', 'correlation']
        outlier_filtering_on_train (bool): Apply SQI filtering independently to Session 1.
        outlier_filtering_on_test (bool): Apply SQI filtering independently to Session 2.
        sqi_train (str or np.ndarray): SQI calculation method or pre-computed array for Session 1.
        sqi_test (str or np.ndarray): SQI calculation method or pre-computed array for Session 2.
        sqi_threshold (float): Absolute minimum SQI score required to keep a beat (0.0 to 1.0).
        sqi_keep_pct (float): Top percentage of beats to keep per subject after filtering.
        probe_fusion_size (int): Number of consecutive Session 2 beats to average before making a decision.
        save_results_and_settings (bool): If True, logs results and parameters to a text file.
        loader (object): Dataset loader instance (used for extracting metadata for logging).
        n_runs (int): Number of independent runs (with varying seeds) for statistical validation.
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.

    Returns:
        tuple: (Rank-1 Accuracy, Rank-5 Accuracy)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for both metrics.
    """

    # ====================================================
    # 0. Capture Hyperparameters for Logger & MULTI-RUN AGGREGATOR
    # ====================================================
    data_stats = {}
    hyperparams = {
        'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr, 'use_template': use_template, 
        'template_fusion_method': template_fusion_method, 'template_size': template_size, 
        'matching_method': matching_method, 'probe_fusion_size': probe_fusion_size, 'val_split': val_split,
        'outlier_filter_train': outlier_filtering_on_train, 'outlier_filter_test': outlier_filtering_on_test
    }

    hyperparams = _add_seed_metadata(
        hyperparams,
        base_seed=seed,
        n_runs=n_runs,
    )

    if n_runs > 1:
        call_args = _prepare_multi_run_arguments(locals())
        base_seed = call_args.get('seed', 42)
        results = []
        evaluation_artifact_runs = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs)...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False 
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            run_wall_clock_started = _start_runtime_stage()
            artifact_sink_token = (
                _EVALUATION_ARTIFACT_SINK.set(
                    evaluation_artifact_runs
                )
            )
            try:
                res, d_stats, h_params = run_cross_session_identification(
                    **call_args
                )
            finally:
                _EVALUATION_ARTIFACT_SINK.reset(
                    artifact_sink_token
                )
            _record_multi_run_time(
                run_index=i + 1,
                seed=call_args["seed"],
                started_at=run_wall_clock_started,
            )
            results.append(res); data_stats = d_stats; hyperparams = h_params

        hyperparams = _add_seed_metadata(
            hyperparams,
            base_seed=base_seed,
            n_runs=n_runs,
        )

        per_run_results = (
            _build_per_run_results(
                results=results,
                seeds=hyperparams[
                    "run_seeds"
                ],
            )
        )

        per_run_evaluation_artifacts = (
            _build_per_run_evaluation_artifacts(
                artifacts=evaluation_artifact_runs,
                seeds=hyperparams[
                    "run_seeds"
                ],
            )
        )
                
        r1_mean, r1_std = np.mean([r[0] for r in results]), np.std([r[0] for r in results])
        r5_mean, r5_std = np.mean([r[1] for r in results]), np.std([r[1] for r in results])
        
        if save_results_and_settings:
            metrics_dict = {"Rank-1 Accuracy": f"{r1_mean:.4f} ± {r1_std:.4f}", "Rank-5 Accuracy": f"{r5_mean:.4f} ± {r5_std:.4f}"}
            _log_experiment_results(
                "Cross-Session Identification",
                metrics_dict,
                data_stats,
                hyperparams,
                loader,
                per_run_results=per_run_results,
                per_run_evaluation_artifacts=per_run_evaluation_artifacts,
            )
        return (r1_mean, r1_std), (r5_mean, r5_std)
    # ----------------------------

    augmentation_config = (
        _normalize_augmentation_config(
            augmentation_config
        )
    )

    hyperparams[
        "augmentation"
    ] = augmentation_config

    _set_seed(seed); device = _get_device(device)
    partition_stage_started = _start_runtime_stage()
    task_title = "Cross-Session Identification"
    mode_str = f"Template ({template_fusion_method}, size={template_size or 'All'})" if use_template else "Softmax Classifier"
    print(f"\n[TASK] {task_title} | Mode: {mode_str} | Match: {matching_method if use_template else 'N/A'}")
    
    # ====================================================
    # 1. DYNAMIC SQI CALCULATION (Independent Sessions)
    # ====================================================
    def _prepare_sqi(sqi_input, x_data, flag, name):
        if not flag: return None
        if sqi_input is None:
            print(f"[WARN] Filtering requested for {name} but sqi scores are None. Skipping {name} filtering.")
            return None
        if isinstance(sqi_input, str):
            print(f"[INFO] Calculating SQI for {name} using method: '{sqi_input}'")
            return np.array(_compute_sqi(x_data, method=sqi_input))
        if isinstance(sqi_input, (list, np.ndarray)):
            return np.array(sqi_input)
        raise TypeError(f"[ERROR] sqi_{name.lower()} must be a string, array, or None.")

    sqi_train = _prepare_sqi(sqi_train, x_train, outlier_filtering_on_train, "Train")
    sqi_test = _prepare_sqi(sqi_test, x_test, outlier_filtering_on_test, "Test")

    # ====================================================
    # 2. APPLY SQI FILTERS
    # ====================================================
    if sqi_train is not None:
        print("\n[INFO] Filtering Session 1 (Enrollment)...")
        x_train, y_train = _apply_outlier_filter(x_train, y_train, sqi_train, sqi_threshold, sqi_keep_pct)

    if sqi_test is not None:
        print("\n[INFO] Filtering Session 2 (Probes)...")
        x_test, y_test = _apply_outlier_filter(
            x_test,
            y_test,
            sqi_test,
            absolute_threshold=sqi_threshold,
            keep_percentage=sqi_keep_pct,
            apply_subject_ranking=False,
        )

    # ====================================================
    # 3. SYNCHRONISE KNOWN SUBJECTS ACROSS SESSIONS
    # ====================================================
    cross_session_partitions = (
        _partition_closed_set_cross_session_samples(
            x_train,
            y_train,
            x_test,
            y_test,
            minimum_session_1_samples=2,
            minimum_session_2_samples=1,
        )
    )

    (
        x_train_full,
        y_train_full,
    ) = cross_session_partitions[
        "session_1"
    ]

    (
        x_test_filtered,
        y_test_filtered,
    ) = cross_session_partitions[
        "session_2"
    ]

    dropped_subjects = (
        cross_session_partitions[
            "dropped_subjects"
        ]
    )

    if len(
        dropped_subjects[
            "insufficient_session_1"
        ]
    ) > 0:
        print(
            "[WARN] Dropping "
            f"{len(dropped_subjects['insufficient_session_1'])} "
            "subjects with fewer than 2 surviving "
            "Session 1 samples."
        )

    # ====================================================
    # 4. ENCODE LABELS
    # ====================================================
    # Encode Labels to 0..N-1 based strictly on the surviving training set classes
    y_train_enc, classes = _encode_labels(y_train_full)
    label_map = {c: i for i, c in enumerate(classes)}
    y_test_enc = np.array([label_map[l] for l in y_test_filtered])
    
    # ====================================================
    # 5. RESUME STANDARD PIPELINE
    # ====================================================
    # Validation Split (from Session 1)
    if val_split > 0.0:
        X_tr, X_val, y_tr, y_val = train_test_split(
            x_train_full, y_train_enc, test_size=val_split, stratify=y_train_enc, random_state=seed
        )
        val_loader = _make_loader(X_val, y_val, batch_size, shuffle=False)
        print(f"Session 1 Split: Train={len(X_tr)}, Val={len(X_val)} | Session 2 Probes={len(x_test_filtered)}")
    else:
        X_tr, y_tr = x_train_full, y_train_enc
        X_val, val_loader = None, None
        print(f"Session 1 Split: Train={len(X_tr)}, Val=0 | Session 2 Probes={len(x_test_filtered)}")

    X_tr, y_tr = (
        _augment_training_partition(
            X_tr,
            y_tr,
            augmentation_config,
            seed,
        )
    )

    train_loader = _make_loader(
        X_tr,
        y_tr,
        batch_size,
        shuffle=True,
    )
    probe_loader = _make_loader(x_test_filtered, y_test_enc, batch_size, shuffle=False)
    
    # Train Model
    _record_runtime_stage(
        "Partition Preparation",
        partition_stage_started,
    )

    model = model_class(in_channels=_detect_channels(x_train_full), num_classes=len(classes), include_top=True).to(device)

    hyperparams.update(
        _summarize_model_complexity(model)
    )

    if intelligent_weight_loading:
        from utils import CacheManager
        cache = CacheManager(
            base_dir=getattr(
                loader,
                "cache_dir",
                DEFAULT_CACHE_DIR,
            )
        )
        train_config = {
            "training_regime": "cross_session_closed_set",
            "model": model_class.__name__, "epochs": epochs, "batch_size": batch_size, "lr": lr, 
            "val_split": val_split, "seed": seed, "outlier_train": outlier_filtering_on_train, 
            "sqi_thresh": sqi_threshold, "classes": len(classes), "data_shape": X_tr.shape,
            "augmentation": augmentation_config,
        }

        train_config = _build_weight_cache_config(
            loader,
            train_config,
            training_samples=X_tr,
            training_labels=y_tr,
        )
        
        cached_model, uid = _timed_runtime_call(
            "Weight Cache Read",
            cache.get_weight_cache,
            train_config,
            model,
            device,
        )
        if cached_model:
            print(f"\n[INFO] Loaded pre-trained weights (Hash: {uid}). Skipping training!")
            model = cached_model
        else:
            print(f"\n[INFO] Training new Cross-Session model (Hash: {uid})...")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
            model = _run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs)
            _timed_runtime_call(
                "Weight Cache Write",
                cache.save_weight_cache,
                model,
                train_config,
                uid,
            )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
        model = _run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs)
    
    # ====================================================
    # 6. EVALUATION STRATEGY
    # ====================================================
    if not use_template:
        # STRATEGY A: Standard Softmax Classification
        print("[INFO] Bypassing Templates. Using standard Softmax Classifier trained on Session 1...")
        model.eval()
        all_probs, all_trues = [], []
        with torch.no_grad():
            for xb, yb in probe_loader:
                xb = xb.to(device)
                probs = torch.softmax(model(xb), dim=1)
                all_probs.append(probs.cpu().numpy())
                all_trues.append(yb.cpu().numpy())
        final_scores = np.vstack(all_probs)
        final_labels = np.concatenate(all_trues)
        
    else:
        # STRATEGY B: Template Matching
        print(f"[INFO] Building Enrollment Templates from Session 1...")
        model.include_top = False # Switch to Feature Extractor
        
        enroll_loader = _make_loader(x_train_full, y_train_enc, batch_size, shuffle=False)
        emb_enroll, lab_enroll = _get_embeddings(model, enroll_loader, device)
        
        gallery_emb, gallery_lab = _create_templates(
            emb_enroll, lab_enroll, method=template_fusion_method, max_beats=template_size
        )
        
        emb_probe, lab_probe = _get_embeddings(model, probe_loader, device)

        # raw_scores shape: (N_Probes, N_Gallery_Items)
        raw_scores = _compute_score_matrix(emb_probe, gallery_emb, method=matching_method)
        
        # Collapse raw_scores into class scores cleanly
        scores = np.full((len(emb_probe), len(classes)), -np.inf)
        for class_idx in range(len(classes)):
            gallery_mask = (gallery_lab == class_idx)
            if np.any(gallery_mask):
                scores[:, class_idx] = np.max(raw_scores[:, gallery_mask], axis=1)
                
        final_scores = scores
        final_labels = lab_probe
        
        # Restore model
        model.include_top = True

    # ====================================================
    # 7. APPLY SCORE-LEVEL FUSION
    # ====================================================
    final_scores, final_labels = _apply_score_fusion(
        final_scores, final_labels, fusion_size=probe_fusion_size
    )

    if visualize and use_template:
        viz = Visualizer()
        viz.plot_embeddings(emb_probe, final_labels, title="Cross-Session Probe Embeddings (T-SNE)")

    rank1, rank5 = _compute_metrics_identification(final_scores, final_labels)

    evaluation_artifacts = (
        _build_identification_curve_artifacts(
            final_scores,
            final_labels,
        )
    )

    _record_evaluation_artifact(
        evaluation_artifacts
    )

    # Update hyperparams dictionary dynamically
    hyperparams['epochs'] = f"{epochs} (stopped at {model.actual_epochs})" if model.actual_epochs < epochs else epochs

    data_stats = {
        "Total Cross-Session Subjects": len(classes),
        "Enrollment (S1) Samples": len(x_train_full),
        "Probe (S2) Samples": len(x_test_filtered),
    }

    if _return_stats:
        return (rank1, rank5), data_stats, hyperparams

    if save_results_and_settings:
        _log_experiment_results(
            task_title,
            {
                "Rank-1 Accuracy": rank1,
                "Rank-5 Accuracy": rank5,
            },
            data_stats,
            hyperparams,
            loader,
            evaluation_artifacts=evaluation_artifacts,
        )

    # 8. Report Identification Metrics
    return rank1, rank5

# =============================================================================
# TASK 6: CROSS-SESSION VERIFICATION
# =============================================================================
def run_cross_session_verification(x_train, y_train, x_test, y_test, model_class, epochs=150, 
                                   batch_size=256, lr=1e-3, val_split=0.0, num_pairs=10000, 
                                   sampling_mode="all", seed=42, device=None, visualize=False, 
                                   use_template=False, template_fusion_method='mean', template_size=None, 
                                   matching_method='cosine', outlier_filtering_on_train=False, 
                                   outlier_filtering_on_test=False, sqi_train=None, sqi_test=None, 
                                   sqi_threshold=0.05, sqi_keep_pct=0.8, use_deployment_evaluation=False,
                                   target_fars=None,
                                   save_results_and_settings=False, loader=None, 
                                   n_runs=1, _return_stats=False,
                                   intelligent_weight_loading=True,
                                   augmentation_config=None):
    """
    Cross-Session Verification Pipeline (Temporal Robustness 1:1).
    Attempts to verify if a subject is who they claim to be across different time-separated recording sessions.

    Args:
        x_train (np.ndarray): Input ECG signals from Session 1.
        y_train (np.ndarray): Labels for Session 1.
        x_test (np.ndarray): Input ECG signals from Session 2.
        y_test (np.ndarray): Labels for Session 2.
        model_class (nn.Module): The PyTorch model architecture class to instantiate.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per training batch.
        lr (float): Learning rate for the Adam optimizer.
        val_split (float): Fraction of Session 1 data to use for early stopping.
        num_pairs (int): Total number of Genuine and Impostor pairs to generate.
        sampling_mode (str): Logic used to pair beats together.
            Options: ['all', 'balanced', 'random']
        seed (int): Random seed for reproducibility.
        device (str): Computation device ('cuda', 'cpu', or 'auto').
        visualize (bool): If True, generates t-SNE scatter plots of the cross-session embeddings.
        use_template (bool):
            - False: Evaluates raw temporal space (Session 2 beats paired vs other Session 2 beats).
            - True: Simulates Authentication (Session 2 probes paired against Session 1 enrollment templates).
        template_fusion_method (str): Logic used to create the Session 1 templates.
            Options: ['mean', 'median', 'trimmed_mean', 'representative',
            'soft_centrality', 'geometric_median', 'none']
        template_size (int, optional): Number of Session 1 beats used for enrollment. None uses all available.
        matching_method (str): Distance/Similarity metric used to score the pairs.
            Options: ['cosine', 'euclidean', 'manhattan', 'correlation']
        outlier_filtering_on_train (bool): Apply SQI filtering independently to Session 1.
        outlier_filtering_on_test (bool): Apply SQI filtering independently to Session 2.
        sqi_train (str or np.ndarray): SQI calculation method or pre-computed array for Session 1.
        sqi_test (str or np.ndarray): SQI calculation method or pre-computed array for Session 2.
        sqi_threshold (float): Absolute minimum SQI score required to keep a beat (0.0 to 1.0).
        sqi_keep_pct (float): Top percentage of beats to keep per subject after filtering.
        use_deployment_evaluation (bool): Uses Session 1 Validation data to calculate a Global Threshold.
        save_results_and_settings (bool): If True, logs results and parameters to a text file.
        loader (object): Dataset loader instance (used for extracting metadata for logging).
        n_runs (int): Number of independent runs (with varying seeds) for statistical validation.
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.

    Returns:
        tuple: (EER, AUC, d-prime, TAR @ 0.1% FAR)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for all four metrics.
    """

    _validate_deployment_evaluation(
        use_deployment_evaluation,
        val_split,
        "Cross-Session Verification",
    )
    
    # ====================================================
    # 0. Capture Hyperparameters for Logger & MULTI-RUN AGGREGATOR
    # ====================================================
    data_stats = {}
    hyperparams = {
        'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr, 'num_pairs': num_pairs, 'use_template': use_template, 
        'template_fusion_method': template_fusion_method, 'template_size': template_size, 
        'matching_method': matching_method, 'val_split': val_split,
        'outlier_filter_train': outlier_filtering_on_train, 'outlier_filter_test': outlier_filtering_on_test
    }

    hyperparams = _add_seed_metadata(
        hyperparams,
        base_seed=seed,
        n_runs=n_runs,
    )

    if n_runs > 1:
        call_args = _prepare_multi_run_arguments(locals())
        base_seed = call_args.get('seed', 42)
        results = []
        evaluation_artifact_runs = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs)...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False 
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            run_wall_clock_started = _start_runtime_stage()
            artifact_sink_token = (
                _EVALUATION_ARTIFACT_SINK.set(
                    evaluation_artifact_runs
                )
            )
            try:
                res, d_stats, h_params = run_cross_session_verification(
                    **call_args
                )
            finally:
                _EVALUATION_ARTIFACT_SINK.reset(
                    artifact_sink_token
                )
            _record_multi_run_time(
                run_index=i + 1,
                seed=call_args["seed"],
                started_at=run_wall_clock_started,
            )
            results.append(res); data_stats = d_stats; hyperparams = h_params

        hyperparams = _add_seed_metadata(
            hyperparams,
            base_seed=base_seed,
            n_runs=n_runs,
        )

        per_run_results = (
            _build_per_run_results(
                results=results,
                seeds=hyperparams[
                    "run_seeds"
                ],
            )
        )

        per_run_evaluation_artifacts = (
            _build_per_run_evaluation_artifacts(
                artifacts=evaluation_artifact_runs,
                seeds=hyperparams[
                    "run_seeds"
                ],
            )
        )
                
        metrics_t = list(zip(*results))
        means, stds = [np.mean(m) for m in metrics_t], [np.std(m) for m in metrics_t]
        
        if save_results_and_settings:
            metrics_dict = {
                "EER": f"{means[0]:.4f} ± {stds[0]:.4f}", "AUC": f"{means[1]:.4f} ± {stds[1]:.4f}", 
                "d-prime": f"{means[2]:.4f} ± {stds[2]:.4f}", "TAR@0.1%FAR": f"{means[3]:.4f} ± {stds[3]:.4f}"
            }
            _log_experiment_results(
                "Cross-Session Verification",
                metrics_dict,
                data_stats,
                hyperparams,
                loader,
                per_run_results=per_run_results,
                per_run_evaluation_artifacts=per_run_evaluation_artifacts,
            )
        return tuple(zip(means, stds))
    # ----------------------------

    augmentation_config = (
        _normalize_augmentation_config(
            augmentation_config
        )
    )

    hyperparams[
        "augmentation"
    ] = augmentation_config

    _set_seed(seed); device = _get_device(device)
    partition_stage_started = _start_runtime_stage()
    task_title = "Cross-Session Verification"
    mode_str = f"Template ({template_fusion_method}, size={template_size or 'All'})" if use_template else "Cloud Pairs (Session 2 Only)"
    print(f"\n[TASK] {task_title} | Mode: {mode_str} | Match: {matching_method}")

    # ====================================================
    # 1. DYNAMIC SQI CALCULATION (Independent Sessions)
    # ====================================================
    def _prepare_sqi(sqi_input, x_data, flag, name):
        if not flag: return None
        if sqi_input is None:
            print(f"[WARN] Filtering requested for {name} but sqi scores are None. Skipping {name} filtering.")
            return None
        if isinstance(sqi_input, str):
            print(f"[INFO] Calculating SQI for {name} using method: '{sqi_input}'")
            return np.array(_compute_sqi(x_data, method=sqi_input))
        if isinstance(sqi_input, (list, np.ndarray)):
            return np.array(sqi_input)
        raise TypeError(f"[ERROR] sqi_{name.lower()} must be a string, array, or None.")

    sqi_train = _prepare_sqi(sqi_train, x_train, outlier_filtering_on_train, "Train")
    sqi_test = _prepare_sqi(sqi_test, x_test, outlier_filtering_on_test, "Test")

    # ====================================================
    # 2. APPLY SQI FILTERS
    # ====================================================
    if sqi_train is not None:
        print("\n[INFO] Filtering Session 1 (Enrollment)...")
        x_train, y_train = _apply_outlier_filter(x_train, y_train, sqi_train, sqi_threshold, sqi_keep_pct)

    if sqi_test is not None:
        print("\n[INFO] Filtering Session 2 (Probes)...")
        x_test, y_test = _apply_outlier_filter(
            x_test,
            y_test,
            sqi_test,
            absolute_threshold=sqi_threshold,
            keep_percentage=sqi_keep_pct,
            apply_subject_ranking=False,
        )

    # ====================================================
    # 3. SYNCHRONISE KNOWN SUBJECTS ACROSS SESSIONS
    # ====================================================
    minimum_probe_samples = (
        1
        if use_template
        else 2
    )

    cross_session_partitions = (
        _partition_closed_set_cross_session_samples(
            x_train,
            y_train,
            x_test,
            y_test,
            minimum_session_1_samples=2,
            minimum_session_2_samples=(
                minimum_probe_samples
            ),
        )
    )

    (
        x_train_full,
        y_train_full,
    ) = cross_session_partitions[
        "session_1"
    ]

    (
        x_test_filtered,
        y_test_filtered,
    ) = cross_session_partitions[
        "session_2"
    ]

    dropped_subjects = (
        cross_session_partitions[
            "dropped_subjects"
        ]
    )

    if len(
        dropped_subjects[
            "insufficient_session_1"
        ]
    ) > 0:
        print(
            "[WARN] Dropping "
            f"{len(dropped_subjects['insufficient_session_1'])} "
            "subjects with fewer than 2 surviving "
            "Session 1 samples."
        )

    if len(
        dropped_subjects[
            "insufficient_session_2"
        ]
    ) > 0:
        print(
            "[WARN] Dropping "
            f"{len(dropped_subjects['insufficient_session_2'])} "
            "subjects with fewer than "
            f"{minimum_probe_samples} surviving "
            "Session 2 probe samples."
        )
        
    # ====================================================
    # 4. ENCODE LABELS
    # ====================================================
    y_train_enc, classes = _encode_labels(y_train_full)
    label_map = {c: i for i, c in enumerate(classes)}
    y_test_enc = np.array([label_map[l] for l in y_test_filtered])
    
    # ====================================================
    # 5. RESUME STANDARD PIPELINE
    # ====================================================
    if val_split > 0.0:
        X_tr, X_val, y_tr, y_val = train_test_split(
            x_train_full, y_train_enc, test_size=val_split, stratify=y_train_enc, random_state=seed
        )
        val_loader = _make_loader(X_val, y_val, batch_size, shuffle=False)
        print(f"Session 1 Split: Train={len(X_tr)}, Val={len(X_val)} | Session 2 Probes={len(x_test_filtered)}")
    else:
        X_tr, y_tr = x_train_full, y_train_enc
        X_val, val_loader = None, None
        print(f"Session 1 Split: Train={len(X_tr)}, Val=0 | Session 2 Probes={len(x_test_filtered)}")

    X_tr, y_tr = (
        _augment_training_partition(
            X_tr,
            y_tr,
            augmentation_config,
            seed,
        )
    )

    train_loader = _make_loader(
        X_tr,
        y_tr,
        batch_size,
        shuffle=True,
    )
    probe_loader = _make_loader(x_test_filtered, y_test_enc, batch_size, shuffle=False)
    
    # Train Model
    _record_runtime_stage(
        "Partition Preparation",
        partition_stage_started,
    )

    model = model_class(in_channels=_detect_channels(x_train_full), num_classes=len(classes), include_top=True).to(device)

    hyperparams.update(
        _summarize_model_complexity(model)
    )

    if intelligent_weight_loading:
        from utils import CacheManager
        cache = CacheManager(
            base_dir=getattr(
                loader,
                "cache_dir",
                DEFAULT_CACHE_DIR,
            )
        )
        train_config = {
            "training_regime": "cross_session_closed_set",
            "model": model_class.__name__, "epochs": epochs, "batch_size": batch_size, "lr": lr, 
            "val_split": val_split, "seed": seed, "outlier_train": outlier_filtering_on_train, 
            "sqi_thresh": sqi_threshold, "classes": len(classes), "data_shape": X_tr.shape,
            "augmentation": augmentation_config,
        }

        train_config = _build_weight_cache_config(
            loader,
            train_config,
            training_samples=X_tr,
            training_labels=y_tr,
        )
        
        cached_model, uid = _timed_runtime_call(
            "Weight Cache Read",
            cache.get_weight_cache,
            train_config,
            model,
            device,
        )
        if cached_model:
            print(f"\n[INFO] Loaded pre-trained weights (Hash: {uid}). Skipping training!")
            model = cached_model
        else:
            print(f"\n[INFO] Training new Cross-Session model (Hash: {uid})...")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
            model = _run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs)
            _timed_runtime_call(
                "Weight Cache Write",
                cache.save_weight_cache,
                model,
                train_config,
                uid,
            )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()    
        model = _run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs)
    
    # Switch to Feature Extractor
    model.include_top = False

    # ====================================================
    # 6. MODEL CALIBRATION (Optional)
    # ====================================================
    if use_deployment_evaluation:
        print("\n[INFO] --- DEPLOYMENT THRESHOLD CALIBRATION ---")
        calib_loader = val_loader
        calib_name = "Validation"
        
        print(f"[INFO] Extracting features for Calibration (Session 1 {calib_name} Set)...")
        calib_emb, calib_lab = _get_embeddings(model, calib_loader, device)
        
        print(f"[INFO] Generating Calibration Pairs to find Global Threshold...")
        calib_scores, calib_pair_labels = _generate_pairs(
            embeddings1=calib_emb, labels1=calib_lab, embeddings2=None, labels2=None,
            num_pairs=num_pairs, sampling_mode=sampling_mode, matching_method=matching_method
        )
        global_threshold = _find_optimal_threshold(calib_scores, calib_pair_labels)
        print(f"[INFO] Optimal Global Threshold Found: {global_threshold:.4f}")

    # ====================================================
    # 7. EVALUATION STRATEGY
    # ====================================================
    emb_probe, lab_probe = _get_embeddings(model, probe_loader, device)

    if not use_template:
        # STRATEGY A: Session 2 vs Session 2 (Intra-session unseen evaluation)
        print(f"[INFO] Bypassing Templates. Generating pairs exclusively from Session 2...")
        scores, labels_pair = _generate_pairs(
            embeddings1=emb_probe, 
            labels1=lab_probe, 
            embeddings2=None, # None forces test vs test matching
            labels2=None, 
            num_pairs=num_pairs, 
            sampling_mode=sampling_mode, 
            matching_method=matching_method
        )
    else:
        # STRATEGY B: Session 2 Probes vs Session 1 Templates (Authentication Simulation)
        print(f"[INFO] Building Enrollment Templates from Session 1...")
        enroll_loader = _make_loader(x_train_full, y_train_enc, batch_size, shuffle=False)
        emb_enroll, lab_enroll = _get_embeddings(model, enroll_loader, device)
        
        templates, temp_labels = _create_templates(
            emb_enroll, lab_enroll, method=template_fusion_method, max_beats=template_size
        )
        
        scores, labels_pair = _generate_pairs(
            embeddings1=emb_probe, # Session 2 Probes
            labels1=lab_probe, 
            embeddings2=templates, # Session 1 Templates
            labels2=temp_labels, 
            num_pairs=num_pairs, 
            sampling_mode=sampling_mode, 
            matching_method=matching_method
        )

    if visualize:
        viz = Visualizer()
        viz.plot_embeddings(emb_probe, lab_probe, title="Cross-Session Probe Embeddings (T-SNE)")

    # 8. Apply Deployment Calibration
    if use_deployment_evaluation:
        _evaluate_with_global_threshold(scores, labels_pair, global_threshold)

    eer, auc_val, dprime, tar = _compute_metrics_verification(scores, labels_pair)

    evaluation_artifacts = (
        _build_verification_curve_artifacts(
            scores,
            labels_pair,
            target_fars=target_fars,
        )
    )

    _record_evaluation_artifact(
        evaluation_artifacts
    )

    # Update hyperparams dictionary dynamically
    hyperparams['epochs'] = f"{epochs} (stopped at {model.actual_epochs})" if model.actual_epochs < epochs else epochs

    data_stats = {
        "Total Cross-Session Subjects": len(classes),
        "Training (S1) Samples": len(X_tr),
        "Validation (S1) Samples": (
            len(X_val)
            if X_val is not None
            else 0
        ),
        "Enrollment (S1) Samples": len(
            x_train_full
        ),
        "Probe (S2) Samples": len(
            x_test_filtered
        ),
    }

    data_stats.update(
        _get_verification_pair_statistics(
            labels_pair,
            target_far=0.001,
        )
    )

    if _return_stats:
        return (
            eer,
            auc_val,
            dprime,
            tar,
        ), data_stats, hyperparams

    if save_results_and_settings:
        _log_experiment_results(
            task_title,
            {
                "EER": eer,
                "AUC": auc_val,
                "d-prime": dprime,
                "TAR@0.1%FAR": tar,
            },
            data_stats,
            hyperparams,
            loader,
            evaluation_artifacts=evaluation_artifacts,
        )

    # 9. Report Verification Metrics
    return eer, auc_val, dprime, tar

# =============================================================================
# TASK 7: SUBJECT-DISJOINT CROSS-SESSION IDENTIFICATION
# =============================================================================
def run_subject_disjoint_cross_session_identification(
        x_s1, y_s1, x_s2, y_s2, model_class, epochs=150, batch_size=256, lr=1e-3, test_split=0.2, val_split=0.0, 
        seed=42, device=None, visualize=False, use_template=True, template_fusion_method='mean', template_size=None, 
        matching_method='cosine', outlier_filtering_on_train=False, outlier_filtering_on_test=False, sqi_s1=None, 
        sqi_s2=None, sqi_threshold=0.05, sqi_keep_pct=0.8, probe_fusion_size=3, save_results_and_settings=False, 
        loader=None, n_runs=1, _return_stats=False,
        intelligent_weight_loading=True,
        augmentation_config=None):
    """
    The Ultimate Biometric Test: Subject-Disjoint + Temporal Robustness Identification.
    1. Trains a feature extractor on Session 1 of Subject Group A.
    2. Enrolls Unseen Subject Group B using their Session 1 recordings to build a gallery.
    3. Identifies Subject Group B using their Session 2 recordings as probes.

    Args:
        x_s1 (np.ndarray): Input ECG signals from Session 1.
        y_s1 (np.ndarray): Labels for Session 1.
        x_s2 (np.ndarray): Input ECG signals from Session 2.
        y_s2 (np.ndarray): Labels for Session 2.
        model_class (nn.Module): The PyTorch model architecture class to instantiate.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per training batch.
        lr (float): Learning rate for the Adam optimizer.
        test_split (float): Fraction of unique SUBJECTS to isolate for the Group B tests.
        val_split (float): Fraction of Group A subjects to use for early stopping validation.
        seed (int): Random seed for reproducibility.
        device (str): Computation device ('cuda', 'cpu', or 'auto').
        visualize (bool): If True, generates t-SNE scatter plots of the unseen temporal embeddings.
        use_template (bool): MUST be True for this task (requires a gallery to identify unseen subjects).
        template_fusion_method (str): Logic used to enroll unseen Session 1 data into the gallery.
            Options: ['mean', 'median', 'trimmed_mean', 'representative',
            'soft_centrality', 'geometric_median', 'none']
        template_size (int, optional): Number of Session 1 beats to form the gallery. None uses all available.
        matching_method (str): Distance/Similarity metric used to search the gallery.
            Options: ['cosine', 'euclidean', 'manhattan', 'correlation']
        outlier_filtering_on_train (bool): Apply SQI filtering to Session 1 data.
        outlier_filtering_on_test (bool): Apply SQI filtering to Session 2 data.
        sqi_s1 (str or np.ndarray): SQI calculation method or pre-computed array for Session 1.
        sqi_s2 (str or np.ndarray): SQI calculation method or pre-computed array for Session 2.
        sqi_threshold (float): Absolute minimum SQI score required to keep a beat (0.0 to 1.0).
        sqi_keep_pct (float): Top percentage of beats to keep per subject after filtering.
        probe_fusion_size (int): Number of consecutive Session 2 beats to average before searching the gallery.
        save_results_and_settings (bool): If True, logs results and parameters to a text file.
        loader (object): Dataset loader instance (used for extracting metadata for logging).
        n_runs (int): Number of independent runs (with varying seeds) for statistical validation.
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.

    Returns:
        tuple: (Rank-1 Accuracy, Rank-5 Accuracy)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for both metrics.
    """
    
    # ====================================================
    # 0. Capture Hyperparameters for Logger & MULTI-RUN AGGREGATOR
    # ====================================================
    data_stats = {}
    hyperparams = {
        'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr, 'test_split': test_split, 'val_split': val_split,
        'template_fusion_method': template_fusion_method, 'template_size': template_size, 
        'matching_method': matching_method, 'probe_fusion_size': probe_fusion_size,
        'outlier_filter_train': outlier_filtering_on_train, 'outlier_filter_test': outlier_filtering_on_test
    }

    hyperparams = _add_seed_metadata(
        hyperparams,
        base_seed=seed,
        n_runs=n_runs,
    )

    if n_runs > 1:
        call_args = _prepare_multi_run_arguments(locals())
        base_seed = call_args.get('seed', 42)
        results = []
        evaluation_artifact_runs = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs)...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False 
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            run_wall_clock_started = _start_runtime_stage()
            artifact_sink_token = (
                _EVALUATION_ARTIFACT_SINK.set(
                    evaluation_artifact_runs
                )
            )
            try:
                res, d_stats, h_params = run_subject_disjoint_cross_session_identification(
                    **call_args
                )
            finally:
                _EVALUATION_ARTIFACT_SINK.reset(
                    artifact_sink_token
                )
            _record_multi_run_time(
                run_index=i + 1,
                seed=call_args["seed"],
                started_at=run_wall_clock_started,
            )
            results.append(res); data_stats = d_stats; hyperparams = h_params

        hyperparams = _add_seed_metadata(
            hyperparams,
            base_seed=base_seed,
            n_runs=n_runs,
        )

        per_run_results = (
            _build_per_run_results(
                results=results,
                seeds=hyperparams[
                    "run_seeds"
                ],
            )
        )

        per_run_evaluation_artifacts = (
            _build_per_run_evaluation_artifacts(
                artifacts=evaluation_artifact_runs,
                seeds=hyperparams[
                    "run_seeds"
                ],
            )
        )
                
        r1_mean, r1_std = np.mean([r[0] for r in results]), np.std([r[0] for r in results])
        r5_mean, r5_std = np.mean([r[1] for r in results]), np.std([r[1] for r in results])
        
        if save_results_and_settings:
            metrics_dict = {"Rank-1 Accuracy": f"{r1_mean:.4f} ± {r1_std:.4f}", "Rank-5 Accuracy": f"{r5_mean:.4f} ± {r5_std:.4f}"}
            _log_experiment_results(
                "Subject-Disjoint Cross-Session ID",
                metrics_dict,
                data_stats,
                hyperparams,
                loader,
                per_run_results=per_run_results,
                per_run_evaluation_artifacts=per_run_evaluation_artifacts,
            )
        return (r1_mean, r1_std), (r5_mean, r5_std)
    # ----------------------------

    if not use_template:
        raise ValueError("[ERROR] use_template=False is invalid for Identification. Must use templates to build a gallery.")
        
    augmentation_config = (
        _normalize_augmentation_config(
            augmentation_config
        )
    )

    hyperparams[
        "augmentation"
    ] = augmentation_config

    _set_seed(seed); device = _get_device(device)
    partition_stage_started = _start_runtime_stage()
    task_title = "Subject-Disjoint Cross-Session ID"
    mode_str = f"Gallery: Session 1 ({template_fusion_method}, size={template_size or 'All'})"
    print(f"\n[TASK] {task_title} | Mode: {mode_str} | Match: {matching_method}")

    # ====================================================
    # 1. PREPARE & APPLY SQI FILTERS
    # ====================================================
    def _prepare_sqi(sqi_input, x_data, flag, name):
        if not flag: return None
        if sqi_input is None: return None
        if isinstance(sqi_input, str): return np.array(_compute_sqi(x_data, method=sqi_input))
        return np.array(sqi_input)

    sqi_s1 = _prepare_sqi(sqi_s1, x_s1, outlier_filtering_on_train, "Session 1")
    sqi_s2 = _prepare_sqi(sqi_s2, x_s2, outlier_filtering_on_test, "Session 2")

    if sqi_s1 is not None:
        print("\n[INFO] Filtering Session 1 (Representation & Enrollment)...")
        x_s1, y_s1 = _apply_outlier_filter(x_s1, y_s1, sqi_s1, sqi_threshold, sqi_keep_pct)

    if sqi_s2 is not None:
        print("\n[INFO] Filtering Session 2 (Probes)...")
        x_s2, y_s2 = _apply_outlier_filter(
            x_s2,
            y_s2,
            sqi_s2,
            absolute_threshold=sqi_threshold,
            keep_percentage=sqi_keep_pct,
            apply_subject_ranking=False,
        )

    # ====================================================
    # 2. INTERSECT AND SPLIT SUBJECTS (STRICTLY DISJOINT)
    # ====================================================
    # We must only evaluate subjects that successfully completed both sessions.
    common_subs = sorted(list(set(y_s1).intersection(set(y_s2))))
    
    if len(common_subs) < 2: 
        raise ValueError("[ERROR] Not enough common subjects across sessions after filtering.")
        
    # Split the distinct subjects into Train, Val, and Test cohorts
    (
        train_subs,
        val_subs,
        test_subs,
    ) = _split_subject_cohorts(
        common_subs,
        test_split=test_split,
        val_split=val_split,
        seed=seed,
    )

    print(
        "Subject Split: "
        f"Train={len(train_subs)}, "
        f"Val={len(val_subs)}, "
        f"Test={len(test_subs)}"
    )

    partitions = (
        _partition_subject_disjoint_cross_session_samples(
            x_s1,
            y_s1,
            x_s2,
            y_s2,
            train_subjects=train_subs,
            validation_subjects=val_subs,
            test_subjects=test_subs,
        )
    )

    (
        X_train,
        Y_train,
    ) = partitions["train"]

    (
        X_val_s1,
        Y_val_s1,
    ) = partitions["validation"]

    (
        X_enroll,
        Y_enroll,
    ) = partitions["enrollment"]

    (
        X_probe,
        Y_probe,
    ) = partitions["probe"]

    # ====================================================
    # 3. ENCODE LABELS (CRITICAL FIX FOR PYTORCH TENSORS)
    # ====================================================
    # PyTorch Datasets cannot handle raw strings (like "MLS" or "HPS").
    # We must explicitly map these strings to integers (0 to N-1).
    
    # A. Train Labels
    y_train_enc, train_classes = _encode_labels(Y_train)
    num_train_classes = len(train_classes)
    
    # B. Validation Labels (Ensures S1 map to the exact same integers)
    if len(val_subs) > 0:
        val_map = {sub: i for i, sub in enumerate(val_subs)}
        y_val_s1_enc = np.array([val_map[s] for s in Y_val_s1])
    else:
        y_val_s1_enc = None

    # C. Test Labels (Ensures Enroll and Probe map to the exact same integers)
    test_map = {sub: i for i, sub in enumerate(test_subs)}
    y_enroll_enc = np.array([test_map[s] for s in Y_enroll])
    y_probe_enc = np.array([test_map[s] for s in Y_probe])

    # ====================================================
    # 4. LOADERS & CUSTOM TRAINING LOOP
    # ====================================================
    # Create a Validation split from the SEEN Training subjects
    # This gives us the smooth Cross-Entropy loss anchor for the composite metric
    if val_split > 0.0:
        X_tr, X_val_seen, y_tr, y_val_seen = train_test_split(
            X_train, y_train_enc, test_size=val_split, stratify=y_train_enc, random_state=seed
        )
        val_loader_seen = _make_loader(X_val_seen, y_val_seen, batch_size, shuffle=False)
    else:
        X_tr, y_tr = X_train, y_train_enc
        val_loader_seen = None

    X_tr, y_tr = (
        _augment_training_partition(
            X_tr,
            y_tr,
            augmentation_config,
            seed,
        )
    )

    train_loader = _make_loader(
        X_tr,
        y_tr,
        batch_size,
        shuffle=True,
    )
    val_loader_s1 = _make_loader(X_val_s1, y_val_s1_enc, batch_size, shuffle=False) if X_val_s1 is not None else None
    
    # UNSEEN Validation now strictly passes the Session 1 loader only (Intra-Session check)
    val_loader_unseen = val_loader_s1
    
    _record_runtime_stage(
        "Partition Preparation",
        partition_stage_started,
    )

    model = model_class(in_channels=_detect_channels(x_s1), num_classes=num_train_classes, include_top=True).to(device)

    hyperparams.update(
        _summarize_model_complexity(model)
    )

    if intelligent_weight_loading:
        from utils import CacheManager
        cache = CacheManager(
            base_dir=getattr(
                loader,
                "cache_dir",
                DEFAULT_CACHE_DIR,
            )
        )
        train_config = {
            "training_regime": "cross_session_subject_disjoint",
            "model": model_class.__name__, "epochs": epochs, "batch_size": batch_size, "lr": lr, 
            "val_split": val_split, "seed": seed, "outlier_train": outlier_filtering_on_train, 
            "sqi_thresh": sqi_threshold, "classes": num_train_classes, "data_shape": X_tr.shape,
            "augmentation": augmentation_config,
            "matching_method": matching_method # Affects early stopping EER!
        }

        train_config = _build_weight_cache_config(
            loader,
            train_config,
            training_samples=X_tr,
            training_labels=y_tr,
        )
        
        cached_model, uid = _timed_runtime_call(
            "Weight Cache Read",
            cache.get_weight_cache,
            train_config,
            model,
            device,
        )
        if cached_model:
            print(f"\n[INFO] Loaded pre-trained weights (Hash: {uid}). Skipping training!")
            model = cached_model
        else:
            print(f"\n[INFO] Training new Subject-Disjoint Cross-Session model (Hash: {uid})...")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
            model = _run_train_loop_unseen_subjects(
                model=model, train_loader=train_loader, val_loader_seen=val_loader_seen, 
                val_loader_unseen=val_loader_unseen, optimizer=optimizer, criterion=criterion, 
                device=device, epochs=epochs, matching_method=matching_method, patience=40, lr_patience=15
            )
            _timed_runtime_call(
                "Weight Cache Write",
                cache.save_weight_cache,
                model,
                train_config,
                uid,
            )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Single line execution using the Composite Metric loop!
        model = _run_train_loop_unseen_subjects(
            model=model, 
            train_loader=train_loader, 
            val_loader_seen=val_loader_seen, 
            val_loader_unseen=val_loader_unseen, 
            optimizer=optimizer, 
            criterion=criterion, 
            device=device, 
            epochs=epochs, 
            matching_method=matching_method, 
            patience=40,       # Max epochs to wait for composite score improvement
            lr_patience=15     # Epochs to wait before halving Learning Rate
        )

    # ====================================================
    # 5. FINAL INFERENCE ON UNSEEN SUBJECTS
    # ====================================================
    model.include_top = False # Final metric extraction
    
    print(f"[INFO] Building Enrollment Templates for Unseen Subjects from Session 1...")
    enroll_loader = _make_loader(X_enroll, y_enroll_enc, batch_size, shuffle=False)
    emb_enroll, lab_enroll = _get_embeddings(model, enroll_loader, device)
    
    gallery_emb, gallery_lab = _create_templates(
        emb_enroll, lab_enroll, method=template_fusion_method, max_beats=template_size
    )

    print(f"[INFO] Probing with Unseen Subjects from Session 2...")
    probe_loader = _make_loader(X_probe, y_probe_enc, batch_size, shuffle=False)
    emb_probe, lab_probe = _get_embeddings(model, probe_loader, device)

    # Because we mapped y_enroll_enc and y_probe_enc to integers, 
    # gallery_lab and lab_probe are already perfectly aligned from 0 to N-1!
    raw_scores = _compute_score_matrix(emb_probe, gallery_emb, method=matching_method)
    scores = np.full((len(emb_probe), len(test_subs)), -np.inf)
    
    for class_idx in range(len(test_subs)):
        gallery_mask = (gallery_lab == class_idx)
        if np.any(gallery_mask):
            scores[:, class_idx] = np.max(raw_scores[:, gallery_mask], axis=1)

    # ====================================================
    # 6. APPLY SCORE-LEVEL FUSION & EVALUATE
    # ====================================================
    final_scores, final_labels = _apply_score_fusion(scores, lab_probe, fusion_size=probe_fusion_size)

    if visualize:
        viz = Visualizer()
        viz.plot_embeddings(emb_probe, lab_probe, title="Disjoint Cross-Session Embeddings (T-SNE)")

    rank1, rank5 = _compute_metrics_identification(final_scores, final_labels)

    evaluation_artifacts = (
        _build_identification_curve_artifacts(
            final_scores,
            final_labels,
        )
    )

    _record_evaluation_artifact(
        evaluation_artifacts
    )

    # Update hyperparams dictionary dynamically using the model's tracked epochs
    actual_ep = getattr(model, 'actual_epochs', epochs)
    hyperparams['epochs'] = f"{epochs} (stopped at {actual_ep})" if actual_ep < epochs else epochs

    data_stats = {
        "Train Subjects": len(train_subs),
        "Test Subjects": len(test_subs),
        "Train (S1) Samples": len(X_train),
        "Enrollment (S1) Samples": len(X_enroll),
        "Probe (S2) Samples": len(X_probe),
    }

    if _return_stats:
        return (rank1, rank5), data_stats, hyperparams

    if save_results_and_settings:
        _log_experiment_results(
            task_title,
            {
                "Rank-1 Accuracy": rank1,
                "Rank-5 Accuracy": rank5,
            },
            data_stats,
            hyperparams,
            loader,
            evaluation_artifacts=evaluation_artifacts,
        )

    return rank1, rank5


# =============================================================================
# TASK 8: SUBJECT-DISJOINT CROSS-SESSION VERIFICATION
# =============================================================================
def run_subject_disjoint_cross_session_verification(
        x_s1, y_s1, x_s2, y_s2, model_class, epochs=150, batch_size=256, lr=1e-3, test_split=0.2, val_split=0.0, 
        num_pairs=10000, sampling_mode="all", seed=42, device=None, visualize=False, use_template=False, 
        template_fusion_method='mean', template_size=None, matching_method='cosine', outlier_filtering_on_train=False, 
        outlier_filtering_on_test=False, sqi_s1=None, sqi_s2=None, sqi_threshold=0.05, sqi_keep_pct=0.8,
        use_deployment_evaluation=False, target_fars=None,
        save_results_and_settings=False, loader=None, n_runs=1, _return_stats=False,
        intelligent_weight_loading=True,
        augmentation_config=None):
    """
    The Ultimate Biometric Test: Subject-Disjoint + Temporal Robustness 1:1 Verification.
    Verifies the identity of subjects completely excluded from representation learning, across different recording days.
    The model learns generalized features on Session 1 of Subject Group A, and evaluates verification on Subject Group B.

    Args:
        x_s1 (np.ndarray): Input ECG signals from Session 1.
        y_s1 (np.ndarray): Labels for Session 1.
        x_s2 (np.ndarray): Input ECG signals from Session 2.
        y_s2 (np.ndarray): Labels for Session 2.
        model_class (nn.Module): The PyTorch model architecture class to instantiate.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per training batch.
        lr (float): Learning rate for the Adam optimizer.
        test_split (float): Fraction of unique SUBJECTS to isolate for the Group B tests.
        val_split (float): Fraction of Group A subjects to use for early stopping validation.
        num_pairs (int): Total number of Genuine and Impostor pairs to generate for evaluation.
        sampling_mode (str): Logic used to pair beats together.
            Options: ['all', 'balanced', 'random']
        seed (int): Random seed for reproducibility.
        device (str): Computation device ('cuda', 'cpu', or 'auto').
        visualize (bool): If True, generates t-SNE scatter plots of the unseen temporal embeddings.
        use_template (bool):
            - False: Evaluates raw temporal space (Group B's Session 2 paired vs Group B's Session 2).
            - True: Simulates Authentication (Group B's Session 2 probes matched vs Group B's Session 1 templates).
        template_fusion_method (str): Logic used to create Session 1 templates.
            Options: ['mean', 'median', 'trimmed_mean', 'representative',
            'soft_centrality', 'geometric_median', 'none']
        template_size (int, optional): Number of Session 1 beats used for enrollment. None uses all available.
        matching_method (str): Distance/Similarity metric used to score the pairs.
            Options: ['cosine', 'euclidean', 'manhattan', 'correlation']
        outlier_filtering_on_train (bool): Apply SQI filtering to Session 1 data.
        outlier_filtering_on_test (bool): Apply SQI filtering to Session 2 data.
        sqi_s1 (str or np.ndarray): SQI calculation method or pre-computed array for Session 1.
        sqi_s2 (str or np.ndarray): SQI calculation method or pre-computed array for Session 2.
        sqi_threshold (float): Absolute minimum SQI score required to keep a beat (0.0 to 1.0).
        sqi_keep_pct (float): Top percentage of beats to keep per subject after filtering.
        use_deployment_evaluation (bool): Uses unseen validation subjects from Group A to calculate a Global Threshold.
        save_results_and_settings (bool): If True, logs results and parameters to a text file.
        loader (object): Dataset loader instance (used for extracting metadata for logging).
        n_runs (int): Number of independent runs (with varying seeds) for statistical validation.
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.

    Returns:
        tuple: (EER, AUC, d-prime, TAR @ 0.1% FAR)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for all four metrics.
    """

    _validate_deployment_evaluation(
        use_deployment_evaluation,
        val_split,
        "Subject-Disjoint Cross-Session Verification",
    )
    
    # ====================================================
    # 0. Capture Hyperparameters for Logger & MULTI-RUN AGGREGATOR
    # ====================================================
    data_stats = {}
    hyperparams = {
        'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr, 'test_split': test_split, 'val_split': val_split, 
        'num_pairs': num_pairs, 'use_template': use_template, 'template_fusion_method': template_fusion_method,
        'template_size': template_size, 'matching_method': matching_method, 'outlier_filter_train': outlier_filtering_on_train, 
        'outlier_filter_test': outlier_filtering_on_test
    }

    hyperparams = _add_seed_metadata(
        hyperparams,
        base_seed=seed,
        n_runs=n_runs,
    )

    if n_runs > 1:
        call_args = _prepare_multi_run_arguments(locals())
        base_seed = call_args.get('seed', 42)
        results = []
        evaluation_artifact_runs = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs)...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False 
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            run_wall_clock_started = _start_runtime_stage()
            artifact_sink_token = (
                _EVALUATION_ARTIFACT_SINK.set(
                    evaluation_artifact_runs
                )
            )
            try:
                res, d_stats, h_params = run_subject_disjoint_cross_session_verification(
                    **call_args
                )
            finally:
                _EVALUATION_ARTIFACT_SINK.reset(
                    artifact_sink_token
                )
            _record_multi_run_time(
                run_index=i + 1,
                seed=call_args["seed"],
                started_at=run_wall_clock_started,
            )
            results.append(res); data_stats = d_stats; hyperparams = h_params

        hyperparams = _add_seed_metadata(
            hyperparams,
            base_seed=base_seed,
            n_runs=n_runs,
        )

        per_run_results = (
            _build_per_run_results(
                results=results,
                seeds=hyperparams[
                    "run_seeds"
                ],
            )
        )

        per_run_evaluation_artifacts = (
            _build_per_run_evaluation_artifacts(
                artifacts=evaluation_artifact_runs,
                seeds=hyperparams[
                    "run_seeds"
                ],
            )
        )
                
        metrics_t = list(zip(*results))
        means, stds = [np.mean(m) for m in metrics_t], [np.std(m) for m in metrics_t]
        
        if save_results_and_settings:
            metrics_dict = {
                "EER": f"{means[0]:.4f} ± {stds[0]:.4f}", "AUC": f"{means[1]:.4f} ± {stds[1]:.4f}", 
                "d-prime": f"{means[2]:.4f} ± {stds[2]:.4f}", "TAR@0.1%FAR": f"{means[3]:.4f} ± {stds[3]:.4f}"
            }
            _log_experiment_results(
                (
                    "Subject-Disjoint "
                    "Cross-Session Verification"
                ),
                metrics_dict,
                data_stats,
                hyperparams,
                loader,
                per_run_results=per_run_results,
                per_run_evaluation_artifacts=per_run_evaluation_artifacts,
            )
        return tuple(zip(means, stds))
    # ----------------------------

    augmentation_config = (
        _normalize_augmentation_config(
            augmentation_config
        )
    )

    hyperparams[
        "augmentation"
    ] = augmentation_config

    _set_seed(seed); device = _get_device(device)
    partition_stage_started = _start_runtime_stage()
    task_title = "Subject-Disjoint Cross-Session Verification"
    mode_str = f"Template ({template_fusion_method}, S1 Enroll -> S2 Probe)" if use_template else "Cloud Pairs (S2 vs S2)"
    print(f"\n[TASK] {task_title} | Mode: {mode_str} | Match: {matching_method}")

    # ====================================================
    # 1. PREPARE & APPLY SQI FILTERS
    # ====================================================
    def _prepare_sqi(sqi_input, x_data, flag, name):
        if not flag: return None
        if sqi_input is None: return None
        if isinstance(sqi_input, str): return np.array(_compute_sqi(x_data, method=sqi_input))
        return np.array(sqi_input)

    sqi_s1 = _prepare_sqi(sqi_s1, x_s1, outlier_filtering_on_train, "Session 1")
    sqi_s2 = _prepare_sqi(sqi_s2, x_s2, outlier_filtering_on_test, "Session 2")

    if sqi_s1 is not None:
        print("\n[INFO] Filtering Session 1 (Representation & Enrollment)...")
        x_s1, y_s1 = _apply_outlier_filter(x_s1, y_s1, sqi_s1, sqi_threshold, sqi_keep_pct)

    if sqi_s2 is not None:
        print("\n[INFO] Filtering Session 2 (Probes)...")
        x_s2, y_s2 = _apply_outlier_filter(
            x_s2,
            y_s2,
            sqi_s2,
            absolute_threshold=sqi_threshold,
            keep_percentage=sqi_keep_pct,
            apply_subject_ranking=False,
        )

    # ====================================================
    # 2. INTERSECT AND SPLIT SUBJECTS
    # ====================================================
    common_subs = sorted(list(set(y_s1).intersection(set(y_s2))))
    
    if len(common_subs) < 2: 
        raise ValueError("[ERROR] Not enough common subjects across sessions after filtering.")
        
    (
        train_subs,
        val_subs,
        test_subs,
    ) = _split_subject_cohorts(
        common_subs,
        test_split=test_split,
        val_split=val_split,
        seed=seed,
    )

    print(
        "Subject Split: "
        f"Train={len(train_subs)}, "
        f"Val={len(val_subs)}, "
        f"Test={len(test_subs)}"
    )

    partitions = (
        _partition_subject_disjoint_cross_session_samples(
            x_s1,
            y_s1,
            x_s2,
            y_s2,
            train_subjects=train_subs,
            validation_subjects=val_subs,
            test_subjects=test_subs,
        )
    )

    (
        X_train,
        Y_train,
    ) = partitions["train"]

    (
        X_val_s1,
        Y_val_s1,
    ) = partitions["validation"]

    (
        X_enroll,
        Y_enroll,
    ) = partitions["enrollment"]

    (
        X_probe,
        Y_probe,
    ) = partitions["probe"]

    # ====================================================
    # 3. ENCODE LABELS (CRITICAL FIX FOR PYTORCH TENSORS)
    # ====================================================
    y_train_enc, train_classes = _encode_labels(Y_train)
    num_train_classes = len(train_classes)
    
    if len(val_subs) > 0:
        val_map = {sub: i for i, sub in enumerate(val_subs)}
        y_val_s1_enc = np.array([val_map[s] for s in Y_val_s1])
    else:
        y_val_s1_enc = None

    test_map = {sub: i for i, sub in enumerate(test_subs)}
    y_enroll_enc = np.array([test_map[s] for s in Y_enroll])
    y_probe_enc = np.array([test_map[s] for s in Y_probe])

    # ====================================================
    # 4. LOADERS & CUSTOM TRAINING LOOP
    # ====================================================
    # Create a Validation split from the SEEN Training subjects
    # This gives us the smooth Cross-Entropy loss anchor for the composite metric
    if val_split > 0.0:
        X_tr, X_val_seen, y_tr, y_val_seen = train_test_split(
            X_train, y_train_enc, test_size=val_split, stratify=y_train_enc, random_state=seed
        )
        val_loader_seen = _make_loader(X_val_seen, y_val_seen, batch_size, shuffle=False)
    else:
        X_tr, y_tr = X_train, y_train_enc
        val_loader_seen = None

    X_tr, y_tr = (
        _augment_training_partition(
            X_tr,
            y_tr,
            augmentation_config,
            seed,
        )
    )

    train_loader = _make_loader(
        X_tr,
        y_tr,
        batch_size,
        shuffle=True,
    )
    val_loader_s1 = _make_loader(X_val_s1, y_val_s1_enc, batch_size, shuffle=False) if X_val_s1 is not None else None
    
    # UNSEEN Validation now strictly passes the Session 1 loader only (Intra-Session check)
    val_loader_unseen = val_loader_s1
    
    _record_runtime_stage(
        "Partition Preparation",
        partition_stage_started,
    )

    model = model_class(in_channels=_detect_channels(x_s1), num_classes=num_train_classes, include_top=True).to(device)

    hyperparams.update(
        _summarize_model_complexity(model)
    )

    if intelligent_weight_loading:
        from utils import CacheManager
        cache = CacheManager(
            base_dir=getattr(
                loader,
                "cache_dir",
                DEFAULT_CACHE_DIR,
            )
        )
        train_config = {
            "training_regime": "cross_session_subject_disjoint",
            "model": model_class.__name__, "epochs": epochs, "batch_size": batch_size, "lr": lr, 
            "val_split": val_split, "seed": seed, "outlier_train": outlier_filtering_on_train, 
            "sqi_thresh": sqi_threshold, "classes": num_train_classes, "data_shape": X_tr.shape,
            "augmentation": augmentation_config,
            "matching_method": matching_method # Affects early stopping EER!
        }

        train_config = _build_weight_cache_config(
            loader,
            train_config,
            training_samples=X_tr,
            training_labels=y_tr,
        )

        cached_model, uid = _timed_runtime_call(
            "Weight Cache Read",
            cache.get_weight_cache,
            train_config,
            model,
            device,
        )
        if cached_model:
            print(f"\n[INFO] Loaded pre-trained weights (Hash: {uid}). Skipping training!")
            model = cached_model
        else:
            print(f"\n[INFO] Training new Subject-Disjoint Cross-Session model (Hash: {uid})...")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
            model = _run_train_loop_unseen_subjects(
                model=model, train_loader=train_loader, val_loader_seen=val_loader_seen, 
                val_loader_unseen=val_loader_unseen, optimizer=optimizer, criterion=criterion, 
                device=device, epochs=epochs, matching_method=matching_method, patience=40, lr_patience=15
            )
            _timed_runtime_call(
                "Weight Cache Write",
                cache.save_weight_cache,
                model,
                train_config,
                uid,
            )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Single line execution using the Composite Metric loop!
        model = _run_train_loop_unseen_subjects(
            model=model, 
            train_loader=train_loader, 
            val_loader_seen=val_loader_seen, 
            val_loader_unseen=val_loader_unseen, 
            optimizer=optimizer, 
            criterion=criterion, 
            device=device, 
            epochs=epochs, 
            matching_method=matching_method, 
            patience=40,       # Max epochs to wait for composite score improvement
            lr_patience=15     # Epochs to wait before halving Learning Rate
        )

    # ====================================================
    # 5. MODEL CALIBRATION (Optional)
    # ====================================================
    model.include_top = False
    
    if use_deployment_evaluation:
        print("\n[INFO] --- DEPLOYMENT THRESHOLD CALIBRATION ---")
        calib_loader = val_loader_s1
        calib_name = "Unseen Validation (Session 1)"
            
        print(f"[INFO] Extracting features for Calibration ({calib_name})...")
        calib_emb_s1, calib_lab_s1 = _get_embeddings(model, calib_loader, device)
        
        # Calibration relies entirely on Session 1 features
        print(f"[INFO] Generating Calibration Pairs to find Global Threshold...")
        calib_scores, calib_pair_labels = _generate_pairs(
            embeddings1=calib_emb_s1, labels1=calib_lab_s1, 
            embeddings2=None, labels2=None,
            num_pairs=num_pairs, sampling_mode=sampling_mode, matching_method=matching_method
        )
        global_threshold = _find_optimal_threshold(calib_scores, calib_pair_labels)
        print(f"[INFO] Optimal Global Threshold Found: {global_threshold:.4f}")

    # ====================================================
    # 6. EVALUATION STRATEGY ON UNSEEN TEST SUBJECTS
    # ====================================================
    probe_loader = _make_loader(X_probe, y_probe_enc, batch_size, shuffle=False)
    emb_probe, lab_probe = _get_embeddings(model, probe_loader, device)

    if not use_template:
        print(f"[INFO] Bypassing Templates. Generating pairs entirely from Session 2 for Unseen Subjects...")
        scores, labels_pair = _generate_pairs(
            embeddings1=emb_probe, labels1=lab_probe, 
            embeddings2=None, labels2=None, 
            num_pairs=num_pairs, sampling_mode=sampling_mode, matching_method=matching_method
        )
    else:
        print(f"[INFO] Building Enrollment Templates for Unseen Subjects from Session 1...")
        enroll_loader = _make_loader(X_enroll, y_enroll_enc, batch_size, shuffle=False)
        emb_enroll, lab_enroll = _get_embeddings(model, enroll_loader, device)
        
        templates, temp_labels = _create_templates(
            emb_enroll, lab_enroll, method=template_fusion_method, max_beats=template_size
        )
        
        scores, labels_pair = _generate_pairs(
            embeddings1=emb_probe, # Session 2 Probes
            labels1=lab_probe, 
            embeddings2=templates, # Session 1 Templates
            labels2=temp_labels, 
            num_pairs=num_pairs, 
            sampling_mode=sampling_mode, 
            matching_method=matching_method
        )
        
    if visualize:
        viz = Visualizer()
        viz.plot_embeddings(emb_probe, lab_probe, title="Disjoint Cross-Session Embeddings (T-SNE)")

    if use_deployment_evaluation:
        _evaluate_with_global_threshold(scores, labels_pair, global_threshold)

    eer, auc_val, dprime, tar = _compute_metrics_verification(scores, labels_pair)

    evaluation_artifacts = (
        _build_verification_curve_artifacts(
            scores,
            labels_pair,
            target_fars=target_fars,
        )
    )

    _record_evaluation_artifact(
        evaluation_artifacts
    )

    # Update hyperparams dictionary dynamically using the model's tracked epochs
    actual_ep = getattr(model, 'actual_epochs', epochs)
    hyperparams['epochs'] = f"{epochs} (stopped at {actual_ep})" if actual_ep < epochs else epochs

    data_stats = {
        "Train Subjects": len(train_subs),
        "Train Samples": len(X_train),
        "Validation Subjects": len(
            val_subs
        ),
        "Validation (S1) Samples": (
            len(X_val_s1)
            if X_val_s1 is not None
            else 0
        ),
        "Test Subjects": len(test_subs),
        "Enrollment (S1) Samples": len(
            X_enroll
        ),
        "Probe (S2) Samples": len(
            X_probe
        ),
    }

    data_stats.update(
        _get_verification_pair_statistics(
            labels_pair,
            target_far=0.001,
        )
    )

    if _return_stats:
        return (
            eer,
            auc_val,
            dprime,
            tar,
        ), data_stats, hyperparams

    if save_results_and_settings:
        _log_experiment_results(
            task_title,
            {
                "EER": eer,
                "AUC": auc_val,
                "d-prime": dprime,
                "TAR@0.1%FAR": tar,
            },
            data_stats,
            hyperparams,
            loader,
            evaluation_artifacts=evaluation_artifacts,
        )

    return eer, auc_val, dprime, tar
# =============================================================================
