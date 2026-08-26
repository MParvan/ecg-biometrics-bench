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
from artifact_provenance import (
    STATE_DICT_HASH_FORMAT,
    build_weight_compatibility_identity,
    canonical_state_dict_sha256,
)
from experiment_provenance import (
    build_experiment_implementation_identity,
    build_result_record_provenance,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy import stats

from utils import (
    _apply_score_fusion, _generate_fused_verification_pairs, _source_order_indices, _make_loader, _encode_labels,
    _select_multi_templates, _reduce_template_scores_to_identities, _generate_multi_template_verification_pairs,
    _apply_outlier_filter, _compute_sqi, _compute_score_matrix,
    _get_embeddings, _create_templates, _generate_pairs, _resolve_pair_sampling_arguments,
    _find_optimal_threshold, _evaluate_with_global_threshold, _summarize_verification_pairs,
    _build_identification_curve_artifacts,
    _build_verification_curve_artifacts,
    _compute_metrics_identification, _compute_metrics_verification,
    _run_training_loop, _run_train_loop_unseen_subjects, _train_epoch, _detect_channels,
    DEFAULT_CACHE_DIR, DEFAULT_RESULTS_DIR, resolve_artifact_path,
    _build_loader_cache_identity, _fingerprint_array_collection,
    _prepare_reproducibility_backend, _setup_reproducibility,
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


_ACTIVE_PROFILING_DEVICE_TYPE = "auto"

def _set_active_profiling_device_type(device):
    global _ACTIVE_PROFILING_DEVICE_TYPE
    if device is None or device == "auto":
        _ACTIVE_PROFILING_DEVICE_TYPE = "auto"
    else:
        try:
            _ACTIVE_PROFILING_DEVICE_TYPE = torch.device(device).type
        except Exception:
            _ACTIVE_PROFILING_DEVICE_TYPE = "auto"

def _should_query_cuda():
    if _ACTIVE_PROFILING_DEVICE_TYPE == "cpu":
        return False
    return torch.cuda.is_available()


def _collect_software_environment():
    """
    Collect the software and hardware environment used by an experiment.
    """
    environment = {
        "Python": platform.python_version(),
        "Operating System": platform.platform(),
        "PyTorch": str(torch.__version__),
        "NumPy": _get_installed_package_version("numpy"),
        "SciPy": _get_installed_package_version("scipy"),
        "scikit-learn": _get_installed_package_version("scikit-learn"),
        "pandas": _get_installed_package_version("pandas"),
        "NeuroKit2": _get_installed_package_version("neurokit2"),
        "WFDB": _get_installed_package_version("wfdb"),
        "PyYAML": _get_installed_package_version("PyYAML"),
    }

    if _should_query_cuda():
        environment["CUDA Available"] = True
        environment["CUDA Runtime"] = (
            str(torch.version.cuda)
            if torch.version.cuda is not None
            else "not available"
        )
        try:
            environment["CUDA Device"] = torch.cuda.get_device_name(0)
        except Exception:
            environment["CUDA Device"] = "unavailable"
    else:
        environment["CUDA Available"] = None
        environment["CUDA Runtime"] = None
        environment["CUDA Device"] = None

    return environment


# =============================================================================
# COMPUTATIONAL PROFILE
# =============================================================================

_EXPERIMENT_START_TIME = None
_ENTRYPOINT_PROFILE_PENDING_RUNNER = False

_RUNTIME_STAGE_TOTALS = collections.defaultdict(float)
_RUNTIME_STAGE_COUNTS = collections.defaultdict(int)
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
    """
    if not _should_query_cuda():
        return

    try:
        torch.cuda.synchronize()
    except Exception:
        pass


def _initialize_runtime_profile(device=None):
    """
    Initialize a fresh wall-clock, stage, and peak-memory profile.
    """
    global _EXPERIMENT_START_TIME

    _set_active_profiling_device_type(device)
    _reset_runtime_profile()
    _EXPERIMENT_START_TIME = time.perf_counter()

    if _should_query_cuda():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def start_experiment_timer(device=None):
    """Start a profile that the next top-level public runner will adopt."""
    global _ENTRYPOINT_PROFILE_PENDING_RUNNER

    _initialize_runtime_profile(device)
    _ENTRYPOINT_PROFILE_PENDING_RUNNER = True


def _activate_top_level_runtime_profile(
    reproducibility_mode,
    device,
    recursive_run=False,
):
    """Prepare reproducibility and establish one profile per top-level run."""
    global _ENTRYPOINT_PROFILE_PENDING_RUNNER

    mode = _prepare_reproducibility_backend(
        reproducibility_mode,
        device,
    )

    if recursive_run:
        return mode

    if _ENTRYPOINT_PROFILE_PENDING_RUNNER:
        # The CLI timer already includes dataset preparation and must not be
        # reset when its selected runner starts.
        _ENTRYPOINT_PROFILE_PENDING_RUNNER = False
        _set_active_profiling_device_type(device)
    else:
        # A direct public API call owns a fresh profile, even if globals from
        # an earlier completed experiment remain populated in this process.
        _initialize_runtime_profile(device)

    return mode


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
_ORIGINAL_GENERATE_FUSED_VERIFICATION_PAIRS = (
    _generate_fused_verification_pairs
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

_generate_fused_verification_pairs = _make_runtime_wrapper(
    "Probe Fusion",
    _ORIGINAL_GENERATE_FUSED_VERIFICATION_PAIRS,
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

    if _should_query_cuda():
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
    else:
        profile[
            "Peak CUDA Memory (MiB)"
        ] = None

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
    Non-finite floating-point values are converted to None because
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

        return None

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
    canonical_provenance=None,
):
    """
    Build one self-contained machine-readable experiment record.
    """
    normalized_per_run_results = _to_json_compatible(
        (
            []
            if per_run_results is None
            else per_run_results
        )
    )

    record = {
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
            normalized_per_run_results
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
        "canonical_provenance": (
            _to_json_compatible(
                canonical_provenance
            )
        ),
    }

    if (
        len(normalized_per_run_results) > 1
        and all(
            isinstance(run_record, Mapping)
            and all(
                field_name in run_record
                for field_name in (
                    "run_index",
                    "seed",
                    "split_seed",
                    "data_statistics",
                    "trained_weight",
                )
            )
            for run_record in normalized_per_run_results
        )
    ):
        final_run = normalized_per_run_results[-1]
        record["data_statistics_scope"] = {
            "scope": "last_run_snapshot",
            "run_index": final_run["run_index"],
            "seed": final_run["seed"],
            "split_seed": final_run["split_seed"],
        }

    return record


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

    authoritative_configuration = getattr(
        loader,
        "effective_experiment_configuration",
        None,
    )
    configuration_authoritative = isinstance(
        authoritative_configuration,
        Mapping,
    )
    if not configuration_authoritative:
        fallback_hyperparameters = {
            key: value
            for key, value in hyperparams.items()
            if not any(
                marker in str(key).lower()
                for marker in (
                    "cache",
                    "directory",
                    "output",
                    "path",
                    "time",
                )
            )
        }
        fallback_dataset_settings = {
            key: value
            for key, value in dataset_kwargs.items()
            if not any(
                marker in str(key).lower()
                for marker in (
                    "cache",
                    "directory",
                    "output",
                    "path",
                    "results_dir",
                    "time",
                )
            )
        }
        authoritative_configuration = {
            "task": str(task_name),
            "dataset": str(dataset_name),
            "model_hyperparameters": _to_json_compatible(
                fallback_hyperparameters
            ),
            "dataset_and_preprocessing_settings": _to_json_compatible(
                fallback_dataset_settings
            ),
        }

    provenance_context = getattr(
        loader,
        "result_provenance_context",
        {},
    )
    if not isinstance(provenance_context, Mapping):
        provenance_context = {}

    canonical_provenance = build_result_record_provenance(
        effective_configuration=authoritative_configuration,
        configuration_authoritative=configuration_authoritative,
        implementation_identity=(
            build_experiment_implementation_identity()
        ),
        campaign_id=provenance_context.get("campaign_id"),
        smoke_run=provenance_context.get("smoke_run", False),
        hyperparameters=hyperparams,
        per_run_results=per_run_results,
    )

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
            canonical_provenance=canonical_provenance,
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


def _validate_verification_probe_fusion(
    probe_fusion_size,
    use_template,
    use_deployment_evaluation,
    task_name,
):
    """
    Validate probe_fusion_size for a verification runner.

    Multi-beat probe fusion (probe_fusion_size > 1) requires an explicit
    enrollment target and does not compose with threshold-calibrated
    deployment evaluation. probe_fusion_size == 1 preserves the standard
    verification pair-generation path exactly.
    """
    try:
        depth = int(probe_fusion_size)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{task_name}: 'probe_fusion_size' must be an integer greater "
            f"than or equal to 1, received {probe_fusion_size!r}."
        ) from error

    if depth < 1:
        raise ValueError(
            f"{task_name}: 'probe_fusion_size' must be an integer greater "
            f"than or equal to 1, received {depth}."
        )

    if depth > 1:
        if not use_template:
            raise ValueError(
                f"{task_name}: multi-beat probe fusion "
                "(probe_fusion_size > 1) requires use_template=True so "
                "that a distinct enrollment target is defined for the "
                "fused decision."
            )
        if use_deployment_evaluation:
            raise ValueError(
                f"{task_name}: multi-beat probe fusion "
                "(probe_fusion_size > 1) is not compatible with "
                "use_deployment_evaluation. Calibrate the deployment "
                "threshold at probe_fusion_size = 1 and study probe "
                "fusion depths as a separate evaluation."
            )

    return depth


_UNSET = object()
"""Sentinel distinguishing an omitted keyword argument from an explicit
``None`` or other falsy value. ``template_fusion_method`` uses ``None`` as a
genuine, legacy no-fusion selector, so plain ``None`` cannot serve as the
"argument omitted" marker."""


def _validate_multi_template_arguments(
    enrollment_template_mode,
    template_fusion_method,
    num_templates_per_identity,
    template_selection_method,
    template_score_aggregation,
    use_template,
    use_deployment_evaluation,
    task_name,
):
    """
    Validate and resolve the enrollment-template-mode arguments.

    ``enrollment_template_mode='fusion'`` is the existing enrollment path:
    ``template_fusion_method`` selects how enrollment observations combine
    into one template per identity (or, for ``'none'``/``None``, are kept
    unfused). The new multi-template parameters have no meaning in this mode
    and are rejected if explicitly supplied.

    ``enrollment_template_mode='multi_template'`` selects a fixed number of
    representative enrollment observations per identity instead of fusing
    them. ``template_fusion_method`` has no meaning in this mode: supplying
    it explicitly (any value, including ``'mean'``, ``'none'`` or ``None``)
    is rejected rather than silently ignored, and ``num_templates_per_identity``
    is required. It also requires ``use_template=True`` and is not compatible
    with ``use_deployment_evaluation``.

    Returns the resolved
    ``(enrollment_template_mode, num_templates_per_identity,
    template_selection_method, template_score_aggregation)``.
    """
    if enrollment_template_mode not in {"fusion", "multi_template"}:
        raise ValueError(
            f"{task_name}: 'enrollment_template_mode' must be 'fusion' or "
            f"'multi_template', received {enrollment_template_mode!r}."
        )

    if enrollment_template_mode == "fusion":
        conflicting = [
            name
            for name, value in (
                ("num_templates_per_identity", num_templates_per_identity),
                ("template_selection_method", template_selection_method),
                ("template_score_aggregation", template_score_aggregation),
            )
            if value is not None
        ]
        if conflicting:
            raise ValueError(
                f"{task_name}: enrollment_template_mode='fusion' does not "
                "use " + ", ".join(conflicting) + ". Remove them or set "
                "enrollment_template_mode='multi_template'."
            )
        return (
            enrollment_template_mode,
            None,
            None,
            None,
        )

    # enrollment_template_mode == "multi_template"
    if template_fusion_method is not _UNSET:
        raise ValueError(
            f"{task_name}: enrollment_template_mode='multi_template' "
            "selects enrollment templates directly and does not use "
            "'template_fusion_method'. Remove it from the configuration."
        )

    if not use_template:
        raise ValueError(
            f"{task_name}: enrollment_template_mode='multi_template' "
            "requires use_template=True."
        )

    if num_templates_per_identity is None:
        raise ValueError(
            f"{task_name}: enrollment_template_mode='multi_template' "
            "requires 'num_templates_per_identity'."
        )

    if (
        isinstance(num_templates_per_identity, (bool, np.bool_))
        or not isinstance(num_templates_per_identity, (int, np.integer))
        or int(num_templates_per_identity) < 1
    ):
        raise ValueError(
            f"{task_name}: 'num_templates_per_identity' must be a positive "
            f"integer, received {num_templates_per_identity!r}."
        )

    resolved_k = int(num_templates_per_identity)

    resolved_selection_method = (
        "farthest_first_cosine"
        if template_selection_method is None
        else template_selection_method
    )
    if resolved_selection_method != "farthest_first_cosine":
        raise ValueError(
            f"{task_name}: 'template_selection_method' must be "
            "'farthest_first_cosine', received "
            f"{template_selection_method!r}."
        )

    resolved_aggregation = (
        "max"
        if template_score_aggregation is None
        else template_score_aggregation
    )
    if resolved_aggregation != "max":
        raise ValueError(
            f"{task_name}: 'template_score_aggregation' must be 'max', "
            f"received {template_score_aggregation!r}."
        )

    if use_deployment_evaluation:
        raise ValueError(
            f"{task_name}: enrollment_template_mode='multi_template' is "
            "not compatible with use_deployment_evaluation. Identity-level "
            "aggregation over multiple stored templates changes the score "
            "distribution a deployment threshold is calibrated on. "
            "Calibrate with enrollment_template_mode='fusion', or disable "
            "deployment evaluation."
        )

    return (
        enrollment_template_mode,
        resolved_k,
        resolved_selection_method,
        resolved_aggregation,
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
    x_enrollment=None,
    y_enrollment=None,
    minimum_enrollment_samples=1,
):
    """
    Synchronize known identities across training, enrollment, and probe data.

    The first and second session arguments retain their historical meaning as
    training and probe inputs. When a separate enrollment partition is omitted,
    enrollment reuses the training partition. This preserves the two-partition
    interface while permitting an independent enrollment source.

    Original sample order is preserved within every role.
    """
    x_train = np.asarray(
        x_session_1
    )
    y_train = np.asarray(
        y_session_1
    )
    x_probe = np.asarray(
        x_session_2
    )
    y_probe = np.asarray(
        y_session_2
    )

    if (
        (x_enrollment is None)
        != (y_enrollment is None)
    ):
        raise ValueError(
            "x_enrollment and y_enrollment must either both "
            "be provided or both be omitted."
        )

    enrollment_reuses_training = (
        x_enrollment is None
    )

    if enrollment_reuses_training:
        x_enrollment = x_train
        y_enrollment = y_train
    else:
        x_enrollment = np.asarray(
            x_enrollment
        )
        y_enrollment = np.asarray(
            y_enrollment
        )

    role_arrays = (
        (
            "Session 1"
            if enrollment_reuses_training
            else "Training",
            x_train,
            y_train,
        ),
        (
            "Enrollment",
            x_enrollment,
            y_enrollment,
        ),
        (
            "Session 2"
            if enrollment_reuses_training
            else "Probe",
            x_probe,
            y_probe,
        ),
    )

    for role_name, role_x, role_y in role_arrays:
        if role_x.ndim < 1:
            raise ValueError(
                f"{role_name} samples must "
                "contain a sample dimension."
            )

        if role_y.ndim != 1:
            raise ValueError(
                f"{role_name} labels must "
                "be one-dimensional."
            )

        if len(role_x) != len(role_y):
            raise ValueError(
                f"{role_name} samples and "
                "labels are misaligned."
            )

    minimum_values = {
        "minimum_session_1_samples": (
            minimum_session_1_samples
        ),
        "minimum_enrollment_samples": (
            minimum_enrollment_samples
        ),
        "minimum_session_2_samples": (
            minimum_session_2_samples
        ),
    }

    normalized_minimums = {}

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

        value = int(value)

        if value < 1:
            raise ValueError(
                f"{parameter_name} must be a positive integer."
            )

        normalized_minimums[
            parameter_name
        ] = value

    minimum_train_samples = (
        normalized_minimums[
            "minimum_session_1_samples"
        ]
    )

    minimum_enrollment_samples = (
        normalized_minimums[
            "minimum_enrollment_samples"
        ]
    )

    minimum_probe_samples = (
        normalized_minimums[
            "minimum_session_2_samples"
        ]
    )

    train_subjects = np.unique(
        y_train
    )

    enrollment_subjects = np.unique(
        y_enrollment
    )

    probe_subjects = np.unique(
        y_probe
    )

    common_subjects = np.intersect1d(
        np.intersect1d(
            train_subjects,
            enrollment_subjects,
        ),
        probe_subjects,
    )

    if len(common_subjects) < 2:
        raise ValueError(
            "At least two identities must be shared by "
            "the required cross-session roles."
        )

    train_counts = {
        subject: int(
            np.sum(
                y_train == subject
            )
        )
        for subject in common_subjects
    }

    enrollment_counts = {
        subject: int(
            np.sum(
                y_enrollment == subject
            )
        )
        for subject in common_subjects
    }

    probe_counts = {
        subject: int(
            np.sum(
                y_probe == subject
            )
        )
        for subject in common_subjects
    }

    insufficient_train = np.asarray(
        [
            subject
            for subject in common_subjects
            if (
                train_counts[subject]
                < minimum_train_samples
            )
        ],
        dtype=common_subjects.dtype,
    )

    insufficient_enrollment = np.asarray(
        [
            subject
            for subject in common_subjects
            if (
                enrollment_counts[subject]
                < minimum_enrollment_samples
            )
        ],
        dtype=common_subjects.dtype,
    )

    insufficient_probe = np.asarray(
        [
            subject
            for subject in common_subjects
            if (
                probe_counts[subject]
                < minimum_probe_samples
            )
        ],
        dtype=common_subjects.dtype,
    )

    eligible_subjects = np.asarray(
        [
            subject
            for subject in common_subjects
            if (
                train_counts[subject]
                >= minimum_train_samples
                and enrollment_counts[subject]
                >= minimum_enrollment_samples
                and probe_counts[subject]
                >= minimum_probe_samples
            )
        ],
        dtype=common_subjects.dtype,
    )

    if len(eligible_subjects) < 2:
        raise ValueError(
            "At least two shared identities must satisfy "
            "the required cross-session sample counts."
        )

    train_mask = np.isin(
        y_train,
        eligible_subjects,
    )

    enrollment_mask = np.isin(
        y_enrollment,
        eligible_subjects,
    )

    probe_mask = np.isin(
        y_probe,
        eligible_subjects,
    )

    train_partition = (
        x_train[train_mask],
        y_train[train_mask],
    )

    enrollment_partition = (
        x_enrollment[enrollment_mask],
        y_enrollment[enrollment_mask],
    )

    probe_partition = (
        x_probe[probe_mask],
        y_probe[probe_mask],
    )

    expected_subjects = set(
        eligible_subjects.tolist()
    )

    for role_name, role_partition in (
        ("training", train_partition),
        ("enrollment", enrollment_partition),
        ("probe", probe_partition),
    ):
        retained_subjects = set(
            np.unique(
                role_partition[1]
            ).tolist()
        )

        if retained_subjects != expected_subjects:
            raise RuntimeError(
                f"{role_name.capitalize()} partitioning "
                "did not preserve exactly the eligible identities."
            )

    all_subjects = np.union1d(
        np.union1d(
            train_subjects,
            enrollment_subjects,
        ),
        probe_subjects,
    )

    train_indices = np.flatnonzero(
        train_mask
    )

    enrollment_indices = np.flatnonzero(
        enrollment_mask
    )

    probe_indices = np.flatnonzero(
        probe_mask
    )

    return {
        "train": train_partition,
        "enrollment": enrollment_partition,
        "probe": probe_partition,

        # Backward-compatible aliases for callers that still use the
        # historical two-session vocabulary.
        "session_1": train_partition,
        "session_2": probe_partition,

        "indices": {
            "train": train_indices,
            "enrollment": enrollment_indices,
            "probe": probe_indices,
            "session_1": train_indices,
            "session_2": probe_indices,
        },
        "subjects": eligible_subjects,
        "enrollment_reuses_training": (
            enrollment_reuses_training
        ),
        "dropped_subjects": {
            "missing_training": np.setdiff1d(
                all_subjects,
                train_subjects,
            ),
            "missing_enrollment": np.setdiff1d(
                all_subjects,
                enrollment_subjects,
            ),
            "missing_probe": np.setdiff1d(
                all_subjects,
                probe_subjects,
            ),
            "insufficient_training": (
                insufficient_train
            ),
            "insufficient_enrollment": (
                insufficient_enrollment
            ),
            "insufficient_probe": (
                insufficient_probe
            ),

            # Backward-compatible diagnostics.
            "session_1_only": np.setdiff1d(
                train_subjects,
                probe_subjects,
            ),
            "session_2_only": np.setdiff1d(
                probe_subjects,
                train_subjects,
            ),
            "insufficient_session_1": (
                insufficient_train
            ),
            "insufficient_session_2": (
                insufficient_probe
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

    def subject_indices(subjects):
        return np.flatnonzero(np.isin(y, subjects))

    return {
        "train": train_partition,
        "validation": validation_partition,
        "test": test_partition,
        "indices": {
            "train": subject_indices(train_subjects),
            "validation": (
                subject_indices(validation_subjects)
                if len(validation_subjects) > 0
                else np.empty((0,), dtype=int)
            ),
            "test": subject_indices(test_subjects),
        },
    }



def _partition_subject_disjoint_cross_session_samples(
    x_s1,
    y_s1,
    x_s2,
    y_s2,
    train_subjects,
    validation_subjects,
    test_subjects,
    x_enrollment=None,
    y_enrollment=None,
):
    """
    Assign subject-disjoint training, enrollment, and probe partitions.

    The historical two-session interface remains valid: when a separate
    enrollment array is omitted, held-out identities enroll from the training
    source. When enrollment data is supplied explicitly, held-out identities
    enroll from that source instead.

    Representation-learning and validation samples always come from the
    training role. Probe samples always come from the probe role.
    """
    x_train = np.asarray(
        x_s1
    )
    y_train = np.asarray(
        y_s1
    )
    x_probe = np.asarray(
        x_s2
    )
    y_probe = np.asarray(
        y_s2
    )

    if (
        (x_enrollment is None)
        != (y_enrollment is None)
    ):
        raise ValueError(
            "x_enrollment and y_enrollment must either both "
            "be provided or both be omitted."
        )

    enrollment_reuses_training = (
        x_enrollment is None
    )

    if enrollment_reuses_training:
        x_enrollment = x_train
        y_enrollment = y_train
    else:
        x_enrollment = np.asarray(
            x_enrollment
        )
        y_enrollment = np.asarray(
            y_enrollment
        )

    role_arrays = (
        (
            "Session 1"
            if enrollment_reuses_training
            else "Training",
            x_train,
            y_train,
        ),
        (
            "Enrollment",
            x_enrollment,
            y_enrollment,
        ),
        (
            "Session 2"
            if enrollment_reuses_training
            else "Probe",
            x_probe,
            y_probe,
        ),
    )

    for role_name, role_x, role_y in role_arrays:
        if role_y.ndim != 1:
            raise ValueError(
                f"{role_name} labels must be one-dimensional."
            )

        if len(role_x) != len(role_y):
            raise ValueError(
                f"{role_name} samples and labels are misaligned."
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
            np.unique(
                y_train
            ).tolist()
        )
        & set(
            np.unique(
                y_enrollment
            ).tolist()
        )
        & set(
            np.unique(
                y_probe
            ).tolist()
        )
    )

    assigned_subjects = (
        train_set
        | validation_set
        | test_set
    )

    if assigned_subjects != common_subjects:
        if enrollment_reuses_training:
            raise ValueError(
                "The subject cohorts must cover exactly the "
                "subjects shared by Session 1 and Session 2."
            )

        raise ValueError(
            "The subject cohorts must cover exactly the "
            "identities shared by training, enrollment, "
            "and probe roles."
        )

    train_mask = np.isin(
        y_train,
        train_subjects,
    )

    validation_mask = np.isin(
        y_train,
        validation_subjects,
    )

    enrollment_mask = np.isin(
        y_enrollment,
        test_subjects,
    )

    probe_mask = np.isin(
        y_probe,
        test_subjects,
    )

    train_partition = (
        x_train[train_mask],
        y_train[train_mask],
    )

    if len(validation_subjects) > 0:
        validation_partition = (
            x_train[validation_mask],
            y_train[validation_mask],
        )
    else:
        validation_partition = (
            None,
            None,
        )

    enrollment_partition = (
        x_enrollment[enrollment_mask],
        y_enrollment[enrollment_mask],
    )

    probe_partition = (
        x_probe[probe_mask],
        y_probe[probe_mask],
    )

    return {
        "train": train_partition,
        "validation": validation_partition,
        "enrollment": enrollment_partition,
        "probe": probe_partition,
        "indices": {
            "train": np.flatnonzero(
                train_mask
            ),
            "validation": (
                np.flatnonzero(
                    validation_mask
                )
                if len(
                    validation_subjects
                ) > 0
                else np.empty(
                    (0,),
                    dtype=int,
                )
            ),
            "enrollment": np.flatnonzero(
                enrollment_mask
            ),
            "probe": np.flatnonzero(
                probe_mask
            ),
        },
    }

def _split_enrollment_probe_embeddings(
    embeddings,
    labels,
    subjects,
    template_size,
    provenance=None,
):
    """
    Split each subject's embeddings into an enrollment gallery and a probe set.

    Without ``provenance`` the first ``template_size`` embeddings of each
    subject, in input order, enrol and the rest probe (backward-compatible
    behavior). With ``provenance`` the enrollment *membership* is the genuinely
    first ``template_size`` beats in source order; the remaining beats probe in
    their original input order. The returned ``indices["enrollment"]`` and
    ``indices["probe"]`` align exactly with the returned embedding/label
    arrays, so a caller can subset a parallel array such as provenance.
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

    if provenance is not None:
        provenance.validate(len(labels))

    enrollment_embeddings = []
    enrollment_labels = []

    probe_embeddings = []
    probe_labels = []

    enrollment_index_blocks = []
    probe_index_blocks = []

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

        if provenance is None:
            # Backward-compatible input-order behavior: the first template_size
            # embeddings of the subject enrol.
            enrollment_indices = subject_indices[:template_size]
            probe_indices = subject_indices[template_size:]
        else:
            # Enrol the genuinely first template_size beats in source order;
            # remaining beats stay probes in their original input order.
            source_order = _source_order_indices(
                provenance.subset(subject_indices)
            )
            enrollment_indices = subject_indices[source_order][:template_size]
            enrolled = set(enrollment_indices.tolist())
            probe_indices = np.asarray(
                [index for index in subject_indices if index not in enrolled],
                dtype=int,
            )

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

        enrollment_index_blocks.append(enrollment_indices)
        probe_index_blocks.append(probe_indices)

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
        "indices": {
            "enrollment": np.concatenate(enrollment_index_blocks),
            "probe": np.concatenate(probe_index_blocks),
        },
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

    call_args[
        "reproducibility_mode"
    ] = _prepare_reproducibility_backend(
        call_args.get(
            "reproducibility_mode",
            "seeded",
        ),
        call_args.get("device"),
    )

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

def _collect_validation_arrays(val_split, scope):
    """
    Gather the validation arrays that can influence the selected/final
    weights, for the weight-cache fingerprint.

    Validation affects the returned weights through best-checkpoint
    selection, learning-rate rollback and early stopping. When validation is
    inactive (``val_split`` not positive) this returns ``None`` so the
    weight-cache identity is unchanged. Only arrays present in the caller's
    scope and not ``None`` are included, which covers the seen validation
    (all tasks) and the unseen-subject / session validation (subject-disjoint
    and cross-session tasks).
    """
    try:
        active = float(val_split) > 0.0
    except (TypeError, ValueError):
        active = False

    if not active:
        return None

    candidate_names = (
        "X_val",
        "y_val",
        "X_val_seen",
        "y_val_seen",
        "X_val_s1",
        "y_val_s1_enc",
    )
    collected = {
        name: scope[name]
        for name in candidate_names
        if scope.get(name) is not None
    }
    return collected or None


def _build_weight_cache_config(
    loader,
    training_config,
    training_samples=None,
    training_labels=None,
    validation_arrays=None,
    training_role_only_loader_identity=False,
):
    """
    Build the complete identity for reusable trained model weights.

    A training-role-only loader identity may be requested when enrollment and
    probe source selectors affect evaluation but cannot affect fitted weights.

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

    if validation_arrays:
        # Validation can change the selected/final weights (best-checkpoint
        # selection, LR rollback, early stopping); its arrays therefore
        # belong to the identity when validation is active. The field is
        # omitted entirely when inactive, preserving existing identities.
        fingerprintable = {
            name: array
            for name, array in validation_arrays.items()
            if array is not None
        }
        if fingerprintable:
            complete_config[
                "validation_partition"
            ] = _fingerprint_array_collection(fingerprintable)
    loader_identity = _build_loader_cache_identity(
        loader
    )

    if (
        training_role_only_loader_identity
        and isinstance(loader_identity, dict)
    ):
        loader_identity = copy.deepcopy(
            loader_identity
        )

        settings = loader_identity.get(
            "settings"
        )

        if isinstance(settings, dict):
            evaluation_only_settings = {
                "enroll_sessions",
                "enrol_sessions",
                "probe_sessions",
                "required_cross_sessions",
                "enrol_parts",
                "enroll_parts",
                "test_parts",
                "enroll_record_indices",
                "probe_record_indices",
            }

            for setting_name in (
                evaluation_only_settings
            ):
                settings.pop(
                    setting_name,
                    None,
                )

    complete_config[
        "loader_identity"
    ] = loader_identity

    compatibility_identity = (
        build_weight_compatibility_identity()
    )
    complete_config[
        "implementation_identity"
    ] = compatibility_identity[
        "implementation"
    ]
    complete_config[
        "dependency_identity"
    ] = compatibility_identity[
        "dependencies"
    ]

    return complete_config


def _build_weight_artifact_context(
    training_samples,
    num_classes,
    resolved_split_seed,
    reproducibility_state,
):
    """Describe runner-known training facts excluded from cache identity."""
    return {
        "model_constructor_arguments": {
            "in_channels": int(
                _detect_channels(training_samples)
            ),
            "num_classes": int(num_classes),
            "include_top": True,
        },
        "resolved_split_seed": int(resolved_split_seed),
        "reproducibility_state": copy.deepcopy(
            reproducibility_state
        ),
        "training_components": {
            "optimizer": "torch.optim.Adam",
            "loss": "torch.nn.CrossEntropyLoss",
            "scheduler": "framework rollback learning-rate policy",
        },
    }

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


def _aggregate_multi_run_metrics(
    results,
):
    """
    Aggregate seed-level metrics while preserving unavailable values.

    A metric is unavailable at aggregate level when any contributing run is
    unavailable or non-finite; only complete run sets are summarized.
    """
    rows = [
        tuple(result)
        for result in results
    ]

    if not rows:
        return tuple()

    metric_count = len(
        rows[0]
    )

    if any(
        len(row) != metric_count
        for row in rows
    ):
        raise ValueError(
            "Every multi-run result must contain "
            "the same number of metrics."
        )

    aggregates = []

    for metric_values in zip(
        *rows
    ):
        numeric_values = []

        for value in metric_values:
            if value is None:
                numeric_values = None
                break

            try:
                numeric_value = float(
                    value
                )
            except (
                TypeError,
                ValueError,
            ):
                numeric_values = None
                break

            if not np.isfinite(
                numeric_value
            ):
                numeric_values = None
                break

            numeric_values.append(
                numeric_value
            )

        if numeric_values is None:
            aggregates.append(
                (
                    None,
                    None,
                )
            )
            continue

        values = np.asarray(
            numeric_values,
            dtype=float,
        )

        aggregates.append(
            (
                float(
                    np.mean(
                        values
                    )
                ),
                float(
                    np.std(
                        values
                    )
                ),
            )
        )

    return tuple(
        aggregates
    )


def _format_multi_run_metric(
    aggregate,
):
    """
    Format one multi-run metric without fabricating unavailable values.
    """
    mean_value, std_value = aggregate

    if (
        mean_value is None
        or std_value is None
    ):
        return None

    return (
        f"{mean_value:.4f} "
        "\u00b1 "
        f"{std_value:.4f}"
    )



def _apply_identification_metric_reportability(
    results,
    artifacts,
):
    """
    Mask seed-level Rank-5 values that are not suitable for headline reporting.

    Rank-1 remains available. Rank-5 is represented as None when its
    identification artifact marks it as nonreportable, allowing the shared
    missing-aware aggregation path to preserve that status across runs.
    """
    result_rows = [
        tuple(result)
        for result in results
    ]
    artifact_rows = list(
        artifacts
    )

    if len(result_rows) != len(
        artifact_rows
    ):
        raise ValueError(
            "Identification results and artifacts "
            "must contain the same number of runs."
        )

    reportable_rows = []

    for result, artifact in zip(
        result_rows,
        artifact_rows,
    ):
        if len(result) != 2:
            raise ValueError(
                "Each identification result must "
                "contain Rank-1 and Rank-5."
            )

        if (
            not isinstance(
                artifact,
                Mapping,
            )
            or artifact.get("type")
            != "identification"
        ):
            raise ValueError(
                "Each identification result requires "
                "a matching identification artifact."
            )

        if (
            "rank_5_reportable"
            not in artifact
        ):
            raise ValueError(
                "Identification artifact is missing "
                "Rank-5 reportability metadata."
            )

        rank_5 = (
            result[1]
            if bool(
                artifact[
                    "rank_5_reportable"
                ]
            )
            else None
        )

        reportable_rows.append(
            (
                result[0],
                rank_5,
            )
        )

    return reportable_rows


def _build_per_run_results(
    results,
    seeds,
    metric_names=None,
    split_seeds=None,
    data_statistics=None,
    trained_weight_references=None,
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

    provenance_sequences = (
        split_seeds,
        data_statistics,
        trained_weight_references,
    )
    provenance_supplied = [
        sequence is not None
        for sequence in provenance_sequences
    ]

    if any(provenance_supplied) and not all(provenance_supplied):
        raise ValueError(
            "Split seeds, data statistics, and trained-weight references "
            "must be supplied together."
        )

    complete_provenance = all(provenance_supplied)

    if complete_provenance:
        split_seeds = list(split_seeds)
        data_statistics = list(data_statistics)
        trained_weight_references = list(trained_weight_references)

        sequence_lengths = {
            len(results),
            len(seeds),
            len(split_seeds),
            len(data_statistics),
            len(trained_weight_references),
        }
        if len(sequence_lengths) != 1:
            raise ValueError(
                "Multi-run results, seeds, resolved split seeds, data "
                "statistics, and trained-weight references must have "
                "matching lengths."
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

        run_record = {
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

        if complete_provenance:
            statistics = data_statistics[run_index - 1]
            weight_reference = trained_weight_references[run_index - 1]

            if not isinstance(statistics, Mapping):
                raise ValueError(
                    "Every per-run data-statistics value must be a mapping."
                )
            if not isinstance(weight_reference, Mapping):
                raise ValueError(
                    "Every trained-weight reference must be a mapping."
                )

            run_record.update(
                {
                    "split_seed": int(
                        split_seeds[run_index - 1]
                    ),
                    "data_statistics": copy.deepcopy(
                        _to_json_compatible(statistics)
                    ),
                    "trained_weight": copy.deepcopy(
                        _to_json_compatible(weight_reference)
                    ),
                }
            )

        per_run_results.append(run_record)

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
    raw_values = list(
        values
    )

    if not raw_values:
        return None

    numeric_values = []

    for value in raw_values:
        if value is None:
            return None

        try:
            numeric_value = float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            return None

        if not np.isfinite(
            numeric_value
        ):
            return None

        numeric_values.append(
            numeric_value
        )

    values = np.asarray(
        numeric_values,
        dtype=float,
    )

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

_TRAINED_WEIGHT_REFERENCE_SINK = ContextVar(
    "_TRAINED_WEIGHT_REFERENCE_SINK",
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


def _is_lowercase_sha256(value):
    """Return whether value is one lowercase hexadecimal SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _build_persisted_trained_weight_reference(
    cache_metadata,
    source,
):
    """Build a compact reference from validated immutable metadata."""
    if source not in {
        "cache_hit",
        "trained_and_saved",
    }:
        raise ValueError(
            f"Unsupported persisted trained-weight source: {source!r}."
        )
    if not isinstance(cache_metadata, Mapping):
        raise RuntimeError(
            "Validated trained-weight cache metadata is unavailable."
        )

    artifact_metadata = cache_metadata.get("weight_artifact")
    identity = (
        artifact_metadata.get("identity")
        if isinstance(artifact_metadata, Mapping)
        else None
    )
    if not isinstance(identity, Mapping):
        raise RuntimeError(
            "Validated trained-weight artifact identity is unavailable."
        )

    weight_uid = identity.get("weight_uid")
    state_hash_format = identity.get("state_dict_hash_format")
    state_hash = identity.get("state_dict_sha256")
    payload_hash = identity.get("payload_sha256")

    if not isinstance(weight_uid, str) or not weight_uid:
        raise RuntimeError("Validated trained-weight UID is unavailable.")
    if state_hash_format != STATE_DICT_HASH_FORMAT:
        raise RuntimeError(
            "Validated trained-weight state hash format is unsupported."
        )
    if not _is_lowercase_sha256(state_hash):
        raise RuntimeError(
            "Validated trained-weight state checksum is unavailable."
        )
    if not _is_lowercase_sha256(payload_hash):
        raise RuntimeError(
            "Validated trained-weight payload checksum is unavailable."
        )

    return {
        "persisted": True,
        "source": source,
        "weight_uid": weight_uid,
        "state_dict_hash_format": state_hash_format,
        "state_dict_sha256": state_hash,
        "payload_sha256": payload_hash,
    }


def _build_in_memory_trained_weight_reference(model):
    """Identify the final model state without claiming persistence."""
    return {
        "persisted": False,
        "source": "trained_not_persisted",
        "weight_uid": None,
        "state_dict_hash_format": STATE_DICT_HASH_FORMAT,
        "state_dict_sha256": canonical_state_dict_sha256(
            model.state_dict()
        ),
        "payload_sha256": None,
    }


def _record_trained_weight_reference(reference):
    """Record one defensive run-local weight reference when a sink is active."""
    sink = _TRAINED_WEIGHT_REFERENCE_SINK.get()
    if sink is not None:
        sink.append(copy.deepcopy(reference))


def _record_persisted_trained_weight(cache_metadata, source):
    """Record validated persisted weight metadata only when requested."""
    if _TRAINED_WEIGHT_REFERENCE_SINK.get() is None:
        return
    _record_trained_weight_reference(
        _build_persisted_trained_weight_reference(
            cache_metadata,
            source,
        )
    )


def _record_newly_saved_trained_weight(cache, uid):
    """Validate a new cache artifact and record its compact identity."""
    if _TRAINED_WEIGHT_REFERENCE_SINK.get() is None:
        return
    cache_metadata = cache.get_weight_artifact_metadata(uid)
    if cache_metadata is None:
        raise RuntimeError(
            "Newly saved trained-weight metadata could not be validated."
        )
    _record_persisted_trained_weight(
        cache_metadata,
        "trained_and_saved",
    )


def _record_in_memory_trained_weight(model):
    """Record the final tensor identity without creating a cache artifact."""
    if _TRAINED_WEIGHT_REFERENCE_SINK.get() is None:
        return
    _record_trained_weight_reference(
        _build_in_memory_trained_weight_reference(model)
    )


def _run_recursive_with_provenance(
    runner,
    call_args,
    evaluation_artifact_sink,
):
    """Execute one recursive run with isolated provenance sinks."""
    trained_weight_sink = []
    evaluation_token = _EVALUATION_ARTIFACT_SINK.set(
        evaluation_artifact_sink
    )
    trained_weight_token = _TRAINED_WEIGHT_REFERENCE_SINK.set(
        trained_weight_sink
    )
    try:
        recursive_result = runner(**call_args)
    finally:
        _TRAINED_WEIGHT_REFERENCE_SINK.reset(trained_weight_token)
        _EVALUATION_ARTIFACT_SINK.reset(evaluation_token)

    if len(trained_weight_sink) != 1:
        raise RuntimeError(
            "Each successful recursive run must record exactly one "
            "trained-weight reference."
        )

    return recursive_result, copy.deepcopy(trained_weight_sink[0])


def _add_seed_metadata(
    hyperparams,
    base_seed,
    n_runs,
    split_seed=None,
):
    """
    Add the complete random-seed schedule to experiment metadata.

    Multi-run experiments use consecutive training seeds beginning at
    ``base_seed``. ``split_seed`` records the data-role allocation policy:
    when ``None`` the split follows the per-run training seed
    (``split_seed_policy = "follow_seed"``); otherwise the partition is held
    fixed at the configured value (``"fixed"``). Field names are neutral; the
    manuscript supplies any interpretation.

    A copied dictionary is returned so the caller's original metadata is not
    modified unexpectedly.
    """
    base_seed = int(base_seed)
    n_runs = int(n_runs)

    if n_runs < 1:
        raise ValueError(
            "n_runs must be greater than or equal to 1."
        )

    run_seeds = [
        base_seed + run_index
        for run_index in range(n_runs)
    ]

    if split_seed is None:
        resolved_split_seeds = list(run_seeds)
        split_seed_policy = "follow_seed"
        configured_split_seed = None
    else:
        configured_split_seed = int(split_seed)
        resolved_split_seeds = [configured_split_seed] * n_runs
        split_seed_policy = "fixed"

    updated_hyperparams = dict(hyperparams)

    updated_hyperparams.update(
        {
            "base_seed": base_seed,
            "n_runs": n_runs,
            "run_seeds": run_seeds,
            "configured_split_seed": configured_split_seed,
            "resolved_split_seeds": resolved_split_seeds,
            "split_seed_policy": split_seed_policy,
        }
    )

    return updated_hyperparams

# =============================================================================
# TASK 1: CLOSED-SET IDENTIFICATION
# =============================================================================
def run_closed_set_identification(x, y, model_class, epochs=150, batch_size=256, 
                                  lr=1e-3, test_split=0.2, val_split=0.0, seed=42, 
                                  device=None, visualize=False, use_template=False, 
                                  template_fusion_method=_UNSET, template_size=None,
                                  matching_method='cosine', outlier_filtering_on_train=False,
                                  outlier_filtering_on_test=False, sqi_scores=None,
                                  sqi_threshold=0.05, sqi_keep_pct=0.8, probe_fusion_size=1,
                                  save_results_and_settings=False, loader=None, 
                                  n_runs=1, _return_stats=False,
                                  intelligent_weight_loading=True,
                                  augmentation_config=None, split_seed=None, provenance=None,
        enrollment_template_mode='fusion', num_templates_per_identity=None,
        template_selection_method=None, template_score_aggregation=None,
        reproducibility_mode="seeded"):
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
        seed (int): Training/general stochastic seed controlling model initialization, training DataLoader shuffling, augmentation, and other stochastic operations not governed by split_seed (such as validation-EER pair sampling). When split_seed is None, seed also supplies the resolved data-role allocation seed.
        split_seed (int or None): Optional seed for randomized data-role allocation such as train/test beat splits, subject-cohort splits, and validation allocation. None follows the current per-run training seed; an explicit integer holds the randomized partition fixed across training seeds where such a partition exists.
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
        probe_fusion_size (int): Probe fusion depth. With 1, each probe beat
            yields one decision. With k>1, complete non-overlapping groups of k
            probe beats within one source block (subject, session, record,
            segment), ordered by source provenance, are averaged into one
            decision; an incomplete final group is dropped.
        save_results_and_settings (bool): If True, logs results and parameters to a text file.
        loader (object): Dataset loader instance (used for extracting metadata for logging).
        n_runs (int): Number of repeated runs using consecutive training seeds. The data-role split schedule follows the split_seed policy (it follows the training seeds when split_seed is None, and stays fixed when an explicit split_seed is given).
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.
        reproducibility_mode (str): Reproducibility policy ('seeded' or 'strict').

    Returns:
        tuple: (Rank-1 Accuracy, Rank-5 Accuracy)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for both metrics.
    """

    reproducibility_mode = _activate_top_level_runtime_profile(
        reproducibility_mode,
        device,
        recursive_run=_return_stats,
    )

    (
        enrollment_template_mode,
        num_templates_per_identity,
        template_selection_method,
        template_score_aggregation,
    ) = _validate_multi_template_arguments(
        enrollment_template_mode,
        template_fusion_method,
        num_templates_per_identity,
        template_selection_method,
        template_score_aggregation,
        use_template,
        False,
        "Closed-Set Identification",
    )

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
        split_seed=split_seed,
    )

    # 2. MULTI-RUN AGGREGATOR (Handles statistical validation)
    if n_runs > 1:
        # Capture current arguments to repeat the experiment
        call_args = _prepare_multi_run_arguments(locals())
        base_seed = call_args.get('seed', 42)
        results = []
        evaluation_artifact_runs = []
        data_statistics_runs = []
        trained_weight_references = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs) for Statistical Validation...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False # Prevent 5 pop-up windows
            
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            
            # Recursive call to execute a single seed
            run_wall_clock_started = _start_runtime_stage()
            (
                (res, d_stats, h_params),
                trained_weight_reference,
            ) = _run_recursive_with_provenance(
                run_closed_set_identification,
                call_args,
                evaluation_artifact_runs,
            )
            _record_multi_run_time(
                run_index=i + 1,
                seed=call_args["seed"],
                started_at=run_wall_clock_started,
            )
            
            results.append(res)
            data_statistics_runs.append(copy.deepcopy(d_stats))
            trained_weight_references.append(trained_weight_reference)
            # Preserve metadata from the last successful run for the final log file
            data_stats = d_stats  
            hyperparams = h_params 

        hyperparams = _add_seed_metadata(
            hyperparams,
            base_seed=base_seed,
            n_runs=n_runs,
            split_seed=split_seed,
        )

        reportable_results = (
            _apply_identification_metric_reportability(
                results,
                evaluation_artifact_runs,
            )
        )

        per_run_results = (
            _build_per_run_results(
                results=reportable_results,
                seeds=hyperparams[
                    "run_seeds"
                ],
                split_seeds=hyperparams[
                    "resolved_split_seeds"
                ],
                data_statistics=data_statistics_runs,
                trained_weight_references=trained_weight_references,
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
        metric_aggregates = (
            _aggregate_multi_run_metrics(
                reportable_results
            )
        )
        
        print(f"\n[MULTI-RUN RESULTS | {n_runs} Runs]")
        print(
            "Rank-1 Acc: "
            f"{_format_multi_run_metric(metric_aggregates[0]) or 'N/A'} | "
            "Rank-5 Acc: "
            f"{_format_multi_run_metric(metric_aggregates[1]) or 'N/A'}"
        )
        
        if save_results_and_settings:
            metrics_dict = {
                "Rank-1 Accuracy": _format_multi_run_metric(
                    metric_aggregates[0]
                ),
                "Rank-5 Accuracy": _format_multi_run_metric(
                    metric_aggregates[1]
                ),
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
        return tuple(metric_aggregates)

    augmentation_config = (
        _normalize_augmentation_config(
            augmentation_config
        )
    )

    hyperparams[
        "augmentation"
    ] = augmentation_config

    if enrollment_template_mode == "fusion":
        if template_fusion_method is _UNSET:
            template_fusion_method = "mean"
    else:
        template_fusion_method = None

    hyperparams["enrollment_template_mode"] = enrollment_template_mode
    hyperparams["template_fusion_method"] = template_fusion_method

    if enrollment_template_mode == "multi_template":
        hyperparams["num_templates_per_identity"] = num_templates_per_identity
        hyperparams["template_selection_method"] = template_selection_method
        hyperparams["template_score_aggregation"] = template_score_aggregation

    device, reproducibility_state = _setup_reproducibility(
        seed=seed,
        device=device,
        reproducibility_mode=reproducibility_mode,
    )
    _set_active_profiling_device_type(device)
    hyperparams["reproducibility_mode"] = reproducibility_mode
    hyperparams["reproducibility_state"] = reproducibility_state
    resolved_split_seed = seed if split_seed is None else split_seed
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
    if provenance is not None:
        provenance = provenance.subset(valid_mask)

    # ====================================================
    # 3. SPLIT DATA & SQI SCORES
    # ====================================================
    train_test_partitions = (
        _split_closed_set_samples(
            x,
            y,
            holdout_split=test_split,
            seed=resolved_split_seed,
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

    provenance_train = None
    provenance_test = None
    if provenance is not None:
        provenance_train = provenance.subset(
            train_test_partitions["indices"]["retained"]
        )
        provenance_test = provenance.subset(
            train_test_partitions["indices"]["holdout"]
        )

    # ====================================================
    # 4. APPLY FILTERS INDEPENDENTLY
    # ====================================================
    if outlier_filtering_on_train and sqi_train is not None:
        print("\n[INFO] Filtering Train Set (Enrollment)...")
        if provenance_train is not None:
            X_train, y_train, retained_indices = _apply_outlier_filter(
                X_train, y_train, sqi_train, absolute_threshold=sqi_threshold, keep_percentage=sqi_keep_pct, return_indices=True
            )
            provenance_train = provenance_train.subset(retained_indices)
        else:
            X_train, y_train = _apply_outlier_filter(
                X_train, y_train, sqi_train, absolute_threshold=sqi_threshold, keep_percentage=sqi_keep_pct
            )

    if outlier_filtering_on_test and sqi_test is not None:
        print("\n[INFO] Filtering Test Set (Probes)...")
        if provenance_test is not None:
            X_test, y_test, retained_indices = _apply_outlier_filter(
                X_test, y_test, sqi_test, absolute_threshold=sqi_threshold, keep_percentage=sqi_keep_pct, apply_subject_ranking=False, return_indices=True
            )
            provenance_test = provenance_test.subset(retained_indices)
        else:
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
    if provenance_test is not None:
        provenance_test = provenance_test.subset(test_mask)

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
            seed=resolved_split_seed,
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
        "reproducibility_mode": reproducibility_mode,
        }

        validation_arrays = _collect_validation_arrays(
            val_split, locals()
        )
        train_config = _build_weight_cache_config(
            loader,
            train_config,
            training_samples=X_tr,
            training_labels=y_tr,
            validation_arrays=validation_arrays,
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
            _record_persisted_trained_weight(
                model.weight_artifact_metadata,
                "cache_hit",
            )
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
                artifact_context=_build_weight_artifact_context(
                    X_tr,
                    len(classes),
                    resolved_split_seed,
                    reproducibility_state,
                ),
            )
            _record_newly_saved_trained_weight(cache, uid)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
        model = _run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs)
        _record_in_memory_trained_weight(model)

    # 4. Evaluation Logic
    if enrollment_template_mode == "multi_template":
        print(
            "[INFO] Building multi-template gallery "
            f"({num_templates_per_identity} templates/identity, "
            f"{template_selection_method})..."
        )
        model.include_top = False
        train_extract_loader = _make_loader(X_train, y_train_enc, batch_size, shuffle=False)
        train_emb, train_lab = _get_embeddings(model, train_extract_loader, device)
        (
            mt_templates,
            mt_template_identities,
            mt_template_source_indices,
            template_diagnostics,
        ) = _select_multi_templates(
            train_emb, train_lab,
            provenance=provenance_train,
            num_templates_per_identity=num_templates_per_identity,
            template_selection_method=template_selection_method,
            max_beats=template_size,
        )

        test_emb, test_lab = _get_embeddings(model, test_loader, device)
        raw_scores = _compute_score_matrix(test_emb, mt_templates, method=matching_method)

        if probe_fusion_size > 1:
            fused_template_scores, fused_probe_labels, fusion_diagnostics = _apply_score_fusion(
                raw_scores, test_lab, fusion_size=probe_fusion_size,
                provenance=provenance_test, return_diagnostics=True,
            )
        else:
            fused_template_scores = raw_scores
            fused_probe_labels = test_lab
            fusion_diagnostics = {
                "fusion_size": 1,
                "raw_probe_observations": int(len(test_lab)),
                "fused_probe_decisions": int(len(test_lab)),
                "dropped_remainder_observations": 0,
                "source_blocks_below_fusion_size": 0,
                "identities_without_a_fused_decision": 0,
            }

        if probe_fusion_size > 1 and len(fused_probe_labels) == 0:
            raise ValueError(
                "Probe fusion produced no complete fused decisions for any "
                "identity: with the configured probe fusion size no source "
                "segment holds enough probe beats to form a group. Reduce the "
                "probe fusion size or provide more probe data."
            )

        identity_order = np.arange(len(classes))
        final_scores = _reduce_template_scores_to_identities(
            fused_template_scores, mt_template_identities, identity_order,
            template_score_aggregation=template_score_aggregation,
        )
        final_labels = fused_probe_labels

        model.include_top = True
    elif not use_template:
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
            train_emb, train_lab, method=template_fusion_method, max_beats=template_size, provenance=provenance_train
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
    if enrollment_template_mode != "multi_template":
        final_scores, final_labels, fusion_diagnostics = _apply_score_fusion(
            final_scores,
            final_labels,
            fusion_size=probe_fusion_size,
            provenance=provenance_test,
            return_diagnostics=True,
        )
        if probe_fusion_size > 1 and len(final_labels) == 0:
            raise ValueError(
                "Probe fusion produced no complete fused decisions for any "
                "identity: with the configured probe fusion size no source "
                "segment holds enough probe beats to form a group. Reduce the "
                "probe fusion size or provide more probe data."
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
        "Probe Fusion": fusion_diagnostics,
    }
    if enrollment_template_mode == "multi_template":
        data_stats["Enrollment Templates"] = template_diagnostics

    if _return_stats:
        return (rank1, rank5), data_stats, hyperparams

    if save_results_and_settings:
        _log_experiment_results(
            task_title,
            {
                "Rank-1 Accuracy": rank1,
                "Rank-5 Accuracy": (
                    rank5
                    if evaluation_artifacts[
                        "rank_5_reportable"
                    ]
                    else None
                ),
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
                                test_split=0.2, val_split=0.0, num_pairs=None,
                                sampling_mode=None, seed=42, device=None, visualize=False,
                                use_template=False, template_fusion_method=_UNSET,
                                template_size=None, matching_method='cosine',
                                outlier_filtering_on_train=False, outlier_filtering_on_test=False, 
                                sqi_scores=None, sqi_threshold=0.05, sqi_keep_pct=0.8, 
                                use_deployment_evaluation=False, target_fars=None,
                                save_results_and_settings=False, 
                                loader=None, n_runs=1, _return_stats=False,
                                intelligent_weight_loading=True,
                                augmentation_config=None, split_seed=None, provenance=None,
        *,
        pair_sampling_budget=None,
        pair_sampling_mode=None,
        max_impostor_pairs=1000000,
        pair_sampling_seed=42,
        probe_fusion_size=1,
        enrollment_template_mode='fusion', num_templates_per_identity=None,
        template_selection_method=None, template_score_aggregation=None,
        reproducibility_mode="seeded"):
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
        pair_sampling_mode (str or None): Verification comparison strategy. Options: ['all', 'all_genuine', 'balanced', 'random'].
        pair_sampling_budget (int or None): Requested comparison budget for balanced and random sampling.
        max_impostor_pairs (int): Maximum number of impostor comparisons retained by all_genuine.
        pair_sampling_seed (int): Dedicated seed for stochastic verification-pair sampling.
        num_pairs (int or None): Legacy alias for pair_sampling_budget.
        sampling_mode (str or None): Legacy alias for pair_sampling_mode.
        seed (int): Training/general stochastic seed controlling model initialization, training DataLoader shuffling, augmentation, and other stochastic operations not governed by split_seed (such as validation-EER pair sampling). When split_seed is None, seed also supplies the resolved data-role allocation seed.
        split_seed (int or None): Optional seed for randomized data-role allocation such as train/test beat splits, subject-cohort splits, and validation allocation. None follows the current per-run training seed; an explicit integer holds the randomized partition fixed across training seeds where such a partition exists.
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
        n_runs (int): Number of repeated runs using consecutive training seeds. The data-role split schedule follows the split_seed policy (it follows the training seeds when split_seed is None, and stays fixed when an explicit split_seed is given).
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.
        reproducibility_mode (str): Reproducibility policy ('seeded' or 'strict').

    Returns:
        tuple: (EER, AUC, d-prime, TAR @ 0.1% FAR)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for all four metrics.
    """

    reproducibility_mode = _activate_top_level_runtime_profile(
        reproducibility_mode,
        device,
        recursive_run=_return_stats,
    )

    (
        enrollment_template_mode,
        num_templates_per_identity,
        template_selection_method,
        template_score_aggregation,
    ) = _validate_multi_template_arguments(
        enrollment_template_mode,
        template_fusion_method,
        num_templates_per_identity,
        template_selection_method,
        template_score_aggregation,
        use_template,
        use_deployment_evaluation,
        "Closed-Set Verification",
    )

    _validate_deployment_evaluation(
        use_deployment_evaluation,
        val_split,
        "Closed-Set Verification",
    )

    probe_fusion_size = _validate_verification_probe_fusion(
        probe_fusion_size,
        use_template,
        use_deployment_evaluation,
        "Closed-Set Verification",
    )

    (
        pair_sampling_mode,
        pair_sampling_budget,
    ) = _resolve_pair_sampling_arguments(
        pair_sampling_mode=pair_sampling_mode,
        pair_sampling_budget=pair_sampling_budget,
        sampling_mode=sampling_mode,
        num_pairs=num_pairs,
    )

    if pair_sampling_mode not in {
        "balanced",
        "random",
    }:
        pair_sampling_budget = None

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

    hyperparams.pop(
        "num_pairs",
        None,
    )
    hyperparams.pop(
        "sampling_mode",
        None,
    )
    hyperparams.update(
        {
            "pair_sampling_mode": pair_sampling_mode,
            "pair_sampling_budget": pair_sampling_budget,
            "max_impostor_pairs": (
                max_impostor_pairs
                if pair_sampling_mode == "all_genuine"
                else None
            ),
            "pair_sampling_seed": (
                (42 if pair_sampling_seed is None else pair_sampling_seed)
                if pair_sampling_mode == "all_genuine"
                else pair_sampling_seed
                if pair_sampling_mode in {"balanced", "random"}
                else None
            ),
            "probe_fusion_size": probe_fusion_size,
        }
    )

    hyperparams = _add_seed_metadata(
        hyperparams,
        base_seed=seed,
        n_runs=n_runs,
        split_seed=split_seed,
    )

    # --- MULTI-RUN AGGREGATOR ---
    if n_runs > 1:
        call_args = _prepare_multi_run_arguments(locals())
        base_seed = call_args.get('seed', 42)
        results = []
        evaluation_artifact_runs = []
        data_statistics_runs = []
        trained_weight_references = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs)...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False 
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            run_wall_clock_started = _start_runtime_stage()
            (
                (res, d_stats, h_params),
                trained_weight_reference,
            ) = _run_recursive_with_provenance(
                run_closed_set_verification,
                call_args,
                evaluation_artifact_runs,
            )
            _record_multi_run_time(
                run_index=i + 1,
                seed=call_args["seed"],
                started_at=run_wall_clock_started,
            )
            results.append(res)
            data_statistics_runs.append(copy.deepcopy(d_stats))
            trained_weight_references.append(trained_weight_reference)
            data_stats = d_stats
            hyperparams = h_params

        hyperparams = _add_seed_metadata(
            hyperparams,
            base_seed=base_seed,
            n_runs=n_runs,
            split_seed=split_seed,
        )

        per_run_results = (
            _build_per_run_results(
                results=results,
                seeds=hyperparams[
                    "run_seeds"
                ],
                split_seeds=hyperparams[
                    "resolved_split_seeds"
                ],
                data_statistics=data_statistics_runs,
                trained_weight_references=trained_weight_references,
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
                
        metric_aggregates = (
            _aggregate_multi_run_metrics(
                results
            )
        )
        
        if save_results_and_settings:
            metrics_dict = {
                "EER": _format_multi_run_metric(metric_aggregates[0]), "AUC": _format_multi_run_metric(metric_aggregates[1]),
                "d-prime": _format_multi_run_metric(metric_aggregates[2]), "TAR@0.1%FAR": _format_multi_run_metric(metric_aggregates[3])
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
        return tuple(metric_aggregates)
    # ----------------------------

    augmentation_config = (
        _normalize_augmentation_config(
            augmentation_config
        )
    )

    hyperparams[
        "augmentation"
    ] = augmentation_config

    if enrollment_template_mode == "fusion":
        if template_fusion_method is _UNSET:
            template_fusion_method = "mean"
    else:
        template_fusion_method = None

    hyperparams["enrollment_template_mode"] = enrollment_template_mode
    hyperparams["template_fusion_method"] = template_fusion_method

    if enrollment_template_mode == "multi_template":
        hyperparams["num_templates_per_identity"] = num_templates_per_identity
        hyperparams["template_selection_method"] = template_selection_method
        hyperparams["template_score_aggregation"] = template_score_aggregation

    device, reproducibility_state = _setup_reproducibility(
        seed=seed,
        device=device,
        reproducibility_mode=reproducibility_mode,
    )
    _set_active_profiling_device_type(device)
    hyperparams["reproducibility_mode"] = reproducibility_mode
    hyperparams["reproducibility_state"] = reproducibility_state
    resolved_split_seed = seed if split_seed is None else split_seed
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
    if provenance is not None:
        provenance = provenance.subset(valid_mask)

    # ====================================================
    # 3. SPLIT DATA & SQI SCORES
    # ====================================================
    train_test_partitions = (
        _split_closed_set_samples(
            x,
            y,
            holdout_split=test_split,
            seed=resolved_split_seed,
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

    provenance_train = None
    provenance_test = None
    if provenance is not None:
        provenance_train = provenance.subset(
            train_test_partitions["indices"]["retained"]
        )
        provenance_test = provenance.subset(
            train_test_partitions["indices"]["holdout"]
        )

    # ====================================================
    # 4. APPLY FILTERS INDEPENDENTLY
    # ====================================================
    if outlier_filtering_on_train and sqi_train is not None:
        print("\n[INFO] Filtering Train Set (Enrollment)...")
        if provenance_train is not None:
            X_train, y_train, retained_indices = _apply_outlier_filter(
                X_train, y_train, sqi_train, absolute_threshold=sqi_threshold,
                keep_percentage=sqi_keep_pct, return_indices=True
            )
            provenance_train = provenance_train.subset(retained_indices)
        else:
            X_train, y_train = _apply_outlier_filter(
                X_train, y_train, sqi_train, absolute_threshold=sqi_threshold,
                keep_percentage=sqi_keep_pct
            )

    if outlier_filtering_on_test and sqi_test is not None:
        print("\n[INFO] Filtering Test Set (Probes)...")
        if provenance_test is not None:
            X_test, y_test, retained_indices = _apply_outlier_filter(
                X_test, y_test, sqi_test, absolute_threshold=sqi_threshold,
                keep_percentage=sqi_keep_pct, apply_subject_ranking=False,
                return_indices=True,
            )
            provenance_test = provenance_test.subset(retained_indices)
        else:
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
    if provenance_test is not None:
        provenance_test = provenance_test.subset(test_mask)

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
            seed=resolved_split_seed,
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
            "reproducibility_mode": reproducibility_mode,
        }

        validation_arrays = _collect_validation_arrays(
            val_split, locals()
        )
        train_config = _build_weight_cache_config(
            loader,
            train_config,
            training_samples=X_tr,
            training_labels=y_tr,
            validation_arrays=validation_arrays,
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
            _record_persisted_trained_weight(
                model.weight_artifact_metadata,
                "cache_hit",
            )
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
                artifact_context=_build_weight_artifact_context(
                    X_tr,
                    len(classes),
                    resolved_split_seed,
                    reproducibility_state,
                ),
            )
            _record_newly_saved_trained_weight(cache, uid)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        model = _run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs)
        _record_in_memory_trained_weight(model)
        
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
            pair_sampling_budget=pair_sampling_budget, pair_sampling_mode=pair_sampling_mode, max_impostor_pairs=max_impostor_pairs, pair_sampling_seed=pair_sampling_seed, matching_method=matching_method
        )
        global_threshold = _find_optimal_threshold(calib_scores, calib_pair_labels)
        print(f"[INFO] Optimal Global Threshold Found: {global_threshold:.4f}")
    
    # Extract Test Embeddings (Probes)
    test_emb, test_lab = _get_embeddings(model, test_loader, device)

    # ====================================================
    # 9. EVALUATION STRATEGY
    # ====================================================
    fusion_diagnostics = None
    template_diagnostics = None

    if enrollment_template_mode == "multi_template":
        print(
            "[INFO] Building multi-template gallery "
            f"({num_templates_per_identity} templates/identity, "
            f"{template_selection_method})..."
        )
        train_extract_loader = _make_loader(X_train, y_train_enc, batch_size, shuffle=False)
        train_emb, train_lab = _get_embeddings(model, train_extract_loader, device)
        (
            mt_templates,
            mt_template_identities,
            mt_template_source_indices,
            template_diagnostics,
        ) = _select_multi_templates(
            train_emb, train_lab,
            provenance=provenance_train,
            num_templates_per_identity=num_templates_per_identity,
            template_selection_method=template_selection_method,
            max_beats=template_size,
        )
        scores, labels_pair, fusion_diagnostics = _generate_multi_template_verification_pairs(
            probe_embeddings=test_emb,
            probe_labels=test_lab,
            probe_provenance=provenance_test,
            template_embeddings=mt_templates,
            template_identities=mt_template_identities,
            num_templates_per_identity=num_templates_per_identity,
            probe_fusion_size=probe_fusion_size,
            matching_method=matching_method,
            pair_sampling_mode=pair_sampling_mode,
            pair_sampling_budget=pair_sampling_budget,
            max_impostor_pairs=max_impostor_pairs,
            pair_sampling_seed=pair_sampling_seed,
            template_score_aggregation=template_score_aggregation,
        )
    elif not use_template:
        # STRATEGY A: Test vs Test (Intra-session unseen evaluation)
        print(f"[INFO] Bypassing Templates. Generating pairs exclusively from Test split...")
        scores, labels_pair = _generate_pairs(
            embeddings1=test_emb, 
            labels1=test_lab, 
            embeddings2=None, # None forces Test vs Test matching
            labels2=None, 
            pair_sampling_budget=pair_sampling_budget,
            pair_sampling_mode=pair_sampling_mode,
            max_impostor_pairs=max_impostor_pairs, pair_sampling_seed=pair_sampling_seed, matching_method=matching_method
        )
    else:
        # STRATEGY B: Test vs Train Templates (Authentication Simulation)
        print(f"[INFO] Building Enrollment Templates from Train split...")
        # [FIX]: Use y_train_enc to match the neural network encoding output
        train_extract_loader = _make_loader(X_train, y_train_enc, batch_size, shuffle=False)
        train_emb, train_lab = _get_embeddings(model, train_extract_loader, device)
        
        templates, temp_labels = _create_templates(
            train_emb, train_lab, method=template_fusion_method,
            max_beats=template_size, provenance=provenance_train
        )
        
        if probe_fusion_size > 1:
            scores, labels_pair, fusion_diagnostics = _generate_fused_verification_pairs(
                probe_embeddings=test_emb,
                probe_labels=test_lab,
                probe_provenance=provenance_test,
                template_embeddings=templates,
                template_identities=temp_labels,
                group_size=probe_fusion_size,
                matching_method=matching_method,
                pair_sampling_mode=pair_sampling_mode,
                pair_sampling_budget=pair_sampling_budget,
                max_impostor_pairs=max_impostor_pairs,
                pair_sampling_seed=pair_sampling_seed,
            )
        else:
            scores, labels_pair = _generate_pairs(
                embeddings1=test_emb, # Probes
                labels1=test_lab,
                embeddings2=templates, # Enrollment
                labels2=temp_labels,
                pair_sampling_budget=pair_sampling_budget,
                pair_sampling_mode=pair_sampling_mode,
                max_impostor_pairs=max_impostor_pairs, pair_sampling_seed=pair_sampling_seed, matching_method=matching_method
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

    if fusion_diagnostics is not None:
        data_stats["Probe Fusion"] = fusion_diagnostics
    if template_diagnostics is not None:
        data_stats["Enrollment Templates"] = template_diagnostics

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
                                        template_fusion_method=_UNSET, template_size=1,
                                        matching_method='cosine', outlier_filtering_on_train=False, 
                                        outlier_filtering_on_test=False, sqi_scores=None, 
                                        sqi_threshold=0.05, sqi_keep_pct=0.8, probe_fusion_size=1,
                                        save_results_and_settings=False, loader=None, 
                                        n_runs=1, _return_stats=False,
                                        intelligent_weight_loading=True,
                                        augmentation_config=None, split_seed=None, provenance=None,
        enrollment_template_mode='fusion', num_templates_per_identity=None,
        template_selection_method=None, template_score_aggregation=None,
        reproducibility_mode="seeded"):
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
        seed (int): Training/general stochastic seed controlling model initialization, training DataLoader shuffling, augmentation, and other stochastic operations not governed by split_seed (such as validation-EER pair sampling). When split_seed is None, seed also supplies the resolved data-role allocation seed.
        split_seed (int or None): Optional seed for randomized data-role allocation such as train/test beat splits, subject-cohort splits, and validation allocation. None follows the current per-run training seed; an explicit integer holds the randomized partition fixed across training seeds where such a partition exists.
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
        probe_fusion_size (int): Probe fusion depth. With 1, each probe beat
            yields one decision. With k>1, complete non-overlapping groups of k
            probe beats within one source block (subject, session, record,
            segment), ordered by source provenance, are averaged into one
            decision; an incomplete final group is dropped.
        save_results_and_settings (bool): If True, logs results and parameters to a text file.
        loader (object): Dataset loader instance (used for extracting metadata for logging).
        n_runs (int): Number of repeated runs using consecutive training seeds. The data-role split schedule follows the split_seed policy (it follows the training seeds when split_seed is None, and stays fixed when an explicit split_seed is given).
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.
        reproducibility_mode (str): Reproducibility policy ('seeded' or 'strict').

    Returns:
        tuple: (Rank-1 Accuracy, Rank-5 Accuracy)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for both metrics.
    """   

    reproducibility_mode = _activate_top_level_runtime_profile(
        reproducibility_mode,
        device,
        recursive_run=_return_stats,
    )

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

    (
        enrollment_template_mode,
        num_templates_per_identity,
        template_selection_method,
        template_score_aggregation,
    ) = _validate_multi_template_arguments(
        enrollment_template_mode,
        template_fusion_method,
        num_templates_per_identity,
        template_selection_method,
        template_score_aggregation,
        use_template,
        False,
        "Subject-Disjoint Identification",
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
        split_seed=split_seed,
    )

    if n_runs > 1:
        call_args = _prepare_multi_run_arguments(locals())
        base_seed = call_args.get('seed', 42)
        results = []
        evaluation_artifact_runs = []
        data_statistics_runs = []
        trained_weight_references = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs)...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False 
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            run_wall_clock_started = _start_runtime_stage()
            (
                (res, d_stats, h_params),
                trained_weight_reference,
            ) = _run_recursive_with_provenance(
                run_subject_disjoint_identification,
                call_args,
                evaluation_artifact_runs,
            )
            _record_multi_run_time(
                run_index=i + 1,
                seed=call_args["seed"],
                started_at=run_wall_clock_started,
            )
            results.append(res)
            data_statistics_runs.append(copy.deepcopy(d_stats))
            trained_weight_references.append(trained_weight_reference)
            data_stats = d_stats
            hyperparams = h_params

        hyperparams = _add_seed_metadata(
            hyperparams,
            base_seed=base_seed,
            n_runs=n_runs,
            split_seed=split_seed,
        )

        reportable_results = (
            _apply_identification_metric_reportability(
                results,
                evaluation_artifact_runs,
            )
        )

        per_run_results = (
            _build_per_run_results(
                results=reportable_results,
                seeds=hyperparams[
                    "run_seeds"
                ],
                split_seeds=hyperparams[
                    "resolved_split_seeds"
                ],
                data_statistics=data_statistics_runs,
                trained_weight_references=trained_weight_references,
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
                
        metric_aggregates = (
            _aggregate_multi_run_metrics(
                reportable_results
            )
        )
        
        if save_results_and_settings:
            metrics_dict = {
                "Rank-1 Accuracy": _format_multi_run_metric(
                    metric_aggregates[0]
                ),
                "Rank-5 Accuracy": _format_multi_run_metric(
                    metric_aggregates[1]
                ),
            }
            _log_experiment_results(
                "Subject-Disjoint Identification",
                metrics_dict,
                data_stats,
                hyperparams,
                loader,
                per_run_results=per_run_results,
                per_run_evaluation_artifacts=per_run_evaluation_artifacts,
            )
        return tuple(metric_aggregates)
    # ----------------------------

    augmentation_config = (
        _normalize_augmentation_config(
            augmentation_config
        )
    )

    hyperparams[
        "augmentation"
    ] = augmentation_config

    if enrollment_template_mode == "fusion":
        if template_fusion_method is _UNSET:
            template_fusion_method = "mean"
    else:
        template_fusion_method = None

    hyperparams["enrollment_template_mode"] = enrollment_template_mode
    hyperparams["template_fusion_method"] = template_fusion_method

    if enrollment_template_mode == "multi_template":
        hyperparams["num_templates_per_identity"] = num_templates_per_identity
        hyperparams["template_selection_method"] = template_selection_method
        hyperparams["template_score_aggregation"] = template_score_aggregation

    device, reproducibility_state = _setup_reproducibility(
        seed=seed,
        device=device,
        reproducibility_mode=reproducibility_mode,
    )
    _set_active_profiling_device_type(device)
    hyperparams["reproducibility_mode"] = reproducibility_mode
    hyperparams["reproducibility_state"] = reproducibility_state
    resolved_split_seed = seed if split_seed is None else split_seed
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
    if provenance is not None:
        provenance = provenance.subset(valid_mask)

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
        seed=resolved_split_seed,
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

    provenance_test = None
    if provenance is not None:
        provenance_test = provenance.subset(partitions["indices"]["test"])

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
        if provenance_test is not None:
            X_test, y_test, retained_indices = _apply_outlier_filter(
                X_test, y_test, sqi_test, absolute_threshold=sqi_threshold,
                keep_percentage=sqi_keep_pct, apply_subject_ranking=False,
                return_indices=True,
            )
            provenance_test = provenance_test.subset(retained_indices)
        else:
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
    if provenance_test is not None:
        provenance_test = provenance_test.subset(test_survivor_mask)
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
            X_train, y_train_remap, test_size=val_split, stratify=y_train_remap, random_state=resolved_split_seed
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
            "reproducibility_mode": reproducibility_mode,
            "matching_method": matching_method  # Affects early stopping EER!
        }

        validation_arrays = _collect_validation_arrays(
            val_split, locals()
        )
        train_config = _build_weight_cache_config(
            loader,
            train_config,
            training_samples=X_tr,
            training_labels=y_tr,
            validation_arrays=validation_arrays,
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
            _record_persisted_trained_weight(
                model.weight_artifact_metadata,
                "cache_hit",
            )
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
                artifact_context=_build_weight_artifact_context(
                    X_tr,
                    num_train_classes,
                    resolved_split_seed,
                    reproducibility_state,
                ),
            )
            _record_newly_saved_trained_weight(cache, uid)
    
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Call the updated custom loop passing both validation loaders of seen and unseen subjects!
        model = _run_train_loop_unseen_subjects(
            model, train_loader, val_loader_seen, val_loader_unseen, optimizer, criterion, device, 
            epochs, matching_method=matching_method, patience=40, lr_patience=15
        )
        _record_in_memory_trained_weight(model)

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
            provenance=provenance_test,
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

    provenance_probe = None
    if provenance_test is not None:
        provenance_probe = provenance_test.subset(
            enrollment_probe_partitions["indices"]["probe"]
        )

    template_diagnostics = None

    if enrollment_template_mode == "multi_template":
        provenance_enroll = None
        if provenance_test is not None:
            provenance_enroll = provenance_test.subset(
                enrollment_probe_partitions["indices"]["enrollment"]
            )

        print(
            "[INFO] Building multi-template gallery "
            f"({num_templates_per_identity} templates/identity, "
            f"{template_selection_method})..."
        )
        (
            mt_templates,
            mt_template_identities,
            mt_template_source_indices,
            template_diagnostics,
        ) = _select_multi_templates(
            emb_enroll, lab_enroll,
            provenance=provenance_enroll,
            num_templates_per_identity=num_templates_per_identity,
            template_selection_method=template_selection_method,
            max_beats=None,
        )
        gallery_emb = mt_templates

        probe_mapped = np.array([test_sub_map[l] for l in lab_probe])
        raw_scores = _compute_score_matrix(emb_probe, mt_templates, method=matching_method)

        if probe_fusion_size > 1:
            fused_template_scores, fused_probe_mapped, fusion_diagnostics = _apply_score_fusion(
                raw_scores, probe_mapped, fusion_size=probe_fusion_size,
                provenance=provenance_probe, return_diagnostics=True,
            )
        else:
            fused_template_scores = raw_scores
            fused_probe_mapped = probe_mapped
            fusion_diagnostics = {
                "fusion_size": 1,
                "raw_probe_observations": int(len(probe_mapped)),
                "fused_probe_decisions": int(len(probe_mapped)),
                "dropped_remainder_observations": 0,
                "source_blocks_below_fusion_size": 0,
                "identities_without_a_fused_decision": 0,
            }

        if probe_fusion_size > 1 and len(fused_probe_mapped) == 0:
            raise ValueError(
                "Probe fusion produced no complete fused decisions for any "
                "identity: with the configured probe fusion size no source "
                "segment holds enough probe beats to form a group. Reduce the "
                "probe fusion size or provide more probe data."
            )

        final_scores = _reduce_template_scores_to_identities(
            fused_template_scores, mt_template_identities, np.asarray(test_subs_final),
            template_score_aggregation=template_score_aggregation,
        )
        final_labels = fused_probe_mapped
    else:
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
        final_scores, final_labels, fusion_diagnostics = _apply_score_fusion(
            scores,
            probe_mapped,
            fusion_size=probe_fusion_size,
            provenance=provenance_probe,
            return_diagnostics=True,
        )
        if probe_fusion_size > 1 and len(final_labels) == 0:
            raise ValueError(
                "Probe fusion produced no complete fused decisions for any "
                "identity: with the configured probe fusion size no source "
                "segment holds enough probe beats to form a group. Reduce the "
                "probe fusion size or provide more probe data."
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
        "Probe Fusion": fusion_diagnostics,
    }
    if template_diagnostics is not None:
        data_stats["Enrollment Templates"] = template_diagnostics

    if _return_stats:
        return (rank1, rank5), data_stats, hyperparams

    if save_results_and_settings:
        _log_experiment_results(
            task_title,
            {
                "Rank-1 Accuracy": rank1,
                "Rank-5 Accuracy": (
                    rank5
                    if evaluation_artifacts[
                        "rank_5_reportable"
                    ]
                    else None
                ),
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
                                      test_split=0.2, val_split=0.0, num_pairs=None,
                                      sampling_mode=None, seed=42, device=None,
                                      visualize=False, use_template=False, template_fusion_method=_UNSET,
                                      template_size=1, matching_method='cosine',
                                      outlier_filtering_on_train=False, outlier_filtering_on_test=False, 
                                      sqi_scores=None, sqi_threshold=0.05, sqi_keep_pct=0.8, 
                                      use_deployment_evaluation=False, target_fars=None,
                                      save_results_and_settings=False, 
                                      loader=None, n_runs=1, _return_stats=False,
                                      intelligent_weight_loading=True,
                                      augmentation_config=None, split_seed=None, provenance=None,
        *,
        pair_sampling_budget=None,
        pair_sampling_mode=None,
        max_impostor_pairs=1000000,
        pair_sampling_seed=42,
        probe_fusion_size=1,
        enrollment_template_mode='fusion', num_templates_per_identity=None,
        template_selection_method=None, template_score_aggregation=None,
        reproducibility_mode="seeded"):
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
        pair_sampling_mode (str or None): Verification comparison strategy. Options: ['all', 'all_genuine', 'balanced', 'random'].
        pair_sampling_budget (int or None): Requested comparison budget for balanced and random sampling.
        max_impostor_pairs (int): Maximum number of impostor comparisons retained by all_genuine.
        pair_sampling_seed (int): Dedicated seed for stochastic verification-pair sampling.
        num_pairs (int or None): Legacy alias for pair_sampling_budget.
        sampling_mode (str or None): Legacy alias for pair_sampling_mode.
        seed (int): Training/general stochastic seed controlling model initialization, training DataLoader shuffling, augmentation, and other stochastic operations not governed by split_seed (such as validation-EER pair sampling). When split_seed is None, seed also supplies the resolved data-role allocation seed.
        split_seed (int or None): Optional seed for randomized data-role allocation such as train/test beat splits, subject-cohort splits, and validation allocation. None follows the current per-run training seed; an explicit integer holds the randomized partition fixed across training seeds where such a partition exists.
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
        n_runs (int): Number of repeated runs using consecutive training seeds. The data-role split schedule follows the split_seed policy (it follows the training seeds when split_seed is None, and stays fixed when an explicit split_seed is given).
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.
        reproducibility_mode (str): Reproducibility policy ('seeded' or 'strict').

    Returns:
        tuple: (EER, AUC, d-prime, TAR @ 0.1% FAR)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for all four metrics.
    """

    reproducibility_mode = _activate_top_level_runtime_profile(
        reproducibility_mode,
        device,
        recursive_run=_return_stats,
    )

    (
        enrollment_template_mode,
        num_templates_per_identity,
        template_selection_method,
        template_score_aggregation,
    ) = _validate_multi_template_arguments(
        enrollment_template_mode,
        template_fusion_method,
        num_templates_per_identity,
        template_selection_method,
        template_score_aggregation,
        use_template,
        use_deployment_evaluation,
        "Subject-Disjoint Verification",
    )

    _validate_deployment_evaluation(
        use_deployment_evaluation,
        val_split,
        "Subject-Disjoint Verification",
    )

    probe_fusion_size = _validate_verification_probe_fusion(
        probe_fusion_size,
        use_template,
        use_deployment_evaluation,
        "Subject-Disjoint Verification",
    )

    (
        pair_sampling_mode,
        pair_sampling_budget,
    ) = _resolve_pair_sampling_arguments(
        pair_sampling_mode=pair_sampling_mode,
        pair_sampling_budget=pair_sampling_budget,
        sampling_mode=sampling_mode,
        num_pairs=num_pairs,
    )

    if pair_sampling_mode not in {
        "balanced",
        "random",
    }:
        pair_sampling_budget = None

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

    hyperparams.pop(
        "num_pairs",
        None,
    )
    hyperparams.pop(
        "sampling_mode",
        None,
    )
    hyperparams.update(
        {
            "pair_sampling_mode": pair_sampling_mode,
            "pair_sampling_budget": pair_sampling_budget,
            "max_impostor_pairs": (
                max_impostor_pairs
                if pair_sampling_mode == "all_genuine"
                else None
            ),
            "pair_sampling_seed": (
                (42 if pair_sampling_seed is None else pair_sampling_seed)
                if pair_sampling_mode == "all_genuine"
                else pair_sampling_seed
                if pair_sampling_mode in {"balanced", "random"}
                else None
            ),
            "probe_fusion_size": probe_fusion_size,
        }
    )

    hyperparams = _add_seed_metadata(
        hyperparams,
        base_seed=seed,
        n_runs=n_runs,
        split_seed=split_seed,
    )

    if n_runs > 1:
        call_args = _prepare_multi_run_arguments(locals())
        base_seed = call_args.get('seed', 42)
        results = []
        evaluation_artifact_runs = []
        data_statistics_runs = []
        trained_weight_references = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs)...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False 
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            run_wall_clock_started = _start_runtime_stage()
            (
                (res, d_stats, h_params),
                trained_weight_reference,
            ) = _run_recursive_with_provenance(
                run_subject_disjoint_verification,
                call_args,
                evaluation_artifact_runs,
            )
            _record_multi_run_time(
                run_index=i + 1,
                seed=call_args["seed"],
                started_at=run_wall_clock_started,
            )
            results.append(res)
            data_statistics_runs.append(copy.deepcopy(d_stats))
            trained_weight_references.append(trained_weight_reference)
            data_stats = d_stats
            hyperparams = h_params

        hyperparams = _add_seed_metadata(
            hyperparams,
            base_seed=base_seed,
            n_runs=n_runs,
            split_seed=split_seed,
        )

        per_run_results = (
            _build_per_run_results(
                results=results,
                seeds=hyperparams[
                    "run_seeds"
                ],
                split_seeds=hyperparams[
                    "resolved_split_seeds"
                ],
                data_statistics=data_statistics_runs,
                trained_weight_references=trained_weight_references,
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
                
        metric_aggregates = (
            _aggregate_multi_run_metrics(
                results
            )
        )
        
        if save_results_and_settings:
            metrics_dict = {
                "EER": _format_multi_run_metric(metric_aggregates[0]), "AUC": _format_multi_run_metric(metric_aggregates[1]),
                "d-prime": _format_multi_run_metric(metric_aggregates[2]), "TAR@0.1%FAR": _format_multi_run_metric(metric_aggregates[3])
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
        return tuple(metric_aggregates)
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

    if enrollment_template_mode == "fusion":
        if template_fusion_method is _UNSET:
            template_fusion_method = "mean"
    else:
        template_fusion_method = None

    hyperparams["enrollment_template_mode"] = enrollment_template_mode
    hyperparams["template_fusion_method"] = template_fusion_method

    if enrollment_template_mode == "multi_template":
        hyperparams["num_templates_per_identity"] = num_templates_per_identity
        hyperparams["template_selection_method"] = template_selection_method
        hyperparams["template_score_aggregation"] = template_score_aggregation

    device, reproducibility_state = _setup_reproducibility(
        seed=seed,
        device=device,
        reproducibility_mode=reproducibility_mode,
    )
    _set_active_profiling_device_type(device)
    hyperparams["reproducibility_mode"] = reproducibility_mode
    hyperparams["reproducibility_state"] = reproducibility_state
    resolved_split_seed = seed if split_seed is None else split_seed
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
    if provenance is not None:
        provenance = provenance.subset(valid_mask)

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
        seed=resolved_split_seed,
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

    provenance_test = None
    if provenance is not None:
        provenance_test = provenance.subset(
            partitions["indices"]["test"]
        )

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
        if provenance_test is not None:
            X_test, y_test, retained_indices = _apply_outlier_filter(
                X_test,
                y_test,
                sqi_test,
                absolute_threshold=sqi_threshold,
                keep_percentage=sqi_keep_pct,
                apply_subject_ranking=False,
                return_indices=True,
            )
            provenance_test = provenance_test.subset(retained_indices)
        else:
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
    if provenance_test is not None:
        provenance_test = provenance_test.subset(test_survivor_mask)
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
            X_train, y_train_remap, test_size=val_split, stratify=y_train_remap, random_state=resolved_split_seed
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
            "reproducibility_mode": reproducibility_mode,
            "matching_method": matching_method  # Affects early stopping EER!
        }

        validation_arrays = _collect_validation_arrays(
            val_split, locals()
        )
        train_config = _build_weight_cache_config(
            loader,
            train_config,
            training_samples=X_tr,
            training_labels=y_tr,
            validation_arrays=validation_arrays,
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
            _record_persisted_trained_weight(
                model.weight_artifact_metadata,
                "cache_hit",
            )
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
                artifact_context=_build_weight_artifact_context(
                    X_tr,
                    num_train_classes,
                    resolved_split_seed,
                    reproducibility_state,
                ),
            )
            _record_newly_saved_trained_weight(cache, uid)
    
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
        _record_in_memory_trained_weight(model)

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
            pair_sampling_budget=pair_sampling_budget, pair_sampling_mode=pair_sampling_mode, max_impostor_pairs=max_impostor_pairs, pair_sampling_seed=pair_sampling_seed, matching_method=matching_method
        )
        global_threshold = _find_optimal_threshold(calib_scores, calib_pair_labels)
        print(f"[INFO] Optimal Global Threshold Found: {global_threshold:.4f}")
        
    # ====================================================
    # 8. FINAL INFERENCE ON UNSEEN TEST SUBJECTS
    # ====================================================
    test_emb, test_lab = _get_embeddings(model, test_loader, device)

    # 9. Evaluation Strategy
    fusion_diagnostics = None
    template_diagnostics = None

    if enrollment_template_mode == "multi_template":
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
                provenance=provenance_test,
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

        provenance_enroll = None
        provenance_probe = None
        if provenance_test is not None:
            provenance_enroll = provenance_test.subset(
                enrollment_probe_partitions["indices"]["enrollment"]
            )
            provenance_probe = provenance_test.subset(
                enrollment_probe_partitions["indices"]["probe"]
            )

        print(
            "[INFO] Building multi-template gallery "
            f"({num_templates_per_identity} templates/identity, "
            f"{template_selection_method})..."
        )
        (
            mt_templates,
            mt_template_identities,
            mt_template_source_indices,
            template_diagnostics,
        ) = _select_multi_templates(
            emb_enroll, lab_enroll,
            provenance=provenance_enroll,
            num_templates_per_identity=num_templates_per_identity,
            template_selection_method=template_selection_method,
            max_beats=None,
        )
        scores, labels_pair, fusion_diagnostics = _generate_multi_template_verification_pairs(
            probe_embeddings=emb_probe,
            probe_labels=lab_probe,
            probe_provenance=provenance_probe,
            template_embeddings=mt_templates,
            template_identities=mt_template_identities,
            num_templates_per_identity=num_templates_per_identity,
            probe_fusion_size=probe_fusion_size,
            matching_method=matching_method,
            pair_sampling_mode=pair_sampling_mode,
            pair_sampling_budget=pair_sampling_budget,
            max_impostor_pairs=max_impostor_pairs,
            pair_sampling_seed=pair_sampling_seed,
            template_score_aggregation=template_score_aggregation,
        )
    elif not use_template:
        print(f"[INFO] Bypassing Templates. Generating pairs entirely from Unseen Test Subjects...")
        scores, labels_pair = _generate_pairs(
            embeddings1=test_emb, labels1=test_lab, 
            embeddings2=None, labels2=None, 
            pair_sampling_budget=pair_sampling_budget, pair_sampling_mode=pair_sampling_mode, max_impostor_pairs=max_impostor_pairs, pair_sampling_seed=pair_sampling_seed, matching_method=matching_method
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
                provenance=provenance_test,
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

        provenance_probe = None
        if provenance_test is not None:
            provenance_probe = provenance_test.subset(
                enrollment_probe_partitions["indices"]["probe"]
            )
        
        templates, temp_labels = _create_templates(
            emb_enroll, lab_enroll, method=template_fusion_method, max_beats=None
        )
        
        if probe_fusion_size > 1:
            scores, labels_pair, fusion_diagnostics = _generate_fused_verification_pairs(
                probe_embeddings=emb_probe,
                probe_labels=lab_probe,
                probe_provenance=provenance_probe,
                template_embeddings=templates,
                template_identities=temp_labels,
                group_size=probe_fusion_size,
                matching_method=matching_method,
                pair_sampling_mode=pair_sampling_mode,
                pair_sampling_budget=pair_sampling_budget,
                max_impostor_pairs=max_impostor_pairs,
                pair_sampling_seed=pair_sampling_seed,
            )
        else:
            scores, labels_pair = _generate_pairs(
                embeddings1=emb_probe, labels1=lab_probe,
                embeddings2=templates, labels2=temp_labels,
                pair_sampling_budget=pair_sampling_budget, pair_sampling_mode=pair_sampling_mode, max_impostor_pairs=max_impostor_pairs, pair_sampling_seed=pair_sampling_seed, matching_method=matching_method
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

    if fusion_diagnostics is not None:
        data_stats["Probe Fusion"] = fusion_diagnostics
    if template_diagnostics is not None:
        data_stats["Enrollment Templates"] = template_diagnostics

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

def _prepare_cross_session_enrollment_role(
    x_enroll,
    y_enroll,
    provenance_enroll,
    use_enrollment,
):
    """
    Validate an optional physically separate enrollment role.

    Enrollment data is unused when ``use_enrollment`` is false. When enrollment
    is required but no separate arrays are supplied, the partition helper
    reuses the already processed training role.
    """
    if not use_enrollment:
        return (
            None,
            None,
            None,
            False,
        )

    if (
        (x_enroll is None)
        != (y_enroll is None)
    ):
        raise ValueError(
            "x_enroll and y_enroll must either both "
            "be provided or both be omitted."
        )

    if x_enroll is None:
        if provenance_enroll is not None:
            raise ValueError(
                "provenance_enroll requires explicit "
                "x_enroll and y_enroll arrays."
            )

        return (
            None,
            None,
            None,
            False,
        )

    x_enroll = np.asarray(
        x_enroll
    )

    y_enroll = np.asarray(
        y_enroll
    )

    if x_enroll.ndim < 1:
        raise ValueError(
            "Enrollment samples must contain "
            "a sample dimension."
        )

    if y_enroll.ndim != 1:
        raise ValueError(
            "Enrollment labels must be "
            "one-dimensional."
        )

    if len(x_enroll) != len(y_enroll):
        raise ValueError(
            "Enrollment samples and labels "
            "are misaligned."
        )

    if (
        provenance_enroll is not None
        and len(provenance_enroll)
        != len(y_enroll)
    ):
        raise ValueError(
            "Enrollment provenance is misaligned "
            "with enrollment samples and labels."
        )

    return (
        x_enroll,
        y_enroll,
        provenance_enroll,
        True,
    )

def run_cross_session_identification(x_train, y_train, x_test, y_test, model_class, epochs=150, 
                                     batch_size=256, lr=1e-3, val_split=0.0, seed=42, device=None, 
                                     visualize=False, use_template=False, template_fusion_method=_UNSET,
                                     template_size=None, matching_method='cosine',
                                     outlier_filtering_on_train=False, outlier_filtering_on_test=False, 
                                     sqi_train=None, sqi_test=None, sqi_threshold=0.05, 
                                     sqi_keep_pct=0.8, probe_fusion_size=1, save_results_and_settings=False,
                                     loader=None, n_runs=1, _return_stats=False,
                                     intelligent_weight_loading=True,
                                     augmentation_config=None, split_seed=None, provenance_s1=None, provenance_s2=None,
                                     x_enroll=None, y_enroll=None, provenance_enroll=None,
        enrollment_template_mode='fusion', num_templates_per_identity=None,
        template_selection_method=None, template_score_aggregation=None,
        reproducibility_mode="seeded"):
    """
    Cross-session identification with protocol-defined data roles.

    Training data fits the representation or classifier. Probe data is used for
    identification. When template matching is enabled, enrollment data forms
    the gallery and defaults to the training source when omitted.

    Args:
        x_train (np.ndarray): ECG samples assigned to the training role.
        y_train (np.ndarray): Labels for the training samples.
        x_test (np.ndarray): ECG samples assigned to the probe role.
        y_test (np.ndarray): Labels for the probe samples.
        model_class (nn.Module): The PyTorch model architecture class to instantiate.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per training batch.
        lr (float): Learning rate for the Adam optimizer.
        val_split (float): Fraction of Session 1 data to use for early stopping.
        seed (int): Training/general stochastic seed controlling model initialization, training DataLoader shuffling, augmentation, and other stochastic operations not governed by split_seed (such as validation-EER pair sampling). When split_seed is None, seed also supplies the resolved data-role allocation seed.
        split_seed (int or None): Optional seed for randomized data-role allocation such as train/test beat splits, subject-cohort splits, and validation allocation. None follows the current per-run training seed; an explicit integer holds the randomized partition fixed across training seeds where such a partition exists. For cross-session tasks the evaluation partition is protocol-defined and fixed; split_seed affects only a randomized validation allocation when validation is active.
        device (str): Computation device ('cuda', 'cpu', or 'auto').
        visualize (bool): If True, generates t-SNE scatter plots of the cross-session embeddings.
        use_template (bool):
            - False: Uses the classifier fitted on training data to classify probes.
            - True: Builds an enrollment gallery and metric-matches probe embeddings.
        template_fusion_method (str): Logic used to create enrollment templates.
            Options: ['mean', 'median', 'trimmed_mean', 'representative',
            'soft_centrality', 'geometric_median', 'none']
        template_size (int, optional): Number of enrollment beats used per template. None uses all available.
        matching_method (str): Distance/Similarity metric for template matching.
            Options: ['cosine', 'euclidean', 'manhattan', 'correlation']
        outlier_filtering_on_train (bool): Apply SQI filtering independently to Session 1.
        outlier_filtering_on_test (bool): Apply SQI filtering independently to Session 2.
        sqi_train (str or np.ndarray): SQI calculation method or pre-computed array for Session 1.
        sqi_test (str or np.ndarray): SQI calculation method or pre-computed array for Session 2.
        sqi_threshold (float): Absolute minimum SQI score required to keep a beat (0.0 to 1.0).
        sqi_keep_pct (float): Top percentage of beats to keep per subject after filtering.
        probe_fusion_size (int): Probe fusion depth. With 1, each probe beat
            yields one decision. With k>1, complete non-overlapping groups of k
            probe beats within one source block (subject, session, record,
            segment), ordered by source provenance, are averaged into one
            decision; an incomplete final group is dropped.
        save_results_and_settings (bool): If True, logs results and parameters to a text file.
        loader (object): Dataset loader instance (used for extracting metadata for logging).
        n_runs (int): Number of repeated runs using consecutive training seeds. The data-role split schedule follows the split_seed policy (it follows the training seeds when split_seed is None, and stays fixed when an explicit split_seed is given).
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.
        reproducibility_mode (str): Reproducibility policy ('seeded' or 'strict').

    Returns:
        tuple: (Rank-1 Accuracy, Rank-5 Accuracy)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for both metrics.
    """

    reproducibility_mode = _activate_top_level_runtime_profile(
        reproducibility_mode,
        device,
        recursive_run=_return_stats,
    )

    (
        enrollment_template_mode,
        num_templates_per_identity,
        template_selection_method,
        template_score_aggregation,
    ) = _validate_multi_template_arguments(
        enrollment_template_mode,
        template_fusion_method,
        num_templates_per_identity,
        template_selection_method,
        template_score_aggregation,
        use_template,
        False,
        "Cross-Session Identification",
    )

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
        split_seed=split_seed,
    )

    if n_runs > 1:
        call_args = _prepare_multi_run_arguments(locals())
        base_seed = call_args.get('seed', 42)
        results = []
        evaluation_artifact_runs = []
        data_statistics_runs = []
        trained_weight_references = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs)...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False 
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            run_wall_clock_started = _start_runtime_stage()
            (
                (res, d_stats, h_params),
                trained_weight_reference,
            ) = _run_recursive_with_provenance(
                run_cross_session_identification,
                call_args,
                evaluation_artifact_runs,
            )
            _record_multi_run_time(
                run_index=i + 1,
                seed=call_args["seed"],
                started_at=run_wall_clock_started,
            )
            results.append(res)
            data_statistics_runs.append(copy.deepcopy(d_stats))
            trained_weight_references.append(trained_weight_reference)
            data_stats = d_stats
            hyperparams = h_params

        hyperparams = _add_seed_metadata(
            hyperparams,
            base_seed=base_seed,
            n_runs=n_runs,
            split_seed=split_seed,
        )

        reportable_results = (
            _apply_identification_metric_reportability(
                results,
                evaluation_artifact_runs,
            )
        )

        per_run_results = (
            _build_per_run_results(
                results=reportable_results,
                seeds=hyperparams[
                    "run_seeds"
                ],
                split_seeds=hyperparams[
                    "resolved_split_seeds"
                ],
                data_statistics=data_statistics_runs,
                trained_weight_references=trained_weight_references,
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
                
        metric_aggregates = (
            _aggregate_multi_run_metrics(
                reportable_results
            )
        )
        
        if save_results_and_settings:
            metrics_dict = {
                "Rank-1 Accuracy": _format_multi_run_metric(
                    metric_aggregates[0]
                ),
                "Rank-5 Accuracy": _format_multi_run_metric(
                    metric_aggregates[1]
                ),
            }
            _log_experiment_results(
                "Cross-Session Identification",
                metrics_dict,
                data_stats,
                hyperparams,
                loader,
                per_run_results=per_run_results,
                per_run_evaluation_artifacts=per_run_evaluation_artifacts,
            )
        return tuple(metric_aggregates)
    # ----------------------------

    augmentation_config = (
        _normalize_augmentation_config(
            augmentation_config
        )
    )

    hyperparams[
        "augmentation"
    ] = augmentation_config

    if enrollment_template_mode == "fusion":
        if template_fusion_method is _UNSET:
            template_fusion_method = "mean"
    else:
        template_fusion_method = None

    hyperparams["enrollment_template_mode"] = enrollment_template_mode
    hyperparams["template_fusion_method"] = template_fusion_method

    if enrollment_template_mode == "multi_template":
        hyperparams["num_templates_per_identity"] = num_templates_per_identity
        hyperparams["template_selection_method"] = template_selection_method
        hyperparams["template_score_aggregation"] = template_score_aggregation

    device, reproducibility_state = _setup_reproducibility(
        seed=seed,
        device=device,
        reproducibility_mode=reproducibility_mode,
    )
    _set_active_profiling_device_type(device)
    hyperparams["reproducibility_mode"] = reproducibility_mode
    hyperparams["reproducibility_state"] = reproducibility_state
    resolved_split_seed = seed if split_seed is None else split_seed
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
        print("\n[INFO] Filtering training data...")
        if provenance_s1 is not None:
            x_train, y_train, retained_indices = _apply_outlier_filter(
                x_train, y_train, sqi_train, sqi_threshold, sqi_keep_pct, return_indices=True
            )
            provenance_s1 = provenance_s1.subset(retained_indices)
        else:
            x_train, y_train = _apply_outlier_filter(x_train, y_train, sqi_train, sqi_threshold, sqi_keep_pct)

    if sqi_test is not None:
        print("\n[INFO] Filtering probe data...")
        if provenance_s2 is not None:
            x_test, y_test, retained_indices = _apply_outlier_filter(
                x_test, y_test, sqi_test, absolute_threshold=sqi_threshold,
                keep_percentage=sqi_keep_pct, apply_subject_ranking=False,
                return_indices=True,
            )
            provenance_s2 = provenance_s2.subset(retained_indices)
        else:
            x_test, y_test = _apply_outlier_filter(
                x_test,
                y_test,
                sqi_test,
                absolute_threshold=sqi_threshold,
                keep_percentage=sqi_keep_pct,
                apply_subject_ranking=False,
            )

    (
        x_enroll_prepared,
        y_enroll_prepared,
        provenance_enroll_prepared,
        enrollment_is_explicit,
    ) = _prepare_cross_session_enrollment_role(
        x_enroll=x_enroll,
        y_enroll=y_enroll,
        provenance_enroll=provenance_enroll,
        use_enrollment=use_template,
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
            x_enrollment=x_enroll_prepared,
            y_enrollment=y_enroll_prepared,
        )
    )

    (
        x_train_full,
        y_train_full,
    ) = cross_session_partitions[
        "session_1"
    ]

    (
        x_enroll_filtered,
        y_enroll_filtered,
    ) = cross_session_partitions[
        "enrollment"
    ]

    (
        x_test_filtered,
        y_test_filtered,
    ) = cross_session_partitions[
        "probe"
    ]

    template_provenance = None
    provenance_probe = None

    if (
        enrollment_is_explicit
        and provenance_enroll_prepared is not None
    ):
        template_provenance = (
            provenance_enroll_prepared.subset(
                cross_session_partitions[
                    "indices"
                ][
                    "enrollment"
                ]
            )
        )
    elif (
        not enrollment_is_explicit
        and provenance_s1 is not None
    ):
        template_provenance = provenance_s1.subset(
            cross_session_partitions[
                "indices"
            ][
                "enrollment"
            ]
        )

    if provenance_s2 is not None:
        provenance_probe = provenance_s2.subset(
            cross_session_partitions[
                "indices"
            ][
                "probe"
            ]
        )

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
    y_enroll_enc = (
        np.array(
            [
                label_map[label]
                for label in y_enroll_filtered
            ]
        )
        if use_template
        else None
    )
    
    # ====================================================
    # 5. RESUME STANDARD PIPELINE
    # ====================================================
    # Validation split from the training role
    if val_split > 0.0:
        X_tr, X_val, y_tr, y_val = train_test_split(
            x_train_full, y_train_enc, test_size=val_split, stratify=y_train_enc, random_state=resolved_split_seed
        )
        val_loader = _make_loader(X_val, y_val, batch_size, shuffle=False)
        print(f"Training Split: Train={len(X_tr)}, Val={len(X_val)} | Probes={len(x_test_filtered)}")
    else:
        X_tr, y_tr = x_train_full, y_train_enc
        X_val, val_loader = None, None
        print(f"Training Split: Train={len(X_tr)}, Val=0 | Probes={len(x_test_filtered)}")

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
            "reproducibility_mode": reproducibility_mode,
        }

        validation_arrays = _collect_validation_arrays(
            val_split, locals()
        )
        train_config = _build_weight_cache_config(
            loader,
            train_config,
            training_samples=X_tr,
            training_labels=y_tr,
            validation_arrays=validation_arrays,
            training_role_only_loader_identity=True,
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
            _record_persisted_trained_weight(
                model.weight_artifact_metadata,
                "cache_hit",
            )
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
                artifact_context=_build_weight_artifact_context(
                    X_tr,
                    len(classes),
                    resolved_split_seed,
                    reproducibility_state,
                ),
            )
            _record_newly_saved_trained_weight(cache, uid)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
        model = _run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs)
        _record_in_memory_trained_weight(model)
    
    # ====================================================
    # 6. EVALUATION STRATEGY
    # ====================================================
    template_diagnostics = None

    if enrollment_template_mode == "multi_template":
        print(
            "[INFO] Building multi-template gallery "
            f"({num_templates_per_identity} templates/identity, "
            f"{template_selection_method})..."
        )
        model.include_top = False

        enroll_loader = _make_loader(
            x_enroll_filtered,
            y_enroll_enc,
            batch_size,
            shuffle=False,
        )
        emb_enroll, lab_enroll = _get_embeddings(model, enroll_loader, device)
        (
            mt_templates,
            mt_template_identities,
            mt_template_source_indices,
            template_diagnostics,
        ) = _select_multi_templates(
            emb_enroll, lab_enroll,
            provenance=template_provenance,
            num_templates_per_identity=num_templates_per_identity,
            template_selection_method=template_selection_method,
            max_beats=template_size,
        )

        emb_probe, lab_probe = _get_embeddings(model, probe_loader, device)
        raw_scores = _compute_score_matrix(emb_probe, mt_templates, method=matching_method)

        if probe_fusion_size > 1:
            fused_template_scores, fused_probe_labels, fusion_diagnostics = _apply_score_fusion(
                raw_scores, lab_probe, fusion_size=probe_fusion_size,
                provenance=provenance_probe, return_diagnostics=True,
            )
        else:
            fused_template_scores = raw_scores
            fused_probe_labels = lab_probe
            fusion_diagnostics = {
                "fusion_size": 1,
                "raw_probe_observations": int(len(lab_probe)),
                "fused_probe_decisions": int(len(lab_probe)),
                "dropped_remainder_observations": 0,
                "source_blocks_below_fusion_size": 0,
                "identities_without_a_fused_decision": 0,
            }

        if probe_fusion_size > 1 and len(fused_probe_labels) == 0:
            raise ValueError(
                "Probe fusion produced no complete fused decisions for any "
                "identity: with the configured probe fusion size no source "
                "segment holds enough probe beats to form a group. Reduce the "
                "probe fusion size or provide more probe data."
            )

        identity_order = np.arange(len(classes))
        final_scores = _reduce_template_scores_to_identities(
            fused_template_scores, mt_template_identities, identity_order,
            template_score_aggregation=template_score_aggregation,
        )
        final_labels = fused_probe_labels

        model.include_top = True
    elif not use_template:
        # STRATEGY A: Standard Softmax Classification
        print("[INFO] Bypassing templates. Using the classifier fitted on training data...")
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
        print("[INFO] Building enrollment templates...")
        model.include_top = False # Switch to Feature Extractor
        
        enroll_loader = _make_loader(
            x_enroll_filtered,
            y_enroll_enc,
            batch_size,
            shuffle=False,
        )
        emb_enroll, lab_enroll = _get_embeddings(model, enroll_loader, device)
        
        gallery_emb, gallery_lab = _create_templates(
            emb_enroll, lab_enroll, method=template_fusion_method, max_beats=template_size, provenance=template_provenance
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
    if enrollment_template_mode != "multi_template":
        final_scores, final_labels, fusion_diagnostics = _apply_score_fusion(
            final_scores,
            final_labels,
            fusion_size=probe_fusion_size,
            provenance=provenance_probe,
            return_diagnostics=True,
        )
        if probe_fusion_size > 1 and len(final_labels) == 0:
            raise ValueError(
                "Probe fusion produced no complete fused decisions for any "
                "identity: with the configured probe fusion size no source "
                "segment holds enough probe beats to form a group. Reduce the "
                "probe fusion size or provide more probe data."
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
        "Training Samples": len(x_train_full),
        "Enrollment Samples": (
            len(x_enroll_filtered)
            if use_template
            else 0
        ),
        "Probe Samples": len(x_test_filtered),
        "Probe Fusion": fusion_diagnostics,
    }
    if template_diagnostics is not None:
        data_stats["Enrollment Templates"] = template_diagnostics

    if _return_stats:
        return (rank1, rank5), data_stats, hyperparams

    if save_results_and_settings:
        _log_experiment_results(
            task_title,
            {
                "Rank-1 Accuracy": rank1,
                "Rank-5 Accuracy": (
                    rank5
                    if evaluation_artifacts[
                        "rank_5_reportable"
                    ]
                    else None
                ),
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
                                   batch_size=256, lr=1e-3, val_split=0.0, num_pairs=None,
                                   sampling_mode=None, seed=42, device=None, visualize=False,
                                   use_template=False, template_fusion_method=_UNSET, template_size=None,
                                   matching_method='cosine', outlier_filtering_on_train=False, 
                                   outlier_filtering_on_test=False, sqi_train=None, sqi_test=None, 
                                   sqi_threshold=0.05, sqi_keep_pct=0.8, use_deployment_evaluation=False,
                                   target_fars=None,
                                   save_results_and_settings=False, loader=None, 
                                   n_runs=1, _return_stats=False,
                                   intelligent_weight_loading=True,
                                   augmentation_config=None, split_seed=None, provenance_s1=None, provenance_s2=None,
                                   x_enroll=None, y_enroll=None, provenance_enroll=None,
        *,
        pair_sampling_budget=None,
        pair_sampling_mode=None,
        max_impostor_pairs=1000000,
        pair_sampling_seed=42,
        probe_fusion_size=1,
        enrollment_template_mode='fusion', num_templates_per_identity=None,
        template_selection_method=None, template_score_aggregation=None,
        reproducibility_mode="seeded"):
    """
    Cross-session verification with protocol-defined data roles.

    Training data fits the representation. Probe data supplies biometric
    comparisons, and template-based evaluation uses enrollment templates.

    Args:
        x_train (np.ndarray): ECG samples assigned to the training role.
        y_train (np.ndarray): Labels for the training samples.
        x_test (np.ndarray): ECG samples assigned to the probe role.
        y_test (np.ndarray): Labels for the probe samples.
        model_class (nn.Module): The PyTorch model architecture class to instantiate.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per training batch.
        lr (float): Learning rate for the Adam optimizer.
        val_split (float): Fraction of Session 1 data to use for early stopping.
        pair_sampling_mode (str or None): Verification comparison strategy. Options: ['all', 'all_genuine', 'balanced', 'random'].
        pair_sampling_budget (int or None): Requested comparison budget for balanced and random sampling.
        max_impostor_pairs (int): Maximum number of impostor comparisons retained by all_genuine.
        pair_sampling_seed (int): Dedicated seed for stochastic verification-pair sampling.
        num_pairs (int or None): Legacy alias for pair_sampling_budget.
        sampling_mode (str or None): Legacy alias for pair_sampling_mode.
        seed (int): Training/general stochastic seed controlling model initialization, training DataLoader shuffling, augmentation, and other stochastic operations not governed by split_seed (such as validation-EER pair sampling). When split_seed is None, seed also supplies the resolved data-role allocation seed.
        split_seed (int or None): Optional seed for randomized data-role allocation such as train/test beat splits, subject-cohort splits, and validation allocation. None follows the current per-run training seed; an explicit integer holds the randomized partition fixed across training seeds where such a partition exists. For cross-session tasks the evaluation partition is protocol-defined and fixed; split_seed affects only a randomized validation allocation when validation is active.
        device (str): Computation device ('cuda', 'cpu', or 'auto').
        visualize (bool): If True, generates t-SNE scatter plots of the cross-session embeddings.
        use_template (bool):
            - False: Generates verification pairs entirely from probe data.
            - True: Compares probe embeddings against enrollment templates.
        template_fusion_method (str): Logic used to create enrollment templates.
            Options: ['mean', 'median', 'trimmed_mean', 'representative',
            'soft_centrality', 'geometric_median', 'none']
        template_size (int, optional): Number of enrollment beats used per template. None uses all available.
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
        n_runs (int): Number of repeated runs using consecutive training seeds. The data-role split schedule follows the split_seed policy (it follows the training seeds when split_seed is None, and stays fixed when an explicit split_seed is given).
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.
        reproducibility_mode (str): Reproducibility policy ('seeded' or 'strict').

    Returns:
        tuple: (EER, AUC, d-prime, TAR @ 0.1% FAR)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for all four metrics.
    """

    reproducibility_mode = _activate_top_level_runtime_profile(
        reproducibility_mode,
        device,
        recursive_run=_return_stats,
    )

    (
        enrollment_template_mode,
        num_templates_per_identity,
        template_selection_method,
        template_score_aggregation,
    ) = _validate_multi_template_arguments(
        enrollment_template_mode,
        template_fusion_method,
        num_templates_per_identity,
        template_selection_method,
        template_score_aggregation,
        use_template,
        use_deployment_evaluation,
        "Cross-Session Verification",
    )

    _validate_deployment_evaluation(
        use_deployment_evaluation,
        val_split,
        "Cross-Session Verification",
    )

    probe_fusion_size = _validate_verification_probe_fusion(
        probe_fusion_size,
        use_template,
        use_deployment_evaluation,
        "Cross-Session Verification",
    )
    
    (
        pair_sampling_mode,
        pair_sampling_budget,
    ) = _resolve_pair_sampling_arguments(
        pair_sampling_mode=pair_sampling_mode,
        pair_sampling_budget=pair_sampling_budget,
        sampling_mode=sampling_mode,
        num_pairs=num_pairs,
    )

    if pair_sampling_mode not in {
        "balanced",
        "random",
    }:
        pair_sampling_budget = None

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

    hyperparams.pop(
        "num_pairs",
        None,
    )
    hyperparams.pop(
        "sampling_mode",
        None,
    )
    hyperparams.update(
        {
            "pair_sampling_mode": pair_sampling_mode,
            "pair_sampling_budget": pair_sampling_budget,
            "max_impostor_pairs": (
                max_impostor_pairs
                if pair_sampling_mode == "all_genuine"
                else None
            ),
            "pair_sampling_seed": (
                (42 if pair_sampling_seed is None else pair_sampling_seed)
                if pair_sampling_mode == "all_genuine"
                else pair_sampling_seed
                if pair_sampling_mode in {"balanced", "random"}
                else None
            ),
            "probe_fusion_size": probe_fusion_size,
        }
    )

    hyperparams = _add_seed_metadata(
        hyperparams,
        base_seed=seed,
        n_runs=n_runs,
        split_seed=split_seed,
    )

    if n_runs > 1:
        call_args = _prepare_multi_run_arguments(locals())
        base_seed = call_args.get('seed', 42)
        results = []
        evaluation_artifact_runs = []
        data_statistics_runs = []
        trained_weight_references = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs)...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False 
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            run_wall_clock_started = _start_runtime_stage()
            (
                (res, d_stats, h_params),
                trained_weight_reference,
            ) = _run_recursive_with_provenance(
                run_cross_session_verification,
                call_args,
                evaluation_artifact_runs,
            )
            _record_multi_run_time(
                run_index=i + 1,
                seed=call_args["seed"],
                started_at=run_wall_clock_started,
            )
            results.append(res)
            data_statistics_runs.append(copy.deepcopy(d_stats))
            trained_weight_references.append(trained_weight_reference)
            data_stats = d_stats
            hyperparams = h_params

        hyperparams = _add_seed_metadata(
            hyperparams,
            base_seed=base_seed,
            n_runs=n_runs,
            split_seed=split_seed,
        )

        per_run_results = (
            _build_per_run_results(
                results=results,
                seeds=hyperparams[
                    "run_seeds"
                ],
                split_seeds=hyperparams[
                    "resolved_split_seeds"
                ],
                data_statistics=data_statistics_runs,
                trained_weight_references=trained_weight_references,
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
                
        metric_aggregates = (
            _aggregate_multi_run_metrics(
                results
            )
        )
        
        if save_results_and_settings:
            metrics_dict = {
                "EER": _format_multi_run_metric(metric_aggregates[0]), "AUC": _format_multi_run_metric(metric_aggregates[1]),
                "d-prime": _format_multi_run_metric(metric_aggregates[2]), "TAR@0.1%FAR": _format_multi_run_metric(metric_aggregates[3])
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
        return tuple(metric_aggregates)
    # ----------------------------

    augmentation_config = (
        _normalize_augmentation_config(
            augmentation_config
        )
    )

    hyperparams[
        "augmentation"
    ] = augmentation_config

    if enrollment_template_mode == "fusion":
        if template_fusion_method is _UNSET:
            template_fusion_method = "mean"
    else:
        template_fusion_method = None

    hyperparams["enrollment_template_mode"] = enrollment_template_mode
    hyperparams["template_fusion_method"] = template_fusion_method

    if enrollment_template_mode == "multi_template":
        hyperparams["num_templates_per_identity"] = num_templates_per_identity
        hyperparams["template_selection_method"] = template_selection_method
        hyperparams["template_score_aggregation"] = template_score_aggregation

    device, reproducibility_state = _setup_reproducibility(
        seed=seed,
        device=device,
        reproducibility_mode=reproducibility_mode,
    )
    _set_active_profiling_device_type(device)
    hyperparams["reproducibility_mode"] = reproducibility_mode
    hyperparams["reproducibility_state"] = reproducibility_state
    resolved_split_seed = seed if split_seed is None else split_seed
    partition_stage_started = _start_runtime_stage()
    task_title = "Cross-Session Verification"
    mode_str = f"Template ({template_fusion_method}, size={template_size or 'All'})" if use_template else "Probe-only pairs"
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
        print("\n[INFO] Filtering training data...")
        if provenance_s1 is not None:
            x_train, y_train, retained_indices = _apply_outlier_filter(
                x_train,
                y_train,
                sqi_train,
                sqi_threshold,
                sqi_keep_pct,
                return_indices=True,
            )
            provenance_s1 = provenance_s1.subset(retained_indices)
        else:
            x_train, y_train = _apply_outlier_filter(
                x_train,
                y_train,
                sqi_train,
                sqi_threshold,
                sqi_keep_pct,
            )

    if sqi_test is not None:
        print("\n[INFO] Filtering probe data...")
        if provenance_s2 is not None:
            x_test, y_test, retained_indices = _apply_outlier_filter(
                x_test,
                y_test,
                sqi_test,
                absolute_threshold=sqi_threshold,
                keep_percentage=sqi_keep_pct,
                apply_subject_ranking=False,
                return_indices=True,
            )
            provenance_s2 = provenance_s2.subset(retained_indices)
        else:
            x_test, y_test = _apply_outlier_filter(
                x_test,
                y_test,
                sqi_test,
                absolute_threshold=sqi_threshold,
                keep_percentage=sqi_keep_pct,
                apply_subject_ranking=False,
            )

    (
        x_enroll_prepared,
        y_enroll_prepared,
        provenance_enroll_prepared,
        enrollment_is_explicit,
    ) = _prepare_cross_session_enrollment_role(
        x_enroll=x_enroll,
        y_enroll=y_enroll,
        provenance_enroll=provenance_enroll,
        use_enrollment=use_template,
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
            x_enrollment=x_enroll_prepared,
            y_enrollment=y_enroll_prepared,
        )
    )

    (
        x_train_full,
        y_train_full,
    ) = cross_session_partitions[
        "session_1"
    ]

    (
        x_enroll_filtered,
        y_enroll_filtered,
    ) = cross_session_partitions[
        "enrollment"
    ]

    (
        x_test_filtered,
        y_test_filtered,
    ) = cross_session_partitions[
        "probe"
    ]

    template_provenance = None
    provenance_probe = None

    if (
        enrollment_is_explicit
        and provenance_enroll_prepared is not None
    ):
        template_provenance = (
            provenance_enroll_prepared.subset(
                cross_session_partitions[
                    "indices"
                ][
                    "enrollment"
                ]
            )
        )
    elif (
        not enrollment_is_explicit
        and provenance_s1 is not None
    ):
        template_provenance = provenance_s1.subset(
            cross_session_partitions[
                "indices"
            ][
                "enrollment"
            ]
        )

    if provenance_s2 is not None:
        provenance_probe = provenance_s2.subset(
            cross_session_partitions[
                "indices"
            ][
                "probe"
            ]
        )

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
    y_enroll_enc = (
        np.array(
            [
                label_map[label]
                for label in y_enroll_filtered
            ]
        )
        if use_template
        else None
    )
    
    # ====================================================
    # 5. RESUME STANDARD PIPELINE
    # ====================================================
    if val_split > 0.0:
        X_tr, X_val, y_tr, y_val = train_test_split(
            x_train_full, y_train_enc, test_size=val_split, stratify=y_train_enc, random_state=resolved_split_seed
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
            "reproducibility_mode": reproducibility_mode,
        }

        validation_arrays = _collect_validation_arrays(
            val_split, locals()
        )
        train_config = _build_weight_cache_config(
            loader,
            train_config,
            training_samples=X_tr,
            training_labels=y_tr,
            validation_arrays=validation_arrays,
            training_role_only_loader_identity=True,
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
            _record_persisted_trained_weight(
                model.weight_artifact_metadata,
                "cache_hit",
            )
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
                artifact_context=_build_weight_artifact_context(
                    X_tr,
                    len(classes),
                    resolved_split_seed,
                    reproducibility_state,
                ),
            )
            _record_newly_saved_trained_weight(cache, uid)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()    
        model = _run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs)
        _record_in_memory_trained_weight(model)
    
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
            pair_sampling_budget=pair_sampling_budget, pair_sampling_mode=pair_sampling_mode, max_impostor_pairs=max_impostor_pairs, pair_sampling_seed=pair_sampling_seed, matching_method=matching_method
        )
        global_threshold = _find_optimal_threshold(calib_scores, calib_pair_labels)
        print(f"[INFO] Optimal Global Threshold Found: {global_threshold:.4f}")

    # ====================================================
    # 7. EVALUATION STRATEGY
    # ====================================================
    emb_probe, lab_probe = _get_embeddings(model, probe_loader, device)

    fusion_diagnostics = None
    template_diagnostics = None

    if enrollment_template_mode == "multi_template":
        print(
            "[INFO] Building multi-template gallery "
            f"({num_templates_per_identity} templates/identity, "
            f"{template_selection_method})..."
        )
        enroll_loader = _make_loader(
            x_enroll_filtered,
            y_enroll_enc,
            batch_size,
            shuffle=False,
        )
        emb_enroll, lab_enroll = _get_embeddings(model, enroll_loader, device)
        (
            mt_templates,
            mt_template_identities,
            mt_template_source_indices,
            template_diagnostics,
        ) = _select_multi_templates(
            emb_enroll, lab_enroll,
            provenance=template_provenance,
            num_templates_per_identity=num_templates_per_identity,
            template_selection_method=template_selection_method,
            max_beats=template_size,
        )
        scores, labels_pair, fusion_diagnostics = _generate_multi_template_verification_pairs(
            probe_embeddings=emb_probe,
            probe_labels=lab_probe,
            probe_provenance=provenance_probe,
            template_embeddings=mt_templates,
            template_identities=mt_template_identities,
            num_templates_per_identity=num_templates_per_identity,
            probe_fusion_size=probe_fusion_size,
            matching_method=matching_method,
            pair_sampling_mode=pair_sampling_mode,
            pair_sampling_budget=pair_sampling_budget,
            max_impostor_pairs=max_impostor_pairs,
            pair_sampling_seed=pair_sampling_seed,
            template_score_aggregation=template_score_aggregation,
        )
    elif not use_template:
        # STRATEGY A: probe-only verification
        print("[INFO] Bypassing templates. Generating verification pairs from probe data...")
        scores, labels_pair = _generate_pairs(
            embeddings1=emb_probe, 
            labels1=lab_probe, 
            embeddings2=None, # None forces test vs test matching
            labels2=None, 
            pair_sampling_budget=pair_sampling_budget,
            pair_sampling_mode=pair_sampling_mode,
            max_impostor_pairs=max_impostor_pairs, pair_sampling_seed=pair_sampling_seed, matching_method=matching_method
        )
    else:
        # STRATEGY B: enrollment-template verification
        print("[INFO] Building enrollment templates...")
        enroll_loader = _make_loader(
            x_enroll_filtered,
            y_enroll_enc,
            batch_size,
            shuffle=False,
        )
        emb_enroll, lab_enroll = _get_embeddings(model, enroll_loader, device)
        
        templates, temp_labels = _create_templates(
            emb_enroll,
            lab_enroll,
            method=template_fusion_method,
            max_beats=template_size,
            provenance=template_provenance,
        )
        
        if probe_fusion_size > 1:
            scores, labels_pair, fusion_diagnostics = _generate_fused_verification_pairs(
                probe_embeddings=emb_probe,
                probe_labels=lab_probe,
                probe_provenance=provenance_probe,
                template_embeddings=templates,
                template_identities=temp_labels,
                group_size=probe_fusion_size,
                matching_method=matching_method,
                pair_sampling_mode=pair_sampling_mode,
                pair_sampling_budget=pair_sampling_budget,
                max_impostor_pairs=max_impostor_pairs,
                pair_sampling_seed=pair_sampling_seed,
            )
        else:
            scores, labels_pair = _generate_pairs(
                embeddings1=emb_probe,
                labels1=lab_probe,
                embeddings2=templates,
                labels2=temp_labels,
                pair_sampling_budget=pair_sampling_budget,
                pair_sampling_mode=pair_sampling_mode,
                max_impostor_pairs=max_impostor_pairs, pair_sampling_seed=pair_sampling_seed, matching_method=matching_method
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
        "Training Samples": len(X_tr),
        "Validation Samples": (
            len(X_val)
            if X_val is not None
            else 0
        ),
        "Enrollment Samples": (
            len(x_enroll_filtered)
            if use_template
            else 0
        ),
        "Probe Samples": len(
            x_test_filtered
        ),
    }

    data_stats.update(
        _get_verification_pair_statistics(
            labels_pair,
            target_far=0.001,
        )
    )

    if fusion_diagnostics is not None:
        data_stats["Probe Fusion"] = fusion_diagnostics
    if template_diagnostics is not None:
        data_stats["Enrollment Templates"] = template_diagnostics

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
        seed=42, device=None, visualize=False, use_template=True, template_fusion_method=_UNSET, template_size=None,
        matching_method='cosine', outlier_filtering_on_train=False, outlier_filtering_on_test=False, sqi_s1=None, 
        sqi_s2=None, sqi_threshold=0.05, sqi_keep_pct=0.8, probe_fusion_size=1, save_results_and_settings=False,
        loader=None, n_runs=1, _return_stats=False,
        intelligent_weight_loading=True,
        augmentation_config=None, split_seed=None, provenance_s1=None, provenance_s2=None,
        x_enroll=None, y_enroll=None, provenance_enroll=None,
        enrollment_template_mode='fusion', num_templates_per_identity=None,
        template_selection_method=None, template_score_aggregation=None,
        reproducibility_mode="seeded"):
    """
    Subject-disjoint cross-session identification.
    1. Fits a feature extractor using the training role for Subject Group A.
    2. Builds a gallery for unseen Subject Group B from the enrollment role.
    3. Identifies Subject Group B using the probe role.

    Args:
        x_s1 (np.ndarray): ECG samples assigned to the training role.
        y_s1 (np.ndarray): Labels for the training samples.
        x_s2 (np.ndarray): ECG samples assigned to the probe role.
        y_s2 (np.ndarray): Labels for the probe samples.
        model_class (nn.Module): The PyTorch model architecture class to instantiate.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per training batch.
        lr (float): Learning rate for the Adam optimizer.
        test_split (float): Fraction of unique SUBJECTS to isolate for the Group B tests.
        val_split (float): Fraction of Group A subjects to use for early stopping validation.
        seed (int): Training/general stochastic seed controlling model initialization, training DataLoader shuffling, augmentation, and other stochastic operations not governed by split_seed (such as validation-EER pair sampling). When split_seed is None, seed also supplies the resolved data-role allocation seed.
        split_seed (int or None): Optional seed for randomized data-role allocation such as train/test beat splits, subject-cohort splits, and validation allocation. None follows the current per-run training seed; an explicit integer holds the randomized partition fixed across training seeds where such a partition exists.
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
        probe_fusion_size (int): Probe fusion depth. With 1, each probe beat
            yields one decision. With k>1, complete non-overlapping groups of k
            probe beats within one source block (subject, session, record,
            segment), ordered by source provenance, are averaged into one
            decision; an incomplete final group is dropped.
        save_results_and_settings (bool): If True, logs results and parameters to a text file.
        loader (object): Dataset loader instance (used for extracting metadata for logging).
        n_runs (int): Number of repeated runs using consecutive training seeds. The data-role split schedule follows the split_seed policy (it follows the training seeds when split_seed is None, and stays fixed when an explicit split_seed is given).
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.
        reproducibility_mode (str): Reproducibility policy ('seeded' or 'strict').

    Returns:
        tuple: (Rank-1 Accuracy, Rank-5 Accuracy)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for both metrics.
    """
    reproducibility_mode = _activate_top_level_runtime_profile(
        reproducibility_mode,
        device,
        recursive_run=_return_stats,
    )
    
    (
        enrollment_template_mode,
        num_templates_per_identity,
        template_selection_method,
        template_score_aggregation,
    ) = _validate_multi_template_arguments(
        enrollment_template_mode,
        template_fusion_method,
        num_templates_per_identity,
        template_selection_method,
        template_score_aggregation,
        use_template,
        False,
        "Subject-Disjoint Cross-Session Identification",
    )

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
        split_seed=split_seed,
    )

    if n_runs > 1:
        call_args = _prepare_multi_run_arguments(locals())
        base_seed = call_args.get('seed', 42)
        results = []
        evaluation_artifact_runs = []
        data_statistics_runs = []
        trained_weight_references = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs)...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False 
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            run_wall_clock_started = _start_runtime_stage()
            (
                (res, d_stats, h_params),
                trained_weight_reference,
            ) = _run_recursive_with_provenance(
                run_subject_disjoint_cross_session_identification,
                call_args,
                evaluation_artifact_runs,
            )
            _record_multi_run_time(
                run_index=i + 1,
                seed=call_args["seed"],
                started_at=run_wall_clock_started,
            )
            results.append(res)
            data_statistics_runs.append(copy.deepcopy(d_stats))
            trained_weight_references.append(trained_weight_reference)
            data_stats = d_stats
            hyperparams = h_params

        hyperparams = _add_seed_metadata(
            hyperparams,
            base_seed=base_seed,
            n_runs=n_runs,
            split_seed=split_seed,
        )

        reportable_results = (
            _apply_identification_metric_reportability(
                results,
                evaluation_artifact_runs,
            )
        )

        per_run_results = (
            _build_per_run_results(
                results=reportable_results,
                seeds=hyperparams[
                    "run_seeds"
                ],
                split_seeds=hyperparams[
                    "resolved_split_seeds"
                ],
                data_statistics=data_statistics_runs,
                trained_weight_references=trained_weight_references,
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
                
        metric_aggregates = (
            _aggregate_multi_run_metrics(
                reportable_results
            )
        )
        
        if save_results_and_settings:
            metrics_dict = {
                "Rank-1 Accuracy": _format_multi_run_metric(
                    metric_aggregates[0]
                ),
                "Rank-5 Accuracy": _format_multi_run_metric(
                    metric_aggregates[1]
                ),
            }
            _log_experiment_results(
                "Subject-Disjoint Cross-Session ID",
                metrics_dict,
                data_stats,
                hyperparams,
                loader,
                per_run_results=per_run_results,
                per_run_evaluation_artifacts=per_run_evaluation_artifacts,
            )
        return tuple(metric_aggregates)
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

    if enrollment_template_mode == "fusion":
        if template_fusion_method is _UNSET:
            template_fusion_method = "mean"
    else:
        template_fusion_method = None

    hyperparams["enrollment_template_mode"] = enrollment_template_mode
    hyperparams["template_fusion_method"] = template_fusion_method

    if enrollment_template_mode == "multi_template":
        hyperparams["num_templates_per_identity"] = num_templates_per_identity
        hyperparams["template_selection_method"] = template_selection_method
        hyperparams["template_score_aggregation"] = template_score_aggregation

    device, reproducibility_state = _setup_reproducibility(
        seed=seed,
        device=device,
        reproducibility_mode=reproducibility_mode,
    )
    _set_active_profiling_device_type(device)
    hyperparams["reproducibility_mode"] = reproducibility_mode
    hyperparams["reproducibility_state"] = reproducibility_state
    resolved_split_seed = seed if split_seed is None else split_seed
    partition_stage_started = _start_runtime_stage()
    task_title = "Subject-Disjoint Cross-Session ID"
    mode_str = f"Gallery: Enrollment ({template_fusion_method}, size={template_size or 'All'})"
    print(f"\n[TASK] {task_title} | Mode: {mode_str} | Match: {matching_method}")

    # ====================================================
    # 1. PREPARE & APPLY SQI FILTERS
    # ====================================================
    def _prepare_sqi(sqi_input, x_data, flag, name):
        if not flag: return None
        if sqi_input is None: return None
        if isinstance(sqi_input, str): return np.array(_compute_sqi(x_data, method=sqi_input))
        return np.array(sqi_input)

    sqi_s1 = _prepare_sqi(sqi_s1, x_s1, outlier_filtering_on_train, "Training")
    sqi_s2 = _prepare_sqi(sqi_s2, x_s2, outlier_filtering_on_test, "Session 2")

    if sqi_s1 is not None:
        print("\n[INFO] Filtering training data...")
        if provenance_s1 is not None:
            x_s1, y_s1, retained_indices = _apply_outlier_filter(
                x_s1, y_s1, sqi_s1, sqi_threshold, sqi_keep_pct, return_indices=True
            )
            provenance_s1 = provenance_s1.subset(retained_indices)
        else:
            x_s1, y_s1 = _apply_outlier_filter(x_s1, y_s1, sqi_s1, sqi_threshold, sqi_keep_pct)

    if sqi_s2 is not None:
        print("\n[INFO] Filtering probe data...")
        if provenance_s2 is not None:
            x_s2, y_s2, retained_indices = _apply_outlier_filter(
                x_s2, y_s2, sqi_s2, absolute_threshold=sqi_threshold,
                keep_percentage=sqi_keep_pct, apply_subject_ranking=False,
                return_indices=True,
            )
            provenance_s2 = provenance_s2.subset(retained_indices)
        else:
            x_s2, y_s2 = _apply_outlier_filter(
                x_s2,
                y_s2,
                sqi_s2,
                absolute_threshold=sqi_threshold,
                keep_percentage=sqi_keep_pct,
                apply_subject_ranking=False,
            )

    (
        x_enroll_prepared,
        y_enroll_prepared,
        provenance_enroll_prepared,
        enrollment_is_explicit,
    ) = _prepare_cross_session_enrollment_role(
        x_enroll=x_enroll,
        y_enroll=y_enroll,
        provenance_enroll=provenance_enroll,
        use_enrollment=True,
    )

    # ====================================================
    # 2. INTERSECT AND SPLIT SUBJECTS (STRICTLY DISJOINT)
    # ====================================================
    # Subject cohorts contain only identities available in every required role.
    eligible_subjects = set(
        y_s1
    ).intersection(
        set(y_s2)
    )

    if enrollment_is_explicit:
        eligible_subjects &= set(
            y_enroll_prepared
        )

    common_subs = sorted(
        eligible_subjects
    )
    
    if len(common_subs) < 2: 
        raise ValueError("[ERROR] Not enough common subjects across required roles after filtering.")
        
    # Split the distinct subjects into Train, Val, and Test cohorts
    (
        train_subs,
        val_subs,
        test_subs,
    ) = _split_subject_cohorts(
        common_subs,
        test_split=test_split,
        val_split=val_split,
        seed=resolved_split_seed,
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
            x_enrollment=x_enroll_prepared,
            y_enrollment=y_enroll_prepared,
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

    template_provenance = None
    provenance_probe = None

    if (
        enrollment_is_explicit
        and provenance_enroll_prepared is not None
    ):
        template_provenance = (
            provenance_enroll_prepared.subset(
                partitions[
                    "indices"
                ][
                    "enrollment"
                ]
            )
        )
    elif (
        not enrollment_is_explicit
        and provenance_s1 is not None
    ):
        template_provenance = provenance_s1.subset(
            partitions[
                "indices"
            ][
                "enrollment"
            ]
        )

    if provenance_s2 is not None:
        provenance_probe = provenance_s2.subset(
            partitions["indices"]["probe"]
        )

    # ====================================================
    # 3. ENCODE LABELS
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
            X_train, y_train_enc, test_size=val_split, stratify=y_train_enc, random_state=resolved_split_seed
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
    
    # Unseen-subject validation uses only training-role samples.
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
            "reproducibility_mode": reproducibility_mode,
            "matching_method": matching_method # Affects early stopping EER!
        }

        validation_arrays = _collect_validation_arrays(
            val_split, locals()
        )
        train_config = _build_weight_cache_config(
            loader,
            train_config,
            training_samples=X_tr,
            training_labels=y_tr,
            validation_arrays=validation_arrays,
            training_role_only_loader_identity=True,
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
            _record_persisted_trained_weight(
                model.weight_artifact_metadata,
                "cache_hit",
            )
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
                artifact_context=_build_weight_artifact_context(
                    X_tr,
                    num_train_classes,
                    resolved_split_seed,
                    reproducibility_state,
                ),
            )
            _record_newly_saved_trained_weight(cache, uid)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Train with the subject-disjoint validation objective.
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
        _record_in_memory_trained_weight(model)

    # ====================================================
    # 5. FINAL INFERENCE ON UNSEEN SUBJECTS
    # ====================================================
    model.include_top = False # Final metric extraction
    
    print("[INFO] Building enrollment templates for unseen subjects...")
    enroll_loader = _make_loader(X_enroll, y_enroll_enc, batch_size, shuffle=False)
    emb_enroll, lab_enroll = _get_embeddings(model, enroll_loader, device)
    
    template_diagnostics = None

    if enrollment_template_mode == "multi_template":
        print(
            "[INFO] Building multi-template gallery "
            f"({num_templates_per_identity} templates/identity, "
            f"{template_selection_method})..."
        )
        (
            mt_templates,
            mt_template_identities,
            mt_template_source_indices,
            template_diagnostics,
        ) = _select_multi_templates(
            emb_enroll, lab_enroll,
            provenance=template_provenance,
            num_templates_per_identity=num_templates_per_identity,
            template_selection_method=template_selection_method,
            max_beats=template_size,
        )

        print("[INFO] Evaluating unseen subjects with probe data...")
        probe_loader = _make_loader(X_probe, y_probe_enc, batch_size, shuffle=False)
        emb_probe, lab_probe = _get_embeddings(model, probe_loader, device)

        raw_scores = _compute_score_matrix(emb_probe, mt_templates, method=matching_method)

        if probe_fusion_size > 1:
            fused_template_scores, fused_probe_labels, fusion_diagnostics = _apply_score_fusion(
                raw_scores, lab_probe, fusion_size=probe_fusion_size,
                provenance=provenance_probe, return_diagnostics=True,
            )
        else:
            fused_template_scores = raw_scores
            fused_probe_labels = lab_probe
            fusion_diagnostics = {
                "fusion_size": 1,
                "raw_probe_observations": int(len(lab_probe)),
                "fused_probe_decisions": int(len(lab_probe)),
                "dropped_remainder_observations": 0,
                "source_blocks_below_fusion_size": 0,
                "identities_without_a_fused_decision": 0,
            }

        if probe_fusion_size > 1 and len(fused_probe_labels) == 0:
            raise ValueError(
                "Probe fusion produced no complete fused decisions for any "
                "identity: with the configured probe fusion size no source "
                "segment holds enough probe beats to form a group. Reduce the "
                "probe fusion size or provide more probe data."
            )

        identity_order = np.arange(len(test_subs))
        final_scores = _reduce_template_scores_to_identities(
            fused_template_scores, mt_template_identities, identity_order,
            template_score_aggregation=template_score_aggregation,
        )
        final_labels = fused_probe_labels
    else:
        gallery_emb, gallery_lab = _create_templates(
            emb_enroll, lab_enroll, method=template_fusion_method, max_beats=template_size, provenance=template_provenance
        )

        print("[INFO] Evaluating unseen subjects with probe data...")
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
        final_scores, final_labels, fusion_diagnostics = _apply_score_fusion(
            scores,
            lab_probe,
            fusion_size=probe_fusion_size,
            provenance=provenance_probe,
            return_diagnostics=True,
        )
        if probe_fusion_size > 1 and len(final_labels) == 0:
            raise ValueError(
                "Probe fusion produced no complete fused decisions for any "
                "identity: with the configured probe fusion size no source "
                "segment holds enough probe beats to form a group. Reduce the "
                "probe fusion size or provide more probe data."
            )

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
        "Training Samples": len(X_train),
        "Enrollment Samples": len(X_enroll),
        "Probe Samples": len(X_probe),
        "Probe Fusion": fusion_diagnostics,
    }
    if template_diagnostics is not None:
        data_stats["Enrollment Templates"] = template_diagnostics

    if _return_stats:
        return (rank1, rank5), data_stats, hyperparams

    if save_results_and_settings:
        _log_experiment_results(
            task_title,
            {
                "Rank-1 Accuracy": rank1,
                "Rank-5 Accuracy": (
                    rank5
                    if evaluation_artifacts[
                        "rank_5_reportable"
                    ]
                    else None
                ),
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
        num_pairs=None, sampling_mode=None, seed=42, device=None, visualize=False, use_template=False,
        template_fusion_method=_UNSET, template_size=None, matching_method='cosine', outlier_filtering_on_train=False,
        outlier_filtering_on_test=False, sqi_s1=None, sqi_s2=None, sqi_threshold=0.05, sqi_keep_pct=0.8,
        use_deployment_evaluation=False, target_fars=None,
        save_results_and_settings=False, loader=None, n_runs=1, _return_stats=False,
        intelligent_weight_loading=True,
        augmentation_config=None, split_seed=None, provenance_s1=None, provenance_s2=None,
        x_enroll=None, y_enroll=None, provenance_enroll=None,
        *,
        pair_sampling_budget=None,
        pair_sampling_mode=None,
        max_impostor_pairs=1000000,
        pair_sampling_seed=42,
        probe_fusion_size=1,
        enrollment_template_mode='fusion', num_templates_per_identity=None,
        template_selection_method=None, template_score_aggregation=None,
        reproducibility_mode="seeded"):
    """
    Subject-disjoint cross-session verification.

    Representation learning uses Subject Group A from the training role.
    Evaluation uses held-out Subject Group B from the probe role and, when
    templates are enabled, from the enrollment role.

    Args:
        x_s1 (np.ndarray): ECG samples assigned to the training role.
        y_s1 (np.ndarray): Labels for the training samples.
        x_s2 (np.ndarray): ECG samples assigned to the probe role.
        y_s2 (np.ndarray): Labels for the probe samples.
        model_class (nn.Module): The PyTorch model architecture class to instantiate.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per training batch.
        lr (float): Learning rate for the Adam optimizer.
        test_split (float): Fraction of unique SUBJECTS to isolate for the Group B tests.
        val_split (float): Fraction of Group A subjects to use for early stopping validation.
        pair_sampling_mode (str or None): Verification comparison strategy. Options: ['all', 'all_genuine', 'balanced', 'random'].
        pair_sampling_budget (int or None): Requested comparison budget for balanced and random sampling.
        max_impostor_pairs (int): Maximum number of impostor comparisons retained by all_genuine.
        pair_sampling_seed (int): Dedicated seed for stochastic verification-pair sampling.
        num_pairs (int or None): Legacy alias for pair_sampling_budget.
        sampling_mode (str or None): Legacy alias for pair_sampling_mode.
        seed (int): Training/general stochastic seed controlling model initialization, training DataLoader shuffling, augmentation, and other stochastic operations not governed by split_seed (such as validation-EER pair sampling). When split_seed is None, seed also supplies the resolved data-role allocation seed.
        split_seed (int or None): Optional seed for randomized data-role allocation such as train/test beat splits, subject-cohort splits, and validation allocation. None follows the current per-run training seed; an explicit integer holds the randomized partition fixed across training seeds where such a partition exists.
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
        n_runs (int): Number of repeated runs using consecutive training seeds. The data-role split schedule follows the split_seed policy (it follows the training seeds when split_seed is None, and stays fixed when an explicit split_seed is given).
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.
        reproducibility_mode (str): Reproducibility policy ('seeded' or 'strict').

    Returns:
        tuple: (EER, AUC, d-prime, TAR @ 0.1% FAR)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for all four metrics.
    """

    reproducibility_mode = _activate_top_level_runtime_profile(
        reproducibility_mode,
        device,
        recursive_run=_return_stats,
    )

    (
        enrollment_template_mode,
        num_templates_per_identity,
        template_selection_method,
        template_score_aggregation,
    ) = _validate_multi_template_arguments(
        enrollment_template_mode,
        template_fusion_method,
        num_templates_per_identity,
        template_selection_method,
        template_score_aggregation,
        use_template,
        use_deployment_evaluation,
        "Subject-Disjoint Cross-Session Verification",
    )

    _validate_deployment_evaluation(
        use_deployment_evaluation,
        val_split,
        "Subject-Disjoint Cross-Session Verification",
    )

    probe_fusion_size = _validate_verification_probe_fusion(
        probe_fusion_size,
        use_template,
        use_deployment_evaluation,
        "Subject-Disjoint Cross-Session Verification",
    )
    
    (
        pair_sampling_mode,
        pair_sampling_budget,
    ) = _resolve_pair_sampling_arguments(
        pair_sampling_mode=pair_sampling_mode,
        pair_sampling_budget=pair_sampling_budget,
        sampling_mode=sampling_mode,
        num_pairs=num_pairs,
    )

    if pair_sampling_mode not in {
        "balanced",
        "random",
    }:
        pair_sampling_budget = None

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

    hyperparams.pop(
        "num_pairs",
        None,
    )
    hyperparams.pop(
        "sampling_mode",
        None,
    )
    hyperparams.update(
        {
            "pair_sampling_mode": pair_sampling_mode,
            "pair_sampling_budget": pair_sampling_budget,
            "max_impostor_pairs": (
                max_impostor_pairs
                if pair_sampling_mode == "all_genuine"
                else None
            ),
            "pair_sampling_seed": (
                (42 if pair_sampling_seed is None else pair_sampling_seed)
                if pair_sampling_mode == "all_genuine"
                else pair_sampling_seed
                if pair_sampling_mode in {"balanced", "random"}
                else None
            ),
            "probe_fusion_size": probe_fusion_size,
        }
    )

    hyperparams = _add_seed_metadata(
        hyperparams,
        base_seed=seed,
        n_runs=n_runs,
        split_seed=split_seed,
    )

    if n_runs > 1:
        call_args = _prepare_multi_run_arguments(locals())
        base_seed = call_args.get('seed', 42)
        results = []
        evaluation_artifact_runs = []
        data_statistics_runs = []
        trained_weight_references = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs)...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False 
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            run_wall_clock_started = _start_runtime_stage()
            (
                (res, d_stats, h_params),
                trained_weight_reference,
            ) = _run_recursive_with_provenance(
                run_subject_disjoint_cross_session_verification,
                call_args,
                evaluation_artifact_runs,
            )
            _record_multi_run_time(
                run_index=i + 1,
                seed=call_args["seed"],
                started_at=run_wall_clock_started,
            )
            results.append(res)
            data_statistics_runs.append(copy.deepcopy(d_stats))
            trained_weight_references.append(trained_weight_reference)
            data_stats = d_stats
            hyperparams = h_params

        hyperparams = _add_seed_metadata(
            hyperparams,
            base_seed=base_seed,
            n_runs=n_runs,
            split_seed=split_seed,
        )

        per_run_results = (
            _build_per_run_results(
                results=results,
                seeds=hyperparams[
                    "run_seeds"
                ],
                split_seeds=hyperparams[
                    "resolved_split_seeds"
                ],
                data_statistics=data_statistics_runs,
                trained_weight_references=trained_weight_references,
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
                
        metric_aggregates = (
            _aggregate_multi_run_metrics(
                results
            )
        )
        
        if save_results_and_settings:
            metrics_dict = {
                "EER": _format_multi_run_metric(metric_aggregates[0]), "AUC": _format_multi_run_metric(metric_aggregates[1]),
                "d-prime": _format_multi_run_metric(metric_aggregates[2]), "TAR@0.1%FAR": _format_multi_run_metric(metric_aggregates[3])
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
        return tuple(metric_aggregates)
    # ----------------------------

    augmentation_config = (
        _normalize_augmentation_config(
            augmentation_config
        )
    )

    hyperparams[
        "augmentation"
    ] = augmentation_config

    if enrollment_template_mode == "fusion":
        if template_fusion_method is _UNSET:
            template_fusion_method = "mean"
    else:
        template_fusion_method = None

    hyperparams["enrollment_template_mode"] = enrollment_template_mode
    hyperparams["template_fusion_method"] = template_fusion_method

    if enrollment_template_mode == "multi_template":
        hyperparams["num_templates_per_identity"] = num_templates_per_identity
        hyperparams["template_selection_method"] = template_selection_method
        hyperparams["template_score_aggregation"] = template_score_aggregation

    device, reproducibility_state = _setup_reproducibility(
        seed=seed,
        device=device,
        reproducibility_mode=reproducibility_mode,
    )
    _set_active_profiling_device_type(device)
    hyperparams["reproducibility_mode"] = reproducibility_mode
    hyperparams["reproducibility_state"] = reproducibility_state
    resolved_split_seed = seed if split_seed is None else split_seed
    partition_stage_started = _start_runtime_stage()
    task_title = "Subject-Disjoint Cross-Session Verification"
    mode_str = f"Template ({template_fusion_method}, Enrollment -> Probe)" if use_template else "Probe-only pairs"
    print(f"\n[TASK] {task_title} | Mode: {mode_str} | Match: {matching_method}")

    # ====================================================
    # 1. PREPARE & APPLY SQI FILTERS
    # ====================================================
    def _prepare_sqi(sqi_input, x_data, flag, name):
        if not flag: return None
        if sqi_input is None: return None
        if isinstance(sqi_input, str): return np.array(_compute_sqi(x_data, method=sqi_input))
        return np.array(sqi_input)

    sqi_s1 = _prepare_sqi(sqi_s1, x_s1, outlier_filtering_on_train, "Training")
    sqi_s2 = _prepare_sqi(sqi_s2, x_s2, outlier_filtering_on_test, "Session 2")

    if sqi_s1 is not None:
        print("\n[INFO] Filtering training data...")
        if provenance_s1 is not None:
            x_s1, y_s1, retained_indices = _apply_outlier_filter(
                x_s1,
                y_s1,
                sqi_s1,
                sqi_threshold,
                sqi_keep_pct,
                return_indices=True,
            )
            provenance_s1 = provenance_s1.subset(retained_indices)
        else:
            x_s1, y_s1 = _apply_outlier_filter(
                x_s1,
                y_s1,
                sqi_s1,
                sqi_threshold,
                sqi_keep_pct,
            )

    if sqi_s2 is not None:
        print("\n[INFO] Filtering probe data...")
        if provenance_s2 is not None:
            x_s2, y_s2, retained_indices = _apply_outlier_filter(
                x_s2,
                y_s2,
                sqi_s2,
                absolute_threshold=sqi_threshold,
                keep_percentage=sqi_keep_pct,
                apply_subject_ranking=False,
                return_indices=True,
            )
            provenance_s2 = provenance_s2.subset(retained_indices)
        else:
            x_s2, y_s2 = _apply_outlier_filter(
                x_s2,
                y_s2,
                sqi_s2,
                absolute_threshold=sqi_threshold,
                keep_percentage=sqi_keep_pct,
                apply_subject_ranking=False,
            )

    (
        x_enroll_prepared,
        y_enroll_prepared,
        provenance_enroll_prepared,
        enrollment_is_explicit,
    ) = _prepare_cross_session_enrollment_role(
        x_enroll=x_enroll,
        y_enroll=y_enroll,
        provenance_enroll=provenance_enroll,
        use_enrollment=use_template,
    )

    # ====================================================
    # 2. INTERSECT AND SPLIT SUBJECTS
    # ====================================================
    eligible_subjects = set(
        y_s1
    ).intersection(
        set(y_s2)
    )

    if enrollment_is_explicit:
        eligible_subjects &= set(
            y_enroll_prepared
        )

    common_subs = sorted(
        eligible_subjects
    )
    
    if len(common_subs) < 2: 
        raise ValueError("[ERROR] Not enough common subjects across required roles after filtering.")
        
    (
        train_subs,
        val_subs,
        test_subs,
    ) = _split_subject_cohorts(
        common_subs,
        test_split=test_split,
        val_split=val_split,
        seed=resolved_split_seed,
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
            x_enrollment=x_enroll_prepared,
            y_enrollment=y_enroll_prepared,
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

    template_provenance = None
    provenance_probe = None

    if (
        enrollment_is_explicit
        and provenance_enroll_prepared is not None
    ):
        template_provenance = (
            provenance_enroll_prepared.subset(
                partitions[
                    "indices"
                ][
                    "enrollment"
                ]
            )
        )
    elif (
        not enrollment_is_explicit
        and provenance_s1 is not None
    ):
        template_provenance = provenance_s1.subset(
            partitions[
                "indices"
            ][
                "enrollment"
            ]
        )

    if provenance_s2 is not None:
        provenance_probe = provenance_s2.subset(
            partitions["indices"]["probe"]
        )

    # ====================================================
    # 3. ENCODE LABELS
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
            X_train, y_train_enc, test_size=val_split, stratify=y_train_enc, random_state=resolved_split_seed
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
    
    # Unseen-subject validation uses only training-role samples.
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
            "reproducibility_mode": reproducibility_mode,
            "matching_method": matching_method # Affects early stopping EER!
        }

        validation_arrays = _collect_validation_arrays(
            val_split, locals()
        )
        train_config = _build_weight_cache_config(
            loader,
            train_config,
            training_samples=X_tr,
            training_labels=y_tr,
            validation_arrays=validation_arrays,
            training_role_only_loader_identity=True,
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
            _record_persisted_trained_weight(
                model.weight_artifact_metadata,
                "cache_hit",
            )
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
                artifact_context=_build_weight_artifact_context(
                    X_tr,
                    num_train_classes,
                    resolved_split_seed,
                    reproducibility_state,
                ),
            )
            _record_newly_saved_trained_weight(cache, uid)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Train with the subject-disjoint validation objective.
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
        _record_in_memory_trained_weight(model)

    # ====================================================
    # 5. MODEL CALIBRATION (Optional)
    # ====================================================
    model.include_top = False
    
    if use_deployment_evaluation:
        print("\n[INFO] --- DEPLOYMENT THRESHOLD CALIBRATION ---")
        calib_loader = val_loader_s1
        calib_name = "Unseen validation (training role)"
            
        print(f"[INFO] Extracting features for Calibration ({calib_name})...")
        calib_emb_s1, calib_lab_s1 = _get_embeddings(model, calib_loader, device)
        
        # Calibration relies entirely on training-role validation features.
        print(f"[INFO] Generating Calibration Pairs to find Global Threshold...")
        calib_scores, calib_pair_labels = _generate_pairs(
            embeddings1=calib_emb_s1, labels1=calib_lab_s1, 
            embeddings2=None, labels2=None,
            pair_sampling_budget=pair_sampling_budget, pair_sampling_mode=pair_sampling_mode, max_impostor_pairs=max_impostor_pairs, pair_sampling_seed=pair_sampling_seed, matching_method=matching_method
        )
        global_threshold = _find_optimal_threshold(calib_scores, calib_pair_labels)
        print(f"[INFO] Optimal Global Threshold Found: {global_threshold:.4f}")

    # ====================================================
    # 6. EVALUATION STRATEGY ON UNSEEN TEST SUBJECTS
    # ====================================================
    probe_loader = _make_loader(X_probe, y_probe_enc, batch_size, shuffle=False)
    emb_probe, lab_probe = _get_embeddings(model, probe_loader, device)

    fusion_diagnostics = None
    template_diagnostics = None

    if enrollment_template_mode == "multi_template":
        print(
            "[INFO] Building multi-template gallery "
            f"({num_templates_per_identity} templates/identity, "
            f"{template_selection_method})..."
        )
        enroll_loader = _make_loader(X_enroll, y_enroll_enc, batch_size, shuffle=False)
        emb_enroll, lab_enroll = _get_embeddings(model, enroll_loader, device)
        (
            mt_templates,
            mt_template_identities,
            mt_template_source_indices,
            template_diagnostics,
        ) = _select_multi_templates(
            emb_enroll, lab_enroll,
            provenance=template_provenance,
            num_templates_per_identity=num_templates_per_identity,
            template_selection_method=template_selection_method,
            max_beats=template_size,
        )
        scores, labels_pair, fusion_diagnostics = _generate_multi_template_verification_pairs(
            probe_embeddings=emb_probe,
            probe_labels=lab_probe,
            probe_provenance=provenance_probe,
            template_embeddings=mt_templates,
            template_identities=mt_template_identities,
            num_templates_per_identity=num_templates_per_identity,
            probe_fusion_size=probe_fusion_size,
            matching_method=matching_method,
            pair_sampling_mode=pair_sampling_mode,
            pair_sampling_budget=pair_sampling_budget,
            max_impostor_pairs=max_impostor_pairs,
            pair_sampling_seed=pair_sampling_seed,
            template_score_aggregation=template_score_aggregation,
        )
    elif not use_template:
        print("[INFO] Bypassing templates. Generating verification pairs from probe data for unseen subjects...")
        scores, labels_pair = _generate_pairs(
            embeddings1=emb_probe, labels1=lab_probe, 
            embeddings2=None, labels2=None, 
            pair_sampling_budget=pair_sampling_budget, pair_sampling_mode=pair_sampling_mode, max_impostor_pairs=max_impostor_pairs, pair_sampling_seed=pair_sampling_seed, matching_method=matching_method
        )
    else:
        print("[INFO] Building enrollment templates for unseen subjects...")
        enroll_loader = _make_loader(X_enroll, y_enroll_enc, batch_size, shuffle=False)
        emb_enroll, lab_enroll = _get_embeddings(model, enroll_loader, device)
        
        templates, temp_labels = _create_templates(
            emb_enroll,
            lab_enroll,
            method=template_fusion_method,
            max_beats=template_size,
            provenance=template_provenance,
        )
        
        if probe_fusion_size > 1:
            scores, labels_pair, fusion_diagnostics = _generate_fused_verification_pairs(
                probe_embeddings=emb_probe,
                probe_labels=lab_probe,
                probe_provenance=provenance_probe,
                template_embeddings=templates,
                template_identities=temp_labels,
                group_size=probe_fusion_size,
                matching_method=matching_method,
                pair_sampling_mode=pair_sampling_mode,
                pair_sampling_budget=pair_sampling_budget,
                max_impostor_pairs=max_impostor_pairs,
                pair_sampling_seed=pair_sampling_seed,
            )
        else:
            scores, labels_pair = _generate_pairs(
                embeddings1=emb_probe,
                labels1=lab_probe,
                embeddings2=templates,
                labels2=temp_labels,
                pair_sampling_budget=pair_sampling_budget,
                pair_sampling_mode=pair_sampling_mode,
                max_impostor_pairs=max_impostor_pairs, pair_sampling_seed=pair_sampling_seed, matching_method=matching_method
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
        "Training Samples": len(X_train),
        "Validation Subjects": len(
            val_subs
        ),
        "Validation Samples": (
            len(X_val_s1)
            if X_val_s1 is not None
            else 0
        ),
        "Test Subjects": len(test_subs),
        "Enrollment Samples": (
            len(X_enroll)
            if use_template
            else 0
        ),
        "Probe Samples": len(
            X_probe
        ),
    }

    data_stats.update(
        _get_verification_pair_statistics(
            labels_pair,
            target_far=0.001,
        )
    )

    if fusion_diagnostics is not None:
        data_stats["Probe Fusion"] = fusion_diagnostics
    if template_diagnostics is not None:
        data_stats["Enrollment Templates"] = template_diagnostics

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
# PUBLIC TASK RUNNER INVENTORY
# =============================================================================
# The authoritative task-number -> runner mapping. Provenance classification
# tests build their coverage from this table instead of a separately
# maintained list, so adding, removing, or renaming a public task runner is
# a single, deliberate edit here rather than several silently-drifting ones.
PUBLIC_TASK_RUNNERS = {
    1: run_closed_set_identification,
    2: run_closed_set_verification,
    3: run_subject_disjoint_identification,
    4: run_subject_disjoint_verification,
    5: run_cross_session_identification,
    6: run_cross_session_verification,
    7: run_subject_disjoint_cross_session_identification,
    8: run_subject_disjoint_cross_session_verification,
}
