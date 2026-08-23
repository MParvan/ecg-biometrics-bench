import numpy as np
import pandas as pd
import os
from pathlib import Path
import glob
from typing import List, Optional, Dict
from collections.abc import Mapping
import datetime
import shutil
import zipfile
import tempfile
import requests
import yaml
import wfdb
import re
import patoolib
import collections
import hashlib
import statistics
from tqdm import tqdm  # For real-time progress bars
from preprocessing import Preprocessing

# =============================================================================
# PER-BEAT SOURCE PROVENANCE
# =============================================================================
# Every returned beat can be traced back to the recording, acquisition and
# source segment it was extracted from. The representation is columnar: one
# array per field, all sharing the first axis with the returned ``X``/``y``.
# Subject identity stays in ``y`` and is not duplicated here.
#
# ``acquisition_time`` is the genuine acquisition date or datetime when the
# source states one and ``None`` otherwise; it is never fabricated. When a role
# must be ordered and some acquisitions state no time, ``acquisition_order``
# supplies a stable dataset-defined order instead, which is a deterministic
# source order rather than a chronology. ``source_segment_order`` carries a
# numeric source position (a start minute or sample) so segments order without
# relying on lexical ``source_segment_id`` comparison.

PROVENANCE_COLUMNS = (
    "record_id",
    "session_id",
    "acquisition_time",
    "acquisition_order",
    "source_segment_id",
    "source_segment_order",
    "beat_ordinal",
    "rpeak_index",
)

_PROVENANCE_OBJECT_COLUMNS = frozenset(
    {
        "record_id",
        "session_id",
        "acquisition_time",
        "source_segment_id",
    }
)

_PROVENANCE_INT_COLUMNS = frozenset(
    {
        "acquisition_order",
        "beat_ordinal",
        "rpeak_index",
    }
)


def _provenance_column_dtype(name):
    if name in _PROVENANCE_OBJECT_COLUMNS:
        return object
    if name in _PROVENANCE_INT_COLUMNS:
        return np.int64
    return np.float64


class BeatProvenance:
    """Columnar per-beat source provenance aligned to ``X`` and ``y``."""

    __slots__ = ("columns",)

    def __init__(self, columns):
        self.columns = dict(columns)
        lengths = {len(value) for value in self.columns.values()}
        if len(lengths) > 1:
            raise ValueError(
                "All provenance columns must share one length."
            )

    def __len__(self):
        if not self.columns:
            return 0
        return len(next(iter(self.columns.values())))

    @classmethod
    def empty(cls):
        return cls(
            {
                name: np.empty(
                    (0,),
                    dtype=_provenance_column_dtype(name),
                )
                for name in PROVENANCE_COLUMNS
            }
        )

    def validate(self, expected_length=None):
        missing = [
            name for name in PROVENANCE_COLUMNS if name not in self.columns
        ]
        if missing:
            raise ValueError(
                f"Provenance is missing columns: {missing}."
            )

        lengths = {
            len(self.columns[name]) for name in PROVENANCE_COLUMNS
        }
        if len(lengths) != 1:
            raise ValueError(
                "Provenance columns have unequal lengths."
            )

        length = lengths.pop()
        if expected_length is not None and length != expected_length:
            raise ValueError(
                f"Provenance length {length} does not match the expected "
                f"{expected_length}."
            )
        return length

    def subset(self, indices):
        indices = np.asarray(indices)
        return BeatProvenance(
            {
                name: np.asarray(column)[indices]
                for name, column in self.columns.items()
            }
        )

    @classmethod
    def concatenate(cls, parts):
        parts = [part for part in parts if part is not None and len(part) > 0]
        if not parts:
            return cls.empty()
        return cls(
            {
                name: np.concatenate(
                    [part.columns[name] for part in parts]
                )
                for name in PROVENANCE_COLUMNS
            }
        )

    def to_cache_dict(self, prefix="provenance__"):
        return {
            f"{prefix}{name}": np.asarray(
                self.columns[name],
                dtype=_provenance_column_dtype(name),
            )
            for name in PROVENANCE_COLUMNS
        }

    @classmethod
    def from_cache_dict(
        cls,
        arrays,
        prefix="provenance__",
        expected_length=None,
    ):
        keys = [f"{prefix}{name}" for name in PROVENANCE_COLUMNS]
        if any(key not in arrays for key in keys):
            return None

        columns = {
            name: np.asarray(arrays[f"{prefix}{name}"])
            for name in PROVENANCE_COLUMNS
        }
        try:
            provenance = cls(columns)
            provenance.validate(expected_length)
        except ValueError:
            return None
        return provenance


class _ProvenanceBuilder:
    """Accumulate provenance blocks as beats are extracted."""

    def __init__(self):
        self._blocks = {name: [] for name in PROVENANCE_COLUMNS}

    def add_block(
        self,
        beat_count,
        record_id,
        session_id,
        acquisition_time,
        acquisition_order,
        source_segment_id,
        source_segment_order,
        rpeak_indices=None,
    ):
        if beat_count <= 0:
            return

        if rpeak_indices is None:
            rpeaks = np.full(beat_count, -1, dtype=np.int64)
        else:
            rpeaks = np.asarray(rpeak_indices, dtype=np.int64)
            if len(rpeaks) != beat_count:
                raise ValueError(
                    "rpeak_indices length must match beat_count."
                )

        self._blocks["record_id"].append(
            np.full(beat_count, record_id, dtype=object)
        )
        self._blocks["session_id"].append(
            np.full(beat_count, session_id, dtype=object)
        )
        self._blocks["acquisition_time"].append(
            np.array([acquisition_time] * beat_count, dtype=object)
        )
        self._blocks["acquisition_order"].append(
            np.full(beat_count, int(acquisition_order), dtype=np.int64)
        )
        self._blocks["source_segment_id"].append(
            np.full(beat_count, source_segment_id, dtype=object)
        )
        self._blocks["source_segment_order"].append(
            np.full(
                beat_count,
                float(source_segment_order),
                dtype=np.float64,
            )
        )
        self._blocks["beat_ordinal"].append(
            np.arange(beat_count, dtype=np.int64)
        )
        self._blocks["rpeak_index"].append(rpeaks)

    def build(self):
        columns = {}
        for name in PROVENANCE_COLUMNS:
            blocks = self._blocks[name]
            if blocks:
                columns[name] = np.concatenate(blocks)
            else:
                columns[name] = np.empty(
                    (0,),
                    dtype=_provenance_column_dtype(name),
                )
        return BeatProvenance(columns)



def _finalize_loader_output(
    x_list,
    y_list,
    provenance_builder,
    return_provenance,
):
    """Assemble the loader return value, optionally with aligned provenance."""
    if x_list:
        samples = np.vstack(x_list)
        labels = np.array(y_list)
    else:
        samples = np.empty((0, 0))
        labels = np.empty((0,))

    if not return_provenance:
        return samples, labels

    provenance = (
        provenance_builder.build()
        if provenance_builder is not None
        else BeatProvenance.empty()
    )
    provenance.validate(len(labels))
    return samples, labels, provenance

# =============================================================================
# CONFIGURATION LOADING
# =============================================================================
def load_config(config_name: str = "config.yaml") -> dict:
    """
    Robust config loader that searches multiple locations.
    Priority:
    1. Same directory as this script.
    2. Current working directory (CWD).
    3. Parent directory of CWD.
    """
    search_paths = [
        Path(__file__).resolve().parent / config_name,
        Path.cwd() / config_name,
        Path.cwd().parent / config_name
    ]

    for path in search_paths:
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f)
                
    raise FileNotFoundError(f"Could not find '{config_name}'. Checked: {[str(p) for p in search_paths]}")

CONFIG = load_config()


# =============================================================================
# PREPROCESSING CONFIGURATION
# =============================================================================
DEFAULT_PREPROCESSING_CONFIG = {
    "mode": "beat",
    "pre_s": 0.2,
    "post_s": 0.4,
    "resample_len": None,
    "window_s": 5.0,
    "stride_s": 1.0,
    "rpeak_method": "pantompkins",
    "align_peak": True,
    "align_window_s": 0.10,
    "filter_method": "butter",
    "filter_parameters": {
        "low": 0.5,
        "high": 40.0,
        "order": 4,
    },
    "normalization_method": "zscore",
}


def _normalize_preprocessing_config(*configurations):
    """
    Merge and validate preprocessing settings using one canonical schema.

    Configurations are applied from left to right. The function accepts the
    historical keys used by earlier repository versions, but always returns
    the complete canonical mapping recorded in caches and experiment logs.
    """
    normalized = {
        **DEFAULT_PREPROCESSING_CONFIG,
        "filter_parameters": dict(
            DEFAULT_PREPROCESSING_CONFIG["filter_parameters"]
        ),
    }

    canonical_keys = {
        "mode",
        "pre_s",
        "post_s",
        "resample_len",
        "window_s",
        "stride_s",
        "rpeak_method",
        "align_peak",
        "align_window_s",
        "filter_method",
        "filter_parameters",
        "normalization_method",
    }

    legacy_keys = {
        "window_len",
        "stride",
        "bandpass",
        "lowcut",
        "highcut",
        "filter_order",
        "filter_kwargs",
        "normalize",
        "norm_method",
    }

    for configuration in configurations:
        if configuration is None:
            continue

        if not isinstance(configuration, Mapping):
            raise ValueError(
                "Preprocessing configuration must be a mapping or None."
            )

        configuration = dict(configuration)
        unknown_keys = (
            set(configuration)
            - canonical_keys
            - legacy_keys
        )

        if unknown_keys:
            raise ValueError(
                "Unknown preprocessing parameter(s): "
                + ", ".join(sorted(str(key) for key in unknown_keys))
            )

        translated = {}

        for key in canonical_keys - {"filter_parameters"}:
            if key in configuration:
                translated[key] = configuration[key]

        if (
            "window_s" not in configuration
            and "window_len" in configuration
        ):
            translated["window_s"] = configuration["window_len"]

        if (
            "stride_s" not in configuration
            and "stride" in configuration
        ):
            translated["stride_s"] = configuration["stride"]

        if (
            "filter_method" not in configuration
            and "bandpass" in configuration
        ):
            bandpass = configuration["bandpass"]

            if not isinstance(bandpass, (bool, np.bool_)):
                raise ValueError("bandpass must be Boolean.")

            translated["filter_method"] = (
                "butter" if bandpass else None
            )

        if (
            "normalization_method" not in configuration
            and "norm_method" in configuration
        ):
            translated["normalization_method"] = configuration[
                "norm_method"
            ]
        elif (
            "normalization_method" not in configuration
            and "normalize" in configuration
        ):
            normalize = configuration["normalize"]

            if not isinstance(normalize, (bool, np.bool_)):
                raise ValueError("normalize must be Boolean.")

            translated["normalization_method"] = (
                "zscore" if normalize else None
            )

        if (
            "filter_method" in translated
            and translated["filter_method"]
            != normalized["filter_method"]
        ):
            normalized["filter_parameters"] = {}

        normalized.update(translated)

        filter_parameters = {}

        if "filter_kwargs" in configuration:
            legacy_filter_parameters = configuration["filter_kwargs"]

            if not isinstance(legacy_filter_parameters, Mapping):
                raise ValueError("filter_kwargs must be a mapping.")

            filter_parameters.update(legacy_filter_parameters)

        if "lowcut" in configuration:
            filter_parameters["low"] = configuration["lowcut"]

        if "highcut" in configuration:
            filter_parameters["high"] = configuration["highcut"]

        if "filter_order" in configuration:
            filter_parameters["order"] = configuration["filter_order"]

        if "filter_parameters" in configuration:
            explicit_filter_parameters = configuration[
                "filter_parameters"
            ]

            if not isinstance(explicit_filter_parameters, Mapping):
                raise ValueError(
                    "filter_parameters must be a mapping."
                )

            filter_parameters.update(explicit_filter_parameters)

        normalized["filter_parameters"].update(filter_parameters)

    normalized["mode"] = str(normalized["mode"]).strip().lower()

    if normalized["mode"] not in {"beat", "blind"}:
        raise ValueError("mode must be 'beat' or 'blind'.")

    for parameter_name in [
        "pre_s",
        "post_s",
        "window_s",
        "stride_s",
    ]:
        value = normalized[parameter_name]

        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
            or float(value) <= 0.0
        ):
            raise ValueError(
                f"{parameter_name} must be a finite positive number."
            )

        normalized[parameter_name] = float(value)

    resample_len = normalized["resample_len"]

    if resample_len is not None:
        if (
            isinstance(resample_len, (bool, np.bool_))
            or not isinstance(resample_len, (int, np.integer))
            or int(resample_len) < 1
        ):
            raise ValueError(
                "resample_len must be a positive integer or None."
            )

        normalized["resample_len"] = int(resample_len)

    rpeak_method = normalized["rpeak_method"]

    if not isinstance(rpeak_method, str) or not rpeak_method.strip():
        raise ValueError("rpeak_method must be a non-empty string.")

    normalized["rpeak_method"] = rpeak_method.strip().lower()

    if not isinstance(normalized["align_peak"], (bool, np.bool_)):
        raise ValueError("align_peak must be Boolean.")

    normalized["align_peak"] = bool(normalized["align_peak"])

    align_window_s = normalized["align_window_s"]

    # The width is a duration, so every accepted value is a positive number of
    # seconds. Omitting the key takes the documented default; an explicit null
    # is reported rather than filled in, so that the file and the run always
    # describe the same search.
    if align_window_s is None:
        raise ValueError(
            "align_window_s must be a positive number of seconds. Omit the "
            "key to take the default of "
            f"{DEFAULT_PREPROCESSING_CONFIG['align_window_s']} s, which is the "
            "width the reported results were produced with."
        )

    if isinstance(align_window_s, bool) or not np.isscalar(align_window_s):
        raise ValueError(
            "align_window_s must be a positive number of seconds."
        )

    align_window_s = float(align_window_s)

    if not np.isfinite(align_window_s) or align_window_s <= 0.0:
        raise ValueError(
            "align_window_s must be a positive number of seconds."
        )

    normalized["align_window_s"] = align_window_s

    filter_method = normalized["filter_method"]

    if isinstance(filter_method, str):
        filter_method = filter_method.strip().lower()

        if filter_method in {"", "none", "null"}:
            filter_method = None

    if filter_method not in {None, "butter", "fir", "notch", "savgol"}:
        raise ValueError(
            "filter_method must be one of: butter, fir, notch, "
            "savgol, or None."
        )

    normalized["filter_method"] = filter_method

    filter_parameters = dict(normalized["filter_parameters"])

    if filter_method == "butter":
        order = filter_parameters.get("order", 4)

        if (
            isinstance(order, (bool, np.bool_))
            or not isinstance(order, (int, np.integer))
            or int(order) < 1
        ):
            raise ValueError(
                "Butterworth filter order must be a positive integer."
            )

        filter_parameters["order"] = int(order)

        for cutoff_name in ["low", "high"]:
            cutoff = filter_parameters.get(cutoff_name)

            if cutoff is None:
                continue

            if (
                isinstance(cutoff, (bool, np.bool_))
                or not isinstance(
                    cutoff,
                    (int, float, np.integer, np.floating),
                )
                or not np.isfinite(cutoff)
                or float(cutoff) <= 0.0
            ):
                raise ValueError(
                    f"Butterworth {cutoff_name} cutoff must be "
                    "a positive finite number or None."
                )

            filter_parameters[cutoff_name] = float(cutoff)

        low = filter_parameters.get("low")
        high = filter_parameters.get("high")

        if low is not None and high is not None and low >= high:
            raise ValueError(
                "Butterworth cutoffs must satisfy low < high."
            )

    if filter_method is None:
        filter_parameters = {}

    normalized["filter_parameters"] = filter_parameters

    normalization_method = normalized["normalization_method"]

    if isinstance(normalization_method, str):
        normalization_method = normalization_method.strip().lower()

        if normalization_method in {"", "none", "null"}:
            normalization_method = None

    if normalization_method not in {None, "zscore", "minmax"}:
        raise ValueError(
            "normalization_method must be 'zscore', 'minmax', or None."
        )

    normalized["normalization_method"] = normalization_method

    return normalized


def _preprocess_signal(preprocessor, signal, fs, preprocessing_config):
    """Apply the complete effective preprocessing configuration."""
    config = _normalize_preprocessing_config(preprocessing_config)

    return preprocessor.preprocess_ecg(
        ecg=signal,
        fs=fs,
        mode=config["mode"],
        pre_s=config["pre_s"],
        post_s=config["post_s"],
        resample_len=config["resample_len"],
        window_s=config["window_s"],
        stride_s=config["stride_s"],
        rpeak_method=config["rpeak_method"],
        align_peak=config["align_peak"],
        align_window_s=config["align_window_s"],
        filter_method=config["filter_method"],
        filter_kwargs=dict(config["filter_parameters"]),
        norm_method=config["normalization_method"],
    )

# =============================================================================
# CONTINUOUS-RECORDING TEMPORAL PARTITIONS
# =============================================================================
# MIT-BIH and NSRDB are single continuous recordings per subject rather than
# session-structured datasets. Their biometric partitions are therefore defined
# as explicit minute windows into one timeline, which makes it possible for a
# careless configuration to place the same physical samples in both the
# enrollment and the probe partition. The helpers below make the realized
# windows explicit and reject overlapping enrollment/probe definitions.

_CONTINUOUS_PARTITION_ROLES = (
    "train",
    "enrol",
    "test",
)


def _normalize_minute_ranges(ranges, role):
    """
    Validate one partition definition and return sorted (start, end) minutes.

    ``None`` and empty definitions are returned as an empty list because not
    every regime populates every partition.
    """
    if ranges is None:
        return []

    if isinstance(ranges, Mapping):
        raise ValueError(
            f"{role}_parts must be a sequence of "
            "(start_minute, end_minute) pairs."
        )

    normalized = []

    for entry in ranges:
        if isinstance(entry, (str, bytes)) or not hasattr(entry, "__iter__"):
            raise ValueError(
                f"Every {role}_parts entry must be a "
                "(start_minute, end_minute) pair."
            )

        bounds = list(entry)

        if len(bounds) != 2:
            raise ValueError(
                f"Every {role}_parts entry must contain exactly two "
                f"minute boundaries, received {bounds!r}."
            )

        try:
            start_minute = float(bounds[0])
            end_minute = float(bounds[1])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Every {role}_parts boundary must be numeric, "
                f"received {bounds!r}."
            ) from error

        if not (np.isfinite(start_minute) and np.isfinite(end_minute)):
            raise ValueError(
                f"Every {role}_parts boundary must be finite, "
                f"received {bounds!r}."
            )

        if start_minute < 0.0:
            raise ValueError(
                f"{role}_parts cannot start before minute 0, "
                f"received {bounds!r}."
            )

        if start_minute >= end_minute:
            raise ValueError(
                f"Every {role}_parts entry must satisfy "
                f"start_minute < end_minute, received {bounds!r}."
            )

        normalized.append((start_minute, end_minute))

    normalized.sort()

    return normalized


def _merge_minute_ranges(ranges):
    """
    Collapse sorted minute ranges into their minimal disjoint coverage.
    """
    coverage = []

    for start_minute, end_minute in sorted(ranges):
        if coverage and start_minute <= coverage[-1][1]:
            previous_start, previous_end = coverage[-1]
            coverage[-1] = (
                previous_start,
                max(previous_end, end_minute),
            )
        else:
            coverage.append((start_minute, end_minute))

    return coverage


def _minute_range_intersections(first_ranges, second_ranges):
    """
    Return every non-empty intersection between two minute coverages.
    """
    intersections = []

    for first_start, first_end in first_ranges:
        for second_start, second_end in second_ranges:
            overlap_start = max(first_start, second_start)
            overlap_end = min(first_end, second_end)

            if overlap_start < overlap_end:
                intersections.append(
                    (overlap_start, overlap_end)
                )

    return _merge_minute_ranges(intersections)


def _minute_range_separation(first_ranges, second_ranges):
    """
    Return the smallest temporal gap in minutes between two coverages.

    Touching-but-disjoint coverages such as (0, 5) and (5, 10) return 0.0.
    ``None`` is returned when either coverage is empty.
    """
    if not first_ranges or not second_ranges:
        return None

    separations = []

    for first_start, first_end in first_ranges:
        for second_start, second_end in second_ranges:
            if first_end <= second_start:
                separations.append(second_start - first_end)
            elif second_end <= first_start:
                separations.append(first_start - second_end)
            else:
                separations.append(0.0)

    return min(separations)


def _total_covered_minutes(ranges):
    """
    Return the total duration in minutes spanned by a merged coverage.
    """
    return float(
        sum(
            end_minute - start_minute
            for start_minute, end_minute in ranges
        )
    )


def audit_continuous_temporal_partitions(
    train_parts=None,
    enrol_parts=None,
    test_parts=None,
    temporal_guard_minutes=0.0,
):
    """
    Audit the enrollment and probe windows of a continuous-recording split.

    ``train_parts`` supplies representation-learning samples and
    ``enrol_parts`` supplies gallery/template samples. The two roles may share
    source windows or be identical by design. Overlap between either role and
    ``test_parts`` places identical physical samples on the source and probe
    sides of the comparison and is rejected.

    Args:
        train_parts: Minute ranges used for representation learning.
        enrol_parts: Minute ranges used for gallery/template construction.
        test_parts: Minute ranges used for probes.
        temporal_guard_minutes: Minimum required separation in minutes
            between the enrollment coverage and the probe coverage. The
            default of 0.0 permits directly adjacent windows and only
            rejects true overlap.

    Returns:
        dict: The realized coverages, their durations, the achieved
        separation, and the guard that was enforced.

    Raises:
        ValueError: If any partition is malformed, if enrollment and probe
            windows overlap, or if the achieved separation is smaller than
            ``temporal_guard_minutes``.
    """
    try:
        temporal_guard_minutes = float(temporal_guard_minutes)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "temporal_guard_minutes must be numeric."
        ) from error

    if not np.isfinite(temporal_guard_minutes):
        raise ValueError(
            "temporal_guard_minutes must be finite."
        )

    if temporal_guard_minutes < 0.0:
        raise ValueError(
            "temporal_guard_minutes cannot be negative."
        )

    normalized_parts = {
        "train": _normalize_minute_ranges(train_parts, "train"),
        "enrol": _normalize_minute_ranges(enrol_parts, "enrol"),
        "test": _normalize_minute_ranges(test_parts, "test"),
    }

    coverage = {
        role: _merge_minute_ranges(normalized_parts[role])
        for role in _CONTINUOUS_PARTITION_ROLES
    }

    enrollment_coverage = _merge_minute_ranges(
        normalized_parts["train"] + normalized_parts["enrol"]
    )
    probe_coverage = coverage["test"]

    # Reported separately so a response letter can cite either check.
    leakage_checks = {
        "train_vs_test": _minute_range_intersections(
            coverage["train"],
            probe_coverage,
        ),
        "enrol_vs_test": _minute_range_intersections(
            coverage["enrol"],
            probe_coverage,
        ),
    }

    violations = []

    for check_name, intersections in leakage_checks.items():
        if intersections:
            violations.append(
                f"{check_name.replace('_', ' ')} overlap at "
                + ", ".join(
                    f"[{start:g}, {end:g}) minutes"
                    for start, end in intersections
                )
            )

    if violations:
        raise ValueError(
            "Continuous-recording partitions share samples between "
            "enrollment and probe windows: "
            + "; ".join(violations)
            + ". Enrollment and probe minute ranges must be disjoint so "
            "that reported performance is not inflated by evaluating on "
            "signal already seen during training or enrollment."
        )

    achieved_separation = _minute_range_separation(
        enrollment_coverage,
        probe_coverage,
    )

    if (
        temporal_guard_minutes > 0.0
        and achieved_separation is not None
        and achieved_separation < temporal_guard_minutes
    ):
        raise ValueError(
            "Continuous-recording partitions are separated by only "
            f"{achieved_separation:g} minutes, which is below the "
            f"requested guard band of {temporal_guard_minutes:g} minutes. "
            "Widen the gap between the enrollment and probe windows or "
            "lower temporal_guard_minutes."
        )

    return {
        "partitions": {
            role: [list(window) for window in coverage[role]]
            for role in _CONTINUOUS_PARTITION_ROLES
        },
        "enrollment_coverage": [
            list(window) for window in enrollment_coverage
        ],
        "probe_coverage": [
            list(window) for window in probe_coverage
        ],
        "covered_minutes": {
            "enrollment": _total_covered_minutes(enrollment_coverage),
            "probe": _total_covered_minutes(probe_coverage),
        },
        "train_enrol_shared_coverage": [
            list(window)
            for window in _minute_range_intersections(
                coverage["train"],
                coverage["enrol"],
            )
        ],
        "achieved_separation_minutes": achieved_separation,
        "temporal_guard_minutes": temporal_guard_minutes,
        "overlap_free": True,
    }


# =============================================================================
# RECORD-ORDER REGIME SELECTION
# =============================================================================
# ECG-ID, PTB, and PTB-XL contain a variable number of recordings per subject
# without a uniform session structure, so their regimes are derived from the
# deterministic record order. The selection rules are defined once here and
# used by every loader, which means the temporal-causality audit reports the
# same assignment the pipeline actually trains and evaluates on rather than a
# re-implementation of it.

RECORD_ORDER_SPLIT_MODES = (
    "single-cross-session",
    "single-shot-short-term",
    "leave-last-out-short-term",
    "single-shot-long-term",
    "leave-last-out-long-term",
)

# How the record-order regimes treat a recording whose header states no
# acquisition date: it is left unknown and takes no part in an ordering that
# would read elapsed time into it. Every loader that orders recordings by date
# carries this value so that it reaches the cache identity, because a cache
# built under a different policy describes a different partition.
TEMPORAL_DATE_POLICY = "known_dates_only"

# The predefined record-order regimes retain their historical two-sided
# Train/Enroll versus Probe semantics. This separate mode exposes explicit
# recording positions for protocols that need independent training,
# enrollment, and probe sources.
CUSTOM_RECORD_SPLIT_MODE = "custom-record-split"
RECORD_BASED_SPLIT_MODES = (
    ("all-available", "single-session")
    + RECORD_ORDER_SPLIT_MODES
    + (CUSTOM_RECORD_SPLIT_MODE,)
)


def verify_sampling_rate(declared_fs, dataset_name, record_name, measured_fs):
    """
    Check a recording's sampling rate against the rate the configuration declares.

    The rate stored in the file is what preprocessing uses, so this does not
    override it. Its purpose is to catch a silent mismatch: a repository that
    re-releases a database at a different rate, or a configuration edited to a
    value the files do not carry. Either would leave every filter cutoff and
    beat window subtly wrong while the pipeline reported success.

    Takes the declared rate rather than the configuration mapping, so that the
    key a loader depends on is visible at the call site.

    Args:
        declared_fs: Rate declared for the dataset, or None to skip the check.
        dataset_name: Name used in the message.
        record_name: Recording the rate was read from.
        measured_fs: Sampling rate carried by the file.

    Raises:
        ValueError: When the file disagrees with the declared rate.
    """
    if declared_fs is None or measured_fs is None:
        return

    if int(declared_fs) != int(measured_fs):
        raise ValueError(
            f"{dataset_name}: {record_name} is sampled at {int(measured_fs)} Hz "
            f"but config.yaml declares {int(declared_fs)} Hz. Preprocessing "
            "uses the rate stored in the file, so correct the configuration or "
            "re-download the database before relying on the result."
        )



def _canonical_record_role(role):
    """
    Normalize a record-based data role.

    Boolean values remain accepted for compatibility with the historical
    enrollment/probe logging helper.
    """
    if isinstance(role, (bool, np.bool_)):
        return "enrollment" if bool(role) else "probe"

    if not isinstance(role, str):
        raise ValueError(
            "Record role must be 'train', 'enrollment', or 'probe'."
        )

    normalized = role.strip().lower()

    aliases = {
        "train": "train",
        "enrol": "enrollment",
        "enroll": "enrollment",
        "enrollment": "enrollment",
        "probe": "probe",
        "test": "probe",
    }

    if normalized not in aliases:
        raise ValueError(
            "Record role must be 'train', 'enrollment', or 'probe'."
        )

    return aliases[normalized]


def _normalize_record_indices(indices, argument_name):
    """
    Validate one explicit recording-index selector.

    Indices are zero-based positions in the loader's deterministic per-subject
    recording order. The normalized result is sorted so YAML ordering cannot
    change source ordering.
    """
    if indices is None:
        return None

    if isinstance(indices, (str, bytes)) or not isinstance(
        indices,
        (list, tuple, np.ndarray),
    ):
        raise ValueError(
            f"{argument_name} must be a non-empty sequence of "
            "zero-based integer recording indices."
        )

    normalized = []

    for value in indices:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise ValueError(
                f"{argument_name} must contain only integer "
                "recording indices."
            )

        value = int(value)

        if value < 0:
            raise ValueError(
                f"{argument_name} cannot contain negative indices."
            )

        normalized.append(value)

    if not normalized:
        raise ValueError(
            f"{argument_name} cannot be empty."
        )

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            f"{argument_name} cannot contain duplicate indices."
        )

    return tuple(sorted(normalized))


def _resolve_custom_record_indices(
    train_record_indices,
    enroll_record_indices,
    probe_record_indices,
):
    """
    Resolve an explicit Train/Enroll/Probe recording assignment.

    Enrollment defaults to training when omitted. Training and enrollment may
    share recordings, while the probe role must remain physically disjoint from
    both.
    """
    train = _normalize_record_indices(
        train_record_indices,
        "train_record_indices",
    )
    probe = _normalize_record_indices(
        probe_record_indices,
        "probe_record_indices",
    )

    if train is None:
        raise ValueError(
            "custom-record-split requires train_record_indices."
        )

    if probe is None:
        raise ValueError(
            "custom-record-split requires probe_record_indices."
        )

    if enroll_record_indices is None:
        enrollment = tuple(train)
    else:
        enrollment = _normalize_record_indices(
            enroll_record_indices,
            "enroll_record_indices",
        )

    source_side = set(train) | set(enrollment)
    shared_probe_indices = source_side & set(probe)

    if shared_probe_indices:
        raise ValueError(
            "Probe recordings must be disjoint from training and "
            "enrollment recordings. Shared record index/indices: "
            f"{sorted(shared_probe_indices)}."
        )

    return train, enrollment, probe


def _configure_record_role_indices(
    loader,
    data_split_mode,
    train_record_indices,
    enroll_record_indices,
    probe_record_indices,
):
    """
    Attach explicit record-role selectors only to custom-record-split loaders.
    """
    supplied = any(
        value is not None
        for value in (
            train_record_indices,
            enroll_record_indices,
            probe_record_indices,
        )
    )

    if data_split_mode != CUSTOM_RECORD_SPLIT_MODE:
        if supplied:
            raise ValueError(
                "Explicit record-role selectors require "
                "data_split_mode='custom-record-split'."
            )
        return

    (
        train,
        enrollment,
        probe,
    ) = _resolve_custom_record_indices(
        train_record_indices,
        enroll_record_indices,
        probe_record_indices,
    )

    loader.train_record_indices = train
    loader.enroll_record_indices = enrollment
    loader.probe_record_indices = probe


def select_record_role_partition(
    records,
    data_split_mode,
    role,
    train_record_indices=None,
    enroll_record_indices=None,
    probe_record_indices=None,
):
    """
    Select one semantic role from a record-based dataset.

    Existing record-order modes delegate unchanged to
    ``select_record_order_partition``. ``custom-record-split`` instead uses
    explicit zero-based recording positions. A subject is eligible only when
    every recording required by all three roles exists, so an incomplete
    subject is omitted consistently from Train, Enroll, and Probe.
    """
    role = _canonical_record_role(role)

    if data_split_mode != CUSTOM_RECORD_SPLIT_MODE:
        return select_record_order_partition(
            records,
            data_split_mode,
            is_enrollment=(role != "probe"),
        )

    (
        train,
        enrollment,
        probe,
    ) = _resolve_custom_record_indices(
        train_record_indices,
        enroll_record_indices,
        probe_record_indices,
    )

    selectors = {
        "train": train,
        "enrollment": enrollment,
        "probe": probe,
    }

    required_indices = (
        set(train)
        | set(enrollment)
        | set(probe)
    )

    if not records:
        return [], False

    if max(required_indices) >= len(records):
        return [], False

    selected = [
        records[index]
        for index in selectors[role]
    ]

    return selected, True


def select_record_order_partition(records, data_split_mode, is_enrollment):
    """
    Select the recordings assigned to one partition of a record-order regime.

    Args:
        records: Recordings for a single subject, already sorted by
            (date, record order).
        data_split_mode: One of ``RECORD_ORDER_SPLIT_MODES``.
        is_enrollment: True for the training/enrollment partition, False for
            the probe partition.

    Returns:
        tuple: ``(selected_records, subject_is_eligible)``. Subjects that lack
        the structure a regime requires yield ``([], False)`` and are dropped
        by the caller.
    """
    if data_split_mode not in RECORD_ORDER_SPLIT_MODES:
        raise ValueError(
            f"Unsupported record-order split mode: {data_split_mode!r}. "
            f"Use one of {list(RECORD_ORDER_SPLIT_MODES)}."
        )

    # Every regime below reads meaning into the order of the dates: which
    # recording came first, which day is the last one, how far apart the two
    # partitions sit. A recording whose header states no acquisition date
    # carries none of that evidence, so it takes no part here. Such recordings
    # stay available to "all-available", which draws on every recording without
    # ordering them.
    records = [
        record
        for record in records
        if record.get("date") is not None
    ]

    if not records:
        return [], False

    unique_dates = sorted({record["date"] for record in records})
    day1_date = unique_dates[0]
    day1_records = [
        record
        for record in records
        if record["date"] == day1_date
    ]

    if data_split_mode == "single-cross-session":
        if len(records) < 2:
            return [], False

        return (
            [records[0]] if is_enrollment else [records[1]],
            True,
        )

    if data_split_mode == "single-shot-short-term":
        if len(day1_records) < 2:
            return [], False

        return (
            [day1_records[0]]
            if is_enrollment
            else day1_records[1:],
            True,
        )

    if data_split_mode == "leave-last-out-short-term":
        if len(day1_records) < 2:
            return [], False

        return (
            day1_records[:-1]
            if is_enrollment
            else [day1_records[-1]],
            True,
        )

    if data_split_mode == "single-shot-long-term":
        if len(unique_dates) < 2:
            return [], False

        return (
            day1_records
            if is_enrollment
            else [
                record
                for record in records
                if record["date"] > day1_date
            ],
            True,
        )

    # leave-last-out-long-term
    if len(unique_dates) < 2:
        return [], False

    last_date = unique_dates[-1]

    return (
        [
            record
            for record in records
            if record["date"] < last_date
        ]
        if is_enrollment
        else [
            record
            for record in records
            if record["date"] == last_date
        ],
        True,
    )


def _record_sort_key(record, records):
    """
    Return the deterministic (date, position) ordering key of one recording.
    """
    return (
        record["date"],
        records.index(record),
    )



def _record_partition_assignment(loader, role, subject_id, records):
    """
    Log the recordings assigned to one semantic data role.

    Boolean role values remain supported for existing callers: True maps to
    enrollment and False maps to probe. Explicit three-role protocols use the
    ``train``, ``enrollment``, and ``probe`` keys independently.
    """
    role = _canonical_record_role(role)

    partition_log = getattr(
        loader,
        "partition_assignment_log",
        None,
    )

    if partition_log is None:
        partition_log = {}
        loader.partition_assignment_log = partition_log

    for role_name in (
        "train",
        "enrollment",
        "probe",
    ):
        partition_log.setdefault(
            role_name,
            {},
        )

    partition_log[role][str(subject_id)] = [
        {
            "filename": str(
                record.get("filename", "")
            ),
            "date": str(
                record.get("date", "")
            ),
        }
        for record in records
    ]



def summarize_partition_log(loader):
    """
    Summarize record assignments and check source-side separation from probes.

    Training and enrollment are allowed to share recordings. Probe recordings
    are checked against the union of both roles. Historical two-role logs that
    contain only enrollment and probe entries remain valid.
    """
    partition_log = getattr(
        loader,
        "partition_assignment_log",
        None,
    )

    if not partition_log:
        return None

    training_by_subject = partition_log.get(
        "train",
        {},
    )
    enrollment_by_subject = partition_log.get(
        "enrollment",
        {},
    )
    probe_by_subject = partition_log.get(
        "probe",
        {},
    )

    source_subjects = (
        set(training_by_subject)
        | set(enrollment_by_subject)
    )

    shared_subjects = sorted(
        source_subjects
        & set(probe_by_subject)
    )

    violations = []

    for subject_id in shared_subjects:
        source_entries = (
            list(
                training_by_subject.get(
                    subject_id,
                    [],
                )
            )
            + list(
                enrollment_by_subject.get(
                    subject_id,
                    [],
                )
            )
        )

        probe_entries = probe_by_subject[
            subject_id
        ]

        source_files = {
            entry["filename"]
            for entry in source_entries
        }

        probe_files = {
            entry["filename"]
            for entry in probe_entries
        }

        shared_files = (
            source_files
            & probe_files
        )

        if shared_files:
            violations.append(
                {
                    "subject": subject_id,
                    "reasons": [
                        "training/enrollment and probe partitions "
                        "share recording(s): "
                        f"{sorted(shared_files)}"
                    ],
                }
            )
            continue

        source_dates = [
            entry["date"]
            for entry in source_entries
            if entry.get("date") not in {
                "",
                "None",
            }
        ]

        probe_dates = [
            entry["date"]
            for entry in probe_entries
            if entry.get("date") not in {
                "",
                "None",
            }
        ]

        if not (
            source_dates
            and probe_dates
        ):
            continue

        if max(source_dates) > min(probe_dates):
            violations.append(
                {
                    "subject": subject_id,
                    "reasons": [
                        "a training/enrollment recording is dated "
                        "after the first probe recording"
                    ],
                }
            )

    return {
        "data_split_mode": getattr(
            loader,
            "data_split_mode",
            None,
        ),
        "subjects_trained": len(
            training_by_subject
        ),
        "subjects_enrolled": len(
            enrollment_by_subject
        ),
        "subjects_probed": len(
            probe_by_subject
        ),
        "subjects_audited": len(
            shared_subjects
        ),
        "violations": violations,
        "enrollment_precedes_probe": (
            not violations
        ),
    }


def audit_record_order_causality(records_by_subject, data_split_mode):
    """
    Verify that enrollment recordings strictly precede probe recordings.

    The audit re-uses ``select_record_order_partition``, so it reflects the
    assignment the pipeline performs rather than an independent restatement
    of the protocol. A regime is causal when, for every eligible subject, the
    latest enrollment recording occurs no later than the earliest probe
    recording and the two partitions share no recording.

    Args:
        records_by_subject: Mapping of subject identifier to its recordings,
            sorted by (date, record order). Only ``date`` and an identifying
            key such as ``filename`` are required, so the audit can run on
            header metadata without loading any signal.
        data_split_mode: One of ``RECORD_ORDER_SPLIT_MODES``.

    Returns:
        dict: Per-subject assignment, aggregate counts, and any violations.
    """
    subject_reports = []
    violations = []
    separations = []
    eligible_subjects = 0
    total_enrollment_records = 0
    total_probe_records = 0

    for subject_id in sorted(records_by_subject):
        records = list(records_by_subject[subject_id])

        (
            enrollment_records,
            enrollment_eligible,
        ) = select_record_order_partition(
            records,
            data_split_mode,
            is_enrollment=True,
        )
        (
            probe_records,
            probe_eligible,
        ) = select_record_order_partition(
            records,
            data_split_mode,
            is_enrollment=False,
        )

        if not (enrollment_eligible and probe_eligible):
            continue

        eligible_subjects += 1
        total_enrollment_records += len(enrollment_records)
        total_probe_records += len(probe_records)

        enrollment_keys = [
            _record_sort_key(record, records)
            for record in enrollment_records
        ]
        probe_keys = [
            _record_sort_key(record, records)
            for record in probe_records
        ]

        subject_violations = []

        shared_positions = {
            key[1] for key in enrollment_keys
        } & {
            key[1] for key in probe_keys
        }

        if shared_positions:
            subject_violations.append(
                "enrollment and probe partitions share "
                f"{len(shared_positions)} recording(s)"
            )

        if enrollment_keys and probe_keys:
            latest_enrollment = max(enrollment_keys)
            earliest_probe = min(probe_keys)

            if latest_enrollment >= earliest_probe:
                subject_violations.append(
                    "an enrollment recording is not earlier than the "
                    "first probe recording"
                )

        if subject_violations:
            violations.append(
                {
                    "subject": subject_id,
                    "reasons": subject_violations,
                }
            )

        separation_days = _partition_separation_days(
            enrollment_keys,
            probe_keys,
        )

        if separation_days is not None:
            separations.append(separation_days)

        subject_reports.append(
            {
                "subject": subject_id,
                "enrollment_records": [
                    str(record.get("filename", ""))
                    for record in enrollment_records
                ],
                "probe_records": [
                    str(record.get("filename", ""))
                    for record in probe_records
                ],
                "latest_enrollment_date": (
                    str(max(enrollment_keys)[0])
                    if enrollment_keys
                    else None
                ),
                "earliest_probe_date": (
                    str(min(probe_keys)[0])
                    if probe_keys
                    else None
                ),
                "separation_days": separation_days,
            }
        )

    return {
        "data_split_mode": data_split_mode,
        "subjects_supplied": len(records_by_subject),
        "subjects_eligible": eligible_subjects,
        "subjects_audited": len(subject_reports),
        "violations": violations,
        "enrollment_precedes_probe": not violations,
        "temporal_separation": summarize_temporal_separation(
            separations,
            enrollment_records=total_enrollment_records,
            probe_records=total_probe_records,
        ),
        "subject_reports": subject_reports,
    }


def _partition_separation_days(enrollment_keys, probe_keys):
    """
    Return the whole days between the last enrollment and the first probe.

    ``None`` when either partition is empty or the recordings carry no usable
    date, so that a dataset without acquisition dates is reported as unknown
    rather than as zero separation.
    """
    if not (enrollment_keys and probe_keys):
        return None

    latest_enrollment = max(enrollment_keys)[0]
    earliest_probe = min(probe_keys)[0]

    if not (
        isinstance(latest_enrollment, datetime.date)
        and isinstance(earliest_probe, datetime.date)
    ):
        return None

    if datetime.date.min in (latest_enrollment, earliest_probe):
        return None

    return (earliest_probe - latest_enrollment).days


def summarize_temporal_separation(
    separations,
    enrollment_records=0,
    probe_records=0,
):
    """
    Describe how far apart in time a regime placed enrollment and probes.

    Some regimes separate recordings without separating days. Reporting the
    measured gap alongside the number of recordings on each side describes what
    a regime compared, which the name alone does not settle.

    Returns ``None`` when no subject supplied a usable pair of dates.
    """
    usable = [
        value for value in separations if value is not None
    ]

    if not usable:
        return None

    same_day = sum(1 for value in usable if value == 0)

    return {
        "subjects_measured": len(usable),
        "subjects_same_day": same_day,
        "subjects_different_day": len(usable) - same_day,
        "enrollment_records": enrollment_records,
        "probe_records": probe_records,
        "separates_days": same_day == 0,
        "min_days": min(usable),
        "median_days": float(statistics.median(usable)),
        "max_days": max(usable),
    }


# =============================================================================
# BEAT-MERGE WINDOWING
# =============================================================================
# Merging `num_beats_to_merge` consecutive beats into one sample slides the
# merge window one beat at a time by default, so neighbouring samples share
# beats. That is harmless when the neighbours stay inside a single partition,
# but it means a downstream random split can place two nearly identical
# samples on opposite sides of the train/test boundary. Configuring a stride
# equal to the merge width produces strictly non-overlapping windows.

def _normalize_beat_merge_stride(beat_merge_stride, num_beats_to_merge):
    """
    Validate the merge stride and return it as a positive integer.
    """
    if beat_merge_stride is None:
        return 1

    if isinstance(beat_merge_stride, bool):
        raise ValueError(
            "beat_merge_stride must be an integer, not a Boolean."
        )

    try:
        stride = int(beat_merge_stride)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "beat_merge_stride must be an integer."
        ) from error

    if stride != beat_merge_stride:
        raise ValueError(
            "beat_merge_stride must be a whole number."
        )

    if stride < 1:
        raise ValueError(
            "beat_merge_stride must be at least 1."
        )

    try:
        merge_width = int(num_beats_to_merge)
    except (TypeError, ValueError):
        merge_width = 1

    if merge_width >= 1 and stride > merge_width:
        raise ValueError(
            "beat_merge_stride cannot exceed num_beats_to_merge, "
            f"received stride {stride} for a merge width of "
            f"{merge_width}. A larger stride would silently discard "
            "beats between consecutive samples."
        )

    return stride


def _beat_merge_start_indices(beat_count, num_beats_to_merge, beat_merge_stride):
    """
    Return the start index of every merge window over a beat sequence.
    """
    if beat_count < num_beats_to_merge:
        return range(0)

    return range(
        0,
        beat_count - num_beats_to_merge + 1,
        beat_merge_stride,
    )


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def _download_and_extract(url: str, zip_path: Path, extract_to: Path, dataset_name: str, cleanup: bool = False):
    """
    Helper to download a file with a real-time progress bar and extract it.
    Includes auto-cleanup for corrupt files and handles nested ZIP/RAR archives.
    
    Args:
        url (str): Direct download link.
        zip_path (Path): Where to save the compressed file.
        extract_to (Path): Folder to extract contents into.
        dataset_name (str): For print logging.
        cleanup (bool): If True, deletes the zip/rar file after successful extraction.
    """
    extract_to.mkdir(parents=True, exist_ok=True)
    
    # 1. DOWNLOAD PHASE
    if not zip_path.exists():
        print(f"[INFO] Downloading {dataset_name}...")
        try:
            # Use browser headers to prevent 403 Forbidden errors from sites like Figshare
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            
            response = requests.get(url, stream=True, headers=headers)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            # Write file in chunks with progress bar
            with open(zip_path, "wb") as f, tqdm(
                desc=f"Downloading {dataset_name}", 
                total=total_size, 
                unit='iB', 
                unit_scale=True, 
                unit_divisor=1024
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)
                    
        except Exception as e:
            print(f"[ERR] Download failed: {e}")
            # Clean up partial file so we don't try to unzip a corrupt file later
            if zip_path.exists(): os.remove(zip_path)
            return

    # 2. EXTRACTION PHASE
    # Only extract if the target directory is empty
    if not any(extract_to.iterdir()):
        print(f"[INFO] Extracting {dataset_name}...")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Handle ZIP files
                if zip_path.suffix == ".zip":
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(temp_dir)
                        
                # Handle RAR files (requires 'patool' and system 7-Zip/Unrar)
                elif zip_path.suffix == ".rar":
                    patoolib.extract_archive(str(zip_path), outdir=temp_dir)
                
                # --- Intelligent Move Logic ---
                # Many archives wrap everything in a single top-level folder.
                # We detect this and move the *contents* up one level to keep paths clean.
                content = [d for d in os.listdir(temp_dir)]
                if len(content) == 1 and os.path.isdir(os.path.join(temp_dir, content[0])):
                    src = os.path.join(temp_dir, content[0])
                    for item in os.listdir(src):
                        shutil.move(os.path.join(src, item), extract_to)
                else:
                    # Move everything directly
                    for item in content:
                        shutil.move(os.path.join(temp_dir, item), extract_to)
                        
            print(f"[INFO] {dataset_name} ready.")
            
            # Optional cleanup
            if cleanup: 
                print(f"[INFO] Cleaning up zip file: {zip_path.name}")
                os.remove(zip_path)
                
        except Exception as e:
            print(f"[ERR] Extraction failed: {e}")
            print(f"[ACTION] Deleting corrupt file {zip_path.name} to force re-download on next run.")
            if zip_path.exists(): os.remove(zip_path)
            if extract_to.exists(): shutil.rmtree(extract_to)

# =============================================================================
# 1. ECG-ID
# =============================================================================
class load_ecgid_dataset():
    """
    Robust Loader for the ECG-ID Database.
    Handles automatic downloading, parsing via WFDB, and filtering.

    This dataset consists of 310 recordings from 90 subjects. Recordings vary
    in number per subject and are taken over different days. The database
    documentation gives the per-subject range as 2 to 20 records; the released
    files actually span 1 to 22, and 70 of the 90 subjects were recorded on a
    single day. The long-term regimes below therefore evaluate 20 subjects
    rather than 90.

    Args:
        num_beats_to_merge (int): Number of consecutive beats to fuse into a single sample. 
            Default is 1 (no fusion).
        beat_merge_method (str): Strategy for fusing beats if `num_beats_to_merge` > 1.
            Options: 
                - 'average': Averages the morphology of N beats.
                - 'concat': Flattens N beats into a single continuous vector.
        beat_merge_stride (int): Step between consecutive merge windows.
            The default of 1 slides one beat at a time, so merged samples
            share beats with their neighbours. Set it equal to
            `num_beats_to_merge` for strictly non-overlapping windows.
        data_split_mode (str): Evaluation regime mapping to strictly partition records.
            Options:
                - 'all-available': Loads every record (used for random beat-level splitting).
                - 'single-session': Loads ONLY the 1st record of each subject.
                - 'single-cross-session': 1st record = Train/Enroll, 2nd record = Test/Probe.
                  On ECG-ID the first two records of every eligible subject were
                  acquired on the same day, so this regime separates recordings
                  but not sessions. Use the long-term regimes for a genuine
                  across-day comparison.
                - 'single-shot-short-term': Day 1's 1st record = Enroll, rest of Day 1 = Probe.
                - 'leave-last-out-short-term': Day 1's last record = Probe, rest of Day 1 = Enroll.
                - 'single-shot-long-term': All Day 1 records = Enroll, all future days = Probe.
                - 'leave-last-out-long-term': Last recording day = Probe, all past days = Enroll.
        signal_type (str): Which WFDB channel to extract.
            Options:
                - 'raw': Extracts the unfiltered channel (idx 0).
                - 'filtered': Extracts the hardware-filtered channel (idx 1).
            Defaults to the value in ``config.yaml``, which is 'raw'. The
            filtered channel was produced by the database authors with an
            undocumented filter, so selecting it places an unspecified filter
            in series with the pipeline's own band-pass and gives ECG-ID an
            effective passband that no other dataset in the benchmark shares.
            Reading the raw channel keeps every dataset on the same
            preprocessing path.
        cleanup_zip (bool): If True, deletes the downloaded zip file after extraction.
        **preprocessing_params: kwargs passed directly to the Preprocessing class.
            Common options: mode='beat'|'blind', bandpass=True, normalize=True, window_len=5.0
    """
    def __init__(self, num_beats_to_merge=1, beat_merge_method="average",
                 beat_merge_stride=1,
                 data_split_mode="all-available", signal_type=None,
                 train_record_indices=None, enroll_record_indices=None,
                 probe_record_indices=None,
                 cleanup_zip=False, preprocessing_config=None,
                 **preprocessing_params):
        
        self.preprocessor = Preprocessing()
        self.cfg = CONFIG["datasets"]["ecgid"]
        project_dir = Path(__file__).resolve().parent
        self.data_root = (project_dir / CONFIG["project"]["data_root"]).resolve()
        self.dataset_root = self.data_root / self.cfg["root_dir"]
        self.zip_path = self.data_root / self.cfg["zip_name"]
        self.url = self.cfg["url"]
        
        # An explicit argument wins; otherwise the dataset entry in config.yaml
        # supplies the default, so editing that file changes which channel is
        # read. The resolved value is what reaches the cache identity, so a
        # change of default regenerates cached data rather than reusing it.
        if signal_type is None:
            signal_type = self.cfg.get("signal_type", "raw")

        if signal_type not in {"raw", "filtered"}:
            raise ValueError(
                "signal_type must be either 'raw' or 'filtered'."
            )

        self.signal_type = signal_type
        self.prep_params = _normalize_preprocessing_config(
            self.cfg.get("preprocessing", {}),
            preprocessing_config,
            preprocessing_params,
        )
        self.num_beats = num_beats_to_merge
        self.merge_strategy = beat_merge_method
        self.beat_merge_stride = _normalize_beat_merge_stride(
            beat_merge_stride,
            num_beats_to_merge,
        )
        self.cleanup_zip = cleanup_zip
        
        valid_modes = list(RECORD_BASED_SPLIT_MODES)
        if data_split_mode not in valid_modes:
            raise ValueError(f"Invalid mode: {data_split_mode}. Use {valid_modes}")
        self.data_split_mode = data_split_mode
        _configure_record_role_indices(
            self,
            data_split_mode,
            train_record_indices,
            enroll_record_indices,
            probe_record_indices,
        )
        # Recorded so that the temporal-date policy reaches the cache
        # identity: a cache built under a different policy describes a
        # different partition and must not be reused.
        self.temporal_date_policy = TEMPORAL_DATE_POLICY

    def download(self):
        _download_and_extract(self.url, self.zip_path, self.dataset_root, "ECG-ID", cleanup=self.cleanup_zip)

    def _extract_rec_number(self, filename):
        match = re.search(r'rec_(\d+)', filename)
        return int(match.group(1)) if match else 9999

    def _get_channel_index(self):
        """
        Return the WFDB channel corresponding to the selected signal type.

        Channel 0 is the raw signal and channel 1 is the
        hardware-filtered signal.
        """
        return 0 if self.signal_type == "raw" else 1

    def load_raw_data(self, metadata_only=False):
        """
        Scans the directory structure, reads WFDB headers, and parses recording dates.
        Returns a dictionary: { 'patient_id': [list of recording dicts sorted by time] }

        Args:
            metadata_only (bool): If True, skips signal reading and returns
                only the record identity and date of each recording. Date
                parsing and sorting are unchanged, so a metadata-only pass
                reports the same partition assignment the full pipeline
                produces.
        """
        if not self.dataset_root.exists() or not any(self.dataset_root.iterdir()):
            self.download()
        
        recordings = {}
        undated_records = []
        for subject_dir in tqdm(sorted(self.dataset_root.glob("Person*")), desc="Loading ECG-ID raw files"):
            sid = subject_dir.name.replace("Person_", "")
            recs = []
            hea_files = sorted(subject_dir.glob("*.hea"))
            
            for hea_path in hea_files:
                try:
                    record = (
                        wfdb.rdheader(str(hea_path.with_suffix("")))
                        if metadata_only
                        else wfdb.rdrecord(str(hea_path.with_suffix("")))
                    )
                    rec_date = record.base_date
                    
                    # Robust date parsing from comments if base_date is missing
                    if rec_date is None:
                        for comment in record.comments:
                            if "date" in comment.lower():
                                try: 
                                    date_str = comment.split(":")[-1].strip()
                                    rec_date = datetime.datetime.strptime(date_str, "%d.%m.%Y").date()
                                except: pass
                    # Every released ECG-ID header carries a date, so this stays
                    # unset in practice. Should one ever arrive without a date,
                    # it is kept as unknown rather than assigned a placeholder,
                    # which would otherwise anchor the first day of the
                    # record-order regimes.
                    if rec_date is None:
                        undated_records.append(hea_path.name)

                    verify_sampling_rate(
                        self.cfg.get("fs"),
                        "ECG-ID",
                        hea_path.name,
                        record.fs,
                    )

                    if metadata_only:
                        recs.append({
                            'date': rec_date,
                            'fs': record.fs,
                            'filename': hea_path.name
                        })
                        continue

                    channel_idx = self._get_channel_index()

                    # Safety check just in case a file has only 1 channel
                    if record.p_signal.shape[1] <= channel_idx:
                        channel_idx = 0

                    recs.append({
                        'signal': record.p_signal[:, channel_idx],
                        'date': rec_date,
                        'fs': record.fs,
                        'filename': hea_path.name
                    })
                except ValueError:
                    # Raised deliberately when a recording contradicts the
                    # dataset configuration. That is a defect in the data or the
                    # configuration, not an unreadable file, so it must stop the
                    # run rather than reduce the cohort.
                    raise
                except Exception as read_error:
                    # A record that cannot be read shrinks the evaluated cohort,
                    # so report it rather than letting the run continue on a
                    # silently reduced set of recordings.
                    print(
                        f"[WARN] ECG-ID: skipping {hea_path.name} for subject "
                        f"{sid}: {read_error}"
                    )

            # Dated recordings lead, in acquisition order. Any undated one
            # follows in record-number order, so enumeration stays reproducible
            # without implying where it belongs in time.
            recs.sort(
                key=lambda x: (
                    x['date'] is None,
                    x['date'] if x['date'] is not None else datetime.date.min,
                    self._extract_rec_number(x['filename']),
                )
            )
            recordings[sid] = recs

        if undated_records:
            print(
                f"[INFO] ECG-ID: {len(undated_records)} recording(s) state no "
                "acquisition date. They stay available to 'all-available' and "
                "take no part in the record-order regimes: "
                + ", ".join(sorted(undated_records))
            )

        return recordings

    def _process_signal(self, sig, fs):
        beats = _preprocess_signal(
                self.preprocessor,
                sig,
                fs,
                self.prep_params,
            )
        
        if self.num_beats == 1: return beats
        if len(beats) < self.num_beats: return np.empty((0, beats.shape[1])) 
        
        processed_samples = []
        merge_starts = _beat_merge_start_indices(
            len(beats),
            self.num_beats,
            self.beat_merge_stride,
        )
        for i in merge_starts:
            group = beats[i : i + self.num_beats] 
            if self.merge_strategy == "average":
                processed_samples.append(np.mean(group, axis=0))
            elif self.merge_strategy == "concat":
                processed_samples.append(group.flatten())
        return np.array(processed_samples)

    def load_all_data(self, return_provenance=False):
        """
        Loads dataset for tasks that handle train/test splitting downstream.
        Applies to 'all-available' and 'single-session'.
        """
        if self.data_split_mode not in ["all-available", "single-session"]:
            print(f"[WARN] Calling load_all_data() but mode is '{self.data_split_mode}'.")
            
        data = self.load_raw_data()
        x_list, y_list = [], []
        provenance_builder = _ProvenanceBuilder() if return_provenance else None
        
        for sid, recs in tqdm(data.items(), desc="Processing signals"):
            if not recs: continue
            
            target_recs = recs if self.data_split_mode == "all-available" else [recs[0]]
            
            for acquisition_order, rec in enumerate(target_recs):
                segments = self._process_signal(rec['signal'], rec['fs'])
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
                    if provenance_builder is not None:
                        provenance_builder.add_block(
                            len(segments),
                            record_id=rec['filename'],
                            session_id=rec['filename'],
                            acquisition_time=rec['date'],
                            acquisition_order=acquisition_order,
                            source_segment_id=rec['filename'],
                            source_segment_order=float(acquisition_order),
                        )
                    
        return _finalize_loader_output(x_list, y_list, provenance_builder, return_provenance)

    def load_session(self, session_name, return_provenance=False):
        """
        Loads the partitioned data strictly based on temporal/record boundaries.
        Applies to cross-session and short/long-term tasks.
        """
        role = _canonical_record_role(session_name)
        is_enrollment = role != "probe"
        log_role = (
            role
            if self.data_split_mode == CUSTOM_RECORD_SPLIT_MODE
            else is_enrollment
        )
        log_name = {
            "train": "Training",
            "enrollment": "Enrollment",
            "probe": "Test/Probe",
        }[role]

        data = self.load_raw_data()
        x_list, y_list = [], []
        provenance_builder = _ProvenanceBuilder() if return_provenance else None
        
        kept_subjects, dropped_subjects = 0, 0

        for sid, recs in tqdm(data.items(), desc=f"Processing {log_name}"):
            if not recs: continue
            
            (
                target_recs,
                subject_is_eligible,
            ) = select_record_role_partition(
                recs,
                self.data_split_mode,
                role,
                train_record_indices=getattr(
                    self,
                    "train_record_indices",
                    None,
                ),
                enroll_record_indices=getattr(
                    self,
                    "enroll_record_indices",
                    None,
                ),
                probe_record_indices=getattr(
                    self,
                    "probe_record_indices",
                    None,
                ),
            )

            if not subject_is_eligible:
                dropped_subjects += 1
                continue

            kept_subjects += 1

            _record_partition_assignment(
                self,
                log_role,
                sid,
                target_recs,
            )

            # --- EXTRACTION & SPLITTING ---
            for acquisition_order, rec in enumerate(target_recs):
                segments = self._process_signal(rec['signal'], rec['fs'])
                
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
                    if provenance_builder is not None:
                        provenance_builder.add_block(
                            len(segments),
                            record_id=rec['filename'],
                            session_id=rec['filename'],
                            acquisition_time=rec['date'],
                            acquisition_order=acquisition_order,
                            source_segment_id=rec['filename'],
                            source_segment_order=float(acquisition_order),
                        )

        # Dynamic summary print for all structured tasks during enrollment
        if self.data_split_mode not in ["all-available", "single-session"] and is_enrollment:
            mode_title = self.data_split_mode.replace('-', ' ').title()
            print(f"\n[INFO] {mode_title} Summary: Kept {kept_subjects} subjects. Dropped {dropped_subjects} subjects.")

        return _finalize_loader_output(x_list, y_list, provenance_builder, return_provenance)

# =============================================================================
# 2. HeartPrint
# =============================================================================
class load_heartprint_dataset():
    """
    Dynamic Loader for the HeartPrint Dataset.
    
    HeartPrint is highly structured around distinct physiological and temporal sessions.
    This loader enforces strict mathematical intersection—a subject is only kept if they 
    possess valid data in ALL requested target sessions.

    Session Tags Available for Mapping:
      - 'session1'  (Baseline / Rest)
      - 'session2'  (Rest / Short-Term follow up)
      - 'session3r' (Reading Task / Cognitive State Change)
      - 'session3l' (Very Long-Term / Maximal Time Gap)

    Args:
        data_split_mode (str): The routing logic for data extraction.
            Options:
                - 'single-session': Extracts the exact sessions defined in `session_for_single_session_evaluation` 
                                    and pools them for downstream random-splitting.
                - 'cross-session': Maps data strictly to Train/Enroll/Probe groups based on the session arguments below.
        session_for_single_session_evaluation (str or list): Target session(s) to load if mode is 'single-session'.
            Example: 'session1' or ['session1', 'session2']
        train_sessions (str or list): Session(s) to load when requesting representation learning data. Can be None.
            Example: 'session1'
        enroll_sessions (str or list): Session(s) to load when requesting Gallery enrollment data. Can be None.
            Example: 'session1'
        probe_sessions (str or list): Session(s) to load when requesting Test query data. Can be None.
            Example: 'session3l'
        num_beats_to_merge (int): Number of consecutive beats to fuse into a single sample.
        beat_merge_method (str): Strategy for fusing beats. Options: ['average', 'concat']
        beat_merge_stride (int): Step between consecutive merge windows.
            The default of 1 slides one beat at a time, so merged samples
            share beats with their neighbours. Set it equal to
            `num_beats_to_merge` for strictly non-overlapping windows.
        cleanup_zip (bool): If True, deletes the downloaded zip file after extraction.
        **preprocessing_params: kwargs passed directly to the Preprocessing class.
    """
    # Sessions are read in this order so that enumeration, and the choice of
    # which copy of a repeated recording to keep, do not depend on the
    # filesystem.
    SESSION_ORDER = (
        "session1",
        "session2",
        "session3r",
        "session3l",
    )

    # The visit each session belongs to. S3R holds the recordings made in the
    # reading condition and S3L those separated from the first session by a
    # long interval, but both come from the third visit, so one recording can
    # legitimately appear under both tags.
    SESSION_VISIT = {
        "session1": "visit1",
        "session2": "visit2",
        "session3r": "visit3",
        "session3l": "visit3",
    }

    # Pairs that describe one visit and therefore share recordings. Using one
    # to enrol and the other to probe would compare recordings against
    # themselves for 45 of the 78 subjects present in both.
    SAME_VISIT_SESSION_PAIRS = (
        ("session3r", "session3l"),
    )

    def __init__(self, data_split_mode="cross-session",
                 session_for_single_session_evaluation=["session1"],
                 train_sessions=["session1"],
                 enroll_sessions=None,
                 probe_sessions=["session2"],
                 num_beats_to_merge=1, beat_merge_method="average",
                 beat_merge_stride=1,
                 cleanup_zip=False, preprocessing_config=None,
                 **preprocessing_params):
       
        self.preprocessor = Preprocessing()
        self.cfg = CONFIG["datasets"]["heartprint"]
        project_dir = Path(__file__).resolve().parent
        self.data_root = (project_dir / CONFIG["project"]["data_root"]).resolve()
        self.dataset_root = self.data_root / self.cfg["root_dir"]
        self.zip_path = self.data_root / self.cfg["zip_name"]
        self.url = self.cfg["url"]
        
        self.sample_len = self.cfg.get("sample_length", 3747)
        self.fs = int(self.cfg.get("fs", 250))
        self.prep_params = _normalize_preprocessing_config(
            self.cfg.get("preprocessing", {}),
            preprocessing_config,
            preprocessing_params,
        )
        self.num_beats = num_beats_to_merge
        self.merge_strategy = beat_merge_method
        self.beat_merge_stride = _normalize_beat_merge_stride(
            beat_merge_stride,
            num_beats_to_merge,
        )
        self.cleanup_zip = cleanup_zip

        valid_modes = ["single-session", "cross-session"]
        if data_split_mode not in valid_modes:
            raise ValueError(f"Invalid mode: {data_split_mode}. Use {valid_modes}")
        self.data_split_mode = data_split_mode
        
        # Safely convert strings to lists and handle None
        to_list = lambda x: [x] if isinstance(x, str) else (x if x else [])
        
        self.session_for_single_session_evaluation = to_list(session_for_single_session_evaluation)
        self.train_sessions = to_list(train_sessions)
        self.enroll_sessions = (
            list(self.train_sessions)
            if enroll_sessions is None
            else to_list(enroll_sessions)
        )
        self.probe_sessions = to_list(probe_sessions)
        
        # Enforce exact naming to match folder architectures
        self._normalize_sessions(self.session_for_single_session_evaluation)
        self._normalize_sessions(self.train_sessions)
        self._normalize_sessions(self.enroll_sessions)
        self._normalize_sessions(self.probe_sessions)
        
        self.required_cross_sessions = list(set(self.train_sessions + self.enroll_sessions + self.probe_sessions))

        self._reject_same_visit_pairing()
        
        if self.data_split_mode == "single-session" and not self.session_for_single_session_evaluation:
            raise ValueError("You must provide `session_for_single_session_evaluation`.")
        if self.data_split_mode == "cross-session" and not self.required_cross_sessions:
            raise ValueError("You must provide at least one valid train, enroll, or probe session.")

    def _normalize_sessions(self, session_list):
        """Standardizes session strings to match directory keys natively."""
        for i in range(len(session_list)):
            s = session_list[i].lower().replace("-", "").replace("_", "").replace(" ", "")
            session_list[i] = s

    def download(self):
        """Attempts robust download via Figshare API."""
        if self.dataset_root.exists():
            for root, dirs, files in os.walk(self.dataset_root):
                for d in dirs:
                    if "session" in d.lower(): return 

        self.dataset_root.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Attempting to download HeartPrint...")
        
        try:
            if not self.zip_path.exists():
                match = re.search(r'articles/(\d+)/versions/(\d+)', self.url)
                if match:
                    aid, ver = match.groups()
                    api = f"https://api.figshare.com/v2/articles/{aid}/versions/{ver}"
                    r = requests.get(api); r.raise_for_status()
                    target = next((f for f in r.json()['files'] if 'zip' in f['name'] or 'rar' in f['name']), None)
                    if not target: raise ValueError("Archive not found")
                    dl_url = target['download_url']
                    size = target['size']
                else:
                    dl_url = self.url; size = 0

                with requests.get(dl_url, stream=True) as r:
                    r.raise_for_status()
                    if size == 0: size = int(r.headers.get('content-length', 0))
                    with open(self.zip_path, "wb") as f, tqdm(desc="Downloading", total=size, unit='iB', unit_scale=True) as bar:
                        for chunk in r.iter_content(8192): f.write(chunk); bar.update(len(chunk))

            print(f"[INFO] Attempting extraction...")
            with tempfile.TemporaryDirectory() as temp_dir:
                try: patoolib.extract_archive(str(self.zip_path), outdir=temp_dir)
                except Exception:
                    with zipfile.ZipFile(self.zip_path, "r") as zf: zf.extractall(temp_dir)
                for item in os.listdir(temp_dir): shutil.move(os.path.join(temp_dir, item), self.dataset_root)
            if self.cleanup_zip: os.remove(self.zip_path)

        except Exception as e:
            print(f"[WARN] Automated download failed: {e}")
            print("Please download manually and extract to:", self.dataset_root)

    def _reject_same_visit_pairing(self):
        """
        Refuse a protocol that enrols and probes on two tags of one visit.

        S3R and S3L label the reading condition and the long interval of the
        same third visit, so 135 recordings appear under both. Using one to
        enrol and the other to probe compares recordings against themselves for
        45 of the 78 subjects present in both, which reports a similarity that
        no separation produced.
        """
        enrolment = set(self.train_sessions) | set(self.enroll_sessions)
        probes = set(self.probe_sessions)

        for first, second in self.SAME_VISIT_SESSION_PAIRS:
            crosses = (
                (first in enrolment and second in probes)
                or (second in enrolment and first in probes)
            )

            if crosses:
                raise ValueError(
                    f"HeartPrint: '{first}' and '{second}' are two labels on "
                    "the same visit and share recordings, so one cannot enrol "
                    "while the other probes. Probe against 'session1' or "
                    "'session2' for a comparison across visits."
                )

    def load_raw_data(self):
        """
        Scans the HeartPrint directory and maps valid text files to their explicit Session and Subject ID.
        """
        if not self.dataset_root.exists() or not any(self.dataset_root.iterdir()):
            self.download()

        print("\n[INFO] Scanning directories and pooling HeartPrint files...")
        session_dirs = {}

        # 1. Identify the core session folders safely. Directories are walked in
        # sorted order so that the same tree always yields the same folder for
        # a given session tag.
        for root, dirs, files in os.walk(self.dataset_root):
            dirs.sort()
            for d in dirs:
                d_norm = d.lower().replace("-", "").replace("_", "").replace(" ", "")
                if "session1" in d_norm: session_dirs["session1"] = Path(root) / d
                elif "session2" in d_norm: session_dirs["session2"] = Path(root) / d
                elif "session3r" in d_norm or ("session3" in d_norm and "r" in d_norm): session_dirs["session3r"] = Path(root) / d
                elif "session3l" in d_norm or ("session3" in d_norm and "l" in d_norm): session_dirs["session3l"] = Path(root) / d

        recordings = {}

        # Recordings already seen for a subject, keyed by content, together
        # with the visit they were first seen in. Session-2 holds a
        # byte-identical copy of the Session-1 recordings of two subjects who
        # attended only once, which would otherwise make an enrolment and its
        # probe the same signal. A recording shared between session3r and
        # session3l is not the same case: those are two labels on one visit, so
        # the same recording belongs under both and is kept.
        seen_by_subject = {}
        duplicate_count = 0

        # 2. Extract records, ensuring we skip hidden macOS files
        for session_tag in self.SESSION_ORDER:
            sess_path = session_dirs.get(session_tag)
            if sess_path is None:
                continue

            subject_ids = sorted(
                entry.name
                for entry in sess_path.iterdir()
                if entry.is_dir()
            )

            for sid in tqdm(subject_ids, desc=f"Loading {session_tag.upper()} raw files"):
                subj_path = sess_path / sid

                if sid not in recordings: recordings[sid] = {}
                if session_tag not in recordings[sid]: recordings[sid][session_tag] = []

                for fpath in sorted(subj_path.glob("*.txt")):
                    if fpath.name.startswith("._"): continue

                    try:
                        digest = hashlib.sha256(
                            fpath.read_bytes()
                        ).hexdigest()

                        visit = self.SESSION_VISIT[session_tag]
                        first_visit = seen_by_subject.setdefault(
                            sid, {}
                        ).get(digest)

                        if first_visit is not None and first_visit != visit:
                            duplicate_count += 1
                            continue

                        seen_by_subject[sid].setdefault(digest, visit)

                        # High-Speed parsing: Bypasses headers and stops at the sample length
                        df = pd.read_csv(fpath, comment='#', delim_whitespace=True, header=None, nrows=self.sample_len, on_bad_lines='skip')
                        if not df.empty:
                            sig = df.iloc[:, 0].dropna().values.astype(float)
                            sig = sig - np.mean(sig) # Zero-mean baseline
                            recordings[sid][session_tag].append({
                                'signal': sig,
                                'fs': self.fs,
                                'filename': fpath.name,
                            })
                    except Exception as e:
                        print(f"\n[WARN] Failed to read {fpath.name}: {e}")

        if duplicate_count:
            print(
                f"[INFO] HeartPrint: skipped {duplicate_count} recording(s) "
                "that repeat an earlier session of the same subject."
            )

        return recordings

    def _process_signal(self, sig, fs=250):
        """Applies filters, segmentation, and multi-beat merging."""
        if np.isnan(sig).any() or len(sig) < fs or np.std(sig) < 1e-5: 
            return np.empty((0, 0))

        beats = _preprocess_signal(
                self.preprocessor,
                sig,
                fs,
                self.prep_params,
            )
        if self.num_beats == 1: return beats
        if len(beats) < self.num_beats: return np.empty((0, beats.shape[1]))
        
        processed_samples = []
        merge_starts = _beat_merge_start_indices(
            len(beats),
            self.num_beats,
            self.beat_merge_stride,
        )
        for i in merge_starts:
            group = beats[i : i + self.num_beats]
            if self.merge_strategy == "average": processed_samples.append(np.mean(group, axis=0))
            elif self.merge_strategy == "concat": processed_samples.append(group.flatten())
        return np.array(processed_samples)

    def load_all_data(self, return_provenance=False):
        """Safely routes generic all-data requests to the single-session logic."""
        if self.data_split_mode != "single-session":
            print(f"[WARN] Calling load_all_data() but mode is '{self.data_split_mode}'.")
        return self.load_session("train", return_provenance=return_provenance)

    def load_session(self, session_name, return_provenance=False):
        """
        Extracts requested sessions.
        In 'cross-session' mode, enforces strict mathematical intersection, ensuring a 
        subject is present across ALL specified train, enroll, and probe configurations.
        """
        session_name = session_name.lower()
        target_sessions = []
        is_primary_pass = False
        
        if self.data_split_mode == "single-session":
            if session_name in ["probe", "test"]:
                raise ValueError("Cannot load 'test' in single-session mode. Split upstream.")
            target_sessions = self.session_for_single_session_evaluation
            log_name = f"Single-Session Target(s): {target_sessions}"
            is_primary_pass = True
            
        elif self.data_split_mode == "cross-session":
            if session_name in ["train"]:
                target_sessions = self.train_sessions
                is_primary_pass = True
            elif session_name in ["enrol", "enrollment"]:
                target_sessions = self.enroll_sessions
                is_primary_pass = True if not self.train_sessions else False
            elif session_name in ["probe", "test"]:
                target_sessions = self.probe_sessions
            else:
                raise ValueError("session_name must be 'train', 'enrol', or 'test'.")
            log_name = f"Cross-Session ({session_name.title()}): {target_sessions}"

        if not target_sessions:
            return _finalize_loader_output([], [], None, return_provenance)
            
        data = self.load_raw_data()
        x_list, y_list = [], []
        provenance_builder = _ProvenanceBuilder() if return_provenance else None
        
        kept_subjects, dropped_subjects = 0, 0

        for sid, tagged_sessions in tqdm(data.items(), desc=f"Processing {log_name}"):
            
            # --- STRICT INTERSECTION LOGIC ---
            if self.data_split_mode == "single-session":
                is_valid = all(s in tagged_sessions for s in target_sessions)
            elif self.data_split_mode == "cross-session":
                # Strict: Subject MUST have data in ALL requested global sets
                is_valid = all(s in tagged_sessions for s in self.required_cross_sessions)

            if is_valid:
                kept_subjects += 1
                record_order = 0
                for s in target_sessions:
                    # HeartPrint frequently has 2 to 6 files per session. This naturally pools them!
                    for signal_dict in tagged_sessions[s]:
                        segments = self._process_signal(signal_dict['signal'], signal_dict['fs'])
                        if len(segments) > 0:
                            x_list.append(segments)
                            y_list.extend([sid] * len(segments))
                            if provenance_builder is not None:
                                provenance_builder.add_block(
                                    len(segments),
                                    record_id=signal_dict['filename'],
                                    session_id=s,
                                    acquisition_time=None,
                                    acquisition_order=record_order,
                                    source_segment_id=signal_dict['filename'],
                                    source_segment_order=float(record_order),
                                )
                        record_order += 1
            else:
                dropped_subjects += 1

        if is_primary_pass:
            print(f"\n[INFO] HeartPrint Evaluation Summary ({self.data_split_mode.title()}):")
            print(f"       Kept {kept_subjects} mathematically matched subjects. Dropped {dropped_subjects} subjects due to missing session data.")

        return _finalize_loader_output(x_list, y_list, provenance_builder, return_provenance)

# =============================================================================
# 3. PTB (Physikalisch-Technische Bundesanstalt)
# =============================================================================
class load_ptb_dataset():
    """
    Robust Loader for the PTB Diagnostic ECG Database.
    Handles multi-lead parsing, clinical header filtering, and chronologically mapped biometric tasks.

    This dataset contains 549 records from 290 subjects. Many subjects have 
    severe clinical pathologies (e.g., Myocardial Infarction). 

    Args:
        leads (list of str): Target ECG leads to extract. 
            Options: Any valid 12-lead string (e.g., ['i', 'v5', 'ii']) or 'all' for all 15 available channels.
            Default: ['i']
        data_split_mode (str): Evaluation regime mapping.
            Options:
                - 'all-available': Loads every record (used for random beat-level splitting).
                - 'single-session': Loads ONLY the 1st record of each subject.
                - 'single-cross-session': 1st record = Train/Enroll, 2nd record = Test/Probe.
                - 'single-shot-short-term': Day 1's 1st record = Enroll, rest of Day 1 = Probe.
                - 'leave-last-out-short-term': Day 1's last record = Probe, rest of Day 1 = Enroll.
                - 'single-shot-long-term': All Day 1 records = Enroll, all future days = Probe.
                - 'leave-last-out-long-term': Last recording day = Probe, all past days = Enroll.
        only_healthy (bool): If True, strictly drops subjects with clinical pathologies, 
                             keeping only the ~52 healthy control volunteers.
        num_beats_to_merge (int): Number of consecutive beats to fuse into a single sample.
        beat_merge_method (str): Strategy for fusing beats. Options: ['average', 'concat']
        beat_merge_stride (int): Step between consecutive merge windows.
            The default of 1 slides one beat at a time, so merged samples
            share beats with their neighbours. Set it equal to
            `num_beats_to_merge` for strictly non-overlapping windows.
        cleanup_zip (bool): If True, deletes the downloaded zip file after extraction.
        **preprocessing_params: kwargs passed directly to the Preprocessing class.
    """
    def __init__(self, leads=['i'], data_split_mode="all-available",
                 only_healthy=False, num_beats_to_merge=1, beat_merge_method="average",
                 beat_merge_stride=1,
                 train_record_indices=None, enroll_record_indices=None,
                 probe_record_indices=None,
                 cleanup_zip=False, preprocessing_config=None,
                 **preprocessing_params):
       
        self.preprocessor = Preprocessing()
        self.cfg = CONFIG["datasets"]["ptb"]
        project_dir = Path(__file__).resolve().parent
        self.data_root = (project_dir / CONFIG["project"]["data_root"]).resolve()
        self.dataset_root = self.data_root / self.cfg["root_dir"]
        self.zip_path = self.data_root / self.cfg["zip_name"]
        self.url = self.cfg["url"]
        
        self.prep_params = _normalize_preprocessing_config(
            self.cfg.get("preprocessing", {}),
            preprocessing_config,
            preprocessing_params,
        )
        self.target_leads = [l.lower() for l in leads] if isinstance(leads, list) else leads
        self.only_healthy = only_healthy
        self.num_beats = num_beats_to_merge
        self.merge_strategy = beat_merge_method
        self.beat_merge_stride = _normalize_beat_merge_stride(
            beat_merge_stride,
            num_beats_to_merge,
        )
        self.cleanup_zip = cleanup_zip
        
        valid_modes = list(RECORD_BASED_SPLIT_MODES)
        if data_split_mode not in valid_modes:
            raise ValueError(f"Invalid mode: {data_split_mode}. Use {valid_modes}")
        self.data_split_mode = data_split_mode
        _configure_record_role_indices(
            self,
            data_split_mode,
            train_record_indices,
            enroll_record_indices,
            probe_record_indices,
        )
        # Recorded so that the temporal-date policy reaches the cache
        # identity: a cache built under a different policy describes a
        # different partition and must not be reused.
        self.temporal_date_policy = TEMPORAL_DATE_POLICY

    def download(self):
        """Downloads and extracts the dataset if not already present."""
        _download_and_extract(self.url, self.zip_path, self.dataset_root, "PTB", cleanup=self.cleanup_zip)

    def _is_healthy(self, record):
        """Robust check for healthy status using multiple header fields."""
        healthy_keywords = ["healthy", "control", "volunteer", "donor", "normal"]
        target_fields = ["reason for admission", "clinical classification", "diagnose"]
        for comment in record.comments:
            c_lower = comment.lower()
            if any(field in c_lower for field in target_fields):
                if any(kw in c_lower for kw in healthy_keywords): return True
        return False

    def _parse_date_from_comments(self, comments):
        """Extracts date from PTB comments like '# ECG date: 05/06/1997'."""
        for comment in comments:
            if "ecg date" in comment.lower():
                try:
                    date_str = comment.split(":")[-1].strip()
                    for fmt in ["%d/%m/%Y", "%d-%b-%y", "%d.%m.%Y"]:
                        try: return datetime.datetime.strptime(date_str, fmt).date()
                        except ValueError: continue
                except: pass
        return None

    def _get_lead_indices(self, available_leads):
        """Maps requested lead names (e.g., 'i', 'v5') to channel indices."""
        avail_norm = [l.lower().replace("ecg", "").strip() for l in available_leads]
        if self.target_leads == 'all': return list(range(len(available_leads)))
        indices = []
        for req in self.target_leads:
            req = req.strip().lower()
            try: indices.append(avail_norm.index(req))
            except ValueError: pass 
        return indices

    def load_raw_data(self, metadata_only=False):
        """
        Loads all WFDB records, parses metadata, and sorts chronologically.
        A record whose header states no acquisition date keeps ``date`` as
        ``None`` and is ordered after the dated ones by record name.

        Args:
            metadata_only (bool): If True, skips signal reading and returns
                only the record identity and date of each recording. Date
                parsing and sorting are unchanged, so a metadata-only pass
                reports the same partition assignment the full pipeline
                produces.
        """
        if not self.dataset_root.exists() or not any(self.dataset_root.iterdir()):
            self.download()
        
        recordings = {}
        files = list(self.dataset_root.rglob("*.hea"))
        
        # Group files by patient folder (e.g., patient001)
        patient_groups = {}
        for f in files:
            pid = f.parent.name
            if pid not in patient_groups: patient_groups[pid] = []
            patient_groups[pid].append(f)

        # 541 of the 549 records state an acquisition date in the header; the
        # remaining eight state none. An unknown acquisition time is kept as
        # unknown rather than filled in, so that it can never stand in as
        # evidence of elapsed time. Undated recordings remain available to
        # "all-available", which does not order recordings, and take no part in
        # the record-order regimes, which do.
        #
        # Seven of the eight belong to patients whose only recording is that
        # record, so every regime requiring two recordings already drops them.
        # The eighth is patient180/s0561_re, and that patient keeps six dated
        # recordings across four distinct days, so the subject stays eligible
        # throughout; only the undated recording sits out the ordering.
        dated_count = 0
        undated_records = []

        for sid, p_files in tqdm(sorted(patient_groups.items()), desc="Loading PTB raw files"):
            recs = []
            p_files = sorted(p_files) 

            if self.only_healthy:
                try:
                    first_header = wfdb.rdheader(str(p_files[0].with_suffix("")))
                    if not self._is_healthy(first_header): continue 
                except: continue

            for hea in p_files:
                try:
                    rec_header = wfdb.rdheader(str(hea.with_suffix("")))

                    verify_sampling_rate(
                        self.cfg.get("fs"),
                        "PTB",
                        hea.name,
                        rec_header.fs,
                    )

                    lead_indices = self._get_lead_indices(rec_header.sig_name)
                    if not lead_indices: continue

                    data = None
                    if not metadata_only:
                        data, _ = wfdb.rdsamp(str(hea.with_suffix("")), channels=lead_indices)

                    dt = rec_header.base_date
                    if dt is None: dt = self._parse_date_from_comments(rec_header.comments)

                    if dt is None:
                        full_dt = None
                        undated_records.append(f"{sid}/{hea.stem}")
                    else:
                        full_dt = datetime.datetime.combine(dt, datetime.time.min)
                        dated_count += 1

                    record_entry = {
                        "fs": rec_header.fs,
                        "date": full_dt,
                        "filename": hea.name,
                    }

                    if not metadata_only:
                        record_entry["signal"] = data

                    recs.append(record_entry)
                except ValueError:
                    # Raised deliberately when a recording contradicts the
                    # dataset configuration, which must stop the run rather
                    # than quietly reduce the cohort.
                    raise
                except Exception as read_error:
                    # A record that cannot be read shrinks the evaluated
                    # cohort, so report it rather than dropping it in silence.
                    print(
                        f"[WARN] PTB: skipping {hea.name} for {sid}: "
                        f"{read_error}"
                    )

            if recs:
                # Dated recordings lead, in acquisition order. Undated ones
                # follow in record-name order, which keeps enumeration
                # reproducible without implying where they belong in time.
                recs.sort(
                    key=lambda x: (
                        x["date"] is None,
                        x["date"] if x["date"] is not None else datetime.datetime.min,
                        x["filename"],
                    )
                )
                recordings[sid] = recs

        if undated_records:
            print(
                f"[INFO] PTB: {dated_count} recording(s) state an acquisition "
                f"date and {len(undated_records)} do not. The undated "
                "recordings stay available to 'all-available' and take no part "
                "in the record-order regimes: "
                + ", ".join(sorted(undated_records))
            )

        return recordings

    def _process_signal(self, sig, fs):
        """Applies filters, segmentation, and multi-beat merging."""
        n_channels = sig.shape[1]
        processed_channels = []
        for c in range(n_channels):
            processed_channels.append(_preprocess_signal(
                self.preprocessor,
                sig[:, c],
                fs,
                self.prep_params,
            ))
        if not processed_channels: return np.empty((0, n_channels, 0))
        min_len = min([len(ch) for ch in processed_channels])
        if min_len == 0: return np.empty((0, n_channels, 0))
        
        beats_multi = np.stack([ch[:min_len] for ch in processed_channels], axis=1)
        if self.num_beats == 1:
            return beats_multi[:, 0, :] if n_channels == 1 else beats_multi
        if len(beats_multi) < self.num_beats: return np.empty((0, n_channels, 0))
        
        merged_samples = []
        merge_starts = _beat_merge_start_indices(
            len(beats_multi),
            self.num_beats,
            self.beat_merge_stride,
        )
        for i in merge_starts:
            group = beats_multi[i : i + self.num_beats]
            if self.merge_strategy == "average":
                merged = np.mean(group, axis=0)
                if n_channels == 1: merged = merged.squeeze(0)
                merged_samples.append(merged)
            elif self.merge_strategy == "concat":
                merged = group.transpose(1, 0, 2).reshape(n_channels, -1)
                if n_channels == 1: merged = merged.squeeze(0)
                merged_samples.append(merged)
        return np.array(merged_samples)

    def load_all_data(self, return_provenance=False):
        """
        Loads dataset for tasks that handle train/test splitting downstream.
        Applies to 'all-available' and 'single-session'.
        """
        if self.data_split_mode not in ["all-available", "single-session"]:
            print(f"[WARN] Calling load_all_data() but mode is '{self.data_split_mode}'.")
            
        data = self.load_raw_data()
        x_list, y_list = [], []
        provenance_builder = _ProvenanceBuilder() if return_provenance else None
        
        for sid, recs in tqdm(data.items(), desc="Processing signals"):
            if not recs: continue
            
            target_recs = recs if self.data_split_mode == "all-available" else [recs[0]]
            
            for acquisition_order, rec in enumerate(target_recs):
                segments = self._process_signal(rec['signal'], rec['fs'])
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
                    if provenance_builder is not None:
                        provenance_builder.add_block(
                            len(segments),
                            record_id=rec['filename'],
                            session_id=rec['filename'],
                            acquisition_time=rec['date'],
                            acquisition_order=acquisition_order,
                            source_segment_id=rec['filename'],
                            source_segment_order=float(acquisition_order),
                        )
                    
        return _finalize_loader_output(x_list, y_list, provenance_builder, return_provenance)

    def load_session(self, session_name, return_provenance=False):
        """
        Loads the partitioned data strictly based on temporal/record boundaries.
        Applies to cross-session and short/long-term tasks.
        """
        role = _canonical_record_role(session_name)
        is_enrollment = role != "probe"
        log_role = (
            role
            if self.data_split_mode == CUSTOM_RECORD_SPLIT_MODE
            else is_enrollment
        )
        log_name = {
            "train": "Training",
            "enrollment": "Enrollment",
            "probe": "Test/Probe",
        }[role]

        data = self.load_raw_data()
        x_list, y_list = [], []
        provenance_builder = _ProvenanceBuilder() if return_provenance else None
        
        kept_subjects, dropped_subjects = 0, 0

        for sid, recs in tqdm(data.items(), desc=f"Processing {log_name}"):
            if not recs: continue
            
            (
                target_recs,
                subject_is_eligible,
            ) = select_record_role_partition(
                recs,
                self.data_split_mode,
                role,
                train_record_indices=getattr(
                    self,
                    "train_record_indices",
                    None,
                ),
                enroll_record_indices=getattr(
                    self,
                    "enroll_record_indices",
                    None,
                ),
                probe_record_indices=getattr(
                    self,
                    "probe_record_indices",
                    None,
                ),
            )

            if not subject_is_eligible:
                dropped_subjects += 1
                continue

            kept_subjects += 1

            _record_partition_assignment(
                self,
                log_role,
                sid,
                target_recs,
            )

            # --- EXTRACTION & SIGNAL PROCESSING ---
            for acquisition_order, rec in enumerate(target_recs):
                segments = self._process_signal(rec['signal'], rec['fs'])
                
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
                    if provenance_builder is not None:
                        provenance_builder.add_block(
                            len(segments),
                            record_id=rec['filename'],
                            session_id=rec['filename'],
                            acquisition_time=rec['date'],
                            acquisition_order=acquisition_order,
                            source_segment_id=rec['filename'],
                            source_segment_order=float(acquisition_order),
                        )

        # Dynamic summary print for all structured tasks during enrollment
        if self.data_split_mode not in ["all-available", "single-session"] and is_enrollment:
            mode_title = self.data_split_mode.replace('-', ' ').title()
            print(f"\n[INFO] {mode_title} Summary: Kept {kept_subjects} subjects. Dropped {dropped_subjects} subjects.")

        return _finalize_loader_output(x_list, y_list, provenance_builder, return_provenance)

# # =============================================================================
# # 4. CYBHi
# # =============================================================================
class load_cybhi_dataset():
    """
    Dynamic Loader for the CYBHi Dataset.
    
    CYBHi holds two collections published together, with different people,
    hardware and session structure. The short-term collection records one
    sitting per participant; the long-term collection records two visits about
    three months apart.

    Session Tags Available for Mapping:
      - 'short-term_CI' (Informed-consent briefing; no stimulus presented)
      - 'short-term_A1' (Watching a low-arousal video)
      - 'short-term_A2' (Watching a high-arousal video)
      - 'long-term_S1'  (First visit)
      - 'long-term_S2'  (Second visit, about three months later)

    The three short-term moments belong to a single sitting a few minutes
    apart, so pairing them measures tolerance to an induced change in
    emotional arousal rather than to a separation between sessions. Only the
    long-term pair is a comparison across visits.

    The recordings carry heavy 50 Hz mains interference, most of all on the
    electrolycra unit where it dominates the spectrum. The pipeline band-pass
    is therefore doing essential work here and should not be weakened: without
    it these signals are largely hum. The sensor is documented as applying a
    1-30 Hz band-pass in hardware, but mains energy survives in the released
    files, so the data cannot be treated as already band-limited.

    Args:
        data_split_mode (str): The routing logic for data extraction.
            Options:
                - 'single-session': Extracts the exact sessions defined in `session_for_single_session_evaluation` 
                                    and pools them for downstream random-splitting.
                - 'cross-session': Maps data strictly to Train/Enroll/Probe groups based on the session arguments below.
        session_for_single_session_evaluation (str or list): Target session(s) to load if mode is 'single-session'.
            Example: 'long-term_S1'
        train_sessions (str or list): Session(s) to load for representation learning.
            Example: 'long-term_S1'
        enroll_sessions (str or list): Session(s) to load for Gallery enrollment.
            Example: 'long-term_S1'
        probe_sessions (str or list): Session(s) to load for Test queries.
            Example: 'long-term_S2'
        electrode_unit (str): Which short-term acquiring unit to read.
            Options:
                - '8B': hand palms, Ag/AgCl electrodes. The default.
                - '85': index and middle fingers, electrolycra.
                - 'both': pool the two, for an electrode-material comparison.
            Every short-term acquisition was recorded by both units at once,
            so reading one unit keeps the electrode configuration fixed within
            an identity: pooling would let enrollment and probe draw the same
            moment from different sensors. The choice holds subject and moment
            fixed and varies only the electrode, which is the comparison the
            dataset was built for. Defaults to the value in ``config.yaml``.
            The long-term collection has one unit and is unaffected.
        num_beats_to_merge (int): Number of consecutive beats to fuse into a single sample.
        beat_merge_method (str): Strategy for fusing beats. Options: ['average', 'concat']
        beat_merge_stride (int): Step between consecutive merge windows.
            The default of 1 slides one beat at a time, so merged samples
            share beats with their neighbours. Set it equal to
            `num_beats_to_merge` for strictly non-overlapping windows.
        cleanup_zip (bool): If True, deletes the downloaded zip file after extraction.
        **preprocessing_params: kwargs passed directly to the Preprocessing class.
    """
    # Subject codes present in the data that do not correspond to a study
    # participant. VIDEOPRINT is a test acquisition: it has no baseline moment,
    # spans two collection days, and is absent from the shipped participant
    # list.
    EXCLUDED_CODES = frozenset({"VIDEOPRINT"})

    # Every short-term acquisition was recorded simultaneously by two units
    # with different electrodes, distinguished by the filename suffix:
    # 8B from the hand palms with Ag/AgCl electrodes, and 85 from the fingers
    # with electrolycra. The long-term unit, 35, is unaffected by this choice.
    SHORT_TERM_UNITS = ("8B", "85", "both")

    def __init__(self, data_split_mode="cross-session",
                 session_for_single_session_evaluation=["long-term_S1"],
                 train_sessions=["long-term_S1"],
                 enroll_sessions=None,
                 probe_sessions=["long-term_S2"],
                 electrode_unit=None,
                 num_beats_to_merge=1, beat_merge_method="average",
                 beat_merge_stride=1,
                 cleanup_zip=False, preprocessing_config=None,
                 **preprocessing_params):

        self.preprocessor = Preprocessing()
        self.cfg = CONFIG["datasets"]["cybhi"]
        project_dir = Path(__file__).resolve().parent
        self.data_root = (project_dir / CONFIG["project"]["data_root"]).resolve()
        self.dataset_root = self.data_root / self.cfg["root_dir"]
        self.zip_path = self.data_root / self.cfg["zip_name"]
        self.url = self.cfg["url"]

        # An explicit argument wins; otherwise the dataset entry in config.yaml
        # supplies the default. The electrolycra recordings mostly defeat beat
        # detection, so the Ag/AgCl unit is the default and 'both' restores the
        # pooled behaviour for an electrode comparison.
        if electrode_unit is None:
            electrode_unit = self.cfg.get("electrode_unit", "8B")

        if electrode_unit not in self.SHORT_TERM_UNITS:
            raise ValueError(
                f"electrode_unit must be one of {list(self.SHORT_TERM_UNITS)}, "
                f"not {electrode_unit!r}."
            )

        self.electrode_unit = electrode_unit
        self.fs = int(self.cfg.get("fs", 1000))

        self.prep_params = _normalize_preprocessing_config(
            self.cfg.get("preprocessing", {}),
            preprocessing_config,
            preprocessing_params,
        )
        self.num_beats = num_beats_to_merge
        self.merge_strategy = beat_merge_method
        self.beat_merge_stride = _normalize_beat_merge_stride(
            beat_merge_stride,
            num_beats_to_merge,
        )
        self.cleanup_zip = cleanup_zip
        
        valid_modes = ["single-session", "cross-session"]
        if data_split_mode not in valid_modes:
            raise ValueError(f"Invalid mode: {data_split_mode}.")
        self.data_split_mode = data_split_mode
        
        # Format variables to lists gracefully
        to_list = lambda x: [x] if isinstance(x, str) else (x if x else [])
        
        self.session_for_single_session_evaluation = to_list(session_for_single_session_evaluation)
        self.train_sessions = to_list(train_sessions)
        self.enroll_sessions = (
            list(self.train_sessions)
            if enroll_sessions is None
            else to_list(enroll_sessions)
        )
        self.probe_sessions = to_list(probe_sessions)
        
        self.required_cross_sessions = list(set(self.train_sessions + self.enroll_sessions + self.probe_sessions))

        if self.data_split_mode == "single-session" and not self.session_for_single_session_evaluation:
            raise ValueError("You must provide `session_for_single_session_evaluation`.")
        if self.data_split_mode == "cross-session" and not self.required_cross_sessions:
            raise ValueError("You must provide at least one valid train, enroll, or probe session.")

        self._reject_mixed_collections()

    def _reject_mixed_collections(self):
        """
        Refuse a protocol that draws on both CYBHi collections at once.

        The short-term and long-term collections are two separate studies with
        two separate cohorts: 65 participants in one, 63 in the other. Ten
        subject codes appear in both, and each such code belongs to a
        different person in each collection, so an experiment spanning the two
        would file two people under one identity.
        """
        # Only the sessions the active mode reads matter: the other mode's
        # arguments keep their constructor defaults and are never used.
        if self.data_split_mode == "single-session":
            requested = set(self.session_for_single_session_evaluation)
        else:
            requested = set(self.required_cross_sessions)

        short_term = sorted(
            s for s in requested if s.startswith("short-term")
        )
        long_term = sorted(
            s for s in requested if s.startswith("long-term")
        )

        if short_term and long_term:
            raise ValueError(
                "CYBHi: a protocol cannot span the short-term and long-term "
                "collections. They are two separate cohorts, and some subject "
                "codes appear in both while naming different people, so "
                f"combining {short_term} with {long_term} would file two "
                "people under one identity. Evaluate each collection in its "
                "own experiment."
            )

    def download(self):
        _download_and_extract(self.url, self.zip_path, self.dataset_root, "CYBHi", cleanup=self.cleanup_zip)

    def _parse_file_info(self, filename):
        """
        Parse one CYBHi filename:
        [Date] - [SID] - [Session/Intervention] - [Acquiring unit]
        """
        clean = filename.replace('.txt', '').replace('._', '')
        parts = clean.split('-')

        rec_date = None
        sid = "UNKNOWN"
        session_code = "UNKNOWN"
        unit = "UNKNOWN"

        if len(parts) >= 3:
            # 1. Date (e.g., 20110715). A CYBHi recording always states a
            # genuine YYYYMMDD acquisition date, and the long-term collection
            # reads it to order the two visits, so an unparseable date is
            # refused rather than replaced with a stand-in.
            if len(parts[0]) == 8 and parts[0].isdigit():
                try:
                    rec_date = datetime.datetime.strptime(parts[0], "%Y%m%d").date()
                except ValueError:
                    rec_date = None
            if rec_date is None:
                raise ValueError(
                    "CYBHi filename states no parseable acquisition date: "
                    f"{filename!r}. Expected a leading YYYYMMDD field."
                )

            # 2. Subject ID (e.g., MLS)
            sid = parts[1].upper()

            # 3. Session Code (e.g., A1, CI, A0)
            session_code = parts[2].upper()

        if len(parts) >= 4:
            # 4. Acquiring unit (e.g., 8B, 85, 35)
            unit = parts[3].upper()

        return rec_date, sid, session_code, unit

    def _read_signal(self, fpath):
        """Bulletproof Pandas reader using the fast C engine and native comment skipping."""
        try:
            # comment='#' prevents silent crashes on header info
            df = pd.read_csv(fpath, comment='#', delim_whitespace=True, header=None, on_bad_lines='skip')
            
            col_idx = self.cfg.get("ecg_column", 2)
            if col_idx >= df.shape[1]: col_idx = 0
            
            sig = df.iloc[:, col_idx].dropna().values.astype(float)
            sig = sig - np.mean(sig) # Zero-mean baseline
            return sig
        except Exception as e:
            print(f"\n[ERR] Failed to read {fpath.name}: {e}")
            return None

    def load_raw_data(self):
        if not self.dataset_root.exists() or not any(self.dataset_root.iterdir()):
            self.download()

        print("\n[INFO] Scanning directories and pooling CYBHi files...")
        st_pool = {} # Short-Term
        lt_pool = {} # Long-Term

        # 1. Distribute files to the two collections. The collection is decided
        # by the exact directory names on the path, never by substring matching
        # against the whole path: the absolute path also contains the location
        # the repository was cloned into, which must not influence which pool a
        # recording lands in. Directories and files are visited in sorted order
        # so enumeration does not depend on the filesystem.
        for root, dirs, files in os.walk(self.dataset_root):
            dirs.sort()
            components = {part.lower() for part in Path(root).parts}

            # The archive ships a __MACOSX tree of resource forks; skip it.
            if "__macosx" in components:
                continue

            is_st = "short-term" in components
            is_lt = "long-term" in components

            if not is_st and not is_lt: continue

            for f in sorted(files):
                # IMPORTANT: Skip macOS hidden metadata files that crash the reader
                if not f.endswith(".txt") or f.startswith("._"):
                    continue

                fpath = Path(root) / f
                rec_date, sid, session_code, unit = self._parse_file_info(f)

                if sid == "UNKNOWN": continue
                if sid in self.EXCLUDED_CODES: continue

                if is_st:
                    # Keep only the recordings of the selected acquiring unit.
                    if self.electrode_unit != "both" and unit != self.electrode_unit:
                        continue
                    if sid not in st_pool: st_pool[sid] = []
                    st_pool[sid].append({"path": fpath, "date": rec_date, "code": session_code})
                elif is_lt:
                    if sid not in lt_pool: lt_pool[sid] = []
                    lt_pool[sid].append({"path": fpath, "date": rec_date, "code": session_code})

        recordings = {}

        # 2. Load Short-Term signals directly using explicit intervention tags
        for sid, recs in tqdm(st_pool.items(), desc="Loading short-term raw files"):
            if sid not in recordings: recordings[sid] = {}
            for rec in recs:
                tag = f"short-term_{rec['code']}" # Outputs: short-term_CI, short-term_A1, short-term_A2
                if tag not in recordings[sid]: recordings[sid][tag] = []
                
                sig = self._read_signal(rec['path'])
                if sig is not None:
                    recordings[sid][tag].append({'signal': sig, 'fs': self.fs, 'record_id': rec['path'].name, 'date': rec['date']})

        # 3. Load Long-Term signals by date sequence (Month 0 vs Month 3)
        for sid, recs in tqdm(lt_pool.items(), desc="Loading long-term raw files"):
            if sid not in recordings: recordings[sid] = {}
            recs.sort(key=lambda x: x["date"])
            unique_dates = sorted(list(set([r['date'] for r in recs])))
            
            for rec in recs:
                # 1st Date = S1. 2nd Date = S2.
                date_idx = unique_dates.index(rec['date']) + 1
                tag = f"long-term_S{date_idx}" # Outputs: long-term_S1, long-term_S2
                if tag not in recordings[sid]: recordings[sid][tag] = []
                
                sig = self._read_signal(rec['path'])
                if sig is not None:
                    recordings[sid][tag].append({'signal': sig, 'fs': self.fs, 'record_id': rec['path'].name, 'date': rec['date']})

        return recordings

    def _process_signal(self, sig, fs=1000):
        if np.isnan(sig).any() or len(sig) < fs or np.std(sig) < 1e-5: 
            return np.empty((0, 0))

        beats = _preprocess_signal(
                self.preprocessor,
                sig,
                fs,
                self.prep_params,
            )
        if self.num_beats == 1: return beats
        if len(beats) < self.num_beats: return np.empty((0, beats.shape[1]))
        
        processed_samples = []
        merge_starts = _beat_merge_start_indices(
            len(beats),
            self.num_beats,
            self.beat_merge_stride,
        )
        for i in merge_starts:
            group = beats[i : i + self.num_beats]
            if self.merge_strategy == "average": processed_samples.append(np.mean(group, axis=0))
            elif self.merge_strategy == "concat": processed_samples.append(group.flatten())
        return np.array(processed_samples)

    def load_all_data(self, return_provenance=False):
        if self.data_split_mode != "single-session":
            print(f"[WARN] Calling load_all_data() but mode is '{self.data_split_mode}'.")
        return self.load_session("train", return_provenance=return_provenance)

    def load_session(self, session_name, return_provenance=False):
        session_name = session_name.lower()
        target_sessions = []
        is_primary_pass = False
        
        if self.data_split_mode == "single-session":
            if session_name in ["probe", "test"]:
                raise ValueError("Cannot load 'test' in single-session mode. Split upstream.")
            target_sessions = self.session_for_single_session_evaluation
            log_name = f"Single-Session Target(s): {target_sessions}"
            is_primary_pass = True
            
        elif self.data_split_mode == "cross-session":
            if session_name in ["train"]:
                target_sessions = self.train_sessions
                is_primary_pass = True
            elif session_name in ["enrol", "enrollment"]:
                target_sessions = self.enroll_sessions
                is_primary_pass = True if not self.train_sessions else False
            elif session_name in ["probe", "test"]:
                target_sessions = self.probe_sessions
            else:
                raise ValueError("session_name must be 'train', 'enrol', or 'test'.")
            log_name = f"Cross-Session ({session_name.title()}): {target_sessions}"

        if not target_sessions:
            return _finalize_loader_output([], [], None, return_provenance)
            
        data = self.load_raw_data()
        x_list, y_list = [], []
        provenance_builder = _ProvenanceBuilder() if return_provenance else None
        
        kept_subjects, dropped_subjects = 0, 0

        for sid, tagged_sessions in tqdm(data.items(), desc=f"Processing {log_name}"):
            
            if self.data_split_mode == "single-session":
                is_valid = all(s in tagged_sessions for s in target_sessions)
            elif self.data_split_mode == "cross-session":
                # Strict: Subject MUST have data in ALL requested global sets
                is_valid = all(s in tagged_sessions for s in self.required_cross_sessions)

            if is_valid:
                kept_subjects += 1
                record_order = 0
                for s in target_sessions:
                    # The acquiring-unit selection already happened during
                    # enumeration, so each session holds recordings from one
                    # unit unless 'both' was requested explicitly.
                    for signal_dict in tagged_sessions[s]:
                        segments = self._process_signal(signal_dict['signal'], signal_dict['fs'])
                        if len(segments) > 0:
                            x_list.append(segments)
                            y_list.extend([sid] * len(segments))
                            if provenance_builder is not None:
                                provenance_builder.add_block(
                                    len(segments),
                                    record_id=signal_dict['record_id'],
                                    session_id=s,
                                    acquisition_time=signal_dict['date'],
                                    acquisition_order=record_order,
                                    source_segment_id=signal_dict['record_id'],
                                    source_segment_order=float(record_order),
                                )
                        record_order += 1
            else:
                dropped_subjects += 1

        if is_primary_pass:
            print(f"\n[INFO] CYBHi Evaluation Summary ({self.data_split_mode.title()}):")
            print(f"       Kept {kept_subjects} matched subjects. Dropped {dropped_subjects} subjects.")

        return _finalize_loader_output(x_list, y_list, provenance_builder, return_provenance)

# =============================================================================
# 5. MIT-BIH Arrhythmia Database
# =============================================================================
class load_mitbih_dataset():
    """
    Robust Loader for the MIT-BIH Arrhythmia Database.
    
    This dataset consists of 48 continuous ~30-minute recordings from 47 subjects.
    Because there are no distinct "sessions" per subject, biometric evaluation 
    requires slicing the continuous timeline into discrete minute-based chunks.

    Args:
        leads (list of str): Target leads to extract.
            Options: Usually ['MLII'] or ['V1']. Pass 'all' for both available leads.
            Default: ['MLII']
        data_split_mode (str): Evaluation regime mapping.
            Options:
                - 'all-available': Loads the entire 30-minute continuous signal.
                - 'single-segment': Extracts a continuous chunk based on `single_segment_range`.
                - 'custom-split': Manually maps exact minute ranges to Train/Enroll/Probe regimes.
        single_segment_range (tuple): Used only if mode='single-segment'. Defines (start_min, end_min).
            Example: (0, 5) extracts the first 5 minutes.
        train_parts (list of tuples): Minute ranges for training data if mode='custom-split'.
            Example: [(0, 5), (10, 15)]
        enrol_parts (list of tuples): Minute ranges for template enrollment if mode='custom-split'.
        test_parts (list of tuples): Minute ranges for test probes if mode='custom-split'.
            Example: [(25, 30)] extracts the last 5 minutes of the tape.
        num_beats_to_merge (int): Number of consecutive beats to fuse.
        beat_merge_method (str): Strategy for fusing beats. Options: ['average', 'concat']
        beat_merge_stride (int): Step between consecutive merge windows.
            The default of 1 slides one beat at a time, so merged samples
            share beats with their neighbours. Set it equal to
            `num_beats_to_merge` for strictly non-overlapping windows.
        temporal_guard_minutes (float): Minimum separation enforced between
            the enrollment coverage and the probe coverage in 'custom-split'
            mode. The default of 0.0 permits adjacent windows and rejects
            only true overlap.
        cleanup_zip (bool): If True, deletes the downloaded zip file after extraction.
        **preprocessing_params: kwargs passed directly to the Preprocessing class.
    """
    # The database documentation states that records 201 and 202 came from the
    # same male subject, so the 48 records represent 47 people. Mapping the
    # second record onto the first keeps one identity per person.
    SHARED_SUBJECT_RECORDS = {
        "202": "201",
    }

    def __init__(self, leads=['MLII'], data_split_mode="all-available",
                 single_segment_range=(0, 5), train_parts=None, enrol_parts=None,
                 test_parts=None, num_beats_to_merge=1, beat_merge_method="average",
                 beat_merge_stride=1, temporal_guard_minutes=0.0,
                 cleanup_zip=False, preprocessing_config=None,
                 **preprocessing_params):

        self.preprocessor = Preprocessing()
        self.cfg = CONFIG["datasets"]["mitbih"]
        project_dir = Path(__file__).resolve().parent
        self.data_root = (project_dir / CONFIG["project"]["data_root"]).resolve()
        self.dataset_root = self.data_root / self.cfg["root_dir"]
        self.zip_path = self.data_root / self.cfg["zip_name"]
        self.url = self.cfg["url"]
        
        self.prep_params = _normalize_preprocessing_config(
            self.cfg.get("preprocessing", {}),
            preprocessing_config,
            preprocessing_params,
        )
        self.target_leads = [l.lower() for l in leads] if isinstance(leads, list) else leads
        self.num_beats = num_beats_to_merge
        self.merge_strategy = beat_merge_method
        self.beat_merge_stride = _normalize_beat_merge_stride(
            beat_merge_stride,
            num_beats_to_merge,
        )
        self.temporal_guard_minutes = float(temporal_guard_minutes)
        self.cleanup_zip = cleanup_zip

        valid_modes = ["all-available", "single-segment", "custom-split"]
        if data_split_mode not in valid_modes:
            raise ValueError(f"Invalid mode: {data_split_mode}. Use {valid_modes}")
        self.data_split_mode = data_split_mode

        # Segment mappings (in minutes)
        self.single_segment_range = single_segment_range
        self.train_parts = train_parts
        self.enrol_parts = (
            list(train_parts)
            if enrol_parts is None
            and train_parts is not None
            else enrol_parts
        )
        self.test_parts = test_parts
        self.temporal_partition_audit = None

        # Strict validation for custom-split
        if self.data_split_mode == "custom-split":
            if not self.train_parts or not self.test_parts:
                raise ValueError(
                    "For 'custom-split' mode, `train_parts` and `test_parts` cannot be None. "
                    "Please provide minute ranges. Example: train_parts=[(0, 5)], test_parts=[(25, 30)]"
                )

            self.temporal_partition_audit = (
                audit_continuous_temporal_partitions(
                    train_parts=self.train_parts,
                    enrol_parts=self.enrol_parts,
                    test_parts=self.test_parts,
                    temporal_guard_minutes=self.temporal_guard_minutes,
                )
            )

    def download(self):
        """Downloads and extracts the dataset if missing."""
        _download_and_extract(self.url, self.zip_path, self.dataset_root, "MIT-BIH", cleanup=self.cleanup_zip)

    def _get_lead_indices(self, available_leads):
        """Maps requested lead names (e.g., 'MLII') to channel indices."""
        avail_norm = [l.lower().strip() for l in available_leads]
        if self.target_leads == 'all': return list(range(len(available_leads)))
        indices = []
        for req in self.target_leads:
            req = req.strip().lower()
            if req in avail_norm: indices.append(avail_norm.index(req))
            else:
                for i, avail in enumerate(avail_norm):
                    if req in avail or avail in req: indices.append(i); break
        return indices

    def _normalized_source_ranges(self, min_ranges):
        """
        Order the configured ranges by source time and reject a same-role
        overlap, which would otherwise read the same samples into two
        separate segments of one role.
        """
        ordered = _normalize_minute_ranges(min_ranges, "range")
        for earlier, later in zip(ordered, ordered[1:]):
            if later[0] < earlier[1]:
                raise ValueError(
                    "A single role must not request overlapping minute "
                    f"ranges; {earlier} and {later} overlap."
                )
        return ordered

    def load_raw_data(self, min_ranges=None):
        """
        Load each physical record's configured source ranges independently.

        Every configured minute range of a physical recording is read and
        returned as its own segment; ranges are never stitched into one raw
        signal, so no filtering, R-peak detection or beat window crosses a
        boundary between two non-contiguous source ranges. ``None`` reads the
        whole recording as a single segment, which 'all-available' needs.

        Records mapped to the same biometric subject contribute their segments
        under that subject while keeping their own ``record_id``; their raw
        waveforms are never concatenated before processing.

        Returns:
            dict: Subject id mapped to a list of source segments, each
                carrying its signal, sampling rate, physical record identity
                and the numeric source start (in minutes) of the range.
        """
        if not self.dataset_root.exists() or not any(self.dataset_root.iterdir()):
            self.download()

        source_ranges = (
            None if min_ranges is None
            else self._normalized_source_ranges(min_ranges)
        )

        recordings = {}
        record_orders = {}

        for name in tqdm(self._record_names(), desc="Loading MIT-BIH raw files"):
            hea = self.dataset_root / f"{name}.hea"
            sid = self.SHARED_SUBJECT_RECORDS.get(name, name)
            record_id = hea.name
            try:
                rec_header = wfdb.rdheader(str(hea.with_suffix("")))

                verify_sampling_rate(
                    self.cfg.get("fs"),
                    "MIT-BIH",
                    hea.name,
                    rec_header.fs,
                )

                lead_indices = self._get_lead_indices(rec_header.sig_name)
                if not lead_indices:
                    continue

                fs = rec_header.fs
                total_samples = rec_header.sig_len
                record_path = str(hea.with_suffix(""))

                spans = [(0.0, None)] if source_ranges is None else source_ranges

                orders = record_orders.setdefault(sid, {})
                if record_id not in orders:
                    orders[record_id] = len(orders)
                acquisition_order = orders[record_id]

                for (start_min, end_min) in spans:
                    if end_min is None:
                        signal, _ = wfdb.rdsamp(
                            record_path, channels=lead_indices
                        )
                        segment_start = 0.0
                    else:
                        start_idx = max(0, int(start_min * 60 * fs))
                        end_idx = min(int(total_samples), int(end_min * 60 * fs))
                        if start_idx >= end_idx:
                            continue
                        signal, _ = wfdb.rdsamp(
                            record_path, channels=lead_indices,
                            sampfrom=start_idx, sampto=end_idx,
                        )
                        segment_start = float(start_min)

                    recordings.setdefault(sid, []).append(
                        {
                            "signal": signal,
                            "fs": fs,
                            "filename": record_id,
                            "record_id": record_id,
                            "start_min": segment_start,
                            "acquisition_order": acquisition_order,
                        }
                    )
            except ValueError:
                raise
            except Exception as read_error:
                print(
                    f"[WARN] MIT-BIH: skipping {hea.name}: {read_error}"
                )

        return recordings

    def _record_names(self):
        """
        Return the canonical record names, in the order the database lists them.

        The shipped RECORDS file is the authoritative index and is already
        sorted, which makes enumeration reproducible across filesystems. It
        also excludes the x_mitdb directory, whose records are copies of the
        first ten minutes of records listed here and which therefore carry no
        additional identities.
        """
        manifest = self.dataset_root / "RECORDS"

        if not manifest.exists():
            raise FileNotFoundError(
                f"MIT-BIH record list not found at {manifest}. The database "
                "ships this file; re-download if it is missing."
            )

        return [
            line.strip()
            for line in manifest.read_text().splitlines()
            if line.strip()
        ]

    def _process_signal(self, sig, fs):
        """Applies filters, segmentation, and multi-beat merging."""
        n_channels = sig.shape[1]
        processed_channels = []
        for c in range(n_channels):
            processed_channels.append(_preprocess_signal(
                self.preprocessor,
                sig[:, c],
                fs,
                self.prep_params,
            ))
        if not processed_channels: return np.empty((0, n_channels, 0))
        min_len = min([len(ch) for ch in processed_channels])
        if min_len == 0: return np.empty((0, n_channels, 0))
        
        beats_multi = np.stack([ch[:min_len] for ch in processed_channels], axis=1)
        if self.num_beats == 1: return beats_multi[:, 0, :] if n_channels == 1 else beats_multi
        if len(beats_multi) < self.num_beats: return np.empty((0, n_channels, 0))
        
        merged_samples = []
        merge_starts = _beat_merge_start_indices(
            len(beats_multi),
            self.num_beats,
            self.beat_merge_stride,
        )
        for i in merge_starts:
            group = beats_multi[i : i + self.num_beats]
            if self.merge_strategy == "average":
                merged = np.mean(group, axis=0)
                if n_channels == 1: merged = merged.squeeze(0)
                merged_samples.append(merged)
            elif self.merge_strategy == "concat":
                merged = group.transpose(1, 0, 2).reshape(n_channels, -1)
                if n_channels == 1: merged = merged.squeeze(0)
                merged_samples.append(merged)
        return np.array(merged_samples)

    def _slice_signal(self, raw_signal, fs, min_ranges):
        """
        Takes a continuous raw signal and extracts the requested minute boundaries.
        Returns a concatenated raw array to pass to the preprocessor.

        The loader reads those boundaries directly from disk rather than
        reading a whole recording and cutting it, so this is no longer on the
        loading path. It is kept as the reference the reading path is checked
        against: expressing the same selection a second way is what makes the
        equivalence test meaningful.
        """
        if not min_ranges: return np.empty((0, raw_signal.shape[1]))
        
        sliced_chunks = []
        total_samples = raw_signal.shape[0]
        
        for (start_min, end_min) in min_ranges:
            start_idx = int(start_min * 60 * fs)
            end_idx = int(end_min * 60 * fs)
            
            # Boundary protections
            start_idx = max(0, start_idx)
            end_idx = min(total_samples, end_idx)
            
            if start_idx < end_idx:
                sliced_chunks.append(raw_signal[start_idx:end_idx, :])
                
        if sliced_chunks:
            return np.vstack(sliced_chunks)
        return np.empty((0, raw_signal.shape[1]))

    def load_all_data(self, return_provenance=False):
        """
        Handles dataset loading for downstream random-split tasks.
        Applies to 'all-available' and 'single-segment'.
        """
        if self.data_split_mode not in ["all-available", "single-segment"]:
            print(f"[WARN] Calling load_all_data() but mode is '{self.data_split_mode}'.")
            
        # The ranges are applied while reading, so the signal returned here is
        # already the span the protocol asks for and must not be sliced again.
        target_ranges = None
        if self.data_split_mode == "single-segment":
            target_ranges = [self.single_segment_range]

        data = self.load_raw_data(min_ranges=target_ranges)
        x_list, y_list = [], []
        provenance_builder = _ProvenanceBuilder() if return_provenance else None

        for sid, recs in tqdm(data.items(), desc="Processing signals"):
            for rec in recs:
                target_signal = rec['signal']
                fs = rec['fs']

                if target_signal.shape[0] == 0: continue

                segments = self._process_signal(target_signal, fs)
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
                    if provenance_builder is not None:
                        provenance_builder.add_block(
                            len(segments),
                            record_id=rec['record_id'],
                            session_id=rec['record_id'],
                            acquisition_time=None,
                            acquisition_order=rec['acquisition_order'],
                            source_segment_id=f"{rec['record_id']}#{rec['start_min']}",
                            source_segment_order=float(rec['start_min']),
                        )

        return _finalize_loader_output(x_list, y_list, provenance_builder, return_provenance)

    def load_session(self, session_name, return_provenance=False):
        """
        Processes the customized minute-based ranges mapping to Train/Enrol/Test tasks.
        """
        if self.data_split_mode != "custom-split":
            raise ValueError("load_session() is only valid when data_split_mode='custom-split'.")
            
        session_name = session_name.lower()
        target_ranges = []
        
        # Route the correct minute ranges based on the requested session
        if session_name in ["train"]:
            target_ranges = self.train_parts
        elif session_name in ["enrol", "enrollment"]:
            if not self.enrol_parts: return _finalize_loader_output([], [], None, return_provenance)
            target_ranges = self.enrol_parts
        elif session_name in ["test", "probe"]:
            target_ranges = self.test_parts
        else:
            raise ValueError("session_name must be 'train', 'enrol', or 'test'.")

        # Reading the assigned minutes directly avoids pulling in the rest of a
        # half-hour recording only to discard it.
        data = self.load_raw_data(min_ranges=target_ranges)
        x_list, y_list = [], []
        provenance_builder = _ProvenanceBuilder() if return_provenance else None

        subjects_with_data = set()

        for sid, recs in tqdm(data.items(), desc=f"Processing {session_name}"):
            for rec in recs:
                target_signal = rec['signal']
                fs = rec['fs']

                if target_signal.shape[0] == 0: continue

                segments = self._process_signal(target_signal, fs)

                if len(segments) > 0:
                    subjects_with_data.add(sid)
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
                    if provenance_builder is not None:
                        provenance_builder.add_block(
                            len(segments),
                            record_id=rec['record_id'],
                            session_id=rec['record_id'],
                            acquisition_time=None,
                            acquisition_order=rec['acquisition_order'],
                            source_segment_id=f"{rec['record_id']}#{rec['start_min']}",
                            source_segment_order=float(rec['start_min']),
                        )

        kept_subjects = len(subjects_with_data)

        if session_name == "train":
            print(f"\n[INFO] Custom Split Summary: Extracted data for {kept_subjects} subjects.")

        return _finalize_loader_output(x_list, y_list, provenance_builder, return_provenance)

# =============================================================================
# 6. MIT-BIH NSRDB (Normal Sinus Rhythm)
# =============================================================================
class load_nsrdb_dataset():
    """
    Highly Optimized Loader for the MIT-BIH Normal Sinus Rhythm Database (NSRDB).
    
    This dataset consists of 18 extremely long-term Holter recordings (~24 hours continuous).
    To prevent memory overflow, this loader calculates exact byte boundaries and 
    reads ONLY the requested minute slices directly from the disk.

    Args:
        leads (list of str): Target leads to extract. 
            Options: Usually ['ECG1'] or ['ECG2'].
            Default: ['ECG1']
        data_split_mode (str): Evaluation regime mapping.
            Options:
                - 'all-available': Loads the ENTIRE 24-hour signal (Warning: High Memory/RAM usage).
                - 'single-segment': Extracts a continuous chunk based on `single_segment_range`.
                - 'custom-split': Manually maps exact minute ranges to Train/Enroll/Probe regimes.
        single_segment_range (tuple): Used only if mode='single-segment'. Defines (start_min, end_min).
            Example: (0, 60) extracts the first hour.
        train_parts (list of tuples): Minute ranges for training data if mode='custom-split'.
            Example: [(0, 120)] extracts the first 2 hours.
        enrol_parts (list of tuples): Minute ranges for template enrollment if mode='custom-split'.
        test_parts (list of tuples): Minute ranges for test probes if mode='custom-split'.
            Example: [(1380, 1440)] extracts the final hour of the 24-hour tape.
        num_beats_to_merge (int): Number of consecutive beats to fuse.
        beat_merge_method (str): Strategy for fusing beats. Options: ['average', 'concat']
        beat_merge_stride (int): Step between consecutive merge windows.
            The default of 1 slides one beat at a time, so merged samples
            share beats with their neighbours. Set it equal to
            `num_beats_to_merge` for strictly non-overlapping windows.
        temporal_guard_minutes (float): Minimum separation enforced between
            the enrollment coverage and the probe coverage in 'custom-split'
            mode. The default of 0.0 permits adjacent windows and rejects
            only true overlap.
        cleanup_zip (bool): If True, deletes the downloaded zip file after extraction.
        **preprocessing_params: kwargs passed directly to the Preprocessing class.
    """
    def __init__(self, leads=['ECG1'], data_split_mode="all-available",
                 single_segment_range=(0, 60), train_parts=None, enrol_parts=None,
                 test_parts=None, num_beats_to_merge=1, beat_merge_method="average",
                 beat_merge_stride=1, temporal_guard_minutes=0.0,
                 cleanup_zip=False, preprocessing_config=None,
                 **preprocessing_params):

        self.preprocessor = Preprocessing()
        self.cfg = CONFIG["datasets"]["nsrdb"]
        project_dir = Path(__file__).resolve().parent
        self.data_root = (project_dir / CONFIG["project"]["data_root"]).resolve()
        self.dataset_root = self.data_root / self.cfg["root_dir"]
        self.zip_path = self.data_root / self.cfg["zip_name"]
        self.url = self.cfg["url"]
        
        self.prep_params = _normalize_preprocessing_config(
            self.cfg.get("preprocessing", {}),
            preprocessing_config,
            preprocessing_params,
        )
        self.target_leads = [l.lower() for l in leads] if isinstance(leads, list) else leads
        self.num_beats = num_beats_to_merge
        self.merge_strategy = beat_merge_method
        self.beat_merge_stride = _normalize_beat_merge_stride(
            beat_merge_stride,
            num_beats_to_merge,
        )
        self.temporal_guard_minutes = float(temporal_guard_minutes)
        self.cleanup_zip = cleanup_zip

        valid_modes = ["all-available", "single-segment", "custom-split"]
        if data_split_mode not in valid_modes:
            raise ValueError(f"Invalid mode: {data_split_mode}. Use {valid_modes}")
        self.data_split_mode = data_split_mode

        # Segment mappings (in minutes)
        self.single_segment_range = single_segment_range
        self.train_parts = train_parts
        self.enrol_parts = (
            list(train_parts)
            if enrol_parts is None
            and train_parts is not None
            else enrol_parts
        )
        self.test_parts = test_parts
        self.temporal_partition_audit = None

        # Strict validation for custom-split
        if self.data_split_mode == "custom-split":
            if not self.train_parts or not self.test_parts:
                raise ValueError(
                    "For 'custom-split' mode, `train_parts` and `test_parts` cannot be None. "
                    "Please provide minute ranges. Example: train_parts=[(0, 60)], test_parts=[(1380, 1440)]"
                )

            self.temporal_partition_audit = (
                audit_continuous_temporal_partitions(
                    train_parts=self.train_parts,
                    enrol_parts=self.enrol_parts,
                    test_parts=self.test_parts,
                    temporal_guard_minutes=self.temporal_guard_minutes,
                )
            )

    def download(self):
        """Downloads and extracts dataset."""
        _download_and_extract(self.url, self.zip_path, self.dataset_root, "NSRDB", cleanup=self.cleanup_zip)

    def _get_lead_indices(self, available_leads):
        """Finds channel indices."""
        avail_norm = [l.lower().strip() for l in available_leads]
        if self.target_leads == 'all': return list(range(len(available_leads)))
        indices = []
        for req in self.target_leads:
            req = req.strip().lower()
            if req in avail_norm: indices.append(avail_norm.index(req))
            else:
                for i, avail in enumerate(avail_norm):
                    if req in avail or avail in req: indices.append(i); break
        return indices

    def load_raw_data_slices(self, min_ranges=None):
        """
        Core I/O Optimizer: Reads ONLY the specified minute chunks from disk.
        If min_ranges is None, it loads the entire 24h file.
        """
        if not self.dataset_root.exists() or not any(self.dataset_root.iterdir()):
            self.download()
            
        recordings = {}

        for name in tqdm(self._record_names(), desc="Loading NSRDB specific slices"):
            hea = self.dataset_root / f"{name}.hea"
            sid = name
            try:
                # 1. Read lightweight header to get total length and sampling rate
                rec_header = wfdb.rdheader(str(hea.with_suffix("")))
                total_samples = rec_header.sig_len
                fs = rec_header.fs

                verify_sampling_rate(
                    self.cfg.get("fs"),
                    "NSRDB",
                    hea.name,
                    fs,
                )

                lead_indices = self._get_lead_indices(rec_header.sig_name)
                if not lead_indices: continue

                # If no ranges specified, load the entire massive file
                if min_ranges is None:
                    data, _ = wfdb.rdsamp(str(hea.with_suffix("")), channels=lead_indices)
                    if sid not in recordings: recordings[sid] = []
                    recordings[sid].append({"signal": data, "fs": fs, "filename": hea.name, "start_min": 0})
                    continue

                # 2. Iterate through requested chunks and pull them efficiently from disk
                for (start_min, end_min) in min_ranges:
                    sampfrom = int(start_min * 60 * fs)
                    sampto = int(end_min * 60 * fs)
                    
                    # Boundary protection
                    sampfrom = max(0, min(sampfrom, total_samples))
                    sampto = max(0, min(sampto, total_samples))
                    
                    if sampfrom < sampto:
                        data, _ = wfdb.rdsamp(str(hea.with_suffix("")), channels=lead_indices, sampfrom=sampfrom, sampto=sampto)
                        if sid not in recordings: recordings[sid] = []
                        recordings[sid].append({"signal": data, "fs": fs, "filename": hea.name, "start_min": start_min})
                        
            except ValueError:
                # Raised deliberately when a recording contradicts the dataset
                # configuration, which must stop the run rather than quietly
                # reduce the cohort.
                raise
            except Exception as read_error:
                print(
                    f"[WARN] NSRDB: skipping {hea.name}: {read_error}"
                )

        return recordings

    def _record_names(self):
        """
        Return the record names, in the order the database lists them.

        The shipped RECORDS file is the authoritative index and is already
        sorted, which makes enumeration reproducible across filesystems.
        """
        manifest = self.dataset_root / "RECORDS"

        if not manifest.exists():
            raise FileNotFoundError(
                f"NSRDB record list not found at {manifest}. The database "
                "ships this file; re-download if it is missing."
            )

        return [
            line.strip()
            for line in manifest.read_text().splitlines()
            if line.strip()
        ]

    def _process_signal(self, sig, fs):
        """Applies filters, segmentation, and multi-beat merging."""
        n_channels = sig.shape[1]
        processed_channels = []
        for c in range(n_channels):
            processed_channels.append(_preprocess_signal(
                self.preprocessor,
                sig[:, c],
                fs,
                self.prep_params,
            ))
        if not processed_channels: return np.empty((0, n_channels, 0))
        min_len = min([len(ch) for ch in processed_channels])
        if min_len == 0: return np.empty((0, n_channels, 0))
        
        beats_multi = np.stack([ch[:min_len] for ch in processed_channels], axis=1)
        if self.num_beats == 1: return beats_multi[:, 0, :] if n_channels == 1 else beats_multi
        if len(beats_multi) < self.num_beats: return np.empty((0, n_channels, 0))
        
        merged_samples = []
        merge_starts = _beat_merge_start_indices(
            len(beats_multi),
            self.num_beats,
            self.beat_merge_stride,
        )
        for i in merge_starts:
            group = beats_multi[i : i + self.num_beats]
            if self.merge_strategy == "average":
                merged = np.mean(group, axis=0)
                if n_channels == 1: merged = merged.squeeze(0)
                merged_samples.append(merged)
            elif self.merge_strategy == "concat":
                merged = group.transpose(1, 0, 2).reshape(n_channels, -1)
                if n_channels == 1: merged = merged.squeeze(0)
                merged_samples.append(merged)
        return np.array(merged_samples)

    def load_all_data(self, return_provenance=False):
        """
        Handles dataset loading for 'all-available' and 'single-segment'.
        """
        if self.data_split_mode not in ["all-available", "single-segment"]:
            print(f"[WARN] Calling load_all_data() but mode is '{self.data_split_mode}'.")
            
        # Determine ranges
        target_ranges = None
        if self.data_split_mode == "single-segment":
            target_ranges = [self.single_segment_range]
            
        # The optimizer perfectly fetches only what we need from disk
        data = self.load_raw_data_slices(min_ranges=target_ranges)
        
        x_list, y_list = [], []
        provenance_builder = _ProvenanceBuilder() if return_provenance else None
        for sid, recs in tqdm(data.items(), desc="Processing signals"):
            for rec in recs:
                segments = self._process_signal(rec['signal'], rec['fs'])
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
                    if provenance_builder is not None:
                        provenance_builder.add_block(
                            len(segments),
                            record_id=rec['filename'],
                            session_id=rec['filename'],
                            acquisition_time=None,
                            acquisition_order=0,
                            source_segment_id=f"{rec['filename']}#{rec['start_min']}",
                            source_segment_order=float(rec['start_min']),
                        )
                    
        return _finalize_loader_output(x_list, y_list, provenance_builder, return_provenance)

    def load_session(self, session_name, return_provenance=False):
        """
        Processes the customized minute-based ranges mapping to Train/Enrol/Test tasks.
        """
        if self.data_split_mode != "custom-split":
            raise ValueError("load_session() is only valid when data_split_mode='custom-split'.")
            
        session_name = session_name.lower()
        target_ranges = []
        
        # Route the correct minute ranges based on the requested session
        if session_name in ["train"]:
            target_ranges = self.train_parts
        elif session_name in ["enrol", "enrollment"]:
            if not self.enrol_parts: return _finalize_loader_output([], [], None, return_provenance)
            target_ranges = self.enrol_parts
        elif session_name in ["test", "probe"]:
            target_ranges = self.test_parts
        else:
            raise ValueError("session_name must be 'train', 'enrol', or 'test'.")

        # Fetch explicitly only the requested bytes from the massive 24h disk files
        data = self.load_raw_data_slices(min_ranges=target_ranges)
        
        x_list, y_list = [], []
        provenance_builder = _ProvenanceBuilder() if return_provenance else None
        kept_subjects = 0

        for sid, recs in tqdm(data.items(), desc=f"Processing {session_name}"):
            if not recs: continue
            
            subject_has_data = False
            for rec in recs:
                segments = self._process_signal(rec['signal'], rec['fs'])
                if len(segments) > 0:
                    subject_has_data = True
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
                    if provenance_builder is not None:
                        provenance_builder.add_block(
                            len(segments),
                            record_id=rec['filename'],
                            session_id=rec['filename'],
                            acquisition_time=None,
                            acquisition_order=0,
                            source_segment_id=f"{rec['filename']}#{rec['start_min']}",
                            source_segment_order=float(rec['start_min']),
                        )
                    
            if subject_has_data:
                kept_subjects += 1

        if session_name == "train":
            print(f"\n[INFO] Custom Split Summary: Extracted data for {kept_subjects} subjects.")

        return _finalize_loader_output(x_list, y_list, provenance_builder, return_provenance)

# =============================================================================
# 7. PTB-XL (Physikalisch-Technische Bundesanstalt XL)
# =============================================================================
class load_ptbxl_dataset():
    """
    Robust Loader for the PTB-XL Dataset.
    
    This is a massive clinical dataset (21k+ records). Every recording is exactly 10 seconds long.
    To ensure robust feature extraction, records are never split internally. Subjects lacking 
    the required number of discrete 10-second recordings for a given task are strictly dropped.

    Args:
        leads (list of str): Target leads to extract.
            Options: Any valid 12-lead string (e.g., ['i', 'v5', 'avf']) or 'all' for all 12 channels.
            Default: ['i']
        resolution (str): Which of the two stored copies to read.
            Options:
                - 'high': the 500 Hz copy, and the default.
                - 'low': the 100 Hz copy.
            Both copies hold the same 10-second recordings. The 100 Hz copy is
            a resampling rather than a subsample, so the two are not related by
            simple decimation and results obtained at one rate do not carry
            over to the other. Each header is checked against the rate its
            resolution implies.
        only_healthy (bool): If True, keeps only subjects whose baseline
            recording carries the 'NORM' diagnostic code, which is 8,896 of the
            18,869 patients. Note that a record may carry 'NORM' alongside
            another positive finding: requiring 'NORM' to be the only
            positive code would leave 8,411 patients instead.
        data_split_mode (str): Evaluation regime mapping to strictly partition the 10-second records.
            Options:
                - 'all-available': Loads every record (used for random beat-level splitting).
                - 'single-session': Loads ONLY the 1st record of each subject.
                - 'single-cross-session': 1st record = Train/Enroll, 2nd record = Test/Probe.
                - 'single-shot-short-term': Day 1's 1st record = Enroll, rest of Day 1 = Probe.
                - 'leave-last-out-short-term': Day 1's last record = Probe, rest of Day 1 = Enroll.
                - 'single-shot-long-term': All Day 1 records = Enroll, all future days = Probe.
                - 'leave-last-out-long-term': Last recording day = Probe, all past days = Enroll.
        num_beats_to_merge (int): Number of consecutive beats to fuse into a single sample.
        beat_merge_method (str): Strategy for fusing beats. Options: ['average', 'concat']
        beat_merge_stride (int): Step between consecutive merge windows.
            The default of 1 slides one beat at a time, so merged samples
            share beats with their neighbours. Set it equal to
            `num_beats_to_merge` for strictly non-overlapping windows.
        limit_records (int, optional): Hard limit on the number of patients to process (useful for fast debugging).
        cleanup_zip (bool): If True, deletes the downloaded zip file after extraction.
        **preprocessing_params: kwargs passed directly to the Preprocessing class.
    """
    # PTB-XL distributes every recording twice. The rate follows from which
    # copy is read rather than being a free choice, so it lives here instead of
    # in config.yaml, and each header is checked against it.
    SAMPLING_RATES = {
        "high": 500,
        "low": 100,
    }

    def __init__(self, leads=['i'], resolution='high', only_healthy=False,
                 data_split_mode="all-available", num_beats_to_merge=1,
                 beat_merge_method="average", beat_merge_stride=1,
                 limit_records=None,
                 train_record_indices=None, enroll_record_indices=None,
                 probe_record_indices=None,
                 cleanup_zip=False, preprocessing_config=None,
                 **preprocessing_params):

        self.preprocessor = Preprocessing()
        self.cfg = CONFIG["datasets"]["ptbxl"]
        project_dir = Path(__file__).resolve().parent
        self.data_root = (project_dir / CONFIG["project"]["data_root"]).resolve()
        self.dataset_root = self.data_root / self.cfg["root_dir"]
        self.zip_path = self.data_root / self.cfg["zip_name"]
        self.url = self.cfg["url"]
        
        self.prep_params = _normalize_preprocessing_config(
            self.cfg.get("preprocessing", {}),
            preprocessing_config,
            preprocessing_params,
        )
        self.target_leads = [l.lower() for l in leads] if isinstance(leads, list) else leads

        if resolution not in self.SAMPLING_RATES:
            raise ValueError(
                f"resolution must be one of {sorted(self.SAMPLING_RATES)}, "
                f"not {resolution!r}."
            )

        self.resolution = resolution
        self.only_healthy = only_healthy
        self.num_beats = num_beats_to_merge
        self.merge_strategy = beat_merge_method
        self.beat_merge_stride = _normalize_beat_merge_stride(
            beat_merge_stride,
            num_beats_to_merge,
        )
        self.limit_records = limit_records
        self.cleanup_zip = cleanup_zip
        
        valid_modes = list(RECORD_BASED_SPLIT_MODES)
        if data_split_mode not in valid_modes:
            raise ValueError(f"Invalid mode: {data_split_mode}. Use {valid_modes}")
        self.data_split_mode = data_split_mode
        _configure_record_role_indices(
            self,
            data_split_mode,
            train_record_indices,
            enroll_record_indices,
            probe_record_indices,
        )
        # Recorded so that the temporal-date policy reaches the cache
        # identity: a cache built under a different policy describes a
        # different partition and must not be reused.
        self.temporal_date_policy = TEMPORAL_DATE_POLICY

    def download(self):
        """Downloads and extracts the dataset if not already present."""
        _download_and_extract(self.url, self.zip_path, self.dataset_root, "PTB-XL", cleanup=self.cleanup_zip)

    def _get_lead_indices(self, available_leads):
        """Maps requested lead names (e.g., 'i', 'v5') to channel indices."""
        avail_norm = [l.lower().strip() for l in available_leads]
        if self.target_leads == 'all': return list(range(len(available_leads)))
        indices = []
        for req in self.target_leads:
            req = req.strip().lower()
            if req in avail_norm: indices.append(avail_norm.index(req))
            else:
                for i, avail in enumerate(avail_norm):
                    if req == avail: indices.append(i); break
        return indices

    def _is_healthy(self, scp_codes_str):
        """
        Checks if the 'NORM' (Normal ECG) superclass is present in the diagnostic codes.
        PTB-XL stores these as stringified dictionaries (e.g., "{'NORM': 100.0, ...}").
        """
        return "NORM" in str(scp_codes_str)

    def load_raw_data(self, metadata_only=False):
        """
        Parses the official ptbxl_database.csv, loads WFDB records, and groups by patient.
        Ensures strict chronological sorting using official metadata timestamps.

        Args:
            metadata_only (bool): If True, skips signal reading and returns
                only the record identity and date of each recording. The
                grouping, ordering, and eligibility logic is identical, so a
                metadata-only pass reports the same partition assignment the
                full pipeline produces.
        """
        if not self.dataset_root.exists() or not any(self.dataset_root.iterdir()):
            self.download()
        
        csv_path = self.dataset_root / 'ptbxl_database.csv'
        if not csv_path.exists(): raise FileNotFoundError(f"Database CSV not found at {csv_path}")
        
        df = pd.read_csv(csv_path, index_col='ecg_id')
        df['patient_id'] = df['patient_id'].astype(int)
        
        recordings = {}
        unique_patients = df['patient_id'].unique()
        if self.limit_records: 
            unique_patients = unique_patients[:self.limit_records]
        
        fname_col = 'filename_hr' if self.resolution == 'high' else 'filename_lr'

        for pid in tqdm(unique_patients, desc="Loading PTB-XL raw files"):
            # Sort chronologically using precise metadata timestamps
            patient_recs = df[df['patient_id'] == pid].sort_values(by='recording_date')
            
            # --- Healthy Control Check ---
            # We evaluate the baseline (first) recording to determine subject eligibility
            if self.only_healthy:
                baseline_codes = patient_recs.iloc[0]['scp_codes']
                if not self._is_healthy(baseline_codes):
                    continue

            recs_list = []
            for ecg_id, row in patient_recs.iterrows():
                fname_rel = row[fname_col]
                full_path = self.dataset_root / fname_rel
                
                if not (full_path.parent / (full_path.name + ".hea")).exists(): continue

                try:
                    rec_header = wfdb.rdheader(str(full_path))

                    verify_sampling_rate(
                        self.SAMPLING_RATES[self.resolution],
                        "PTB-XL",
                        fname_rel,
                        rec_header.fs,
                    )

                    lead_indices = self._get_lead_indices(rec_header.sig_name)
                    if not lead_indices: continue

                    # Store exact datetime for Day 1 splits
                    rec_dt = pd.to_datetime(row['recording_date']).date()

                    if metadata_only:
                        recs_list.append({
                            "fs": rec_header.fs,
                            "date": rec_dt,
                            "filename": str(fname_rel),
                        })
                        continue

                    data, _ = wfdb.rdsamp(str(full_path), channels=lead_indices)

                    recs_list.append({
                        "signal": data,
                        "fs": rec_header.fs,
                        "date": rec_dt,
                        "filename": str(fname_rel),
                    })
                except ValueError:
                    # Raised deliberately when a recording contradicts the
                    # dataset configuration, which must stop the run rather
                    # than quietly reduce the cohort.
                    raise
                except Exception as read_error:
                    # With 21,799 records a silent skip is invisible, so report
                    # which recording was dropped and why.
                    print(
                        f"[WARN] PTB-XL: skipping {fname_rel} for patient "
                        f"{pid}: {read_error}"
                    )

            if recs_list: recordings[str(pid)] = recs_list
            
        return recordings

    def _process_signal(self, sig, fs):
        """Applies filters, segmentation, and multi-beat merging."""
        n_channels = sig.shape[1]
        processed_channels = []
        for c in range(n_channels):
            processed_channels.append(_preprocess_signal(
                self.preprocessor,
                sig[:, c],
                fs,
                self.prep_params,
            ))
        if not processed_channels: return np.empty((0, n_channels, 0))
        min_len = min([len(ch) for ch in processed_channels])
        if min_len == 0: return np.empty((0, n_channels, 0))
        
        beats_multi = np.stack([ch[:min_len] for ch in processed_channels], axis=1)
        if self.num_beats == 1: return beats_multi[:, 0, :] if n_channels == 1 else beats_multi
        if len(beats_multi) < self.num_beats: return np.empty((0, n_channels, 0))
        
        merged_samples = []
        merge_starts = _beat_merge_start_indices(
            len(beats_multi),
            self.num_beats,
            self.beat_merge_stride,
        )
        for i in merge_starts:
            group = beats_multi[i : i + self.num_beats]
            if self.merge_strategy == "average":
                merged = np.mean(group, axis=0)
                if n_channels == 1: merged = merged.squeeze(0)
                merged_samples.append(merged)
            elif self.merge_strategy == "concat":
                merged = group.transpose(1, 0, 2).reshape(n_channels, -1)
                if n_channels == 1: merged = merged.squeeze(0)
                merged_samples.append(merged)
        return np.array(merged_samples)

    def load_all_data(self, return_provenance=False):
        """
        Loads dataset for tasks that handle train/test splitting downstream.
        Applies to 'all-available' and 'single-session'.
        """
        if self.data_split_mode not in ["all-available", "single-session"]:
            print(f"[WARN] Calling load_all_data() but mode is '{self.data_split_mode}'.")
            
        data = self.load_raw_data()
        x_list, y_list = [], []
        provenance_builder = _ProvenanceBuilder() if return_provenance else None
        
        for sid, recs in tqdm(data.items(), desc="Processing signals"):
            if not recs: continue
            
            target_recs = recs if self.data_split_mode == "all-available" else [recs[0]]
            
            for acquisition_order, rec in enumerate(target_recs):
                segments = self._process_signal(rec['signal'], rec['fs'])
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
                    if provenance_builder is not None:
                        provenance_builder.add_block(
                            len(segments),
                            record_id=rec['filename'],
                            session_id=rec['filename'],
                            acquisition_time=None,
                            acquisition_order=acquisition_order,
                            source_segment_id=rec['filename'],
                            source_segment_order=float(acquisition_order),
                        )
                    
        return _finalize_loader_output(x_list, y_list, provenance_builder, return_provenance)

    def load_session(self, session_name, return_provenance=False):
        """
        Loads the partitioned data strictly based on temporal/record boundaries.
        No intra-record splitting is permitted to preserve complete 10s windows.
        """
        role = _canonical_record_role(session_name)
        is_enrollment = role != "probe"
        log_role = (
            role
            if self.data_split_mode == CUSTOM_RECORD_SPLIT_MODE
            else is_enrollment
        )
        log_name = {
            "train": "Training",
            "enrollment": "Enrollment",
            "probe": "Test/Probe",
        }[role]

        data = self.load_raw_data()
        x_list, y_list = [], []
        provenance_builder = _ProvenanceBuilder() if return_provenance else None
        
        kept_subjects, dropped_subjects = 0, 0

        for sid, recs in tqdm(data.items(), desc=f"Processing {log_name}"):
            if not recs: continue
            
            (
                target_recs,
                subject_is_eligible,
            ) = select_record_role_partition(
                recs,
                self.data_split_mode,
                role,
                train_record_indices=getattr(
                    self,
                    "train_record_indices",
                    None,
                ),
                enroll_record_indices=getattr(
                    self,
                    "enroll_record_indices",
                    None,
                ),
                probe_record_indices=getattr(
                    self,
                    "probe_record_indices",
                    None,
                ),
            )

            if not subject_is_eligible:
                dropped_subjects += 1
                continue

            kept_subjects += 1

            _record_partition_assignment(
                self,
                log_role,
                sid,
                target_recs,
            )

            # --- EXTRACTION & SIGNAL PROCESSING ---
            for acquisition_order, rec in enumerate(target_recs):
                segments = self._process_signal(rec['signal'], rec['fs'])
                
                if len(segments) > 0:
                    x_list.append(segments)
                    y_list.extend([sid] * len(segments))
                    if provenance_builder is not None:
                        provenance_builder.add_block(
                            len(segments),
                            record_id=rec['filename'],
                            session_id=rec['filename'],
                            acquisition_time=None,
                            acquisition_order=acquisition_order,
                            source_segment_id=rec['filename'],
                            source_segment_order=float(acquisition_order),
                        )

        # Dynamic summary print during the enrollment pass
        if self.data_split_mode not in ["all-available", "single-session"] and is_enrollment:
            mode_title = self.data_split_mode.replace('-', ' ').title()
            print(f"\n[INFO] {mode_title} Summary: Kept {kept_subjects} subjects. Dropped {dropped_subjects} subjects.")

        return _finalize_loader_output(x_list, y_list, provenance_builder, return_provenance)
# =============================================================================