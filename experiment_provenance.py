"""Canonical experiment identities and exact structured-result selection."""

from __future__ import annotations

import enum
import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral, Real
from pathlib import Path
from typing import Callable, Optional

from artifact_provenance import (
    ImplementationSourceError,
    build_implementation_group,
    canonical_json_bytes,
    collect_creation_provenance,
)


SCIENTIFIC_CONFIGURATION_SCHEMA = (
    "ecg-biometrics/scientific-configuration-v1"
)
IMPLEMENTATION_IDENTITY_SCHEMA = (
    "ecg-biometrics/experiment-implementation-v1"
)
RESULT_PROVENANCE_SCHEMA = (
    "ecg-biometrics/experiment-result-provenance-v1"
)
RECORD_LOCATOR_HASH_FORMAT = (
    "ecg-biometrics/canonical-json-record-sha256"
)


EXPERIMENT_IMPLEMENTATION_MODULES = (
    ("load_dataset", "load_dataset"),
    ("preprocessing", "preprocessing"),
    ("filtering", "filtering"),
    ("representation", "representation"),
    ("models", "models"),
    ("run", "run"),
    ("utils", "utils"),
    ("data_augmentation", "data_augmentation"),
    ("main", "main"),
    ("artifact_provenance", "artifact_provenance"),
    ("experiment_provenance", "experiment_provenance"),
)


EFFECTIVE_CONFIGURATION_ADMINISTRATIVE_FIELDS = frozenset(
    {
        "cache_dir",
        "campaign_id",
        "campaign_label",
        "config",
        "experiment_time",
        "explicitly_configured_arguments",
        "hostname",
        "intelligent_data_loading",
        "intelligent_weight_loading",
        "num_pairs",
        "log_path",
        "output_dir",
        "record_locator",
        "results_dir",
        "sampling_mode",
        "save_results",
        "smoke_run",
        "timestamp",
        "visualize",
    }
)


EFFECTIVE_CONFIGURATION_SCIENTIFIC_FIELDS = frozenset(
    {
        "augmentation_copies",
        "augmentation_method",
        "augmentation_parameters",
        "batch_size",
        "beat_merge_stride",
        "data_split_mode",
        "dataset",
        "device",
        "electrode_unit",
        "enrol_parts",
        "enroll_record_indices",
        "enroll_sessions",
        "enrollment_template_mode",
        "epochs",
        "lr",
        "matching_method",
        "max_impostor_pairs",
        "model",
        "n_runs",
        "num_beats_to_merge",
        "num_templates_per_identity",
        "outlier_filtering_on_test",
        "outlier_filtering_on_train",
        "pair_sampling_budget",
        "pair_sampling_mode",
        "pair_sampling_seed",
        "preprocessing_parameters",
        "probe_fusion_size",
        "probe_record_indices",
        "probe_sessions",
        "reproducibility_mode",
        "seed",
        "session_for_single_session_evaluation",
        "signal_type",
        "single_segment_range",
        "split_seed",
        "sqi_keep_pct",
        "sqi_method",
        "sqi_threshold",
        "target_fars",
        "task",
        "template_fusion_method",
        "template_score_aggregation",
        "template_selection_method",
        "template_size",
        "temporal_guard_minutes",
        "test_parts",
        "test_split",
        "train_parts",
        "train_record_indices",
        "train_sessions",
        "use_augmentation",
        "use_deployment_evaluation",
        "use_template",
        "val_split",
    }
)


RUNNER_ADMINISTRATIVE_PARAMETERS = frozenset(
    {
        "_return_stats",
        "intelligent_weight_loading",
        "loader",
        "save_results_and_settings",
        "visualize",
    }
)


RUNNER_DATA_PARAMETERS = frozenset(
    {
        "provenance",
        "provenance_enroll",
        "provenance_s1",
        "provenance_s2",
        "sqi_scores",
        "sqi_s1",
        "sqi_s2",
        "sqi_test",
        "sqi_train",
        "x",
        "x_enroll",
        "x_s1",
        "x_s2",
        "x_test",
        "x_train",
        "y",
        "y_enroll",
        "y_s1",
        "y_s2",
        "y_test",
        "y_train",
    }
)


RUNNER_SCIENTIFIC_PARAMETERS = frozenset(
    {
        "augmentation_config",
        "batch_size",
        "device",
        "enrollment_template_mode",
        "epochs",
        "lr",
        "matching_method",
        "max_impostor_pairs",
        "model_class",
        "n_runs",
        "num_pairs",
        "num_templates_per_identity",
        "outlier_filtering_on_test",
        "outlier_filtering_on_train",
        "pair_sampling_budget",
        "pair_sampling_mode",
        "pair_sampling_seed",
        "probe_fusion_size",
        "reproducibility_mode",
        "sampling_mode",
        "seed",
        "split_seed",
        "sqi_keep_pct",
        "sqi_threshold",
        "target_fars",
        "template_fusion_method",
        "template_score_aggregation",
        "template_selection_method",
        "template_size",
        "test_split",
        "use_deployment_evaluation",
        "use_template",
        "val_split",
    }
)


# Every name in RUNNER_SCIENTIFIC_PARAMETERS that is not itself an effective-
# configuration scientific field name is a value main.py derives before
# dispatch (a runtime object such as a model class) or a legacy call-time
# alias, rather than a canonical field read directly off resolved arguments.
# This mapping states, explicitly, which effective-configuration scientific
# field(s) already represent each such parameter's scientific semantics, so
# verify_runner_scientific_parameters_are_represented can detect a future
# scientific runner parameter that was classified without ever being wired
# to canonical identity.
RUNNER_DERIVED_SCIENTIFIC_PARAMETER_SOURCES = {
    "model_class": ("model",),
    "augmentation_config": (
        "use_augmentation",
        "augmentation_method",
        "augmentation_copies",
        "augmentation_parameters",
    ),
    "num_pairs": ("pair_sampling_budget",),
    "sampling_mode": ("pair_sampling_mode",),
}


def verify_runner_scientific_parameters_are_represented():
    """Return the scientific runner parameters with no identity source.

    A scientific runner parameter is represented either because it is
    itself a classified effective-configuration scientific field (same
    name), or because it is listed in
    ``RUNNER_DERIVED_SCIENTIFIC_PARAMETER_SOURCES`` with every declared
    source field also classified as scientific. Anything else is an
    omission: a scientific runner parameter whose value is not provably
    covered by the canonical scientific configuration identity.
    """
    unrepresented = []
    for name in sorted(RUNNER_SCIENTIFIC_PARAMETERS):
        if name in EFFECTIVE_CONFIGURATION_SCIENTIFIC_FIELDS:
            continue
        sources = RUNNER_DERIVED_SCIENTIFIC_PARAMETER_SOURCES.get(name)
        if sources and all(
            source in EFFECTIVE_CONFIGURATION_SCIENTIFIC_FIELDS
            for source in sources
        ):
            continue
        unrepresented.append(name)
    return unrepresented


class ProvenanceError(RuntimeError):
    """Base class for experiment and result provenance failures."""


class CanonicalConfigurationError(ProvenanceError, ValueError):
    """Raised when effective configuration values are not canonicalizable."""


class ResultCollectionError(ProvenanceError):
    """Raised when an exact intended structured result cannot be proven."""


def _canonicalize_identity_value(value, location="configuration"):
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, Integral):
        return int(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalConfigurationError(
                f"{location} must not contain NaN or infinity."
            )
        return 0.0 if value == 0.0 else value

    if isinstance(value, Real):
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise CanonicalConfigurationError(
                f"{location} must not contain NaN or infinity."
            )
        return 0.0 if numeric_value == 0.0 else numeric_value

    if isinstance(value, enum.Enum):
        return _canonicalize_identity_value(
            value.value,
            location=location,
        )

    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalConfigurationError(
                    f"{location} contains a non-string mapping key: {key!r}."
                )
            normalized[key] = _canonicalize_identity_value(
                item,
                location=f"{location}.{key}",
            )
        return normalized

    if isinstance(value, (list, tuple)):
        return [
            _canonicalize_identity_value(
                item,
                location=f"{location}[{index}]",
            )
            for index, item in enumerate(value)
        ]

    raise CanonicalConfigurationError(
        f"{location} contains unsupported canonical value type "
        f"{type(value).__name__}."
    )


def classify_effective_configuration_fields(configuration):
    """Fail when a resolved configuration field has no identity policy."""
    if not isinstance(configuration, Mapping):
        raise CanonicalConfigurationError(
            "Effective experiment configuration must be a mapping."
        )

    classified = (
        EFFECTIVE_CONFIGURATION_SCIENTIFIC_FIELDS
        | EFFECTIVE_CONFIGURATION_ADMINISTRATIVE_FIELDS
    )
    unknown = sorted(set(configuration) - classified)
    if unknown:
        raise CanonicalConfigurationError(
            "Unclassified effective configuration field(s): "
            + ", ".join(unknown)
        )


def _normalize_seed_schedule(configuration):
    normalized = dict(configuration)
    if not {"seed", "n_runs", "split_seed"}.issubset(normalized):
        return normalized

    base_seed = int(normalized["seed"])
    run_count = int(normalized["n_runs"])
    if run_count < 1:
        raise CanonicalConfigurationError(
            "Resolved n_runs must be greater than or equal to one."
        )

    run_seeds = [base_seed + index for index in range(run_count)]
    configured_split_seed = normalized.pop("split_seed")
    if configured_split_seed is None:
        resolved_split_seeds = list(run_seeds)
    else:
        resolved_split_seeds = [int(configured_split_seed)] * run_count

    normalized["run_seeds"] = run_seeds
    normalized["resolved_split_seeds"] = resolved_split_seeds
    return normalized


def _normalize_inactive_settings(configuration):
    """Remove resolved settings that the selected execution branch ignores."""
    normalized = dict(configuration)

    if normalized.get("use_augmentation") is False:
        for field_name in (
            "augmentation_method",
            "augmentation_copies",
            "augmentation_parameters",
        ):
            normalized.pop(field_name, None)

    if not (
        normalized.get("outlier_filtering_on_train")
        or normalized.get("outlier_filtering_on_test")
    ):
        for field_name in (
            "sqi_method",
            "sqi_threshold",
            "sqi_keep_pct",
        ):
            normalized.pop(field_name, None)

    task = normalized.get("task")
    use_template = bool(normalized.get("use_template"))
    if not use_template:
        for field_name in (
            "enrollment_template_mode",
            "num_templates_per_identity",
            "template_fusion_method",
            "template_score_aggregation",
            "template_selection_method",
            "template_size",
        ):
            normalized.pop(field_name, None)
        if task in {1, 3, 5, 7}:
            normalized.pop("matching_method", None)

    enrollment_consumed = task == 7 or (
        task in {5, 6, 8} and use_template
    )
    if not enrollment_consumed:
        for field_name in (
            "enrol_parts",
            "enroll_record_indices",
            "enroll_sessions",
        ):
            normalized.pop(field_name, None)

    return normalized


DECLARED_REAL_SCIENTIFIC_FIELDS = frozenset(
    {
        "lr",
        "test_split",
        "val_split",
        "sqi_threshold",
        "sqi_keep_pct",
    }
)


def _normalize_declared_real_fields(configuration):
    """Coerce declared real-valued scientific fields to float.

    Each of these fields is always semantically a real number (a learning
    rate, a split fraction, a quality threshold): an integer-valued literal
    such as ``val_split: 0`` and its float form ``val_split: 0.0`` describe
    the same scientific configuration, and YAML preserves whichever literal
    type was written. Only these explicitly declared-real fields are
    touched -- this does not generalize to arbitrary nested preprocessing
    values, where integer and float can carry distinct meaning. Booleans
    are left untouched; a declared-real field is never legitimately boolean.
    """
    normalized = dict(configuration)
    for field_name in DECLARED_REAL_SCIENTIFIC_FIELDS:
        if field_name not in normalized:
            continue
        value = normalized[field_name]
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, Integral, float, Real)):
            normalized[field_name] = float(value)
    return normalized


def build_scientific_configuration_identity(
    effective_configuration,
    *,
    authoritative=True,
):
    """Return strict canonical scientific configuration content and digest."""
    if authoritative:
        classify_effective_configuration_fields(effective_configuration)

    scientific = {
        key: value
        for key, value in effective_configuration.items()
        if key not in EFFECTIVE_CONFIGURATION_ADMINISTRATIVE_FIELDS
    }
    scientific = _normalize_inactive_settings(scientific)
    scientific = _normalize_seed_schedule(scientific)
    scientific = _normalize_declared_real_fields(scientific)
    canonical_configuration = _canonicalize_identity_value(scientific)
    identity_payload = {
        "schema": SCIENTIFIC_CONFIGURATION_SCHEMA,
        "configuration": canonical_configuration,
    }
    return {
        **identity_payload,
        "sha256": hashlib.sha256(
            canonical_json_bytes(identity_payload)
        ).hexdigest(),
        "authoritative": bool(authoritative),
    }


@lru_cache(maxsize=1)
def _default_experiment_source_identity():
    return build_implementation_group(
        "experiment_execution",
        EXPERIMENT_IMPLEMENTATION_MODULES,
    )


def build_experiment_implementation_identity(
    *,
    source_reader: Optional[Callable[[str], bytes]] = None,
    repository_root: Optional[os.PathLike] = None,
    git_runner=None,
):
    """Return deterministic source identity plus separate repository state."""
    if source_reader is None:
        source_identity = _default_experiment_source_identity()
    else:
        source_identity = build_implementation_group(
            "experiment_execution",
            EXPERIMENT_IMPLEMENTATION_MODULES,
            source_reader=source_reader,
        )

    creation = collect_creation_provenance(
        repository_root=repository_root,
        git_runner=git_runner,
    )
    return {
        "schema": IMPLEMENTATION_IDENTITY_SCHEMA,
        "source": source_identity,
        "source_sha256": source_identity["aggregate_sha256"],
        "git": creation["git"],
    }


def _normalize_seed_list(value):
    if not isinstance(value, (list, tuple)):
        return None
    try:
        seeds = [int(seed) for seed in value]
    except (TypeError, ValueError):
        return None
    return seeds


def _build_completion(hyperparameters, per_run_results):
    expected_count = hyperparameters.get("n_runs", 1)
    try:
        expected_count = int(expected_count)
    except (TypeError, ValueError):
        expected_count = 0

    expected_seeds = _normalize_seed_list(
        hyperparameters.get("run_seeds")
    )
    expected_split_seeds = _normalize_seed_list(
        hyperparameters.get("resolved_split_seeds")
    )
    if expected_seeds is None and expected_count == 1:
        try:
            expected_seeds = [int(hyperparameters["base_seed"])]
        except (KeyError, TypeError, ValueError):
            expected_seeds = None

    run_records = list(per_run_results or [])
    completed_indices = []
    completed_seeds = []
    completed_split_seeds = []
    completion_source = "per_run_results"

    if run_records:
        for record in run_records:
            if not isinstance(record, Mapping):
                completed_indices = []
                completed_seeds = []
                break
            try:
                completed_indices.append(int(record["run_index"]))
                completed_seeds.append(int(record["seed"]))
                if expected_split_seeds is not None:
                    completed_split_seeds.append(int(record["split_seed"]))
            except (KeyError, TypeError, ValueError):
                completed_indices = []
                completed_seeds = []
                completed_split_seeds = []
                break
    elif expected_count == 1 and expected_seeds is not None:
        completed_indices = [1]
        completed_seeds = list(expected_seeds)
        if expected_split_seeds is not None:
            completed_split_seeds = list(expected_split_seeds)
        completion_source = "successful_single_run_record"

    complete = (
        expected_count >= 1
        and expected_seeds is not None
        and len(expected_seeds) == expected_count
        and completed_indices == list(range(1, expected_count + 1))
        and completed_seeds == expected_seeds
        and (
            expected_split_seeds is None
            or (
                len(expected_split_seeds) == expected_count
                and completed_split_seeds == expected_split_seeds
            )
        )
    )
    return {
        "expected": {
            "run_count": expected_count,
            "run_seeds": expected_seeds,
            "resolved_split_seeds": expected_split_seeds,
        },
        "completed": {
            "run_count": len(completed_seeds),
            "run_seeds": completed_seeds,
            "resolved_split_seeds": completed_split_seeds,
            "per_run_indices": completed_indices,
            "source": completion_source,
        },
        "complete": complete,
    }


def build_result_record_provenance(
    *,
    effective_configuration,
    configuration_authoritative,
    implementation_identity,
    campaign_id,
    smoke_run,
    hyperparameters,
    per_run_results,
):
    """Build machine-verifiable provenance for one successful result record."""
    scientific_identity = build_scientific_configuration_identity(
        effective_configuration,
        authoritative=configuration_authoritative,
    )
    completion = _build_completion(
        hyperparameters,
        per_run_results,
    )
    reasons = []
    if not configuration_authoritative:
        reasons.append("authoritative_effective_configuration_unavailable")
    if smoke_run:
        reasons.append("smoke_run")
    if not completion["complete"]:
        reasons.append("incomplete_run_schedule")

    git_identity = implementation_identity.get("git", {})
    if git_identity.get("status") != "available":
        reasons.append("source_revision_unavailable")
    elif git_identity.get("dirty") is not False:
        reasons.append("dirty_source_tree")

    return {
        "schema": RESULT_PROVENANCE_SCHEMA,
        "scientific_configuration": scientific_identity,
        "implementation": implementation_identity,
        "execution": {
            "campaign_id": (
                None if campaign_id is None else str(campaign_id)
            ),
            "smoke_run": bool(smoke_run),
            "status": "success",
            "successful": True,
            "completion": completion,
        },
        "publication_eligibility": {
            "eligible": not reasons,
            "reasons": reasons,
        },
    }


@dataclass(frozen=True)
class ResultLogSnapshot:
    path: Path
    existed: bool
    size_bytes: int
    prefix_sha256: str
    file_identity: Optional[tuple]
    ended_with_newline: bool


def capture_result_log_snapshot(path):
    """Capture the exact byte prefix and file identity before execution."""
    path = Path(path)
    if not path.exists():
        return ResultLogSnapshot(
            path=path,
            existed=False,
            size_bytes=0,
            prefix_sha256=hashlib.sha256(b"").hexdigest(),
            file_identity=None,
            ended_with_newline=True,
        )
    if not path.is_file():
        raise ResultCollectionError(
            f"Structured result path is not a regular file: {path}"
        )

    content = path.read_bytes()
    stat = path.stat()
    return ResultLogSnapshot(
        path=path,
        existed=True,
        size_bytes=len(content),
        prefix_sha256=hashlib.sha256(content).hexdigest(),
        file_identity=(stat.st_dev, stat.st_ino),
        ended_with_newline=(not content or content.endswith(b"\n")),
    )


def _parse_jsonl_bytes(content, path):
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ResultCollectionError(
            f"Structured result data is not valid UTF-8: {path}"
        ) from error

    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ResultCollectionError(
                f"Malformed structured result at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(record, Mapping):
            raise ResultCollectionError(
                f"Structured result at {path}:{line_number} must be an object."
            )
        records.append((line_number, dict(record)))
    return records


def _record_digest(record):
    return hashlib.sha256(canonical_json_bytes(record)).hexdigest()


def _build_record_locator(path, result_root, line_number, record):
    path = Path(path).resolve()
    result_root = Path(result_root).resolve()
    try:
        relative_path = path.relative_to(result_root)
    except ValueError as error:
        raise ResultCollectionError(
            "Structured result path is outside the declared result root."
        ) from error
    return {
        "relative_path": relative_path.as_posix(),
        "line_number": int(line_number),
        "line_number_base": 1,
        "record_hash_format": RECORD_LOCATOR_HASH_FORMAT,
        "record_sha256": _record_digest(record),
    }


def validate_result_record(
    record,
    *,
    expected_scientific_sha256,
    expected_implementation=None,
    required_campaign_id=None,
    publication_mode=True,
    expected_smoke_run=False,
):
    """Validate that one record is the intended completed execution."""
    provenance = record.get("canonical_provenance")
    if not isinstance(provenance, Mapping):
        raise ResultCollectionError(
            "Structured result lacks canonical experiment provenance."
        )
    if provenance.get("schema") != RESULT_PROVENANCE_SCHEMA:
        raise ResultCollectionError("Unsupported result provenance schema.")

    scientific = provenance.get("scientific_configuration")
    if not isinstance(scientific, Mapping) or (
        scientific.get("sha256") != expected_scientific_sha256
    ):
        raise ResultCollectionError(
            "Structured result scientific configuration identity mismatches."
        )

    implementation = provenance.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ResultCollectionError(
            "Structured result implementation identity is missing."
        )
    if expected_implementation is not None:
        if implementation.get("source_sha256") != expected_implementation.get(
            "source_sha256"
        ):
            raise ResultCollectionError(
                "Structured result implementation source identity mismatches."
            )
        expected_commit = expected_implementation.get("git", {}).get("commit")
        if expected_commit not in (None, "unavailable") and (
            implementation.get("git", {}).get("commit") != expected_commit
        ):
            raise ResultCollectionError(
                "Structured result source commit mismatches."
            )

    execution = provenance.get("execution")
    if not isinstance(execution, Mapping):
        raise ResultCollectionError("Execution provenance is missing.")
    if execution.get("successful") is not True or (
        execution.get("status") != "success"
    ):
        raise ResultCollectionError("Structured result is unsuccessful.")
    completion = execution.get("completion")
    if not isinstance(completion, Mapping) or completion.get("complete") is not True:
        raise ResultCollectionError("Structured result has incomplete runs.")
    if bool(execution.get("smoke_run")) != bool(expected_smoke_run):
        raise ResultCollectionError("Structured result smoke status mismatches.")
    if required_campaign_id is not None and (
        execution.get("campaign_id") != str(required_campaign_id)
    ):
        raise ResultCollectionError("Structured result campaign identity mismatches.")

    if publication_mode:
        eligibility = provenance.get("publication_eligibility")
        if not isinstance(eligibility, Mapping) or (
            eligibility.get("eligible") is not True
        ):
            reasons = (
                eligibility.get("reasons")
                if isinstance(eligibility, Mapping)
                else ["missing_eligibility"]
            )
            raise ResultCollectionError(
                "Structured result is not publication eligible: "
                + ", ".join(str(reason) for reason in reasons)
            )
        if execution.get("smoke_run") is not False:
            raise ResultCollectionError(
                "Smoke results cannot be selected for publication output."
            )

        # The stored eligibility flag is derived metadata, not independent
        # evidence: re-check the underlying source-cleanliness evidence in
        # this same record rather than trusting that it was set correctly.
        implementation_git = implementation.get("git")
        if not isinstance(implementation_git, Mapping) or (
            implementation_git.get("status") != "available"
        ):
            raise ResultCollectionError(
                "Structured result implementation source revision is not "
                "available; publication requires a resolvable Git state."
            )
        if implementation_git.get("dirty") is not False:
            raise ResultCollectionError(
                "Structured result implementation source tree was dirty; "
                "publication requires a clean source revision."
            )

    return dict(record)


def collect_appended_result(
    path,
    snapshot,
    *,
    result_root,
    expected_scientific_sha256,
    expected_implementation=None,
    required_campaign_id=None,
    publication_mode=True,
    expected_smoke_run=False,
):
    """Require exactly one new record appended to the captured file prefix."""
    path = Path(path)
    if path.resolve() != snapshot.path.resolve():
        raise ResultCollectionError("Snapshot path does not match result path.")
    if not path.is_file():
        raise ResultCollectionError("Expected structured result file was not created.")

    content = path.read_bytes()
    if len(content) < snapshot.size_bytes:
        raise ResultCollectionError("Structured result file was truncated.")
    prefix = content[:snapshot.size_bytes]
    if hashlib.sha256(prefix).hexdigest() != snapshot.prefix_sha256:
        raise ResultCollectionError(
            "Structured result file no longer has the captured prefix."
        )
    if snapshot.existed:
        stat = path.stat()
        if (stat.st_dev, stat.st_ino) != snapshot.file_identity:
            raise ResultCollectionError("Structured result file was replaced.")
    if snapshot.size_bytes and not snapshot.ended_with_newline:
        raise ResultCollectionError(
            "Captured structured result did not end at a line boundary."
        )

    appended = content[snapshot.size_bytes:]
    records = _parse_jsonl_bytes(appended, path)
    if len(records) != 1:
        raise ResultCollectionError(
            "Execution must append exactly one structured result record; "
            f"observed {len(records)}."
        )
    relative_line_number, record = records[0]
    prefix_line_count = prefix.count(b"\n")
    line_number = prefix_line_count + relative_line_number
    validated = validate_result_record(
        record,
        expected_scientific_sha256=expected_scientific_sha256,
        expected_implementation=expected_implementation,
        required_campaign_id=required_campaign_id,
        publication_mode=publication_mode,
        expected_smoke_run=expected_smoke_run,
    )
    return {
        "record": validated,
        "locator": _build_record_locator(
            path,
            result_root,
            line_number,
            validated,
        ),
    }


def select_exact_result_record(
    path,
    *,
    result_root,
    expected_scientific_sha256,
    expected_implementation=None,
    required_campaign_id=None,
    publication_mode=True,
    expected_smoke_run=False,
):
    """Select exactly one matching record from an existing structured log."""
    path = Path(path)
    if not path.is_file():
        raise ResultCollectionError(f"Structured result file does not exist: {path}")
    records = _parse_jsonl_bytes(path.read_bytes(), path)
    matches = []
    diagnostics = []
    for line_number, record in records:
        try:
            validated = validate_result_record(
                record,
                expected_scientific_sha256=expected_scientific_sha256,
                expected_implementation=expected_implementation,
                required_campaign_id=required_campaign_id,
                publication_mode=publication_mode,
                expected_smoke_run=expected_smoke_run,
            )
        except ResultCollectionError as error:
            diagnostics.append(str(error))
            continue
        matches.append((line_number, validated))

    if len(matches) != 1:
        detail = "; ".join(sorted(set(diagnostics))[:3])
        raise ResultCollectionError(
            "Expected exactly one matching structured result record; "
            f"observed {len(matches)}."
            + (f" Diagnostics: {detail}" if detail else "")
        )

    line_number, record = matches[0]
    return {
        "record": record,
        "locator": _build_record_locator(
            path,
            result_root,
            line_number,
            record,
        ),
    }


def read_legacy_latest_record(path):
    """Explicitly read the last valid record for exploratory compatibility."""
    path = Path(path)
    if not path.is_file():
        return None
    latest = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, Mapping):
            latest = dict(record)
    return latest
