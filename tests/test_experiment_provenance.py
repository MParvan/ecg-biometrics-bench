import copy
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

import experiment_provenance as provenance
import main
import run


def _configuration(**overrides):
    configuration = {
        "dataset": "ecgid",
        "task": 1,
        "model": "resnet1d",
        "epochs": 2,
        "seed": 42,
        "n_runs": 2,
        "split_seed": None,
        "results_dir": "results",
        "cache_dir": "cache",
        "save_results": True,
        "campaign_id": "campaign-a",
        "smoke_run": False,
    }
    configuration.update(overrides)
    return configuration


def _git_runner(dirty=False, commit="abc123"):
    outputs = {
        ("rev-parse", "HEAD"): commit,
        ("rev-parse", "--abbrev-ref", "HEAD"): "main",
        ("status", "--porcelain"): " M run.py" if dirty else "",
    }
    return lambda root, arguments: outputs[tuple(arguments)]


def _implementation(dirty=False, source_suffix=""):
    return provenance.build_experiment_implementation_identity(
        source_reader=lambda module: (
            f"MODULE = {module!r}\n{source_suffix}".encode("utf-8")
        ),
        git_runner=_git_runner(dirty=dirty),
    )


def _canonical_record(
    *,
    configuration=None,
    implementation=None,
    campaign_id="campaign-a",
    smoke_run=False,
    per_run_results=None,
):
    configuration = configuration or _configuration()
    implementation = implementation or _implementation()
    if per_run_results is None:
        per_run_results = [
            {
                "run_index": 1,
                "seed": 42,
                "split_seed": 42,
                "metrics": {"A": 0.8},
            },
            {
                "run_index": 2,
                "seed": 43,
                "split_seed": 43,
                "metrics": {"A": 0.9},
            },
        ]
    canonical = provenance.build_result_record_provenance(
        effective_configuration=configuration,
        configuration_authoritative=True,
        implementation_identity=implementation,
        campaign_id=campaign_id,
        smoke_run=smoke_run,
        hyperparameters={
            "n_runs": 2,
            "run_seeds": [42, 43],
            "resolved_split_seeds": [42, 43],
        },
        per_run_results=per_run_results,
    )
    return {
        "task": "Synthetic",
        "dataset": "ecgid",
        "per_run_results": per_run_results,
        "results": {"A": {"mean": 0.85, "std": 0.05}},
        "canonical_provenance": canonical,
    }


def _write_record(path, record, mode="w"):
    with Path(path).open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False))
        handle.write("\n")


def _collect(path, snapshot, record, **overrides):
    identity = record["canonical_provenance"]["scientific_configuration"]
    implementation = record["canonical_provenance"]["implementation"]
    arguments = {
        "result_root": Path(path).parent,
        "expected_scientific_sha256": identity["sha256"],
        "expected_implementation": implementation,
        "required_campaign_id": "campaign-a",
        "publication_mode": True,
        "expected_smoke_run": False,
    }
    arguments.update(overrides)
    return provenance.collect_appended_result(path, snapshot, **arguments)


def test_canonical_configuration_is_order_independent_and_strict():
    first = _configuration(
        augmentation_parameters={"sigma": 0.1, "copies": [1, 2]},
    )
    second = dict(reversed(list(first.items())))
    second["augmentation_parameters"] = {
        "copies": [1, 2],
        "sigma": 0.1,
    }
    assert provenance.build_scientific_configuration_identity(first)[
        "sha256"
    ] == provenance.build_scientific_configuration_identity(second)["sha256"]

    with pytest.raises(provenance.CanonicalConfigurationError, match="unsupported"):
        provenance.build_scientific_configuration_identity(
            _configuration(augmentation_parameters={"bad": object()})
        )
    with pytest.raises(provenance.CanonicalConfigurationError, match="NaN"):
        provenance.build_scientific_configuration_identity(
            _configuration(epochs=float("nan"))
        )
    with pytest.raises(provenance.CanonicalConfigurationError, match="infinity"):
        provenance.build_scientific_configuration_identity(
            _configuration(epochs=float("inf"))
        )


def test_administrative_metadata_does_not_change_scientific_identity(tmp_path):
    first = _configuration(
        results_dir=tmp_path / "first",
        cache_dir=tmp_path / "cache-a",
        campaign_id="campaign-a",
        experiment_time="2026-01-01T00:00:00",
        output_dir="one",
    )
    second = _configuration(
        results_dir=str(tmp_path / "second"),
        cache_dir=str(tmp_path / "cache-b"),
        campaign_id="campaign-b",
        experiment_time="2027-02-02T00:00:00",
        output_dir="two",
    )
    assert provenance.build_scientific_configuration_identity(first)[
        "sha256"
    ] == provenance.build_scientific_configuration_identity(second)["sha256"]


def test_result_affecting_change_changes_scientific_identity():
    baseline = provenance.build_scientific_configuration_identity(
        _configuration(epochs=2)
    )
    changed = provenance.build_scientific_configuration_identity(
        _configuration(epochs=3)
    )
    assert baseline["sha256"] != changed["sha256"]


def test_inactive_settings_do_not_change_effective_identity():
    first = _configuration(
        task=1,
        use_augmentation=False,
        augmentation_method="gaussian",
        augmentation_copies=1,
        augmentation_parameters={},
        use_template=False,
        template_fusion_method="mean",
        template_size=None,
        matching_method="cosine",
        outlier_filtering_on_train=False,
        outlier_filtering_on_test=False,
        sqi_method="kurtosis",
        sqi_threshold=0.05,
        sqi_keep_pct=0.8,
        enroll_sessions=["session-a"],
    )
    second = dict(first)
    second.update(
        {
            "augmentation_method": "cutout",
            "augmentation_copies": 9,
            "augmentation_parameters": {"width": 20},
            "template_fusion_method": "median",
            "template_size": 8,
            "matching_method": "euclidean",
            "sqi_method": "skewness",
            "sqi_threshold": 0.9,
            "sqi_keep_pct": 0.2,
            "enroll_sessions": ["session-b"],
        }
    )
    assert provenance.build_scientific_configuration_identity(first)[
        "sha256"
    ] == provenance.build_scientific_configuration_identity(second)["sha256"]

    enabled = dict(first)
    enabled.update(
        {
            "use_augmentation": True,
            "augmentation_method": "gaussian",
        }
    )
    changed_enabled = dict(enabled)
    changed_enabled["augmentation_method"] = "cutout"
    assert provenance.build_scientific_configuration_identity(enabled)[
        "sha256"
    ] != provenance.build_scientific_configuration_identity(changed_enabled)[
        "sha256"
    ]


def test_resolved_split_seed_semantics_normalize_equivalent_values():
    followed = provenance.build_scientific_configuration_identity(
        _configuration(n_runs=1, seed=42, split_seed=None)
    )
    fixed_equivalent = provenance.build_scientific_configuration_identity(
        _configuration(n_runs=1, seed=42, split_seed=42)
    )
    materially_different = provenance.build_scientific_configuration_identity(
        _configuration(n_runs=1, seed=42, split_seed=123)
    )
    assert followed["sha256"] == fixed_equivalent["sha256"]
    assert followed["sha256"] != materially_different["sha256"]
    assert followed["configuration"]["run_seeds"] == [42]
    assert followed["configuration"]["resolved_split_seeds"] == [42]


def test_declared_real_fields_normalize_int_and_float_equivalents():
    integral = provenance.build_scientific_configuration_identity(
        _configuration(
            lr=1,
            test_split=0,
            val_split=0,
            sqi_threshold=0,
            sqi_keep_pct=1,
            outlier_filtering_on_train=True,
        )
    )
    real = provenance.build_scientific_configuration_identity(
        _configuration(
            lr=1.0,
            test_split=0.0,
            val_split=0.0,
            sqi_threshold=0.0,
            sqi_keep_pct=1.0,
            outlier_filtering_on_train=True,
        )
    )
    assert integral["sha256"] == real["sha256"]
    assert isinstance(integral["configuration"]["lr"], float)
    assert isinstance(integral["configuration"]["val_split"], float)

    # A genuinely different declared-real value must still change identity.
    different = provenance.build_scientific_configuration_identity(
        _configuration(lr=2, test_split=0, val_split=0)
    )
    assert integral["sha256"] != different["sha256"]

    # A field that is genuinely an integer (not declared real) must keep
    # distinguishing 1 from 1.0.
    integer_field = provenance.build_scientific_configuration_identity(
        _configuration(n_runs=1)
    )
    float_valued_integer_field = provenance.build_scientific_configuration_identity(
        _configuration(n_runs=1.0)
    )
    assert integer_field["sha256"] != float_valued_integer_field["sha256"]

    # Booleans must not collide with 0/1 on a declared-real field: the
    # normalizer explicitly leaves bool values untouched.
    boolean_valued = provenance.build_scientific_configuration_identity(
        _configuration(lr=1, test_split=0, val_split=True)
    )
    zero_valued = provenance.build_scientific_configuration_identity(
        _configuration(lr=1, test_split=0, val_split=0)
    )
    assert boolean_valued["sha256"] != zero_valued["sha256"]
    assert boolean_valued["configuration"]["val_split"] is True


def test_resolved_parser_defaults_and_explicit_equivalents_match():
    default_arguments, _ = main.parse_experiment_arguments(
        ["--dataset", "ecgid", "--task", "1"]
    )
    explicit_arguments, _ = main.parse_experiment_arguments(
        [
            "--dataset",
            "ecgid",
            "--task",
            "1",
            "--epochs",
            "150",
            "--split_seed",
            "42",
            "--results_dir",
            "different-output-location",
        ]
    )
    default_identity = provenance.build_scientific_configuration_identity(
        main.build_effective_configuration(default_arguments)
    )
    explicit_identity = provenance.build_scientific_configuration_identity(
        main.build_effective_configuration(explicit_arguments)
    )
    assert default_identity["sha256"] == explicit_identity["sha256"]


def test_legacy_configuration_aliases_share_canonical_identity(tmp_path):
    canonical_path = tmp_path / "canonical.yaml"
    legacy_path = tmp_path / "legacy.yaml"
    canonical_path.write_text(
        yaml.safe_dump(
            {
                "dataset": "ecgid",
                "task": 2,
                "pair_sampling_mode": "balanced",
                "pair_sampling_budget": 320,
            }
        ),
        encoding="utf-8",
    )
    legacy_path.write_text(
        yaml.safe_dump(
            {
                "dataset": "ecgid",
                "task": 2,
                "sampling_mode": "balanced",
                "num_pairs": 320,
            }
        ),
        encoding="utf-8",
    )
    canonical_arguments, _ = main.parse_experiment_arguments(
        ["--config", str(canonical_path)]
    )
    legacy_arguments, _ = main.parse_experiment_arguments(
        ["--config", str(legacy_path)]
    )
    canonical_identity = provenance.build_scientific_configuration_identity(
        main.build_effective_configuration(canonical_arguments)
    )
    legacy_identity = provenance.build_scientific_configuration_identity(
        main.build_effective_configuration(legacy_arguments)
    )
    assert canonical_identity["sha256"] == legacy_identity["sha256"]


def test_unclassified_effective_field_fails_exhaustiveness_guard():
    with pytest.raises(provenance.CanonicalConfigurationError, match="Unclassified"):
        provenance.build_scientific_configuration_identity(
            _configuration(new_result_affecting_option=True)
        )


def test_all_parser_fields_have_an_explicit_identity_classification():
    parser_fields = {
        action.dest
        for action in main.get_parser()._actions
        if action.dest != "help"
    }
    classified = (
        provenance.EFFECTIVE_CONFIGURATION_SCIENTIFIC_FIELDS
        | provenance.EFFECTIVE_CONFIGURATION_ADMINISTRATIVE_FIELDS
    )
    assert parser_fields <= classified
    assert (
        provenance.EFFECTIVE_CONFIGURATION_SCIENTIFIC_FIELDS
        & provenance.EFFECTIVE_CONFIGURATION_ADMINISTRATIVE_FIELDS
    ) == set()


def test_public_task_runner_inventory_has_exactly_eight_entries():
    # A change to this count means a public task runner was added or
    # removed. Every test below iterates run.PUBLIC_TASK_RUNNERS rather
    # than a separately hardcoded list, so that change is the single place
    # coverage must be deliberately re-reviewed.
    assert len(run.PUBLIC_TASK_RUNNERS) == 8
    assert set(run.PUBLIC_TASK_RUNNERS) == set(range(1, 9))


def test_all_eight_runner_parameters_are_explicitly_classified():
    categories = (
        provenance.RUNNER_SCIENTIFIC_PARAMETERS,
        provenance.RUNNER_ADMINISTRATIVE_PARAMETERS,
        provenance.RUNNER_DATA_PARAMETERS,
    )
    classified = set().union(*categories)
    assert all(
        not (categories[left] & categories[right])
        for left in range(len(categories))
        for right in range(left + 1, len(categories))
    )
    for task, runner in run.PUBLIC_TASK_RUNNERS.items():
        parameters = set(inspect.signature(runner).parameters)
        assert parameters - classified == set(), task


def test_all_eight_runners_route_structured_output_through_central_provenance():
    for task, runner in run.PUBLIC_TASK_RUNNERS.items():
        source = inspect.getsource(runner)
        assert "_build_per_run_results" in source, task
        assert "_log_experiment_results" in source, task

    logger_source = inspect.getsource(run._log_experiment_results)
    assert "build_result_record_provenance" in logger_source
    assert "canonical_provenance=canonical_provenance" in logger_source


def test_runner_scientific_parameters_are_all_represented_in_identity():
    # Every scientific runner parameter must be traceable to canonical
    # scientific identity: either it is itself a classified effective-
    # configuration field, or it is an explicitly documented derived value
    # (RUNNER_DERIVED_SCIENTIFIC_PARAMETER_SOURCES) whose declared source
    # fields are themselves classified as scientific. This detects a future
    # scientific runner argument added without a provenance mapping, not
    # merely a self-consistent classification list.
    assert provenance.verify_runner_scientific_parameters_are_represented() == []


def test_derived_scientific_parameter_omission_is_detected():
    unrepresented = provenance.verify_runner_scientific_parameters_are_represented
    scientific = set(provenance.RUNNER_SCIENTIFIC_PARAMETERS)
    sources = dict(provenance.RUNNER_DERIVED_SCIENTIFIC_PARAMETER_SOURCES)
    try:
        provenance.RUNNER_SCIENTIFIC_PARAMETERS = scientific | {
            "hypothetical_future_parameter"
        }
        assert "hypothetical_future_parameter" in unrepresented()

        provenance.RUNNER_DERIVED_SCIENTIFIC_PARAMETER_SOURCES[
            "hypothetical_future_parameter"
        ] = ("cache_dir",)
        assert "hypothetical_future_parameter" in unrepresented()
    finally:
        provenance.RUNNER_SCIENTIFIC_PARAMETERS = scientific
        provenance.RUNNER_DERIVED_SCIENTIFIC_PARAMETER_SOURCES.clear()
        provenance.RUNNER_DERIVED_SCIENTIFIC_PARAMETER_SOURCES.update(sources)


def test_live_scientific_runner_parameters_are_provably_represented():
    # Ties together the full chain: every runner in run.PUBLIC_TASK_
    # RUNNERS, its LIVE inspect.signature parameter names, the RUNNER_*
    # classification, RUNNER_DERIVED_SCIENTIFIC_PARAMETER_SOURCES, and the
    # effective scientific configuration field classification. The other
    # tests in this module prove these pieces are each internally sound;
    # this one proves they are wired to each other against live data, not
    # merely self-consistent in isolation.
    categories = (
        provenance.RUNNER_SCIENTIFIC_PARAMETERS,
        provenance.RUNNER_ADMINISTRATIVE_PARAMETERS,
        provenance.RUNNER_DATA_PARAMETERS,
    )
    classified = set().union(*categories)

    live_parameters = set()
    for task, runner in run.PUBLIC_TASK_RUNNERS.items():
        parameters = set(inspect.signature(runner).parameters)
        assert parameters - classified == set(), task
        live_parameters |= parameters

    # Restrict to parameters that are both live on at least one of the
    # eight runners AND classified scientific -- the set a future scientific
    # parameter would actually land in.
    live_scientific = live_parameters & provenance.RUNNER_SCIENTIFIC_PARAMETERS
    assert live_scientific, "expected at least one live scientific parameter"

    def _is_represented(name):
        if name in provenance.EFFECTIVE_CONFIGURATION_SCIENTIFIC_FIELDS:
            return True
        sources = provenance.RUNNER_DERIVED_SCIENTIFIC_PARAMETER_SOURCES.get(name)
        return bool(sources) and all(
            source in provenance.EFFECTIVE_CONFIGURATION_SCIENTIFIC_FIELDS
            for source in sources
        )

    unrepresented = sorted(
        name for name in live_scientific if not _is_represented(name)
    )
    assert unrepresented == []

    # Cross-check against the standalone production guard: it must agree,
    # proving this test is not a parallel reimplementation that could
    # silently diverge from the function actually available for reuse.
    assert provenance.verify_runner_scientific_parameters_are_represented() == []


def test_scientific_parameter_missing_from_a_live_signature_would_fail():
    # Simulates the exact failure the correction targets: a parameter
    # classified scientific and live on a real runner signature, but with
    # no representation path at all. Uses a real runner's actual parameter
    # name set plus one synthetic addition, so the live-signature coupling
    # itself is exercised, not just the static frozenset in isolation.
    sample_runner = next(iter(run.PUBLIC_TASK_RUNNERS.values()))
    live_parameters = set(inspect.signature(sample_runner).parameters) | {
        "hypothetical_future_parameter"
    }
    scientific = set(provenance.RUNNER_SCIENTIFIC_PARAMETERS) | {
        "hypothetical_future_parameter"
    }

    def _is_represented(name):
        if name in provenance.EFFECTIVE_CONFIGURATION_SCIENTIFIC_FIELDS:
            return True
        sources = provenance.RUNNER_DERIVED_SCIENTIFIC_PARAMETER_SOURCES.get(name)
        return bool(sources) and all(
            source in provenance.EFFECTIVE_CONFIGURATION_SCIENTIFIC_FIELDS
            for source in sources
        )

    live_scientific = live_parameters & scientific
    unrepresented = sorted(
        name for name in live_scientific if not _is_represented(name)
    )
    assert unrepresented == ["hypothetical_future_parameter"]


def test_implementation_source_identity_is_deterministic_and_content_based():
    first = _implementation(source_suffix="VALUE = 1\n")
    second = _implementation(source_suffix="VALUE = 1\n")
    changed = _implementation(source_suffix="VALUE = 2\n")
    assert first["source_sha256"] == second["source_sha256"]
    assert first["source_sha256"] != changed["source_sha256"]


def test_git_state_is_separate_from_source_digest():
    clean = _implementation(dirty=False)
    dirty = _implementation(dirty=True)
    other_commit = provenance.build_experiment_implementation_identity(
        source_reader=lambda module: f"MODULE = {module!r}\n".encode(),
        git_runner=_git_runner(dirty=False, commit="def456"),
    )
    assert clean["source_sha256"] == dirty["source_sha256"]
    assert clean["source_sha256"] == other_commit["source_sha256"]
    assert clean["git"]["dirty"] is False
    assert dirty["git"]["dirty"] is True


def test_result_provenance_records_completion_campaign_and_eligibility():
    record = _canonical_record()
    canonical = record["canonical_provenance"]
    assert canonical["scientific_configuration"]["sha256"]
    assert canonical["implementation"]["source_sha256"]
    assert canonical["execution"]["campaign_id"] == "campaign-a"
    assert canonical["execution"]["smoke_run"] is False
    assert canonical["execution"]["status"] == "success"
    assert canonical["execution"]["completion"] == {
        "expected": {
            "run_count": 2,
            "run_seeds": [42, 43],
            "resolved_split_seeds": [42, 43],
        },
        "completed": {
            "run_count": 2,
            "run_seeds": [42, 43],
            "resolved_split_seeds": [42, 43],
            "per_run_indices": [1, 2],
            "source": "per_run_results",
        },
        "complete": True,
    }
    assert canonical["publication_eligibility"] == {
        "eligible": True,
        "reasons": [],
    }


def test_structured_logger_persists_authoritative_canonical_provenance(tmp_path):
    configuration = _configuration(results_dir=str(tmp_path))
    loader = SimpleNamespace(
        cfg={"root_dir": "ecgid", "preprocessing": {}},
        prep_params={},
        results_dir=tmp_path,
        effective_experiment_configuration=configuration,
        result_provenance_context={
            "campaign_id": "campaign-a",
            "smoke_run": False,
        },
    )
    per_run_results = [
        {
            "run_index": 1,
            "seed": 42,
            "split_seed": 42,
            "metrics": {"A": 0.8},
            "data_statistics": {"run": 1},
            "trained_weight": {"state_dict_sha256": "1" * 64},
        },
        {
            "run_index": 2,
            "seed": 43,
            "split_seed": 43,
            "metrics": {"A": 0.9},
            "data_statistics": {"run": 2},
            "trained_weight": {"state_dict_sha256": "2" * 64},
        },
    ]
    with patch(
        "run.build_experiment_implementation_identity",
        return_value=_implementation(),
    ), patch(
        "run._collect_software_environment",
        return_value={},
    ), patch(
        "run._collect_source_revision",
        return_value={},
    ), patch(
        "run._collect_runtime_profile",
        return_value={},
    ):
        run._log_experiment_results(
            "Synthetic Task",
            {"A": "0.8500 ± 0.0500"},
            {"run": 2},
            {
                "n_runs": 2,
                "run_seeds": [42, 43],
                "resolved_split_seeds": [42, 43],
            },
            loader=loader,
            per_run_results=per_run_results,
        )

    record = json.loads(
        (tmp_path / "ecgid" / "Synthetic_Task.jsonl").read_text(
            encoding="utf-8"
        )
    )
    canonical = record["canonical_provenance"]
    assert canonical["scientific_configuration"]["authoritative"] is True
    assert canonical["execution"]["campaign_id"] == "campaign-a"
    assert canonical["execution"]["completion"]["complete"] is True
    assert canonical["publication_eligibility"]["eligible"] is True
    assert record["per_run_results"] == per_run_results


def test_result_provenance_does_not_fabricate_missing_runs():
    record = _canonical_record(
        per_run_results=[
            {
                "run_index": 1,
                "seed": 42,
                "split_seed": 42,
                "metrics": {"A": 0.8},
            }
        ]
    )
    completion = record["canonical_provenance"]["execution"]["completion"]
    assert completion["completed"]["run_count"] == 1
    assert completion["complete"] is False
    assert record["canonical_provenance"]["publication_eligibility"] == {
        "eligible": False,
        "reasons": ["incomplete_run_schedule"],
    }


def test_dirty_smoke_and_fallback_records_are_not_publication_eligible():
    dirty = _canonical_record(implementation=_implementation(dirty=True))
    assert dirty["canonical_provenance"]["publication_eligibility"] == {
        "eligible": False,
        "reasons": ["dirty_source_tree"],
    }
    smoke = _canonical_record(smoke_run=True)
    assert "smoke_run" in smoke["canonical_provenance"][
        "publication_eligibility"
    ]["reasons"]

    fallback = provenance.build_result_record_provenance(
        effective_configuration={"task": "Synthetic"},
        configuration_authoritative=False,
        implementation_identity=_implementation(),
        campaign_id=None,
        smoke_run=False,
        hyperparameters={"n_runs": 1, "run_seeds": [42]},
        per_run_results=None,
    )
    assert fallback["scientific_configuration"]["sha256"]
    assert fallback["publication_eligibility"]["eligible"] is False
    assert "authoritative_effective_configuration_unavailable" in fallback[
        "publication_eligibility"
    ]["reasons"]


def test_exactly_one_new_matching_record_returns_exact_locator(tmp_path):
    path = tmp_path / "results" / "task.jsonl"
    path.parent.mkdir()
    stale = _canonical_record(campaign_id="old-campaign")
    _write_record(path, stale)
    snapshot = provenance.capture_result_log_snapshot(path)
    record = _canonical_record()
    _write_record(path, record, mode="a")

    collected = _collect(path, snapshot, record)
    assert collected["record"] == record
    assert collected["locator"]["relative_path"] == "task.jsonl"
    assert collected["locator"]["line_number"] == 2
    assert collected["locator"]["line_number_base"] == 1
    assert collected["locator"]["record_sha256"] == hashlib.sha256(
        provenance.canonical_json_bytes(record)
    ).hexdigest()


@pytest.mark.parametrize("append_mode", ["none", "wrong", "unrelated"])
def test_stale_or_missing_new_record_never_falls_back(tmp_path, append_mode):
    path = tmp_path / "task.jsonl"
    record = _canonical_record()
    _write_record(path, record)
    snapshot = provenance.capture_result_log_snapshot(path)
    if append_mode == "wrong":
        wrong = _canonical_record(configuration=_configuration(epochs=9))
        _write_record(path, wrong, mode="a")
    elif append_mode == "unrelated":
        unrelated = _canonical_record(campaign_id="other")
        _write_record(path, unrelated, mode="a")

    with pytest.raises(provenance.ResultCollectionError):
        _collect(path, snapshot, record)


def test_multiple_new_records_fail_closed(tmp_path):
    path = tmp_path / "task.jsonl"
    snapshot = provenance.capture_result_log_snapshot(path)
    record = _canonical_record()
    _write_record(path, record)
    _write_record(path, record, mode="a")
    with pytest.raises(provenance.ResultCollectionError, match="exactly one"):
        _collect(path, snapshot, record)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda record: record["canonical_provenance"]["execution"].update(
                {"campaign_id": "wrong"}
            ),
            "campaign",
        ),
        (
            lambda record: record["canonical_provenance"]["implementation"].update(
                {"source_sha256": "0" * 64}
            ),
            "implementation",
        ),
        (
            lambda record: record["canonical_provenance"]["execution"].update(
                {"smoke_run": True}
            ),
            "smoke",
        ),
        (
            lambda record: record["canonical_provenance"]["execution"].update(
                {"successful": False, "status": "failed"}
            ),
            "unsuccessful",
        ),
        (
            lambda record: record["canonical_provenance"]["execution"][
                "completion"
            ].update({"complete": False}),
            "incomplete",
        ),
        (
            # Contradictory record: publication_eligibility.eligible is
            # left True (untouched) while the underlying Git evidence in
            # the SAME record says the source tree was dirty. Validation
            # must re-derive cleanliness from that evidence rather than
            # trusting the stored eligibility summary.
            lambda record: record["canonical_provenance"]["implementation"][
                "git"
            ].update({"dirty": True}),
            "dirty",
        ),
        (
            # Same contradiction for source-revision availability.
            lambda record: record["canonical_provenance"]["implementation"][
                "git"
            ].update({"status": "unavailable"}),
            "not available",
        ),
    ],
)
def test_mismatched_or_ineligible_new_record_fails(tmp_path, mutation, message):
    path = tmp_path / "task.jsonl"
    snapshot = provenance.capture_result_log_snapshot(path)
    expected = _canonical_record()
    actual = copy.deepcopy(expected)
    mutation(actual)
    _write_record(path, actual)
    with pytest.raises(provenance.ResultCollectionError, match=message):
        _collect(path, snapshot, expected)


def test_missing_canonical_provenance_and_malformed_append_fail(tmp_path):
    path = tmp_path / "task.jsonl"
    snapshot = provenance.capture_result_log_snapshot(path)
    expected = _canonical_record()
    _write_record(path, {"task": "legacy"})
    with pytest.raises(provenance.ResultCollectionError, match="lacks canonical"):
        _collect(path, snapshot, expected)

    path.unlink()
    snapshot = provenance.capture_result_log_snapshot(path)
    path.write_text('{"truncated":', encoding="utf-8")
    with pytest.raises(provenance.ResultCollectionError, match="Malformed"):
        _collect(path, snapshot, expected)


def test_replaced_or_truncated_file_fails_closed(tmp_path):
    path = tmp_path / "task.jsonl"
    old = _canonical_record(campaign_id="old")
    _write_record(path, old)
    snapshot = provenance.capture_result_log_snapshot(path)
    expected = _canonical_record()

    path.write_text("", encoding="utf-8")
    with pytest.raises(provenance.ResultCollectionError, match="truncated"):
        _collect(path, snapshot, expected)

    _write_record(path, old)
    snapshot = provenance.capture_result_log_snapshot(path)
    replacement = tmp_path / "replacement.jsonl"
    _write_record(replacement, old)
    _write_record(replacement, expected, mode="a")
    os.replace(replacement, path)
    with pytest.raises(provenance.ResultCollectionError, match="replaced"):
        _collect(path, snapshot, expected)


def test_existing_exact_selection_rejects_legacy_and_ambiguity(tmp_path):
    path = tmp_path / "task.jsonl"
    path.write_text('{"task":"legacy"}\n', encoding="utf-8")
    record = _canonical_record()
    _write_record(path, record, mode="a")
    identity = record["canonical_provenance"]["scientific_configuration"]
    selected = provenance.select_exact_result_record(
        path,
        result_root=tmp_path,
        expected_scientific_sha256=identity["sha256"],
        expected_implementation=record["canonical_provenance"]["implementation"],
        required_campaign_id="campaign-a",
    )
    assert selected["record"] == record
    assert selected["locator"]["line_number"] == 2

    _write_record(path, record, mode="a")
    with pytest.raises(provenance.ResultCollectionError, match="observed 2"):
        provenance.select_exact_result_record(
            path,
            result_root=tmp_path,
            expected_scientific_sha256=identity["sha256"],
            expected_implementation=record["canonical_provenance"][
                "implementation"
            ],
            required_campaign_id="campaign-a",
        )


def test_legacy_read_is_explicit_and_does_not_invent_identity(tmp_path):
    path = tmp_path / "legacy.jsonl"
    path.write_text('{"task":"old"}\nnot-json\n', encoding="utf-8")
    record = provenance.read_legacy_latest_record(path)
    assert record == {"task": "old"}
    assert "canonical_provenance" not in record


def test_non_finite_configuration_value_fails_before_task_dispatch(monkeypatch):
    # json.loads accepts bare NaN/Infinity literals unless explicitly
    # rejected, so a JSON-mapping CLI argument can carry a non-finite value
    # past ordinary argument validation. The canonical identity must be
    # rejected at launch, before any dataset is loaded.
    argv = [
        "main.py",
        "--dataset", "ecgid",
        "--task", "1",
        "--model", "resnet1d",
        "--preprocessing_parameters", '{"x": NaN}',
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with patch.object(main, "load_ecgid_dataset") as loader_mock:
        with pytest.raises(SystemExit):
            main.main()
    loader_mock.assert_not_called()


def test_unclassified_effective_field_fails_before_task_dispatch(monkeypatch):
    argv = [
        "main.py",
        "--dataset", "ecgid",
        "--task", "1",
        "--model", "resnet1d",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    original_build = main.build_effective_configuration

    def _with_unclassified_field(args):
        configuration = original_build(args)
        configuration["new_result_affecting_option"] = True
        return configuration

    with patch.object(
        main,
        "build_effective_configuration",
        _with_unclassified_field,
    ), patch.object(main, "load_ecgid_dataset") as loader_mock:
        with pytest.raises(SystemExit):
            main.main()
    loader_mock.assert_not_called()


def test_normal_configuration_passes_the_fail_fast_identity_check(monkeypatch):
    # A normal, fully classified configuration must not be rejected by the
    # same fail-fast check; it should reach dataset loading unchanged.
    argv = [
        "main.py",
        "--dataset", "ecgid",
        "--task", "1",
        "--model", "resnet1d",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with patch.object(
        main,
        "load_ecgid_dataset",
        side_effect=RuntimeError("stop after reaching dataset loading"),
    ) as loader_mock:
        with pytest.raises(RuntimeError, match="stop after reaching dataset loading"):
            main.main()
    loader_mock.assert_called_once()
