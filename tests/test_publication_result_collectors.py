import inspect
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import experiment_provenance as provenance
from scripts import build_artifact_manifest as manifest_tool
from scripts import make_figures
from scripts import reproduce_tables
from scripts import statistical_comparisons


def test_publication_collectors_do_not_use_latest_record_fallback():
    assert "read_latest_record" not in inspect.getsource(
        reproduce_tables.collect_rows
    )
    assert "collect_configuration_record" in inspect.getsource(
        reproduce_tables.collect_rows
    )
    assert "collect_configuration_record" in inspect.getsource(
        make_figures.collect_series
    )
    assert "collect_configuration_record" in inspect.getsource(
        manifest_tool.collect_configuration_coverage
    )
    assert "select_exact_result_record" in inspect.getsource(
        statistical_comparisons._resolve_experiment_spec
    )


def test_configuration_log_path_has_no_recursive_latest_fallback():
    source = inspect.getsource(
        reproduce_tables.ConfigurationEntry.structured_log_path
    )
    assert "rglob" not in source
    assert "matches[-1]" not in source


def test_reproduction_driver_snapshots_and_validates_new_append():
    source = inspect.getsource(reproduce_tables.run_configurations)
    assert "capture_result_log_snapshot" in source
    assert "collect_appended_result" in source


def test_reproduction_publication_collection_rejects_legacy(tmp_path):
    path = tmp_path / "task.jsonl"
    path.write_text('{"task":"legacy"}\n', encoding="utf-8")
    expectation = {
        "log_path": path,
        "results_root": tmp_path,
        "scientific_identity": {"sha256": "1" * 64},
        "implementation_identity": {
            "source_sha256": "2" * 64,
            "git": {"commit": "abc"},
        },
    }
    with patch.object(
        reproduce_tables,
        "_resolve_execution_expectation",
        return_value=expectation,
    ):
        with pytest.raises(
            provenance.ResultCollectionError,
            match="observed 0",
        ):
            reproduce_tables.collect_configuration_record(
                SimpleNamespace(),
                publication_mode=True,
            )


def test_figure_collection_uses_verified_record_and_keeps_locator():
    entry = SimpleNamespace(
        task_type="verification",
        setting="closed_set",
        protocol="single_session",
        path=SimpleNamespace(name="config.yaml"),
    )
    collected = {
        "record": {
            "per_run_results": [
                {"seed": 42, "metrics": {"EER": 0.1}},
                {"seed": 43, "metrics": {"EER": 0.2}},
            ]
        },
        "locator": {
            "relative_path": "ecgid/task.jsonl",
            "line_number": 3,
            "record_sha256": "3" * 64,
        },
    }
    with patch.object(
        make_figures,
        "build_experiment_implementation_identity",
        return_value={},
    ), patch.object(
        make_figures,
        "collect_configuration_record",
        return_value=collected,
    ) as collector:
        series, missing, locators = make_figures.collect_series(
            [entry],
            "EER",
            include_provenance=True,
        )

    assert missing == []
    assert series["closed_set"]["single_session"] == {
        42: 0.1,
        43: 0.2,
    }
    assert locators[
        "closed_set/single_session/verification"
    ] == collected["locator"]
    assert collector.call_args.kwargs["publication_mode"] is True


def test_manifest_coverage_propagates_verified_locator():
    entry = SimpleNamespace(
        path=SimpleNamespace(
            name="config.yaml",
            relative_to=lambda root: SimpleNamespace(
                __str__=lambda self: "configs/config.yaml"
            ),
        ),
        table=5,
        dataset="ecgid",
        protocol="single_session",
        setting="closed_set",
        task=1,
        task_type="identification",
        structured_log_path=lambda: SimpleNamespace(name="task.jsonl"),
    )
    locator = {
        "relative_path": "ecgid/task.jsonl",
        "line_number": 1,
        "record_sha256": "4" * 64,
    }
    collected = {
        "record": {
            "experiment_time": "2026-01-01T00:00:00",
            "per_run_results": [{"seed": 42}],
        },
        "locator": locator,
    }
    with patch.object(
        manifest_tool,
        "build_experiment_implementation_identity",
        return_value={},
    ), patch.object(
        manifest_tool,
        "collect_configuration_record",
        return_value=collected,
    ):
        coverage = manifest_tool.collect_configuration_coverage([entry])

    assert coverage[0]["executed"] is True
    assert coverage[0]["publication_provenance_verified"] is True
    assert coverage[0]["result_locator"] == locator
    assert coverage[0]["result_log"] == "ecgid/task.jsonl"


def test_statistics_manifest_rejects_positional_latest_by_default(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text(
        json.dumps(
            {
                "per_run_results": [
                    {"seed": 42, "metrics": {"EER": 0.1}}
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "reference": {"path": path.name, "record_index": -1},
        "comparisons": [{"path": path.name, "record_index": -1}],
    }
    with pytest.raises(ValueError, match="Publication experiment specification"):
        statistical_comparisons.analyse_manifest(
            manifest,
            manifest_path=tmp_path / "manifest.yaml",
        )


def test_figure_cli_refuses_partial_publication_output(tmp_path):
    arguments = SimpleNamespace(
        format=None,
        confidence_level=0.95,
        font_size=9,
        config_root=tmp_path,
        dataset="ecgid",
        metric="EER",
        campaign_id=None,
        allow_exploratory_results=False,
        output_dir=tmp_path / "figures",
    )
    parser = SimpleNamespace(parse_args=lambda argv: arguments)

    with patch.object(make_figures, "build_parser", return_value=parser), \
            patch.object(make_figures, "apply_publication_style"), \
            patch.object(make_figures, "discover_configurations", return_value=[]), \
            patch.object(make_figures, "filter_configurations", return_value=[]), \
            patch.object(
                make_figures,
                "collect_series",
                return_value=(
                    {"closed_set": {"single_session": {42: 0.1}}},
                    ["missing configuration"],
                    {},
                ),
            ):
        with pytest.raises(SystemExit, match="Refusing to render"):
            make_figures.main([])

    assert not arguments.output_dir.exists()


def test_table_cli_refuses_incomplete_publication_output(tmp_path):
    output_dir = tmp_path / "tables"
    arguments = SimpleNamespace(
        dry_run=False,
        run=False,
        collect=True,
        config_root=tmp_path,
        table=None,
        dataset=None,
        task=None,
        smoke=False,
        extra_arguments=None,
        campaign_id=None,
        continue_on_error=False,
        allow_exploratory_results=False,
        output_dir=output_dir,
    )
    parser = SimpleNamespace(parse_args=lambda argv: arguments)
    row = {
        "table": 1,
        "dataset": "ecgid",
        "protocol": "single_session",
        "setting": "closed_set",
        "missing": ["no verified record"],
        "sources": {},
    }

    with patch.object(reproduce_tables, "build_parser", return_value=parser), \
            patch.object(
                reproduce_tables,
                "discover_configurations",
                return_value=[SimpleNamespace()],
            ), patch.object(
                reproduce_tables,
                "filter_configurations",
                return_value=[SimpleNamespace()],
            ), patch.object(
                reproduce_tables,
                "collect_rows",
                return_value={(1,): row},
            ), patch.object(
                reproduce_tables,
                "render_markdown",
                return_value="incomplete",
            ):
        assert reproduce_tables.main([]) == 1

    assert not output_dir.exists()


def test_reproduce_tables_real_parser_defaults_to_publication_mode():
    parser = reproduce_tables.build_parser()
    absent = parser.parse_args([])
    explicit = parser.parse_args(["--allow-exploratory-results"])
    assert absent.allow_exploratory_results is False
    assert explicit.allow_exploratory_results is True


def test_make_figures_real_parser_defaults_to_publication_mode():
    parser = make_figures.build_parser()
    base_argv = ["--dataset", "ecgid"]
    absent = parser.parse_args(base_argv)
    explicit = parser.parse_args(base_argv + ["--allow-exploratory-results"])
    assert absent.allow_exploratory_results is False
    assert explicit.allow_exploratory_results is True


def test_build_artifact_manifest_real_parser_defaults_to_publication_mode():
    parser = manifest_tool.build_parser()
    absent = parser.parse_args([])
    explicit = parser.parse_args(["--allow-exploratory-results"])
    assert absent.allow_exploratory_results is False
    assert explicit.allow_exploratory_results is True


def test_statistical_comparisons_manifest_defaults_to_publication_mode():
    # statistical_comparisons has no CLI flag of its own for this switch;
    # the manifest file is the real, production-parsed source of the
    # setting. Capture the publication_mode actually derived from a real
    # analyse_manifest() call rather than asserting the literal expression.
    captured = {}

    def _capture(*args, **kwargs):
        captured["publication_mode"] = kwargs.get("publication_mode")
        raise RuntimeError("stop after capture")

    base_manifest = {
        "reference": {"path": "reference.jsonl"},
        "comparisons": [{"path": "comparison.jsonl"}],
    }

    with patch.object(
        statistical_comparisons,
        "_resolve_experiment_spec",
        _capture,
    ):
        with pytest.raises(RuntimeError, match="stop after capture"):
            statistical_comparisons.analyse_manifest(base_manifest)
    assert captured["publication_mode"] is True

    captured.clear()
    explicit_manifest = dict(base_manifest, allow_exploratory_results=True)
    with patch.object(
        statistical_comparisons,
        "_resolve_experiment_spec",
        _capture,
    ):
        with pytest.raises(RuntimeError, match="stop after capture"):
            statistical_comparisons.analyse_manifest(explicit_manifest)
    assert captured["publication_mode"] is False


def test_manifest_cli_refuses_unverified_publication_output(tmp_path):
    output_path = tmp_path / "manifest.json"
    arguments = SimpleNamespace(
        skip_checksums=True,
        cache_dir=tmp_path / "cache",
        config_root=tmp_path,
        results_dir=None,
        campaign_id=None,
        allow_exploratory_results=False,
        output_json=output_path,
        output_markdown=None,
    )
    parser = SimpleNamespace(parse_args=lambda argv: arguments)
    coverage = [{"collection_error": "no verified record"}]
    summary = {
        "configurations_executed": 0,
        "configurations_total": 1,
        "trained_weight_files": 0,
        "preprocessed_array_files": 0,
        "total_artifact_gigabytes": 0.0,
    }

    with patch.object(manifest_tool, "build_parser", return_value=parser), \
            patch.object(manifest_tool, "collect_weight_artifacts", return_value=[]), \
            patch.object(
                manifest_tool,
                "collect_dataset_cache_artifacts",
                return_value=[],
            ), patch.object(
                manifest_tool,
                "discover_configurations",
                return_value=[],
            ), patch.object(
                manifest_tool,
                "collect_configuration_coverage",
                return_value=coverage,
            ), patch.object(manifest_tool, "summarize", return_value=summary):
        assert manifest_tool.main([]) == 1

    assert not output_path.exists()
