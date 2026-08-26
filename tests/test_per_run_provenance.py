import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

import run
from artifact_provenance import (
    STATE_DICT_HASH_FORMAT,
    canonical_state_dict_sha256,
)


class TinyProvenanceModel(nn.Module):
    def __init__(self, in_channels, num_classes, include_top=True):
        super().__init__()
        self.include_top = include_top
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, inputs):
        features = inputs.mean(dim=-1)
        if self.include_top:
            return self.classifier(features)
        return features


def _weight_reference(index=1):
    digit = format(index % 16, "x")
    return {
        "persisted": True,
        "source": "trained_and_saved",
        "weight_uid": f"uid-{index}",
        "state_dict_hash_format": STATE_DICT_HASH_FORMAT,
        "state_dict_sha256": digit * 64,
        "payload_sha256": format((index + 1) % 16, "x") * 64,
    }


def _effective_reproducibility_state():
    return {
        "requested_mode": "seeded",
        "effective_mode": "seeded",
        "requested_device": "cpu",
        "device_type": "cpu",
        "device_name": "cpu",
        "deterministic_algorithms_enabled": False,
        "cudnn_deterministic": None,
        "cudnn_benchmark": None,
        "cublas_workspace_config": None,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": None,
        "cudnn_version": None,
    }


def _synthetic_data():
    generator = np.random.default_rng(123)
    samples = generator.normal(size=(18, 2, 16)).astype(np.float32)
    labels = np.repeat(np.arange(3), 6)
    return samples, labels


def _complete_training(captured_models):
    def complete(model, *args, **kwargs):
        model.actual_epochs = 1
        captured_models.append(model)
        return model

    return complete


def _run_single_with_weight_sink(
    tmp_path,
    intelligent_weight_loading,
):
    samples, labels = _synthetic_data()
    cache_dir = tmp_path / "cache"
    references = []
    captured_models = []
    sink_token = run._TRAINED_WEIGHT_REFERENCE_SINK.set(references)
    training = _complete_training(captured_models)
    try:
        with patch(
            "run._activate_top_level_runtime_profile",
            return_value="seeded",
        ), patch(
            "run._setup_reproducibility",
            return_value=("cpu", _effective_reproducibility_state()),
        ), patch(
            "run._run_training_loop",
            side_effect=training,
        ) as training_mock:
            result = run.run_closed_set_identification(
                samples,
                labels,
                TinyProvenanceModel,
                epochs=1,
                batch_size=4,
                test_split=0.5,
                val_split=0.0,
                seed=42,
                split_seed=None,
                device="cpu",
                loader=SimpleNamespace(cache_dir=cache_dir),
                _return_stats=True,
                intelligent_weight_loading=intelligent_weight_loading,
                reproducibility_mode="seeded",
            )
    finally:
        run._TRAINED_WEIGHT_REFERENCE_SINK.reset(sink_token)

    assert len(references) == 1
    return {
        "result": result,
        "reference": references[0],
        "captured_models": captured_models,
        "training_call_count": training_mock.call_count,
        "cache_dir": cache_dir,
    }


def test_complete_per_run_records_preserve_metrics_and_copy_provenance():
    shared_statistics = {
        "samples": 10,
        "nested": {"values": [1]},
    }
    first_weight = _weight_reference(1)
    second_weight = _weight_reference(2)
    records = run._build_per_run_results(
        results=[(0.8, 0.9), (0.7, 0.85)],
        seeds=[42, 43],
        split_seeds=[42, 43],
        data_statistics=[shared_statistics, shared_statistics],
        trained_weight_references=[first_weight, second_weight],
    )

    assert [record["run_index"] for record in records] == [1, 2]
    assert records[0] == {
        "run_index": 1,
        "seed": 42,
        "split_seed": 42,
        "metrics": {
            "Rank-1 Accuracy": 0.8,
            "Rank-5 Accuracy": 0.9,
        },
        "data_statistics": shared_statistics,
        "trained_weight": first_weight,
    }

    shared_statistics["nested"]["values"].append(2)
    first_weight["source"] = "changed"
    records[0]["data_statistics"]["nested"]["values"].append(3)
    assert records[0]["trained_weight"]["source"] == "trained_and_saved"
    assert records[1]["data_statistics"]["nested"]["values"] == [1]


def test_historical_per_run_helper_usage_remains_compatible():
    records = run._build_per_run_results(
        results=[(0.8, 0.9)],
        seeds=[42],
    )
    assert records == [
        {
            "run_index": 1,
            "seed": 42,
            "metrics": {
                "Rank-1 Accuracy": 0.8,
                "Rank-5 Accuracy": 0.9,
            },
        }
    ]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("seeds", [42]),
        ("split_seeds", [42]),
        ("data_statistics", [{"run": 1}]),
        ("trained_weight_references", [_weight_reference(1)]),
    ],
)
def test_per_run_provenance_length_mismatch_is_rejected(field_name, value):
    arguments = {
        "results": [(0.8, 0.9), (0.7, 0.8)],
        "seeds": [42, 43],
        "split_seeds": [42, 43],
        "data_statistics": [{"run": 1}, {"run": 2}],
        "trained_weight_references": [
            _weight_reference(1),
            _weight_reference(2),
        ],
    }
    arguments[field_name] = value
    with pytest.raises(ValueError, match="match|matching lengths"):
        run._build_per_run_results(**arguments)


def test_partial_per_run_provenance_is_rejected():
    with pytest.raises(ValueError, match="supplied together"):
        run._build_per_run_results(
            results=[(0.8, 0.9)],
            seeds=[42],
            split_seeds=[42],
        )


@pytest.mark.parametrize(
    ("configured_split_seed", "expected_split_seeds"),
    [
        (None, [42, 43, 44]),
        (123, [123, 123, 123]),
    ],
)
def test_per_run_split_seeds_follow_authoritative_schedule(
    configured_split_seed,
    expected_split_seeds,
):
    hyperparameters = run._add_seed_metadata(
        {},
        base_seed=42,
        n_runs=3,
        split_seed=configured_split_seed,
    )
    records = run._build_per_run_results(
        results=[(0.8, 0.9)] * 3,
        seeds=hyperparameters["run_seeds"],
        split_seeds=hyperparameters["resolved_split_seeds"],
        data_statistics=[{"run": index} for index in range(3)],
        trained_weight_references=[
            _weight_reference(index + 1) for index in range(3)
        ],
    )
    assert [record["seed"] for record in records] == [42, 43, 44]
    assert [record["split_seed"] for record in records] == (
        expected_split_seeds
    )


def test_structured_multi_run_record_marks_last_run_statistics_scope():
    per_run_results = run._build_per_run_results(
        results=[(0.8, 0.9), (0.7, 0.85)],
        seeds=[42, 43],
        split_seeds=[42, 43],
        data_statistics=[{"run": 1}, {"run": 2}],
        trained_weight_references=[
            _weight_reference(1),
            _weight_reference(2),
        ],
    )
    record = run._build_structured_experiment_record(
        experiment_time=__import__("datetime").datetime(2026, 8, 26),
        task_name="Synthetic Task",
        dataset_name="synthetic",
        metrics_dict={"Rank-1 Accuracy": "0.7500 ± 0.0500"},
        data_stats={"run": 2},
        hyperparams={},
        dataset_kwargs={},
        software_environment={},
        source_revision={},
        runtime_profile={},
        per_run_results=per_run_results,
    )
    assert isinstance(record["data_statistics"], dict)
    assert record["data_statistics"] == {"run": 2}
    assert record["data_statistics_scope"] == {
        "scope": "last_run_snapshot",
        "run_index": 2,
        "seed": 43,
        "split_seed": 43,
    }
    assert record["per_run_results"][0]["data_statistics"] == {"run": 1}
    json.dumps(record, allow_nan=False)

    single_record = run._build_structured_experiment_record(
        experiment_time=__import__("datetime").datetime(2026, 8, 26),
        task_name="Synthetic Task",
        dataset_name="synthetic",
        metrics_dict={"Rank-1 Accuracy": 0.8},
        data_stats={"run": 1},
        hyperparams={},
        dataset_kwargs={},
        software_environment={},
        source_revision={},
        runtime_profile={},
    )
    assert "data_statistics_scope" not in single_record


def test_uncertainty_reads_only_metrics_from_enriched_records():
    records = run._build_per_run_results(
        results=[(0.1, 0.9, 1.5, 0.7), (0.2, 0.8, 1.2, 0.6)],
        seeds=[42, 43],
        split_seeds=[42, 43],
        data_statistics=[
            {"EER": 999, "nested": {"values": [1]}},
            {"EER": -999, "nested": {"values": [2]}},
        ],
        trained_weight_references=[
            _weight_reference(1),
            _weight_reference(2),
        ],
    )
    summary = run._summarize_per_run_uncertainty(records)
    assert summary["runs"] == 2
    assert set(summary["metrics"]) == {
        "EER",
        "AUC",
        "d-prime",
        "TAR@0.1%FAR",
    }
    assert summary["metrics"]["EER"]["mean"] == pytest.approx(0.15)


def test_newly_saved_and_cache_hit_references_match_real_artifact(tmp_path):
    first = _run_single_with_weight_sink(tmp_path, True)
    reference = first["reference"]
    assert first["training_call_count"] == 1
    assert reference["persisted"] is True
    assert reference["source"] == "trained_and_saved"
    assert reference["state_dict_hash_format"] == STATE_DICT_HASH_FORMAT
    assert str(tmp_path) not in json.dumps(reference)

    payload_path = (
        first["cache_dir"]
        / "weights"
        / f"{reference['weight_uid']}.pth"
    )
    payload_bytes = payload_path.read_bytes()
    saved_state = torch.load(
        payload_path,
        map_location="cpu",
        weights_only=True,
    )
    assert reference["state_dict_sha256"] == canonical_state_dict_sha256(
        saved_state
    )
    assert reference["payload_sha256"] == hashlib.sha256(
        payload_bytes
    ).hexdigest()

    second = _run_single_with_weight_sink(tmp_path, True)
    assert second["training_call_count"] == 0
    assert second["reference"] == {
        **reference,
        "source": "cache_hit",
    }


def test_cache_disabled_records_in_memory_state_without_artifact(tmp_path):
    execution = _run_single_with_weight_sink(tmp_path, False)
    reference = execution["reference"]
    assert execution["training_call_count"] == 1
    assert reference == {
        "persisted": False,
        "source": "trained_not_persisted",
        "weight_uid": None,
        "state_dict_hash_format": STATE_DICT_HASH_FORMAT,
        "state_dict_sha256": canonical_state_dict_sha256(
            execution["captured_models"][0].state_dict()
        ),
        "payload_sha256": None,
    }
    assert not list(tmp_path.rglob("*.pth"))


def test_recursive_weight_sink_does_not_reuse_stale_reference():
    evaluation_artifacts = []

    def successful_runner(**kwargs):
        run._record_evaluation_artifact({"type": "identification"})
        run._record_trained_weight_reference(_weight_reference(1))
        return (0.8, 0.9), {"run": 1}, {}

    result, reference = run._run_recursive_with_provenance(
        successful_runner,
        {},
        evaluation_artifacts,
    )
    assert result[1] == {"run": 1}
    assert reference == _weight_reference(1)
    assert run._TRAINED_WEIGHT_REFERENCE_SINK.get() is None

    def missing_reference_runner(**kwargs):
        run._record_evaluation_artifact({"type": "identification"})
        return (0.7, 0.8), {"run": 2}, {}

    with pytest.raises(RuntimeError, match="exactly one"):
        run._run_recursive_with_provenance(
            missing_reference_runner,
            {},
            evaluation_artifacts,
        )


def test_real_multi_run_persists_complete_run_local_provenance(tmp_path):
    samples, labels = _synthetic_data()
    cache_dir = tmp_path / "multi-cache"
    captured_models = []

    with patch(
        "run._prepare_reproducibility_backend",
        return_value="seeded",
    ), patch(
        "run._activate_top_level_runtime_profile",
        return_value="seeded",
    ), patch(
        "run._setup_reproducibility",
        return_value=("cpu", _effective_reproducibility_state()),
    ), patch(
        "run._run_training_loop",
        side_effect=_complete_training(captured_models),
    ), patch(
        "run._log_experiment_results",
    ) as logger:
        aggregate = run.run_closed_set_identification(
            samples,
            labels,
            TinyProvenanceModel,
            epochs=1,
            batch_size=4,
            test_split=0.5,
            val_split=0.0,
            seed=42,
            split_seed=None,
            device="cpu",
            loader=SimpleNamespace(cache_dir=cache_dir),
            n_runs=2,
            save_results_and_settings=True,
            intelligent_weight_loading=True,
            reproducibility_mode="seeded",
        )

    assert isinstance(aggregate, tuple)
    assert len(aggregate) == 2
    assert all(isinstance(metric, tuple) for metric in aggregate)
    assert logger.call_count == 1
    per_run_results = logger.call_args.kwargs["per_run_results"]
    assert [record["run_index"] for record in per_run_results] == [1, 2]
    assert [record["seed"] for record in per_run_results] == [42, 43]
    assert [record["split_seed"] for record in per_run_results] == [42, 43]
    assert per_run_results[0]["data_statistics"] is not (
        per_run_results[1]["data_statistics"]
    )
    assert logger.call_args.args[2] == per_run_results[-1]["data_statistics"]

    metric_rows = [
        tuple(record["metrics"].values())
        for record in per_run_results
    ]
    assert aggregate == run._aggregate_multi_run_metrics(metric_rows)

    for record in per_run_results:
        reference = record["trained_weight"]
        assert reference["source"] == "trained_and_saved"
        payload_path = cache_dir / "weights" / f"{reference['weight_uid']}.pth"
        state = torch.load(
            payload_path,
            map_location="cpu",
            weights_only=True,
        )
        assert reference["state_dict_sha256"] == canonical_state_dict_sha256(
            state
        )
        assert reference["payload_sha256"] == hashlib.sha256(
            payload_path.read_bytes()
        ).hexdigest()
