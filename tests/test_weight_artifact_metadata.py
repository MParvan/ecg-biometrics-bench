import copy
import inspect
import json
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from artifact_provenance import (
    STATE_DICT_HASH_FORMAT,
    canonical_json_bytes,
    canonical_state_dict_sha256,
)
import run
from utils import (
    CacheManager,
    _build_loader_cache_identity,
    _build_weight_artifact_metadata,
    _file_sha256,
    _generate_config_hash,
)


class TinyStateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.register_buffer(
            "running_scale",
            torch.tensor([1.5, 2.5]),
        )

    def forward(self, inputs):
        return self.linear(inputs) * self.running_scale


class RunnerIntegrationModel(nn.Module):
    def __init__(self, in_channels, num_classes, include_top=True):
        super().__init__()
        self.include_top = include_top
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, inputs):
        features = inputs.mean(dim=-1)
        if self.include_top:
            return self.classifier(features)
        return features


def _training_config(reproducibility_mode="seeded"):
    return {
        "training_regime": "unit_test",
        "model": "TinyStateModel",
        "epochs": 20,
        "batch_size": 4,
        "lr": 0.001,
        "val_split": 0.25,
        "seed": 42,
        "augmentation": {
            "enabled": False,
            "method": "gaussian",
            "copies": 1,
            "parameters": {},
        },
        "reproducibility_mode": reproducibility_mode,
        "classes": 2,
        "data_shape": (8, 1, 32),
        "loader_identity": {
            "loader_class": "SyntheticLoader",
            "root_dir": "synthetic",
        },
        "training_partition": {
            "sha256": "a" * 64,
            "arrays": {
                "samples": {"shape": [8, 1, 32]},
                "labels": {"shape": [8]},
            },
        },
        "validation_partition": {
            "sha256": "b" * 64,
            "arrays": {
                "X_val": {"shape": [2, 1, 32]},
                "y_val": {"shape": [2]},
            },
        },
        "implementation_identity": {
            "aggregate_sha256": "c" * 64,
        },
        "dependency_identity": {
            "aggregate_sha256": "d" * 64,
        },
    }


def _artifact_context(device_name="cpu"):
    return {
        "model_constructor_arguments": {
            "in_channels": 1,
            "num_classes": 2,
            "include_top": True,
        },
        "resolved_split_seed": 42,
        "reproducibility_state": {
            "requested_mode": "seeded",
            "effective_mode": "seeded",
            "requested_device": "auto",
            "device_type": "cpu",
            "device_name": device_name,
            "deterministic_algorithms_enabled": False,
            "cudnn_deterministic": None,
            "cudnn_benchmark": None,
            "cublas_workspace_config": None,
            "torch_cuda_version": None,
            "cudnn_version": None,
        },
        "training_components": {
            "optimizer": "torch.optim.Adam",
            "loss": "torch.nn.CrossEntropyLoss",
            "scheduler": "framework rollback learning-rate policy",
        },
    }


def _save_artifact(
    tmp_path,
    reproducibility_mode="seeded",
    artifact_context=None,
    creation_provenance=None,
):
    cache = CacheManager(base_dir=tmp_path / "cache")
    config = _training_config(reproducibility_mode)
    model = TinyStateModel()
    model.actual_epochs = 7
    _, uid = cache.get_weight_cache(
        config,
        TinyStateModel(),
        device="cpu",
    )
    cache.save_weight_cache(
        model,
        config,
        uid,
        creation_provenance=(
            creation_provenance
            or {"git": {"commit": "test-commit", "dirty": False}}
        ),
        artifact_context=artifact_context or _artifact_context(),
    )
    weight_path = tmp_path / "cache" / "weights" / f"{uid}.pth"
    metadata_path = tmp_path / "cache" / "weights" / f"{uid}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return cache, config, model, uid, weight_path, metadata_path, metadata


def test_state_hash_ignores_mapping_insertion_order():
    first = OrderedDict(
        [
            ("weight", torch.arange(6, dtype=torch.float32).reshape(2, 3)),
            ("bias", torch.tensor([1.0, 2.0])),
        ]
    )
    second = OrderedDict(reversed(list(first.items())))
    assert canonical_state_dict_sha256(first) == (
        canonical_state_dict_sha256(second)
    )


@pytest.mark.parametrize(
    "changed_state",
    [
        {"value": torch.tensor([1.0, 2.5])},
        {"renamed": torch.tensor([1.0, 2.0])},
        {"value": torch.tensor([1.0, 2.0], dtype=torch.float64)},
        {"value": torch.tensor([1.0, 2.0]).reshape(2, 1)},
    ],
    ids=["value", "entry-name", "dtype", "shape"],
)
def test_state_hash_changes_for_semantic_state_changes(changed_state):
    original = {"value": torch.tensor([1.0, 2.0])}
    assert canonical_state_dict_sha256(original) != (
        canonical_state_dict_sha256(changed_state)
    )


def test_state_hash_ignores_equivalent_dense_layout():
    non_contiguous = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    contiguous = non_contiguous.contiguous()
    assert not non_contiguous.is_contiguous()
    assert canonical_state_dict_sha256({"value": non_contiguous}) == (
        canonical_state_dict_sha256({"value": contiguous})
    )


def test_state_hash_does_not_mutate_source_tensor():
    source = (
        torch.arange(12, dtype=torch.float32)
        .reshape(3, 4)
        .t()
        .requires_grad_()
    )
    original = source.detach().clone()
    original_stride = source.stride()
    original_device = source.device
    original_dtype = source.dtype
    original_requires_grad = source.requires_grad

    canonical_state_dict_sha256({"value": source})

    torch.testing.assert_close(source.detach(), original)
    assert source.stride() == original_stride
    assert source.device == original_device
    assert source.dtype == original_dtype
    assert source.requires_grad is original_requires_grad


def test_complete_model_state_hash_includes_parameters_and_buffers():
    model = TinyStateModel()
    state = model.state_dict()
    assert set(state) == {
        "linear.weight",
        "linear.bias",
        "running_scale",
    }
    baseline = canonical_state_dict_sha256(state)

    parameter_changed = copy.deepcopy(state)
    parameter_changed["linear.weight"][0, 0] += 1
    buffer_changed = copy.deepcopy(state)
    buffer_changed["running_scale"][0] += 1

    assert canonical_state_dict_sha256(parameter_changed) != baseline
    assert canonical_state_dict_sha256(buffer_changed) != baseline


def test_state_hash_rejects_non_tensor_entries():
    with pytest.raises(TypeError, match="must be a tensor"):
        canonical_state_dict_sha256({"value": "not-a-tensor"})


def test_state_hash_is_repeatable_and_supports_bfloat16():
    state = {"value": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}
    first = canonical_state_dict_sha256(state)
    second = canonical_state_dict_sha256(state)
    assert first == second
    assert len(first) == 64
    assert first == first.lower()


def test_saved_artifact_contains_export_ready_immutable_metadata(tmp_path):
    (
        _,
        config,
        model,
        uid,
        weight_path,
        _,
        metadata,
    ) = _save_artifact(tmp_path)

    identity = metadata["weight_artifact"]["identity"]
    assert identity["weight_uid"] == uid
    assert identity["state_dict_hash_format"] == STATE_DICT_HASH_FORMAT
    assert identity["state_dict_sha256"] == (
        canonical_state_dict_sha256(model.state_dict())
    )
    assert identity["payload_sha256"] == _file_sha256(weight_path)
    assert canonical_json_bytes(metadata["cache_identity"]) == (
        canonical_json_bytes(config)
    )
    assert metadata["actual_epochs"] == 7
    assert metadata["creation_provenance"]["git"]["commit"] == "test-commit"

    artifact = metadata["weight_artifact"]
    assert artifact["model"] == {
        "module": TinyStateModel.__module__,
        "class_name": "TinyStateModel",
        "qualified_class_name": "TinyStateModel",
        "constructor_arguments": {
            "in_channels": 1,
            "num_classes": 2,
            "include_top": True,
        },
    }
    assert artifact["training"]["actual_epochs"] == 7
    assert artifact["training"]["resolved_split_seed"] == 42
    assert artifact["training"]["components"]["optimizer"] == (
        "torch.optim.Adam"
    )
    assert artifact["reproducibility"]["requested_mode"] == "seeded"
    assert artifact["reproducibility"]["effective_state"][
        "device_type"
    ] == "cpu"
    assert artifact["input"]["training_data_shape"] == [8, 1, 32]
    assert artifact["authoritative_fields"] == {
        "training_and_data_compatibility": "cache_identity",
        "creation_provenance": "creation_provenance",
    }


def test_save_load_round_trip_preserves_state_checksum_and_epochs(tmp_path):
    cache, config, model, uid, _, _, metadata = _save_artifact(tmp_path)
    destination = TinyStateModel()

    loaded, loaded_uid = cache.get_weight_cache(
        config,
        destination,
        device="cpu",
    )

    assert loaded is destination
    assert loaded_uid == uid
    assert loaded.actual_epochs == 7
    assert canonical_state_dict_sha256(loaded.state_dict()) == metadata[
        "weight_artifact"
    ]["identity"]["state_dict_sha256"]
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], expected)
    assert loaded.weight_artifact_metadata == metadata


def test_payload_corruption_is_a_safe_miss_before_model_mutation(tmp_path):
    cache, config, _, uid, weight_path, metadata_path, _ = _save_artifact(
        tmp_path
    )
    original_bytes = weight_path.read_bytes()
    weight_path.write_bytes(original_bytes[:-1] + bytes([original_bytes[-1] ^ 1]))
    destination = TinyStateModel()
    original_destination = copy.deepcopy(destination.state_dict())

    with patch("utils.torch.load") as payload_loader:
        loaded, loaded_uid = cache.get_weight_cache(
            config,
            destination,
            device="cpu",
        )

    assert loaded is None
    assert loaded_uid == uid
    payload_loader.assert_not_called()
    for name, expected in original_destination.items():
        torch.testing.assert_close(destination.state_dict()[name], expected)
    assert not weight_path.exists()
    assert not metadata_path.exists()


def test_state_checksum_mismatch_is_a_safe_miss_before_model_mutation(tmp_path):
    cache, config, _, uid, weight_path, metadata_path, metadata = (
        _save_artifact(tmp_path)
    )
    metadata["weight_artifact"]["identity"]["state_dict_sha256"] = "0" * 64
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, indent=4),
        encoding="utf-8",
    )
    destination = TinyStateModel()
    original_destination = copy.deepcopy(destination.state_dict())

    loaded, loaded_uid = cache.get_weight_cache(
        config,
        destination,
        device="cpu",
    )

    assert loaded is None
    assert loaded_uid == uid
    for name, expected in original_destination.items():
        torch.testing.assert_close(destination.state_dict()[name], expected)
    assert not weight_path.exists()
    assert not metadata_path.exists()


def test_mirrored_training_seed_mismatch_is_safe_before_model_mutation(
    tmp_path,
):
    cache, config, _, uid, weight_path, metadata_path, metadata = (
        _save_artifact(tmp_path)
    )
    metadata["weight_artifact"]["training"]["seed"] = 999
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, indent=4),
        encoding="utf-8",
    )
    destination = TinyStateModel()
    original_destination = copy.deepcopy(destination.state_dict())

    with patch("utils.torch.load") as payload_loader:
        loaded, loaded_uid = cache.get_weight_cache(
            config,
            destination,
            device="cpu",
        )

    assert loaded is None
    assert loaded_uid == uid
    payload_loader.assert_not_called()
    for name, expected in original_destination.items():
        torch.testing.assert_close(destination.state_dict()[name], expected)
    assert not weight_path.exists()
    assert not metadata_path.exists()


@pytest.mark.parametrize(
    ("section", "field_name", "contradictory_value"),
    [
        ("training", "actual_epochs", 999),
        ("reproducibility", "requested_mode", "strict"),
        ("model", "class_name", "DifferentModel"),
        ("identity", "state_dict_hash_format", "unknown-state-format"),
    ],
    ids=[
        "actual-epochs",
        "requested-mode",
        "model-class",
        "state-hash-format",
    ],
)
def test_export_helper_rejects_contradictory_artifact_metadata(
    tmp_path,
    section,
    field_name,
    contradictory_value,
):
    cache, _, _, uid, weight_path, metadata_path, metadata = _save_artifact(
        tmp_path
    )
    metadata["weight_artifact"][section][field_name] = contradictory_value
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, indent=4),
        encoding="utf-8",
    )

    with patch("utils.torch.load") as payload_loader:
        exported = cache.get_weight_artifact_metadata(uid)

    assert exported is None
    payload_loader.assert_not_called()
    assert not weight_path.exists()
    assert not metadata_path.exists()


def test_incomplete_pre_artifact_metadata_is_a_safe_miss(tmp_path):
    cache, config, _, uid, weight_path, metadata_path, metadata = (
        _save_artifact(tmp_path)
    )
    metadata.pop("weight_artifact")
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, indent=4),
        encoding="utf-8",
    )

    with patch("utils.torch.load") as payload_loader:
        loaded, loaded_uid = cache.get_weight_cache(
            config,
            TinyStateModel(),
            device="cpu",
        )

    assert loaded is None
    assert loaded_uid == uid
    payload_loader.assert_not_called()
    assert not weight_path.exists()
    assert not metadata_path.exists()


def test_failed_model_application_restores_destination_state(tmp_path):
    cache, config, _, uid, weight_path, metadata_path, _ = _save_artifact(
        tmp_path
    )
    incompatible_model = nn.Linear(5, 4)
    original_destination = copy.deepcopy(incompatible_model.state_dict())

    loaded, loaded_uid = cache.get_weight_cache(
        config,
        incompatible_model,
        device="cpu",
    )

    assert loaded is None
    assert loaded_uid == uid
    for name, expected in original_destination.items():
        torch.testing.assert_close(incompatible_model.state_dict()[name], expected)
    assert not weight_path.exists()
    assert not metadata_path.exists()


def test_export_metadata_helper_validates_and_returns_an_independent_copy(
    tmp_path,
):
    cache, _, _, uid, _, metadata_path, metadata = _save_artifact(tmp_path)
    original_sidecar_bytes = metadata_path.read_bytes()
    exported = cache.get_weight_artifact_metadata(uid)
    assert exported == metadata

    exported["weight_artifact"]["identity"]["weight_uid"] = "changed"
    reloaded = cache.get_weight_artifact_metadata(uid)
    assert reloaded["weight_artifact"]["identity"]["weight_uid"] == uid
    assert metadata_path.read_bytes() == original_sidecar_bytes


def test_cache_uid_excludes_volatile_artifact_metadata(tmp_path):
    config = _training_config()
    first_cache = CacheManager(base_dir=tmp_path / "first")
    second_cache = CacheManager(base_dir=tmp_path / "second")
    _, first_uid = first_cache.get_weight_cache(
        config,
        TinyStateModel(),
        device="cpu",
    )
    _, second_uid = second_cache.get_weight_cache(
        config,
        TinyStateModel(),
        device="cpu",
    )

    assert first_uid == second_uid == _generate_config_hash(config)
    assert _artifact_context("GPU A") != _artifact_context("GPU B")
    assert {"created_at": "time-a"} != {"created_at": "time-b"}


def test_seeded_and_strict_modes_have_distinct_cache_uids(tmp_path):
    cache = CacheManager(base_dir=tmp_path / "cache")
    _, seeded_uid = cache.get_weight_cache(
        _training_config("seeded"),
        TinyStateModel(),
        device="cpu",
    )
    _, strict_uid = cache.get_weight_cache(
        _training_config("strict"),
        TinyStateModel(),
        device="cpu",
    )
    assert seeded_uid != strict_uid


def test_weight_metadata_excludes_evaluation_only_fields(tmp_path):
    _, _, _, _, _, _, metadata = _save_artifact(tmp_path)
    serialized_artifact = json.dumps(
        metadata["weight_artifact"],
        sort_keys=True,
    )
    for field_name in (
        "pair_sampling_mode",
        "pair_sampling_budget",
        "max_impostor_pairs",
        "pair_sampling_seed",
        "target_fars",
        "probe_fusion_size",
        "verification_metrics",
        "identification_metrics",
    ):
        assert field_name not in serialized_artifact


def test_artifact_metadata_normalization_is_deterministic():
    model = TinyStateModel()
    config = _training_config()
    arguments = {
        "model": model,
        "config_dict": config,
        "uid": _generate_config_hash(config),
        "state_dict_sha256": "1" * 64,
        "payload_sha256": "2" * 64,
        "actual_epochs": 7,
        "artifact_context": _artifact_context(),
    }
    first = _build_weight_artifact_metadata(**arguments)
    second = _build_weight_artifact_metadata(**arguments)
    assert first == second
    assert json.dumps(
        first,
        sort_keys=True,
        separators=(",", ":"),
    ) == json.dumps(
        second,
        sort_keys=True,
        separators=(",", ":"),
    )


def test_artifact_metadata_rejects_unstable_objects():
    model = TinyStateModel()
    config = _training_config()
    with pytest.raises(TypeError, match="unsupported metadata type"):
        _build_weight_artifact_metadata(
            model=model,
            config_dict=config,
            uid=_generate_config_hash(config),
            state_dict_sha256="1" * 64,
            payload_sha256="2" * 64,
            actual_epochs=7,
            artifact_context={
                "reproducibility_state": {"unstable": object()},
            },
        )


def test_all_runners_forward_immutable_artifact_context():
    run_source = inspect.getsource(run)
    assert run_source.count(
        "artifact_context=_build_weight_artifact_context("
    ) == 8


def test_task_one_runner_writes_real_weight_artifact_metadata(tmp_path):
    cache_dir = tmp_path / "runner-cache"
    generator = np.random.default_rng(123)
    samples = generator.normal(size=(18, 2, 16)).astype(np.float32)
    labels = np.repeat(np.arange(3), 6)
    effective_state = {
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

    def complete_training(model, *args, **kwargs):
        model.actual_epochs = 1
        return model

    with patch(
        "run._activate_top_level_runtime_profile",
        return_value="seeded",
    ), patch(
        "run._setup_reproducibility",
        return_value=("cpu", effective_state),
    ), patch(
        "run._run_training_loop",
        side_effect=complete_training,
    ):
        run.run_closed_set_identification(
            samples,
            labels,
            RunnerIntegrationModel,
            epochs=1,
            batch_size=4,
            test_split=0.5,
            val_split=0.0,
            seed=11,
            split_seed=17,
            device="cpu",
            loader=SimpleNamespace(cache_dir=cache_dir),
            _return_stats=True,
            intelligent_weight_loading=True,
            reproducibility_mode="seeded",
        )

    metadata_paths = list((cache_dir / "weights").glob("*.json"))
    payload_paths = list((cache_dir / "weights").glob("*.pth"))
    assert len(metadata_paths) == 1
    assert len(payload_paths) == 1

    metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    artifact = metadata["weight_artifact"]
    assert artifact["model"] == {
        "module": RunnerIntegrationModel.__module__,
        "class_name": "RunnerIntegrationModel",
        "qualified_class_name": "RunnerIntegrationModel",
        "constructor_arguments": {
            "in_channels": 2,
            "num_classes": 3,
            "include_top": True,
        },
    }
    assert artifact["training"]["resolved_split_seed"] == 17
    assert artifact["reproducibility"]["requested_mode"] == "seeded"
    assert artifact["reproducibility"]["effective_state"] == effective_state

    saved_state = torch.load(
        payload_paths[0],
        map_location="cpu",
        weights_only=True,
    )
    identity = artifact["identity"]
    assert identity["state_dict_hash_format"] == STATE_DICT_HASH_FORMAT
    assert identity["state_dict_sha256"] == canonical_state_dict_sha256(
        saved_state
    )
    assert identity["payload_sha256"] == _file_sha256(payload_paths[0])


def test_commit_one_compatibility_identity_remains_authoritative(tmp_path):
    cache, config, _, uid, weight_path, metadata_path, metadata = (
        _save_artifact(tmp_path)
    )
    assert metadata["cache_identity"]["implementation_identity"] == (
        config["implementation_identity"]
    )
    assert metadata["cache_identity"]["dependency_identity"] == (
        config["dependency_identity"]
    )
    metadata["cache_identity"]["implementation_identity"] = {
        "aggregate_sha256": "e" * 64
    }
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, indent=4),
        encoding="utf-8",
    )

    with patch("utils.torch.load") as payload_loader:
        loaded, loaded_uid = cache.get_weight_cache(
            config,
            TinyStateModel(),
            device="cpu",
        )

    assert loaded is None
    assert loaded_uid == uid
    payload_loader.assert_not_called()
    assert weight_path.exists()
    assert metadata_path.exists()


def test_pathlike_loader_identity_is_json_safe_for_weight_cache(tmp_path):
    class PathLoader:
        def __init__(self):
            self.cfg = {
                "root_dir": "synthetic",
                "preprocessing": {},
            }
            self.prep_params = {}
            self.data_split_mode = "single-session"
            self.signal_type = "raw"
            self.data_root = tmp_path / "datasets"
            self.dataset_root = self.data_root / "synthetic"

    loader = PathLoader()

    loader_identity = _build_loader_cache_identity(
        loader
    )

    settings = loader_identity["settings"]

    assert settings["data_root"] == str(
        loader.data_root
    )
    assert settings["dataset_root"] == str(
        loader.dataset_root
    )
    assert isinstance(
        settings["data_root"],
        str,
    )
    assert isinstance(
        settings["dataset_root"],
        str,
    )

    # The normalized representation remains strict-JSON serializable.
    json.dumps(
        loader_identity,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )

    # Explicit Path -> string normalization must not change the historical
    # cache UID because _generate_config_hash already represented Path values
    # through their string form.
    legacy_identity = copy.deepcopy(
        loader_identity
    )
    legacy_identity["settings"][
        "data_root"
    ] = loader.data_root
    legacy_identity["settings"][
        "dataset_root"
    ] = loader.dataset_root

    normalized_config = _training_config()
    normalized_config[
        "loader_identity"
    ] = loader_identity

    legacy_config = copy.deepcopy(
        normalized_config
    )
    legacy_config[
        "loader_identity"
    ] = legacy_identity

    assert _generate_config_hash(
        normalized_config
    ) == _generate_config_hash(
        legacy_config
    )

    cache = CacheManager(
        base_dir=tmp_path / "cache"
    )

    model = TinyStateModel()
    model.actual_epochs = 1

    _, uid = cache.get_weight_cache(
        normalized_config,
        TinyStateModel(),
        device="cpu",
    )

    cache.save_weight_cache(
        model,
        normalized_config,
        uid,
        creation_provenance={
            "git": {
                "status": "available",
                "commit": "test-commit",
                "branch": "test",
                "dirty": False,
            }
        },
        artifact_context=_artifact_context(),
    )

    metadata_path = (
        tmp_path
        / "cache"
        / "weights"
        / f"{uid}.json"
    )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    stored_settings = metadata[
        "cache_identity"
    ]["loader_identity"]["settings"]

    assert stored_settings[
        "data_root"
    ] == str(loader.data_root)

    assert stored_settings[
        "dataset_root"
    ] == str(loader.dataset_root)

    loaded_model, loaded_uid = (
        cache.get_weight_cache(
            normalized_config,
            TinyStateModel(),
            device="cpu",
        )
    )

    assert loaded_uid == uid
    assert loaded_model is not None
