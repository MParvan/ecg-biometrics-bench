import os
import sys
import pytest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch
import random
import numpy as np

# Adjust path to import from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
import run


_PROHIBITED_CPU_CUDA_APIS = (
    "is_available",
    "is_initialized",
    "synchronize",
    "reset_peak_memory_stats",
    "max_memory_allocated",
    "get_device_name",
    "manual_seed_all",
)


@contextmanager
def _assert_no_cuda_calls():
    mocks = {}
    patchers = [
        patch(f"torch.cuda.{api_name}")
        for api_name in _PROHIBITED_CPU_CUDA_APIS
    ]
    try:
        for api_name, patcher in zip(
            _PROHIBITED_CPU_CUDA_APIS,
            patchers,
        ):
            mocks[api_name] = patcher.start()
        yield mocks
    finally:
        for patcher in reversed(patchers):
            patcher.stop()
        for cuda_mock in mocks.values():
            cuda_mock.assert_not_called()


@pytest.fixture(autouse=True)
def _restore_process_global_state():
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.default_generator.get_state()
    deterministic_enabled = (
        torch.are_deterministic_algorithms_enabled()
    )
    deterministic_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    experiment_start_time = run._EXPERIMENT_START_TIME
    entrypoint_profile_pending = (
        run._ENTRYPOINT_PROFILE_PENDING_RUNNER
    )
    profiling_device_type = run._ACTIVE_PROFILING_DEVICE_TYPE
    runtime_stage_totals = dict(run._RUNTIME_STAGE_TOTALS)
    runtime_stage_counts = dict(run._RUNTIME_STAGE_COUNTS)
    runtime_run_times = list(run._RUNTIME_RUN_TIMES)

    yield

    random.setstate(python_rng_state)
    np.random.set_state(numpy_rng_state)
    torch.default_generator.set_state(torch_rng_state)
    torch.use_deterministic_algorithms(
        deterministic_enabled,
        warn_only=deterministic_warn_only,
    )
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
    run._EXPERIMENT_START_TIME = experiment_start_time
    run._ENTRYPOINT_PROFILE_PENDING_RUNNER = (
        entrypoint_profile_pending
    )
    run._ACTIVE_PROFILING_DEVICE_TYPE = profiling_device_type
    run._RUNTIME_STAGE_TOTALS.clear()
    run._RUNTIME_STAGE_TOTALS.update(runtime_stage_totals)
    run._RUNTIME_STAGE_COUNTS.clear()
    run._RUNTIME_STAGE_COUNTS.update(runtime_stage_counts)
    run._RUNTIME_RUN_TIMES[:] = runtime_run_times


class MockModel:
    __name__ = "MockModel"

    def __init__(self, *args, **kwargs):
        self.actual_epochs = 1

    def to(self, *args, **kwargs):
        return self

    def __call__(self, *args, **kwargs):
        batch_size = len(args[0]) if args else 1
        return torch.zeros(batch_size, 2)

    def parameters(self):
        return [torch.nn.Parameter(torch.zeros(1))]

    def buffers(self):
        return []

    def eval(self):
        return self


def _run_representative_single_cpu(cache_dir):
    x = np.random.default_rng(42).random((10, 64))
    y = np.tile(np.arange(2), 5)

    with patch(
        "run._run_training_loop",
        return_value=MockModel(),
    ), patch(
        "run._save_compact_evaluation_outputs",
        return_value=([], []),
    ), patch(
        "run._log_experiment_results",
    ), patch(
        "run._compute_metrics_identification",
        return_value=(1.0, 1.0),
    ), patch(
        "run._build_identification_curve_artifacts",
    ), patch(
        "run._build_loader_cache_identity",
        return_value="mock_uid",
    ), patch(
        "utils.CacheManager.get_weight_cache",
        return_value=(None, "mock_uid"),
    ), patch(
        "utils.CacheManager.save_weight_cache",
    ):
        return run.run_closed_set_identification(
            x,
            y,
            MockModel,
            epochs=1,
            device="cpu",
            test_split=0.5,
            loader=SimpleNamespace(cache_dir=cache_dir),
        )


def test_normalize_reproducibility_mode():
    assert utils._normalize_reproducibility_mode("seeded") == "seeded"
    assert utils._normalize_reproducibility_mode("strict") == "strict"
    with pytest.raises(ValueError):
        utils._normalize_reproducibility_mode("unknown")


@patch("torch.cuda.is_available", return_value=False)
def test_prepare_backend_cpu(mock_is_available):
    # Should not query CUDA
    utils._prepare_reproducibility_backend("strict", "cpu")
    mock_is_available.assert_not_called()


@patch("torch.cuda.is_initialized", return_value=False)
def test_prepare_backend_cuda_strict(mock_is_initialized):
    with patch.dict(os.environ, {}, clear=True):
        utils._prepare_reproducibility_backend("strict", "cuda")
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


@patch("torch.cuda.is_initialized", return_value=True)
def test_prepare_backend_cuda_strict_already_initialized(mock_is_initialized):
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError, match="initialized without a supported"):
            utils._prepare_reproducibility_backend("strict", "cuda")


@patch("torch.cuda.is_initialized", return_value=False)
def test_prepare_backend_cuda_strict_preserves_16_8(mock_is_initialized):
    with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}, clear=True):
        utils._prepare_reproducibility_backend("strict", "cuda")
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_set_seed_cpu_uses_real_cpu_generator_without_cuda():
    with patch("torch.cuda.manual_seed_all") as mock_torch_cuda:
        utils._set_seed(42, device_type="cpu")
        first_torch_draw = torch.rand(4)
        first_python_draw = random.random()
        first_numpy_draw = np.random.random()

        utils._set_seed(42, device_type="cpu")
        second_torch_draw = torch.rand(4)
        second_python_draw = random.random()
        second_numpy_draw = np.random.random()

    torch.testing.assert_close(first_torch_draw, second_torch_draw)
    assert first_python_draw == second_python_draw
    assert first_numpy_draw == second_numpy_draw
    mock_torch_cuda.assert_not_called()


def test_set_seed_cuda_seeds_cuda_once():
    with patch("torch.cuda.manual_seed_all") as mock_torch_cuda:
        utils._set_seed(42, device_type="cuda")

    mock_torch_cuda.assert_called_once_with(42)


@patch("torch.are_deterministic_algorithms_enabled", return_value=True)
@patch("torch.is_deterministic_algorithms_warn_only_enabled", return_value=False)
@patch("torch.use_deterministic_algorithms")
def test_configure_backend_cuda_strict(mock_use, mock_warn, mock_enabled):
    with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, clear=True):
        utils._configure_reproducibility_backend("strict", "cuda")
        mock_use.assert_called_with(True, warn_only=False)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False


@patch("torch.are_deterministic_algorithms_enabled", return_value=False)
@patch("torch.use_deterministic_algorithms")
def test_configure_backend_cuda_seeded(mock_use, mock_enabled):
    utils._configure_reproducibility_backend("seeded", "cuda")
    mock_use.assert_called_with(False, warn_only=False)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is False


@patch("torch.are_deterministic_algorithms_enabled", return_value=True)
@patch("torch.is_deterministic_algorithms_warn_only_enabled", return_value=False)
@patch("torch.use_deterministic_algorithms")
def test_configure_backend_cpu_strict(mock_use, mock_warn, mock_enabled):
    # Should not touch cudnn flags
    orig_det = torch.backends.cudnn.deterministic
    utils._configure_reproducibility_backend("strict", "cpu")
    assert torch.backends.cudnn.deterministic == orig_det


@patch("torch.are_deterministic_algorithms_enabled", return_value=False)
@patch("torch.use_deterministic_algorithms")
def test_configure_backend_cpu_seeded(mock_use, mock_enabled):
    orig_det = torch.backends.cudnn.deterministic
    utils._configure_reproducibility_backend("seeded", "cpu")
    assert torch.backends.cudnn.deterministic == orig_det


@patch("torch.are_deterministic_algorithms_enabled", return_value=False)
def test_collect_reproducibility_state_cpu(mock_enabled):
    state = utils._collect_reproducibility_state("strict", "cpu")
    assert state["device_type"] == "cpu"
    assert state["cudnn_deterministic"] is None
    assert state["cudnn_benchmark"] is None
    assert state["cublas_workspace_config"] is None
    assert state["torch_cuda_version"] is None
    assert state["cudnn_version"] is None

def test_direct_single_cpu_avoids_cuda_and_starts_fresh_profile(tmp_path):
    run._ENTRYPOINT_PROFILE_PENDING_RUNNER = False
    run._EXPERIMENT_START_TIME = -1.0
    run._ACTIVE_PROFILING_DEVICE_TYPE = "auto"
    run._RUNTIME_STAGE_TOTALS["stale stage"] = 123.0
    run._RUNTIME_STAGE_COUNTS["stale stage"] = 7
    run._RUNTIME_RUN_TIMES.append(
        {"run_index": 99, "seed": 999, "seconds": 456.0}
    )

    with _assert_no_cuda_calls():
        _run_representative_single_cpu(tmp_path / "cache")

    assert run._EXPERIMENT_START_TIME != -1.0
    assert "stale stage" not in run._RUNTIME_STAGE_TOTALS
    assert "stale stage" not in run._RUNTIME_STAGE_COUNTS
    assert all(
        entry["run_index"] != 99
        for entry in run._RUNTIME_RUN_TIMES
    )
    assert run._ACTIVE_PROFILING_DEVICE_TYPE == "cpu"


def test_direct_multi_run_cpu_sets_profile_before_outer_timing():
    x = np.random.default_rng(42).random((10, 64))
    y = np.tile(np.arange(2), 5)
    outer_runner = run.run_closed_set_identification
    recursive_results = [
        ((0.5, 0.6), {}, {}),
        ((0.7, 0.8), {}, {}),
    ]

    def recursive_run_double(**kwargs):
        run._record_trained_weight_reference(
            {
                "persisted": False,
                "source": "trained_not_persisted",
                "weight_uid": None,
                "state_dict_hash_format": run.STATE_DICT_HASH_FORMAT,
                "state_dict_sha256": "0" * 64,
                "payload_sha256": None,
            }
        )
        return recursive_results.pop(0)

    run._ENTRYPOINT_PROFILE_PENDING_RUNNER = False
    run._ACTIVE_PROFILING_DEVICE_TYPE = "auto"

    with _assert_no_cuda_calls(), patch.object(
        run,
        "run_closed_set_identification",
        side_effect=recursive_run_double,
    ) as recursive_runner, patch.object(
        run,
        "_apply_identification_metric_reportability",
        return_value=[(0.5, 0.6), (0.7, 0.8)],
    ), patch.object(
        run,
        "_build_per_run_evaluation_artifacts",
        return_value=[],
    ):
        result = outer_runner(
            x,
            y,
            MockModel,
            device="cpu",
            n_runs=2,
        )

    assert recursive_runner.call_count == 2
    np.testing.assert_allclose(
        np.asarray(result, dtype=float),
        np.asarray(((0.6, 0.1), (0.7, 0.1))),
    )
    assert run._ACTIVE_PROFILING_DEVICE_TYPE == "cpu"


def test_cli_started_profile_is_adopted_without_reset(tmp_path):
    run.start_experiment_timer(device="cpu")
    cli_start_time = run._EXPERIMENT_START_TIME
    run._RUNTIME_STAGE_TOTALS["Data Preparation (inclusive)"] = 12.5
    run._RUNTIME_STAGE_COUNTS["Data Preparation (inclusive)"] = 1

    with _assert_no_cuda_calls():
        _run_representative_single_cpu(tmp_path / "cache")

    assert run._EXPERIMENT_START_TIME == cli_start_time
    assert run._RUNTIME_STAGE_TOTALS[
        "Data Preparation (inclusive)"
    ] == 12.5
    assert run._RUNTIME_STAGE_COUNTS[
        "Data Preparation (inclusive)"
    ] == 1
    assert run._ENTRYPOINT_PROFILE_PENDING_RUNNER is False


def test_recursive_run_does_not_reset_outer_profile():
    run._initialize_runtime_profile(device="cpu")
    outer_start_time = run._EXPERIMENT_START_TIME
    run._RUNTIME_STAGE_TOTALS["outer stage"] = 3.0
    run._RUNTIME_STAGE_COUNTS["outer stage"] = 1
    run._RUNTIME_RUN_TIMES.append(
        {"run_index": 1, "seed": 42, "seconds": 2.0}
    )

    with _assert_no_cuda_calls():
        run._activate_top_level_runtime_profile(
            "seeded",
            "cpu",
            recursive_run=True,
        )

    assert run._EXPERIMENT_START_TIME == outer_start_time
    assert run._RUNTIME_STAGE_TOTALS["outer stage"] == 3.0
    assert run._RUNTIME_STAGE_COUNTS["outer stage"] == 1
    assert run._RUNTIME_RUN_TIMES == [
        {"run_index": 1, "seed": 42, "seconds": 2.0}
    ]


@pytest.mark.parametrize(
    ("first_mode", "second_mode"),
    [
        ("seeded", "seeded"),
        ("seeded", "strict"),
        ("strict", "seeded"),
        ("strict", "strict"),
    ],
)
def test_same_process_deterministic_policy_transitions(
    first_mode,
    second_mode,
):
    original_enabled = torch.are_deterministic_algorithms_enabled()
    original_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    try:
        with _assert_no_cuda_calls():
            _, first_state = utils._setup_reproducibility(
                seed=41,
                device="cpu",
                reproducibility_mode=first_mode,
            )
            _, second_state = utils._setup_reproducibility(
                seed=42,
                device="cpu",
                reproducibility_mode=second_mode,
            )

        assert first_state[
            "deterministic_algorithms_enabled"
        ] is (first_mode == "strict")
        assert second_state[
            "deterministic_algorithms_enabled"
        ] is (second_mode == "strict")
    finally:
        torch.use_deterministic_algorithms(
            original_enabled,
            warn_only=original_warn_only,
        )


def test_weight_cache_uid_differs_by_reproducibility_mode(tmp_path):
    training_samples = np.arange(12, dtype=np.float32).reshape(3, 4)
    training_labels = np.array([0, 1, 0], dtype=np.int64)
    compatibility_identity = {
        "implementation": {"aggregate_sha256": "implementation"},
        "dependencies": {"aggregate_sha256": "dependencies"},
    }

    with patch.object(
        run,
        "build_weight_compatibility_identity",
        return_value=compatibility_identity,
    ):
        seeded_config = run._build_weight_cache_config(
            loader=None,
            training_config={
                "model": "MockModel",
                "reproducibility_mode": "seeded",
            },
            training_samples=training_samples,
            training_labels=training_labels,
        )
        strict_config = run._build_weight_cache_config(
            loader=None,
            training_config={
                "model": "MockModel",
                "reproducibility_mode": "strict",
            },
            training_samples=training_samples,
            training_labels=training_labels,
        )

    cache = utils.CacheManager(base_dir=tmp_path)
    model = torch.nn.Linear(1, 1)
    seeded_model, seeded_uid = cache.get_weight_cache(
        seeded_config,
        model,
        "cpu",
    )
    strict_model, strict_uid = cache.get_weight_cache(
        strict_config,
        model,
        "cpu",
    )

    assert seeded_model is None
    assert strict_model is None
    assert seeded_uid != strict_uid
    assert utils._generate_config_hash(
        seeded_config
    ) == seeded_uid
    assert utils._generate_config_hash(
        strict_config
    ) == strict_uid
    assert "reproducibility_state" not in seeded_config
    assert "reproducibility_state" not in strict_config


def test_requested_auto_and_effective_cpu_metadata_are_distinct():
    with patch(
        "torch.cuda.is_available",
        return_value=False,
    ) as mock_is_available, patch(
        "torch.cuda.manual_seed_all",
    ) as mock_manual_seed_all:
        resolved_device, state = utils._setup_reproducibility(
            seed=42,
            device="auto",
            reproducibility_mode="seeded",
        )

    assert resolved_device == "cpu"
    assert state["requested_device"] == "auto"
    assert state["device_type"] == "cpu"
    assert state["device_name"] == "cpu"
    mock_is_available.assert_called_once_with()
    mock_manual_seed_all.assert_not_called()


def test_cpu_software_environment_cuda_fields():
    run._ACTIVE_PROFILING_DEVICE_TYPE = "cpu"
    env = run._collect_software_environment()
    assert env["CUDA Available"] is None
    assert env["CUDA Runtime"] is None
    assert env["CUDA Device"] is None
