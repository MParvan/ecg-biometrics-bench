import argparse
import copy
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


SUPPORTED_METHODS = (
    "gaussian",
    "amplitude",
    "timeshift",
    "baseline_wander",
    "time_warp",
    "cutout",
    "emg_noise",
    "istft_augment",
)

DEFAULT_METHODS = (
    "gaussian",
    "amplitude",
    "timeshift",
    "time_warp",
    "cutout",
)

DEFAULT_METHOD_PARAMETERS = {
    "gaussian": {
        "std": 0.02,
        "relative": True,
    },
    "amplitude": {
        "scale_range": [0.9, 1.1],
    },
    "timeshift": {
        "max_shift": 10,
    },
    "baseline_wander": {
        "freq": 0.3,
        "amp": 0.05,
    },
    "time_warp": {
        "sigma": 0.1,
        "num_knots": 4,
    },
    "cutout": {
        "num_holes": 1,
        "length": 20,
    },
    "emg_noise": {
        "std": 0.03,
    },
    "istft_augment": {
        "nperseg": 128,
        "noverlap": 64,
        "noise_std": 0.02,
        "log_power": True,
    },
}

METHODS_REQUIRING_SAMPLING_FREQUENCY = {
    "baseline_wander",
    "emg_noise",
    "istft_augment",
}



def _load_yaml_mapping(path):
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        content = yaml.safe_load(file) or {}

    if not isinstance(content, dict):
        raise ValueError(
            f"Configuration file must contain a mapping: {path}"
        )

    return content



def _normalize_methods(methods):
    normalized = []

    for method in methods:
        if not isinstance(method, str):
            raise ValueError(
                "Every augmentation method must be a string."
            )

        method = (
            method
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported augmentation method {method!r}."
            )

        if method not in normalized:
            normalized.append(method)

    if not normalized:
        raise ValueError(
            "At least one augmentation method is required."
        )

    return normalized



def _load_parameter_overrides(path):
    if path is None:
        return {}

    overrides = _load_yaml_mapping(path)
    normalized = {}

    for method, parameters in overrides.items():
        normalized_method = _normalize_methods(
            [method]
        )[0]

        if not isinstance(parameters, dict):
            raise ValueError(
                "Each augmentation-plan entry must map a method "
                "to a parameter mapping."
            )

        normalized[normalized_method] = dict(
            parameters
        )

    return normalized



def _get_method_parameters(
    method,
    parameter_overrides,
    sampling_frequency,
):
    parameters = copy.deepcopy(
        DEFAULT_METHOD_PARAMETERS[method]
    )

    if method in parameter_overrides:
        parameters.update(
            parameter_overrides[method]
        )

    if method in METHODS_REQUIRING_SAMPLING_FREQUENCY:
        if sampling_frequency is None:
            raise ValueError(
                f"Augmentation method {method!r} requires "
                "--sampling-frequency."
            )

        parameters["fs"] = float(
            sampling_frequency
        )

    return parameters



def _build_variant_config(
    base_config,
    method,
    copies,
    n_runs,
    parameters=None,
):
    config = copy.deepcopy(
        base_config
    )

    enabled = method is not None

    config.update(
        {
            "use_augmentation": enabled,
            "augmentation_method": (
                method
                if enabled
                else "gaussian"
            ),
            "augmentation_copies": int(copies),
            "augmentation_parameters": (
                dict(parameters or {})
                if enabled
                else {}
            ),
            "visualize": False,
            "save_results_and_settings": True,
            "intelligent_weight_loading": False,
        }
    )

    if n_runs is not None:
        config["n_runs"] = int(
            n_runs
        )

    return config



def _snapshot_result_logs(results_root):
    results_root = Path(
        results_root
    )

    if not results_root.exists():
        return {}

    return {
        path.resolve(): path.stat().st_mtime_ns
        for path in results_root.rglob("*.txt")
        if path.is_file()
    }



def _find_changed_result_log(
    before_snapshot,
    results_root,
):
    results_root = Path(
        results_root
    )

    if not results_root.exists():
        return None

    changed = []

    for path in results_root.rglob(
        "*.txt"
    ):
        if not path.is_file():
            continue

        resolved_path = path.resolve()
        modified_time = path.stat().st_mtime_ns
        previous_time = before_snapshot.get(
            resolved_path
        )

        if (
            previous_time is None
            or modified_time > previous_time
        ):
            changed.append(
                (
                    modified_time,
                    resolved_path,
                )
            )

    if not changed:
        return None

    changed.sort(
        key=lambda item: item[0]
    )

    return changed[-1][1]



def _parse_last_section(
    path,
    section_name,
):
    path = Path(path)
    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    section_header = (
        f"[{section_name}]"
    )

    matching_indices = [
        index
        for index, line in enumerate(
            lines
        )
        if line.strip() == section_header
    ]

    if not matching_indices:
        raise ValueError(
            f"Section {section_header} was not found in {path}."
        )

    values = {}

    for line in lines[
        matching_indices[-1] + 1:
    ]:
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("["):
            break

        if set(stripped) <= {
            "-",
            "=",
        }:
            break

        if ":" not in stripped:
            continue

        key, value = stripped.split(
            ":",
            1,
        )

        values[
            key.strip()
        ] = value.strip()

    return values



def _safe_scenario_name(value):
    return "".join(
        character
        if character.isalnum()
        or character in {
            "-",
            "_",
        }
        else "_"
        for character in value
    )



def _write_summary(
    rows,
    output_directory,
):
    output_directory = Path(
        output_directory
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_keys = sorted(
        {
            metric_name
            for row in rows
            for metric_name in row.get(
                "metrics",
                {},
            )
        }
    )

    fixed_fields = [
        "scenario",
        "use_augmentation",
        "augmentation_method",
        "augmentation_copies",
        "n_runs",
        "status",
        "return_code",
        "duration_seconds",
        "result_log",
        "console_log",
    ]

    csv_path = (
        output_directory
        / "summary.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                fixed_fields
                + metric_keys
            ),
        )
        writer.writeheader()

        for row in rows:
            flat_row = {
                field: row.get(
                    field,
                    "",
                )
                for field in fixed_fields
            }

            flat_row.update(
                row.get(
                    "metrics",
                    {},
                )
            )

            writer.writerow(
                flat_row
            )

    json_path = (
        output_directory
        / "summary.json"
    )

    json_path.write_text(
        json.dumps(
            rows,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return (
        csv_path,
        json_path,
    )



def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run a baseline and controlled ECG augmentation "
            "ablation through main.py."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name accepted by main.py.",
    )

    parser.add_argument(
        "--task",
        required=True,
        type=int,
        choices=range(1, 9),
        help="Biometric task number from 1 to 8.",
    )

    parser.add_argument(
        "--config",
        default="experiment_settings.yaml",
        help="Base experiment YAML file.",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(
            DEFAULT_METHODS
        ),
        help=(
            "Augmentation methods to compare. The baseline is "
            "included automatically unless --skip-baseline is used."
        ),
    )

    parser.add_argument(
        "--copies",
        type=int,
        default=1,
        help="Augmented copies appended per original training sample.",
    )

    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
        help="Override n_runs from the base YAML.",
    )

    parser.add_argument(
        "--sampling-frequency",
        type=float,
        default=None,
        help=(
            "Required when baseline_wander, emg_noise, or "
            "istft_augment is selected."
        ),
    )

    parser.add_argument(
        "--augmentation-plan",
        default=None,
        help=(
            "Optional YAML mapping from method names to parameter "
            "overrides."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated configs, logs, and summaries.",
    )

    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Run only augmentation methods without the baseline.",
    )

    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with later methods after one run fails.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate variant YAML files without starting training.",
    )

    return parser



def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.copies < 1:
        parser.error(
            "--copies must be at least 1."
        )

    if (
        args.n_runs is not None
        and args.n_runs < 1
    ):
        parser.error(
            "--n-runs must be at least 1."
        )

    if (
        args.sampling_frequency is not None
        and args.sampling_frequency <= 0.0
    ):
        parser.error(
            "--sampling-frequency must be greater than zero."
        )

    project_root = Path(
        __file__
    ).resolve().parents[1]

    config_path = Path(
        args.config
    )

    if not config_path.is_absolute():
        config_path = (
            project_root
            / config_path
        )

    if not config_path.exists():
        parser.error(
            f"Base configuration file not found: {config_path}"
        )

    try:
        methods = _normalize_methods(
            args.methods
        )

        parameter_overrides = (
            _load_parameter_overrides(
                args.augmentation_plan
            )
        )

        base_config = _load_yaml_mapping(
            config_path
        )
    except ValueError as error:
        parser.error(
            str(error)
        )

    if args.output_dir is None:
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        output_directory = (
            project_root
            / "results"
            / "augmentation_ablation"
            / (
                f"{args.dataset}_task{args.task}_"
                f"{timestamp}"
            )
        )
    else:
        output_directory = Path(
            args.output_dir
        )

        if not output_directory.is_absolute():
            output_directory = (
                project_root
                / output_directory
            )

    config_directory = (
        output_directory
        / "configs"
    )

    console_directory = (
        output_directory
        / "console"
    )

    config_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    console_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    scenarios = []

    if not args.skip_baseline:
        scenarios.append(
            (
                "baseline",
                None,
                {},
            )
        )

    try:
        for method in methods:
            scenarios.append(
                (
                    method,
                    method,
                    _get_method_parameters(
                        method,
                        parameter_overrides,
                        args.sampling_frequency,
                    ),
                )
            )
    except ValueError as error:
        parser.error(
            str(error)
        )

    rows = []
    results_root = (
        project_root
        / "results"
    )

    for (
        scenario,
        method,
        parameters,
    ) in scenarios:
        safe_name = _safe_scenario_name(
            scenario
        )

        variant_config = (
            _build_variant_config(
                base_config,
                method=method,
                copies=args.copies,
                n_runs=args.n_runs,
                parameters=parameters,
            )
        )

        variant_config_path = (
            config_directory
            / f"{safe_name}.yaml"
        )

        variant_config_path.write_text(
            yaml.safe_dump(
                variant_config,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        console_log_path = (
            console_directory
            / f"{safe_name}.log"
        )

        row = {
            "scenario": scenario,
            "use_augmentation": (
                method is not None
            ),
            "augmentation_method": (
                method or "none"
            ),
            "augmentation_copies": (
                args.copies
                if method is not None
                else 0
            ),
            "n_runs": variant_config.get(
                "n_runs",
                "",
            ),
            "status": "dry_run",
            "return_code": "",
            "duration_seconds": 0.0,
            "result_log": "",
            "console_log": str(
                console_log_path
            ),
            "metrics": {},
        }

        print(
            "\n"
            + "=" * 72
        )
        print(
            f"SCENARIO: {scenario}"
        )
        print(
            f"CONFIG:   {variant_config_path}"
        )
        print(
            "=" * 72
        )

        if args.dry_run:
            rows.append(
                row
            )
            continue

        before_snapshot = (
            _snapshot_result_logs(
                results_root
            )
        )

        command = [
            sys.executable,
            str(
                project_root
                / "main.py"
            ),
            "--dataset",
            args.dataset,
            "--task",
            str(args.task),
            "--config",
            str(
                variant_config_path
            ),
        ]

        start_time = time.perf_counter()

        completed = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        duration_seconds = (
            time.perf_counter()
            - start_time
        )

        console_text = (
            completed.stdout
            + (
                "\n[STDERR]\n"
                if completed.stderr
                else ""
            )
            + completed.stderr
        )

        console_log_path.write_text(
            console_text,
            encoding="utf-8",
        )

        changed_result_log = (
            _find_changed_result_log(
                before_snapshot,
                results_root,
            )
        )

        row.update(
            {
                "return_code": completed.returncode,
                "duration_seconds": round(
                    duration_seconds,
                    4,
                ),
                "status": (
                    "success"
                    if completed.returncode == 0
                    else "failed"
                ),
            }
        )

        if changed_result_log is not None:
            row["result_log"] = str(
                changed_result_log
            )

            try:
                row["metrics"] = (
                    _parse_last_section(
                        changed_result_log,
                        "RESULTS",
                    )
                )
            except ValueError as error:
                row["status"] = (
                    "result_parse_failed"
                )
                row["metrics"] = {
                    "Parse Error": str(error),
                }
        elif completed.returncode == 0:
            row["status"] = (
                "missing_result_log"
            )

        rows.append(
            row
        )

        _write_summary(
            rows,
            output_directory,
        )

        print(
            f"Status: {row['status']}"
        )
        print(
            f"Console log: {console_log_path}"
        )

        if row["result_log"]:
            print(
                f"Result log: {row['result_log']}"
            )

        if row["metrics"]:
            print(
                "Metrics:"
            )

            for key, value in (
                row["metrics"].items()
            ):
                print(
                    f"  {key}: {value}"
                )

        if (
            completed.returncode != 0
            and not args.keep_going
        ):
            print(
                "\n[ERROR] Ablation stopped after a failed scenario."
            )
            break

    csv_path, json_path = (
        _write_summary(
            rows,
            output_directory,
        )
    )

    print(
        "\nAblation outputs:"
    )
    print(
        f"  CSV:  {csv_path}"
    )
    print(
        f"  JSON: {json_path}"
    )
    print(
        f"  Configs and console logs: {output_directory}"
    )


if __name__ == "__main__":
    main()
