"""
Run the shipped paper configurations and assemble the reported result tables.

The configurations under ``configs/paper_reproduction`` are the single source
of truth for what was run. This driver discovers them, groups them into the
published result tables, executes the ones you ask for, and rebuilds each
table from the structured JSONL records the runs write.

Every table row combines two configurations: an identification run supplies
Rank-1 and Rank-5, and the matching verification run supplies EER, AUC,
d-prime, and TAR@0.1%FAR. The driver pairs them automatically.

Typical use:

    # See the plan without running anything.
    python -m scripts.reproduce_tables --table 5 --dry-run

    # Reproduce one dataset end to end.
    python -m scripts.reproduce_tables --dataset ecgid --run

    # Rebuild the tables from runs that already finished.
    python -m scripts.reproduce_tables --collect --output-dir reproduced_tables

    # Confirm the wiring on CPU in minutes, not GPU-days.
    python -m scripts.reproduce_tables --dataset ecgid --run --smoke
"""

import argparse
import csv
import json
import contextlib
import io
import re
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import yaml

from experiment_provenance import (
    ResultCollectionError,
    build_experiment_implementation_identity,
    build_scientific_configuration_identity,
    capture_result_log_snapshot,
    collect_appended_result,
    read_legacy_latest_record,
    select_exact_result_record,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_ROOT = PROJECT_ROOT / "configs" / "paper_reproduction"

# Table grouping. Datasets are tabulated together when they share structural
# characteristics, which is why the continuous recordings and the
# session-structured datasets each have their own table.
TABLE_BY_DATASET = OrderedDict(
    [
        ("ecgid", 5),
        ("ptb", 5),
        ("ptbxl", 5),
        ("mitbih", 6),
        ("nsrdb", 6),
        ("heartprint", 7),
        ("cybhi", 7),
    ]
)

# Row order within each dataset, from least to most demanding protocol.
PROTOCOL_ORDER = {
    "ecgid": [
        "all_available",
        "single_session",
        "single_cross_session",
        "single_shot_short_term",
        "leave_last_out_short_term",
        "single_shot_long_term",
        "leave_last_out_long_term",
    ],
    "mitbih": [
        "single_segment",
        "short_term",
        "multi_shot",
        "long_term",
    ],
    "cybhi": [
        "baseline_ci",
        "baseline_s1",
        "intervention_a1",
        "intervention_a2",
        "long_term",
    ],
    "heartprint": [
        "single_session",
        "short_term",
        "short_term_reverse",
        "cognitive",
        "cognitive_s1_s2_s3r",
        "long_term",
        "long_term_s1_s2_s3l",
    ],
}
PROTOCOL_ORDER["ptb"] = PROTOCOL_ORDER["ecgid"]
PROTOCOL_ORDER["ptbxl"] = PROTOCOL_ORDER["ecgid"]
PROTOCOL_ORDER["nsrdb"] = PROTOCOL_ORDER["mitbih"]

# Display labels for the protocol slugs used in filenames.
PROTOCOL_LABELS = {
    "all_available": "All available",
    "single_session": "Single session",
    "single_cross_session": "Single cross session",
    "single_shot_short_term": "SS Short-term",
    "leave_last_out_short_term": "LLO Short-term",
    "single_shot_long_term": "SS Long-term",
    "leave_last_out_long_term": "LLO Long-term",
    "single_segment": "Single segment",
    "short_term": "Short-term",
    "short_term_reverse": "Short-term (Rev.)",
    "multi_shot": "Multi-shot",
    "long_term": "Long-term",
    "baseline_ci": "Baseline (CI)",
    "baseline_s1": "Baseline (S1)",
    "intervention_a1": "Intervention (A1)",
    "intervention_a2": "Intervention (A2)",
    "cognitive": "Cognitive (S3R)",
    "cognitive_s1_s2_s3r": "Cognitive (S1+S2 -> S3R)",
    "long_term_s1_s2_s3l": "Long-term (S1+S2 -> S3L)",
}

DATASET_LABELS = {
    "ecgid": "ECG-ID",
    "ptb": "PTB",
    "ptbxl": "PTB-XL",
    "mitbih": "MIT-BIH",
    "nsrdb": "NSRDB",
    "heartprint": "Heartprint",
    "cybhi": "CYBHi",
}

SETTING_LABELS = {
    "closed_set": "Closed set",
    "subject_disjoint": "Subject-disjoint",
}

# The two evaluation settings, reported in this order.
SETTING_ORDER = ("closed_set", "subject_disjoint")

IDENTIFICATION_METRICS = (
    "Rank-1 Accuracy",
    "Rank-5 Accuracy",
)
VERIFICATION_METRICS = (
    "EER",
    "AUC",
    "d-prime",
    "TAR@0.1%FAR",
)
TABLE_METRICS = IDENTIFICATION_METRICS + VERIFICATION_METRICS

# Task name to structured-log filename, as written by run._log_experiment_results.
TASK_LOG_NAMES = {
    1: "Closed-Set_Identification",
    2: "Closed-Set_Verification",
    3: "Subject-Disjoint_Identification",
    4: "Subject-Disjoint_Verification",
    5: "Cross-Session_Identification",
    6: "Cross-Session_Verification",
    7: "Subject-Disjoint_Cross-Session_ID",
    8: "Subject-Disjoint_Cross-Session_Verification",
}

# Settings overridden in --smoke mode so the full plan can be exercised on CPU.
SMOKE_OVERRIDES = {
    "epochs": 1,
    "n_runs": 1,
    "batch_size": 32,
    "num_pairs": 200,
    "device": "cpu",
}

CONFIG_NAME_PATTERN = re.compile(
    r"^(?P<dataset>[a-z]+)_(?P<protocol>.+)_"
    r"(?P<setting>closed_set|subject_disjoint)_"
    r"task(?P<task>\d{2})_(?P<task_type>identification|verification)$"
)


class ConfigurationEntry:
    """
    One shipped configuration, parsed into its table coordinates.
    """

    def __init__(self, path, configuration):
        self.path = path
        self.configuration = configuration

        match = CONFIG_NAME_PATTERN.match(path.stem)

        if match is None:
            raise ValueError(
                f"Configuration name does not follow the expected "
                f"convention: {path.name}"
            )

        self.dataset = match.group("dataset")
        self.protocol = match.group("protocol")
        self.setting = match.group("setting")
        self.task = int(match.group("task"))
        self.task_type = match.group("task_type")

        if self.dataset not in TABLE_BY_DATASET:
            raise ValueError(
                f"Unknown dataset '{self.dataset}' in {path.name}"
            )

        self.table = TABLE_BY_DATASET[self.dataset]

    @property
    def row_key(self):
        return (
            self.dataset,
            self.protocol,
            self.setting,
        )

    @property
    def results_dir(self):
        return self.configuration.get("results_dir")

    def structured_log_path(self, results_dir=None, dataset=None):
        """
        Locate the JSONL record this configuration writes.

        ``run._log_experiment_results`` appends the dataset root directory
        name to the configured results directory, then names the file after
        the task.
        """
        results_dir = self.results_dir if results_dir is None else results_dir

        if not results_dir:
            return None

        base = Path(results_dir)

        if not base.is_absolute():
            base = (PROJECT_ROOT / base).resolve()

        log_name = TASK_LOG_NAMES.get(self.task)

        if log_name is None:
            return None

        # The dataset subdirectory is derived from the loader's root_dir.
        dataset = self.dataset if dataset is None else str(dataset)
        return base / dataset / f"{log_name}.jsonl"


def discover_configurations(config_root=CONFIG_ROOT):
    """
    Load and parse every shipped reproduction configuration.
    """
    if not config_root.exists():
        raise SystemExit(
            f"Configuration directory not found: {config_root}"
        )

    entries = []

    for path in sorted(config_root.rglob("*.yaml")):
        with path.open("r", encoding="utf-8") as config_file:
            configuration = yaml.safe_load(config_file) or {}

        entries.append(
            ConfigurationEntry(path, configuration)
        )

    if not entries:
        raise SystemExit(
            f"No configurations found under {config_root}"
        )

    return entries


def filter_configurations(entries, tables=None, datasets=None, tasks=None):
    """
    Restrict the plan to the requested tables, datasets, or task numbers.
    """
    selected = entries

    if tables:
        selected = [
            entry
            for entry in selected
            if entry.table in set(tables)
        ]

    if datasets:
        selected = [
            entry
            for entry in selected
            if entry.dataset in set(datasets)
        ]

    if tasks:
        selected = [
            entry
            for entry in selected
            if entry.task in set(tasks)
        ]

    return selected


def sort_key(entry):
    """
    Order entries the way the result tables read.
    """
    protocol_order = PROTOCOL_ORDER.get(entry.dataset, [])

    try:
        protocol_index = protocol_order.index(entry.protocol)
    except ValueError:
        protocol_index = len(protocol_order)

    return (
        entry.table,
        list(TABLE_BY_DATASET).index(entry.dataset),
        protocol_index,
        entry.protocol,
        SETTING_ORDER.index(entry.setting),
        entry.task_type,
    )


def build_command(
    entry,
    smoke=False,
    extra_arguments=None,
    campaign_id=None,
):
    """
    Build the command line that reproduces one configuration.
    """
    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--config",
        str(entry.path),
    ]

    if smoke:
        for name, value in SMOKE_OVERRIDES.items():
            command.extend([f"--{name}", str(value)])
        command.append("--smoke_run")

    if campaign_id is not None:
        command.extend(["--campaign_id", str(campaign_id)])

    if extra_arguments:
        command.extend(extra_arguments)

    return command


def _resolve_execution_expectation(
    entry,
    *,
    smoke=False,
    extra_arguments=None,
    campaign_id=None,
    implementation_identity=None,
):
    """Resolve the same effective configuration the launched CLI will use."""
    import main as experiment_main

    command = build_command(
        entry,
        smoke=smoke,
        extra_arguments=extra_arguments,
        campaign_id=campaign_id,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        arguments, _ = experiment_main.parse_experiment_arguments(
            command[2:]
        )
    effective_configuration = experiment_main.build_effective_configuration(
        arguments
    )
    scientific_identity = build_scientific_configuration_identity(
        effective_configuration
    )
    log_path = entry.structured_log_path(
        results_dir=arguments.results_dir,
        dataset=arguments.dataset,
    )
    results_root = Path(arguments.results_dir)
    if not results_root.is_absolute():
        results_root = (PROJECT_ROOT / results_root).resolve()
    if implementation_identity is None:
        implementation_identity = build_experiment_implementation_identity()
    return {
        "command": command,
        "effective_configuration": effective_configuration,
        "scientific_identity": scientific_identity,
        "implementation_identity": implementation_identity,
        "log_path": log_path,
        "results_root": results_root,
    }


def collect_configuration_record(
    entry,
    *,
    campaign_id=None,
    publication_mode=True,
    smoke=False,
    extra_arguments=None,
    implementation_identity=None,
):
    """Select the one exact result belonging to a shipped configuration."""
    if not publication_mode:
        log_path = entry.structured_log_path()
        record = read_legacy_latest_record(log_path)
        if record is None:
            raise ResultCollectionError(
                f"No exploratory result record exists: {log_path}"
            )
        return {
            "record": record,
            "locator": None,
        }

    expectation = _resolve_execution_expectation(
        entry,
        smoke=smoke,
        extra_arguments=extra_arguments,
        campaign_id=campaign_id,
        implementation_identity=implementation_identity,
    )
    return select_exact_result_record(
        expectation["log_path"],
        result_root=expectation["results_root"],
        expected_scientific_sha256=(
            expectation["scientific_identity"]["sha256"]
        ),
        expected_implementation=expectation["implementation_identity"],
        required_campaign_id=campaign_id,
        publication_mode=True,
        expected_smoke_run=smoke,
    )


def run_configurations(entries, smoke=False, extra_arguments=None,
                       continue_on_error=False, campaign_id=None,
                       return_collected=False):
    """
    Execute the selected configurations sequentially.
    """
    failures = []
    collected_results = []

    implementation_identity = build_experiment_implementation_identity()

    for index, entry in enumerate(entries, start=1):
        expectation = _resolve_execution_expectation(
            entry,
            smoke=smoke,
            extra_arguments=extra_arguments,
            campaign_id=campaign_id,
            implementation_identity=implementation_identity,
        )
        command = expectation["command"]
        snapshot = capture_result_log_snapshot(expectation["log_path"])

        print("=" * 72)
        print(f"[{index}/{len(entries)}] {entry.path.name}")
        print("=" * 72)

        started_at = time.time()
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
        )
        elapsed = time.time() - started_at

        if completed.returncode == 0:
            try:
                collected = collect_appended_result(
                    expectation["log_path"],
                    snapshot,
                    result_root=expectation["results_root"],
                    expected_scientific_sha256=(
                        expectation["scientific_identity"]["sha256"]
                    ),
                    expected_implementation=implementation_identity,
                    required_campaign_id=campaign_id,
                    publication_mode=not smoke,
                    expected_smoke_run=smoke,
                )
            except ResultCollectionError as error:
                failures.append((entry.path.name, "result provenance"))
                print(f"[FAIL] {error}\n")
                if not continue_on_error:
                    break
                continue

            print(
                f"[OK] finished in {elapsed:.1f}s; "
                f"record {collected['locator']['relative_path']}:"
                f"{collected['locator']['line_number']}\n"
            )
            collected_results.append(
                {
                    "configuration": entry.path.name,
                    **collected,
                }
            )
            continue

        failures.append(
            (entry.path.name, completed.returncode)
        )
        print(
            f"[FAIL] exit status {completed.returncode} "
            f"after {elapsed:.1f}s\n"
        )

        if not continue_on_error:
            break

    if return_collected:
        return failures, collected_results
    return failures


def format_metric(value):
    """
    Render a metric for display, or mark it as missing.
    """
    if value is None:
        return "-"

    if isinstance(value, dict):
        if "mean" in value and "std" in value:
            return f"{value['mean']:.4f} +/- {value['std']:.4f}"

        if "display" in value:
            return str(value["display"])

        return "-"

    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"

    return str(value)


def collect_rows(
    entries,
    *,
    campaign_id=None,
    publication_mode=True,
    smoke=False,
    implementation_identity=None,
):
    """
    Pair identification and verification runs into result table rows.
    """
    rows = OrderedDict()

    for entry in sorted(entries, key=sort_key):
        row = rows.setdefault(
            entry.row_key,
            {
                "table": entry.table,
                "dataset": entry.dataset,
                "protocol": entry.protocol,
                "setting": entry.setting,
                "metrics": {},
                "sources": {},
                "missing": [],
            },
        )

        log_path = entry.structured_log_path()

        if log_path is None:
            row["missing"].append(
                f"{entry.task_type}: no result log for {entry.path.name}"
            )
            continue

        try:
            collected = collect_configuration_record(
                entry,
                campaign_id=campaign_id,
                publication_mode=publication_mode,
                smoke=smoke,
                implementation_identity=implementation_identity,
            )
        except (ResultCollectionError, FileNotFoundError) as error:
            row["missing"].append(
                f"{entry.task_type}: {error}"
            )
            continue

        record = collected["record"]

        results = record.get("results", {})
        expected_metrics = (
            IDENTIFICATION_METRICS
            if entry.task_type == "identification"
            else VERIFICATION_METRICS
        )

        for metric_name in expected_metrics:
            if metric_name in results:
                row["metrics"][metric_name] = results[metric_name]

        row["sources"][entry.task_type] = (
            collected["locator"]
            if collected["locator"] is not None
            else {
                "legacy_exploratory": True,
                "relative_path": log_path.name,
            }
        )

    return rows


def render_markdown(rows, table_number):
    """
    Render one result table as Markdown.
    """
    header = (
        ["Dataset", "Data Split", "Setting"]
        + [
            "Rank-1",
            "Rank-5",
            "EER",
            "AUC",
            "d-prime",
            "TAR@0.1%FAR",
        ]
    )

    lines = [
        f"### Table {table_number}",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]

    previous_dataset = None
    previous_protocol = None

    for row in rows.values():
        if row["table"] != table_number:
            continue

        dataset_label = DATASET_LABELS.get(
            row["dataset"],
            row["dataset"],
        )
        protocol_label = PROTOCOL_LABELS.get(
            row["protocol"],
            row["protocol"].replace("_", " "),
        )

        cells = [
            dataset_label if row["dataset"] != previous_dataset else "",
            (
                protocol_label
                if (row["dataset"], row["protocol"])
                != (previous_dataset, previous_protocol)
                else ""
            ),
            SETTING_LABELS.get(row["setting"], row["setting"]),
        ]

        for metric_name in TABLE_METRICS:
            cells.append(
                format_metric(
                    row["metrics"].get(metric_name)
                )
            )

        lines.append("| " + " | ".join(cells) + " |")

        previous_dataset = row["dataset"]
        previous_protocol = row["protocol"]

    lines.append("")

    return "\n".join(lines)


def write_csv(rows, output_path):
    """
    Write every collected row to a single flat CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "table",
                "dataset",
                "data_split",
                "setting",
            ]
            + list(TABLE_METRICS)
            + ["missing"]
        )

        for row in rows.values():
            writer.writerow(
                [
                    row["table"],
                    DATASET_LABELS.get(
                        row["dataset"],
                        row["dataset"],
                    ),
                    PROTOCOL_LABELS.get(
                        row["protocol"],
                        row["protocol"],
                    ),
                    SETTING_LABELS.get(
                        row["setting"],
                        row["setting"],
                    ),
                ]
                + [
                    format_metric(
                        row["metrics"].get(metric_name)
                    )
                    for metric_name in TABLE_METRICS
                ]
                + ["; ".join(row["missing"])]
            )


def describe_plan(entries):
    """
    Print the configurations that would be executed.
    """
    print(f"{len(entries)} configuration(s) selected\n")

    current_table = None
    current_dataset = None

    for entry in sorted(entries, key=sort_key):
        if entry.table != current_table:
            current_table = entry.table
            current_dataset = None
            print(f"Table {current_table}")

        if entry.dataset != current_dataset:
            current_dataset = entry.dataset
            print(
                f"  {DATASET_LABELS.get(current_dataset, current_dataset)}"
            )

        print(
            f"    task {entry.task} "
            f"{entry.task_type:<15} "
            f"{entry.protocol}/{entry.setting}"
        )

    print()


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the shipped paper configurations and rebuild the "
            "result tables."
        ),
    )

    parser.add_argument(
        "--config-root",
        type=str,
        default=str(CONFIG_ROOT),
        help="Directory containing the reproduction configurations.",
    )
    parser.add_argument(
        "--table",
        type=int,
        action="append",
        choices=[5, 6, 7],
        default=None,
        help="Restrict to one result table. Repeatable.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        action="append",
        choices=list(TABLE_BY_DATASET),
        default=None,
        help="Restrict to one dataset. Repeatable.",
    )
    parser.add_argument(
        "--task",
        type=int,
        action="append",
        choices=list(range(1, 9)),
        default=None,
        help="Restrict to one task number. Repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and the commands without executing them.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the selected configurations.",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Rebuild the tables from result logs that already exist.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run with reduced epochs, seeds, and pair counts on CPU. "
            "Verifies the plan end to end; the numbers are not the "
            "reported ones."
        ),
    )
    parser.add_argument(
        "--campaign-id",
        type=str,
        default=None,
        help="Require and record this administrative campaign identifier.",
    )
    parser.add_argument(
        "--allow-exploratory-results",
        action="store_true",
        help=(
            "Explicitly allow legacy latest-record reads. Such records are "
            "not publication eligible."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going after a configuration fails.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for the assembled CSV and Markdown tables.",
    )
    parser.add_argument(
        "--extra-arg",
        type=str,
        action="append",
        default=None,
        dest="extra_arguments",
        help=(
            "Extra argument forwarded to main.py, e.g. "
            "--extra-arg --intelligent_weight_loading. Repeatable."
        ),
    )

    return parser


def main(argv=None):
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if not (
        arguments.dry_run
        or arguments.run
        or arguments.collect
    ):
        parser.error(
            "Choose at least one of --dry-run, --run, or --collect."
        )

    entries = discover_configurations(
        Path(arguments.config_root)
    )
    entries = filter_configurations(
        entries,
        tables=arguments.table,
        datasets=arguments.dataset,
        tasks=arguments.task,
    )

    if not entries:
        raise SystemExit(
            "No configurations match the requested filters."
        )

    if arguments.dry_run:
        describe_plan(entries)

        for entry in sorted(entries, key=sort_key):
            command = build_command(
                entry,
                smoke=arguments.smoke,
                extra_arguments=arguments.extra_arguments,
                campaign_id=arguments.campaign_id,
            )
            print(" ".join(command))

        print()

    exit_status = 0

    if arguments.run:
        if arguments.smoke:
            print(
                "[WARN] Smoke mode is active. The resulting numbers are "
                "not the reported ones; it verifies the plan only.\n"
            )

        failures, collected_results = run_configurations(
            entries,
            smoke=arguments.smoke,
            extra_arguments=arguments.extra_arguments,
            continue_on_error=arguments.continue_on_error,
            campaign_id=arguments.campaign_id,
            return_collected=True,
        )

        if failures:
            exit_status = 1
            print(f"{len(failures)} configuration(s) failed:")
            for name, status in failures:
                print(f"  {name} (exit {status})")
            print()

    if arguments.collect:
        rows = collect_rows(
            entries,
            campaign_id=arguments.campaign_id,
            publication_mode=not arguments.allow_exploratory_results,
            smoke=arguments.smoke,
        )

        incomplete = [
            row for row in rows.values() if row["missing"]
        ]

        tables = sorted(
            {row["table"] for row in rows.values()}
        )

        rendered = "\n".join(
            render_markdown(rows, table_number)
            for table_number in tables
        )

        print(rendered)

        if incomplete:
            exit_status = 1
            print(
                f"[WARN] {len(incomplete)} row(s) are incomplete because "
                "the corresponding runs have not been executed yet:"
            )
            for row in incomplete[:10]:
                print(
                    f"  Table {row['table']} "
                    f"{row['dataset']}/{row['protocol']}/{row['setting']}"
                )
                for reason in row["missing"]:
                    print(f"    {reason}")
            print()

        may_write_outputs = (
            not incomplete
            or arguments.allow_exploratory_results
        )

        if arguments.output_dir and may_write_outputs:
            output_dir = Path(arguments.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            write_csv(
                rows,
                output_dir / "paper_tables.csv",
            )

            markdown_path = output_dir / "paper_tables.md"
            markdown_path.write_text(
                rendered,
                encoding="utf-8",
            )

            provenance_path = (
                output_dir
                / "paper_tables_result_provenance.json"
            )
            provenance_path.write_text(
                json.dumps(
                    {
                        "campaign_id": arguments.campaign_id,
                        "publication_provenance_verified": (
                            not arguments.allow_exploratory_results
                        ),
                        "rows": {
                            "/".join(map(str, row_key)): row["sources"]
                            for row_key, row in rows.items()
                        },
                    },
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )

            print(f"[INFO] Tables written to {output_dir}")
        elif arguments.output_dir:
            print(
                "[ERROR] Refusing to write publication tables because "
                "one or more intended rows lack exact, "
                "provenance-verified result records."
            )

    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
