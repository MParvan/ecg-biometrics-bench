"""
Enumerate the trained weights and result records a release actually contains.

The cache stores files under opaque content hashes, so listing the directory
does not reveal which model, dataset, or seed produced any given file. This
script reads the metadata each cache entry carries and turns the tree into a
manifest that names, for every artifact:

- the dataset, protocol, model, seed, and training regime it belongs to
- the epochs actually trained, which may differ from the configured maximum
- a SHA-256 checksum, so a download can be verified
- the result table and row it supports, where that can be determined

Result records are enumerated the same way, so a reader can see at a glance
which of the 150 shipped configurations have been executed.

Usage:

    python -m scripts.build_artifact_manifest \
        --cache-dir ../ecg-biometrics-artifacts/cache \
        --results-dir ../ecg-biometrics-artifacts/paper_results \
        --output-json artifact_manifest.json \
        --output-markdown ARTIFACTS.md
"""

import argparse
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reproduce_tables import (  # noqa: E402
    DATASET_LABELS,
    PROTOCOL_LABELS,
    SETTING_LABELS,
    TASK_LOG_NAMES,
    discover_configurations,
    read_latest_record,
)

CHECKSUM_CHUNK_SIZE = 8 * 1024 * 1024

# Training regimes recorded in the weight-cache metadata, mapped to the
# manuscript setting they belong to.
REGIME_SETTINGS = {
    "intra_session_closed_set": "closed_set",
    "intra_session_subject_disjoint": "subject_disjoint",
    "cross_session_closed_set": "closed_set",
    "cross_session_subject_disjoint": "subject_disjoint",
}


def compute_checksum(path):
    """
    Return the SHA-256 checksum of one file.
    """
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(CHECKSUM_CHUNK_SIZE),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _loader_dataset_name(metadata):
    """
    Recover the dataset name from a cache entry's loader identity.
    """
    loader_identity = metadata.get("loader_identity") or {}
    root_dir = loader_identity.get("root_dir")

    if root_dir:
        return str(root_dir).lower()

    loader_class = loader_identity.get("loader_class", "")

    # e.g. load_ecgid_dataset -> ecgid
    if loader_class.startswith("load_") and loader_class.endswith(
        "_dataset"
    ):
        return loader_class[len("load_"):-len("_dataset")]

    return None


def _loader_split_mode(metadata):
    """
    Recover the configured split mode from a cache entry's loader identity.
    """
    loader_identity = metadata.get("loader_identity") or {}
    settings = loader_identity.get("settings") or {}

    return settings.get("data_split_mode")


def collect_weight_artifacts(cache_dir, compute_checksums=True):
    """
    Describe every trained-weight entry in a cache directory.
    """
    weight_dir = Path(cache_dir) / "weights"

    if not weight_dir.exists():
        return []

    artifacts = []

    for weight_path in sorted(weight_dir.glob("*.pth")):
        metadata_path = weight_path.with_suffix(".json")

        metadata = {}

        if metadata_path.exists():
            try:
                with metadata_path.open(
                    "r",
                    encoding="utf-8",
                ) as metadata_file:
                    metadata = json.load(metadata_file)
            except (OSError, json.JSONDecodeError):
                metadata = {}

        regime = metadata.get("training_regime")

        entry = OrderedDict(
            [
                ("artifact_type", "trained_weights"),
                ("file", weight_path.name),
                ("cache_uid", weight_path.stem),
                ("size_bytes", weight_path.stat().st_size),
                ("dataset", _loader_dataset_name(metadata)),
                ("data_split_mode", _loader_split_mode(metadata)),
                ("training_regime", regime),
                ("setting", REGIME_SETTINGS.get(regime)),
                ("model", metadata.get("model")),
                ("seed", metadata.get("seed")),
                ("configured_epochs", metadata.get("epochs")),
                ("trained_epochs", metadata.get("actual_epochs")),
                ("batch_size", metadata.get("batch_size")),
                ("learning_rate", metadata.get("lr")),
                ("num_classes", metadata.get("classes")),
                ("metadata_present", bool(metadata)),
            ]
        )

        if compute_checksums:
            entry["sha256"] = compute_checksum(weight_path)

        artifacts.append(entry)

    return artifacts


def collect_dataset_cache_artifacts(cache_dir, compute_checksums=True):
    """
    Describe every preprocessed-array entry in a cache directory.
    """
    data_dir = Path(cache_dir) / "data"

    if not data_dir.exists():
        return []

    artifacts = []

    for data_path in sorted(data_dir.glob("*.npz")):
        metadata_path = data_path.with_suffix(".json")

        metadata = {}

        if metadata_path.exists():
            try:
                with metadata_path.open(
                    "r",
                    encoding="utf-8",
                ) as metadata_file:
                    metadata = json.load(metadata_file)
            except (OSError, json.JSONDecodeError):
                metadata = {}

        entry = OrderedDict(
            [
                ("artifact_type", "preprocessed_arrays"),
                ("file", data_path.name),
                ("cache_uid", data_path.stem),
                ("size_bytes", data_path.stat().st_size),
                ("dataset", metadata.get("dataset")),
                ("task_type", metadata.get("task_type")),
                ("data_split_mode", metadata.get("split_mode")),
                (
                    "num_beats_to_merge",
                    metadata.get("num_beats_to_merge"),
                ),
                ("metadata_present", bool(metadata)),
            ]
        )

        if compute_checksums:
            entry["sha256"] = compute_checksum(data_path)

        artifacts.append(entry)

    return artifacts


def collect_configuration_coverage(entries):
    """
    Report which shipped configurations have produced a result record.
    """
    coverage = []

    for entry in entries:
        log_path = entry.structured_log_path()
        record = (
            read_latest_record(log_path) if log_path else None
        )

        seeds = []
        run_count = 0

        if record:
            per_run_results = record.get("per_run_results") or []
            run_count = len(per_run_results)
            seeds = [
                run_record.get("seed")
                for run_record in per_run_results
            ]

        coverage.append(
            OrderedDict(
                [
                    ("configuration", entry.path.name),
                    (
                        "relative_path",
                        str(
                            entry.path.relative_to(PROJECT_ROOT)
                        ),
                    ),
                    ("table", entry.table),
                    ("dataset", entry.dataset),
                    ("protocol", entry.protocol),
                    ("setting", entry.setting),
                    ("task", entry.task),
                    ("task_type", entry.task_type),
                    (
                        "task_log_name",
                        TASK_LOG_NAMES.get(entry.task),
                    ),
                    ("executed", record is not None),
                    (
                        "result_log",
                        str(log_path) if log_path else None,
                    ),
                    (
                        "experiment_time",
                        (
                            record.get("experiment_time")
                            if record
                            else None
                        ),
                    ),
                    ("runs", run_count),
                    ("seeds", seeds),
                ]
            )
        )

    return coverage


def summarize(weight_artifacts, data_artifacts, coverage):
    """
    Build the counts a reader needs before downloading anything.
    """
    executed = [row for row in coverage if row["executed"]]

    datasets_covered = sorted(
        {
            row["dataset"]
            for row in executed
            if row["dataset"]
        }
    )

    total_bytes = sum(
        artifact["size_bytes"]
        for artifact in weight_artifacts + data_artifacts
    )

    return OrderedDict(
        [
            ("configurations_total", len(coverage)),
            ("configurations_executed", len(executed)),
            (
                "configurations_pending",
                len(coverage) - len(executed),
            ),
            ("datasets_with_results", datasets_covered),
            ("trained_weight_files", len(weight_artifacts)),
            ("preprocessed_array_files", len(data_artifacts)),
            ("total_artifact_bytes", total_bytes),
            (
                "total_artifact_gigabytes",
                round(total_bytes / (1024 ** 3), 3),
            ),
        ]
    )


def render_markdown(manifest):
    """
    Render the manifest as a release-ready document.
    """
    summary = manifest["summary"]

    lines = [
        "# Released artifacts",
        "",
        "This file is generated by "
        "`python -m scripts.build_artifact_manifest`. It records exactly "
        "which artifacts exist, what produced them, and which reported "
        "table row each one supports.",
        "",
        "## Summary",
        "",
        f"- Shipped configurations: {summary['configurations_total']}",
        f"- Executed: {summary['configurations_executed']}",
        f"- Pending: {summary['configurations_pending']}",
        "- Datasets with results: "
        + (
            ", ".join(
                DATASET_LABELS.get(name, name)
                for name in summary["datasets_with_results"]
            )
            or "none"
        ),
        f"- Trained weight files: {summary['trained_weight_files']}",
        "- Preprocessed array files: "
        f"{summary['preprocessed_array_files']}",
        "- Total size: "
        f"{summary['total_artifact_gigabytes']} GiB",
        "",
        "## Configuration coverage",
        "",
        "| Table | Dataset | Data split | Setting | Task | Executed | Runs | Seeds |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for row in manifest["configuration_coverage"]:
        seeds = row["seeds"]
        seed_text = (
            f"{min(seeds)}-{max(seeds)}"
            if seeds and all(
                isinstance(seed, int) for seed in seeds
            )
            else "-"
        )

        lines.append(
            "| {table} | {dataset} | {protocol} | {setting} | "
            "{task} ({task_type}) | {executed} | {runs} | {seeds} |".format(
                table=row["table"],
                dataset=DATASET_LABELS.get(
                    row["dataset"],
                    row["dataset"],
                ),
                protocol=PROTOCOL_LABELS.get(
                    row["protocol"],
                    row["protocol"],
                ),
                setting=SETTING_LABELS.get(
                    row["setting"],
                    row["setting"],
                ),
                task=row["task"],
                task_type=row["task_type"],
                executed="yes" if row["executed"] else "no",
                runs=row["runs"] or "-",
                seeds=seed_text,
            )
        )

    lines.extend(
        [
            "",
            "## Trained weights",
            "",
        ]
    )

    if manifest["trained_weights"]:
        lines.extend(
            [
                "| File | Dataset | Split mode | Regime | Model | Seed | "
                "Epochs | Size (MiB) | SHA-256 |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )

        for artifact in manifest["trained_weights"]:
            checksum = artifact.get("sha256", "")

            lines.append(
                "| {file} | {dataset} | {split} | {regime} | {model} | "
                "{seed} | {epochs} | {size} | {checksum} |".format(
                    file=artifact["file"],
                    dataset=artifact["dataset"] or "-",
                    split=artifact["data_split_mode"] or "-",
                    regime=artifact["training_regime"] or "-",
                    model=artifact["model"] or "-",
                    seed=artifact["seed"] if artifact["seed"] is not None else "-",
                    epochs=(
                        artifact["trained_epochs"]
                        if artifact["trained_epochs"] is not None
                        else artifact["configured_epochs"] or "-"
                    ),
                    size=round(
                        artifact["size_bytes"] / (1024 ** 2),
                        2,
                    ),
                    checksum=(
                        f"`{checksum[:16]}...`" if checksum else "-"
                    ),
                )
            )
    else:
        lines.append(
            "No trained weights were found in the configured cache "
            "directory. Weight caching is disabled by default; enable it "
            "with `--intelligent_weight_loading` to retain the trained "
            "feature extractors."
        )

    lines.extend(["", "## Reproducing a row", ""])
    lines.extend(
        [
            "Each table row is produced by two configurations: an "
            "identification run supplying Rank-1 and Rank-5, and a "
            "verification run supplying EER, AUC, d-prime, and "
            "TAR@0.1%FAR. To reproduce one row:",
            "",
            "```bash",
            "python main.py --config configs/paper_reproduction/<dataset>/"
            "<dataset>_<protocol>_<setting>_task<NN>_identification.yaml",
            "python main.py --config configs/paper_reproduction/<dataset>/"
            "<dataset>_<protocol>_<setting>_task<NN>_verification.yaml",
            "python -m scripts.reproduce_tables --dataset <dataset> "
            "--collect",
            "```",
            "",
        ]
    )

    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate the trained weights, cached arrays, and result "
            "records a release contains."
        ),
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="../ecg-biometrics-artifacts/cache",
        help=(
            "Cache directory holding data/ and weights/ subdirectories."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help=(
            "Optional results root. When omitted, each configuration's "
            "own results_dir is used."
        ),
    )
    parser.add_argument(
        "--config-root",
        type=str,
        default=str(
            PROJECT_ROOT / "configs" / "paper_reproduction"
        ),
        help="Directory containing the reproduction configurations.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write the manifest to this JSON file.",
    )
    parser.add_argument(
        "--output-markdown",
        type=str,
        default=None,
        help="Write the human-readable manifest to this Markdown file.",
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help=(
            "Skip SHA-256 computation. Much faster on a large cache, but "
            "the manifest can no longer verify a download."
        ),
    )

    return parser


def main(argv=None):
    parser = build_parser()
    arguments = parser.parse_args(argv)

    compute_checksums = not arguments.skip_checksums

    cache_dir = Path(arguments.cache_dir)

    if not cache_dir.is_absolute():
        cache_dir = (PROJECT_ROOT / cache_dir).resolve()

    weight_artifacts = collect_weight_artifacts(
        cache_dir,
        compute_checksums=compute_checksums,
    )
    data_artifacts = collect_dataset_cache_artifacts(
        cache_dir,
        compute_checksums=compute_checksums,
    )

    entries = discover_configurations(
        Path(arguments.config_root)
    )

    if arguments.results_dir:
        results_root = Path(arguments.results_dir)

        if not results_root.is_absolute():
            results_root = (
                PROJECT_ROOT / results_root
            ).resolve()

        for entry in entries:
            entry.configuration["results_dir"] = str(
                results_root
                / entry.dataset
                / entry.protocol
                / entry.setting
                / f"task{entry.task:02d}_{entry.task_type}"
            )

    coverage = collect_configuration_coverage(entries)

    manifest = OrderedDict(
        [
            (
                "summary",
                summarize(
                    weight_artifacts,
                    data_artifacts,
                    coverage,
                ),
            ),
            ("cache_directory", str(cache_dir)),
            ("checksums_computed", compute_checksums),
            ("configuration_coverage", coverage),
            ("trained_weights", weight_artifacts),
            ("preprocessed_arrays", data_artifacts),
        ]
    )

    summary = manifest["summary"]

    print(
        f"{summary['configurations_executed']} of "
        f"{summary['configurations_total']} configurations executed; "
        f"{summary['trained_weight_files']} weight file(s), "
        f"{summary['preprocessed_array_files']} array file(s), "
        f"{summary['total_artifact_gigabytes']} GiB total."
    )

    if arguments.output_json:
        output_path = Path(arguments.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                manifest,
                output_file,
                indent=2,
                default=str,
            )

        print(f"[INFO] Manifest written to {output_path}")

    if arguments.output_markdown:
        markdown_path = Path(arguments.output_markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            render_markdown(manifest),
            encoding="utf-8",
        )

        print(f"[INFO] Manifest written to {markdown_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
