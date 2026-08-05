"""
Check that every configured dataset can be downloaded and unpacked.

Dataset acquisition is the first thing a new user runs and the first thing
that can fail, usually for reasons that have nothing to do with the framework:
a repository is unreachable, an archive format needs a tool the machine does
not have, or a transfer is silently truncated. This script exercises that path
for every dataset and prints one table, so a single run reports the state of
all of them instead of surfacing one failure at a time.

Only acquisition is checked. Signals are not parsed and no preprocessing runs,
so the script finishes in the time the downloads take and needs no GPU.

Usage:

    python -m scripts.verify_datasets
    python -m scripts.verify_datasets --datasets ecgid heartprint
    python -m scripts.verify_datasets --check-tools
    python -m scripts.verify_datasets --output-json dataset_report.json
"""

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import load_dataset  # noqa: E402
from load_dataset import (  # noqa: E402
    MissingArchiveToolError,
    available_rar_backend,
    detect_archive_format,
)

# Ordered smallest first, so an unreachable network or a missing tool is
# reported before a multi-gigabyte transfer has been started.
DATASETS = (
    "ecgid",
    "heartprint",
    "cybhi",
    "mitbih",
    "nsrdb",
    "ptb",
    "ptbxl",
)

# Datasets whose archive needs an external unpacking tool, and are therefore
# the ones that fail on an otherwise healthy machine.
EXTERNAL_TOOL_DATASETS = ("heartprint",)


def _loader_for(dataset: str):
    """Return the loader class registered for a dataset name."""
    return getattr(load_dataset, f"load_{dataset}_dataset")


def _directory_summary(root: Path) -> dict:
    """
    Count the files below a directory and measure their total size.

    Returns zeroes rather than raising if the directory does not exist, so a
    dataset that failed to download still produces a row in the report.
    """
    if not root.exists():
        return {"files": 0, "bytes": 0, "extensions": {}}

    files = 0
    total = 0
    extensions: dict = {}

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        files += 1
        try:
            total += path.stat().st_size
        except OSError:
            pass
        suffix = path.suffix.lower() or "(none)"
        extensions[suffix] = extensions.get(suffix, 0) + 1

    # Only the handful of most common extensions are worth reporting; the rest
    # are checksums and licence files that say nothing about dataset health.
    ranked = sorted(extensions.items(), key=lambda item: -item[1])[:4]

    return {"files": files, "bytes": total, "extensions": dict(ranked)}


def _format_size(num_bytes: int) -> str:
    """Render a byte count in the largest unit that keeps it readable."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


def verify_dataset(dataset: str, keep_archive: bool = True) -> dict:
    """
    Download and unpack one dataset, reporting what arrived on disk.

    Args:
        dataset (str): Dataset name as used by --dataset elsewhere.
        keep_archive (bool): If False, the archive is removed once unpacked.

    Returns:
        dict: A record describing the outcome, always including a "status" of
            "ok", "already-present", "missing-tool", or "failed".
    """
    record = {"dataset": dataset, "status": "failed", "detail": ""}

    try:
        loader = _loader_for(dataset)(cleanup_zip=not keep_archive)
    except Exception as exc:
        record["detail"] = f"could not construct the loader: {exc}"
        return record

    root = Path(loader.dataset_root)
    record["path"] = str(root)
    was_present = root.exists() and any(root.iterdir())

    try:
        loader.download()
    except MissingArchiveToolError as exc:
        record["status"] = "missing-tool"
        record["detail"] = str(exc)
        return record
    except Exception as exc:
        record["detail"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        return record

    summary = _directory_summary(root)
    record.update(summary)

    if summary["files"] == 0:
        record["detail"] = (
            "the download reported success but the directory is empty"
        )
        return record

    archive = Path(loader.zip_path)
    if archive.exists():
        record["archive"] = archive.name
        record["archive_format"] = detect_archive_format(archive)

    record["status"] = "already-present" if was_present else "ok"
    return record


def _print_tool_report() -> None:
    """Report which external unpacking tools this machine provides."""
    print("External archive tools")
    print("-" * 62)

    backend = available_rar_backend()
    for name in load_dataset._RAR_BACKENDS:
        location = shutil.which(name)
        mark = "yes" if location else " no"
        print(f"  {mark}  {name:<8} {location or ''}")

    print()
    if backend:
        print(f"RAR archives can be unpacked using '{backend}'.")
    else:
        print(
            "No RAR tool is installed, so the following datasets cannot be\n"
            f"unpacked automatically: {', '.join(EXTERNAL_TOOL_DATASETS)}.\n"
            "  Debian/Ubuntu (including Google Colab):  apt-get install -y unar\n"
            "  macOS (Homebrew):                        brew install unar\n"
            "  Windows:                                 install 7-Zip or WinRAR"
        )
    print()


def _print_table(records) -> None:
    """Print one row per dataset, followed by a count of each outcome."""
    print()
    print(f"{'dataset':<12}{'status':<16}{'files':>8}{'size':>12}  contents")
    print("-" * 74)

    for record in records:
        contents = ", ".join(
            f"{count}{suffix}" for suffix, count in record.get("extensions", {}).items()
        )
        print(
            f"{record['dataset']:<12}"
            f"{record['status']:<16}"
            f"{record.get('files', 0):>8}"
            f"{_format_size(record.get('bytes', 0)):>12}  "
            f"{contents}"
        )

    print()

    for record in records:
        if record["status"] in ("ok", "already-present"):
            continue
        print(f"[{record['status']}] {record['dataset']}")
        for line in str(record.get("detail", "")).splitlines():
            print(f"    {line}")
        print()

    counts: dict = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    print("summary: " + ", ".join(f"{n} {status}" for status, n in sorted(counts.items())))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Download and unpack every dataset, reporting what arrived.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="Datasets to check. Defaults to all of them.",
    )
    parser.add_argument(
        "--check-tools",
        action="store_true",
        help="Report the available archive tools and exit without downloading.",
    )
    parser.add_argument(
        "--delete-archives",
        action="store_true",
        help="Remove each archive once it has been unpacked, to save disk space.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write the full report, including tracebacks, to this file.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    _print_tool_report()
    if args.check_tools:
        return 0

    records = []
    for dataset in args.datasets:
        print(f"=== {dataset} ===")
        records.append(
            verify_dataset(dataset, keep_archive=not args.delete_archives)
        )

    _print_table(records)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"\nReport written to {args.output_json}")

    failed = [r for r in records if r["status"] not in ("ok", "already-present")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
