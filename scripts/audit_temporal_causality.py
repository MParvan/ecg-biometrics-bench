"""
Verify that enrollment templates are never built from probe or future data.

The audit answers two questions that cannot be settled by reading result
tables:

1. For the record-order regimes (ECG-ID, PTB, PTB-XL), does every subject's
   enrollment partition consist only of recordings that precede its probe
   recordings?
2. For the continuous recordings (MIT-BIH, NSRDB), are the enrollment and
   probe minute windows disjoint, and how far apart are they?

Both checks reuse the same selection code the training pipeline runs, so the
report describes the partitions that were actually evaluated rather than a
restatement of the intended protocol. Record-order audits read WFDB headers
only, so they complete in seconds and need no GPU.

Usage:

    python -m scripts.audit_temporal_causality \
        --dataset ecgid \
        --data_split_mode leave-last-out-long-term \
        --output-json causality_ecgid_llo_long.json

    python -m scripts.audit_temporal_causality \
        --dataset mitbih \
        --train_parts 0 5 --train_parts 12.5 17.5 \
        --test_parts 25 30

    python -m scripts.audit_temporal_causality --config path/to/experiment.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from load_dataset import (  # noqa: E402
    RECORD_ORDER_SPLIT_MODES,
    audit_continuous_temporal_partitions,
    audit_record_order_causality,
    load_ecgid_dataset,
    load_ptb_dataset,
    load_ptbxl_dataset,
)

RECORD_ORDER_LOADERS = {
    "ecgid": load_ecgid_dataset,
    "ptb": load_ptb_dataset,
    "ptbxl": load_ptbxl_dataset,
}

CONTINUOUS_DATASETS = (
    "mitbih",
    "nsrdb",
)

SUPPORTED_DATASETS = tuple(RECORD_ORDER_LOADERS) + CONTINUOUS_DATASETS


def _normalize_minute_ranges(raw_ranges):
    """
    Convert repeated --*_parts pairs into (start, end) minute tuples.
    """
    if not raw_ranges:
        return None

    return [
        (float(start), float(end))
        for start, end in raw_ranges
    ]


def _load_configuration(config_path):
    """
    Read the dataset and protocol fields of an experiment YAML.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise SystemExit(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        configuration = yaml.safe_load(config_file) or {}

    if not isinstance(configuration, dict):
        raise SystemExit(
            "The YAML configuration must contain a key-value mapping."
        )

    return configuration


def audit_record_order_dataset(
    dataset,
    data_split_mode,
    signal_type=None,
    only_healthy=False,
    limit_records=None,
):
    """
    Audit one record-order dataset using header metadata only.
    """
    loader_class = RECORD_ORDER_LOADERS[dataset]

    loader_kwargs = {
        "data_split_mode": data_split_mode,
    }

    if dataset == "ecgid":
        loader_kwargs["signal_type"] = signal_type

    if dataset in {"ptb", "ptbxl"}:
        loader_kwargs["only_healthy"] = only_healthy

    if dataset == "ptbxl" and limit_records is not None:
        loader_kwargs["limit_records"] = limit_records

    loader = loader_class(**loader_kwargs)

    records_by_subject = loader.load_raw_data(
        metadata_only=True,
    )

    report = audit_record_order_causality(
        records_by_subject,
        data_split_mode,
    )

    report["dataset"] = dataset
    report["audit_type"] = "record_order"

    return report


def audit_continuous_dataset(
    dataset,
    train_parts,
    enrol_parts,
    test_parts,
    temporal_guard_minutes,
):
    """
    Audit one continuous-recording dataset from its minute windows.
    """
    try:
        report = audit_continuous_temporal_partitions(
            train_parts=train_parts,
            enrol_parts=enrol_parts,
            test_parts=test_parts,
            temporal_guard_minutes=temporal_guard_minutes,
        )
    except ValueError as error:
        report = {
            "overlap_free": False,
            "error": str(error),
        }

    report["dataset"] = dataset
    report["audit_type"] = "continuous_window"

    return report


def summarize_report(report):
    """
    Render a short human-readable verdict for the terminal.
    """
    lines = [
        "=" * 72,
        f"Temporal causality audit: {report['dataset']}",
        "=" * 72,
    ]

    if report["audit_type"] == "record_order":
        lines.extend(
            [
                f"Regime                : {report['data_split_mode']}",
                f"Subjects supplied     : {report['subjects_supplied']}",
                f"Subjects eligible     : {report['subjects_eligible']}",
                f"Subjects audited      : {report['subjects_audited']}",
                f"Violations            : {len(report['violations'])}",
            ]
        )

        verdict = (
            "PASS - every enrollment recording precedes its probe"
            if report["enrollment_precedes_probe"]
            else "FAIL - enrollment uses probe or future recordings"
        )
    else:
        if "error" in report:
            lines.append(f"Error                 : {report['error']}")
            verdict = "FAIL - enrollment and probe windows are not disjoint"
        else:
            lines.extend(
                [
                    "Enrollment coverage   : "
                    + ", ".join(
                        f"[{start:g}, {end:g}) min"
                        for start, end in report["enrollment_coverage"]
                    ),
                    "Probe coverage        : "
                    + ", ".join(
                        f"[{start:g}, {end:g}) min"
                        for start, end in report["probe_coverage"]
                    ),
                    "Enrollment duration   : "
                    f"{report['covered_minutes']['enrollment']:g} min",
                    "Probe duration        : "
                    f"{report['covered_minutes']['probe']:g} min",
                    "Achieved separation   : "
                    f"{report['achieved_separation_minutes']:g} min",
                    "Guard band enforced   : "
                    f"{report['temporal_guard_minutes']:g} min",
                ]
            )
            verdict = "PASS - enrollment and probe windows are disjoint"

    lines.extend(["-" * 72, verdict, ""])

    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Verify that enrollment templates never use probe or future data."
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Experiment YAML to audit. Dataset and protocol fields are read "
            "from the file; explicit flags override them."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=SUPPORTED_DATASETS,
        help="Dataset to audit.",
    )
    parser.add_argument(
        "--data_split_mode",
        type=str,
        default=None,
        choices=list(RECORD_ORDER_SPLIT_MODES),
        help="Record-order regime for ECG-ID, PTB, or PTB-XL.",
    )
    parser.add_argument(
        "--signal_type",
        type=str,
        default=None,
        choices=["raw", "filtered"],
        help="ECG-ID channel selection. Omit to use the dataset default "
             "in config.yaml.",
    )
    parser.add_argument(
        "--only_healthy",
        action="store_true",
        help="Restrict PTB or PTB-XL to healthy control subjects.",
    )
    parser.add_argument(
        "--limit_records",
        type=int,
        default=None,
        help="Audit only the first N PTB-XL patients.",
    )
    parser.add_argument(
        "--train_parts",
        type=float,
        nargs=2,
        action="append",
        metavar=("START_MINUTE", "END_MINUTE"),
        default=None,
        help="Training minute range for MIT-BIH or NSRDB. Repeatable.",
    )
    parser.add_argument(
        "--enrol_parts",
        "--enroll_parts",
        dest="enrol_parts",
        type=float,
        nargs=2,
        action="append",
        metavar=("START_MINUTE", "END_MINUTE"),
        default=None,
        help="Enrollment minute range for MIT-BIH or NSRDB. Repeatable.",
    )
    parser.add_argument(
        "--test_parts",
        type=float,
        nargs=2,
        action="append",
        metavar=("START_MINUTE", "END_MINUTE"),
        default=None,
        help="Probe minute range for MIT-BIH or NSRDB. Repeatable.",
    )
    parser.add_argument(
        "--temporal_guard_minutes",
        type=float,
        default=0.0,
        help=(
            "Minimum required separation between the enrollment and probe "
            "coverage (default: 0.0, which rejects overlap only)."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write the complete report to this JSON file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the terminal summary.",
    )

    return parser


def main(argv=None):
    parser = build_parser()
    arguments = parser.parse_args(argv)

    dataset = arguments.dataset
    data_split_mode = arguments.data_split_mode
    train_parts = _normalize_minute_ranges(arguments.train_parts)
    enrol_parts = _normalize_minute_ranges(arguments.enrol_parts)
    test_parts = _normalize_minute_ranges(arguments.test_parts)
    signal_type = arguments.signal_type
    temporal_guard_minutes = arguments.temporal_guard_minutes

    if arguments.config is not None:
        configuration = _load_configuration(arguments.config)

        dataset = dataset or configuration.get("dataset")
        data_split_mode = data_split_mode or configuration.get(
            "data_split_mode"
        )
        train_parts = train_parts or _normalize_minute_ranges(
            configuration.get("train_parts")
        )
        enrol_parts = enrol_parts or _normalize_minute_ranges(
            configuration.get("enrol_parts")
        )
        test_parts = test_parts or _normalize_minute_ranges(
            configuration.get("test_parts")
        )

        # Match the precedence of every other field here: an explicit command
        # line wins, the file fills in what was left unset, and a value absent
        # from both is resolved by the loader from config.yaml.
        signal_type = signal_type or configuration.get("signal_type")

        if configuration.get("temporal_guard_minutes") is not None:
            temporal_guard_minutes = float(
                configuration["temporal_guard_minutes"]
            )

    if dataset is None:
        parser.error(
            "A dataset is required, either via --dataset or --config."
        )

    if dataset not in SUPPORTED_DATASETS:
        parser.error(
            f"Dataset '{dataset}' has no temporal-causality audit. "
            f"Supported datasets: {list(SUPPORTED_DATASETS)}. "
            "Session-structured datasets are checked by the "
            "enrollment/probe session-disjointness rule in main.py."
        )

    if dataset in CONTINUOUS_DATASETS:
        if not train_parts or not test_parts:
            parser.error(
                f"Auditing '{dataset}' requires --train_parts and "
                "--test_parts, or a --config that defines them."
            )

        report = audit_continuous_dataset(
            dataset,
            train_parts,
            enrol_parts,
            test_parts,
            temporal_guard_minutes,
        )
        audit_passed = report.get("overlap_free", False)
    else:
        if data_split_mode not in RECORD_ORDER_SPLIT_MODES:
            parser.error(
                f"Auditing '{dataset}' requires --data_split_mode from "
                f"{list(RECORD_ORDER_SPLIT_MODES)}. Single-session regimes "
                "have no enrollment/probe temporal ordering to verify."
            )

        report = audit_record_order_dataset(
            dataset,
            data_split_mode,
            signal_type=signal_type,
            only_healthy=arguments.only_healthy,
            limit_records=arguments.limit_records,
        )
        audit_passed = report["enrollment_precedes_probe"]

    if not arguments.quiet:
        print(summarize_report(report))

    if arguments.output_json is not None:
        output_path = Path(arguments.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                report,
                output_file,
                indent=2,
                default=str,
            )

        print(f"[INFO] Report written to {output_path}")

    return 0 if audit_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
