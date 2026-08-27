import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _required_session_orders(hash_seed):
    code = r"""
import json

from load_dataset import (
    load_cybhi_dataset,
    load_heartprint_dataset,
)


heartprint = load_heartprint_dataset(
    data_split_mode="single-session",
    session_for_single_session_evaluation=["session1"],
    train_sessions=["session1"],
    enroll_sessions=["session1"],
    probe_sessions=["session2"],
)

cybhi = load_cybhi_dataset(
    data_split_mode="single-session",
    session_for_single_session_evaluation=["long-term_S1"],
    train_sessions=["long-term_S1"],
    enroll_sessions=["long-term_S1"],
    probe_sessions=["long-term_S2"],
)

print(
    json.dumps(
        {
            "heartprint": heartprint.required_cross_sessions,
            "cybhi": cybhi.required_cross_sessions,
        },
        sort_keys=True,
    )
)
"""

    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(hash_seed)

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            code,
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    return json.loads(
        completed.stdout.strip()
    )


def test_required_cross_session_order_is_hash_seed_independent():
    results = [
        _required_session_orders(seed)
        for seed in range(1, 17)
    ]

    expected = {
        "heartprint": [
            "session1",
            "session2",
        ],
        "cybhi": [
            "long-term_S1",
            "long-term_S2",
        ],
    }

    assert all(
        result == expected
        for result in results
    )


def test_required_cross_session_order_preserves_reverse_role_direction():
    code = r"""
import json

from load_dataset import (
    load_cybhi_dataset,
    load_heartprint_dataset,
)


heartprint = load_heartprint_dataset(
    data_split_mode="cross-session",
    train_sessions=["session2"],
    enroll_sessions=["session2"],
    probe_sessions=["session1"],
)

cybhi = load_cybhi_dataset(
    data_split_mode="cross-session",
    train_sessions=["long-term_S2"],
    enroll_sessions=["long-term_S2"],
    probe_sessions=["long-term_S1"],
)

print(
    json.dumps(
        {
            "heartprint": {
                "train": heartprint.train_sessions,
                "enroll": heartprint.enroll_sessions,
                "probe": heartprint.probe_sessions,
                "required": heartprint.required_cross_sessions,
            },
            "cybhi": {
                "train": cybhi.train_sessions,
                "enroll": cybhi.enroll_sessions,
                "probe": cybhi.probe_sessions,
                "required": cybhi.required_cross_sessions,
            },
        },
        sort_keys=True,
    )
)
"""

    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "7"

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            code,
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(
        completed.stdout.strip()
    )

    assert result["heartprint"] == {
        "train": ["session2"],
        "enroll": ["session2"],
        "probe": ["session1"],
        "required": [
            "session2",
            "session1",
        ],
    }

    assert result["cybhi"] == {
        "train": ["long-term_S2"],
        "enroll": ["long-term_S2"],
        "probe": ["long-term_S1"],
        "required": [
            "long-term_S2",
            "long-term_S1",
        ],
    }
