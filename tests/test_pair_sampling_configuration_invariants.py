"""
Pin verification pair sampling across the experiment configuration corpus.

Verification experiments retain every genuine comparison and use a bounded,
reproducible sample of impostor comparisons. Identification experiments do not
carry verification-only pair-sampling parameters.
"""

import sys
import unittest
from collections import Counter
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_ROOT = PROJECT_ROOT / "configs"

VERIFICATION_TASKS = {
    2,
    4,
    6,
    8,
}

IDENTIFICATION_TASKS = {
    1,
    3,
    5,
    7,
}

PAIR_KEYS = {
    "sampling_mode",
    "num_pairs",
    "pair_sampling_mode",
    "pair_sampling_budget",
    "max_impostor_pairs",
    "pair_sampling_seed",
}

EXPECTED_TASK_COUNTS = {
    1: 9,
    2: 11,
    3: 9,
    4: 11,
    5: 46,
    6: 51,
    7: 46,
    8: 51,
}

EXPECTED_FOLDER_COUNTS = {
    "model_comparison": 84,
    "paper_reproduction": 150,
}


def configuration_paths():
    if not CONFIG_ROOT.exists():
        return []

    return sorted(
        CONFIG_ROOT.rglob(
            "*.yaml"
        )
    )


def load(path):
    with path.open(
        encoding="utf-8"
    ) as handle:
        return (
            yaml.safe_load(handle)
            or {}
        )


class PairSamplingConfigurationInvariants(
    unittest.TestCase
):
    @classmethod
    def setUpClass(cls):
        cls.configurations = [
            (
                path,
                load(path),
            )
            for path in configuration_paths()
        ]

        if not cls.configurations:
            raise unittest.SkipTest(
                "experiment configuration corpus "
                "is not present"
            )

    def test_complete_configuration_matrix_is_pinned(self):
        self.assertEqual(
            len(self.configurations),
            234,
        )

        task_counts = Counter(
            config.get(
                "task"
            )
            for _, config
            in self.configurations
        )

        self.assertEqual(
            dict(
                sorted(
                    task_counts.items()
                )
            ),
            EXPECTED_TASK_COUNTS,
        )

        folder_counts = Counter(
            path.relative_to(
                CONFIG_ROOT
            ).parts[0]
            for path, _
            in self.configurations
        )

        self.assertEqual(
            dict(
                sorted(
                    folder_counts.items()
                )
            ),
            EXPECTED_FOLDER_COUNTS,
        )

    def test_verification_configs_use_all_genuine_sampling(self):
        checked = 0

        for path, config in self.configurations:
            if (
                config.get(
                    "task"
                )
                not in VERIFICATION_TASKS
            ):
                continue

            checked += 1

            self.assertEqual(
                config.get(
                    "pair_sampling_mode"
                ),
                "all_genuine",
                path.name,
            )

            self.assertEqual(
                config.get(
                    "max_impostor_pairs"
                ),
                1000000,
                path.name,
            )

            self.assertEqual(
                config.get(
                    "pair_sampling_seed"
                ),
                42,
                path.name,
            )

            self.assertNotIn(
                "pair_sampling_budget",
                config,
                path.name,
            )

            self.assertNotIn(
                "sampling_mode",
                config,
                path.name,
            )

            self.assertNotIn(
                "num_pairs",
                config,
                path.name,
            )

        self.assertEqual(
            checked,
            124,
        )

    def test_identification_configs_omit_pair_sampling_settings(self):
        checked = 0

        for path, config in self.configurations:
            if (
                config.get(
                    "task"
                )
                not in IDENTIFICATION_TASKS
            ):
                continue

            checked += 1

            present = (
                PAIR_KEYS
                & set(
                    config
                )
            )

            self.assertFalse(
                present,
                (
                    f"{path.name} contains "
                    "verification-only pair settings: "
                    f"{sorted(present)}"
                ),
            )

        self.assertEqual(
            checked,
            110,
        )


if __name__ == "__main__":
    unittest.main()
