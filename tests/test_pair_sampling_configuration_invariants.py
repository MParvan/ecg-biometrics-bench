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

import utils

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

PINNED_ADDITIONAL_TARGET_FARS = [
    0.1,
    0.01,
    0.0001,
]

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

    def test_verification_configs_pin_additional_target_fars(self):
        verification_checked = 0
        identification_checked = 0

        for path, config in self.configurations:
            task = config.get("task")

            if task in VERIFICATION_TASKS:
                verification_checked += 1
                self.assertEqual(
                    config.get("target_fars"),
                    PINNED_ADDITIONAL_TARGET_FARS,
                    path.name,
                )
                self.assertNotIn(
                    utils._MANDATORY_VERIFICATION_FAR,
                    config["target_fars"],
                    path.name,
                )
            elif task in IDENTIFICATION_TASKS:
                identification_checked += 1
                self.assertNotIn(
                    "target_fars",
                    config,
                    path.name,
                )

        self.assertEqual(verification_checked, 124)
        self.assertEqual(identification_checked, 110)
        self.assertEqual(
            utils._resolve_target_fars(
                PINNED_ADDITIONAL_TARGET_FARS
            ),
            [0.0001, 0.001, 0.01, 0.1],
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

    def test_no_configuration_enables_multi_template_enrollment(self):
        # Every shipped configuration uses the existing fusion enrollment
        # path. None of them opts into multi-template enrollment, and none
        # carries a multi-template-only parameter with fusion left implicit.
        multi_template_only_keys = {
            "num_templates_per_identity",
            "template_selection_method",
            "template_score_aggregation",
        }

        for path, config in self.configurations:
            self.assertIn(
                config.get("enrollment_template_mode", "fusion"),
                ("fusion",),
                f"{path.name} does not use the fusion reproduction path",
            )

            present = multi_template_only_keys & set(config)
            self.assertFalse(
                present,
                (
                    f"{path.name} sets multi-template-only parameter(s) "
                    f"while enrollment_template_mode is not set: "
                    f"{sorted(present)}"
                ),
            )


if __name__ == "__main__":
    unittest.main()
