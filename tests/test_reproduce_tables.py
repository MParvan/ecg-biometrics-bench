import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import reproduce_tables as driver

SHIPPED_CONFIG_ROOT = PROJECT_ROOT / "configs" / "paper_reproduction"


class ShippedConfigurationDiscoveryTests(unittest.TestCase):
    """
    Every shipped configuration must be parseable into table coordinates.
    """

    @classmethod
    def setUpClass(cls):
        cls.entries = driver.discover_configurations(
            SHIPPED_CONFIG_ROOT
        )

    def test_all_configurations_are_discovered(self):
        self.assertEqual(len(self.entries), 150)

    def test_every_entry_maps_to_a_table(self):
        for entry in self.entries:
            with self.subTest(config=entry.path.name):
                self.assertIn(entry.table, (5, 6, 7))

    def test_expected_configuration_counts_per_dataset(self):
        counts = {}

        for entry in self.entries:
            counts[entry.dataset] = counts.get(entry.dataset, 0) + 1

        self.assertEqual(
            counts,
            {
                "ecgid": 28,
                "ptb": 28,
                "ptbxl": 14,
                "mitbih": 16,
                "nsrdb": 16,
                "cybhi": 20,
                "heartprint": 28,
            },
        )

    def test_task_number_matches_the_task_type(self):
        identification_tasks = {1, 3, 5, 7}
        verification_tasks = {2, 4, 6, 8}

        for entry in self.entries:
            with self.subTest(config=entry.path.name):
                if entry.task_type == "identification":
                    self.assertIn(
                        entry.task,
                        identification_tasks,
                    )
                else:
                    self.assertIn(
                        entry.task,
                        verification_tasks,
                    )

    def test_task_number_matches_the_setting(self):
        subject_disjoint_tasks = {3, 4, 7, 8}

        for entry in self.entries:
            with self.subTest(config=entry.path.name):
                if entry.setting == "subject_disjoint":
                    self.assertIn(
                        entry.task,
                        subject_disjoint_tasks,
                    )
                else:
                    self.assertNotIn(
                        entry.task,
                        subject_disjoint_tasks,
                    )

    def test_configuration_task_field_matches_the_filename(self):
        for entry in self.entries:
            with self.subTest(config=entry.path.name):
                self.assertEqual(
                    entry.configuration.get("task"),
                    entry.task,
                )

    def test_configuration_dataset_field_matches_the_filename(self):
        for entry in self.entries:
            with self.subTest(config=entry.path.name):
                self.assertEqual(
                    entry.configuration.get("dataset"),
                    entry.dataset,
                )

    def test_every_protocol_has_a_manuscript_label(self):
        for entry in self.entries:
            with self.subTest(protocol=entry.protocol):
                self.assertIn(
                    entry.protocol,
                    driver.PROTOCOL_LABELS,
                )

    def test_every_protocol_has_a_defined_row_order(self):
        for entry in self.entries:
            with self.subTest(
                dataset=entry.dataset,
                protocol=entry.protocol,
            ):
                self.assertIn(
                    entry.protocol,
                    driver.PROTOCOL_ORDER[entry.dataset],
                )

    def test_each_row_pairs_identification_with_verification(self):
        by_row = {}

        for entry in self.entries:
            by_row.setdefault(entry.row_key, set()).add(
                entry.task_type
            )

        for row_key, task_types in by_row.items():
            with self.subTest(row=row_key):
                if row_key[0] == "ptbxl":
                    # Identification metrics were not reported for PTB-XL.
                    self.assertEqual(
                        task_types,
                        {"verification"},
                    )
                else:
                    self.assertEqual(
                        task_types,
                        {"identification", "verification"},
                    )


class FilteringTests(unittest.TestCase):
    """
    Selection flags restrict the plan without reordering it.
    """

    def setUp(self):
        self.entries = driver.discover_configurations(
            SHIPPED_CONFIG_ROOT
        )

    def test_table_filter(self):
        selected = driver.filter_configurations(
            self.entries,
            tables=[6],
        )

        self.assertEqual(len(selected), 32)
        self.assertEqual(
            {entry.dataset for entry in selected},
            {"mitbih", "nsrdb"},
        )

    def test_dataset_filter(self):
        selected = driver.filter_configurations(
            self.entries,
            datasets=["cybhi"],
        )

        self.assertEqual(len(selected), 20)

    def test_task_filter(self):
        selected = driver.filter_configurations(
            self.entries,
            tasks=[8],
        )

        self.assertTrue(selected)
        self.assertTrue(
            all(entry.task == 8 for entry in selected)
        )

    def test_combined_filters(self):
        selected = driver.filter_configurations(
            self.entries,
            tables=[6],
            datasets=["nsrdb"],
            tasks=[5, 6],
        )

        self.assertEqual(len(selected), 6)


class SmokeCommandTests(unittest.TestCase):
    """
    Smoke mode must override the expensive settings and nothing else.
    """

    def test_smoke_overrides_are_appended(self):
        entry = driver.discover_configurations(
            SHIPPED_CONFIG_ROOT
        )[0]

        command = driver.build_command(entry, smoke=True)

        for name, value in driver.SMOKE_OVERRIDES.items():
            self.assertIn(f"--{name}", command)
            self.assertIn(str(value), command)

    def test_default_command_has_no_overrides(self):
        entry = driver.discover_configurations(
            SHIPPED_CONFIG_ROOT
        )[0]

        command = driver.build_command(entry)

        self.assertEqual(
            command[-2:],
            ["--config", str(entry.path)],
        )

    def test_extra_arguments_are_forwarded(self):
        entry = driver.discover_configurations(
            SHIPPED_CONFIG_ROOT
        )[0]

        command = driver.build_command(
            entry,
            extra_arguments=["--intelligent_weight_loading"],
        )

        self.assertEqual(
            command[-1],
            "--intelligent_weight_loading",
        )


class TableAssemblyTests(unittest.TestCase):
    """
    Collection pairs the two runs of a row and reports incomplete rows.

    The fixture writes an isolated configuration tree and result tree so the
    test never reads or writes the real artifact directory.
    """

    def setUp(self):
        self.root = Path(
            tempfile.mkdtemp(prefix="reproduce_tables_")
        )
        self.config_root = self.root / "configs"
        self.results_root = self.root / "results"
        (self.config_root / "mitbih").mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def write_configuration(self, name, task, results_subdir):
        path = self.config_root / "mitbih" / f"{name}.yaml"
        results_dir = self.results_root / results_subdir

        path.write_text(
            yaml.safe_dump(
                {
                    "dataset": "mitbih",
                    "task": task,
                    "results_dir": str(results_dir),
                }
            ),
            encoding="utf-8",
        )

        return results_dir

    def write_result_log(self, results_dir, task, metrics):
        target = results_dir / "mitbih"
        target.mkdir(parents=True, exist_ok=True)

        log_path = (
            target / f"{driver.TASK_LOG_NAMES[task]}.jsonl"
        )

        record = {
            "task": driver.TASK_LOG_NAMES[task],
            "dataset": "mitbih",
            "results": {
                name: {
                    "mean": value,
                    "std": 0.01,
                    "display": f"{value} +/- 0.01",
                }
                for name, value in metrics.items()
            },
        }

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def test_identification_and_verification_merge_into_one_row(self):
        identification_dir = self.write_configuration(
            "mitbih_long_term_closed_set_task05_identification",
            5,
            "id",
        )
        verification_dir = self.write_configuration(
            "mitbih_long_term_closed_set_task06_verification",
            6,
            "ver",
        )

        self.write_result_log(
            identification_dir,
            5,
            {
                "Rank-1 Accuracy": 0.81,
                "Rank-5 Accuracy": 0.93,
            },
        )
        self.write_result_log(
            verification_dir,
            6,
            {
                "EER": 0.09,
                "AUC": 0.97,
                "d-prime": 2.7,
                "TAR@0.1%FAR": 0.55,
            },
        )

        entries = driver.discover_configurations(
            self.config_root
        )
        rows = driver.collect_rows(entries)

        self.assertEqual(len(rows), 1)

        row = next(iter(rows.values()))

        self.assertEqual(row["missing"], [])
        self.assertEqual(
            set(row["metrics"]),
            set(driver.TABLE_METRICS),
        )
        self.assertEqual(
            driver.format_metric(
                row["metrics"]["Rank-1 Accuracy"]
            ),
            "0.8100 +/- 0.0100",
        )

    def test_missing_run_is_reported_not_silently_dropped(self):
        self.write_configuration(
            "mitbih_long_term_closed_set_task05_identification",
            5,
            "id",
        )
        verification_dir = self.write_configuration(
            "mitbih_long_term_closed_set_task06_verification",
            6,
            "ver",
        )

        self.write_result_log(
            verification_dir,
            6,
            {
                "EER": 0.09,
                "AUC": 0.97,
                "d-prime": 2.7,
                "TAR@0.1%FAR": 0.55,
            },
        )

        entries = driver.discover_configurations(
            self.config_root
        )
        rows = driver.collect_rows(entries)
        row = next(iter(rows.values()))

        self.assertTrue(row["missing"])
        self.assertIn(
            "identification",
            row["missing"][0],
        )

    def test_latest_record_wins_when_a_log_has_several(self):
        results_dir = self.write_configuration(
            "mitbih_long_term_closed_set_task05_identification",
            5,
            "id",
        )

        self.write_result_log(
            results_dir,
            5,
            {
                "Rank-1 Accuracy": 0.10,
                "Rank-5 Accuracy": 0.20,
            },
        )
        self.write_result_log(
            results_dir,
            5,
            {
                "Rank-1 Accuracy": 0.81,
                "Rank-5 Accuracy": 0.93,
            },
        )

        entries = driver.discover_configurations(
            self.config_root
        )
        rows = driver.collect_rows(entries)
        row = next(iter(rows.values()))

        self.assertEqual(
            row["metrics"]["Rank-1 Accuracy"]["mean"],
            0.81,
        )

    def test_markdown_marks_absent_metrics(self):
        self.write_configuration(
            "mitbih_long_term_closed_set_task05_identification",
            5,
            "id",
        )

        entries = driver.discover_configurations(
            self.config_root
        )
        rows = driver.collect_rows(entries)

        markdown = driver.render_markdown(rows, 6)

        self.assertIn("### Table 6", markdown)
        self.assertIn("| - |", markdown)


class MetricFormattingTests(unittest.TestCase):
    """
    Metric rendering must never invent precision it does not have.
    """

    def test_mean_and_std_are_rendered(self):
        self.assertEqual(
            driver.format_metric(
                {"mean": 0.5, "std": 0.25}
            ),
            "0.5000 +/- 0.2500",
        )

    def test_scalar_is_rendered(self):
        self.assertEqual(
            driver.format_metric(0.5),
            "0.5000",
        )

    def test_missing_metric_is_marked(self):
        self.assertEqual(
            driver.format_metric(None),
            "-",
        )

    def test_display_only_value_falls_back_to_display(self):
        self.assertEqual(
            driver.format_metric({"display": "0.5 +/- 0.1"}),
            "0.5 +/- 0.1",
        )


if __name__ == "__main__":
    unittest.main()
