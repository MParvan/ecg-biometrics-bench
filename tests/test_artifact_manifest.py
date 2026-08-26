import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_artifact_manifest as manifest_tool


class ChecksumTests(unittest.TestCase):
    """
    Checksums must identify the exact bytes a reader downloads.
    """

    def setUp(self):
        self.root = Path(
            tempfile.mkdtemp(prefix="manifest_checksum_")
        )

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_checksum_matches_hashlib(self):
        import hashlib

        payload = b"weights" * 1000
        path = self.root / "sample.bin"
        path.write_bytes(payload)

        self.assertEqual(
            manifest_tool.compute_checksum(path),
            hashlib.sha256(payload).hexdigest(),
        )

    def test_different_content_yields_different_checksum(self):
        first = self.root / "a.bin"
        second = self.root / "b.bin"
        first.write_bytes(b"alpha")
        second.write_bytes(b"beta")

        self.assertNotEqual(
            manifest_tool.compute_checksum(first),
            manifest_tool.compute_checksum(second),
        )


class WeightArtifactTests(unittest.TestCase):
    """
    Weight entries must be described by what produced them.
    """

    def setUp(self):
        self.root = Path(
            tempfile.mkdtemp(prefix="manifest_weights_")
        )
        self.weight_dir = self.root / "weights"
        self.weight_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def write_entry(self, uid, metadata):
        (self.weight_dir / f"{uid}.pth").write_bytes(
            b"fake weights"
        )

        if metadata is not None:
            (self.weight_dir / f"{uid}.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

    def test_metadata_fields_are_recovered(self):
        self.write_entry(
            "abc123",
            {
                "training_regime": "cross_session_closed_set",
                "model": "DeepECG",
                "epochs": 250,
                "actual_epochs": 137,
                "batch_size": 256,
                "lr": 0.001,
                "seed": 42,
                "classes": 90,
                "loader_identity": {
                    "loader_class": "load_ecgid_dataset",
                    "root_dir": "ecgid",
                    "settings": {
                        "data_split_mode": "leave-last-out-long-term",
                    },
                },
            },
        )

        artifacts = manifest_tool.collect_weight_artifacts(
            self.root
        )

        self.assertEqual(len(artifacts), 1)

        artifact = artifacts[0]

        self.assertEqual(artifact["dataset"], "ecgid")
        self.assertEqual(
            artifact["data_split_mode"],
            "leave-last-out-long-term",
        )
        self.assertEqual(artifact["model"], "DeepECG")
        self.assertEqual(artifact["seed"], 42)
        self.assertEqual(artifact["configured_epochs"], 250)
        self.assertEqual(artifact["trained_epochs"], 137)
        self.assertEqual(artifact["setting"], "closed_set")
        self.assertIn("sha256", artifact)

    def test_dataset_falls_back_to_the_loader_class(self):
        self.write_entry(
            "def456",
            {
                "loader_identity": {
                    "loader_class": "load_heartprint_dataset",
                },
            },
        )

        artifacts = manifest_tool.collect_weight_artifacts(
            self.root
        )

        self.assertEqual(
            artifacts[0]["dataset"],
            "heartprint",
        )

    def test_missing_metadata_is_reported_not_fatal(self):
        self.write_entry("ghi789", None)

        artifacts = manifest_tool.collect_weight_artifacts(
            self.root
        )

        self.assertEqual(len(artifacts), 1)
        self.assertFalse(
            artifacts[0]["metadata_present"]
        )
        self.assertIsNone(artifacts[0]["dataset"])

    def test_corrupt_metadata_is_reported_not_fatal(self):
        (self.weight_dir / "bad.pth").write_bytes(b"x")
        (self.weight_dir / "bad.json").write_text(
            "{not json",
            encoding="utf-8",
        )

        artifacts = manifest_tool.collect_weight_artifacts(
            self.root
        )

        self.assertEqual(len(artifacts), 1)
        self.assertFalse(
            artifacts[0]["metadata_present"]
        )

    def test_checksums_can_be_skipped(self):
        self.write_entry("abc123", {"model": "DeepECG"})

        artifacts = manifest_tool.collect_weight_artifacts(
            self.root,
            compute_checksums=False,
        )

        self.assertNotIn("sha256", artifacts[0])

    def test_absent_cache_directory_is_empty_not_fatal(self):
        self.assertEqual(
            manifest_tool.collect_weight_artifacts(
                self.root / "does_not_exist"
            ),
            [],
        )


class CoverageTests(unittest.TestCase):
    """
    Coverage must distinguish executed configurations from pending ones.
    """

    def setUp(self):
        self.entries = manifest_tool.discover_configurations(
            PROJECT_ROOT / "configs" / "paper_reproduction"
        )
        self.root = Path(
            tempfile.mkdtemp(prefix="manifest_coverage_")
        )

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_all_configurations_are_listed(self):
        coverage = manifest_tool.collect_configuration_coverage(
            self.entries,
            publication_mode=False,
        )

        self.assertEqual(len(coverage), 150)

    def test_unexecuted_configurations_are_marked(self):
        # Point every configuration at an empty results tree.
        for entry in self.entries:
            entry.configuration["results_dir"] = str(self.root)

        coverage = manifest_tool.collect_configuration_coverage(
            self.entries,
            publication_mode=False,
        )

        self.assertTrue(
            all(not row["executed"] for row in coverage)
        )

    def test_executed_configuration_reports_its_seeds(self):
        entry = self.entries[0]
        entry.configuration["results_dir"] = str(self.root)

        log_dir = self.root / entry.dataset
        log_dir.mkdir(parents=True, exist_ok=True)

        log_name = manifest_tool.TASK_LOG_NAMES[entry.task]
        record = {
            "experiment_time": "2026-08-02T12:00:00",
            "results": {},
            "per_run_results": [
                {"run_index": index + 1, "seed": 42 + index}
                for index in range(5)
            ],
        }

        (log_dir / f"{log_name}.jsonl").write_text(
            json.dumps(record) + "\n",
            encoding="utf-8",
        )

        coverage = manifest_tool.collect_configuration_coverage(
            [entry],
            publication_mode=False,
        )

        self.assertTrue(coverage[0]["executed"])
        self.assertEqual(coverage[0]["runs"], 5)
        self.assertEqual(
            coverage[0]["seeds"],
            [42, 43, 44, 45, 46],
        )


class SummaryAndRenderingTests(unittest.TestCase):
    """
    The rendered manifest must be readable with or without artifacts.
    """

    def _coverage(self, executed_count):
        return [
            {
                "configuration": f"config_{index}.yaml",
                "table": 5,
                "dataset": "ecgid",
                "protocol": "all_available",
                "setting": "closed_set",
                "task": 1,
                "task_type": "identification",
                "executed": index < executed_count,
                "runs": 5 if index < executed_count else 0,
                "seeds": (
                    [42, 43, 44, 45, 46]
                    if index < executed_count
                    else []
                ),
            }
            for index in range(4)
        ]

    def test_summary_counts_executed_and_pending(self):
        summary = manifest_tool.summarize(
            [],
            [],
            self._coverage(3),
        )

        self.assertEqual(
            summary["configurations_total"],
            4,
        )
        self.assertEqual(
            summary["configurations_executed"],
            3,
        )
        self.assertEqual(
            summary["configurations_pending"],
            1,
        )

    def test_summary_totals_artifact_bytes(self):
        summary = manifest_tool.summarize(
            [{"size_bytes": 1024}],
            [{"size_bytes": 2048}],
            self._coverage(0),
        )

        self.assertEqual(
            summary["total_artifact_bytes"],
            3072,
        )

    def test_markdown_renders_without_weights(self):
        coverage = self._coverage(1)
        manifest = {
            "summary": manifest_tool.summarize(
                [],
                [],
                coverage,
            ),
            "configuration_coverage": coverage,
            "trained_weights": [],
        }

        markdown = manifest_tool.render_markdown(manifest)

        self.assertIn("# Released artifacts", markdown)
        self.assertIn(
            "No trained weights were found",
            markdown,
        )

    def test_markdown_renders_weight_rows(self):
        coverage = self._coverage(1)
        weights = [
            {
                "file": "abc.pth",
                "dataset": "ecgid",
                "data_split_mode": "all-available",
                "training_regime": "intra_session_closed_set",
                "model": "DeepECG",
                "seed": 42,
                "configured_epochs": 250,
                "trained_epochs": 137,
                "size_bytes": 3 * 1024 * 1024,
                "sha256": "0" * 64,
            }
        ]
        manifest = {
            "summary": manifest_tool.summarize(
                weights,
                [],
                coverage,
            ),
            "configuration_coverage": coverage,
            "trained_weights": weights,
        }

        markdown = manifest_tool.render_markdown(manifest)

        self.assertIn("abc.pth", markdown)
        self.assertIn("DeepECG", markdown)
        self.assertIn("137", markdown)


if __name__ == "__main__":
    unittest.main()
