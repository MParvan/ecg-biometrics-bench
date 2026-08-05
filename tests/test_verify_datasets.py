import io
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import load_dataset
from scripts import verify_datasets


class FakeLoader:
    """Stands in for a dataset loader without touching the network."""

    def __init__(self, root, archive, on_download=None):
        self.dataset_root = root
        self.zip_path = archive
        self._on_download = on_download

    def download(self):
        if self._on_download is not None:
            self._on_download()


class DatasetCoverageTests(unittest.TestCase):
    def test_every_configured_dataset_is_checked(self):
        # A dataset added to the framework but missing here would silently go
        # unverified, which is the failure this script exists to prevent.
        configured = set(load_dataset.CONFIG["datasets"])
        self.assertEqual(set(verify_datasets.DATASETS), configured)

    def test_each_name_resolves_to_a_loader(self):
        for dataset in verify_datasets.DATASETS:
            self.assertTrue(callable(verify_datasets._loader_for(dataset)))


class DirectorySummaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def test_files_are_counted_recursively(self):
        (self.directory / "nested").mkdir()
        (self.directory / "a.hea").write_text("x")
        (self.directory / "nested" / "b.dat").write_text("yy")

        summary = verify_datasets._directory_summary(self.directory)

        self.assertEqual(summary["files"], 2)
        self.assertEqual(summary["bytes"], 3)
        self.assertEqual(summary["extensions"], {".hea": 1, ".dat": 1})

    def test_a_missing_directory_reports_zeroes(self):
        summary = verify_datasets._directory_summary(self.directory / "absent")
        self.assertEqual(summary["files"], 0)
        self.assertEqual(summary["bytes"], 0)


class VerifyDatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.root = self.directory / "ecgid"
        self.archive = self.directory / "ECGID_Dataset.zip"

    def _patch_loader(self, loader):
        return mock.patch.object(
            verify_datasets, "_loader_for", return_value=lambda **kwargs: loader
        )

    def test_a_successful_download_is_reported_as_ok(self):
        def populate():
            self.root.mkdir(parents=True)
            (self.root / "record.hea").write_text("header")

        loader = FakeLoader(self.root, self.archive, on_download=populate)

        with self._patch_loader(loader):
            record = verify_datasets.verify_dataset("ecgid")

        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["files"], 1)

    def test_an_existing_dataset_is_distinguished_from_a_fresh_one(self):
        self.root.mkdir(parents=True)
        (self.root / "record.hea").write_text("header")
        loader = FakeLoader(self.root, self.archive)

        with self._patch_loader(loader):
            record = verify_datasets.verify_dataset("ecgid")

        self.assertEqual(record["status"], "already-present")

    def test_a_missing_tool_is_its_own_status(self):
        # This has to be separable from a genuine failure, because the fix is
        # installing a package rather than retrying the download.
        def refuse():
            raise load_dataset.MissingArchiveToolError("install unar")

        loader = FakeLoader(self.root, self.archive, on_download=refuse)

        with self._patch_loader(loader):
            record = verify_datasets.verify_dataset("heartprint")

        self.assertEqual(record["status"], "missing-tool")
        self.assertIn("install unar", record["detail"])

    def test_a_download_error_is_reported_with_a_traceback(self):
        def fail():
            raise RuntimeError("connection reset")

        loader = FakeLoader(self.root, self.archive, on_download=fail)

        with self._patch_loader(loader):
            record = verify_datasets.verify_dataset("ecgid")

        self.assertEqual(record["status"], "failed")
        self.assertIn("connection reset", record["detail"])
        self.assertIn("traceback", record)

    def test_a_silent_no_op_download_is_caught(self):
        # A download that returns without error but writes nothing must not be
        # reported as success, or the failure surfaces much later as a parsing
        # error against an empty directory.
        loader = FakeLoader(self.root, self.archive)

        with self._patch_loader(loader):
            record = verify_datasets.verify_dataset("ecgid")

        self.assertEqual(record["status"], "failed")
        self.assertIn("empty", record["detail"])

    def test_the_archive_format_is_reported_when_present(self):
        def populate():
            self.root.mkdir(parents=True)
            (self.root / "record.hea").write_text("header")

        self.archive.write_bytes(b"Rar!\x1a\x07\x01\x00payload")
        loader = FakeLoader(self.root, self.archive, on_download=populate)

        with self._patch_loader(loader):
            record = verify_datasets.verify_dataset("heartprint")

        self.assertEqual(record["archive_format"], "rar")


class ExitCodeTests(unittest.TestCase):
    def test_check_tools_exits_zero_without_downloading(self):
        buffer = io.StringIO()
        with mock.patch.object(verify_datasets, "verify_dataset") as verify:
            with redirect_stdout(buffer):
                code = verify_datasets.main(["--check-tools"])

        self.assertEqual(code, 0)
        verify.assert_not_called()

    def test_a_failure_produces_a_nonzero_exit_code(self):
        buffer = io.StringIO()
        failure = {"dataset": "ecgid", "status": "failed", "detail": "no route"}

        with mock.patch.object(verify_datasets, "verify_dataset", return_value=failure):
            with redirect_stdout(buffer):
                code = verify_datasets.main(["--datasets", "ecgid"])

        self.assertEqual(code, 1)
        self.assertIn("no route", buffer.getvalue())

    def test_success_produces_a_zero_exit_code(self):
        buffer = io.StringIO()
        success = {
            "dataset": "ecgid",
            "status": "ok",
            "files": 3,
            "bytes": 100,
            "extensions": {".hea": 3},
        }

        with mock.patch.object(verify_datasets, "verify_dataset", return_value=success):
            with redirect_stdout(buffer):
                code = verify_datasets.main(["--datasets", "ecgid"])

        self.assertEqual(code, 0)

    def test_the_report_can_be_written_as_json(self):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        destination = directory / "report.json"
        success = {"dataset": "ecgid", "status": "ok", "files": 1, "bytes": 1}

        with mock.patch.object(verify_datasets, "verify_dataset", return_value=success):
            with redirect_stdout(io.StringIO()):
                verify_datasets.main(
                    ["--datasets", "ecgid", "--output-json", str(destination)]
                )

        self.assertTrue(destination.exists())
        self.assertIn("ecgid", destination.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
