import io
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import load_dataset

_sleep_patch = None


def setUpModule():
    # Retry backoff is real time that would otherwise be spent in the suite.
    global _sleep_patch
    _sleep_patch = mock.patch.object(load_dataset.time, "sleep")
    _sleep_patch.start()


def tearDownModule():
    _sleep_patch.stop()


# Leading bytes of each archive format, used to build fixtures whose header is
# real even though the payload is not.
ZIP_MAGIC = b"PK\x03\x04"
RAR4_MAGIC = b"Rar!\x1a\x07\x00"
RAR5_MAGIC = b"Rar!\x1a\x07\x01\x00"


def build_zip(path, entries):
    """Write a ZIP archive containing the given {name: text} mapping."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in entries.items():
            archive.writestr(name, text)
    return path


class FakeResponse:
    """Minimal stand-in for a streamed requests response."""

    def __init__(self, payload, headers=None, status=200, reason="OK"):
        self.payload = payload
        self.headers = headers if headers is not None else {}
        self.status = status
        self.status_code = status
        self.reason = reason

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            # The real exception carries the response, which is how a retryable
            # status is told apart from a refusal.
            raise requests.HTTPError(f"{self.status} {self.reason}", response=self)

    def iter_content(self, chunk_size=1):
        stream = io.BytesIO(self.payload)
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                return
            yield chunk


class ArchiveFormatDetectionTests(unittest.TestCase):
    """
    Archives must be identified by content, because repositories publish them
    under names that do not always match the format.
    """

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def test_zip_is_detected(self):
        path = build_zip(self.directory / "archive.zip", {"a.txt": "a"})
        self.assertEqual(load_dataset.detect_archive_format(path), "zip")

    def test_both_rar_generations_are_detected(self):
        for name, magic in (("four.rar", RAR4_MAGIC), ("five.rar", RAR5_MAGIC)):
            path = self.directory / name
            path.write_bytes(magic + b"payload")
            self.assertEqual(load_dataset.detect_archive_format(path), "rar")

    def test_extension_does_not_override_content(self):
        path = self.directory / "actually_a_rar.zip"
        path.write_bytes(RAR5_MAGIC + b"payload")
        self.assertEqual(load_dataset.detect_archive_format(path), "rar")

    def test_unrecognised_and_missing_files_return_none(self):
        html = self.directory / "error_page.zip"
        html.write_bytes(b"<!DOCTYPE html><html>Not found</html>")
        self.assertIsNone(load_dataset.detect_archive_format(html))
        self.assertIsNone(load_dataset.detect_archive_format(self.directory / "absent"))

    def test_empty_file_is_not_an_archive(self):
        empty = self.directory / "empty.zip"
        empty.touch()
        self.assertIsNone(load_dataset.detect_archive_format(empty))


class ArchiveSuffixReconciliationTests(unittest.TestCase):
    """
    A RAR published under a .zip name must be renamed, because extraction
    tools dispatch on the file name.
    """

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def test_mismatched_suffix_is_corrected(self):
        path = self.directory / "Heartprint_Dataset.zip"
        path.write_bytes(RAR5_MAGIC + b"payload")

        corrected = load_dataset._reconcile_archive_suffix(path)

        self.assertEqual(corrected.name, "Heartprint_Dataset.rar")
        self.assertTrue(corrected.exists())
        self.assertFalse(path.exists())

    def test_matching_suffix_is_left_alone(self):
        path = build_zip(self.directory / "dataset.zip", {"a.txt": "a"})
        self.assertEqual(load_dataset._reconcile_archive_suffix(path), path)

    def test_unrecognised_file_is_left_alone(self):
        path = self.directory / "dataset.zip"
        path.write_bytes(b"not an archive at all")
        self.assertEqual(load_dataset._reconcile_archive_suffix(path), path)


class RarBackendTests(unittest.TestCase):
    """
    Python cannot unpack RAR without an external tool, so its absence must be
    reported as a named, installable dependency.
    """

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def test_backend_is_reported_when_present(self):
        with mock.patch.object(load_dataset.shutil, "which", side_effect=lambda name: (
            "/usr/bin/unar" if name == "unar" else None
        )):
            self.assertEqual(load_dataset.available_rar_backend(), "unar")

    def test_absent_backend_reports_none(self):
        with mock.patch.object(load_dataset.shutil, "which", return_value=None):
            self.assertIsNone(load_dataset.available_rar_backend())

    def test_missing_backend_raises_an_actionable_error(self):
        archive = self.directory / "dataset.rar"
        archive.write_bytes(RAR5_MAGIC + b"payload")
        destination = self.directory / "heartprint"

        with mock.patch.object(load_dataset.shutil, "which", return_value=None):
            with self.assertRaises(load_dataset.MissingArchiveToolError) as caught:
                load_dataset._unpack_archive(archive, self.directory, destination)

        message = str(caught.exception)
        # The message has to name a package the reader can install and the
        # directory to unpack into by hand, or it is not actionable.
        self.assertIn("unar", message)
        self.assertIn("apt-get install", message)
        self.assertIn("brew install", message)
        self.assertIn(str(destination), message)

    def test_unrecognised_archive_is_reported_as_such(self):
        archive = self.directory / "dataset.zip"
        archive.write_bytes(b"<html>rate limited</html>")

        with self.assertRaises(RuntimeError) as caught:
            load_dataset._unpack_archive(archive, self.directory, self.directory)

        message = str(caught.exception)
        self.assertIn("not a recognised archive", message)
        # The body of the response is what identifies the real cause, so it
        # has to reach the message before the file is discarded.
        self.assertIn("rate limited", message)
        self.assertIn("25 bytes", message)

    def test_an_empty_file_is_described_as_empty(self):
        archive = self.directory / "dataset.zip"
        archive.touch()

        with self.assertRaises(RuntimeError) as caught:
            load_dataset._unpack_archive(archive, self.directory, self.directory)

        self.assertIn("empty", str(caught.exception))


class DownloadIntegrityTests(unittest.TestCase):
    """
    A truncated transfer must be rejected at download time; if it survives, the
    failure surfaces later as an extraction error that describes the wrong
    problem.
    """

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def test_complete_download_is_written(self):
        payload = b"x" * 4096
        response = FakeResponse(payload, {"content-length": str(len(payload))})
        destination = self.directory / "dataset.zip"

        with mock.patch.object(load_dataset.requests, "get", return_value=response):
            load_dataset._download_file("http://example.invalid", destination, "Test")

        self.assertEqual(destination.read_bytes(), payload)

    def test_truncated_download_raises_and_is_removed(self):
        payload = b"x" * 100
        # The server advertises more than it delivers, which is what a dropped
        # connection looks like from the client side.
        response = FakeResponse(payload, {"content-length": "5000"})
        destination = self.directory / "dataset.zip"

        with mock.patch.object(load_dataset.requests, "get", return_value=response):
            with self.assertRaises(RuntimeError) as caught:
                load_dataset._download_file("http://example.invalid", destination, "Test")

        self.assertIn("5000", str(caught.exception))
        self.assertFalse(destination.exists())

    def test_compressed_transfer_is_not_treated_as_truncated(self):
        payload = b"x" * 4096
        response = FakeResponse(
            payload,
            {"content-length": "120", "Content-Encoding": "gzip"},
        )
        destination = self.directory / "dataset.zip"

        with mock.patch.object(load_dataset.requests, "get", return_value=response):
            load_dataset._download_file("http://example.invalid", destination, "Test")

        self.assertEqual(destination.read_bytes(), payload)

    def test_missing_content_length_is_accepted(self):
        payload = b"x" * 512
        response = FakeResponse(payload, {})
        destination = self.directory / "dataset.zip"

        with mock.patch.object(load_dataset.requests, "get", return_value=response):
            load_dataset._download_file("http://example.invalid", destination, "Test")

        self.assertEqual(destination.read_bytes(), payload)

    def test_a_published_size_is_verified_against_the_transfer(self):
        # The repository's own metadata is a stronger check than the response
        # headers, which describe only what the server meant to send.
        payload = b"x" * 100
        response = FakeResponse(payload, {"content-length": "100"})
        destination = self.directory / "dataset.zip"

        with mock.patch.object(load_dataset.requests, "get", return_value=response):
            with self.assertRaises(RuntimeError) as caught:
                load_dataset._download_file(
                    "http://example.invalid", destination, "Test", expected_size=22601018
                )

        self.assertIn("22601018", str(caught.exception))
        self.assertFalse(destination.exists())

    def test_a_matching_published_size_is_accepted(self):
        payload = b"x" * 100
        response = FakeResponse(payload, {"content-length": "100"})
        destination = self.directory / "dataset.zip"

        with mock.patch.object(load_dataset.requests, "get", return_value=response):
            load_dataset._download_file(
                "http://example.invalid", destination, "Test", expected_size=100
            )

        self.assertEqual(destination.read_bytes(), payload)

    def test_an_empty_body_is_rejected_even_with_a_200_status(self):
        # A server can answer successfully and send nothing, which would
        # otherwise be stored as a zero-byte archive.
        response = FakeResponse(b"", {})
        destination = self.directory / "dataset.zip"

        with mock.patch.object(load_dataset.requests, "get", return_value=response):
            with self.assertRaises(RuntimeError) as caught:
                load_dataset._download_file("http://example.invalid", destination, "Test")

        self.assertIn("empty", str(caught.exception))
        self.assertFalse(destination.exists())

    def test_http_error_raises_and_leaves_no_partial_file(self):
        response = FakeResponse(b"", {}, status=404)
        destination = self.directory / "dataset.zip"

        with mock.patch.object(load_dataset.requests, "get", return_value=response):
            with self.assertRaises(RuntimeError):
                load_dataset._download_file("http://example.invalid", destination, "Test")

        self.assertFalse(destination.exists())


class DownloadIdentityTests(unittest.TestCase):
    """
    Repositories serve plain clients and browsers differently, and in opposite
    directions, so the order the two identities are tried in is load-bearing.
    """

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.destination = self.directory / "dataset.zip"

    def test_the_first_attempt_sends_no_browser_user_agent(self):
        # Presenting a browser identity to a repository that serves plain
        # clients directly returns an interactive response instead of the file.
        payload = b"x" * 64
        seen = []

        def capture(url, stream=False, headers=None, timeout=None):
            seen.append(headers)
            return FakeResponse(payload, {"content-length": str(len(payload))})

        with mock.patch.object(load_dataset.requests, "get", side_effect=capture):
            load_dataset._download_file("http://example.invalid", self.destination, "Test")

        self.assertEqual(len(seen), 1)
        self.assertNotIn("User-Agent", seen[0])

    def test_a_refusal_is_retried_with_a_browser_user_agent(self):
        payload = b"x" * 64
        seen = []

        def capture(url, stream=False, headers=None, timeout=None):
            seen.append(headers)
            if len(seen) == 1:
                return FakeResponse(b"", {}, status=403)
            return FakeResponse(payload, {"content-length": str(len(payload))})

        with mock.patch.object(load_dataset.requests, "get", side_effect=capture):
            load_dataset._download_file("http://example.invalid", self.destination, "Test")

        self.assertEqual(len(seen), 2)
        self.assertIn("Mozilla", seen[1]["User-Agent"])
        self.assertEqual(self.destination.read_bytes(), payload)

    def test_an_accepted_but_empty_response_is_not_a_download(self):
        # A 202 carries no file, yet raise_for_status treats it as success.
        seen = []

        def capture(url, stream=False, headers=None, timeout=None):
            seen.append(headers)
            return FakeResponse(b"", {"content-length": "0"}, status=202)

        with mock.patch.object(load_dataset.requests, "get", side_effect=capture):
            with self.assertRaises(RuntimeError) as caught:
                load_dataset._download_file("http://example.invalid", self.destination, "Test")

        self.assertIn("202", str(caught.exception))
        self.assertFalse(self.destination.exists())

    def test_both_failures_are_reported_together(self):
        def capture(url, stream=False, headers=None, timeout=None):
            return FakeResponse(b"", {}, status=404)

        with mock.patch.object(load_dataset.requests, "get", side_effect=capture):
            with self.assertRaises(RuntimeError) as caught:
                load_dataset._download_file("http://example.invalid", self.destination, "Test")

        # Both attempts matter when diagnosing, because a host that refuses
        # one identity and not the other narrows the cause immediately.
        self.assertIn("; then ", str(caught.exception))

    def test_a_partial_response_status_is_accepted(self):
        # Range-capable hosts answer 206 for a normal download.
        payload = b"x" * 64

        def capture(url, stream=False, headers=None, timeout=None):
            return FakeResponse(payload, {"content-length": str(len(payload))}, status=206)

        with mock.patch.object(load_dataset.requests, "get", side_effect=capture):
            load_dataset._download_file("http://example.invalid", self.destination, "Test")

        self.assertEqual(self.destination.read_bytes(), payload)


class TransientFailureTests(unittest.TestCase):
    """
    Several archives are hundreds of megabytes and served slowly, so a gateway
    dropping the transfer is routine and must not be reported as unavailable.
    """

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.destination = self.directory / "dataset.zip"

    def test_a_gateway_timeout_is_retried_with_the_same_identity(self):
        payload = b"x" * 64
        seen = []

        def capture(url, stream=False, headers=None, timeout=None):
            seen.append(headers)
            if len(seen) == 1:
                return FakeResponse(b"", {}, status=504, reason="Gateway Time-out")
            return FakeResponse(payload, {"content-length": str(len(payload))})

        with mock.patch.object(load_dataset.requests, "get", side_effect=capture):
            load_dataset._download_file("http://example.invalid", self.destination, "Test")

        self.assertEqual(len(seen), 2)
        # Switching identity here would waste the retry on a host that has
        # already shown it accepts this one.
        self.assertNotIn("User-Agent", seen[1])
        self.assertEqual(self.destination.read_bytes(), payload)

    def test_a_dropped_connection_is_retried(self):
        payload = b"x" * 64
        calls = []

        def capture(url, stream=False, headers=None, timeout=None):
            calls.append(headers)
            if len(calls) == 1:
                raise load_dataset.requests.ConnectionError("connection reset")
            return FakeResponse(payload, {"content-length": str(len(payload))})

        with mock.patch.object(load_dataset.requests, "get", side_effect=capture):
            load_dataset._download_file("http://example.invalid", self.destination, "Test")

        self.assertEqual(self.destination.read_bytes(), payload)

    def test_a_refusal_is_not_retried_unchanged(self):
        # A 403 will repeat identically, so the retry belongs to the other
        # identity rather than to the same request.
        seen = []

        def capture(url, stream=False, headers=None, timeout=None):
            seen.append(headers)
            return FakeResponse(b"", {}, status=403, reason="Forbidden")

        with mock.patch.object(load_dataset.requests, "get", side_effect=capture):
            with self.assertRaises(RuntimeError):
                load_dataset._download_file("http://example.invalid", self.destination, "Test")

        self.assertEqual(len(seen), 2)
        self.assertNotIn("User-Agent", seen[0])
        self.assertIn("Mozilla", seen[1]["User-Agent"])

    def test_transient_failures_are_bounded(self):
        seen = []

        def capture(url, stream=False, headers=None, timeout=None):
            seen.append(headers)
            return FakeResponse(b"", {}, status=503, reason="Service Unavailable")

        with mock.patch.object(load_dataset.requests, "get", side_effect=capture):
            with self.assertRaises(RuntimeError):
                load_dataset._download_file("http://example.invalid", self.destination, "Test")

        expected = load_dataset._TRANSIENT_ATTEMPTS * len(load_dataset._DOWNLOAD_IDENTITIES)
        self.assertEqual(len(seen), expected)


class DownloadAndExtractTests(unittest.TestCase):
    """
    End-to-end behaviour of the acquisition helper, including the states left
    behind when something goes wrong.
    """

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.destination = self.directory / "dataset_root"

    def test_wrapper_directory_is_flattened(self):
        archive = build_zip(
            self.directory / "dataset.zip",
            {"Wrapper/session1/a.txt": "a", "Wrapper/session2/b.txt": "b"},
        )

        load_dataset._download_and_extract(
            "http://example.invalid", archive, self.destination, "Test"
        )

        self.assertTrue((self.destination / "session1" / "a.txt").exists())
        self.assertTrue((self.destination / "session2" / "b.txt").exists())

    def test_a_lone_top_level_directory_is_always_treated_as_a_wrapper(self):
        # The heuristic cannot tell a packaging wrapper from a directory that
        # is itself meaningful, so an archive holding exactly one directory
        # always loses that level. Recorded here because it decides the layout
        # every loader then scans.
        archive = build_zip(self.directory / "dataset.zip", {"session1/a.txt": "a"})

        load_dataset._download_and_extract(
            "http://example.invalid", archive, self.destination, "Test"
        )

        self.assertTrue((self.destination / "a.txt").exists())
        self.assertFalse((self.destination / "session1").exists())

    def test_multiple_top_level_entries_are_preserved(self):
        archive = build_zip(
            self.directory / "dataset.zip",
            {"session1/a.txt": "a", "session2/b.txt": "b"},
        )

        load_dataset._download_and_extract(
            "http://example.invalid", archive, self.destination, "Test"
        )

        self.assertTrue((self.destination / "session1" / "a.txt").exists())
        self.assertTrue((self.destination / "session2" / "b.txt").exists())

    def test_populated_destination_is_not_downloaded_again(self):
        self.destination.mkdir(parents=True)
        (self.destination / "existing.txt").write_text("kept")

        with mock.patch.object(load_dataset, "_download_file") as download:
            load_dataset._download_and_extract(
                "http://example.invalid",
                self.directory / "absent.zip",
                self.destination,
                "Test",
            )

        download.assert_not_called()
        self.assertEqual((self.destination / "existing.txt").read_text(), "kept")

    def test_interrupted_extraction_is_discarded_and_retried(self):
        archive = build_zip(
            self.directory / "dataset.zip",
            {"session1/a.txt": "a", "session2/b.txt": "b"},
        )

        # Reproduce the state a run killed part-way through leaves behind: some
        # files in place, and the marker still present.
        self.destination.mkdir(parents=True)
        (self.destination / "half_moved.txt").write_text("stale")
        load_dataset._extraction_marker(self.destination).touch()

        load_dataset._download_and_extract(
            "http://example.invalid", archive, self.destination, "Test"
        )

        self.assertFalse((self.destination / "half_moved.txt").exists())
        self.assertTrue((self.destination / "session1" / "a.txt").exists())
        self.assertFalse(load_dataset._extraction_marker(self.destination).exists())

    def test_marker_is_cleared_after_a_successful_extraction(self):
        archive = build_zip(self.directory / "dataset.zip", {"a.txt": "a"})

        load_dataset._download_and_extract(
            "http://example.invalid", archive, self.destination, "Test"
        )

        self.assertFalse(load_dataset._extraction_marker(self.destination).exists())

    def test_marker_lives_outside_the_dataset_directory(self):
        # The loaders walk the dataset directory recursively, so a sentinel
        # inside it would be picked up as data.
        marker = load_dataset._extraction_marker(self.destination)
        self.assertEqual(marker.parent, self.destination.parent)

    def test_corrupt_archive_is_deleted_to_force_a_fresh_download(self):
        archive = self.directory / "dataset.zip"
        archive.write_bytes(b"<html>truncated</html>")

        with self.assertRaises(RuntimeError):
            load_dataset._download_and_extract(
                "http://example.invalid", archive, self.destination, "Test"
            )

        self.assertFalse(archive.exists())

    def test_archive_is_kept_when_only_the_tool_is_missing(self):
        # The archive is intact and the problem is a local one, so discarding
        # it would force an avoidable second transfer once the tool is present.
        archive = self.directory / "dataset.rar"
        archive.write_bytes(RAR5_MAGIC + b"payload")

        with mock.patch.object(load_dataset.shutil, "which", return_value=None):
            with self.assertRaises(load_dataset.MissingArchiveToolError):
                load_dataset._download_and_extract(
                    "http://example.invalid", archive, self.destination, "Test"
                )

        self.assertTrue(archive.exists())

    def test_download_failure_propagates_rather_than_returning(self):
        # Returning quietly leaves the caller to fail later while parsing a
        # directory that was never populated.
        response = FakeResponse(b"", {}, status=503)

        with mock.patch.object(load_dataset.requests, "get", return_value=response):
            with self.assertRaises(RuntimeError):
                load_dataset._download_and_extract(
                    "http://example.invalid",
                    self.directory / "absent.zip",
                    self.destination,
                    "Test",
                )

    def test_cleanup_removes_the_archive_after_success(self):
        archive = build_zip(self.directory / "dataset.zip", {"a.txt": "a"})

        load_dataset._download_and_extract(
            "http://example.invalid", archive, self.destination, "Test", cleanup=True
        )

        self.assertFalse(archive.exists())
        self.assertTrue((self.destination / "a.txt").exists())

    def test_archive_saved_under_the_wrong_suffix_still_extracts(self):
        # The HeartPrint case: the configured name says .zip, the payload is
        # not a zip. Detection has to win over the file name.
        real = build_zip(
            self.directory / "real.zip",
            {"session1/a.txt": "a", "session2/b.txt": "b"},
        )
        misnamed = self.directory / "dataset.rar"
        misnamed.write_bytes(real.read_bytes())
        real.unlink()

        load_dataset._download_and_extract(
            "http://example.invalid", misnamed, self.destination, "Test"
        )

        self.assertTrue((self.destination / "session1" / "a.txt").exists())


class HeartprintArchiveResolutionTests(unittest.TestCase):
    """
    The HeartPrint record bundles several files, only one of which is the
    archive, and its published format is not the one the configuration names.
    """

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def _figshare_payload(self):
        return {
            "files": [
                {"name": "Heartprint Metadata.xlsx", "download_url": "http://x/1", "size": 1},
                {"name": "20120509_0457_02_ECG.txt", "download_url": "http://x/2", "size": 2},
                {"name": "Heartprint.rar", "download_url": "http://x/3", "size": 22601018},
            ]
        }

    def test_archive_is_selected_over_the_loose_files(self):
        loader = load_dataset.load_heartprint_dataset()
        loader.dataset_root = self.directory / "heartprint"
        loader.zip_path = self.directory / "Heartprint_Dataset.zip"

        api_response = mock.Mock()
        api_response.json.return_value = self._figshare_payload()
        api_response.raise_for_status.return_value = None

        captured = {}

        def record_call(url, zip_path, extract_to, name, cleanup=False, expected_size=None):
            captured["url"] = url
            captured["zip_path"] = zip_path
            captured["expected_size"] = expected_size

        with mock.patch.object(load_dataset.requests, "get", return_value=api_response), \
             mock.patch.object(load_dataset, "_download_and_extract", side_effect=record_call):
            loader.download()

        self.assertEqual(captured["url"], "http://x/3")
        # The saved name has to carry the real format so extraction dispatches
        # correctly rather than handing a RAR to the zip reader.
        self.assertEqual(captured["zip_path"].suffix, ".rar")
        # The published size travels with it, so a short transfer is caught at
        # download time rather than as an extraction failure.
        self.assertEqual(captured["expected_size"], 22601018)

    def test_an_already_downloaded_archive_keeps_its_name(self):
        loader = load_dataset.load_heartprint_dataset()
        loader.dataset_root = self.directory / "heartprint"
        loader.zip_path = self.directory / "Heartprint_Dataset.zip"
        loader.zip_path.write_bytes(RAR5_MAGIC + b"payload")

        api_response = mock.Mock()
        api_response.json.return_value = self._figshare_payload()
        api_response.raise_for_status.return_value = None

        captured = {}

        def record_call(url, zip_path, extract_to, name, cleanup=False, expected_size=None):
            captured["zip_path"] = zip_path

        with mock.patch.object(load_dataset.requests, "get", return_value=api_response), \
             mock.patch.object(load_dataset, "_download_and_extract", side_effect=record_call):
            loader.download()

        self.assertEqual(captured["zip_path"].name, "Heartprint_Dataset.zip")

    def test_api_failure_falls_back_to_the_record_url(self):
        loader = load_dataset.load_heartprint_dataset()
        loader.dataset_root = self.directory / "heartprint"
        loader.zip_path = self.directory / "Heartprint_Dataset.zip"

        captured = {}

        def record_call(url, zip_path, extract_to, name, cleanup=False, expected_size=None):
            captured["url"] = url

        with mock.patch.object(load_dataset.requests, "get", side_effect=IOError("no route")), \
             mock.patch.object(load_dataset, "_download_and_extract", side_effect=record_call):
            loader.download()

        self.assertEqual(captured["url"], loader.url)

    def test_an_extracted_dataset_is_not_fetched_again(self):
        loader = load_dataset.load_heartprint_dataset()
        loader.dataset_root = self.directory / "heartprint"
        (loader.dataset_root / "Session1").mkdir(parents=True)

        with mock.patch.object(load_dataset, "_download_and_extract") as acquire:
            loader.download()

        acquire.assert_not_called()


if __name__ == "__main__":
    unittest.main()
