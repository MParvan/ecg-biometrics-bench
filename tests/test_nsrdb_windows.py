"""
Pin NSRDB enumeration and the reach of its probe windows.

Every subject contributes one continuous recording, so a minute window that
runs past the end of a recording silently removes that subject from whichever
partition asked for it. The recordings differ in length by more than two and a
half hours, which makes the latest usable window a property of the shortest
recording rather than of the protocol.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from load_dataset import load_nsrdb_dataset


FS = 128


def write_record(directory, name, minutes):
    samples = int(minutes * 60 * FS)
    ramp = np.linspace(-0.5, 0.5, samples, dtype=np.float64)
    signal = np.stack([ramp, -ramp], axis=1)

    wfdb.wrsamp(
        record_name=name,
        fs=FS,
        units=["mV", "mV"],
        sig_name=["ECG1", "ECG2"],
        p_signal=signal,
        write_dir=str(directory),
    )

    manifest = directory / "RECORDS"
    existing = (
        [line for line in manifest.read_text().splitlines() if line.strip()]
        if manifest.exists()
        else []
    )
    manifest.write_text("\n".join(existing + [name]) + "\n")


class NSRDBWindowReachTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

        # A long recording and a short one, as the database has.
        write_record(self.directory, "16265", minutes=25)
        write_record(self.directory, "17052", minutes=20)

        self.loader = load_nsrdb_dataset(data_split_mode="all-available")
        self.loader.dataset_root = self.directory

    def test_a_window_inside_every_recording_reaches_every_subject(self):
        recordings = self.loader.load_raw_data_slices(min_ranges=[(10, 15)])

        self.assertEqual(len(recordings), 2)

    def test_a_window_past_the_shortest_recording_drops_that_subject(self):
        """
        The subject is not reported as missing anywhere; it simply does not
        appear, which is how six subjects went unprobed.
        """
        recordings = self.loader.load_raw_data_slices(min_ranges=[(22, 24)])

        self.assertIn("16265", recordings)
        self.assertNotIn("17052", recordings)

    def test_enrolment_and_probe_can_disagree_on_the_cohort(self):
        enrolled = self.loader.load_raw_data_slices(min_ranges=[(0, 5)])
        probed = self.loader.load_raw_data_slices(min_ranges=[(22, 24)])

        self.assertTrue(set(enrolled) - set(probed))


class NSRDBEnumerationTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

        for name in ("19830", "16265", "17052"):
            write_record(self.directory, name, minutes=12)

        self.loader = load_nsrdb_dataset(data_split_mode="all-available")
        self.loader.dataset_root = self.directory

    def test_records_are_read_in_manifest_order(self):
        """
        The downstream split selects samples by position, so the order the
        records arrive in has to come from the manifest rather than the
        filesystem.
        """
        self.assertEqual(
            self.loader._record_names(),
            ["19830", "16265", "17052"],
        )

    def test_a_missing_manifest_is_reported(self):
        (self.directory / "RECORDS").unlink()

        with self.assertRaisesRegex(
            FileNotFoundError,
            "record list not found",
        ):
            self.loader.load_raw_data_slices(min_ranges=[(0, 5)])


class NSRDBConfiguredWindowTests(unittest.TestCase):
    def test_the_shipped_probe_window_ends_before_the_shortest_recording(self):
        """
        The shortest recording runs to 1388 minutes. A probe window ending
        after that silently excludes its subject.
        """
        import yaml

        shortest_minutes = 1388.0
        windows = []

        for path in sorted(
            (PROJECT_ROOT / "configs").rglob("nsrdb*.yaml")
        ):
            configuration = yaml.safe_load(path.read_text())
            for key in ("train_parts", "enrol_parts", "test_parts"):
                value = configuration.get(key)
                if not value:
                    continue
                for start, end in value:
                    windows.append((path.name, key, start, end))

        self.assertTrue(windows)

        overrunning = [
            entry for entry in windows if entry[3] > shortest_minutes
        ]

        self.assertFalse(
            overrunning,
            f"windows extending past the shortest recording: {overrunning}",
        )


if __name__ == "__main__":
    unittest.main()
