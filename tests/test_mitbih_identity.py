"""
Pin how MIT-BIH maps its 48 records onto the people who produced them.

Two properties of the distribution decide the identity count. The database
documentation states that records 201 and 202 came from one man, and the
x_mitdb directory holds copies of the first ten minutes of records that are
already listed in the canonical index. Enumerating the directory tree instead
of the shipped RECORDS file reports 69 identities where there are 45.
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

from load_dataset import load_mitbih_dataset


FS = 360


def write_record(directory, name, minutes=1, leads=("MLII", "V1"), offset=0.0):
    """
    Write one synthetic record and add it to the RECORDS manifest.

    A constant offset per record makes it possible to tell which recording a
    span of samples came from.
    """
    samples = int(minutes * 60 * FS)
    ramp = np.linspace(-0.5, 0.5, samples, dtype=np.float64) + offset
    signal = np.stack([ramp, -ramp], axis=1)

    wfdb.wrsamp(
        record_name=name,
        fs=FS,
        units=["mV", "mV"],
        sig_name=list(leads),
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


class MITBIHEnumerationTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

        for name in ("100", "101", "201", "202"):
            write_record(self.directory, name)

        # A derived copy, in its own directory as the database ships it. It is
        # deliberately left out of the canonical manifest.
        derived = self.directory / "x_mitdb"
        derived.mkdir()
        write_record(derived, "x_100")

        self.loader = load_mitbih_dataset(data_split_mode="all-available")
        self.loader.dataset_root = self.directory

    def test_the_derived_directory_is_not_enumerated(self):
        recordings = self.loader.load_raw_data()

        self.assertFalse(
            [name for name in recordings if name.startswith("x_")],
        )

    def test_records_are_read_in_manifest_order(self):
        """
        A filesystem scan returns entries in whatever order the filesystem
        chooses, and the downstream split selects samples by position.
        """
        self.assertEqual(
            self.loader._record_names(),
            ["100", "101", "201", "202"],
        )

    def test_a_missing_manifest_is_reported(self):
        (self.directory / "RECORDS").unlink()

        with self.assertRaisesRegex(
            FileNotFoundError,
            "record list not found",
        ):
            self.loader.load_raw_data()


class MITBIHSharedSubjectTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

        write_record(self.directory, "100", offset=0.0)
        write_record(self.directory, "201", offset=0.1)
        write_record(self.directory, "202", offset=0.2)

        self.loader = load_mitbih_dataset(data_split_mode="all-available")
        self.loader.dataset_root = self.directory

    def test_the_two_records_of_one_man_become_one_identity(self):
        recordings = self.loader.load_raw_data()

        self.assertIn("201", recordings)
        self.assertNotIn("202", recordings)
        self.assertEqual(len(recordings), 2)

    def test_the_merged_subject_keeps_both_recordings(self):
        recordings = self.loader.load_raw_data()

        single = recordings["100"]["signal"]
        merged = recordings["201"]["signal"]

        self.assertEqual(
            merged.shape[0],
            single.shape[0] * 2,
        )
        self.assertIn("201", recordings["201"]["filename"])
        self.assertIn("202", recordings["201"]["filename"])

    def test_the_merge_appends_rather_than_replaces(self):
        """
        The second recording must extend the first, not overwrite it, so both
        offsets have to survive.
        """
        recordings = self.loader.load_raw_data()
        merged = recordings["201"]["signal"][:, 0]

        half = len(merged) // 2
        self.assertLess(
            float(np.mean(merged[:half])),
            float(np.mean(merged[half:])),
        )

    def test_a_requested_range_is_taken_from_both_recordings(self):
        recordings = self.loader.load_raw_data(min_ranges=[(0, 0.5)])

        single = recordings["100"]["signal"]
        merged = recordings["201"]["signal"]

        self.assertEqual(merged.shape[0], single.shape[0] * 2)


class MITBIHLeadSelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

        write_record(self.directory, "100", leads=("MLII", "V1"))
        write_record(self.directory, "114", leads=("V5", "MLII"))
        write_record(self.directory, "102", leads=("V5", "V2"))

        self.loader = load_mitbih_dataset(data_split_mode="all-available")
        self.loader.dataset_root = self.directory

    def test_a_record_without_the_target_lead_is_dropped(self):
        """
        Reading a different lead for a few subjects would give them a signal
        that differs for a reason unrelated to who they are.
        """
        recordings = self.loader.load_raw_data()

        self.assertNotIn("102", recordings)

    def test_reversed_channels_are_resolved_by_name(self):
        recordings = self.loader.load_raw_data()

        self.assertIn("114", recordings)


if __name__ == "__main__":
    unittest.main()
