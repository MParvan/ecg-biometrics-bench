import datetime
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import load_dataset


def write_synthetic_ecgid_tree(root, subjects=2, day_offsets=(1, 1, 20)):
    """
    Write an ECG-ID-shaped WFDB directory with dated two-channel records.
    """
    sampling_rate = 500
    waveform = np.stack(
        [
            np.sin(
                np.linspace(
                    0,
                    40 * np.pi,
                    sampling_rate * 4,
                )
            ),
            np.sin(
                np.linspace(
                    0,
                    40 * np.pi,
                    sampling_rate * 4,
                )
            ),
        ],
        axis=1,
    ).astype(np.float64)

    for person_index in range(1, subjects + 1):
        person_directory = root / f"Person_{person_index:02d}"
        person_directory.mkdir(parents=True)

        for record_index, day in enumerate(
            day_offsets,
            start=1,
        ):
            wfdb.wrsamp(
                record_name=f"rec_{record_index}",
                fs=sampling_rate,
                units=["mV", "mV"],
                sig_name=[
                    "ECG I raw",
                    "ECG I filtered",
                ],
                p_signal=waveform,
                write_dir=str(person_directory),
                base_date=datetime.date(2024, 1, day),
                base_time=datetime.time(9, 0, 0),
            )


class MetadataOnlyLoadingTests(unittest.TestCase):
    """
    The metadata-only pass must describe exactly the records a full load uses.
    """

    @classmethod
    def setUpClass(cls):
        cls.root = Path(
            tempfile.mkdtemp(
                prefix="ecgid_fixture_",
            )
        )
        write_synthetic_ecgid_tree(cls.root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(
            cls.root,
            ignore_errors=True,
        )

    def build_loader(self, data_split_mode):
        loader = load_dataset.load_ecgid_dataset(
            data_split_mode=data_split_mode,
        )
        loader.dataset_root = self.root

        return loader

    def test_metadata_matches_full_load_ordering(self):
        loader = self.build_loader(
            "leave-last-out-long-term"
        )

        metadata_records = loader.load_raw_data(
            metadata_only=True,
        )
        full_records = loader.load_raw_data()

        self.assertEqual(
            sorted(metadata_records),
            sorted(full_records),
        )

        for subject_id in metadata_records:
            self.assertEqual(
                [
                    (record["date"], record["filename"])
                    for record in metadata_records[subject_id]
                ],
                [
                    (record["date"], record["filename"])
                    for record in full_records[subject_id]
                ],
            )

    def test_metadata_pass_omits_signal_payloads(self):
        loader = self.build_loader(
            "leave-last-out-long-term"
        )

        metadata_records = loader.load_raw_data(
            metadata_only=True,
        )

        for records in metadata_records.values():
            for record in records:
                self.assertNotIn("signal", record)

    def test_full_pass_still_returns_signals(self):
        loader = self.build_loader(
            "leave-last-out-long-term"
        )

        full_records = loader.load_raw_data()

        for records in full_records.values():
            for record in records:
                self.assertIn("signal", record)
                self.assertEqual(
                    record["signal"].ndim,
                    1,
                )

    def test_audit_reports_the_expected_assignment(self):
        loader = self.build_loader(
            "leave-last-out-long-term"
        )

        report = load_dataset.audit_record_order_causality(
            loader.load_raw_data(metadata_only=True),
            "leave-last-out-long-term",
        )

        self.assertTrue(
            report["enrollment_precedes_probe"]
        )
        self.assertEqual(
            report["subjects_audited"],
            2,
        )

        for subject_report in report["subject_reports"]:
            self.assertEqual(
                subject_report["enrollment_records"],
                ["rec_1.hea", "rec_2.hea"],
            )
            self.assertEqual(
                subject_report["probe_records"],
                ["rec_3.hea"],
            )

    def test_every_record_order_regime_is_causal_on_the_fixture(self):
        for mode in load_dataset.RECORD_ORDER_SPLIT_MODES:
            with self.subTest(mode=mode):
                loader = self.build_loader(mode)

                report = load_dataset.audit_record_order_causality(
                    loader.load_raw_data(
                        metadata_only=True,
                    ),
                    mode,
                )

                self.assertTrue(
                    report["enrollment_precedes_probe"],
                    f"{mode} reported {report['violations']}",
                )


if __name__ == "__main__":
    unittest.main()
