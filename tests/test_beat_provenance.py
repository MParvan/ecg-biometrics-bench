"""
Per-beat source provenance: representation, loader emission and cache round-trip.

These tests pin the provenance *infrastructure* only. They do not exercise the
score-fusion or enrollment ordering that later consumes provenance; that
behaviour is covered separately.
"""

import datetime
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from load_dataset import (
    BeatProvenance,
    PROVENANCE_COLUMNS,
    _ProvenanceBuilder,
    load_ecgid_dataset,
    load_mitbih_dataset,
    load_nsrdb_dataset,
)
from utils import CacheManager, _generate_config_hash


def _fixed_segments(count, dim=4):
    return np.ones((count, dim), dtype=float)


def _stub_ecgid(recs):
    loader = load_ecgid_dataset(data_split_mode="all-available")
    loader.load_raw_data = lambda *a, **k: {"S1": recs}
    loader._process_signal = lambda sig, fs: _fixed_segments(3)
    return loader


class PublicApiCompatibility(unittest.TestCase):
    def test_default_call_returns_x_and_y_only(self):
        loader = _stub_ecgid(
            [{"signal": np.zeros(5), "fs": 500, "date": None, "filename": "a.hea"}]
        )
        result = loader.load_all_data()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        x, y = result
        self.assertEqual(len(x), len(y))


class OptInProvenance(unittest.TestCase):
    def test_return_provenance_gives_aligned_bundle(self):
        loader = _stub_ecgid(
            [
                {"signal": np.zeros(5), "fs": 500,
                 "date": datetime.date(2001, 2, 3), "filename": "a.hea"},
                {"signal": np.zeros(5), "fs": 500,
                 "date": None, "filename": "b.hea"},
            ]
        )
        x0, y0 = loader.load_all_data()
        x, y, provenance = loader.load_all_data(return_provenance=True)
        # X/y identical with and without provenance requested.
        self.assertTrue(np.array_equal(x0, x))
        self.assertTrue(np.array_equal(y0, y))
        self.assertIsInstance(provenance, BeatProvenance)
        self.assertEqual(
            provenance.columns["record_id"].tolist(),
            ["a.hea", "a.hea", "a.hea", "b.hea", "b.hea", "b.hea"],
        )


class Alignment(unittest.TestCase):
    def test_every_column_matches_x_and_y_length(self):
        loader = _stub_ecgid(
            [{"signal": np.zeros(5), "fs": 500, "date": None, "filename": "a.hea"},
             {"signal": np.zeros(5), "fs": 500, "date": None, "filename": "b.hea"}]
        )
        x, y, provenance = loader.load_all_data(return_provenance=True)
        provenance.validate(len(y))
        self.assertEqual(len(x), len(y))
        for name in PROVENANCE_COLUMNS:
            self.assertEqual(len(provenance.columns[name]), len(y), name)


class GenuineDateHandling(unittest.TestCase):
    def test_missing_dates_stay_missing_and_present_dates_survive(self):
        loader = _stub_ecgid(
            [{"signal": np.zeros(5), "fs": 500,
              "date": datetime.date(1999, 12, 31), "filename": "dated.hea"},
             {"signal": np.zeros(5), "fs": 500,
              "date": None, "filename": "undated.hea"}]
        )
        _, _, provenance = loader.load_all_data(return_provenance=True)
        times = provenance.columns["acquisition_time"]
        self.assertEqual(times[0], datetime.date(1999, 12, 31))
        self.assertIsNone(times[3])  # undated record's first beat


class StableOrderingMetadata(unittest.TestCase):
    def _nsrdb(self, recs):
        loader = load_nsrdb_dataset(data_split_mode="all-available")
        loader.load_raw_data_slices = lambda *a, **k: {"N1": recs}
        loader._process_signal = lambda sig, fs: _fixed_segments(2)
        return loader

    def test_segment_order_is_numeric_and_independent_of_list_order(self):
        early = {"signal": np.zeros(5), "fs": 128,
                 "filename": "rec.dat", "start_min": 0}
        late = {"signal": np.zeros(5), "fs": 128,
                "filename": "rec.dat", "start_min": 1380}
        # Present the two ranges in reversed list order.
        _, _, provenance = self._nsrdb([late, early]).load_all_data(
            return_provenance=True
        )
        # Sorting by the numeric source position recovers the true range order,
        # so downstream ordering does not depend on enumeration order.
        order = np.argsort(provenance.columns["source_segment_order"], kind="stable")
        recovered = provenance.columns["source_segment_order"][order]
        self.assertTrue(np.all(recovered[:-1] <= recovered[1:]))
        self.assertEqual(set(provenance.columns["source_segment_order"].tolist()),
                         {0.0, 1380.0})


class ContinuousRangesGetDistinctSegments(unittest.TestCase):
    def _nsrdb(self, recs):
        loader = load_nsrdb_dataset(data_split_mode="all-available")
        loader.load_raw_data_slices = lambda *a, **k: {"N1": recs}
        loader._process_signal = lambda sig, fs: _fixed_segments(2)
        return loader

    def test_two_ranges_of_one_recording_get_different_segment_ids(self):
        recs = [
            {"signal": np.zeros(5), "fs": 128, "filename": "rec.dat", "start_min": 0},
            {"signal": np.zeros(5), "fs": 128, "filename": "rec.dat", "start_min": 1380},
        ]
        _, _, provenance = self._nsrdb(recs).load_all_data(return_provenance=True)
        seg_ids = set(provenance.columns["source_segment_id"].tolist())
        self.assertEqual(len(seg_ids), 2, seg_ids)
        # Same physical recording, so record_id is shared.
        self.assertEqual(set(provenance.columns["record_id"].tolist()), {"rec.dat"})


class CacheRoundTrip(unittest.TestCase):
    def _provenance(self, n):
        builder = _ProvenanceBuilder()
        builder.add_block(
            n,
            record_id="r.hea",
            session_id="r.hea",
            acquisition_time=None,
            acquisition_order=0,
            source_segment_id="r.hea#0",
            source_segment_order=0.0,
        )
        return builder.build()

    def test_provenance_survives_the_data_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = CacheManager(base_dir=tmp)
            conf = {"k": "v"}
            x = np.ones((4, 3)); y = np.arange(4)
            prov = self._provenance(4)
            arrays = {"x": x, "y": y}
            arrays.update(prov.to_cache_dict())
            cache.save_data_cache(arrays, conf, _generate_config_hash(conf))
            cached, _ = cache.get_data_cache(conf)
            restored = BeatProvenance.from_cache_dict(
                cached, expected_length=len(cached["y"])
            )
            self.assertIsNotNone(restored)
            self.assertEqual(
                restored.columns["source_segment_id"].tolist(),
                ["r.hea#0"] * 4,
            )

    def test_cache_without_provenance_is_treated_as_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = CacheManager(base_dir=tmp)
            conf = {"k": "v"}
            cache.save_data_cache(
                {"x": np.ones((2, 3)), "y": np.arange(2)}, conf, _generate_config_hash(conf)
            )
            cached, _ = cache.get_data_cache(conf)
            self.assertIsNotNone(cached)  # x/y are there
            restored = BeatProvenance.from_cache_dict(
                cached, expected_length=len(cached["y"])
            )
            self.assertIsNone(restored)  # provenance absent -> rebuild

    def test_length_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = CacheManager(base_dir=tmp)
            conf = {"k": "v"}
            x = np.ones((4, 3)); y = np.arange(4)
            arrays = {"x": x, "y": y}
            arrays.update(self._provenance(4).to_cache_dict())
            cache.save_data_cache(arrays, conf, _generate_config_hash(conf))
            cached, _ = cache.get_data_cache(conf)
            # Ask for a length that does not match the stored provenance.
            self.assertIsNone(
                BeatProvenance.from_cache_dict(cached, expected_length=3)
            )


class CrossSessionBundles(unittest.TestCase):
    def _prov(self, n, tag):
        builder = _ProvenanceBuilder()
        builder.add_block(
            n, record_id=tag, session_id=tag, acquisition_time=None,
            acquisition_order=0, source_segment_id=tag, source_segment_order=0.0,
        )
        return builder.build()

    def test_each_session_bundle_is_validated_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = CacheManager(base_dir=tmp)
            conf = {"k": "v"}
            arrays = {
                "x_s1": np.ones((2, 3)), "y_s1": np.arange(2),
                "x_s2": np.ones((5, 3)), "y_s2": np.arange(5),
            }
            arrays.update(self._prov(2, "s1").to_cache_dict(prefix="provenance_s1__"))
            arrays.update(self._prov(5, "s2").to_cache_dict(prefix="provenance_s2__"))
            cache.save_data_cache(arrays, conf, _generate_config_hash(conf))
            cached, _ = cache.get_data_cache(conf)
            p1 = BeatProvenance.from_cache_dict(
                cached, prefix="provenance_s1__", expected_length=len(cached["y_s1"])
            )
            p2 = BeatProvenance.from_cache_dict(
                cached, prefix="provenance_s2__", expected_length=len(cached["y_s2"])
            )
            self.assertIsNotNone(p1)
            self.assertIsNotNone(p2)
            self.assertEqual(len(p1), 2)
            self.assertEqual(len(p2), 5)
            # A bundle validated against the wrong session length is rejected.
            self.assertIsNone(
                BeatProvenance.from_cache_dict(
                    cached, prefix="provenance_s1__",
                    expected_length=len(cached["y_s2"]),
                )
            )


class MitbihSegmentProvenance(unittest.TestCase):
    """MIT-BIH is a continuous-record dataset: each configured range and each
    physical record is an independent source segment."""

    def _segment(self, record_id, start_min, acquisition_order, n=6):
        return {
            "signal": np.zeros((n, 1)),
            "fs": 360,
            "filename": record_id,
            "record_id": record_id,
            "start_min": start_min,
            "acquisition_order": acquisition_order,
        }

    def _loader(self, recs_by_sid):
        loader = load_mitbih_dataset(data_split_mode="all-available")
        loader.load_raw_data = lambda *a, **k: recs_by_sid
        loader._process_signal = lambda sig, fs: _fixed_segments(2)
        return loader

    def test_two_ranges_of_one_record_are_distinct_segments(self):
        recs = {
            "100": [
                self._segment("100.hea", 0.0, 0),
                self._segment("100.hea", 12.5, 0),
            ]
        }
        _, _, provenance = self._loader(recs).load_all_data(return_provenance=True)
        # D: distinct segment ids; C: every beat belongs to one of the two.
        self.assertEqual(
            set(provenance.columns["source_segment_id"].tolist()),
            {"100.hea#0.0", "100.hea#12.5"},
        )
        # Same physical recording.
        self.assertEqual(set(provenance.columns["record_id"].tolist()), {"100.hea"})
        # E: source order follows the genuine numeric range start.
        self.assertEqual(
            set(provenance.columns["source_segment_order"].tolist()), {0.0, 12.5}
        )
        # F: beat_ordinal resets within each segment.
        self.assertEqual(provenance.columns["beat_ordinal"].tolist(), [0, 1, 0, 1])

    def test_shared_subject_records_stay_distinct(self):
        # G/H: records 201 and 202 are one identity but distinct records.
        recs = {
            "201": [
                self._segment("201.hea", 0.0, 0),
                self._segment("202.hea", 0.0, 1),
            ]
        }
        _, y, provenance = self._loader(recs).load_all_data(return_provenance=True)
        self.assertEqual(
            set(provenance.columns["record_id"].tolist()), {"201.hea", "202.hea"}
        )
        self.assertEqual(set(y.tolist()), {"201"})  # same subject label
        self.assertEqual(
            set(provenance.columns["acquisition_order"].tolist()), {0, 1}
        )
        # Two records at the same start still form two distinct segments.
        self.assertEqual(
            set(provenance.columns["source_segment_id"].tolist()),
            {"201.hea#0.0", "202.hea#0.0"},
        )


if __name__ == "__main__":
    unittest.main()
