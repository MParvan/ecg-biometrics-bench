"""Regression tests for the validation-array weight-cache fingerprint (Issue #5).

Validation can change the selected/final weights (best-checkpoint selection,
learning-rate rollback, early stopping). When validation is active its arrays
therefore participate in the weight-cache identity, under ``validation_partition``.
When validation is inactive the field is omitted entirely, so existing cache
identities are preserved byte-for-byte (no null field, no version/schema key).

These cover the collector (``_collect_validation_arrays``) and the fingerprint
behaviour of ``_build_weight_cache_config``; the ``split_seed`` routing, metadata
and CLI/config behaviour live in ``test_split_seed.py``.
"""

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run  # noqa: E402


class ValidationCacheFingerprintTests(unittest.TestCase):
    """Validation arrays enter the weight-cache identity only when active."""

    def setUp(self):
        # Isolate the validation-fingerprint logic from loader identity.
        self._original = run._build_loader_cache_identity
        run._build_loader_cache_identity = lambda loader: {"loader": "stub"}
        self._x = np.arange(40, dtype=np.float32).reshape(10, 4)
        self._y = np.arange(10)
        self._base = {"model": "m", "epochs": 1}

    def tearDown(self):
        run._build_loader_cache_identity = self._original

    def _build(self, validation_arrays="__absent__"):
        kwargs = dict(
            training_samples=self._x, training_labels=self._y
        )
        if validation_arrays != "__absent__":
            kwargs["validation_arrays"] = validation_arrays
        return run._build_weight_cache_config("L", dict(self._base), **kwargs)

    def test_no_field_when_validation_absent_or_none(self):
        absent = self._build()
        none = self._build(None)
        all_none = self._build({"X_val": None, "y_val": None})
        self.assertNotIn("validation_partition", absent)
        # Identity is preserved (byte-for-byte) across all inactive forms.
        self.assertEqual(absent, none)
        self.assertEqual(absent, all_none)

    def test_field_present_and_sensitive_when_active(self):
        yv = np.arange(5)
        a = np.ones((5, 4), dtype=np.float32)
        b = np.zeros((5, 4), dtype=np.float32)
        cfg_a = self._build({"X_val": a, "y_val": yv})
        cfg_b = self._build({"X_val": b, "y_val": yv})
        cfg_a2 = self._build({"X_val": a, "y_val": yv})
        self.assertIn("validation_partition", cfg_a)
        # Identical training arrays but different active validation arrays give
        # different identities; identical validation reproduces the identity.
        self.assertNotEqual(
            cfg_a["validation_partition"], cfg_b["validation_partition"]
        )
        self.assertEqual(cfg_a, cfg_a2)

    def test_active_identity_differs_from_inactive(self):
        yv = np.arange(5)
        a = np.ones((5, 4), dtype=np.float32)
        self.assertNotEqual(self._build(), self._build({"X_val": a, "y_val": yv}))


class ValidationCollectorTests(unittest.TestCase):
    def test_inactive_returns_none(self):
        arr = np.ones((3, 2))
        self.assertIsNone(run._collect_validation_arrays(0.0, {"X_val": arr}))
        self.assertIsNone(run._collect_validation_arrays(None, {"X_val": arr}))

    def test_active_collects_present_nonnull_arrays(self):
        arr = np.ones((3, 2))
        yv = np.arange(3)
        collected = run._collect_validation_arrays(
            0.2, {"X_val": arr, "y_val": yv, "unrelated": 5}
        )
        self.assertEqual(set(collected), {"X_val", "y_val"})

    def test_active_but_all_none_returns_none(self):
        self.assertIsNone(
            run._collect_validation_arrays(0.2, {"X_val": None, "y_val": None})
        )

    def test_collects_subject_disjoint_and_session_validation_names(self):
        arr = np.ones((3, 2))
        collected = run._collect_validation_arrays(
            0.2,
            {
                "X_val_seen": arr, "y_val_seen": arr,
                "X_val": arr, "y_val": arr,
                "X_val_s1": arr, "y_val_s1_enc": arr,
            },
        )
        self.assertEqual(
            set(collected),
            {"X_val_seen", "y_val_seen", "X_val", "y_val",
             "X_val_s1", "y_val_s1_enc"},
        )


class ValidationCacheNoVersionFieldTests(unittest.TestCase):
    def test_no_cache_version_or_schema_field(self):
        source = inspect.getsource(run._build_weight_cache_config)
        for forbidden in ("cache_version", "schema_version", "cache_schema"):
            self.assertNotIn(forbidden, source)

    def test_no_null_validation_field_written(self):
        # The inactive path must omit the field entirely, never write a null.
        source = inspect.getsource(run._build_weight_cache_config)
        self.assertNotIn('"validation_partition": None', source)


if __name__ == "__main__":
    unittest.main()
