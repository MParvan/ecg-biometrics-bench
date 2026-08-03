import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import analyze_drift_covariates as drift


def build_rows(n, elapsed, heart_rate, amplitude, count):
    return [
        {
            "subject": f"s{index}",
            "elapsed_days": float(elapsed[index]),
            "heart_rate_change_bpm": float(heart_rate[index]),
            "amplitude_log_ratio": float(amplitude[index]),
            "beat_count_ratio": float(count[index]),
        }
        for index in range(n)
    ]


class HeartRateExtractionTests(unittest.TestCase):
    """
    Heart rate must come from detected R-peaks and reject failed detection.
    """

    class FakePreprocessor:
        def __init__(self, peaks):
            self.peaks = peaks

        def detect_r_peaks(self, ecg, fs):
            if self.peaks is None:
                raise RuntimeError("detection failed")

            return np.asarray(self.peaks)

    def test_regular_rhythm_gives_the_expected_rate(self):
        # One beat per 0.8 s at 250 Hz is exactly 75 bpm.
        peaks = np.arange(0, 250 * 10, 200)

        rate = drift._mean_heart_rate(
            np.zeros(2500),
            250,
            self.FakePreprocessor(peaks),
        )

        self.assertAlmostEqual(rate, 75.0, places=6)

    def test_detection_failure_returns_none(self):
        self.assertIsNone(
            drift._mean_heart_rate(
                np.zeros(2500),
                250,
                self.FakePreprocessor(None),
            )
        )

    def test_too_few_peaks_returns_none(self):
        self.assertIsNone(
            drift._mean_heart_rate(
                np.zeros(2500),
                250,
                self.FakePreprocessor([10, 20]),
            )
        )

    def test_implausible_rate_is_rejected(self):
        # Peaks 10 samples apart at 250 Hz would be 1500 bpm.
        peaks = np.arange(0, 500, 10)

        self.assertIsNone(
            drift._mean_heart_rate(
                np.zeros(2500),
                250,
                self.FakePreprocessor(peaks),
            )
        )


class VarianceInflationTests(unittest.TestCase):
    """
    Collinearity must be surfaced, because it is the confounding warning.
    """

    def test_independent_covariates_have_low_inflation(self):
        rng = np.random.default_rng(0)
        matrix = rng.normal(0, 1, (80, 3))

        factors = drift._variance_inflation_factors(
            matrix,
            ["a", "b", "c"],
        )

        for name, value in factors.items():
            with self.subTest(covariate=name):
                self.assertLess(value, 2.0)

    def test_collinear_covariates_are_flagged(self):
        rng = np.random.default_rng(1)
        first = rng.normal(0, 1, 80)
        matrix = np.column_stack(
            [
                first,
                first * 2.0 + rng.normal(0, 0.01, 80),
                rng.normal(0, 1, 80),
            ]
        )

        factors = drift._variance_inflation_factors(
            matrix,
            ["a", "b", "c"],
        )

        self.assertGreater(factors["a"], 5.0)
        self.assertGreater(factors["b"], 5.0)
        self.assertLess(factors["c"], 2.0)

    def test_single_covariate_has_unit_inflation(self):
        matrix = np.random.default_rng(2).normal(0, 1, (20, 1))

        self.assertEqual(
            drift._variance_inflation_factors(matrix, ["a"]),
            {"a": 1.0},
        )


class AssociationRecoveryTests(unittest.TestCase):
    """
    A planted association must be recovered, and a null must stay null.
    """

    def setUp(self):
        rng = np.random.default_rng(5)
        self.n = 60

        self.elapsed = rng.uniform(1, 400, self.n)
        self.heart_rate = rng.uniform(0, 25, self.n)
        self.amplitude = rng.normal(0, 0.3, self.n)
        self.count = rng.normal(0, 0.2, self.n)

        outcome = (
            0.02
            + 0.00025 * self.elapsed
            + 0.0015 * self.heart_rate
            + rng.normal(0, 0.004, self.n)
        )

        self.rows = build_rows(
            self.n,
            self.elapsed,
            self.heart_rate,
            self.amplitude,
            self.count,
        )
        self.scores = {
            f"s{index}": float(outcome[index])
            for index in range(self.n)
        }

    def test_planted_driver_is_the_strongest_association(self):
        report = drift.analyze(
            self.rows,
            self.scores,
            "per_subject_eer",
        )

        univariate = report["univariate"]

        self.assertGreater(
            abs(univariate["elapsed_days"]["spearman_rho"]),
            abs(
                univariate["amplitude_log_ratio"]["spearman_rho"]
            ),
        )
        self.assertLess(
            univariate["elapsed_days"]["p_value"],
            0.001,
        )

    def test_null_covariate_is_not_significant(self):
        report = drift.analyze(
            self.rows,
            self.scores,
            "per_subject_eer",
        )

        self.assertGreater(
            report["univariate"]["beat_count_ratio"]["p_value"],
            0.05,
        )

    def test_joint_model_ranks_the_planted_drivers(self):
        report = drift.analyze(
            self.rows,
            self.scores,
            "per_subject_eer",
        )

        coefficients = report["joint_model"][
            "standardized_coefficients"
        ]

        self.assertGreater(
            abs(coefficients["elapsed_days"]),
            abs(coefficients["heart_rate_change_bpm"]),
        )
        self.assertGreater(
            abs(coefficients["heart_rate_change_bpm"]),
            abs(coefficients["beat_count_ratio"]),
        )

    def test_joint_model_explains_most_variance(self):
        report = drift.analyze(
            self.rows,
            self.scores,
            "per_subject_eer",
        )

        self.assertGreater(
            report["joint_model"]["r_squared"],
            0.8,
        )


class LimitationReportingTests(unittest.TestCase):
    """
    The report must always state what cannot be identified.
    """

    def test_unidentifiable_factors_are_always_present(self):
        report = drift.analyze([], {}, "eer")

        self.assertTrue(
            report["unidentifiable_factors"]
        )
        self.assertTrue(
            any(
                "Electrode shift" in text
                for text in report["unidentifiable_factors"]
            )
        )

    def test_confounding_is_named_in_the_limitations(self):
        report = drift.analyze([], {}, "eer")

        self.assertTrue(
            any(
                "not causal" in text
                for text in report["unidentifiable_factors"]
            )
        )

    def test_summary_leads_with_limitations(self):
        report = drift.analyze([], {}, "eer")
        summary = drift.render_summary(
            report,
            "ecgid",
            "leave-last-out-long-term",
        )

        self.assertIn(
            "Not identifiable from this data",
            summary,
        )

    def test_joint_model_carries_an_interpretation_note(self):
        rng = np.random.default_rng(7)
        n = 40
        rows = build_rows(
            n,
            rng.uniform(1, 400, n),
            rng.uniform(0, 25, n),
            rng.normal(0, 0.3, n),
            rng.normal(0, 0.2, n),
        )
        scores = {
            f"s{index}": float(rng.normal(0.1, 0.02))
            for index in range(n)
        }

        report = drift.analyze(rows, scores, "eer")

        self.assertIn(
            "associations rather than causal effects",
            report["joint_model"]["note"],
        )


class DegenerateInputTests(unittest.TestCase):
    """
    Missing or constant data must be reported, never silently analysed.
    """

    def test_too_few_subjects_is_reported(self):
        rows = build_rows(3, [1, 2, 3], [1, 2, 3], [0, 0, 0], [0, 0, 0])
        scores = {"s0": 0.1, "s1": 0.2, "s2": 0.3}

        report = drift.analyze(rows, scores, "eer")

        self.assertIn("Too few subjects", report["status"])
        self.assertEqual(report["univariate"], {})

    def test_constant_covariate_is_reported_as_uninformative(self):
        rng = np.random.default_rng(3)
        n = 40
        rows = build_rows(
            n,
            rng.uniform(1, 400, n),
            rng.uniform(0, 25, n),
            np.zeros(n),
            rng.normal(0, 0.2, n),
        )
        scores = {
            f"s{index}": float(rng.normal(0.1, 0.02))
            for index in range(n)
        }

        report = drift.analyze(rows, scores, "eer")

        self.assertIn(
            "constant",
            report["univariate"]["amplitude_log_ratio"]["status"],
        )

    def test_subjects_without_scores_are_excluded(self):
        rng = np.random.default_rng(4)
        n = 40
        rows = build_rows(
            n,
            rng.uniform(1, 400, n),
            rng.uniform(0, 25, n),
            rng.normal(0, 0.3, n),
            rng.normal(0, 0.2, n),
        )
        scores = {
            f"s{index}": float(rng.normal(0.1, 0.02))
            for index in range(10)
        }

        report = drift.analyze(rows, scores, "eer")

        self.assertEqual(report["subjects_analysed"], 10)

    def test_non_finite_covariates_are_excluded(self):
        rng = np.random.default_rng(6)
        n = 40
        rows = build_rows(
            n,
            rng.uniform(1, 400, n),
            rng.uniform(0, 25, n),
            rng.normal(0, 0.3, n),
            rng.normal(0, 0.2, n),
        )
        rows[0]["elapsed_days"] = None
        rows[1]["heart_rate_change_bpm"] = float("nan")

        scores = {
            f"s{index}": float(rng.normal(0.1, 0.02))
            for index in range(n)
        }

        report = drift.analyze(rows, scores, "eer")

        self.assertEqual(report["subjects_analysed"], n - 2)


class ScoreFileTests(unittest.TestCase):
    """
    The outcome file contract must be enforced with clear errors.
    """

    def setUp(self):
        self.root = Path(
            tempfile.mkdtemp(prefix="drift_scores_")
        )

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_valid_file_is_loaded(self):
        path = self.root / "scores.csv"
        path.write_text(
            "subject,per_subject_eer\ns0,0.12\ns1,0.21\n",
            encoding="utf-8",
        )

        scores, outcome = drift.load_subject_scores(path)

        self.assertEqual(outcome, "per_subject_eer")
        self.assertEqual(scores, {"s0": 0.12, "s1": 0.21})

    def test_missing_file_is_reported(self):
        with self.assertRaises(SystemExit):
            drift.load_subject_scores(
                self.root / "absent.csv"
            )

    def test_missing_subject_column_is_reported(self):
        path = self.root / "bad.csv"
        path.write_text("id,eer\ns0,0.12\n", encoding="utf-8")

        with self.assertRaises(SystemExit):
            drift.load_subject_scores(path)

    def test_missing_outcome_column_is_reported(self):
        path = self.root / "bad2.csv"
        path.write_text("subject\ns0\n", encoding="utf-8")

        with self.assertRaises(SystemExit):
            drift.load_subject_scores(path)

    def test_unparseable_rows_are_skipped(self):
        path = self.root / "mixed.csv"
        path.write_text(
            "subject,eer\ns0,0.12\ns1,not_a_number\ns2,0.3\n",
            encoding="utf-8",
        )

        scores, _ = drift.load_subject_scores(path)

        self.assertEqual(set(scores), {"s0", "s2"})


if __name__ == "__main__":
    unittest.main()
