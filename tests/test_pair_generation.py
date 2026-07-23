import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

# Make the repository root importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import utils


class PairGenerationTests(unittest.TestCase):
    def test_balanced_intraset_genuine_pairs_never_use_the_same_sample_twice(self):
        """
        In balanced intra-set verification, a genuine pair must contain
        two different samples from the same subject.
        """
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

        labels = np.array([0, 0, 0, 1, 1, 1])

        def mark_self_pair(vector_1, vector_2, method="cosine"):
            # Return 1 only when the exact same sample is paired with itself.
            return 1.0 if np.array_equal(vector_1, vector_2) else 0.0

        np.random.seed(42)

        with patch("utils._compute_pair_score", side_effect=mark_self_pair):
            scores, pair_labels = utils._generate_pairs(
                embeddings1=embeddings,
                labels1=labels,
                embeddings2=None,
                labels2=None,
                num_pairs=200,
                sampling_mode="balanced",
                matching_method="cosine",
            )

        genuine_scores = scores[pair_labels == 1]

        self.assertEqual(len(scores), 200)
        self.assertEqual(np.sum(pair_labels == 1), 100)
        self.assertEqual(np.sum(pair_labels == 0), 100)
        self.assertFalse(
            np.any(genuine_scores == 1.0),
            "A genuine verification pair compared a sample with itself.",
        )


if __name__ == "__main__":
    unittest.main()