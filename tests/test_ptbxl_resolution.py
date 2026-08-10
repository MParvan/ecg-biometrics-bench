"""
Pin how PTB-XL chooses between its two stored copies.

Every recording is distributed twice, at 100 Hz and at 500 Hz, and the two are
not related by simple decimation: the low-rate copy is an anti-aliased
resampling, so a result obtained at one rate does not carry over to the other.
Which copy a run read is therefore part of what the result means, and has to
reach the cache identity.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from load_dataset import load_ptbxl_dataset


class PTBXLResolutionTests(unittest.TestCase):
    def test_the_default_is_the_five_hundred_hertz_copy(self):
        loader = load_ptbxl_dataset()

        self.assertEqual(loader.resolution, "high")
        self.assertEqual(
            loader.SAMPLING_RATES[loader.resolution],
            500,
        )

    def test_the_low_resolution_copy_is_one_hundred_hertz(self):
        loader = load_ptbxl_dataset(resolution="low")

        self.assertEqual(
            loader.SAMPLING_RATES[loader.resolution],
            100,
        )

    def test_an_unknown_resolution_is_rejected(self):
        """
        Anything other than the two stored copies must fail at construction
        rather than silently falling through to one of them.
        """
        with self.assertRaisesRegex(
            ValueError,
            "resolution must be one of",
        ):
            load_ptbxl_dataset(resolution="medium")

    def test_the_resolution_reaches_the_cache_identity(self):
        """
        Reading a different copy produces different arrays, so a run at one
        rate must not be served from a cache built at the other.
        """
        import utils

        high = utils._generate_config_hash(
            utils._build_loader_cache_identity(
                load_ptbxl_dataset(resolution="high")
            )
        )
        low = utils._generate_config_hash(
            utils._build_loader_cache_identity(
                load_ptbxl_dataset(resolution="low")
            )
        )

        self.assertNotEqual(high, low)

    def test_both_rates_are_declared_for_every_supported_resolution(self):
        loader = load_ptbxl_dataset()

        self.assertEqual(
            set(loader.SAMPLING_RATES),
            {"high", "low"},
        )


if __name__ == "__main__":
    unittest.main()
