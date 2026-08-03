import unittest

from load_dataset import load_ecgid_dataset
from main import parse_experiment_arguments


class ECGIDSignalTypeTests(unittest.TestCase):
    def test_cli_default_is_filtered(self):
        args, _ = parse_experiment_arguments(
            [
                "--dataset",
                "ecgid",
                "--task",
                "1",
            ]
        )

        self.assertEqual(
            args.signal_type,
            "filtered",
        )

    def test_loader_default_is_filtered(self):
        loader = load_ecgid_dataset()

        self.assertEqual(
            loader.signal_type,
            "filtered",
        )

        self.assertEqual(
            loader._get_channel_index(),
            1,
        )

    def test_explicit_filtered_channel(self):
        loader = load_ecgid_dataset(
            signal_type="filtered",
        )

        self.assertEqual(
            loader.signal_type,
            "filtered",
        )

        self.assertEqual(
            loader._get_channel_index(),
            1,
        )

    def test_explicit_raw_channel(self):
        loader = load_ecgid_dataset(
            signal_type="raw",
        )

        self.assertEqual(
            loader.signal_type,
            "raw",
        )

        self.assertEqual(
            loader._get_channel_index(),
            0,
        )

    def test_invalid_signal_type_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "signal_type must be either 'raw' or 'filtered'",
        ):
            load_ecgid_dataset(
                signal_type="noisy",
            )


if __name__ == "__main__":
    unittest.main()
