import unittest

import load_dataset
from load_dataset import load_ecgid_dataset
from main import parse_experiment_arguments


class ECGIDSignalTypeTests(unittest.TestCase):
    def test_cli_leaves_the_channel_unset_by_default(self):
        """
        An omitted flag must stay ``None`` so the loader can fall back to
        config.yaml. A concrete CLI default would silently outrank the file.
        """
        args, _ = parse_experiment_arguments(
            [
                "--dataset",
                "ecgid",
                "--task",
                "1",
            ]
        )

        self.assertIsNone(
            args.signal_type,
        )

    def test_loader_default_is_raw(self):
        loader = load_ecgid_dataset()

        self.assertEqual(
            loader.signal_type,
            "raw",
        )

        self.assertEqual(
            loader._get_channel_index(),
            0,
        )

    def test_config_file_supplies_the_default(self):
        """
        Editing the dataset entry in config.yaml must change which channel is
        read, since that is where a user would reasonably set it.
        """
        dataset_config = load_dataset.CONFIG["datasets"]["ecgid"]
        original = dataset_config.get("signal_type")

        try:
            dataset_config["signal_type"] = "filtered"
            loader = load_ecgid_dataset()

            self.assertEqual(
                loader.signal_type,
                "filtered",
            )

            self.assertEqual(
                loader._get_channel_index(),
                1,
            )
        finally:
            dataset_config["signal_type"] = original

    def test_explicit_argument_outranks_the_config_file(self):
        dataset_config = load_dataset.CONFIG["datasets"]["ecgid"]
        original = dataset_config.get("signal_type")

        try:
            dataset_config["signal_type"] = "raw"
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
        finally:
            dataset_config["signal_type"] = original

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

    def test_invalid_config_value_is_rejected(self):
        """
        A typo in config.yaml must fail loudly rather than silently falling
        back to one of the two channels.
        """
        dataset_config = load_dataset.CONFIG["datasets"]["ecgid"]
        original = dataset_config.get("signal_type")

        try:
            dataset_config["signal_type"] = "unfiltered"

            with self.assertRaisesRegex(
                ValueError,
                "signal_type must be either 'raw' or 'filtered'",
            ):
                load_ecgid_dataset()
        finally:
            dataset_config["signal_type"] = original


if __name__ == "__main__":
    unittest.main()
