import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

import main


class CLITextEncodingTests(unittest.TestCase):
    def test_help_text_encodes_with_windows_cp1252(self):
        """
        The command-line help must be printable by the default encoding
        commonly used by Windows PowerShell and Command Prompt.
        """
        help_text = main.get_parser().format_help()

        encoded_help = help_text.encode(
            "cp1252",
        )

        self.assertIsInstance(
            encoded_help,
            bytes,
        )

        self.assertIn(
            b"DEEP LEARNING ECG BIOMETRICS FRAMEWORK",
            encoded_help,
        )

        self.assertIn(
            b"--preprocessing_parameters",
            encoded_help,
        )

    def test_main_entry_point_contains_only_ascii_text(self):
        """
        Keep the CLI entry point free from characters that may fail under
        legacy Windows console encodings.
        """
        source_path = Path(
            main.__file__
        )

        source_text = source_path.read_text(
            encoding="utf-8",
        )

        non_ascii_characters = sorted(
            {
                character
                for character in source_text
                if ord(character) > 127
            }
        )

        self.assertEqual(
            non_ascii_characters,
            [],
        )


if __name__ == "__main__":
    unittest.main()