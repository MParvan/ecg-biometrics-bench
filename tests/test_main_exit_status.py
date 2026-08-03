import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import _terminate_pipeline_with_error


class MainExitStatusTests(unittest.TestCase):
    def test_pipeline_failure_exits_with_status_one(self):
        captured_stderr = StringIO()
        error = RuntimeError("synthetic pipeline failure")

        with redirect_stderr(captured_stderr):
            with self.assertRaises(SystemExit) as context:
                _terminate_pipeline_with_error(error)

        self.assertEqual(context.exception.code, 1)

        error_output = captured_stderr.getvalue()

        self.assertIn(
            "[CRITICAL ERROR] Pipeline Failed",
            error_output,
        )
        self.assertIn(
            "synthetic pipeline failure",
            error_output,
        )
        self.assertIn(
            "RuntimeError",
            error_output,
        )


if __name__ == "__main__":
    unittest.main()