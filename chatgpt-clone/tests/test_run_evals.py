"""The suite runner reports failures and continues with the other suite."""
import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_evals


class SuiteRunnerTests(unittest.TestCase):
    def test_runs_both_and_reports_failure(self):
        output = io.StringIO()
        with patch.object(run_evals.subprocess, "run", side_effect=[SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)]) as run, contextlib.redirect_stdout(output):
            status = run_evals.run_suites(["manual", "deepeval"])
        self.assertEqual(status, 1)
        self.assertEqual(run.call_count, 2)
        self.assertIn("manual: FAIL", output.getvalue())
        self.assertIn("deepeval: PASS", output.getvalue())
        self.assertIn("1/2 suites passed", output.getvalue())

    def test_launch_error_does_not_stop_other_suite(self):
        output = io.StringIO()
        with patch.object(run_evals.subprocess, "run", side_effect=[OSError("cannot start"), SimpleNamespace(returncode=0)]) as run, contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            status = run_evals.run_suites(["manual", "deepeval"])
        self.assertEqual(status, 1)
        self.assertEqual(run.call_count, 2)
        self.assertIn("manual: ERROR", output.getvalue())

    def test_single_selected_suite_passes(self):
        with patch.object(run_evals.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run, contextlib.redirect_stdout(io.StringIO()):
            status = run_evals.run_suites(["manual"])
        self.assertEqual(status, 0)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(Path(run.call_args.args[0][-1]).name, "evaluation.py")


if __name__ == "__main__":
    unittest.main()
