"""Regression tests for the findings from the uncommitted-changes review: crash-mode selection, the
JUnit writer and error report, and the CLI's refusal of bad options."""
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from ckpt_chaos.byo import _select
from ckpt_chaos.invariants import ResumeOutcome
from ckpt_chaos.report import _xml_safe, junit, junit_error
from ckpt_chaos.runner import Row

ROOT = Path(__file__).resolve().parent.parent
EVENTS = [(n, "open", f"f{n}", n % 2 == 0) for n in range(1, 21)]  # every even event is tornable


class Select(unittest.TestCase):
    def test_each_mode_is_sampled_across_the_whole_save(self):
        points = _select(EVENTS, 8, ("before", "term"))
        before = [n for n, m in points if m == "before"]
        term = [n for n, m in points if m == "term"]
        self.assertEqual(before, [1, 6, 11, 16])  # 20 events, 4 samples: stride 5, spread over the whole save
        self.assertEqual(term, [1, 6, 11, 16])

    def test_budget_is_split_and_the_total_respected(self):
        points = _select(EVENTS, 10, ("before", "torn", "term"))
        self.assertEqual(len(points), 10)

    def test_torn_only_samples_tornable_events(self):
        points = _select(EVENTS, 40, ("torn",))
        self.assertTrue(points)
        self.assertTrue(all(n % 2 == 0 for n, _ in points))

    def test_unknown_mode_is_refused_not_silently_empty(self):
        with self.assertRaises(SystemExit):
            _select(EVENTS, 10, ("befor",))

    def test_zero_max_points_is_refused(self):
        with self.assertRaises(SystemExit):
            _select(EVENTS, 0, ("before",))


class Junit(unittest.TestCase):
    def row(self, label, verdict, raised=None):
        return Row(label, True, False, ResumeOutcome(raised, None if raised else 0, 5, 0.0), verdict)

    def test_output_parses_and_counts_failures(self):
        text = junit([self.row("#1 open:a [before]", "PASS"), self.row("#2 open:b [torn]", "HARD_FAIL", raised="ExitCode1")],
                     fail_on={"HARD_FAIL"})
        suite = ET.fromstring(text)
        self.assertEqual(suite.get("tests"), "2")
        self.assertEqual(suite.get("failures"), "1")

    def test_control_characters_in_a_label_still_give_well_formed_xml(self):
        text = junit([self.row("#1 open:bad\x01name [before]", "PASS")], fail_on={"HARD_FAIL"})
        ET.fromstring(text)  # raises if the file is not well-formed
        self.assertNotIn("\x01", text)

    def test_xml_safe_keeps_normal_text(self):
        self.assertEqual(_xml_safe("a & b [c]:d"), "a & b [c]:d")

    def test_error_report_is_a_parseable_single_error(self):
        suite = ET.fromstring(junit_error("your command failed <badly> & \x02 died"))
        self.assertEqual(suite.get("errors"), "1")
        self.assertEqual(len(suite.findall("testcase")), 1)


class CliRefusals(unittest.TestCase):
    def run_cli(self, *args):
        work = Path(tempfile.mkdtemp(prefix="ckptchaos_refuse_")) / "work"
        return subprocess.run([sys.executable, "-m", "ckpt_chaos", "run", "--work", str(work), *args, "--",
                               sys.executable, "-c", "pass", "{out}"], cwd=ROOT, capture_output=True, text=True, timeout=120)

    def test_typo_in_modes_fails_the_run(self):
        p = self.run_cli("--modes", "torm")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("--modes takes", p.stderr)

    def test_empty_modes_fails_the_run(self):
        p = self.run_cli("--modes", "")
        self.assertNotEqual(p.returncode, 0)

    def test_max_points_zero_is_refused(self):
        p = self.run_cli("--max-points", "0")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("--max-points must be at least 1", p.stderr)

    def test_junit_is_written_as_an_error_when_the_run_cannot_start(self):
        out = Path(tempfile.mkdtemp(prefix="ckptchaos_junit_err_")) / "junit.xml"
        p = subprocess.run([sys.executable, "-m", "ckpt_chaos", "run", "--junit", str(out), "--",
                            sys.executable, "-c", "pass"], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertNotEqual(p.returncode, 0)
        suite = ET.parse(out).getroot()
        self.assertEqual(suite.get("errors"), "1")


if __name__ == "__main__":
    unittest.main()
