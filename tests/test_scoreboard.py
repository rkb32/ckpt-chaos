"""`ckpt-chaos scoreboard`: the control is not a crash point, and failing verdicts are counted."""
import json
import tempfile
import unittest
from pathlib import Path

from ckpt_chaos.scoreboard import markdown, summarize

PASS = {"raised": None, "resumed_from_step": 5, "newest_complete_step": 5, "weights_diff": 0.0, "ranks_agree": True}
RAISED = {"raised": "ExitCode1", "resumed_from_step": None, "newest_complete_step": 5, "weights_diff": None, "ranks_agree": True}


def make_run(records: list[dict], meta: dict | None = None) -> Path:
    run = Path(tempfile.mkdtemp(prefix="ckptchaos_sb_"))
    for i, rec in enumerate(records, 1):
        d = run / f"pt_{i:03d}"
        d.mkdir()
        (d / "result.json").write_text(json.dumps(rec), encoding="utf-8")
    if meta is not None:
        (run / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return run


class Summarize(unittest.TestCase):
    def test_control_is_not_counted_as_a_crash_point(self):
        run = make_run([{"label": "no crash (control)", "rc": 0, "outcome": PASS},
                        {"label": "#1 open:a [before]", "rc": 137, "outcome": PASS},
                        {"label": "#2 open:b [before]", "rc": 137, "outcome": RAISED}])
        s = summarize(run, "demo")
        self.assertEqual(s["points"], 2)
        self.assertEqual(s["counts"]["PASS"], 1)
        self.assertEqual(s["counts"]["HARD_FAIL"], 1)

    def test_a_dead_control_is_counted_as_control_fail(self):
        run = make_run([{"label": "no crash (control)", "rc": 1, "outcome": RAISED}])
        self.assertEqual(summarize(run)["counts"]["CONTROL_FAIL"], 1)

    def test_versions_come_from_meta(self):
        run = make_run([{"label": "#1 x [before]", "rc": 137, "outcome": PASS}],
                       meta={"versions": {"torch": "2.14.1", "python": "3.14.8"}})
        self.assertIn("torch 2.14.1", markdown([summarize(run, "t")]))

    def test_table_has_one_row_per_run_and_escapes_pipes(self):
        run = make_run([{"label": "#1 x [before]", "rc": 137, "outcome": PASS}])
        text = markdown([summarize(run, "a|b")])
        self.assertEqual(len([l for l in text.splitlines() if l.startswith("| ")]), 2)  # header + one run
        self.assertIn("a\\|b", text)


if __name__ == "__main__":
    unittest.main()
