"""`ckpt-chaos repro`: selection, the generated script and the draft issue, then an end-to-end run of the script."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ckpt_chaos.repro import issue, script, select, write

ROOT = Path(__file__).resolve().parent.parent
ROTATING = ROOT / "examples" / "rotating_json.py"
HOME = str(Path.home())


def meta(work: str) -> dict:
    return {"train": [sys.executable, str(ROTATING), "--out", "{out}", "--bug"], "resume": [sys.executable, str(ROTATING), "--out", "{out}"],
            "step_regex": r"resumed from step (\d+)", "checkpoint_glob": "{out}/ckpt-*.json", "ignore_size": False,
            "result_file": "{out}/result.json", "result_cmd": None, "work": work, "reference_steps": [5, 10, 15, 20],
            "versions": {"python": "3.12.0", "platform": "test", "ckpt-chaos": "0.1.2", "torch": "2.14.1"}}


def record(order: int, point, outcome: dict, event=("open", "ckpt-10.json.tmp"), tail: str = "") -> dict:
    label = "no crash (control)" if point is None else f"#{point[0]} {event[0]}:{event[1]} [{point[1]}]"
    rec = {"order": order, "label": label, "rc": 1, "crashed": True, "drift": False, "outcome": outcome}
    if point is not None:
        rec.update(point=list(point), event=list(event), resume_tail=tail)
    return rec


def make_run(records: list[dict], work: str) -> Path:
    run = Path(tempfile.mkdtemp(prefix="ckptchaos_repro_"))
    (run / "meta.json").write_text(json.dumps(meta(work)), encoding="utf-8")
    for rec in records:
        d = run / f"pt_{rec['order']:03d}"
        d.mkdir()
        (d / "result.json").write_text(json.dumps(rec), encoding="utf-8")
    return run


LOST = {"raised": None, "resumed_from_step": 0, "newest_complete_step": 5, "weights_diff": 0.0, "ranks_agree": True}
RAISED = {"raised": "ExitCode1", "resumed_from_step": None, "newest_complete_step": 5, "weights_diff": None, "ranks_agree": True}
PASS = {"raised": None, "resumed_from_step": 5, "newest_complete_step": 5, "weights_diff": 0.0, "ranks_agree": True}


class Select(unittest.TestCase):
    def test_takes_one_of_each_verdict_before_repeating(self):
        work = tempfile.mkdtemp()
        run = make_run([record(1, (4, "before"), LOST), record(2, (5, "before"), LOST), record(3, (6, "before"), RAISED),
                        record(4, (7, "before"), PASS)], work)
        verdicts = [v for v, _ in select(run, 2)]
        self.assertEqual(verdicts, ["HARD_FAIL", "LOST_WORK"])  # not two LOST_WORK
        self.assertEqual(len(select(run, 10)), 3)  # PASS is never written

    def test_control_is_never_selected(self):
        run = make_run([record(0, None, LOST)], tempfile.mkdtemp())
        self.assertEqual(select(run, 5), [])


class Generated(unittest.TestCase):
    def setUp(self):
        self.work = str(Path(tempfile.mkdtemp()) / "work")
        self.rec = record(4, (4, "before"), LOST, tail=f"resumed from step 0\n{HOME}/secret/x")
        self.meta = meta(self.work)

    def test_script_is_valid_python_and_carries_the_point(self):
        text = script(self.meta, self.rec)
        compile(text, "repro.py", "exec")  # raises on a syntax error
        self.assertIn("CRASH_AT = 4", text)
        self.assertIn("MODE = 'before'", text)
        self.assertIn("EXPECT_EVENT = ['open', 'ckpt-10.json.tmp']", text)
        self.assertNotIn("@@", text)  # every token was filled in

    def test_issue_states_the_verdict_and_hides_local_paths(self):
        text = issue(self.meta, self.rec, "LOST_WORK")
        self.assertIn("**LOST_WORK**", text)
        self.assertIn("started from step 0, but the newest complete checkpoint was step 5", text)
        self.assertIn("pip install ckpt-chaos==0.1.2", text)
        self.assertIn("steps 5, 10, 15, 20", text)
        self.assertNotIn(HOME, text)

    def test_write_refuses_a_run_without_meta(self):
        run = Path(tempfile.mkdtemp())
        with self.assertRaises(SystemExit):
            write(run, run / "repro")


class EndToEnd(unittest.TestCase):
    def test_a_real_lost_work_run_writes_a_script_that_reproduces_it(self):
        work = Path(tempfile.mkdtemp(prefix="ckptchaos_e2e_")) / "work"
        run = subprocess.run([sys.executable, "-m", "ckpt_chaos", "run", "--checkpoint-glob", "{out}/ckpt-*.json",
                              "--step-regex", r"resumed from step (\d+)", "--result-file", "{out}/result.json",
                              "--work", str(work), "--jobs", "2", "--",
                              sys.executable, str(ROTATING), "--out", "{out}", "--bug"],
                             cwd=ROOT, capture_output=True, text=True, timeout=300)
        self.assertIn("LOST_WORK", run.stdout, run.stdout + run.stderr)
        dest = work.parent / "repro"
        out = subprocess.run([sys.executable, "-m", "ckpt_chaos", "repro", str(work), "--dest", str(dest), "--limit", "1"],
                             cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        folders = sorted(p for p in dest.iterdir() if p.is_dir())
        self.assertEqual(len(folders), 1)
        self.assertTrue((folders[0] / "ISSUE.md").read_text(encoding="utf-8").count("LOST_WORK") >= 1)
        env = dict(os.environ, PROJECT_DIR=str(ROOT), PYTHONPATH=str(ROOT))
        rep = subprocess.run([sys.executable, str(folders[0] / "repro.py")], env=env, cwd=folders[0],
                             capture_output=True, text=True, timeout=600)
        self.assertEqual(rep.returncode, 1, rep.stdout + rep.stderr)
        self.assertIn("reproduced 3 of 3", rep.stdout)

    def test_nothing_to_reproduce_is_said_plainly(self):
        run = make_run([record(1, (4, "before"), PASS)], tempfile.mkdtemp())
        out = subprocess.run([sys.executable, "-m", "ckpt_chaos", "repro", str(run)], cwd=ROOT, capture_output=True,
                             text=True, timeout=60)
        self.assertEqual(out.returncode, 0)
        self.assertIn("nothing to reproduce", out.stdout)


if __name__ == "__main__":
    unittest.main()
