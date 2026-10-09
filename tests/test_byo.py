"""`ckpt-chaos run` on a stdlib-only script: no torch needed, and torn Python-written files are exercised."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ckpt_chaos.byo import _newest_complete, _snapshot

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "examples" / "naive_json.py"
ROTATING = ROOT / "examples" / "rotating_json.py"
CONTRACT = ["--checkpoint-glob", "{out}/ckpt-*.json"]
FAIL_ON_LOST = ["--fail-on", "HARD_FAIL,SILENT_DIVERGENCE,LOST_WORK"]


def run_cli(*extra: str, script: Path = SCRIPT, cli: tuple[str, ...] = ()) -> tuple[int, str]:
    work = Path(tempfile.mkdtemp(prefix="ckptchaos_byo_")) / "work"
    cmd = [sys.executable, "-m", "ckpt_chaos", "run", "--result-file", "{out}/result.json",
           "--step-regex", r"resumed from step (\d+)", "--work", str(work), "--jobs", "2", *cli,
           "--", sys.executable, str(script), "--out", "{out}", *extra]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    return p.returncode, p.stdout + p.stderr


class Byo(unittest.TestCase):
    def test_naive_save_is_caught_by_a_torn_write(self):
        code, out = run_cli()
        self.assertEqual(code, 1, out)
        self.assertIn("HARD_FAIL", out)
        self.assertIn("[torn]", out)

    def test_atomic_save_passes_every_crash_point(self):
        code, out = run_cli("--atomic")
        self.assertEqual(code, 0, out)
        self.assertNotIn("HARD_FAIL", out)
        self.assertNotIn("SILENT_DIVERGENCE", out)
        self.assertIn("PASS", out)

    def test_command_without_out_placeholder_is_refused_with_a_hint(self):
        p = subprocess.run([sys.executable, "-m", "ckpt_chaos", "run", "--", sys.executable, "-c", "pass"],
                           cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("{out}", p.stderr + p.stdout)


class ResumeContract(unittest.TestCase):
    def test_a_correct_resume_uses_the_newest_complete_checkpoint(self):
        code, out = run_cli(script=ROTATING, cli=(*CONTRACT, *FAIL_ON_LOST))
        self.assertEqual(code, 0, out)
        self.assertNotIn("LOST_WORK", out)
        self.assertIn("newest ok", out)
        self.assertIn("steps [5, 10, 15, 20]", out)

    def test_silently_starting_over_is_lost_work(self):
        code, out = run_cli("--bug", script=ROTATING, cli=(*CONTRACT, *FAIL_ON_LOST))
        self.assertEqual(code, 1, out)
        self.assertIn("LOST_WORK", out)
        self.assertIn("resumed from step 0", out)
        self.assertNotIn("HARD_FAIL", out)  # nothing raised: that is exactly why it is dangerous

    def test_contract_check_is_not_a_failure_by_default(self):
        code, out = run_cli("--bug", script=ROTATING, cli=CONTRACT)
        self.assertEqual(code, 0, out)  # LOST_WORK is reported but only fails the run when asked

    def test_glob_without_step_regex_is_refused(self):
        p = subprocess.run([sys.executable, "-m", "ckpt_chaos", "run", "--checkpoint-glob", "{out}/ckpt-*", "--",
                            sys.executable, str(ROTATING), "--out", "{out}"], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("--step-regex", p.stderr + p.stdout)

    def test_glob_matching_nothing_is_refused(self):
        code, out = run_cli(script=ROTATING, cli=("--checkpoint-glob", "{out}/nothing-*.json"))
        self.assertNotEqual(code, 0)
        self.assertIn("matched no checkpoint", out)


class Snapshots(unittest.TestCase):
    def make(self, files: dict[str, int]) -> Path:
        d = Path(tempfile.mkdtemp(prefix="ckptchaos_snap_"))
        for rel, size in files.items():
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_bytes(b"x" * size)
        return d

    def test_files_and_folders_with_a_step_in_the_name(self):
        d = self.make({"checkpoint-10/model.bin": 8, "checkpoint-10/state.json": 2, "ckpt-5.pt": 4, "notes.txt": 1})
        snap = _snapshot("{out}/*", d)
        self.assertEqual(sorted(s for s, _ in snap.values()), [5, 10])  # notes.txt has no number

    def test_a_torn_or_missing_file_is_not_complete(self):
        ref = _snapshot("{out}/checkpoint-*", self.make({"checkpoint-5/a": 8, "checkpoint-5/b": 2,
                                                         "checkpoint-10/a": 8, "checkpoint-10/b": 2}))
        torn = _snapshot("{out}/checkpoint-*", self.make({"checkpoint-5/a": 8, "checkpoint-5/b": 2,
                                                          "checkpoint-10/a": 4, "checkpoint-10/b": 2}))
        partial = _snapshot("{out}/checkpoint-*", self.make({"checkpoint-5/a": 8, "checkpoint-5/b": 2,
                                                             "checkpoint-10/a": 8}))
        self.assertEqual(_newest_complete(ref, ref), 10)
        self.assertEqual(_newest_complete(ref, torn), 5)
        self.assertEqual(_newest_complete(ref, partial), 5)
        self.assertEqual(_newest_complete(ref, torn, ignore_size=True), 10)  # names only: the torn size is ignored
        self.assertEqual(_newest_complete(ref, {}), 0)

    def test_relative_glob_is_taken_inside_out(self):
        snap = _snapshot("ckpt-*.json", self.make({"ckpt-5.json": 3}))
        self.assertEqual(list(snap), ["ckpt-5.json"])


if __name__ == "__main__":
    unittest.main()
