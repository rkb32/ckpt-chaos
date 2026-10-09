"""The GitHub Action: input handling (ckpt_chaos.ci), the Markdown job summary, and action.yml itself."""
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ckpt_chaos.ci import argv_from_env
from ckpt_chaos.invariants import ResumeOutcome
from ckpt_chaos.report import error_markdown, markdown
from ckpt_chaos.runner import Row

ROOT = Path(__file__).resolve().parent.parent
ROTATING = ROOT / "examples" / "rotating_json.py"


def row(label, verdict, *, step=0, newest=0, raised=None, diff=0.0):
    return Row(label, True, False, ResumeOutcome(raised, None if raised else step, newest, None if raised else diff), verdict)


class ArgvFromEnv(unittest.TestCase):
    def test_inputs_become_flags_and_the_command_is_split_after_dashdash(self):
        argv = argv_from_env({"INPUT_COMMAND": 'python train.py --name "a b" --out {out}', "INPUT_CHECKPOINT_GLOB": "{out}/ckpt-*",
                              "INPUT_STEP_REGEX": r"step (\d+)", "INPUT_JOBS": "2", "INPUT_CKPT_IGNORE_SIZE": "true",
                              "INPUT_FAIL_ON": "", "GITHUB_STEP_SUMMARY": "/tmp/s.md"})
        self.assertEqual(argv[0], "run")
        self.assertIn(["--checkpoint-glob", "{out}/ckpt-*"], [argv[i:i + 2] for i in range(len(argv))])
        self.assertIn("--ckpt-ignore-size", argv)
        self.assertNotIn("--fail-on", argv)  # empty input means "use the default"
        self.assertEqual(argv[argv.index("--summary") + 1], "/tmp/s.md")
        self.assertEqual(argv[argv.index("--") + 1:], ["python", "train.py", "--name", "a b", "--out", "{out}"])

    def test_shell_metacharacters_in_inputs_stay_data(self):
        argv = argv_from_env({"INPUT_COMMAND": "python train.py --out {out}", "INPUT_RESULT_CMD": "echo $(whoami); rm -rf /"})
        self.assertEqual(argv[argv.index("--result-cmd") + 1], "echo $(whoami); rm -rf /")

    def test_two_calls_never_share_a_work_directory(self):
        base = tempfile.mkdtemp(prefix="ckptchaos_base_")
        env = {"INPUT_COMMAND": "python x.py --out {out}", "INPUT_WORK_DIR": base}
        calls = [argv_from_env(env) for _ in range(2)]
        works = [a[a.index("--work") + 1] for a in calls]
        self.assertNotEqual(works[0], works[1])
        self.assertTrue(all(Path(w).parent == Path(base) and Path(w).is_dir() for w in works))

    def test_command_is_required(self):
        with self.assertRaises(SystemExit) as cm:
            argv_from_env({"INPUT_COMMAND": "  "})
        self.assertIn("command", str(cm.exception))


class Report(unittest.TestCase):
    ROWS = [row("no crash (control)", "PASS"), row("#1 os.mkdir:out [before]", "PASS"),
            row("#4 open:ckpt-10.json.tmp [before]", "LOST_WORK", step=0, newest=5),
            row("#7 open:a|b [torn]", "HARD_FAIL", raised="ExitCode1")]

    def test_failing_build_lists_the_failing_runs_and_escapes_pipes(self):
        md = markdown(self.ROWS, fail_on={"HARD_FAIL"}, invocation="ckpt-chaos run -- python x.py --out {out}")
        self.assertIn("## ckpt-chaos: FAILED", md)
        self.assertIn("1 of 4 runs", md)
        self.assertIn("### Failing runs", md)
        self.assertIn("a\\|b", md)
        self.assertIn("Reported but not failing the build: 1 LOST_WORK", md)  # LOST_WORK is shown, not gating
        self.assertIn("Newest complete", md)
        self.assertIn("ckpt-chaos run -- python x.py", md)

    def test_passing_build_has_no_failing_section(self):
        md = markdown([row("no crash (control)", "PASS"), row("#1 x [before]", "PASS")], fail_on={"HARD_FAIL"})
        self.assertIn("## ckpt-chaos: PASSED", md)
        self.assertNotIn("Failing runs", md)
        self.assertNotIn("Newest complete", md)  # no checkpoint-glob, so the column would be meaningless

    def test_error_summary(self):
        self.assertIn("could not run", error_markdown("your command failed on its own (exit 2)"))


class Summary(unittest.TestCase):
    def run_cli(self, summary: Path, *extra: str):
        work = Path(tempfile.mkdtemp(prefix="ckptchaos_act_")) / "work"
        cmd = [sys.executable, "-m", "ckpt_chaos", "run", "--checkpoint-glob", "{out}/ckpt-*.json", "--step-regex",
               r"resumed from step (\d+)", "--result-file", "{out}/result.json", "--fail-on", "HARD_FAIL,SILENT_DIVERGENCE,LOST_WORK",
               "--work", str(work), "--jobs", "2", "--summary", str(summary), "--", sys.executable, str(ROTATING), "--out", "{out}", *extra]
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)

    def test_summary_is_appended_and_matches_the_exit_code(self):
        f = Path(tempfile.mkdtemp(prefix="ckptchaos_sum_")) / "summary.md"
        f.write_text("earlier step\n", encoding="utf-8")
        bad = self.run_cli(f, "--bug")
        self.assertEqual(bad.returncode, 1, bad.stdout + bad.stderr)
        text = f.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("earlier step\n"))  # appended, not overwritten
        self.assertIn("## ckpt-chaos: FAILED", text)
        self.assertIn("LOST_WORK", text)
        self.assertNotIn(str(f), text)  # machine-specific flags are not in the reproduce command
        good = self.run_cli(f)
        self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
        self.assertIn("## ckpt-chaos: PASSED", f.read_text(encoding="utf-8"))

    def test_a_job_that_cannot_start_is_explained_in_the_summary(self):
        f = Path(tempfile.mkdtemp(prefix="ckptchaos_sum_")) / "summary.md"
        p = subprocess.run([sys.executable, "-m", "ckpt_chaos", "run", "--summary", str(f), "--", sys.executable, "-c", "import sys; sys.exit(3)",
                            "{out}"], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("could not run", f.read_text(encoding="utf-8"))


class ActionFile(unittest.TestCase):
    TEXT = (ROOT / "action.yml").read_text(encoding="utf-8")

    def test_no_input_is_interpolated_into_a_script(self):
        for m in re.finditer(r"^\s+run:\s*(.*)$", self.TEXT, re.M):
            self.assertNotIn("${{", m.group(1), "inputs must reach scripts through env, not text substitution")

    def test_every_input_used_by_the_entry_point_is_declared(self):
        declared = set(re.findall(r"^  ([a-z-]+):\s*$", self.TEXT.split("inputs:")[1].split("runs:")[0], re.M))
        used = set(re.findall(r"\$\{\{ inputs\.([a-z-]+) \}\}", self.TEXT))
        self.assertLessEqual(used, declared)

    def test_it_is_a_composite_action(self):
        self.assertIn("using: composite", self.TEXT)


if __name__ == "__main__":
    unittest.main()
