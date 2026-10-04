"""`ckpt-chaos run` on a stdlib-only script: no torch needed, and torn Python-written files are exercised."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "examples" / "naive_json.py"


def run_cli(*extra: str) -> tuple[int, str]:
    work = Path(tempfile.mkdtemp(prefix="ckptchaos_byo_")) / "work"
    cmd = [sys.executable, "-m", "ckpt_chaos", "run", "--result-file", "{out}/result.json",
           "--step-regex", r"resumed from step (\d+)", "--work", str(work), "--jobs", "2",
           "--", sys.executable, str(SCRIPT), "--out", "{out}", *extra]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    return p.returncode, p.stdout


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


if __name__ == "__main__":
    unittest.main()
