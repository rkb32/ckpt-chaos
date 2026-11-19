"""SWALLOWED must not fire on a job that prunes checkpoints on purpose (retention rotation), and must fire on a job
that loses a kept checkpoint after a swallowed disk-full error."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
CKPT = ["--checkpoint-glob", "{out}/ckpt-*.json", "--step-regex", r"resumed from step (\d+)",
        "--saved-regex", r"saved step (\d+)", "--result-file", "{out}/result.json"]


def run_cli(job: str, *extra: str, timeout: int = 900) -> subprocess.CompletedProcess:
    work = Path(tempfile.mkdtemp(prefix="ckptchaos_swallow_")) / "work"
    return subprocess.run([PY, "-m", "ckpt_chaos", "run", "--work", str(work), "--jobs", "2", *extra,
                           "--", PY, job, "--out", "{out}"],
                          cwd=ROOT, capture_output=True, text=True, timeout=timeout)


class Swallowed(unittest.TestCase):
    def test_retention_rotation_is_not_swallowed(self):
        # keep-last-1 rotation deletes older checkpoints on purpose: the reference run deletes them too
        p = run_cli("examples/rotate_delete.py", "--modes", "enospc", *CKPT)
        self.assertNotIn("SWALLOWED", p.stdout, p.stdout[-2000:])

    def test_lost_kept_checkpoint_is_swallowed(self):
        p = run_cli("examples/swallow_enospc.py", "--modes", "enospc", *CKPT)
        self.assertIn("SWALLOWED", p.stdout, p.stdout[-2000:])


if __name__ == "__main__":
    unittest.main()
