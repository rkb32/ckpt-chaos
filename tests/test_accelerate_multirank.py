"""accelerate_auto may run with two ranks; accelerate_guarded and accelerate_loop still may not."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ckpt_chaos.runner import validate_bench_ranks

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


class Guard(unittest.TestCase):
    def test_accelerate_auto_accepts_two_ranks(self):
        validate_bench_ranks("accelerate_auto", 2)  # must not raise

    def test_accelerate_guarded_still_rejects_two_ranks(self):
        with self.assertRaises(ValueError):
            validate_bench_ranks("accelerate_guarded", 2)

    def test_accelerate_loop_still_rejects_two_ranks(self):
        with self.assertRaises(ValueError):
            validate_bench_ranks("accelerate_loop", 2)


class TwoRankAuto(unittest.TestCase):
    def test_two_rank_accelerate_auto_labels_its_hard_fails(self):
        # 2-rank bench on accelerate_auto: rank-0 save crashes resume and carry the cause label; the control and the
        # rank-1 points pass. Reproduced locally on main with the guard removed: 8 PASS, 5 HARD_FAIL, all 5 labeled.
        work = Path(tempfile.mkdtemp(prefix="ckptchaos_auto2_")) / "work"
        p = subprocess.run([PY, "-m", "ckpt_chaos", "bench", "accelerate_auto", "--ranks", "2",
                            "--work", str(work), "--jobs", "2"],
                           cwd=ROOT, capture_output=True, text=True, timeout=1500)
        self.assertIn("no crash (control)", p.stdout, p.stdout + p.stderr)
        self.assertIn("HARD_FAIL", p.stdout, p.stdout + p.stderr)
        self.assertIn("PASS", p.stdout, p.stdout + p.stderr)
        rows = [ln for ln in p.stdout.splitlines() if " HARD_FAIL" in ln]  # table rows, not the "HARD_FAIL: n" summary
        self.assertTrue(rows, p.stdout)
        unlabeled = [ln for ln in rows if "[newest checkpoint incomplete]" not in ln]
        self.assertEqual(unlabeled, [], p.stdout)


if __name__ == "__main__":
    unittest.main()
