"""The injector must stop a save at exactly the Nth file event and leave exactly the state a kill would."""
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOT = (
    "import sys,runpy; import ckpt_chaos.inject as i; "
    "i.arm(sys.argv[1], int(sys.argv[2]) or None, sys.argv[3], sys.argv[4]); "
    "sys.argv=[sys.argv[5]]; runpy.run_path(sys.argv[0], run_name='__main__')"
)

SAVE_LOOP = """
    import os
    d = r"{watch}"
    for name in ("model.bin", "optimizer.pt", "state.json"):
        with open(os.path.join(d, name + ".tmp"), "w") as f:
            f.write("x" * 10)
        os.replace(os.path.join(d, name + ".tmp"), os.path.join(d, name))
"""

NATIVE_SAVE = """
    import os, torch
    torch.save({{"w": torch.zeros(4000)}}, os.path.join(r"{watch}", "weights.pt"))
"""


class Harness(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix="ckptchaos_test_"))
        self.watch = self.work / "out"
        self.watch.mkdir()

    def run_child(self, body: str, crash_at: int = 0, mode: str = "before"):
        script = self.work / "job.py"
        script.write_text(textwrap.dedent(body.format(watch=self.watch)))
        for f in self.watch.iterdir():
            f.unlink()
        log = self.work / "events.log"
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        cmd = [sys.executable, "-c", BOOT, str(self.watch), str(crash_at), mode, str(log), str(script)]
        p = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True)
        events = log.read_text().strip().splitlines() if log.exists() else []
        return p.returncode, sorted(f.name for f in self.watch.iterdir()), events

    def test_dry_run_counts_every_mutation(self):
        rc, files, events = self.run_child(SAVE_LOOP)
        self.assertEqual((rc, files, len(events)), (0, ["model.bin", "optimizer.pt", "state.json"], 6))

    def test_crash_before_a_rename_leaves_the_temp_file(self):
        rc, files, _ = self.run_child(SAVE_LOOP, crash_at=4)  # 4th event = rename of optimizer.pt.tmp
        self.assertEqual((rc, files), (137, ["model.bin", "optimizer.pt.tmp"]))

    def test_crash_before_open_leaves_nothing_new(self):
        rc, files, _ = self.run_child(SAVE_LOOP, crash_at=3)  # 3rd event = open of optimizer.pt.tmp
        self.assertEqual((rc, files), (137, ["model.bin"]))

    def test_native_writer_is_counted_and_can_be_torn(self):
        rc, files, events = self.run_child(NATIVE_SAVE)
        full = (self.watch / "weights.pt").stat().st_size
        self.assertEqual((rc, files, len(events)), (0, ["weights.pt"], 1))

        rc, files, _ = self.run_child(NATIVE_SAVE, crash_at=1, mode="torn")
        self.assertEqual((rc, files), (137, ["weights.pt"]))
        self.assertLess((self.watch / "weights.pt").stat().st_size, full)

        rc, files, _ = self.run_child(NATIVE_SAVE, crash_at=1, mode="before")
        self.assertEqual((rc, files), (137, []))


if __name__ == "__main__":
    unittest.main()
