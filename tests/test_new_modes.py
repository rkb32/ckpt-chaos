"""The modes and checks added after 0.1.3: unsynced writes, torn h5py files, the data-order check, and the DCP target."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run_cli(job: list[str], *extra: str, timeout: int = 600) -> subprocess.CompletedProcess:
    work = Path(tempfile.mkdtemp(prefix="ckptchaos_new_")) / "work"
    return subprocess.run([PY, "-m", "ckpt_chaos", "run", "--work", str(work), "--jobs", "2",
                           "--fail-on", "HARD_FAIL,SILENT_DIVERGENCE,LOST_WORK", *extra, "--", PY, *job, "--out", "{out}"],
                          cwd=ROOT, capture_output=True, text=True, timeout=timeout)


class Unsynced(unittest.TestCase):
    CKPT = ["--checkpoint-glob", "{out}/ckpt-*.json", "--step-regex", r"resumed from step (\d+)",
            "--result-file", "{out}/result.json", "--modes", "unsynced"]

    def test_rename_without_fsync_leaves_a_broken_newest_checkpoint(self):
        p = run_cli(["examples/rotating_json.py"], *self.CKPT)
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("HARD_FAIL", p.stdout)

    def test_fsync_before_rename_survives_the_same_model(self):
        p = run_cli(["examples/rotating_json.py", "--fsync"], *self.CKPT)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertNotIn("HARD_FAIL", p.stdout)


class DataOrder(unittest.TestCase):
    OPTS = ["--result-file", "{out}/result.json", "--step-regex", r"resumed from step (\d+)",
            "--modes", "before,torn"]

    def test_a_job_that_restores_its_data_position_passes(self):
        p = run_cli(["examples/data_cursor_json.py"], *self.OPTS)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_a_job_that_loses_its_data_position_is_silent_divergence(self):
        p = run_cli(["examples/data_cursor_json.py", "--bug"], *self.OPTS)
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("SILENT_DIVERGENCE", p.stdout)
        self.assertNotIn("HARD_FAIL", p.stdout)  # both versions finish: only the result tells them apart


try:
    import h5py  # noqa: F401
    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False


@unittest.skipUnless(HAVE_H5PY, "h5py is not installed (no wheel for this Python yet in CI)")
class H5py(unittest.TestCase):
    def test_a_torn_hdf5_file_breaks_the_resume(self):
        p = run_cli(["examples/h5py_json_ckpt.py"], "--result-file", "{out}/result.json",
                    "--step-regex", r"resumed from step (\d+)", "--modes", "before,torn")
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("h5py.File:state.h5 [torn]", p.stdout)
        self.assertIn("HARD_FAIL", p.stdout)


class DCP(unittest.TestCase):
    def test_dcp_matrix_finds_the_crash_in_the_newest_checkpoint(self):
        work = Path(tempfile.mkdtemp(prefix="ckptchaos_dcp_")) / "work"
        p = subprocess.run([PY, "-m", "ckpt_chaos", "bench", "dcp_ddp", "--work", str(work), "--jobs", "2"],
                           cwd=ROOT, capture_output=True, text=True, timeout=900)
        self.assertIn("HARD_FAIL", p.stdout, p.stdout + p.stderr)
        self.assertIn("PASS", p.stdout)


try:
    import tensorstore  # noqa: F401
    HAVE_TS = True
except ImportError:
    HAVE_TS = False


@unittest.skipUnless(HAVE_TS, "tensorstore is not installed")
class Tensorstore(unittest.TestCase):
    def test_a_torn_chunk_breaks_the_resume_and_a_clean_kill_does_not(self):
        p = run_cli(["examples/tensorstore_ckpt.py"], "--result-file", "{out}/result.json",
                    "--step-regex", r"resumed from step (\d+)", "--modes", "before,torn")
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("tensorstore.write:state.zarr [torn]", p.stdout)
        self.assertIn("HARD_FAIL", p.stdout)
        self.assertIn("tensorstore.write:state.zarr [before]", p.stdout)


os.environ.setdefault("KERAS_BACKEND", "torch")
try:
    import keras  # noqa: F401
    HAVE_KERAS = True
except ImportError:
    HAVE_KERAS = False


@unittest.skipUnless(HAVE_KERAS, "keras is not installed")
class Keras(unittest.TestCase):
    def test_a_torn_keras_archive_breaks_the_resume(self):
        p = run_cli(["examples/keras_ckpt.py"], "--result-file", "{out}/result.json",
                    "--step-regex", r"resumed from step (\d+)", "--modes", "before,torn", "--max-points", "12")
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("open:model.keras [torn]", p.stdout)
        self.assertIn("HARD_FAIL", p.stdout)
        self.assertNotIn("SILENT_DIVERGENCE", p.stdout)  # the control must be deterministic


class Accelerate(unittest.TestCase):
    def test_accelerate_matrix_finds_failures_and_passes(self):
        work = Path(tempfile.mkdtemp(prefix="ckptchaos_acc_")) / "work"
        p = subprocess.run([PY, "-m", "ckpt_chaos", "bench", "accelerate_loop", "--work", str(work), "--jobs", "2"],
                           cwd=ROOT, capture_output=True, text=True, timeout=900)
        self.assertIn("HARD_FAIL", p.stdout, p.stdout + p.stderr)
        self.assertIn("PASS", p.stdout)


if __name__ == "__main__":
    unittest.main()
