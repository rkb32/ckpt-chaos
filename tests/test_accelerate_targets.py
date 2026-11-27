import tempfile
import unittest
import zipfile
from pathlib import Path

from ckpt_chaos.runner import _auto_newest, _is_complete, _latest_incomplete


class AutoNewest(unittest.TestCase):
    """accelerate_auto keeps checkpoints in checkpoints/checkpoint_K; K is the save at step (K + 1) * SAVE_EVERY."""

    def test_newest_complete_folder_maps_to_its_step(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            for k, size in [(0, 4), (1, 4), (2, 2)]:  # checkpoint_2 is short: incomplete
                ck = out / "checkpoints" / f"checkpoint_{k}"
                ck.mkdir(parents=True)
                (ck / "a.bin").write_bytes(b"x" * size)
            healthy = {"a.bin": 4}
            self.assertEqual(_auto_newest(out, healthy), (1 + 1) * 5)  # SAVE_EVERY is 5

    def test_no_complete_folder_is_step_zero(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(_auto_newest(Path(d), {"a.bin": 4}), 0)


class RngStateComplete(unittest.TestCase):
    """random_states_*.pkl is a zip container: complete if intact, never unpickled."""

    def test_intact_zip_is_complete(self):
        with tempfile.TemporaryDirectory() as d:
            ck = Path(d)
            with zipfile.ZipFile(ck / "random_states_0.pkl", "w") as z:
                z.writestr("data.pkl", b"not really a pickle, and never loaded")
            self.assertTrue(_is_complete(ck, {"random_states_0.pkl": 0}))

    def test_truncated_zip_is_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            ck = Path(d)
            p = ck / "random_states_0.pkl"
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("data.pkl", b"x" * 2000)
            with open(p, "r+b") as f:
                f.truncate(p.stat().st_size // 2)
            self.assertFalse(_is_complete(ck, {"random_states_0.pkl": 0}))

    def test_missing_file_is_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(_is_complete(Path(d), {"random_states_0.pkl": 0}))


class LatestIncomplete(unittest.TestCase):
    """The folder accelerate_auto resumes from is the highest checkpoint_K, with no completeness check."""

    HEALTHY = {"model.safetensors": 4, "optimizer.bin": 4}

    def _make(self, out, k, files):
        ck = out / "checkpoints" / f"checkpoint_{k}"
        ck.mkdir(parents=True)
        for name, size in files.items():
            (ck / name).write_bytes(b"x" * size)

    def test_half_written_newest_is_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self._make(out, 0, self.HEALTHY)
            self._make(out, 1, {"model.safetensors": 4})  # optimizer.bin missing
            self.assertTrue(_latest_incomplete(out, self.HEALTHY))

    def test_complete_newest_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self._make(out, 0, {"model.safetensors": 4})  # older is broken, newest is fine
            self._make(out, 1, self.HEALTHY)
            self.assertFalse(_latest_incomplete(out, self.HEALTHY))

    def test_no_checkpoints_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(_latest_incomplete(Path(d), self.HEALTHY))


if __name__ == "__main__":
    unittest.main()
