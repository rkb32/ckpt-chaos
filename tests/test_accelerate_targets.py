import tempfile
import unittest
import zipfile
from pathlib import Path

from ckpt_chaos.runner import _auto_newest, _is_complete


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


if __name__ == "__main__":
    unittest.main()
