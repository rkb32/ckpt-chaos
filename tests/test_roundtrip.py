import tempfile
import unittest
from pathlib import Path

from ckpt_chaos.roundtrip import run_roundtrip


class Roundtrip(unittest.TestCase):
    def test_healthy_filesystem_has_no_unreadable_saves(self):
        d = Path(tempfile.mkdtemp(prefix="ckptchaos_rt_"))
        failed, errors = run_roundtrip(d, n=6, writer="torch", size_mb=0.2)
        self.assertEqual((failed, dict(errors)), (0, {}))

    def test_a_damaged_read_is_reported_not_raised(self):
        import ckpt_chaos.roundtrip as rt

        def writer(size_mb):
            def write(path, i, ctx):
                path.write_bytes(b"truncated")
                return None

            def read(path, state):
                import torch

                torch.load(path, weights_only=True)

            return write, read, None

        rt.WRITERS["broken"] = writer
        try:
            d = Path(tempfile.mkdtemp(prefix="ckptchaos_rt_"))
            failed, errors = run_roundtrip(d, n=3, writer="broken")
            self.assertEqual(failed, 3)
            self.assertEqual(len(errors), 1)
        finally:
            del rt.WRITERS["broken"]


if __name__ == "__main__":
    unittest.main()
