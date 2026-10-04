"""`ckpt-chaos roundtrip`: save a checkpoint, load it straight back, repeat. No crash is involved.

For storage that damages or reorders writes. Lightning-AI/pytorch-lightning#21431 reports about 2 in 48
saves unreadable right after `trainer.save_checkpoint` on Docker for Windows/WSL. Run this with --dir on
each filesystem you care about and compare the failure counts.
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path


def _torch_writer(size_mb: float):
    import torch

    def write(path: Path, i: int, ctx):
        g = torch.Generator().manual_seed(i)
        state = {"w": torch.randn(max(1, int(size_mb * 1024 * 1024 / 4)), generator=g), "step": i}
        torch.save(state, path)
        return state

    def read(path: Path, state):
        got = torch.load(path, weights_only=True)
        if got["step"] != state["step"] or not torch.equal(got["w"], state["w"]):
            raise ValueError("the checkpoint that loaded differs from what was saved")

    return write, read, None


def _lightning_writer(size_mb: float):
    import torch
    import lightning.pytorch as pl

    from .targets.lightning_trainer import Lit, loader

    class Padded(Lit):  # a tiny model grown to roughly size_mb with an unused parameter
        def __init__(self):
            super().__init__()
            self.pad = torch.nn.Parameter(torch.zeros(max(1, int(size_mb * 1024 * 1024 / 4))))

    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(Padded(), loader())

    def write(path: Path, i: int, ctx):
        trainer.save_checkpoint(str(path))

    def read(path: Path, state):
        torch.load(path, map_location="cpu", weights_only=False)  # the issue shows torch.load raising too

    return write, read, None


WRITERS = {"torch": _torch_writer, "lightning": _lightning_writer}


def run_roundtrip(directory: Path, n: int, writer: str = "torch", size_mb: float = 20.0, same_path: bool = True,
                  sleep: float = 0.0) -> tuple[int, Counter]:
    """Returns (unreadable saves, error summary -> count)."""
    directory.mkdir(parents=True, exist_ok=True)
    write, read, ctx = WRITERS[writer](size_mb)
    failures: Counter = Counter()
    for i in range(n):
        path = directory / ("roundtrip.ckpt" if same_path else f"roundtrip_{i}.ckpt")
        state = write(path, i, ctx)
        if sleep:
            time.sleep(sleep)
        try:
            read(path, state)
        except Exception as e:  # noqa: BLE001 - every unreadable save is a result
            failures[f"{type(e).__name__}: {str(e)[:90]}"] += 1
        if not same_path:
            path.unlink(missing_ok=True)
    return sum(failures.values()), failures
