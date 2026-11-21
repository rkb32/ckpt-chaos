"""Target: accelerate_loop with a guarded resume. Tries checkpoint-N folders newest first, each with a fresh
Accelerator and model; a folder whose load raises is skipped, and with none loadable the run starts from step 0.

Purpose: separate a bug in the target's resume policy (picks the newest folder with no check) from a bug in
Accelerate's load path. Same phases and CLI as accelerate_loop.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from .accelerate_loop import ARM_STEP, MAX_STEPS, SAVE_EVERY, batch, build, save_checkpoint
from .common import arm_from_env


def train(acc, out, model, opt, start, stop_at=None, arm=False):
    for step in range(start + 1, MAX_STEPS + 1):
        x, y = batch(step)
        opt.zero_grad()
        acc.backward(F.mse_loss(model(x), y))
        opt.step()
        if step % SAVE_EVERY == 0:
            if arm and step == ARM_STEP:
                arm_from_env(0)
            save_checkpoint(acc, out, step)
        if stop_at and step == stop_at:
            return


def guarded_resume(out):
    """Return (acc, model, opt, start_step, skipped), trying each checkpoint newest first."""
    steps = sorted((int(m.group(1)) for d in os.listdir(out) if (m := re.fullmatch(r"checkpoint-(\d+)", d))),
                   reverse=True)
    skipped = []
    for s in steps:
        acc = Accelerator(cpu=True)
        model, opt = build(acc)
        try:
            acc.load_state(os.path.join(out, f"checkpoint-{s}"))
            return acc, model, opt, s, skipped
        except BaseException as e:  # noqa: BLE001 - a bad folder is skipped, the reason is recorded
            skipped.append(f"checkpoint-{s}: {type(e).__name__}")
    acc = Accelerator(cpu=True)
    model, opt = build(acc)
    return acc, model, opt, 0, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["reference", "train", "resume"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.phase == "reference":
        acc = Accelerator(cpu=True)
        model, opt = build(acc)
        train(acc, a.out, model, opt, 0)
        torch.save(acc.unwrap_model(model).state_dict(), os.path.join(a.out, "final_rank0.pt"))
    elif a.phase == "train":
        acc = Accelerator(cpu=True)
        model, opt = build(acc)
        train(acc, a.out, model, opt, 0, stop_at=ARM_STEP, arm=True)
    else:
        outcome = {"raised": None, "message": "", "resumed_from_step": None, "skipped": []}
        try:
            acc, model, opt, start, skipped = guarded_resume(a.out)
            outcome.update(resumed_from_step=start, skipped=skipped)
            train(acc, a.out, model, opt, start)
            torch.save(acc.unwrap_model(model).state_dict(), os.path.join(a.out, "final_rank0.pt"))
        except BaseException as e:  # noqa: BLE001 - every failure mode is a result
            outcome.update(raised=type(e).__name__, message=str(e)[:200])
        with open(os.path.join(a.out, "outcome_rank0.json"), "w") as f:
            json.dump(outcome, f)
        if outcome["raised"]:
            sys.exit(3)


if __name__ == "__main__":
    main()
