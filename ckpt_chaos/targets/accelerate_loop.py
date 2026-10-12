"""Target: a training loop that saves with Hugging Face accelerate's Accelerator.save_state, one folder per save.

accelerate writes in place (no temp folder, no rename, no fsync). Resume picks the highest-numbered checkpoint-N
folder, like most scripts do, with no completeness check, then Accelerator.load_state(). Same phases and CLI as
loop_ddp.py, so the runner drives it unchanged. Single process on CPU here.
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

from .common import arm_from_env

SAVE_EVERY = 5
ARM_STEP = 10  # checkpoint-5 already exists; inject while checkpoint-10 is being saved
MAX_STEPS = 30


def batch(step: int):
    g = torch.Generator().manual_seed(1000 * step)  # a pure function of the step: resume can replay it
    x = torch.randn(16, 8, generator=g)
    return x, x.sum(1, keepdim=True)


def build(acc: Accelerator):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    return acc.prepare(model, opt)


def save_checkpoint(acc: Accelerator, out, step):
    acc.save_state(os.path.join(out, f"checkpoint-{step}"))


def latest_checkpoint(out):
    steps = [int(m.group(1)) for d in os.listdir(out) if (m := re.fullmatch(r"checkpoint-(\d+)", d))]
    if not steps:
        raise FileNotFoundError("no checkpoint-N folder to resume from")
    return os.path.join(out, f"checkpoint-{max(steps)}")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["reference", "train", "resume"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    acc = Accelerator(cpu=True)
    model, opt = build(acc)

    if a.phase == "reference":
        train(acc, a.out, model, opt, 0)
        torch.save(acc.unwrap_model(model).state_dict(), os.path.join(a.out, "final_rank0.pt"))
    elif a.phase == "train":
        train(acc, a.out, model, opt, 0, stop_at=ARM_STEP, arm=True)
    else:
        outcome = {"raised": None, "message": "", "resumed_from_step": None}
        try:
            ckpt = latest_checkpoint(a.out)
            acc.load_state(ckpt)
            start = int(re.fullmatch(r".*checkpoint-(\d+)", ckpt).group(1))
            outcome["resumed_from_step"] = start
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
