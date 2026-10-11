"""Target: a training loop that saves with torch.distributed.checkpoint (DCP), one directory per save.

Save: dcp.save(state, checkpoint_id=checkpoint-N/) writes shard files, then the .metadata file.
Resume: picks the highest-numbered checkpoint-N directory, like most scripts do, with no completeness
check, then dcp.load() into the model and optimizer. Same phases and CLI as loop_ddp.py, so the runner
drives it unchanged. Runs on one process here (WORLD_SIZE=1); DCP's own multi-rank path is not covered yet.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F

from .common import arm_from_env

SAVE_EVERY = 5
ARM_STEP = 10  # checkpoint-5 already exists; inject while checkpoint-10 is being saved
MAX_STEPS = 30


def batch(step: int):
    g = torch.Generator().manual_seed(1000 * step)  # a pure function of the step: resume can replay it
    x = torch.randn(16, 8, generator=g)
    return x, x.sum(1, keepdim=True)


def build():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    return model, opt


def state_of(model, opt, step):
    return {"model": model.state_dict(), "optim": opt.state_dict(), "step": torch.tensor(step)}


def save_checkpoint(out, step, model, opt):
    dcp.save(state_of(model, opt, step), checkpoint_id=os.path.join(out, f"checkpoint-{step}"))


def latest_checkpoint(out):
    steps = [int(m.group(1)) for d in os.listdir(out) if (m := re.fullmatch(r"checkpoint-(\d+)", d))]
    if not steps:
        raise FileNotFoundError("no checkpoint-N directory to resume from")
    return os.path.join(out, f"checkpoint-{max(steps)}")


def load_checkpoint(ckpt, model, opt) -> int:
    for p in model.parameters():  # SGD makes its momentum buffers on the first step; dcp.load then overwrites them
        p.grad = torch.zeros_like(p)
    opt.step()
    for p in model.parameters():
        p.grad = None
    state = state_of(model, opt, 0)
    dcp.load(state, checkpoint_id=ckpt)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["optim"])
    return int(state["step"])


def train(out, model, opt, start, stop_at=None, arm=False):
    for step in range(start + 1, MAX_STEPS + 1):
        x, y = batch(step)
        opt.zero_grad()
        F.mse_loss(model(x), y).backward()
        opt.step()
        if step % SAVE_EVERY == 0:
            if arm and step == ARM_STEP:
                arm_from_env(0)
            save_checkpoint(out, step, model, opt)
        if stop_at and step == stop_at:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["reference", "train", "resume"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model, opt = build()

    if a.phase == "reference":
        train(a.out, model, opt, 0)
        torch.save(model.state_dict(), os.path.join(a.out, "final_rank0.pt"))
    elif a.phase == "train":
        train(a.out, model, opt, 0, stop_at=ARM_STEP, arm=True)
    else:
        outcome = {"raised": None, "message": "", "resumed_from_step": None}
        try:
            start = load_checkpoint(latest_checkpoint(a.out), model, opt)
            outcome["resumed_from_step"] = start
            train(a.out, model, opt, start)
            torch.save(model.state_dict(), os.path.join(a.out, "final_rank0.pt"))
        except BaseException as e:  # noqa: BLE001 - every failure mode is a result; DCP's CheckpointException is a BaseException
            outcome.update(raised=type(e).__name__, message=str(e)[:200])
        with open(os.path.join(a.out, "outcome_rank0.json"), "w") as f:
            json.dump(outcome, f)
        if outcome["raised"]:
            sys.exit(3)


if __name__ == "__main__":
    main()
