"""Target: a hand-written DDP training loop with three ways of installing a checkpoint.

Strategy (env CKPT_CHAOS_STRATEGY):
  direct            write straight into checkpoint-N/, state.json last (what most hand-rolled loops do)
  tmp_rename        every rank writes checkpoint-N.tmp/, then every rank renames it
                    (the shape of the temp-dir fix Hugging Face merged and then reverted)
  tmp_rename_rank0  same, but only rank 0 renames, between two barriers (the correct protocol)

Resume picks the highest-numbered checkpoint-N directory, like most scripts do, with no
completeness check. Same CLI and file protocol as hf_trainer.py, so the runner drives either.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import timedelta

import torch
import torch.nn.functional as F
import torch.distributed as dist

from .common import arm_from_env

SAVE_EVERY = 5
ARM_STEP = 10  # checkpoint-5 already exists; inject while checkpoint-10 is being saved
MAX_STEPS = 30
RANK = int(os.environ.get("RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))
STRATEGY = os.environ.get("CKPT_CHAOS_STRATEGY", "direct")


def barrier():
    if WORLD > 1:
        dist.barrier()


def batch(step: int):
    g = torch.Generator().manual_seed(1000 * step + RANK)  # a pure function of (step, rank): resume can replay it
    x = torch.randn(16, 8, generator=g)
    return x, x.sum(1, keepdim=True)


def build():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    wrapped = torch.nn.parallel.DistributedDataParallel(model) if WORLD > 1 else model
    return model, wrapped, opt


def save_checkpoint(out, step, model, opt):
    final = os.path.join(out, f"checkpoint-{step}")
    work = final if STRATEGY == "direct" else final + ".tmp"
    os.makedirs(work, exist_ok=True)
    if RANK == 0:
        torch.save(model.state_dict(), os.path.join(work, "model.pt"))
        torch.save(opt.state_dict(), os.path.join(work, "optim.pt"))
    torch.save({"step": step}, os.path.join(work, f"rank_{RANK}.pt"))  # stands in for per-rank RNG state
    barrier()
    if RANK == 0:
        with open(os.path.join(work, "state.json"), "w") as f:
            json.dump({"step": step}, f)
    barrier()
    if STRATEGY == "tmp_rename":
        os.rename(work, final)  # every rank: the second one finds the directory already gone
    elif STRATEGY == "tmp_rename_rank0":
        if RANK == 0:
            os.rename(work, final)
        barrier()


def latest_checkpoint(out):
    steps = [int(m.group(1)) for d in os.listdir(out) if (m := re.fullmatch(r"checkpoint-(\d+)", d))]
    if not steps:
        raise FileNotFoundError("no checkpoint-N directory to resume from")
    return os.path.join(out, f"checkpoint-{max(steps)}")


def load_checkpoint(ckpt, model, opt) -> int:
    model.load_state_dict(torch.load(os.path.join(ckpt, "model.pt"), weights_only=True))
    opt.load_state_dict(torch.load(os.path.join(ckpt, "optim.pt"), weights_only=True))
    torch.load(os.path.join(ckpt, f"rank_{RANK}.pt"), weights_only=True)
    with open(os.path.join(ckpt, "state.json")) as f:
        return json.load(f)["step"]


def train(out, model, wrapped, opt, start, stop_at=None, arm=False):
    for step in range(start + 1, MAX_STEPS + 1):
        x, y = batch(step)
        opt.zero_grad()
        F.mse_loss(wrapped(x), y).backward()
        opt.step()
        if step % SAVE_EVERY == 0:
            if arm and step == ARM_STEP:
                arm_from_env(RANK)
            save_checkpoint(out, step, model, opt)
        if stop_at and step == stop_at:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["reference", "train", "resume"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if WORLD > 1:
        dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    model, wrapped, opt = build()

    if a.phase == "reference":
        train(a.out, model, wrapped, opt, 0)
        torch.save(model.state_dict(), os.path.join(a.out, f"final_rank{RANK}.pt"))
    elif a.phase == "train":
        train(a.out, model, wrapped, opt, 0, stop_at=ARM_STEP, arm=True)
    else:
        outcome = {"raised": None, "message": "", "resumed_from_step": None}
        try:
            start = load_checkpoint(latest_checkpoint(a.out), model, opt)
            outcome["resumed_from_step"] = start
            train(a.out, model, wrapped, opt, start)
            torch.save(model.state_dict(), os.path.join(a.out, f"final_rank{RANK}.pt"))
        except Exception as e:  # noqa: BLE001 - every failure mode is a result
            outcome.update(raised=type(e).__name__, message=str(e)[:200])
        with open(os.path.join(a.out, f"outcome_rank{RANK}.json"), "w") as f:
            json.dump(outcome, f)
        if outcome["raised"]:
            sys.exit(3)


if __name__ == "__main__":
    main()
