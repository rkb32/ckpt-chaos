"""Target: a training loop with FSDP (fully sharded) that saves with torch.distributed.checkpoint (DCP).

Same phases and CLI as dcp_ddp.py, so the runner drives it unchanged. The model is wrapped in FSDP
(FULL_SHARD, CPU/gloo), so each rank holds only its shard of the weights and optimizer state. Save and
load go through DCP's state_dict helpers, which return the sharded state each rank must write and read.
Resume picks the highest-numbered checkpoint-N directory with no completeness check, like most scripts do.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from .common import arm_from_env

SAVE_EVERY = 5
ARM_STEP = 10  # checkpoint-5 already exists; inject while checkpoint-10 is being saved
MAX_STEPS = 30
RANK = int(os.environ.get("RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))
OPTS = StateDictOptions(full_state_dict=False, cpu_offload=True, flatten_optimizer_state_dict=True)


def batch(step: int):
    g = torch.Generator().manual_seed(1000 * step + RANK)
    x = torch.randn(16, 8, generator=g)
    return x, x.sum(1, keepdim=True)


def build():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
    if WORLD > 1:
        # device_id=cpu: without it torch 2.x refuses to run FSDP on a CPU-only machine
        model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True,
                     device_id=torch.device("cpu"))
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    return model, opt


def state_of(model, opt, step):
    return {
        "model": get_model_state_dict(model, options=OPTS),
        "optim": get_optimizer_state_dict(model, opt, options=OPTS),
        "step": torch.tensor(step),
    }


def save_checkpoint(out, step, model, opt):
    dcp.save(state_of(model, opt, step), checkpoint_id=os.path.join(out, f"checkpoint-{step}"))


def latest_checkpoint(out):
    steps = [int(m.group(1)) for d in os.listdir(out) if (m := re.fullmatch(r"checkpoint-(\d+)", d))]
    if not steps:
        raise FileNotFoundError("no checkpoint-N directory to resume from")
    return os.path.join(out, f"checkpoint-{max(steps)}")


def load_checkpoint(ckpt, model, opt) -> int:
    for p in model.parameters():  # SGD builds momentum buffers on the first step; DCP then overwrites them
        p.grad = torch.zeros_like(p)
    opt.step()
    for p in model.parameters():
        p.grad = None
    state = state_of(model, opt, 0)
    dcp.load(state, checkpoint_id=ckpt)
    set_model_state_dict(model, state["model"], options=OPTS)
    set_optimizer_state_dict(model, opt, state["optim"], options=OPTS)
    return int(state["step"])


def train(out, model, opt, start, stop_at=None, arm=False):
    for step in range(start + 1, MAX_STEPS + 1):
        x, y = batch(step)
        opt.zero_grad()
        F.mse_loss(model(x), y).backward()
        opt.step()
        if step % SAVE_EVERY == 0:
            if arm and step == ARM_STEP:
                arm_from_env(RANK)
            save_checkpoint(out, step, model, opt)
        if stop_at and step == stop_at:
            return


def final_weights(model):
    # Full weights for the comparison: gather the shards only at the end of the run.
    if WORLD > 1:
        with FSDP.summon_full_params(model, writeback=False):
            return {k: v.detach().clone() for k, v in model.module.state_dict().items()}
    return model.state_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["reference", "train", "resume"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if WORLD > 1:
        dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    model, opt = build()

    if a.phase == "reference":
        train(a.out, model, opt, 0)
        torch.save(final_weights(model), os.path.join(a.out, f"final_rank{RANK}.pt"))
    elif a.phase == "train":
        train(a.out, model, opt, 0, stop_at=ARM_STEP, arm=True)
    else:
        outcome = {"raised": None, "message": "", "resumed_from_step": None}
        try:
            start = load_checkpoint(latest_checkpoint(a.out), model, opt)
            outcome["resumed_from_step"] = start
            train(a.out, model, opt, start)
            torch.save(final_weights(model), os.path.join(a.out, f"final_rank{RANK}.pt"))
        except BaseException as e:  # noqa: BLE001 - every failure mode is a result
            import traceback
            traceback.print_exc()
            outcome.update(raised=type(e).__name__, message=str(e)[:200])
        with open(os.path.join(a.out, f"outcome_rank{RANK}.json"), "w") as f:
            json.dump(outcome, f)
        if outcome["raised"]:
            sys.exit(3)


if __name__ == "__main__":
    main()
