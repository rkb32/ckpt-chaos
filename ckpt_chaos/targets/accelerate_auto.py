"""Target: accelerate_loop using Accelerate's own automatic checkpoint naming. Saves go to project_dir/checkpoints/
checkpoint_K, and resume calls load_state() with no path, so Accelerate picks the folder itself. No resume policy
of this target is involved; a HARD_FAIL here points at Accelerate's load path (or its naming).

The step is recovered from the folder index: save K happens at step (K + 1) * SAVE_EVERY.
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
from accelerate.utils import ProjectConfiguration

from .accelerate_loop import ARM_STEP, MAX_STEPS, SAVE_EVERY, batch, build
from .common import arm_from_env


RANK = int(os.environ.get("RANK", "0"))  # one output file per rank, as the runner expects


def make_acc(out):
    cfg = ProjectConfiguration(project_dir=out, automatic_checkpoint_naming=True)
    return Accelerator(cpu=True, project_config=cfg)


def train(acc, model, opt, start, stop_at=None, arm=False):
    for step in range(start + 1, MAX_STEPS + 1):
        x, y = batch(step)
        opt.zero_grad()
        acc.backward(F.mse_loss(model(x), y))
        opt.step()
        if step % SAVE_EVERY == 0:
            if arm and step == ARM_STEP:
                arm_from_env(RANK)
            acc.save_state()  # automatic naming: project_dir/checkpoints/checkpoint_K
        if stop_at and step == stop_at:
            return


def latest_index(out):
    d = os.path.join(out, "checkpoints")
    idx = [int(m.group(1)) for n in (os.listdir(d) if os.path.isdir(d) else [])
           if (m := re.fullmatch(r"checkpoint_(\d+)", n))]
    return max(idx) if idx else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["reference", "train", "resume"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.phase == "reference":
        acc = make_acc(a.out)
        model, opt = build(acc)
        train(acc, model, opt, 0)
        torch.save(acc.unwrap_model(model).state_dict(), os.path.join(a.out, f"final_rank{RANK}.pt"))
    elif a.phase == "train":
        acc = make_acc(a.out)
        model, opt = build(acc)
        train(acc, model, opt, 0, stop_at=ARM_STEP, arm=True)
    else:
        outcome = {"raised": None, "message": "", "resumed_from_step": None}
        try:
            acc = make_acc(a.out)
            model, opt = build(acc)
            k = latest_index(a.out)
            start = 0
            if k is not None:
                acc.load_state()  # Accelerate picks the latest checkpoint_K itself
                acc.project_configuration.iteration = k + 1  # load_state does not restore it; the next save needs it
                start = (k + 1) * SAVE_EVERY
            outcome["resumed_from_step"] = start
            train(acc, model, opt, start)
            torch.save(acc.unwrap_model(model).state_dict(), os.path.join(a.out, f"final_rank{RANK}.pt"))
        except BaseException as e:  # noqa: BLE001 - every failure mode is a result
            outcome.update(raised=type(e).__name__, message=str(e)[:200])
        with open(os.path.join(a.out, f"outcome_rank{RANK}.json"), "w") as f:
            json.dump(outcome, f)
        if outcome["raised"]:
            sys.exit(3)


if __name__ == "__main__":
    main()
