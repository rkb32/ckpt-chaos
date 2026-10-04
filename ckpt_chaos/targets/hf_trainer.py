"""Target: a tiny GPT-2 trained with HF Trainer on CPU. One phase per child process.

  reference  train 30 steps uninterrupted, save final weights (the ground truth)
  train      train from scratch to step 10, injector armed while checkpoint-10 is saved
  resume     resume_from_checkpoint=True to step 30, record what happened

Runs as 1 process, or as N gloo ranks started by the runner (RANK / WORLD_SIZE in
the environment). The injector is armed through CKPT_CHAOS_* variables; only
CKPT_CHAOS_CRASH_RANK actually crashes, every rank logs its own events.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainerCallback, TrainingArguments

from .common import arm_from_env

SAVE_STEPS = 5
ARM_STEP = 10  # checkpoint-5 already exists; inject while checkpoint-10 is being saved
MAX_STEPS = 30
RANK = int(os.environ.get("RANK", 0))


class DS(torch.utils.data.Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        x = torch.randint(0, 100, (16,), generator=g)
        return {"input_ids": x, "labels": x.clone()}


class Hooks(TrainerCallback):
    def __init__(self, stop_at=None, arm_step=None):
        self.stop_at, self.arm_step, self.started_at = stop_at, arm_step, None

    def on_train_begin(self, args, state, control, **kw):
        self.started_at = state.global_step  # already restored from the checkpoint when resuming

    def on_step_end(self, args, state, control, **kw):
        if self.arm_step and state.global_step == self.arm_step:
            arm_from_env(RANK)
        if self.stop_at and state.global_step == self.stop_at:
            control.should_training_stop = True  # the 30-step LR schedule is kept, like a real kill
        return control


def make_trainer(out, hooks):
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=100, n_positions=16, n_embd=32, n_layer=1, n_head=2))
    args = TrainingArguments(
        output_dir=out, max_steps=MAX_STEPS, save_steps=SAVE_STEPS, per_device_train_batch_size=4,
        use_cpu=True, report_to=[], logging_steps=1000, seed=0, data_seed=0, disable_tqdm=True,
        ddp_timeout=60,  # a rank that lost its peer should fail in a minute, not 30
    )
    return Trainer(model=model, args=args, train_dataset=DS(), callbacks=[hooks])


def save_final(trainer, out):
    torch.save(trainer.model.state_dict(), os.path.join(out, f"final_rank{RANK}.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["reference", "train", "resume"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.phase == "reference":
        t = make_trainer(a.out, Hooks())
        t.train()
        save_final(t, a.out)
    elif a.phase == "train":
        make_trainer(a.out, Hooks(stop_at=ARM_STEP, arm_step=ARM_STEP)).train()
    else:
        hooks = Hooks()
        outcome = {"raised": None, "message": "", "resumed_from_step": None}
        try:
            t = make_trainer(a.out, hooks)
            t.train(resume_from_checkpoint=True)
            outcome["resumed_from_step"] = hooks.started_at
            save_final(t, a.out)
        except Exception as e:  # noqa: BLE001 - every failure mode is a result
            outcome.update(raised=type(e).__name__, message=str(e)[:200])
        with open(os.path.join(a.out, f"outcome_rank{RANK}.json"), "w") as f:
            json.dump(outcome, f)
        if outcome["raised"]:
            sys.exit(3)  # non-zero so the launcher tears the other ranks down, like torchrun would


if __name__ == "__main__":
    main()
