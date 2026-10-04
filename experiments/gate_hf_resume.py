"""Gate test: does HF Trainer survive a checkpoint left broken by a kill?

Trains a tiny GPT-2 on CPU, then damages the newest checkpoint in several ways
a mid-save kill could produce, resumes with resume_from_checkpoint=True, and
compares the final weights with an intact resume.
"""
import os
import shutil
import sys
import tempfile

import torch
import transformers
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

print("transformers", transformers.__version__, "| torch", torch.__version__, "| platform", sys.platform)


class DS(torch.utils.data.Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        x = torch.randint(0, 100, (16,), generator=g)
        return {"input_ids": x, "labels": x.clone()}


class StopAt(TrainerCallback):
    """Stop after a save at `step` while keeping the 30-step LR schedule (a real kill would)."""

    def __init__(self, step):
        self.step = step

    def on_step_end(self, args, state, control, **kw):
        if state.global_step == self.step:
            control.should_training_stop = True
        return control


def make_trainer(out, max_steps, stop_at=None):
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=100, n_positions=16, n_embd=32, n_layer=1, n_head=2)
    model = GPT2LMHeadModel(cfg)
    args = TrainingArguments(
        output_dir=out, max_steps=max_steps, save_steps=5, per_device_train_batch_size=4,
        use_cpu=True, report_to=[], logging_steps=1000, seed=0, data_seed=0, disable_tqdm=True,
    )
    cbs = [StopAt(stop_at)] if stop_at else None
    return Trainer(model=model, args=args, train_dataset=DS(), callbacks=cbs)


def weights(trainer):
    return {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}


def max_diff(a, b):
    return max((a[k] - b[k]).abs().max().item() for k in a)


root = tempfile.mkdtemp(prefix="ckptchaos_")
print("workdir", root)

# Reference 1: uninterrupted 30-step run. Reference 2: clean 20-step run, then intact resume to 30.
t = make_trainer(os.path.join(root, "uninterrupted"), 30); t.train(); w_uninterrupted = weights(t)
run20 = os.path.join(root, "run20")
make_trainer(run20, 30, stop_at=20).train()
print("checkpoints written:", sorted(d for d in os.listdir(run20) if d.startswith("checkpoint")))

intact = os.path.join(root, "intact"); shutil.copytree(run20, intact)
t = make_trainer(intact, 30); t.train(resume_from_checkpoint=True); w_intact = weights(t)
print(f"intact resume vs uninterrupted: max weight diff = {max_diff(w_uninterrupted, w_intact):.3e}")


def rm(ckpt, name):
    os.remove(os.path.join(ckpt, name))


def truncate(ckpt, name):
    p = os.path.join(ckpt, name)
    size = os.path.getsize(p)
    with open(p, "r+b") as f:
        f.truncate(size // 2)


def empty_dir(ckpt):
    for n in os.listdir(ckpt):
        os.remove(os.path.join(ckpt, n))


CASES = {
    "no trainer_state.json": lambda c: rm(c, "trainer_state.json"),
    "no optimizer.pt": lambda c: rm(c, "optimizer.pt"),
    "no rng_state.pth": lambda c: rm(c, "rng_state.pth"),
    "truncated optimizer.pt": lambda c: truncate(c, "optimizer.pt"),
    "truncated model.safetensors": lambda c: truncate(c, "model.safetensors"),
    "empty checkpoint dir": empty_dir,
}

print("\nfiles in a healthy checkpoint:", sorted(os.listdir(os.path.join(run20, "checkpoint-20"))))
print(f"\n{'fault in checkpoint-20':<30}{'auto-resume picks':<20}outcome")
for name, damage in CASES.items():
    d = os.path.join(root, "case_" + str(abs(hash(name))))
    shutil.copytree(run20, d)
    try:
        damage(os.path.join(d, "checkpoint-20"))
    except FileNotFoundError:
        print(f"{name:<30}{'-':<20}SKIPPED (file not written by this version)")
        continue
    picked = os.path.basename(get_last_checkpoint(d) or "none")
    try:
        t = make_trainer(d, 30); t.train(resume_from_checkpoint=True)
        diff = max_diff(w_intact, weights(t))
        verdict = "resumed, weights IDENTICAL to intact" if diff == 0 else f"resumed SILENTLY, weights DIFFER (max diff {diff:.2e})"
    except Exception as e:  # noqa: BLE001 - we want every failure mode
        verdict = f"RAISED {type(e).__name__}: {str(e)[:70]}"
    print(f"{name:<30}{picked:<20}{verdict}")

shutil.rmtree(root, ignore_errors=True)
