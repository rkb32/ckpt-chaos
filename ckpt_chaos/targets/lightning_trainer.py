"""Target: a tiny Lightning model trained with pl.Trainer on CPU (1 process or N gloo ranks).

Same CLI and file protocol as the other targets. ModelCheckpoint(save_last=True) writes
out/checkpoints/ck-<step>.ckpt plus last.ckpt at every epoch end, and resume is the library's own
trainer.fit(ckpt_path="last"). One epoch is SAVE_EVERY steps, so every checkpoint sits on an epoch
boundary and a resume never has to replay half an epoch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from .common import arm_from_env

SAVE_EVERY = 5  # steps per epoch
ARM_EPOCH = 1  # 0-based: the end of epoch index 1 is step 10; checkpoint at step 5 already exists
MAX_EPOCHS = 6  # 30 steps
RANK = int(os.environ.get("RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))


class Lit(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        return F.mse_loss(self.net(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9)


class Hooks(Callback):
    def __init__(self, stop=False):
        self.stop, self.started_at = stop, None

    def on_train_start(self, trainer, pl_module):
        self.started_at = trainer.global_step  # already restored from the checkpoint when resuming

    def on_train_epoch_end(self, trainer, pl_module):
        # Lightning runs ModelCheckpoint after every other callback, so this arms before the save.
        if trainer.current_epoch == ARM_EPOCH:
            arm_from_env(RANK)
            if self.stop:
                trainer.should_stop = True


def loader():
    g = torch.Generator().manual_seed(123)
    x = torch.randn(SAVE_EVERY * 16 * WORLD, 8, generator=g)  # each rank sees SAVE_EVERY batches per epoch
    return DataLoader(TensorDataset(x, x.sum(1, keepdim=True)), batch_size=16, shuffle=False)


def make(out, hooks):
    ck =ModelCheckpoint(dirpath=os.path.join(out, "checkpoints"), filename="ck-{step}", every_n_epochs=1,
                         save_top_k=-1, save_last=True)
    return pl.Trainer(
        max_epochs=MAX_EPOCHS, accelerator="cpu", devices=WORLD, strategy="ddp" if WORLD > 1 else "auto",
        callbacks=[hooks, ck], enable_progress_bar=False, enable_model_summary=False, logger=False,
        default_root_dir=out,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["reference", "train", "resume"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pl.seed_everything(0, workers=False)  # must come before the model is built, or every process starts differently
    model = Lit()

    if a.phase == "reference":
        make(a.out, Hooks()).fit(model, loader())
        torch.save(model.state_dict(), os.path.join(a.out, f"final_rank{RANK}.pt"))
    elif a.phase == "train":
        make(a.out, Hooks(stop=True)).fit(model, loader())
    else:
        hooks = Hooks()
        outcome = {"raised": None, "message": "", "resumed_from_step": None}
        try:
            make(a.out, hooks).fit(model, loader(), ckpt_path="last")
            outcome["resumed_from_step"] = hooks.started_at
            torch.save(model.state_dict(), os.path.join(a.out, f"final_rank{RANK}.pt"))
        except Exception as e:  # noqa: BLE001 - every failure mode is a result
            outcome.update(raised=type(e).__name__, message=str(e)[:200])
        with open(os.path.join(a.out, f"outcome_rank{RANK}.json"), "w") as f:
            json.dump(outcome, f)
        if outcome["raised"]:
            sys.exit(3)


if __name__ == "__main__":
    main()
