"""A plain Hugging Face Trainer job (tiny GPT-2, CPU): saves checkpoint-N every 5 steps and resumes from the newest one.

No ckpt-chaos code in here. The harness only wraps the process and reads the "resumed from step" line.

    ckpt-chaos run --checkpoint-glob "{out}/checkpoint-*" --step-regex "resumed from step (\\d+)" \\
        --result-file "{out}/model.safetensors" --max-points 12 -- python examples/hf_trainer_run.py --out {out}
"""
import argparse

import torch
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint


class DS(torch.utils.data.Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        x = torch.randint(0, 100, (16,), generator=g)
        return {"input_ids": x, "labels": x.clone()}


class ReportStart(TrainerCallback):
    def on_train_begin(self, args, state, control, **kw):
        print(f"resumed from step {state.global_step}", flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=30)
a = ap.parse_args()

torch.manual_seed(0)
model = GPT2LMHeadModel(GPT2Config(vocab_size=100, n_positions=16, n_embd=32, n_layer=1, n_head=2, bos_token_id=0, eos_token_id=0))
args = TrainingArguments(output_dir=a.out, max_steps=a.steps, save_steps=5, per_device_train_batch_size=4, use_cpu=True,
                         report_to=[], logging_steps=1000, seed=0, data_seed=0, disable_tqdm=True)
trainer = Trainer(model=model, args=args, train_dataset=DS(), callbacks=[ReportStart()])
trainer.train(resume_from_checkpoint=get_last_checkpoint(a.out))  # the newest checkpoint-N, or None on a fresh start
trainer.save_model(a.out)
