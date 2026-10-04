"""A deliberately ordinary PyTorch training script, the kind found in thousands of repos.

It resumes from <out>/ckpt.pt if that exists. By default it saves the way most scripts do (torch.save
straight over the old file). With --atomic it writes a temp file and renames it into place.

    ckpt-chaos run --step-regex "resumed from step (\\d+)" --result-file "{out}/final.pt" \\
        -- python examples/naive_torch.py --out {out}
    ckpt-chaos run ... -- python examples/naive_torch.py --out {out} --atomic
"""
import argparse
import os

import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--every", type=int, default=5)
ap.add_argument("--atomic", action="store_true")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
ckpt = os.path.join(a.out, "ckpt.pt")

step = 0
if os.path.exists(ckpt):
    state = torch.load(ckpt, weights_only=True)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    step = state["step"]
print(f"resumed from step {step}", flush=True)

while step < a.steps:
    step += 1
    x = torch.randn(16, 8, generator=torch.Generator().manual_seed(step))
    opt.zero_grad()
    F.mse_loss(model(x), x.sum(1, keepdim=True)).backward()
    opt.step()
    if step % a.every == 0:
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step}
        if a.atomic:
            torch.save(state, ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
        else:
            torch.save(state, ckpt)

torch.save(model.state_dict(), os.path.join(a.out, "final.pt"))
