"""A multi-rank job (torch.distributed, gloo, CPU): every step all-reduces a value across the ranks, then each rank
saves its state to state_rank<r>.json every 5 steps and resumes from it. Rank 0 writes result.json.

Because every step is a collective, a rank that dies leaves its peers waiting: the harness must stop them, which
is what the multi-rank path of ckpt-chaos run does after its peer grace period.

    ckpt-chaos run --ranks 2 --result-file "{out}/result.json" --step-regex "resumed from step (\\d+)" \\
        --fail-on HARD_FAIL,SILENT_DIVERGENCE,LOST_WORK -- python examples/gloo_allreduce_json.py --out {out}
"""
import argparse
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--every", type=int, default=5)
a = ap.parse_args()

RANK = int(os.environ.get("RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))
dist.init_process_group("gloo", timeout=timedelta(seconds=60))
os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, f"state_rank{RANK}.json")

state = {"step": 0, "acc": 0}
if os.path.exists(path):
    with open(path) as f:
        state = json.load(f)
print(f"resumed from step {state['step']}", flush=True)

while state["step"] < a.steps:
    state["step"] += 1
    local = torch.tensor([float(RANK * 1000 + state["step"])])
    dist.all_reduce(local)  # the collective: every rank waits for every other rank here
    state["acc"] = (state["acc"] * 31 + int(local.item())) % 1000003
    if state["step"] % a.every == 0:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, path)

if RANK == 0:
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump({"acc": state["acc"], "world": WORLD}, f)
dist.barrier()
dist.destroy_process_group()
