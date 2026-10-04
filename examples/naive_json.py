"""A stdlib-only "training" script: no torch, no framework. Checkpoints are JSON.

It resumes from <out>/state.json. By default it saves the way most scripts do (open the file and write
over the old one). With --atomic it writes a temp file and renames it into place.

    ckpt-chaos run --step-regex "resumed from step (\\d+)" --result-file "{out}/result.json" \\
        -- python examples/naive_json.py --out {out}
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--every", type=int, default=5)
ap.add_argument("--atomic", action="store_true")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, "state.json")
state = {"step": 0, "acc": 0}
if os.path.exists(path):
    with open(path) as f:
        state = json.load(f)
print(f"resumed from step {state['step']}", flush=True)

while state["step"] < a.steps:
    state["step"] += 1
    state["acc"] = (state["acc"] * 31 + state["step"]) % 1000003  # the "model update"
    if state["step"] % a.every == 0:
        if a.atomic:
            with open(path + ".tmp", "w") as f:
                json.dump(state, f)
            os.replace(path + ".tmp", path)
        else:
            with open(path, "w") as f:
                json.dump(state, f)

with open(os.path.join(a.out, "result.json"), "w") as f:
    json.dump(state, f)
