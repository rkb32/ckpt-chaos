"""A stdlib-only job that catches save errors and carries on, as many loops do with a "best effort" save.

Each save writes ckpt-N.json directly. A full disk raises OSError, which this job logs and ignores, so the
run finishes and reports success with a checkpoint missing. With --strict the error is re-raised.

    ckpt-chaos run --modes enospc --checkpoint-glob "{out}/ckpt-*.json" --step-regex "resumed from step (\d+)" \
        --saved-regex "saved step (\d+)" --result-file "{out}/result.json" -- python examples/swallow_enospc.py --out {out}
"""
import argparse
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--every", type=int, default=5)
ap.add_argument("--strict", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

found = sorted(int(m.group(1)) for n in os.listdir(a.out) if (m := re.fullmatch(r"ckpt-(\d+)\.json", n)))
state = {"step": 0, "acc": 0}
if found:
    with open(os.path.join(a.out, f"ckpt-{found[-1]}.json")) as f:
        state = json.load(f)
print(f"resumed from step {state['step']}", flush=True)

while state["step"] < a.steps:
    state["step"] += 1
    state["acc"] = (state["acc"] * 31 + state["step"]) % 1000003
    if state["step"] % a.every == 0:
        try:
            with open(os.path.join(a.out, f"ckpt-{state['step']}.json"), "w") as f:
                json.dump(state, f)
            print(f"saved step {state['step']}", flush=True)
        except OSError as e:
            if a.strict:
                raise
            print(f"save at step {state['step']} failed, continuing: {e}", flush=True)

with open(os.path.join(a.out, "result.json"), "w") as f:
    json.dump(state, f)
