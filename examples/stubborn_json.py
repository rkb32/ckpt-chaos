"""A stdlib-only job that ignores SIGTERM and takes a while to finish: the case for the grace period.

It saves state.json every 5 steps (the same non-atomic save as naive_json.py) and sleeps 0.2 s per step.
With `--modes term` ckpt-chaos sends it SIGTERM at each file event. It ignores the signal, keeps running,
and gets SIGKILLed when the grace period runs out (CKPT_CHAOS_GRACE_S, default 5 s).

    CKPT_CHAOS_GRACE_S=1 ckpt-chaos run --modes term --checkpoint-glob "{out}/state.json" \\
        --step-regex "resumed from step (\\d+)" -- python examples/stubborn_json.py --out {out}
"""
import argparse
import json
import os
import signal
import time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--every", type=int, default=5)
a = ap.parse_args()

signal.signal(signal.SIGTERM, lambda *_: print("got SIGTERM, ignoring it", flush=True))

os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, "state.json")
state = {"step": 0, "acc": 0}
if os.path.exists(path):
    with open(path) as f:
        state = json.load(f)
print(f"resumed from step {state['step']}", flush=True)

while state["step"] < a.steps:
    state["step"] += 1
    state["acc"] = (state["acc"] * 31 + state["step"]) % 1000003
    time.sleep(0.2)
    if state["step"] % a.every == 0:
        with open(path, "w") as f:
            json.dump(state, f)

with open(os.path.join(a.out, "result.json"), "w") as f:
    json.dump(state, f)
