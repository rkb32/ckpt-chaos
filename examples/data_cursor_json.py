"""A stdlib-only job whose result depends on the order its data is read in: the data-pipeline check.

Each step reads the next sample from a shuffled order. The read position (the cursor) is saved with the
checkpoint, so a resume continues from the same sample. With --bug the cursor is not restored (the step is),
so the resumed job reads samples from the start of the order again. Both versions finish, and both write a
result, so only the result comparison can tell them apart: that is the SILENT_DIVERGENCE this check exists for.

    ckpt-chaos run --result-file "{out}/result.json" --step-regex "resumed from step (\\d+)" \\
        -- python examples/data_cursor_json.py --out {out} [--bug]
"""
import argparse
import json
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--every", type=int, default=5)
ap.add_argument("--bug", action="store_true")
a = ap.parse_args()

SAMPLES = 100
order = list(range(SAMPLES))
random.Random(7).shuffle(order)  # the data order: a pure function of the seed

os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, "state.json")
state = {"step": 0, "cursor": 0, "acc": 0}
if os.path.exists(path):
    with open(path) as f:
        state = json.load(f)
    if a.bug:
        state["cursor"] = 0  # the bug: the step is restored, the position in the data is not
print(f"resumed from step {state['step']}", flush=True)

while state["step"] < a.steps:
    state["step"] += 1
    sample = order[state["cursor"] % SAMPLES]  # the batch this step reads
    state["cursor"] += 1
    state["acc"] = (state["acc"] * 31 + sample * state["step"]) % 1000003  # the "model update"
    if state["step"] % a.every == 0:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, path)  # atomic: the only failure left is the data order, not a torn file

with open(os.path.join(a.out, "result.json"), "w") as f:
    json.dump({"acc": state["acc"]}, f)
