"""A stdlib-only job that keeps one checkpoint file per save (ckpt-5.json, ckpt-10.json, ...).

Every save is atomic (temp file, then rename), so a crash never leaves a torn checkpoint. The default
resume loads the newest ckpt-N.json. With --bug the resume only trusts checkpoints that have a
ckpt-N.json.done marker next to them, and nothing ever writes that marker (think of an async uploader
that never reports back): the job silently starts over from step 0 and no error is raised.

    ckpt-chaos run --checkpoint-glob "{out}/ckpt-*.json" --step-regex "resumed from step (\\d+)" \\
        --result-file "{out}/result.json" -- python examples/rotating_json.py --out {out} --bug
"""
import argparse
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--every", type=int, default=5)
ap.add_argument("--bug", action="store_true")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)

trusted = {}
for name in os.listdir(a.out):
    m = re.fullmatch(r"ckpt-(\d+)\.json", name)
    if m and (not a.bug or os.path.exists(os.path.join(a.out, name + ".done"))):
        trusted[int(m.group(1))] = name

state = {"step": 0, "acc": 0}
if trusted:
    with open(os.path.join(a.out, trusted[max(trusted)])) as f:
        state = json.load(f)
print(f"resumed from step {state['step']}", flush=True)

while state["step"] < a.steps:
    state["step"] += 1
    state["acc"] = (state["acc"] * 31 + state["step"]) % 1000003  # the "model update"
    if state["step"] % a.every == 0:
        path = os.path.join(a.out, f"ckpt-{state['step']}.json")
        with open(path + ".tmp", "w") as f:
            json.dump(state, f)
        os.replace(path + ".tmp", path)

with open(os.path.join(a.out, "result.json"), "w") as f:
    json.dump(state, f)
