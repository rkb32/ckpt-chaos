"""A stdlib-only job that keeps only the newest checkpoint (keep-last-1 rotation, like save_total_limit=1).

Correct order: write the new checkpoint (temp file, then rename), then delete the old one. A crash at any
point leaves at least one complete checkpoint.
With --bug the order is reversed: delete the old checkpoint first, then write the new one. A crash between
the two leaves no checkpoint at all. The resume then finds nothing and starts from step 0 with no error,
which the judge reports as LOST_WORK.

    ckpt-chaos run --checkpoint-glob "{out}/ckpt-*.json" --step-regex "resumed from step (\\d+)" \\
        --saved-regex "saved step (\d+)" --result-file "{out}/result.json" -- python examples/rotate_delete.py --out {out} --bug
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

found = sorted(int(m.group(1)) for n in os.listdir(a.out) if (m := re.fullmatch(r"ckpt-(\d+)\.json", n)))
state = {"step": 0, "acc": 0}
if found:
    with open(os.path.join(a.out, f"ckpt-{found[-1]}.json")) as f:
        state = json.load(f)
print(f"resumed from step {state['step']}", flush=True)


def old_checkpoints():
    return [os.path.join(a.out, n) for n in os.listdir(a.out) if re.fullmatch(r"ckpt-\d+\.json", n)]


def save(step):
    path = os.path.join(a.out, f"ckpt-{step}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(state, f)
    os.replace(path + ".tmp", path)
    print(f"saved step {step}", flush=True)  # reported only after the save returned


while state["step"] < a.steps:
    state["step"] += 1
    state["acc"] = (state["acc"] * 31 + state["step"]) % 1000003  # the "model update"
    if state["step"] % a.every == 0:
        if a.bug:
            for p in old_checkpoints():  # bug: the only complete checkpoint is deleted first
                os.remove(p)
            save(state["step"])
        else:
            save(state["step"])
            for p in old_checkpoints():  # correct: delete old ones only after the new one is in place
                if os.path.basename(p) != f"ckpt-{state['step']}.json":
                    os.remove(p)

with open(os.path.join(a.out, "result.json"), "w") as f:
    json.dump(state, f)
