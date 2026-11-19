"""A stdlib-only job that saves each checkpoint as several objects under one prefix, the way an object
store (S3, GCS) layout does: one object per shard plus a metadata object. Object stores have no directory
rename, so a "move" is a copy followed by a delete, and there is no commit step: the prefix is
complete only when the last object lands.

Resume picks the highest-numbered prefix with no completeness check, like most scripts do.
A local folder stands in for the bucket, so the injector sees each object write as a file event.
The sizes are fixed, so the resume check does not depend on timing.

    ckpt-chaos run --checkpoint-glob "{out}/ckpt-*" --step-regex "resumed from step (\d+)" \
        --saved-regex "saved step (\d+)" --result-file "{out}/result.json" -- python examples/object_store_ckpt.py --out {out}
"""
import argparse
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--every", type=int, default=5)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

steps = sorted(int(m.group(1)) for n in os.listdir(a.out) if (m := re.fullmatch(r"ckpt-(\d+)", n)))
state = {"step": 0, "acc": 0}
if steps:
    with open(os.path.join(a.out, f"ckpt-{steps[-1]}", "meta.json")) as f:
        state = json.load(f)
print(f"resumed from step {state['step']}", flush=True)


def put(prefix, name, data):
    """One object: write it under a temp key, then copy it to its final key and delete the temp (S3 has no rename)."""
    tmp = os.path.join(prefix, name + ".tmp")
    with open(tmp, "w") as f:
        f.write(data)
    with open(tmp) as src, open(os.path.join(prefix, name), "w") as dst:
        dst.write(src.read())
    os.remove(tmp)


while state["step"] < a.steps:
    state["step"] += 1
    state["acc"] = (state["acc"] * 31 + state["step"]) % 1000003
    if state["step"] % a.every == 0:
        prefix = os.path.join(a.out, f"ckpt-{state['step']}")
        os.makedirs(prefix, exist_ok=True)
        put(prefix, "shard-0.bin", "x" * 4096)
        put(prefix, "shard-1.bin", "y" * 4096)
        put(prefix, "meta.json", json.dumps(state))  # the last object: a complete prefix has all three
        print(f"saved step {state['step']}", flush=True)

with open(os.path.join(a.out, "result.json"), "w") as f:
    json.dump(state, f)
