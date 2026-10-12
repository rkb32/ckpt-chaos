"""A job that saves its state as a zarr array with tensorstore, overwriting it in place every 5 steps.

The array is split into 8 chunks (one file each). The step and the accumulator sit in the first chunk. On start the
job reopens the store and resumes from the step it finds. ckpt-chaos counts each write as a file event, so "before"
kills the job before the write, and "torn" lets the write finish and then cuts the chunk file it changed to half.

    ckpt-chaos run --result-file "{out}/result.json" --step-regex "resumed from step (\\d+)" \\
        -- python examples/tensorstore_ckpt.py --out {out}
"""
import argparse
import json
import os

import numpy as np
import tensorstore as ts

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--every", type=int, default=5)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
SIZE = 4096
spec = {"driver": "zarr", "kvstore": {"driver": "file", "path": os.path.join(a.out, "state.zarr")},
        "metadata": {"chunks": [512]}}
store = ts.open(spec, create=True, open=True, dtype=ts.float32, shape=[SIZE]).result()

state = store.read().result()
step, acc = int(state[0]), int(state[1])
if step == 0 and acc == 0 and not np.any(state):
    step, acc = 0, 0  # a fresh store is all zeros
print(f"resumed from step {step}", flush=True)

while step < a.steps:
    step += 1
    acc = (acc * 31 + step) % 1000003  # the "model update"
    if step % a.every == 0:
        arr = np.zeros(SIZE, dtype=np.float32)
        arr[0], arr[1] = step, acc
        store[...] = arr

with open(os.path.join(a.out, "result.json"), "w") as f:
    json.dump({"step": step, "acc": acc}, f)
