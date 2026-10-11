"""A stdlib-free-of-torch job that saves its state as an HDF5 file (h5py), overwriting one file in place.

Every 5 steps it writes <out>/state.h5 with mode "w" (the same in-place shape as naive_json.py), and on start
it resumes from that file if it exists. With ckpt-chaos's h5py support, a "torn" crash cuts the finished file
to half its size at close.

    ckpt-chaos run --result-file "{out}/result.json" --step-regex "resumed from step (\\d+)" \\
        -- python examples/h5py_json_ckpt.py --out {out}
"""
import argparse
import json
import os

import h5py

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--every", type=int, default=5)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, "state.h5")
step, acc = 0, 0
if os.path.exists(path):
    with h5py.File(path, "r") as f:
        step, acc = int(f["step"][()]), int(f["acc"][()])
print(f"resumed from step {step}", flush=True)

while step < a.steps:
    step += 1
    acc = (acc * 31 + step) % 1000003  # the "model update"
    if step % a.every == 0:
        with h5py.File(path, "w") as f:
            f["step"] = step
            f["acc"] = acc
            f["payload"] = [acc] * 4096  # enough bytes that the file is worth tearing

with open(os.path.join(a.out, "result.json"), "w") as f:
    json.dump({"step": step, "acc": acc}, f)
