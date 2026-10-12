"""A Keras 3 job (torch backend, CPU): saves the whole model to model.keras every 5 steps and resumes from it.

Keras writes the .keras archive with zipfile, which opens files through Python's open(), so ckpt-chaos sees every
write as a file event. The job has no ckpt-chaos code; the harness only reads the "resumed from step" line.

    ckpt-chaos run --result-file "{out}/result.json" --step-regex "resumed from step (\\d+)" \\
        --modes before,torn -- python examples/keras_ckpt.py --out {out}
"""
import argparse
import hashlib
import json
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np  # noqa: E402
import keras  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--every", type=int, default=5)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, "model.keras")
keras.utils.set_random_seed(0)
rng = np.random.default_rng(0)
x = rng.normal(size=(64, 8)).astype("float32")
y = x.sum(1, keepdims=True).astype("float32")

if os.path.exists(path):
    model = keras.saving.load_model(path)
    step = int(model.optimizer.iterations.numpy())
else:
    model = keras.Sequential([keras.Input((8,)), keras.layers.Dense(16, activation="tanh"), keras.layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.SGD(0.05, momentum=0.9), loss="mse")
    step = 0
print(f"resumed from step {step}", flush=True)

while step < a.steps:
    model.train_on_batch(x, y)
    step += 1
    if step % a.every == 0:
        model.save(path)  # overwrites the one archive in place, like the other in-place jobs

# the archive's bytes change between runs (zip timestamps), so compare a digest of the weights instead
digest = hashlib.sha256(b"".join(np.round(w.numpy(), 6).tobytes() for w in model.weights)).hexdigest()
with open(os.path.join(a.out, "result.json"), "w") as f:
    json.dump({"step": step, "weights_sha256": digest}, f)
