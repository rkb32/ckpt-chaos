# ckpt-chaos

Crash-consistency testing for ML training checkpoints.

ckpt-chaos kills a training job at **every file boundary of a checkpoint save**, resumes it, and tells you
what came back: a clean resume, a crash that needs manual cleanup, a *silent* resume with different
weights, or lost work. It runs on CPU (gloo ranks), so no GPU is needed.

## Why

Most training stacks treat "`checkpoint-N/` exists" as "checkpoint N is complete". A kill in the middle of
a save leaves a folder that looks valid and is not.

- Hugging Face: [#35525](https://github.com/huggingface/transformers/issues/35525) (open) asks for a fallback to the previous checkpoint when the latest is broken.
- Hugging Face merged an atomic temp-directory save ([#35580](https://github.com/huggingface/transformers/pull/35580)), it broke multi-GPU DDP saves ([#36076](https://github.com/huggingface/transformers/issues/36076)), and it was reverted ([#36112](https://github.com/huggingface/transformers/pull/36112)).
- Lightning: [#21431](https://github.com/Lightning-AI/pytorch-lightning/issues/21431) (open), corrupt checkpoint saves.

It builds on the single-node, single-filesystem study
[Crash-Consistent Checkpointing for AI Training on macOS/APFS](https://arxiv.org/abs/2511.18323), whose
stated limits (one filesystem, one node, a synthetic workload) are this tool's starting points.

## Run it

```bash
python -m ckpt_chaos hf_trainer                      # single process, HF Trainer
python -m ckpt_chaos hf_trainer --ranks 2            # two gloo ranks
python -m ckpt_chaos loop_ddp --ranks 2 --strategy tmp_rename_rank0
python -m ckpt_chaos loop_ddp --ranks 2 --strategy tmp_rename --flake 25   # repeat the fault-free job, count failures
python -m ckpt_chaos hf_trainer --judge runs/<dir>   # re-apply the rules to a finished run
```

Needs `torch` (CPU is fine), `transformers` and `accelerate`. Tests: `python -m unittest discover tests`.

## How it works

1. A dry run counts the file events of one checkpoint save (`inject.py`): CPython audit events for
   open/rename/remove/mkdir, plus wrappers for the native writers `torch.save` and safetensors, which raise no audit events.
2. For every event, one run dies there (`before`: the file never appears). For native writers a second run
   dies with the file `torn` (cut to half its size). With several ranks, every rank is a crash candidate.
3. The job is resumed with the library's own auto-resume, and the final weights are compared with an
   uninterrupted run of the same schedule.
4. `invariants.py` turns each outcome into a verdict.

| Verdict | Meaning |
|---|---|
| `PASS` | resumed, weights identical to the uninterrupted run |
| `HARD_FAIL` | resume raised; the job is down until someone cleans up |
| `SILENT_DIVERGENCE` | resumed, but weights differ or ranks disagree |
| `LOST_WORK` | resumed correctly from an older checkpoint than necessary |

## Results

Setup: transformers 5.18.0 (tiny GPT-2, `resume_from_checkpoint=True`) and Lightning 2.6.6 (tiny MLP,
`fit(ckpt_path="last")`), torch 2.14.1 (CPU), Windows/NTFS. The job is killed while the step-10 checkpoint is
being saved (the step-5 checkpoint already exists).

| Setup | Runs | PASS | HARD_FAIL | SILENT_DIVERGENCE | LOST_WORK |
|---|---|---|---|---|---|
| HF Trainer 5.18.0, 1 process | 19 | 2 | 17 | 0 | 0 |
| HF Trainer 5.18.0, 2 gloo ranks | 22 | 1 | 19 | 2 | 0 |
| Lightning 2.6.6, 1 process | 5 | 3 | 0 | 0 | 2 |
| Lightning 2.6.6, 2 gloo ranks | 5 | 3 | 0 | 0 | 2 |

Each run is one crash point; the control run has no crash and a clean one counts as a PASS.

- **An intact resume is bit-identical** to an uninterrupted run (max weight diff 0.0), with 1 and with 2 ranks,
  so any difference below is caused by the crash.
- **The 17 + 19 hard failures are one root cause, not 36 bugs.** Auto-resume picks the highest-numbered
  `checkpoint-N` folder without checking that it is complete, and `trainer_state.json` is written last, so
  a kill at any earlier point leaves a folder that makes the resume raise (`FileNotFoundError`, `ValueError`,
  `SafetensorError`) until someone deletes it by hand. This is the behaviour behind
  [#35525](https://github.com/huggingface/transformers/issues/35525).
- **Multi-rank only: silent divergence.** If rank 1 dies before writing `rng_state_1.pth`, rank 0 finishes
  the checkpoint and `trainer_state.json`. The resume then succeeds from that folder, logs nothing, and ends
  with weights 1.65e-4 away from the uninterrupted run. The model and optimizer are intact; exact
  reproducibility is lost. A single process cannot produce this state.
- **Lightning is atomic per file, and the matrix shows it.** `_atomic_save` writes each checkpoint to a temp
  file in the system temp directory inside an fsspec transaction, then renames it into place, so no crash
  point leaves a broken checkpoint, with 1 or 2 ranks. The two LOST_WORK rows are a crash between installing
  `ck-step=10.ckpt` and updating `last.ckpt`: `ckpt_path="last"` then resumes from step 5 although a complete
  step-10 checkpoint exists. That redoes 5 steps and corrupts nothing. Because the temp file lives outside the
  watched folder, torn writes cannot reach it and are not exercised.
- **One crash point is clean with a single process** (HF): dying before the folder exists. With 2 ranks even that
  fails, because the surviving rank creates the folder (this depends on the few seconds the survivors run
  on, see Limits).

## Save protocols compared

`loop_ddp` is a small hand-written DDP loop with three ways of installing a checkpoint, run with 2 gloo
ranks. It shows the harness can say PASS, and that it can find a protocol that is broken only sometimes.

| Strategy | What it does | Crash matrix | Fault-free repeat (`--flake 25`) |
|---|---|---|---|
| `direct` | write straight into `checkpoint-N/`, state file last | 1 PASS, 11 HARD_FAIL | not run |
| `tmp_rename_rank0` | write `checkpoint-N.tmp/`, barrier, **rank 0** renames, barrier | **13 / 13 PASS** | **0 / 25 failed** |
| `tmp_rename` | write `checkpoint-N.tmp/`, barrier, **every rank** renames | 13 PASS, 1 HARD_FAIL | **2 / 25 failed** |

- `tmp_rename` is the shape of the temp-directory fix Hugging Face merged and reverted
  ([#35580](https://github.com/huggingface/transformers/pull/35580) / [#36076](https://github.com/huggingface/transformers/issues/36076)):
  every rank tries to move the same shared directory. Here the loser gets `FileNotFoundError` on
  `checkpoint-N.tmp -> checkpoint-N`. It passed 23 of 25 fault-free runs and its control row passed, so a
  single run would have missed it. That is why `--flake N` exists.
- The failing crash-matrix row for `tmp_rename` is not a recovery failure: rank 1 lost the same race during
  a *later* save after the resume. It is reported honestly as what happened, with a cause unrelated to the
  injected crash.
- Not claimed: that Hugging Face's code failed in exactly this way. #36076 was reported on Linux with a
  different error ("Directory not empty"); the Windows race gives the same class of failure, not the same bug.

## Limits

- Crashes are emulated by killing the process (`os._exit`) at a file event. That is not power loss: data
  the OS had not flushed is not lost, and writes are not reordered. Real power-loss testing needs kernel
  level injection or hardware.
- Python-written files (the JSON ones) are not torn yet, only whole-file absent or present.
- After one rank dies, the others run on for a few seconds before they are stopped, which is realistic
  but timing dependent.
- Tested so far: transformers 5.18.0 and Lightning 2.6.6 on Windows/NTFS with CPU gloo ranks. The
  hand-written loop also only on Windows. The crash matrices are single runs per crash point; only the
  fault-free runs have been repeated (`--flake`).

## Roadmap

- `torch.distributed.checkpoint` target
- Linux runs on ext4 and overlayfs (a `Dockerfile` is included but untested), including the filesystem in
  [Lightning #21431](https://github.com/Lightning-AI/pytorch-lightning/issues/21431): a checkpoint saved in
  Docker on Windows/WSL that is sometimes unreadable right after saving, with no crash involved
- torn writes for Python-written files
- power-loss emulation on Linux
