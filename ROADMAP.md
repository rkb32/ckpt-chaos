# Roadmap

Ranked by how much it widens who can use the tool, weighted by the evidence behind it and the effort.
The evidence is what developers report in public. Where I could not reach a source, it says so.

## What developers report

**Where checkpoint trouble shows up** (GitHub issues that mention "checkpoint" together with
*corrupted*, *truncated*, *incomplete*, *preemption* or *killed*; keyword counts, so directional only,
and they include closed issues, comments and plain usage questions):

| Repo | corrupt / truncated / incomplete / preempt / killed | "checkpoint resume" mentions |
|---|---|---|
| tensorflow/tensorflow | 218 | 73 |
| huggingface/transformers | 201 | 590 |
| ray-project/ray | 178 | 290 |
| pytorch/pytorch | 109 | 162 |
| Lightning-AI/pytorch-lightning | 91 | 403 |
| unslothai/unsloth | 58 | 246 |
| NVIDIA-NeMo (Speech) | 54 | 195 |
| modelscope/ms-swift | 53 | 473 |
| ultralytics/ultralytics | 53 | 235 |
| hiyouga/LlamaFactory | 47 | 864 |
| deepspeedai/DeepSpeed | 46 | 132 |
| huggingface/accelerate | 45 | 113 |
| axolotl-ai-cloud/axolotl | 27 | 335 |

**Concrete reports** (read in full, not just titles):

- [verl #7952](https://github.com/verl-project/verl/issues/7952) (open): with async checkpoint saving the marker
  file `latest_checkpointed_iteration.txt` is never written, so `resume_mode=auto` **silently restarts from
  scratch**. Reproduced on CPU by the reporter.
- [Hugging Face #35525](https://github.com/huggingface/transformers/issues/35525) (open): fall back to the previous
  checkpoint when the newest is broken. [#35580](https://github.com/huggingface/transformers/pull/35580) merged an
  atomic save and [#36112](https://github.com/huggingface/transformers/pull/36112) reverted it after it broke
  multi-GPU saves ([#36076](https://github.com/huggingface/transformers/issues/36076)).
- [Lightning #21431](https://github.com/Lightning-AI/pytorch-lightning/issues/21431) (open): a checkpoint saved in
  Docker on Windows/WSL is sometimes unreadable right after saving (about 2 in 48), **with no crash involved**.
- [litdata #263](https://github.com/Lightning-AI/litdata/issues/263): resuming with a changed dataset fails, a data-pipeline
  state problem.
- [karpathy/autoresearch PR #112](https://github.com/karpathy/autoresearch/pull/112) (open): save a checkpoint before
  evaluation, because a crash in evaluation after a full training budget loses the whole run.
- Hugging Face forum threads on [a loss spike when resuming an FSDP sharded checkpoint](https://discuss.huggingface.co/t/loss-spike-when-resuming-from-fsdp-sharded-state-dict-checkpoint-possible-optimizer-state-mismatch/160974)
  and [loss that rises after a resume although the checkpoint loaded](https://discuss.huggingface.co/t/training-resumes-with-increased-loss-despite-checkpoint-loading/105464).

**Papers** (titles and abstracts only; I did not verify their numbers):
[L4](https://arxiv.org/abs/2503.20263) studies reports of failed LLM training jobs,
[ByteCheckpoint](https://arxiv.org/abs/2407.20143) describes a barrier so all workers save atomically,
[Understanding Silent Data Corruption in LLM Training](https://arxiv.org/abs/2502.12340) covers silent failures.

**Not reachable:** X sits behind a login wall and LinkedIn is not indexed by the search tool I used, so
neither is represented above. Reading them needs a signed-in browser.

## Ranked features

| # | Feature | Why | Status |
|---|---|---|---|
| 1 | **Run on any unmodified script** (`ckpt-chaos run`) | The issues above span a dozen frameworks and many custom loops; nobody will rewrite their trainer to try a tool | **done** |
| 2 | Torn writes for Python `open()` files (json, pickle, numpy, yaml) | Most scripts write checkpoints that way, not only through `torch.save` | **done** |
| 3 | **Resume-contract check**: flag "restarted from step 0 although a complete checkpoint existed" | verl #7952 is exactly this, and it is silent. Needs a way to know which checkpoints are complete in `run` mode | **done** (`--checkpoint-glob`; "complete" = same files and sizes as the fault-free run; not yet tried on a real framework's resume; keep-last-N rotation not tracked) |
| 4 | **GitHub Action + job summary** (markdown table, JUnit) | Where developers already are; one line of YAML to adopt | **built** (`action.yml`, `--summary`; Markdown summary only, JUnit not done). Simulated locally and unit-tested; not yet run on a GitHub runner |
| 5 | **Repro generator**: for each failing point, a minimal standalone script plus ready-to-file issue text | Every upstream issue filed with it carries the tool to a framework's own users | next |
| 6 | **Public scoreboard**: framework x version x verdict, regenerated in CI | The shareable result: "we killed N trainers mid-save, here is who survived". Needs more targets (torchtune, accelerate, DeepSpeed, ms-swift, LLaMA-Factory, Axolotl, Unsloth, ultralytics, Ray Train, Keras) | planned |
| 7 | Filesystem mode: save, reload, repeat N times per filesystem (Docker volume, overlayfs, bind mount, NFS) | Lightning #21431 has no crash at all | **done** (`ckpt-chaos roundtrip`). It could **not** reproduce #21431 on Docker Desktop: 0 failures in 8 filesystem x writer combinations. NFS and a WSL distro not tried |
| 8 | Preemption mode: SIGTERM, then a grace period, then SIGKILL | Spot-instance training. Linux only | planned |
| 9 | Data-pipeline state check (same samples after a resume) | litdata #263 and the forum threads | planned |
| 10 | `run` mode with several ranks; native-writer tearing for h5py / tensorstore | Completes distributed and Keras / Orbax coverage | planned |
| 11 | Power-loss emulation (reordered, unflushed writes) on Linux | The study this builds on calls it out; heavy | later |

## Getting it used

Needs a public repository and your say-so, so none of this is done: a PyPI release, an asciicast in the
README, a short write-up of the results, upstream issues that include a reproduction (feature 5),
the badge "crash-tested with ckpt-chaos", and posts where developers talk (GitHub Discussions, Reddit,
Hugging Face and PyTorch forums, X, LinkedIn).
