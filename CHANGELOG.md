# Changelog

## Unreleased

- New example `examples/s3_bench.py` and `examples/s3_ckpt_job.py`: crash test against an S3 API, one crash point per
  PutObject. Default is moto on 127.0.0.1:5005; `S3_ENDPOINT=aws` runs against real AWS.
- Bench reports INVALID instead of classifying: crash runs that do not exit 137, timeouts, and resumes that fail
  without `NoSuchKey`. Cleanup deletes only buckets the run named.
- Job client retries (5 attempts, standard mode) for throttling, 5xx and connection errors.

## 1.0.0

- New S3 crash-test example: `examples/s3_bench.py` and `examples/s3_ckpt_job.py`. Runs against moto by default, or real AWS with `S3_ENDPOINT=aws`.
- Bench reports INVALID instead of guessing when a crash run does not exit cleanly, times out, or fails a resume without `NoSuchKey`.
- Job client retries (5 attempts) on throttling, 5xx, and connection errors.
- `accelerate_auto` HARD_FAIL rows say "newest checkpoint incomplete" when that applies. Single-rank only; the verdict is unchanged.
- Colab notebook: `examples/colab_ckpt_chaos.ipynb`, CPU only, pinned to v0.2.0.
- CI: `tests-green` and `action-green` aggregate checks.
- Known gap: 2-rank `accelerate_auto` HARD_FAILs are unexplained and unlabeled. Tracked in #18.

## 0.2.0

- New verdict **SWALLOWED**: a disk-full error (`--modes enospc`) was caught by the job, the job exited 0, and a
  checkpoint the fault-free run kept is missing at the end. It is in the default `--fail-on` list, so exit codes
  change for existing users. Opt out with `--fail-on HARD_FAIL,SILENT_DIVERGENCE,CONTROL_FAIL`.
- New crash mode `enospc`: the write raises ENOSPC and the job keeps running.
- New `--saved-regex`: the newest save the job reports after it returned defines the checkpoint it was promised,
  instead of what is on disk. Use it for jobs that delete old checkpoints on purpose.
- New target `fsdp_dcp` (FSDP sharding with torch.distributed.checkpoint, CPU/gloo).
- New examples: `rotate_delete.py`, `swallow_enospc.py`, `object_store_ckpt.py` (folder stand-in for an object store).
- Tests: `tests/test_swallow.py`.
