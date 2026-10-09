"""Entry point of the GitHub Action (action.yml): turns its inputs into a `ckpt-chaos run` command line.

The action passes every input as an INPUT_* environment variable, never spliced into a shell script.
"""
from __future__ import annotations

import os
import sys
import tempfile
from typing import Mapping

from .cli import _split, main

VALUE_FLAGS = [("CHECKPOINT_GLOB", "--checkpoint-glob"), ("STEP_REGEX", "--step-regex"), ("RESULT_FILE", "--result-file"),
               ("RESULT_CMD", "--result-cmd"), ("RESUME_COMMAND", "--resume-cmd"), ("MAX_POINTS", "--max-points"),
               ("JOBS", "--jobs"), ("TIMEOUT", "--timeout"), ("FAIL_ON", "--fail-on")]


def argv_from_env(env: Mapping[str, str]) -> list[str]:
    command = env.get("INPUT_COMMAND", "").strip()
    if not command:
        raise SystemExit("the `command` input is required, for example: command: python train.py --output_dir {out}")
    argv = ["run"]
    for key, flag in VALUE_FLAGS:
        value = env.get(f"INPUT_{key}", "")
        if value.strip():
            argv += [flag, value]
    if env.get("INPUT_CKPT_IGNORE_SIZE", "").strip().lower() == "true":
        argv.append("--ckpt-ignore-size")
    if env.get("INPUT_WORK_DIR", "").strip():  # a fresh directory per call: the action may run twice in one job
        os.makedirs(env["INPUT_WORK_DIR"], exist_ok=True)
        argv += ["--work", tempfile.mkdtemp(prefix="run-", dir=env["INPUT_WORK_DIR"])]
    if env.get("GITHUB_STEP_SUMMARY", "").strip():
        argv += ["--summary", env["GITHUB_STEP_SUMMARY"]]
    return argv + ["--"] + _split(command)


if __name__ == "__main__":
    sys.exit(main(argv_from_env(os.environ)))
