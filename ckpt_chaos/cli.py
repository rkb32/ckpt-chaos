"""ckpt-chaos command line.

  ckpt-chaos run   [options] -- <your training command with {out}>     crash-test any training script
  ckpt-chaos bench <target> [--ranks N] [--strategy S] [--flake N]     the built-in targets
  ckpt-chaos judge <run_dir>                                           re-apply the verdict rules to a finished run
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from pathlib import Path

DEFAULT_FAIL_ON = "HARD_FAIL,SILENT_DIVERGENCE,CONTROL_FAIL"


def _split(cmd: str) -> list[str]:
    return shlex.split(cmd, posix=os.name != "nt")


def _exit_code(rows, fail_on: str) -> int:
    bad = {v.strip().upper() for v in fail_on.split(",") if v.strip()}
    return 1 if any(r.verdict in bad for r in rows) else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ckpt-chaos", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="crash-test any training command (put {out} where it takes its output dir)")
    r.add_argument("--resume-cmd", help="command that resumes the job (default: the same command); may use {out}")
    r.add_argument("--result-file", help="file under {out} whose bytes must match the fault-free run, e.g. '{out}/final.pt'")
    r.add_argument("--result-cmd", help="command whose output must match the fault-free run (use for tolerances)")
    r.add_argument("--step-regex", help="regex with one group that finds the resumed step in the resume output")
    r.add_argument("--max-points", type=int, default=40, help="cap on crash points (evenly sampled)")
    r.add_argument("--jobs", type=int, default=1, help="crash points run concurrently (keep 1 on a shared GPU)")
    r.add_argument("--timeout", type=int, default=600, help="seconds per command")
    r.add_argument("--fail-on", default=DEFAULT_FAIL_ON, help=f"verdicts that give a non-zero exit (default {DEFAULT_FAIL_ON})")
    r.add_argument("--work", type=Path, default=Path("ckpt-chaos-runs") / time.strftime("%Y%m%d-%H%M%S"))
    r.add_argument("command", nargs=argparse.REMAINDER, help="-- followed by your training command")

    b = sub.add_parser("bench", help="run a built-in target: hf_trainer, lightning_trainer or loop_ddp")
    b.add_argument("target")
    b.add_argument("--ranks", type=int, default=1, help="gloo ranks on CPU (1 = single process)")
    b.add_argument("--jobs", type=int, default=3, help="runs executed concurrently")
    b.add_argument("--strategy", choices=["direct", "tmp_rename", "tmp_rename_rank0"], default="direct",
                   help="checkpoint install strategy (loop_ddp only)")
    b.add_argument("--flake", type=int, metavar="N", help="repeat the fault-free job N times instead of crashing it")
    b.add_argument("--work", type=Path, default=None)

    j = sub.add_parser("judge", help="re-judge a finished run directory")
    j.add_argument("run_dir", type=Path)

    t = sub.add_parser("roundtrip", help="save a checkpoint and load it straight back N times on a filesystem")
    t.add_argument("--dir", type=Path, required=True, help="directory on the filesystem to test")
    t.add_argument("--n", type=int, default=200)
    t.add_argument("--writer", choices=["torch", "lightning"], default="torch")
    t.add_argument("--size-mb", type=float, default=20.0)
    t.add_argument("--unique-paths", action="store_true", help="a new file each time (default: overwrite one file)")
    t.add_argument("--sleep", type=float, default=0.0, help="seconds between the save and the load")

    a = ap.parse_args(argv)

    if a.cmd == "roundtrip":
        from .roundtrip import run_roundtrip

        failed, errors = run_roundtrip(a.dir, a.n, a.writer, a.size_mb, not a.unique_paths, a.sleep)
        print(f"{a.writer} roundtrip on {a.dir}: {failed}/{a.n} saves unreadable "
              f"({a.size_mb:g} MB, {'one file overwritten' if not a.unique_paths else 'new file each time'}, sleep {a.sleep}s)")
        for msg, count in errors.most_common(5):
            print(f"    {count} x {msg}")
        return 1 if failed else 0

    if a.cmd == "judge":
        from .runner import print_table, rejudge

        rows = rejudge(a.run_dir)
        print_table(rows)
        return _exit_code(rows, DEFAULT_FAIL_ON)

    if a.cmd == "bench":
        from .runner import TARGETS, flake_check, print_table, run_matrix

        if a.target not in TARGETS:
            ap.error(f"target must be one of {sorted(TARGETS)}")
        # absolute, because the child processes run from the package's directory, not from here
        work = (a.work or Path("ckpt-chaos-runs") / time.strftime("%Y%m%d-%H%M%S")).resolve()
        if a.flake:
            failed, total = flake_check(work, a.jobs, a.ranks, a.target, a.strategy, a.flake)
            print(f"{a.target}/{a.strategy}, {a.ranks} rank(s), {a.jobs} concurrent: {failed}/{total} fault-free runs failed")
            return 1 if failed else 0
        rows = run_matrix(work, a.jobs, a.ranks, a.target, a.strategy)
        print_table(rows)
        return _exit_code(rows, DEFAULT_FAIL_ON)

    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command:
        ap.error("give your training command after --, for example: -- python train.py --output_dir {out}")
    from .byo import run_byo
    from .runner import print_table

    rows = run_byo(command, _split(a.resume_cmd) if a.resume_cmd else None, a.work.resolve(), jobs=a.jobs, timeout=a.timeout,
                   max_points=a.max_points, result_file=a.result_file, result_cmd=a.result_cmd, step_regex=a.step_regex)
    print_table(rows)
    return _exit_code(rows, a.fail_on)


if __name__ == "__main__":
    sys.exit(main())
