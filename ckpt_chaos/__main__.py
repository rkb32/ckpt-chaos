"""python -m ckpt_chaos hf_trainer [--ranks N] [--jobs N] [--work DIR]    run the crash matrix
python -m ckpt_chaos loop_ddp --strategy tmp_rename_rank0 --ranks 2     a hand-written loop, chosen save protocol
python -m ckpt_chaos hf_trainer --judge RUN_DIR                          re-apply invariants.py to a finished run
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .runner import ROOT, TARGETS, print_table, rejudge, run_matrix

ap = argparse.ArgumentParser(prog="ckpt_chaos")
ap.add_argument("target", choices=sorted(TARGETS))
ap.add_argument("--ranks", type=int, default=1, help="gloo ranks on CPU (1 = single process)")
ap.add_argument("--jobs", type=int, default=3, help="crash points run concurrently")
ap.add_argument("--strategy", choices=["direct", "tmp_rename", "tmp_rename_rank0"], default="direct",
                help="checkpoint install strategy (loop_ddp only)")
ap.add_argument("--work", type=Path, default=ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S"))
ap.add_argument("--judge", type=Path, help="re-judge an existing run directory instead of running")
args = ap.parse_args()
print_table(rejudge(args.judge) if args.judge else run_matrix(args.work, args.jobs, args.ranks, args.target, args.strategy))
