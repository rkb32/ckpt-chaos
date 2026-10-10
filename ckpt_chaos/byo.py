"""`ckpt-chaos run`: crash-test any training command without changing the user's code.

The command contains `{out}` where it takes its output / checkpoint directory. The harness only ever
creates and writes inside its own per-run `{out}` directories. A fault-free run records every file event
the command makes under `{out}`; then one copy of the command is killed at each of those events (or an
evenly spaced sample of them), the resume command (default: the same command) is run on the same
directory, and the outcome is judged.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .invariants import ResumeOutcome
from .runner import Row, _events, print_table, rejudge

BOOT = Path(__file__).resolve().parent / "_boot"
PKG_PARENT = Path(__file__).resolve().parent.parent
VERSIONED = ("ckpt-chaos", "torch", "transformers", "lightning", "safetensors", "accelerate", "numpy")


def _versions() -> dict:
    """What a bug report needs to say about the machine: Python, OS and the packages that usually matter."""
    from importlib import metadata

    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in VERSIONED:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def _sub(cmd: list[str], out: Path) -> list[str]:
    return [a.replace("{out}", str(out)) for a in cmd]


def _env(extra: dict | None = None) -> dict:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    old = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join([str(BOOT), str(PKG_PARENT)] + ([old] if old else []))
    env.update(extra or {})
    return env


def _run(cmd: list[str], env: dict, cwd: str, timeout: int) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return -9, "TIMEOUT"


def _result(out: Path, result_file: str | None, result_cmd: str | None, cwd: str, timeout: int) -> str | None:
    """A comparable fingerprint of what the run produced, or None if the user gave no way to check."""
    if result_file:
        p = Path(result_file.replace("{out}", str(out)))
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"
    if result_cmd:
        rc, text = _run(shlex.split(result_cmd.replace("{out}", str(out)), posix=os.name != "nt"), _env(), cwd, timeout)
        return text.strip() if rc == 0 else f"ERROR({rc})"
    return None


Snapshot = dict[str, tuple[int, dict[str, int]]]  # checkpoint name -> (step, {file under it: size in bytes})


def _signature(p: Path) -> dict[str, int]:
    if p.is_file():
        return {".": p.stat().st_size}
    return {str(q.relative_to(p)): q.stat().st_size for q in p.rglob("*") if q.is_file()}


def _snapshot(pattern: str, out: Path) -> Snapshot:
    """Every checkpoint matching the glob and what it is made of. The step is the last integer in its name."""
    base = glob.escape(str(out))
    pat = pattern.replace("{out}", base) if "{out}" in pattern else base + os.sep + pattern
    snap: Snapshot = {}
    for hit in glob.glob(pat):
        numbers = re.findall(r"\d+", Path(hit).name)
        if numbers:
            snap[Path(hit).name] = (int(numbers[-1]), _signature(Path(hit)))
    return snap


def _newest_complete(ref: Snapshot, got: Snapshot, ignore_size: bool = False) -> int:
    """Highest step whose checkpoint looks exactly like the fault-free run's (same files, same sizes)."""
    best = 0
    for name, (step, sig) in ref.items():
        if name in got and (set(got[name][1]) == set(sig) if ignore_size else got[name][1] == sig):
            best = max(best, step)
    return best


MODES = ("before", "torn", "term")


def _even(items: list, k: int) -> list:
    """k items spread evenly over the list (all of them if it is short enough)."""
    if k <= 0:
        return []
    if len(items) <= k:
        return items
    stride = len(items) / k
    return [items[int(i * stride)] for i in range(k)]


def _select(events: list, max_points: int, modes: tuple[str, ...]) -> list[tuple[int, str]]:
    """Crash points for each mode. The budget is split between the modes, so every mode is sampled across the
    whole save, and before / term (which share a pool) land on the same file events."""
    unknown = [m for m in modes if m not in MODES]
    if unknown or not modes:
        raise SystemExit(f"crash modes must be one or more of {', '.join(MODES)}; got {list(modes) or 'none'}")
    if max_points < 1:
        raise SystemExit("--max-points must be at least 1")
    share, extra = divmod(max_points, len(modes))
    points: list[tuple[int, str]] = []
    for i, mode in enumerate(modes):
        pool = [n for n, _, _, tornable in events if tornable] if mode == "torn" else [n for n, *_ in events]
        points += [(n, mode) for n in _even(pool, share + (1 if i < extra else 0))]
    return points


def run_byo(train: list[str], resume: list[str] | None, work: Path, *, jobs: int = 1, timeout: int = 600,
            max_points: int = 40, result_file: str | None = None, result_cmd: str | None = None,
            step_regex: str | None = None, modes: tuple[str, ...] = ("before", "torn"),
            checkpoint_glob: str | None = None, ignore_size: bool = False, grace: float = 5.0) -> list[Row]:
    if not grace > 0:
        raise SystemExit("--grace must be a positive number of seconds")
    if not any("{out}" in a for a in train):
        raise SystemExit("put {out} in your command where it takes its output / checkpoint directory, "
                         "for example: -- python train.py --output_dir {out}")
    if checkpoint_glob and not step_regex:
        raise SystemExit("--checkpoint-glob needs --step-regex: to say whether the resume used the newest complete "
                         "checkpoint, the harness must read the step the resume started from")
    cwd = os.getcwd()
    if (work / "reference").exists() or any(work.glob("pt_*")):  # leftovers would be resumed from and skew every verdict
        raise SystemExit(f"{work} already holds the results of an earlier run. Pass a new --work directory "
                         f"(the default creates a fresh one every time).")
    work.mkdir(parents=True, exist_ok=True)
    resume = resume or train
    step_re = re.compile(step_regex) if step_regex else None
    unmatched: list[int] = []

    def armed(wd: Path, out: Path, crash=None, mode="before") -> dict:
        extra = {"CKPT_CHAOS_WATCH": str(out), "CKPT_CHAOS_LOG": str(wd / "events.log"), "CKPT_CHAOS_TORN_PYTHON": "1"}
        if crash:
            extra.update(CKPT_CHAOS_CRASH_AT=str(crash), CKPT_CHAOS_MODE=mode, CKPT_CHAOS_GRACE_S=str(grace))
        return extra

    print("1/3 fault-free reference run (records every file event under {out}) ...", flush=True)
    ref = work / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    ref_out = ref / "out"
    rc, text = _run(_sub(train, ref_out), _env(armed(ref, ref_out)), cwd, timeout)
    if rc != 0:
        raise SystemExit(f"your command failed on its own (exit {rc}) before any crash was injected:\n{text[-1200:]}")
    events = _events(ref / "events.log.rank0")
    if not events:
        raise SystemExit("the command made no file changes under {out}. Does it write its checkpoints there?")
    ref_result = _result(ref_out, result_file, result_cmd, cwd, timeout)
    ref_ckpts: Snapshot = {}
    if checkpoint_glob:
        ref_ckpts = _snapshot(checkpoint_glob, ref_out)
        if not ref_ckpts:
            raise SystemExit(f"--checkpoint-glob {checkpoint_glob!r} matched no checkpoint (with a number in its name) "
                             f"after the fault-free run. Checkpoints the run deletes itself, such as keep-last-N rotation, "
                             f"cannot be tracked: keep them all for this test.")
        print(f"    contract: {len(ref_ckpts)} checkpoints in the fault-free run, steps "
              f"{sorted(s for s, _ in ref_ckpts.values())}", flush=True)
    plan ={n: (n, kind, name, tornable) for n, kind, name, tornable in events}
    # what `ckpt-chaos repro` needs to rebuild one crash point on its own
    (work / "meta.json").write_text(json.dumps({
        "train": train, "resume": resume, "step_regex": step_regex, "checkpoint_glob": checkpoint_glob,
        "ignore_size": ignore_size, "result_file": result_file, "result_cmd": result_cmd, "work": str(work), "cwd": cwd,
        "modes": list(modes), "grace": grace,
        "reference_steps": sorted(s for s, _ in ref_ckpts.values()) if ref_ckpts else None, "versions": _versions(),
    }, indent=1), encoding="utf-8")
    points = _select(events, max_points, modes)
    if not points:  # e.g. only "torn" on a job with no writer that can be torn: say so instead of passing empty-handed
        raise SystemExit(f"no crash points for modes {list(modes)}: the job's {len(events)} file events offer none "
                         f"(torn needs torch.save, safetensors or Python open() writes)")
    print(f"    {len(events)} file events -> {len(points)} crash points (+1 control), {jobs} at a time", flush=True)
    if ref_result is None:
        print("    note: no --result-file / --result-cmd, so silent divergence cannot be detected", flush=True)

    def judge_result(out: Path) -> float:
        got = _result(out, result_file, result_cmd, cwd, timeout)
        return 0.0 if ref_result is None or got == ref_result else 1.0

    def record(order: int, wd: Path, label: str, rc: int, crashed: bool, drift: bool, outcome: ResumeOutcome,
               extra: dict | None = None) -> dict:
        rec = {"order": order, "label": label, "rc": rc, "crashed": crashed, "drift": drift, "outcome": asdict(outcome)}
        rec.update(extra or {})
        (wd / "result.json").write_text(json.dumps(rec), encoding="utf-8")
        return rec

    def control(order: int) -> dict:
        wd = work / "pt_control"
        wd.mkdir(parents=True, exist_ok=True)
        out = wd / "out"
        rc, _ = _run(_sub(train, out), _env(armed(wd, out)), cwd, timeout)
        diff = judge_result(out) if rc == 0 else None
        return record(order, wd, "no crash (control)", rc, False, False,
                      ResumeOutcome(None if rc == 0 else f"ExitCode{rc}", None, 0, diff))

    def one(order: int, point: tuple[int, str]) -> dict:
        n, mode = point
        wd = work / f"pt_{order:03d}"
        wd.mkdir(parents=True, exist_ok=True)
        out = wd / "out"
        rc, _ = _run(_sub(train, out), _env(armed(wd, out, n, mode)), cwd, timeout)
        seen = _events(wd / "events.log.rank0")
        # the crash event must be in the log; after a SIGTERM the job may log more events before it is killed
        drift = not any(e[0] == n and e[:3] == plan[n][:3] for e in seen)
        newest = _newest_complete(ref_ckpts, _snapshot(checkpoint_glob, out), ignore_size) if checkpoint_glob else 0
        rrc, rtext = _run(_sub(resume, out), _env(), cwd, timeout)  # the resume run is not armed
        raised = None if rrc == 0 else ("Timeout" if rtext == "TIMEOUT" else f"ExitCode{rrc}")
        step = None
        if step_re and raised is None:
            m = step_re.search(rtext)
            step = int(m.group(1)) if m else None
            if m is None:
                unmatched.append(order)
        diff = judge_result(out) if raised is None else None
        label = f"#{n} {plan[n][1]}:{plan[n][2]} [{mode}]"
        extra = {"point": [n, mode], "event": [plan[n][1], plan[n][2]], "resume_tail": rtext[-1500:]}
        return record(order, wd, label, rc, rc != 0, drift, ResumeOutcome(raised, step, newest, diff), extra)

    print("2/3 crashing and resuming ...", flush=True)
    with ThreadPoolExecutor(jobs) as ex:
        futures = [ex.submit(control, 0)] + [ex.submit(one, i + 1, p) for i, p in enumerate(points)]
        for f in futures:
            f.result()
    print("3/3 judging\n", flush=True)
    if checkpoint_glob and unmatched:
        print(f"WARNING: --step-regex found no step in the resume output of {len(unmatched)} runs. With "
              f"--checkpoint-glob a missing step counts as 'resumed from step 0' (LOST_WORK): check the regex.\n", flush=True)
    rows = rejudge(work)
    ctl = next((r for r in rows if r.label.startswith("no crash")), None)
    if ctl is not None and ctl.verdict == "SILENT_DIVERGENCE":
        print("WARNING: two fault-free runs of your command produced different results, so training is not "
              "reproducible run to run. Divergence verdicts below are unreliable until you fix seeds / "
              "determinism or give --result-cmd that compares with a tolerance.\n", flush=True)
    return rows
