"""Crash every save boundary, resume, and tabulate what came back.

Running and judging are separate: each crash point saves a result.json with the
raw ResumeOutcome, so changing the rules in invariants.py needs only --judge,
not a re-run. Multi-rank jobs are launched here (not with torchrun, whose
rendezvous fails on Windows torch builds without libuv).
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .invariants import ResumeOutcome, classify

ROOT = Path(__file__).resolve().parent.parent
TARGETS = {
    "hf_trainer": "ckpt_chaos.targets.hf_trainer",
    "loop_ddp": "ckpt_chaos.targets.loop_ddp",
    "lightning_trainer": "ckpt_chaos.targets.lightning_trainer",
}
PEER_GRACE_S = 3  # after one rank dies, how long the others may run on before we tear them down


def _env(extra=None) -> dict:
    env = dict(
        os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8", TRANSFORMERS_VERBOSITY="error",
        HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
    )
    env.update(extra or {})
    return env


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _launch(target: str, phase: str, out: Path, extra=None, ranks: int = 1, timeout: int = 240) -> int:
    """Run the target as `ranks` processes. Returns the first non-zero exit code, else 0."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", TARGETS[target], phase, "--out", str(out)]
    port, procs = _free_port(), []
    for r in range(ranks):
        env = _env(extra)
        if ranks > 1:
            env.update(RANK=str(r), LOCAL_RANK=str(r), WORLD_SIZE=str(ranks), MASTER_ADDR="127.0.0.1",
                       MASTER_PORT=str(port), USE_LIBUV="0")
            if target == "lightning_trainer":  # tells Lightning the ranks are already launched: don't re-spawn
                env["TORCHELASTIC_RUN_ID"] = "ckpt-chaos"
        log = open(out.parent / f"{phase}_rank{r}.log", "w")
        procs.append(subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
    start, first_fail = time.time(), None
    while any(p.poll() is None for p in procs):
        if first_fail is None and any(p.poll() not in (None, 0) for p in procs):
            first_fail = time.time()
        if (first_fail and time.time() - first_fail > PEER_GRACE_S) or time.time() - start > timeout:
            for p in procs:
                if p.poll() is None:
                    p.kill()
            break
        time.sleep(0.2)
    codes = [p.wait() for p in procs]
    return next((c for c in codes if c), 0)


def _profile(ckpt: Path) -> dict[str, int]:
    return {p.name: p.stat().st_size for p in ckpt.iterdir()}


def _is_complete(ckpt: Path, healthy: dict[str, int]) -> bool:
    """Complete = every healthy file is present, binaries at full size, JSON parseable."""
    for name, size in healthy.items():
        p = ckpt / name
        if not p.exists():
            return False
        if name.endswith(".json"):
            try:
                json.loads(p.read_text())
            except ValueError:
                return False
        elif p.stat().st_size != size:
            return False
    return True


def _newest_complete(out: Path, healthy: dict[str, int]) -> int:
    steps = [int(m.group(1)) for p in out.iterdir() if p.is_dir() and (m := re.fullmatch(r"checkpoint-(\d+)", p.name))
             and _is_complete(p, healthy)]
    return max(steps, default=0)


def _lightning_newest(out: Path, healthy: dict[str, int]) -> int:
    """Lightning checkpoints are single .ckpt files: complete means it loads, newest means highest global_step."""
    best = 0
    for p in (out / "checkpoints").glob("*.ckpt"):
        try:
            best = max(best, torch.load(p, map_location="cpu", weights_only=False)["global_step"])
        except Exception:  # noqa: BLE001 - a file that does not load is simply not a complete checkpoint
            pass
    return best


NEWEST = {"lightning_trainer": _lightning_newest}  # targets whose layout is not checkpoint-N/ directories


def _weights_diff(ref: Path, got: Path) -> float:
    a = torch.load(ref, map_location="cpu", weights_only=True)
    b = torch.load(got, map_location="cpu", weights_only=True)
    if a.keys() != b.keys():
        return float("inf")
    return max((a[k] - b[k]).abs().max().item() for k in a)


def _events(log: Path) -> list[tuple[int, str, str, bool]]:
    rows = []
    if log.exists() and log.stat().st_size:
        for line in log.read_text().strip().splitlines():
            n, kind, name, tornable = line.split("\t")
            rows.append((int(n), kind, name, tornable == "1"))
    return rows


def _collect(out: Path, ranks: int, ref_final: Path) -> tuple[str | None, int | None, float | None, bool]:
    """Merge the per-rank resume outcomes into (raised, resumed_from_step, weights_diff, ranks_agree)."""
    ocs = []
    for r in range(ranks):
        p = out / f"outcome_rank{r}.json"
        ocs.append(json.loads(p.read_text()) if p.exists() else {"raised": "ChildDied", "resumed_from_step": None})
    raised = [o["raised"] for o in ocs if o["raised"]]
    if raised:
        real = [x for x in raised if x != "ChildDied"]
        return (real or raised)[0], None, None, True
    steps = {o["resumed_from_step"] for o in ocs}
    finals = [out / f"final_rank{r}.pt" for r in range(ranks)]
    diff = _weights_diff(ref_final, finals[0])
    agree = len(steps) == 1 and all(_weights_diff(finals[0], f) == 0 for f in finals[1:])
    return None, ocs[0]["resumed_from_step"], diff, agree


def _run_point(cfg, order, point) -> dict:
    work, target, ranks, base = cfg["work"], cfg["target"], cfg["ranks"], cfg["base"]
    plan, healthy, ref_final = cfg["plan"], cfg["healthy"], cfg["ref_final"]
    rank, n, mode = point if point else (0, None, "before")
    wd = work / f"pt_{'control' if point is None else f'r{rank}_{n}_{mode}'}"
    out = wd / "out"
    wd.mkdir(parents=True)
    extra = dict(base, CKPT_CHAOS_WATCH=str(out), CKPT_CHAOS_LOG=str(wd / "events.log"))
    if point:
        extra.update(CKPT_CHAOS_CRASH_AT=str(n), CKPT_CHAOS_MODE=mode, CKPT_CHAOS_CRASH_RANK=str(rank))
    rc = _launch(target, "train", out, extra, ranks)

    seen = _events(wd / f"events.log.rank{rank}")
    drift = bool(point) and (not seen or seen[-1][:3] != plan[(rank, n)][:3])
    newest = NEWEST.get(target, _newest_complete)(out, healthy)  # measured before resume repairs anything

    _launch(target, "resume", out, base, ranks)
    raised, resumed, diff, agree = _collect(out, ranks, ref_final)
    tag = "" if ranks == 1 else f"r{rank} "
    label = "no crash (control)" if point is None else f"{tag}{plan[(rank, n)][1]}:{plan[(rank, n)][2]} [{mode}]"
    rec = {
        "order": order, "label": label, "rc": rc, "crashed": rc != 0, "drift": drift,
        "outcome": asdict(ResumeOutcome(raised, resumed, newest, diff, agree)),
    }
    (wd / "result.json").write_text(json.dumps(rec))
    return rec


@dataclass
class Row:
    label: str
    crashed: bool
    drift: bool
    outcome: ResumeOutcome
    verdict: str


def _judge(rec: dict) -> Row:
    o = ResumeOutcome(**rec["outcome"])
    if rec["label"].startswith("no crash") and rec.get("rc", 0) != 0:
        verdict = "CONTROL_FAIL"  # the fault-free run itself died: the save protocol is broken before any crash
    else:
        try:
            verdict = classify(o)
        except NotImplementedError:
            verdict = "(todo)"
    return Row(rec["label"], rec["crashed"], rec["drift"], o, verdict)


def rejudge(work: Path) -> list[Row]:
    recs = sorted((json.loads(p.read_text()) for p in work.glob("pt_*/result.json")), key=lambda r: r["order"])
    return [_judge(r) for r in recs]


def run_matrix(work: Path, jobs: int = 3, ranks: int = 1, target: str = "hf_trainer", strategy: str = "direct") -> list[Row]:
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    base = {"CKPT_CHAOS_STRATEGY": strategy}
    ref = work / "reference" / "out"
    _launch(target, "reference", ref, base, ranks)
    dry = work / "dry" / "out"
    _launch(target, "train", dry, dict(base, CKPT_CHAOS_WATCH=str(dry), CKPT_CHAOS_LOG=str(work / "dry" / "events.log")), ranks)
    if target in NEWEST:
        healthy = {}
    else:
        healthy_dir = dry / "checkpoint-10"
        healthy = _profile(healthy_dir if healthy_dir.exists() else dry / "checkpoint-10.tmp")
    plan = {}
    for r in range(ranks):
        for n, kind, name, tornable in _events(work / "dry" / f"events.log.rank{r}"):
            plan[(r, n)] = (n, kind, name, tornable)
    points = [None] + [(r, n, "before") for (r, n) in plan] + [(r, n, "torn") for (r, n), v in plan.items() if v[3]]
    print(f"{target}/{strategy}, {ranks} rank(s): {len(plan)} save events -> {len(points)} runs (incl. control), "
          f"{jobs} at a time", flush=True)

    cfg = dict(work=work, target=target, ranks=ranks, base=base, plan=plan, healthy=healthy,
               ref_final=ref / "final_rank0.pt")
    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(lambda ip: _run_point(cfg, ip[0], ip[1]), enumerate(points)))
    print(f"done in {time.time() - t0:.0f}s\n")
    return rejudge(work)


def flake_check(work: Path, jobs: int, ranks: int, target: str, strategy: str, n: int) -> tuple[int, int]:
    """Run the fault-free job n times and count failures. A race in the save protocol shows up here
    with no crash injected, and a single control run can pass by luck."""
    base = {"CKPT_CHAOS_STRATEGY": strategy}

    def failed(i: int) -> bool:
        return _launch(target, "reference", work / f"flake_{i}" / "out", base, ranks) != 0

    with ThreadPoolExecutor(jobs) as ex:
        results = list(ex.map(failed, range(n)))
    return sum(results), n


def print_table(rows: list[Row]) -> None:
    print(f"{'crash point':<58}{'newest ok':>10}  {'resume result':<26}{'weights diff':>13}  verdict")
    for r in rows:
        o = r.outcome
        result = f"RAISED {o.raised}" if o.raised else f"resumed from step {o.resumed_from_step}"
        if not o.raised and not o.ranks_agree:
            result += " (ranks differ)"
        diff = "-" if o.weights_diff is None else f"{o.weights_diff:.2e}"
        flag = "" if r.crashed or r.label.startswith("no crash") else "  [DID NOT CRASH]"
        flag += "  [PLAN DRIFT]" if r.drift else ""
        print(f"{r.label:<58}{o.newest_complete_step:>10}  {result:<26}{diff:>13}  {r.verdict}{flag}")
    counts = Counter(r.verdict for r in rows)
    print("\n" + "  ".join(f"{v}: {c}" for v, c in sorted(counts.items())))
