"""`ckpt-chaos scoreboard`: one table of verdict counts across runs, for a README or a shared page.

Each run directory is judged again with the same rules as `ckpt-chaos run`, so the table is never out of step with a run.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .invariants import ResumeOutcome, classify

COLUMNS = ["PASS", "HARD_FAIL", "SILENT_DIVERGENCE", "LOST_WORK", "CONTROL_FAIL"]
VERSION_KEYS = ("torch", "transformers", "lightning", "h5py", "tensorstore")


def summarize(run_dir: Path, name: str | None = None) -> dict:
    counts: Counter = Counter()
    points = 0
    for p in sorted(run_dir.glob("pt_*/result.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("label", "").startswith("no crash"):
            if rec.get("rc", 0) != 0:
                counts["CONTROL_FAIL"] += 1  # same rule as rejudge(): the fault-free run itself died
            continue  # the control is not a crash point, so it is not counted in the points column
        points += 1
        counts[classify(ResumeOutcome(**rec["outcome"]))] += 1
    versions: dict = {}
    meta_file = run_dir / "meta.json"
    if meta_file.exists():
        versions = json.loads(meta_file.read_text(encoding="utf-8")).get("versions", {})
    return {"name": name or run_dir.name, "points": points, "counts": counts, "versions": versions}


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def markdown(rows: list[dict]) -> str:
    head = ["Target", "Crash points"] + COLUMNS + ["Versions"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        ver = ", ".join(f"{k} {r['versions'][k]}" for k in VERSION_KEYS if k in r["versions"]) or "-"
        cells = [_cell(r["name"]), str(r["points"])] + [str(r["counts"].get(c, 0)) for c in COLUMNS] + [_cell(ver)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
