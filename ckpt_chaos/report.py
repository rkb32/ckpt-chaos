"""Markdown for a CI job summary (`$GITHUB_STEP_SUMMARY`) built from the verdict rows."""
from __future__ import annotations

import re
from collections import Counter

ORDER = ["HARD_FAIL", "SILENT_DIVERGENCE", "LOST_WORK", "CONTROL_FAIL", "PASS"]
MEANING = {
    "HARD_FAIL": "the resume raised or exited non-zero",
    "SILENT_DIVERGENCE": "the resume ran, but the result differs from the fault-free run",
    "LOST_WORK": "the resume started from an older step than the newest complete checkpoint, or from scratch",
    "CONTROL_FAIL": "the fault-free control run itself failed",
    "PASS": "the resume matched the fault-free run",
}
MAX_FAILING_ROWS = 100
MAX_ALL_ROWS = 500


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _result(row) -> str:
    o = row.outcome
    if o.raised:
        return f"RAISED {o.raised}"
    if row.label.startswith("no crash") or o.resumed_from_step is None:
        return "-"
    return f"resumed from step {o.resumed_from_step}" + ("" if o.ranks_agree else " (ranks differ)")


def _table(rows, known: bool) -> list[str]:
    head = ["Crash point"] + (["Newest complete"] if known else []) + ["Resume result", "Weights diff", "Verdict"]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        o = r.outcome
        cells = [f"`{_cell(r.label)}`"] + ([str(o.newest_complete_step)] if known else [])
        cells += [_cell(_result(r)), "-" if o.weights_diff is None else f"{o.weights_diff:.2e}", f"**{r.verdict}**"]
        out.append("| " + " | ".join(cells) + " |")
    return out


def markdown(rows, *, fail_on: set[str], invocation: str | None = None) -> str:
    failed = [r for r in rows if r.verdict in fail_on]
    counts = Counter(r.verdict for r in rows)
    known = any(r.outcome.newest_complete_step for r in rows)
    lines = [f"## ckpt-chaos: {'FAILED' if failed else 'PASSED'}", ""]
    if failed:
        lines.append(f"{len(failed)} of {len(rows)} runs could not resume correctly after the job was killed while writing a checkpoint.")
    else:
        lines.append(f"All {len(rows)} runs resumed acceptably after the job was killed while writing a checkpoint.")
    ignored = {v: c for v, c in counts.items() if v != "PASS" and v not in fail_on}
    if ignored:
        lines.append("Reported but not failing the build: " + ", ".join(f"{c} {v}" for v, c in sorted(ignored.items()))
                     + ". Add them to `fail-on` to gate on them.")
    lines += ["", "| Verdict | Runs | Meaning |", "|---|---|---|"]
    lines += [f"| **{v}** | {counts[v]} | {MEANING[v]} |" for v in ORDER if counts.get(v)]
    if failed:
        lines += ["", "### Failing runs", ""] + _table(failed[:MAX_FAILING_ROWS], known)
        if len(failed) > MAX_FAILING_ROWS:
            lines.append(f"\n{len(failed) - MAX_FAILING_ROWS} more failing runs are in the full table.")
    lines += ["", f"<details><summary>All {len(rows)} runs</summary>", ""] + _table(rows[:MAX_ALL_ROWS], known) + ["", "</details>"]
    if invocation:
        lines += ["", "Reproduce locally (`pip install ckpt-chaos`):", "", "```", invocation, "```"]
    return "\n".join(lines) + "\n"


def error_markdown(message: str) -> str:
    return "## ckpt-chaos: could not run\n\n```\n" + message.strip()[-2000:] + "\n```\n"


_XML_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")


def _xml_safe(text: str) -> str:
    """XML 1.0 cannot carry most control characters, even escaped; a file name or an error message can."""
    return _XML_ILLEGAL.sub("?", text)


def junit(rows, *, fail_on: set[str]) -> str:
    """JUnit XML: one test case per crash point, a failure where the verdict is in fail_on. CI systems read this."""
    from xml.sax.saxutils import escape, quoteattr

    failures = sum(1 for r in rows if r.verdict in fail_on)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             f'<testsuite name="ckpt-chaos" tests="{len(rows)}" failures="{failures}">']
    for r in rows:
        name = quoteattr(_xml_safe(r.label))
        if r.verdict in fail_on:
            message = quoteattr(_xml_safe(f"{r.verdict}: {_result(r)}"))
            lines.append(f'  <testcase name={name} classname="ckpt-chaos">'
                         f'<failure message={message}>{escape(MEANING.get(r.verdict, r.verdict))}</failure></testcase>')
        else:
            lines.append(f'  <testcase name={name} classname="ckpt-chaos"/>')
    lines.append("</testsuite>")
    return "\n".join(lines) + "\n"


def junit_error(message: str) -> str:
    """A one-case JUnit file for a run that never got judged, so CI shows the reason instead of a stale or missing file."""
    from xml.sax.saxutils import escape

    text = escape(_xml_safe(message.strip()[-2000:]))
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<testsuite name="ckpt-chaos" tests="1" errors="1">\n'
            f'  <testcase name="ckpt-chaos could not run" classname="ckpt-chaos"><error message="could not run">{text}'
            '</error></testcase>\n</testsuite>\n')
