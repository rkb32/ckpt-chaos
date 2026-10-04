"""What counts as a correct recovery after a crash?

For every crash point the runner resumes the job and records a ResumeOutcome.
classify() turns that outcome into one verdict. The report is only as honest as
these rules, so they are deliberately short and checked in this order:

  1. The resume raised                      -> HARD_FAIL  (job is down until someone cleans up)
  2. It resumed but the weights differ,
     or the ranks disagree                  -> SILENT_DIVERGENCE  (nothing tells you the run is no longer the same run)
  3. It resumed from an older checkpoint
     than the newest complete one           -> LOST_WORK  (correct, but redid finished steps)
  4. Otherwise                              -> PASS

Judgment calls: every exception counts as HARD_FAIL, even a clear one, because
the job still needs a human. Divergence is checked before lost work so a bad
resume can never hide behind a lenient rule. Ranks that resume from different
steps, or end with different weights, count as divergence even if rank 0 matches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Verdict = Literal["PASS", "HARD_FAIL", "SILENT_DIVERGENCE", "LOST_WORK"]

# Exact on CPU: an intact resume reproduced an uninterrupted run with max diff 0.0.
# Loosen this for GPU runs, where reduction order can differ from run to run.
WEIGHT_TOL = 0.0


@dataclass
class ResumeOutcome:
    raised: str | None  # exception type name if resume crashed, else None
    resumed_from_step: int | None  # step training actually continued from (None if it raised)
    newest_complete_step: int  # newest checkpoint fully written before the crash
    weights_diff: float | None  # max |w - w_reference| once the run finishes; None if it raised
    ranks_agree: bool = True  # every rank resumed from the same step and ended with the same weights


def classify(o: ResumeOutcome) -> Verdict:
    if o.raised is not None:
        return "HARD_FAIL"
    if not o.ranks_agree or o.weights_diff is None or o.weights_diff > WEIGHT_TOL:
        return "SILENT_DIVERGENCE"
    if (o.resumed_from_step or 0) < o.newest_complete_step:
        return "LOST_WORK"
    return "PASS"
