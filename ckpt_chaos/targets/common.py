"""Plumbing shared by every target."""
from __future__ import annotations

import os


def arm_from_env(rank: int, lazy: bool = False) -> None:
    """Arm the injector from CKPT_CHAOS_* variables. Every rank logs; only the chosen rank crashes."""
    watch = os.environ.get("CKPT_CHAOS_WATCH")
    if not watch:
        return
    from ckpt_chaos import inject

    log = os.environ.get("CKPT_CHAOS_LOG")
    crash_at = int(os.environ.get("CKPT_CHAOS_CRASH_AT") or 0) or None
    crash_rank = int(os.environ.get("CKPT_CHAOS_CRASH_RANK", "0"))
    inject.arm(
        watch,
        crash_at=crash_at if rank == crash_rank else None,
        mode=os.environ.get("CKPT_CHAOS_MODE", "before"),
        log_path=f"{log}.rank{rank}" if log else None,
        lazy_native=lazy,
        torn_python=os.environ.get("CKPT_CHAOS_TORN_PYTHON") == "1",
    )
