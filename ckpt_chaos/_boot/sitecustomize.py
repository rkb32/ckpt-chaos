"""Arms the crash injector inside an unmodified third-party script.

ckpt-chaos puts this directory first on PYTHONPATH for the command under test. Python imports
`sitecustomize` at startup, so no change to the user's code is needed. Only the main process arms
(DataLoader workers and other children inherit CKPT_CHAOS_ARMED and skip it).
"""
import os
import sys

if os.environ.get("CKPT_CHAOS_WATCH") and not os.environ.get("CKPT_CHAOS_ARMED"):
    try:
        import multiprocessing

        if multiprocessing.current_process().name == "MainProcess":
            from ckpt_chaos.targets.common import arm_from_env

            arm_from_env(int(os.environ.get("RANK", 0)), lazy=True)
            os.environ["CKPT_CHAOS_ARMED"] = "1"
    except Exception as e:  # noqa: BLE001 - never break the user's program because arming failed
        sys.stderr.write(f"[ckpt-chaos] could not arm the injector: {e}\n")
