"""Crash-point injection for training scripts.

Call arm() inside the process under test. It counts file-mutating events under
`watch_dir` and hard-exits at the Nth one, like a SIGKILL in the middle of a
save. Two sources of events:

  * CPython audit events (open-for-write, rename, remove, mkdir, rmtree).
  * Calls to the native writers torch.save and safetensors' save_file, which
    write files from C/Rust and so raise no audit events.

Each event can be crashed in mode "before" (the file never appears) and, for
native writers, mode "torn" (the file appears, cut to half its size). Events
are logged to `log_path` so a report can name the exact boundary.

Limits: audit hooks fire per file operation, not per write(), so Python-written
files (the JSON ones) are not torn yet.
"""
from __future__ import annotations

import os
import sys

_MUTATING = {"os.rename", "os.remove", "os.mkdir", "os.rmdir", "shutil.copyfile", "shutil.rmtree"}
_ONE_PATH = {"open", "os.remove", "os.mkdir", "os.rmdir", "shutil.rmtree"}
_state: dict = {"n": 0, "crash_at": None, "mode": "before", "watch": "", "fd": None}


def _norm(p) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(p)))


def _under(p) -> bool:
    n, w = _norm(p), _state["watch"]
    return n == w or n.startswith(w + os.sep)


def _event(kind: str, path, tornable: bool) -> bool:
    """Count one mutation. Returns True if the caller must tear `path` after writing it."""
    _state["n"] += 1
    n = _state["n"]
    if _state["fd"] is not None:  # os.write raises no audit event, so no recursion
        line = f"{n}\t{kind}\t{os.path.basename(os.fspath(path))}\t{int(tornable)}\n"
        os.write(_state["fd"], line.encode())
    if _state["crash_at"] == n:
        if _state["mode"] == "torn" and tornable:
            return True
        os._exit(137)  # no cleanup, no flush: what a kill leaves behind
    return False


def _is_mutation(event: str, args: tuple) -> bool:
    if event == "open":
        path, mode, flags = args
        if not isinstance(path, (str, os.PathLike)):
            return False
        if mode is not None:
            return any(c in mode for c in "wax+")
        return bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
    return event in _MUTATING


def _hook(event: str, args: tuple) -> None:
    if event != "open" and event not in _MUTATING:
        return
    if not _is_mutation(event, args):
        return
    paths = args[:1] if event in _ONE_PATH else args[:2]
    for a in paths:
        if isinstance(a, (str, os.PathLike)) and _under(a):
            _event(event, a, tornable=False)
            return


def _tear(path: str) -> None:
    os.truncate(path, max(1, os.path.getsize(path) // 2))


def _wrap(orig, kind: str, pos: int, kw: str):
    def wrapper(*a, **k):
        p = a[pos] if len(a) > pos else k.get(kw)
        if isinstance(p, (str, os.PathLike)) and _under(p):
            tear = _event(kind, p, tornable=True)
            result = orig(*a, **k)
            if tear:
                _tear(os.fspath(p))
                os._exit(137)
            return result
        return orig(*a, **k)

    return wrapper


def _install_native_wrappers() -> None:
    for name in ("torch", "safetensors.torch"):  # make sure the modules exist before we patch them
        try:
            __import__(name)
        except ImportError:
            pass
    targets = []
    if "torch" in sys.modules:
        targets.append((sys.modules["torch"].save, "torch.save", 1, "f"))
    if "safetensors.torch" in sys.modules:
        targets.append((sys.modules["safetensors.torch"].save_file, "safetensors.save_file", 1, "filename"))
    for orig, kind, pos, kw in targets:
        wrapper = _wrap(orig, kind, pos, kw)
        for m in list(sys.modules.values()):  # rebind every `from x import save` copy, not just the original name
            d = getattr(m, "__dict__", None)
            if isinstance(d, dict):
                for name, value in list(d.items()):
                    if value is orig:
                        d[name] = wrapper


def arm(watch_dir: str, crash_at: int | None = None, mode: str = "before", log_path: str | None = None) -> None:
    """Start counting mutations under watch_dir; crash at the Nth (1-based) if crash_at is set."""
    _state.update(n=0, crash_at=crash_at, mode=mode, watch=_norm(watch_dir))
    if log_path:
        _state["fd"] = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    _install_native_wrappers()
    sys.addaudithook(_hook)
