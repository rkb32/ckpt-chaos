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

With lazy_native=True (used for unmodified third-party scripts) the native
writers are patched right after torch / safetensors are imported, so scripts
that never import torch pay nothing.

Limits: audit hooks fire per file operation, not per write(), so Python-written
files (the JSON ones) are not torn yet.
"""
from __future__ import annotations

import builtins
import importlib.abc
import importlib.util
import io
import os
import sys

_MUTATING = {"os.rename", "os.remove", "os.mkdir", "os.rmdir", "shutil.copyfile", "shutil.rmtree"}
_ONE_PATH = {"open", "os.remove", "os.mkdir", "os.rmdir", "shutil.rmtree"}
_state: dict = {"n": 0, "crash_at": None, "mode": "before", "watch": "", "fd": None, "hooked": False,
                "py_open": False}


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
    if event == "open" and _state["py_open"] and isinstance(args[1], str):
        return  # builtin open(): the open() wrapper counts it, and can tear it
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

    wrapper._ckpt_chaos = True
    return wrapper


def _native_targets():
    out = []
    if "torch" in sys.modules and hasattr(sys.modules["torch"], "save"):
        out.append((sys.modules["torch"].save, "torch.save", 1, "f"))
    st = sys.modules.get("safetensors.torch")
    if st is not None and hasattr(st, "save_file"):
        out.append((st.save_file, "safetensors.save_file", 1, "filename"))
    return out


def _install_native_wrappers(force_import: bool = False) -> None:
    if force_import:
        for name in ("torch", "safetensors.torch"):  # make sure the modules exist before we patch them
            try:
                __import__(name)
            except ImportError:
                pass
    for orig, kind, pos, kw in _native_targets():
        if getattr(orig, "_ckpt_chaos", False):  # already wrapped: never count an event twice
            continue
        wrapper = _wrap(orig, kind, pos, kw)
        for m in list(sys.modules.values()):  # rebind every `from x import save` copy, not just the original name
            d = getattr(m, "__dict__", None)
            if isinstance(d, dict):
                for name, value in list(d.items()):
                    if value is orig:
                        d[name] = wrapper


class _PostImport(importlib.abc.MetaPathFinder):
    """Patch the native writers right after torch / safetensors.torch finish importing."""

    names = ("torch", "safetensors.torch")

    def __init__(self):
        self.busy: set[str] = set()

    def find_spec(self, name, path, target=None):
        if name not in self.names or name in self.busy:
            return None
        self.busy.add(name)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self.busy.discard(name)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        original = spec.loader.exec_module

        def exec_module(module):
            original(module)
            _install_native_wrappers()

        spec.loader.exec_module = exec_module
        return spec


class _TornFile:
    """A file object that, when closed, cuts the file to half its size and kills the process.

    That is what a crash in the middle of a Python-level write leaves behind, whatever library
    (json, pickle, numpy, yaml) did the writing.
    """

    def __init__(self, f, path: str):
        object.__setattr__(self, "_f", f)
        object.__setattr__(self, "_path", path)

    def __getattr__(self, name):
        return getattr(self._f, name)

    def __iter__(self):
        return iter(self._f)

    def __enter__(self):
        self._f.__enter__()
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._f.close()
        _tear(self._path)
        os._exit(137)


def _open_wrapper(orig):
    def wrapper(file, mode="r", *a, **k):
        if isinstance(file, (str, os.PathLike)) and isinstance(mode, str) and any(c in mode for c in "wax+") \
                and _under(file):
            tear = _event("open", file, tornable=True)
            f = orig(file, mode, *a, **k)
            return _TornFile(f, os.fspath(file)) if tear else f
        return orig(file, mode, *a, **k)

    wrapper._ckpt_chaos = True
    return wrapper


def _install_python_open() -> None:
    if getattr(builtins.open, "_ckpt_chaos", False):
        return
    wrapper = _open_wrapper(builtins.open)
    builtins.open = wrapper
    io.open = wrapper  # pathlib and friends call io.open


def arm(watch_dir: str, crash_at: int | None = None, mode: str = "before", log_path: str | None = None,
        lazy_native: bool = False, torn_python: bool = False) -> None:
    """Start counting mutations under watch_dir; crash at the Nth (1-based) if crash_at is set.

    torn_python also lets mode "torn" cut files written through Python's open() (not just torch.save).
    """
    _state.update(n=0, crash_at=crash_at, mode=mode, watch=_norm(watch_dir), py_open=torn_python)
    if torn_python:
        _install_python_open()
    if log_path:
        _state["fd"] = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    if lazy_native:
        _install_native_wrappers()  # in case torch is already imported
        sys.meta_path.insert(0, _PostImport())
    else:
        _install_native_wrappers(force_import=True)
    if not _state["hooked"]:  # audit hooks cannot be removed, so never add a second one
        sys.addaudithook(_hook)
        _state["hooked"] = True
