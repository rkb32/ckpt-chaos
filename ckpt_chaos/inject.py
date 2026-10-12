"""Crash-point injection for training scripts.

Call arm() inside the process under test. It counts file-mutating events under
`watch_dir` and hard-exits at the Nth one, like a SIGKILL in the middle of a
save. Two sources of events:

  * CPython audit events (open-for-write, rename, remove, mkdir, rmtree).
  * Calls to the native writers torch.save and safetensors' save_file, which
    write files from C/Rust and so raise no audit events.

Each event can be crashed in mode "before" (the file never appears), mode
"torn" (the file appears, cut to half its size; native writers, and Python
open() with torn_python) or mode "term" (the job gets SIGTERM and runs on until
it exits or the grace period CKPT_CHAOS_GRACE_S runs out, as in a preemption).
Events are logged to `log_path` so a report can name the exact boundary.

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
                "py_open": False, "quiet": False, "unsynced": False,
                "durable": {}, "dirty": set(), "fd_path": {}, "zero": set()}


class _quietly:
    """Our own file reads and writes during a rollback: not counted, not tracked."""

    def __enter__(self):
        self.was = _state["quiet"]
        _state["quiet"] = True

    def __exit__(self, *exc):
        _state["quiet"] = self.was


def _read_bytes(p: str):
    with _quietly():
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return f.read()


def _note_open_for_write(p: str) -> None:
    """Before the job truncates or creates p: remember what was on disk, if it was not tracked yet."""
    if p not in _state["durable"]:
        _state["durable"][p] = _read_bytes(p)  # treated as durable: it was there before this run wrote it
    _state["dirty"].add(p)


def _roll_back_unsynced() -> None:
    """Worst case of the POSIX rules: data not fsynced is lost. Every file written since its last fsync reverts
    to its last durable content, or disappears if it never had any. Not power loss: no other reordering.
    """
    with _quietly():
        for p in sorted(_state["dirty"]):
            if p in _state["zero"]:  # renamed into place before its data was synced: the name survives, the data does not
                if os.path.exists(p):
                    with open(p, "wb"):
                        pass
                continue
            base = _state["durable"].get(p)
            if base is None:
                if os.path.exists(p):
                    os.remove(p)
            else:
                with open(p, "wb") as f:
                    f.write(base)
    _state["dirty"].clear()
    _state["zero"].clear()


def _install_fsync() -> None:
    if getattr(os.fsync, "_ckpt_chaos", False):
        return
    orig = os.fsync

    def fsync(fd):
        result = orig(fd)  # the job's data is on disk now
        p = _state["fd_path"].get(fd)
        if p in _state["dirty"]:
            with _quietly():
                _state["durable"][p] = _read_bytes(p)
            _state["dirty"].discard(p)
        return result

    fsync._ckpt_chaos = True
    os.fsync = fsync

    def moved(orig):
        def rename(src, dst, *a, **k):
            result = orig(src, dst, *a, **k)  # the name moves; the data does not, if it was not synced
            s, d = _norm(src), _norm(dst)
            if s in _state["dirty"]:
                _state["dirty"].discard(s)
                _state["dirty"].add(d)
                _state["zero"].add(d)
            return result

        rename._ckpt_chaos = True
        return rename

    if not getattr(os.replace, "_ckpt_chaos", False):
        os.replace = moved(os.replace)
        os.rename = moved(os.rename)


def _norm(p) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(p)))


def _under(p) -> bool:
    n, w = _norm(p), _state["watch"]
    return n == w or n.startswith(w + os.sep)


def _event(kind: str, path, tornable: bool) -> bool:
    """Count one mutation. Returns True if the caller must tear `path` after writing it."""
    if _state["quiet"]:  # our own reads and rollbacks are not the job's events
        return False
    _state["n"] += 1
    n = _state["n"]
    if _state["fd"] is not None:  # os.write raises no audit event, so no recursion
        line = f"{n}\t{kind}\t{os.path.basename(os.fspath(path))}\t{int(tornable)}\n"
        os.write(_state["fd"], line.encode())
    if _state["crash_at"] == n:
        if _state["mode"] == "term":
            _terminate()
            return False
        if _state["mode"] == "unsynced":
            _roll_back_unsynced()
            os._exit(137)
        if _state["mode"] == "torn" and tornable:
            return True
        os._exit(137)  # no cleanup, no flush: what a kill leaves behind
    return False


def _terminate() -> None:
    """A preemption: SIGTERM now, then SIGKILL when the grace period runs out.

    A job with no SIGTERM handler dies at once (the default action). A job with a handler keeps running
    until it exits or the grace period (CKPT_CHAOS_GRACE_S, default 5 s) is over, and then it is killed.
    """
    import signal
    import threading

    try:  # parsed here, inside the job: a bad value must not raise into the user's code
        grace = float(os.environ.get("CKPT_CHAOS_GRACE_S", "5"))
    except ValueError:
        grace = 5.0
    signal.raise_signal(signal.SIGTERM)
    timer = threading.Timer(grace if grace > 0 else 5.0, os._exit, [137])
    timer.daemon = True
    timer.start()


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

    names = ("torch", "safetensors.torch", "h5py", "tensorstore")

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
            if module.__name__ == "h5py":
                _install_h5py_when_ready()
            elif module.__name__ == "tensorstore":
                _install_tensorstore()
            else:
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
            if _state["unsynced"]:
                _note_open_for_write(_norm(file))  # before the open truncates the old content
            f = orig(file, mode, *a, **k)
            if _state["unsynced"]:
                try:
                    _state["fd_path"][f.fileno()] = _norm(file)
                except (AttributeError, OSError, ValueError):
                    pass
            return _TornFile(f, os.fspath(file)) if tear else f
        return orig(file, mode, *a, **k)

    wrapper._ckpt_chaos = True
    return wrapper


def _install_h5py() -> None:
    """Count h5py.File writes under the watch dir, and tear them on close (a torn HDF5 file: half its bytes).

    Like torch.save, the file is written completely by the library, then cut and the process is killed, so
    what is torn is a real, complete HDF5 file that was cut in half. Patches the File class in place, so
    isinstance() and every existing reference keep working.
    """
    try:
        from h5py._hl import files as _files
    except ImportError:
        return
    cls = _files.File
    if getattr(cls.__init__, "_ckpt_chaos", False):
        return
    orig_init, orig_close = cls.__init__, cls.close

    def __init__(self, name, mode=None, *a, **k):
        if isinstance(name, (str, os.PathLike)) and mode in ("w", "w-", "x", "a", "r+") and _under(name):
            tear = _event("h5py.File", name, tornable=True)
            orig_init(self, name, mode, *a, **k)
            if tear:
                object.__setattr__(self, "_ckpt_tear", os.fspath(name))
            return
        orig_init(self, name, mode, *a, **k)

    def close(self):
        path = getattr(self, "_ckpt_tear", None)
        orig_close(self)  # the library writes the whole file here
        if path:
            _tear(path)
            os._exit(137)

    __init__._ckpt_chaos = True
    close._ckpt_chaos = True
    cls.__init__ = __init__
    cls.close = close


def _ts_file_path(spec) -> str | None:
    """The directory a tensorstore spec writes to, when it uses the file kvstore; None otherwise (not tracked)."""
    kv = spec.get("kvstore") if isinstance(spec, dict) else None
    if isinstance(kv, dict) and kv.get("driver") == "file" and isinstance(kv.get("path"), str):
        return os.path.abspath(kv["path"])
    return None


def _walk_stats(root: str) -> dict:
    found = {}
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            found[p] = (st.st_mtime_ns, st.st_size)
    return found


class _StoreProxy:
    """A tensorstore TensorStore whose writes are counted as file events. A write is synchronous here:
    "before" kills before the write starts; "torn" lets the write finish, then cuts the chunk file it changed to half
    its size and kills the process. Reads and everything else pass straight through."""

    def __init__(self, store, path: str):
        object.__setattr__(self, "_s", store)
        object.__setattr__(self, "_p", path)

    def __getattr__(self, name):
        return getattr(self._s, name)

    def __setitem__(self, key, value):
        self._counted(lambda: self._s.__setitem__(key, value))

    def write(self, value, *a, **k):
        def go():
            self._s.write(value, *a, **k).result()  # make it synchronous, so a torn write has finished bytes
        self._counted(go)

    def _counted(self, do):
        before = _walk_stats(self._p)
        tear = _event("tensorstore.write", self._p, tornable=True)
        do()
        if tear:
            changed = sorted(p for p, st in _walk_stats(self._p).items() if before.get(p) != st
                             and not p.endswith(".zarray") and not p.endswith(".zattrs"))
            if changed:
                _tear(changed[0])
            os._exit(137)


class _FutureProxy:
    def __init__(self, fut, path: str):
        object.__setattr__(self, "_f", fut)
        object.__setattr__(self, "_p", path)

    def __getattr__(self, name):
        return getattr(self._f, name)

    def result(self, *a, **k):
        return _StoreProxy(self._f.result(*a, **k), self._p)


def _install_tensorstore() -> None:
    try:
        import tensorstore as ts
    except ImportError:
        return
    if getattr(ts.open, "_ckpt_chaos", False):
        return
    orig_open = ts.open

    def open_(spec, *a, **k):
        fut = orig_open(spec, *a, **k)
        path = _ts_file_path(spec)
        if path is None:
            return fut
        return _FutureProxy(fut, path)

    open_._ckpt_chaos = True
    ts.open = open_


def _install_h5py_when_ready() -> None:
    # h5py.File lives in a submodule that the package imports after its own module body has run
    _install_h5py()


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
    _state.update(n=0, crash_at=crash_at, mode=mode, watch=_norm(watch_dir), py_open=torn_python,
                  unsynced=(mode == "unsynced"))
    if torn_python:
        _install_python_open()
    if mode == "unsynced":
        _install_fsync()  # must be in place before the job's first fsync
    if log_path:
        _state["fd"] = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    if "h5py" in sys.modules:  # already imported: patch now (an import later is caught by _PostImport)
        _install_h5py()
    if "tensorstore" in sys.modules:
        _install_tensorstore()
    if lazy_native:
        _install_native_wrappers()  # in case torch is already imported
        sys.meta_path.insert(0, _PostImport())
    else:
        _install_native_wrappers(force_import=True)
    if not _state["hooked"]:  # audit hooks cannot be removed, so never add a second one
        sys.addaudithook(_hook)
        _state["hooked"] = True
