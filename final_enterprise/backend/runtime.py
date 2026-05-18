from __future__ import annotations

import threading
from dataclasses import dataclass

from history_store import SessionHistoryStore
from seed_data import bootstrap_data
from workflow import GraphRuntime, build_graph_runtime


@dataclass
class AppRuntime:
    graph: GraphRuntime
    history: SessionHistoryStore


_runtime: AppRuntime | None = None
_init_lock = threading.Lock()
_init_thread: threading.Thread | None = None
_init_error: str | None = None
_ready = threading.Event()


def _do_init() -> None:
    global _runtime, _init_error
    try:
        bootstrap_data()
        rt = AppRuntime(graph=build_graph_runtime(), history=SessionHistoryStore())
        with _init_lock:
            _runtime = rt
            _init_error = None
        _ready.set()
    except Exception as exc:
        with _init_lock:
            _init_error = f"{type(exc).__name__}: {exc}"
        _ready.set()


def start_runtime_async() -> None:
    """后台加载 PDF/FAISS/图工作流，避免阻塞 /health。"""
    global _init_thread, _init_error
    with _init_lock:
        if _ready.is_set() and _runtime is not None:
            return
        if _init_thread is not None and _init_thread.is_alive():
            return
        if _init_error and _ready.is_set():
            return
        if not _ready.is_set():
            _ready.clear()
        _init_error = None
        _init_thread = threading.Thread(target=_do_init, name="runtime-init", daemon=True)
        _init_thread.start()


def ensure_init_started() -> None:
    """若初始化线程已退出但未就绪，自动重新拉起（避免 reload 后永远 loading）。"""
    if runtime_ready():
        return
    should_start = False
    with _init_lock:
        alive = _init_thread is not None and _init_thread.is_alive()
        if not alive and not _init_error:
            should_start = True
    if should_start:
        start_runtime_async()


def init_runtime() -> AppRuntime:
    """同步初始化（测试/脚本用）。"""
    global _runtime, _init_error
    _do_init()
    if _runtime is None:
        raise RuntimeError(_init_error or "runtime init failed")
    return _runtime


def runtime_ready() -> bool:
    return _ready.is_set() and _runtime is not None and _init_error is None


def runtime_status() -> dict[str, str | bool]:
    if not _ready.is_set():
        alive = _init_thread is not None and _init_thread.is_alive()
        return {"ready": False, "phase": "loading" if alive else "starting", "error": ""}
    if _init_error:
        return {"ready": False, "phase": "failed", "error": _init_error}
    return {"ready": True, "phase": "ok", "error": ""}


def get_runtime() -> AppRuntime:
    if not runtime_ready():
        st = runtime_status()
        if st.get("phase") == "loading":
            raise RuntimeError("runtime is still loading (PDF/FAISS index); retry in a few seconds")
        if st.get("error"):
            raise RuntimeError(str(st["error"]))
        raise RuntimeError("runtime not initialized")
    return _runtime  # type: ignore[return-value]


def reset_runtime() -> None:
    global _runtime, _init_thread, _init_error
    with _init_lock:
        if _runtime is not None:
            _runtime.graph.close()
        _runtime = None
        _init_error = None
        _init_thread = None
    _ready.clear()
