from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from runtime import ensure_init_started, get_runtime, reset_runtime, runtime_ready, runtime_status
from config import DATA_DIR, Settings
from report_synthesis import llm_status
from database import get_readonly_engine
from sqlalchemy import text


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class ResumeRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    decision: str = Field(..., min_length=1)
    feedback: str = ""


class ResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


def _build_context_window(messages: list[dict], max_messages: int = 10) -> str:
    if not messages:
        return ""
    rows: list[str] = []
    for m in messages[-max_messages:]:
        role = str((m or {}).get("role", "")).strip().lower()
        content = str((m or {}).get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            rows.append(f"用户: {content}")
        elif role == "assistant":
            rows.append(f"助手: {content}")
    return "\n".join(rows).strip()


def _brief(result: dict) -> dict:
    st = result.get("state") or {}
    return {
        "intent": st.get("intent"),
        "route_plan": st.get("route_plan"),
        "route_source": st.get("route_source"),
        "route_confidence": st.get("route_confidence"),
        "route_reason": st.get("route_reason"),
        "context_used": st.get("context_used"),
        "requires_human_approval": st.get("requires_human_approval"),
        "approval_decision": st.get("approval_decision"),
        "node_trace_len": len(result.get("node_trace") or []),
    }


def _source_catalog() -> dict:
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(p.name for p in data_dir.glob("*.pdf"))
    csvs = sorted(p.name for p in data_dir.glob("*.csv"))
    return {"pdf_files": pdfs, "csv_files": csvs}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from runtime import start_runtime_async

    start_runtime_async()
    print("[final-enterprise] loading runtime in background (PDF/FAISS)...", flush=True)
    try:
        yield
    finally:
        reset_runtime()


app = FastAPI(title="Agentic S&OP Final Enterprise API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """存活探针：立即返回，不等待索引加载。"""
    return {"status": "ok", "service": "final-enterprise"}


@app.get("/health/ready")
async def health_ready():
    """就绪探针：索引与工作流加载完成后才 ready。"""
    ensure_init_started()
    st = runtime_status()
    body = {"status": "ready" if runtime_ready() else "loading", **st}
    if not runtime_ready():
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=503, content=body)
    return body


@app.get("/health/llm")
async def health_llm():
    if not runtime_ready():
        return {"configured": False, "reachable": False, "detail": "runtime still loading"}
    st = Settings()
    return llm_status(st)


def _ensure_ready() -> None:
    if not runtime_ready():
        st = runtime_status()
        raise HTTPException(
            status_code=503,
            detail=st.get("error") or "服务正在加载制度索引与检索引擎，请 10–30 秒后重试",
        )


@app.post("/graph/chat")
async def graph_chat(req: ChatRequest):
    _ensure_ready()
    rt = get_runtime()
    prev_messages = rt.history.read(req.session_id)
    context_window = _build_context_window(prev_messages, max_messages=10)
    rt.history.add(req.session_id, "user", req.message)
    try:
        result = rt.graph.invoke(req.session_id, req.message, context_query=context_window)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    if result.get("status") == "completed":
        out = str(result.get("output") or "")
        if out.strip():
            rt.history.add(req.session_id, "assistant", out)

    return {
        "session_id": req.session_id,
        "status": result.get("status"),
        "output": result.get("output"),
        "interrupt": result.get("interrupt"),
        "node_trace": result.get("node_trace", []),
        "sources": result.get("state", {}).get("sources", []),
        "brief": _brief(result),
    }


@app.post("/graph/resume")
async def graph_resume(req: ResumeRequest):
    _ensure_ready()
    rt = get_runtime()
    try:
        payload: dict[str, str] = {"action": req.decision, "feedback": req.feedback or ""}
        result = rt.graph.resume(req.session_id, payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    if result.get("status") == "completed":
        out = str(result.get("output") or "")
        if out.strip():
            rt.history.add(req.session_id, "assistant", out)

    return {
        "session_id": req.session_id,
        "status": result.get("status"),
        "output": result.get("output"),
        "interrupt": result.get("interrupt"),
        "node_trace": result.get("node_trace", []),
        "sources": result.get("state", {}).get("sources", []),
        "brief": _brief(result),
    }


@app.get("/graph/state")
async def graph_state(session_id: str = Query(..., min_length=1)):
    _ensure_ready()
    rt = get_runtime()
    try:
        result = rt.graph.get_state(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e
    return {
        "session_id": session_id,
        "status": result.get("status"),
        "output": result.get("output"),
        "interrupt": result.get("interrupt"),
        "node_trace": result.get("node_trace", []),
        "sources": result.get("state", {}).get("sources", []),
        "brief": _brief(result),
        "state": result.get("state", {}),
    }


@app.get("/graph/pending_approvals")
async def graph_pending_approvals(limit: int = Query(30, ge=1, le=200)):
    _ensure_ready()
    rt = get_runtime()
    sessions = rt.history.list_sessions(limit=limit)
    items = []
    for row in sessions:
        sid = row["session_id"]
        result = rt.graph.get_state(sid)
        if result.get("status") != "interrupted":
            continue
        intr = result.get("interrupt") or {}
        items.append({"session_id": sid, "reason": intr.get("reason"), "brief": _brief(result)})
    return {"count": len(items), "items": items}


@app.get("/session/history")
async def session_history(session_id: str = Query(..., min_length=1)):
    _ensure_ready()
    rt = get_runtime()
    return {"session_id": session_id, "messages": rt.history.read(session_id)}


@app.get("/session/list")
async def session_list(limit: int = Query(30, ge=1, le=200)):
    _ensure_ready()
    rt = get_runtime()
    rows = rt.history.list_sessions(limit)
    return {"count": len(rows), "items": rows}


@app.get("/sources/catalog")
async def sources_catalog():
    cfg = Settings()
    with_tables: dict[str, object] = {"tables": [cfg.sales_table, cfg.mysql_history_table]}
    try:
        with get_readonly_engine().connect() as conn:
            row_count = int(conn.execute(text(f"SELECT COUNT(*) FROM {cfg.sales_table}")).scalar() or 0)
            sample = conn.execute(text(f"SELECT * FROM {cfg.sales_table} LIMIT 2")).mappings().all()
        with_tables["sales_row_count"] = row_count
        with_tables["sales_sample"] = [dict(r) for r in sample]
    except Exception as e:
        with_tables["sales_error"] = f"{type(e).__name__}: {e}"
    return {**_source_catalog(), **with_tables}


@app.post("/session/reset")
async def session_reset(req: ResetRequest):
    _ensure_ready()
    rt = get_runtime()
    rt.history.clear(req.session_id)
    return {"ok": True, "session_id": req.session_id}
