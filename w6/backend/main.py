"""S&OP v6.0 — FastAPI：会话历史持久化（MySQL）+ Text-to-SQL 工具轨迹。

相对 w5 的关键变化：
- 不再依赖 InMemoryChatMessageHistory；`/session/history` 与 `/session/reset` 全部走 MySQL，
  因此后端**重启后刷新前端，依然能续上重启前的对话**。
- 工具可见性 `_tool_visible` 增加 `sql_db_*` 与 `sop_database_query_tool`，
  前端侧栏会同步看到「大模型自动生成的 SQL」。

为减少重复代码：通用回调（`tool_trace_callback`、`perf_callbacks`）通过 sys.path 复用 w5/backend
里的同名模块（在 `agent_runtime.py` 里完成路径注入）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from agent_runtime import (
    LLMInvocationCounter,  # 透传自 w5/backend/perf_callbacks（importlib 唯一别名加载）
    ListToolTraceCallback,  # 透传自 w5/backend/tool_trace_callback
    get_runtime,
    init_runtime,
)

_log = logging.getLogger("uvicorn.error")


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="前端维护的会话 ID")
    message: str = Field(..., min_length=1, description="用户输入")


class SessionResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


# ----------------------------------------------------------------------------- 工具函数

def _message_content_str(m: object) -> str:
    if m is None:
        return ""
    t = getattr(m, "text", None)
    if isinstance(t, str) and t.strip():
        return t
    c = getattr(m, "content", None)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for p in c:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if p.get("type") == "text" and isinstance(p.get("text"), str):
                    parts.append(p["text"])
                elif isinstance(p.get("text"), str):
                    parts.append(p["text"])
        return "".join(parts)
    return str(c) if c is not None else ""


def _agent_invoke_output_text(out: object) -> str:
    if out is None:
        return ""
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        inner = out.get("output")
        if inner is not None:
            return _agent_invoke_output_text(inner)
        try:
            return json.dumps(out, ensure_ascii=False)
        except TypeError:
            return str(out)
    if isinstance(out, (HumanMessage, AIMessage)) or hasattr(out, "content"):
        return _message_content_str(out)
    return str(out)


def _ui_messages_from_history(messages: list) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            c = _message_content_str(m).strip()
            if c:
                rows.append({"role": "user", "content": c})
        elif isinstance(m, AIMessage):
            c = _message_content_str(m).strip()
            if c:
                rows.append({"role": "assistant", "content": c})
    return rows


def _sse_payload(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _tool_visible(name: str) -> bool:
    """决定哪些工具事件下发到前端。

    - `sop_*`：主 Agent 直接调用的工具（含 v6 新工具 `sop_database_query_tool`）。
    - `sql_db_*`：内层 SQL Agent 的标准工具（list_tables / schema / query / query_checker）。
      其中 `sql_db_query` 携带的就是大模型最终生成的 SQL，作业要求必须打印出来。
    - `python_repl`：保留兼容（w4 Pandas 子 Agent 已弃用，但本地调试时可能还存在）。
    """
    if not name:
        return False
    if name.startswith("sop_"):
        return True
    if name.startswith("sql_db_"):
        return True
    if name in ("python_repl", "PythonREPL"):
        return True
    return False


def _chunk_text(chunk: object) -> str:
    if chunk is None:
        return ""
    text_attr = getattr(chunk, "text", None)
    if text_attr is not None:
        s = str(text_attr)
        if s:
            return s
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if p.get("type") == "text" and "text" in p:
                    parts.append(str(p["text"]))
                elif isinstance(p.get("text"), str):
                    parts.append(str(p["text"]))
        return "".join(parts)
    return ""


def _chain_output_user_text(output: object) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, dict):
        if isinstance(output.get("output"), str):
            inner = output["output"].strip()
            if inner:
                return inner
        if isinstance(output.get("output"), dict):
            return _chain_output_user_text(output["output"])
    return ""


def _message_is_tool_calling_turn(msg: object) -> bool:
    if msg is None:
        return False
    tc = getattr(msg, "tool_calls", None) or []
    if tc:
        return True
    itc = getattr(msg, "invalid_tool_calls", None) or []
    return bool(itc)


def _emit_fallback_token(text: str, *, stream_chars: int, fallback_sent: list[bool]) -> str | None:
    if not text or stream_chars > 0 or fallback_sent[0]:
        return None
    fallback_sent[0] = True
    return text


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


def _tool_obs_max_chars() -> int:
    try:
        return max(400, int(os.getenv("SOP_TOOL_OBS_MAX_CHARS", "8000")))
    except ValueError:
        return 8000


def _exception_chain_detail(exc: BaseException, *, max_len: int = 8000) -> str:
    parts: list[str] = [f"{type(exc).__name__}: {exc}"]
    cur: BaseException | None = exc.__cause__
    for _ in range(8):
        if cur is None:
            break
        parts.append(f"caused by: {type(cur).__name__}: {cur}")
        cur = cur.__cause__
    s = " | ".join(parts)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


# ----------------------------------------------------------------------------- SSE

async def _yield_assistant_reply_sse(text: str) -> AsyncIterator[str]:
    if not text:
        return
    if not _env_flag("SOP_SSE_ASSISTANT_TYPING", default="0"):
        yield _sse_payload({"type": "token", "text": text})
        return
    try:
        chunk_sz = max(1, int(os.getenv("SOP_SSE_TYPING_CHARS", "8192")))
    except ValueError:
        chunk_sz = 8192
    try:
        delay_ms = max(0, int(os.getenv("SOP_SSE_TYPING_DELAY_MS", "0")))
    except ValueError:
        delay_ms = 0

    n = len(text)
    i = 0
    while i < n:
        piece = text[i : i + chunk_sz]
        i += chunk_sz
        yield _sse_payload({"type": "token", "text": piece})
        if i < n and delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)


async def _invoke_parity_sse(session_id: str, message: str) -> AsyncIterator[str]:
    rt = get_runtime()
    trace = ListToolTraceCallback()
    cfg = {
        "configurable": {"session_id": session_id},
        "callbacks": [trace],
    }

    def run():
        return rt.agent.invoke({"input": message}, config=cfg)

    try:
        out = await asyncio.to_thread(run)
    except Exception as e:
        _log.exception("SSE invoke_parity invoke failed")
        msg = _exception_chain_detail(e, max_len=4000)
        yield _sse_payload({"type": "error", "message": msg})
        return

    stack: list[str] = []
    for row in trace.entries:
        phase = row.get("phase")
        if phase == "start":
            t = str(row.get("tool") or "")
            stack.append(t)
            if not _tool_visible(t):
                continue
            inp = str(row.get("inputs") or "")
            yield _sse_payload({"type": "tool_start", "tool": t, "inputs": inp[:3000]})
        elif phase == "end":
            t = stack.pop() if stack else ""
            if not _tool_visible(t):
                continue
            obs = str(row.get("observation") or "")
            mobs = _tool_obs_max_chars()
            yield _sse_payload({"type": "tool_end", "tool": t, "observation": obs[:mobs]})

    text = _agent_invoke_output_text(out)
    if text:
        async for line in _yield_assistant_reply_sse(text):
            yield line
    yield _sse_payload({"type": "done"})


async def _agent_event_stream(session_id: str, message: str) -> AsyncIterator[str]:
    rt = get_runtime()
    cfg = {"configurable": {"session_id": session_id}}
    stream_chars = 0
    fallback_sent: list[bool] = [False]
    # w6 使用 SQLChatMessageHistory（同步模式）时，直接走 astream_events 会触发：
    # "Attempting to use an async method when sync mode is turned on"。
    # 因此默认改为 0（invoke 后回放 SSE），如需实验异步事件流可手动设为 1。
    if not _env_flag("SOP_SSE_ASTREAM_EVENTS", default="0"):
        async for line in _invoke_parity_sse(session_id, message):
            yield line
        return

    try:
        async for ev in rt.agent.astream_events({"input": message}, config=cfg, version="v2"):
            kind = ev.get("event")
            name = ev.get("name") or ""

            if kind == "on_chat_model_stream":
                chunk = (ev.get("data") or {}).get("chunk")
                text = _chunk_text(chunk)
                if text:
                    stream_chars += len(text)
                    yield _sse_payload({"type": "token", "text": text})

            elif kind == "on_chat_model_end":
                out = (ev.get("data") or {}).get("output")
                if not _message_is_tool_calling_turn(out):
                    end_text = _chunk_text(out)
                    fb = _emit_fallback_token(
                        end_text, stream_chars=stream_chars, fallback_sent=fallback_sent
                    )
                    if fb:
                        yield _sse_payload({"type": "token", "text": fb})

            elif kind == "on_chain_stream":
                chunk = (ev.get("data") or {}).get("chunk")
                if isinstance(chunk, dict) and not any(
                    k in chunk for k in ("intermediate_step", "actions", "steps")
                ):
                    user_txt = _chain_output_user_text(chunk)
                    fb = _emit_fallback_token(
                        user_txt, stream_chars=stream_chars, fallback_sent=fallback_sent
                    )
                    if fb:
                        yield _sse_payload({"type": "token", "text": fb})

            elif kind == "on_tool_start":
                if not _tool_visible(name):
                    continue
                data = ev.get("data") or {}
                inp = data.get("input")
                if isinstance(inp, dict):
                    try:
                        inp_s = json.dumps(inp, ensure_ascii=False)
                    except TypeError:
                        inp_s = str(inp)
                else:
                    inp_s = str(inp) if inp is not None else ""
                yield _sse_payload(
                    {"type": "tool_start", "tool": name, "inputs": inp_s[:3000]}
                )

            elif kind == "on_tool_end":
                if not _tool_visible(name):
                    continue
                data = ev.get("data") or {}
                out = data.get("output")
                obs = out if isinstance(out, str) else (str(out) if out is not None else "")
                mobs = _tool_obs_max_chars()
                yield _sse_payload(
                    {"type": "tool_end", "tool": name, "observation": obs[:mobs]}
                )

        yield _sse_payload({"type": "done"})
    except Exception as e:
        yield _sse_payload(
            {"type": "error", "message": _exception_chain_detail(e, max_len=4000)}
        )


# ----------------------------------------------------------------------------- Lifespan / App

@asynccontextmanager
async def lifespan(app: FastAPI):
    docs_dir = os.getenv("SOP_DOCS_DIR") or None
    recursive = os.getenv("SOP_RECURSIVE", "").lower() in ("1", "true", "yes")
    force_rebuild = os.getenv("SOP_FORCE_REBUILD", "").lower() in ("1", "true", "yes")
    llm_streaming = _env_flag("SOP_LLM_STREAMING", default="1")
    sse_astream = _env_flag("SOP_SSE_ASTREAM_EVENTS", default="0")
    sse_typing = _env_flag("SOP_SSE_ASSISTANT_TYPING", default="0")
    await asyncio.to_thread(
        init_runtime,
        docs_dir=docs_dir,
        recursive=recursive,
        force_rebuild=force_rebuild,
        llm_streaming=llm_streaming,
    )
    httpx_te = os.getenv("SOP_HTTPX_TRUST_ENV", "1")
    print(
        "[sop-hub-v6] runtime: "
        f"SOP_LLM_STREAMING={int(llm_streaming)} "
        f"SOP_SSE_ASTREAM_EVENTS={int(sse_astream)} "
        f"SOP_SSE_ASSISTANT_TYPING={int(sse_typing)} "
        f"SOP_HTTPX_TRUST_ENV={httpx_te!r} "
        "(MySQL 持久化记忆 + Text-to-SQL 工具 已启用)",
        flush=True,
    )
    yield


app = FastAPI(
    title="S&OP 决策中枢 API",
    version="6.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("SOP_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "sop-hub-v6"}


@app.post("/debug/timing")
async def debug_timing(
    req: ChatRequest,
    include_astream: bool = Query(False, description="额外跑一遍 astream_events 以统计事件数（多一次完整 Agent 费用）"),
):
    if not _env_flag("SOP_DEBUG_TIMING", default="0"):
        raise HTTPException(status_code=403, detail="Set environment variable SOP_DEBUG_TIMING=1 to enable.")

    rt = get_runtime()
    cb = LLMInvocationCounter()
    cfg: dict = {"configurable": {"session_id": req.session_id}, "callbacks": [cb]}

    def run():
        return rt.agent.invoke({"input": req.message}, config=cfg)

    t0 = time.perf_counter()
    try:
        out = await asyncio.to_thread(run)
    except Exception as e:
        raise HTTPException(status_code=500, detail=_exception_chain_detail(e)) from e
    invoke_s = time.perf_counter() - t0
    text = _agent_invoke_output_text(out)

    body: dict = {
        "session_id": req.session_id,
        "invoke_seconds": round(invoke_s, 4),
        "llm_starts": cb.llm_starts,
        "tool_starts": cb.tool_starts,
        "output_chars": len(text or ""),
    }

    if include_astream:
        sid2 = f"{req.session_id}-debug-astream"
        cb2 = LLMInvocationCounter()
        cfg2: dict = {"configurable": {"session_id": sid2}, "callbacks": [cb2]}
        kinds: Counter[str] = Counter()
        t1 = time.perf_counter()
        try:
            async for ev in rt.agent.astream_events(
                {"input": req.message}, config=cfg2, version="v2"
            ):
                k = ev.get("event")
                if isinstance(k, str):
                    kinds[k] += 1
        except Exception as e:
            body["astream_error"] = str(e)
        else:
            body["astream_seconds"] = round(time.perf_counter() - t1, 4)
            body["astream_llm_starts"] = cb2.llm_starts
            body["astream_tool_starts"] = cb2.tool_starts
            body["astream_event_total"] = int(sum(kinds.values()))
            body["astream_on_chat_model_stream"] = int(kinds.get("on_chat_model_stream", 0))
            body["astream_event_top"] = dict(kinds.most_common(15))

    return body


@app.post("/chat")
async def chat(req: ChatRequest):
    rt = get_runtime()
    trace = ListToolTraceCallback()
    cfg = {
        "configurable": {"session_id": req.session_id},
        "callbacks": [trace],
    }

    def run():
        return rt.agent.invoke({"input": req.message}, config=cfg)

    try:
        out = await asyncio.to_thread(run)
    except Exception as e:
        _log.exception("POST /chat invoke failed")
        raise HTTPException(status_code=500, detail=_exception_chain_detail(e)) from e

    text = _agent_invoke_output_text(out)
    return {
        "session_id": req.session_id,
        "output": text,
        "tool_trace": trace.entries,
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(
        _agent_event_stream(req.session_id, req.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/session/reset")
async def session_reset(req: SessionResetRequest):
    rt = get_runtime()
    rt.clear_session(req.session_id)
    return {"ok": True, "session_id": req.session_id}


@app.get("/session/history")
async def session_history(session_id: str = Query(..., min_length=1)):
    """从 MySQL 读取该会话历史。即使后端重启过，只要传相同 session_id 就能续上。"""
    rt = get_runtime()
    msgs = rt.read_messages(session_id)
    return {
        "session_id": session_id,
        "messages": _ui_messages_from_history(list(msgs)),
    }
