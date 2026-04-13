"""S&OP v5.0 — FastAPI：/chat（JSON）与 /chat/stream（SSE）。"""

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

from agent_runtime import get_runtime, init_runtime
from perf_callbacks import LLMInvocationCounter
from tool_trace_callback import ListToolTraceCallback

_log = logging.getLogger("uvicorn.error")


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="前端维护的会话 ID")
    message: str = Field(..., min_length=1, description="用户输入")


class SessionResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


def _message_content_str(m: object) -> str:
    """兼容不同版本 LangChain：content 为 str / list[dict] 等。"""
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
    """AgentExecutor / Runnable 的 invoke 结果转成可 JSON 序列化的 str（避免 AIMessage 等导致 500）。"""
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
    """将 LangChain 记忆转成前端 ChatGPT 式 user/assistant 列表（跳过 tool 等）。"""
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
    if not name:
        return False
    if name.startswith("sop_"):
        return True
    # Pandas 子 Agent 的代码执行步，便于观察「数据分析」内部在跑代码
    if name in ("python_repl", "PythonREPL"):
        return True
    return False


def _chunk_text(chunk: object) -> str:
    """从 AIMessageChunk 等对象中提取可展示的纯文本 token。"""
    if chunk is None:
        return ""
    # BaseMessage / Chunk 的 .text（标准 text block）；兼容部分国产 API 只填满此处
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
                # 少数网关把正文放在非标准键里，尽力抽取
                elif isinstance(p.get("text"), str):
                    parts.append(str(p["text"]))
        return "".join(parts)
    return ""


def _chain_output_user_text(output: object) -> str:
    """从 AgentExecutor / Runnable 的 chain 输出里取出面向用户的 output 字符串。"""
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
    """该轮 LLM 是否在发起 tool_calls（此时不应把整段当作对用户可见的最终回答）。"""
    if msg is None:
        return False
    tc = getattr(msg, "tool_calls", None) or []
    if tc:
        return True
    itc = getattr(msg, "invalid_tool_calls", None) or []
    return bool(itc)


def _emit_fallback_token(
    text: str,
    *,
    stream_chars: int,
    fallback_sent: list[bool],
) -> str | None:
    """仅在本轮尚未收到任何流式 token 时回传整段正文，避免与打字机重复。"""
    if not text or stream_chars > 0 or fallback_sent[0]:
        return None
    fallback_sent[0] = True
    return text


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


def _tool_obs_max_chars() -> int:
    """工具观测写入 SSE / 与前端 SOP_TOOL_OBS_MAX_CHARS 对齐，避免无谓截断。"""
    try:
        return max(400, int(os.getenv("SOP_TOOL_OBS_MAX_CHARS", "8000")))
    except ValueError:
        return 8000


def _exception_chain_detail(exc: BaseException, *, max_len: int = 8000) -> str:
    """OpenAI/httpx 常把根因放在 __cause__，只 str(exc) 会得到笼统的 Connection error。"""
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


async def _yield_assistant_reply_sse(text: str) -> AsyncIterator[str]:
    """仅对用户可见的最终回答做打字机式 SSE（多段 token）；不流式拆解工具观测等。

    注意：分块过小 + 每块 sleep + Streamlit 每个 token 都重绘，会把总耗时的「体感」拖到分钟级；
    默认用大分块、零延迟，只做轻量分段，避免与命令行 print 一次差数量级。
    """
    if not text:
        return
    # 默认关：48 字一块会产生海量 SSE 行，Streamlit iter_lines + markdown 会反压整条链路（比 w4 慢一个数量级）
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
    """与 `POST /chat`、w4 命令行相同的 `invoke` 路径；结束后用 SSE 回放工具与正文（总耗时应一致）。"""
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
            yield _sse_payload(
                {"type": "tool_end", "tool": t, "observation": obs[:mobs]}
            )

    text = _agent_invoke_output_text(out)
    if text:
        async for line in _yield_assistant_reply_sse(text):
            yield line
    yield _sse_payload({"type": "done"})


async def _agent_event_stream(session_id: str, message: str) -> AsyncIterator[str]:
    rt = get_runtime()
    cfg = {"configurable": {"session_id": session_id}}
    stream_chars = 0
    # list 以便在闭包/辅助函数中可变
    fallback_sent: list[bool] = [False]
    # 默认 A 方案：astream_events 真 token；若需与 w4 invoke 同路径可设 SOP_SSE_ASTREAM_EVENTS=0
    if not _env_flag("SOP_SSE_ASTREAM_EVENTS", default="1"):
        async for line in _invoke_parity_sse(session_id, message):
            yield line
        return

    try:
        async for ev in rt.agent.astream_events(
            {"input": message},
            config=cfg,
            version="v2",
        ):
            kind = ev.get("event")
            name = ev.get("name") or ""

            if kind == "on_chat_model_stream":
                chunk = (ev.get("data") or {}).get("chunk")
                text = _chunk_text(chunk)
                if text:
                    stream_chars += len(text)
                    yield _sse_payload({"type": "token", "text": text})

            elif kind == "on_chat_model_end":
                # 兼容：厂商已计费、流式 delta 未进 LangChain 回调时，整段只在 end 里
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
                    {
                        "type": "tool_start",
                        "tool": name,
                        "inputs": inp_s[:3000],
                    }
                )

            elif kind == "on_tool_end":
                if not _tool_visible(name):
                    continue
                data = ev.get("data") or {}
                out = data.get("output")
                obs = out if isinstance(out, str) else (str(out) if out is not None else "")
                mobs = _tool_obs_max_chars()
                yield _sse_payload(
                    {
                        "type": "tool_end",
                        "tool": name,
                        "observation": obs[:mobs],
                    }
                )

        yield _sse_payload({"type": "done"})
    except Exception as e:
        yield _sse_payload(
            {"type": "error", "message": _exception_chain_detail(e, max_len=4000)}
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    docs_dir = os.getenv("SOP_DOCS_DIR") or None
    recursive = os.getenv("SOP_RECURSIVE", "").lower() in ("1", "true", "yes")
    force_rebuild = os.getenv("SOP_FORCE_REBUILD", "").lower() in ("1", "true", "yes")
    # 默认 A 方案：主模型 streaming=True 供 on_chat_model_stream；追求 w4 速度可设 SOP_LLM_STREAMING=0
    llm_streaming = _env_flag("SOP_LLM_STREAMING", default="1")
    sse_astream = _env_flag("SOP_SSE_ASTREAM_EVENTS", default="1")
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
        "[sop-hub-v5] runtime: "
        f"SOP_LLM_STREAMING={int(llm_streaming)} "
        f"SOP_SSE_ASTREAM_EVENTS={int(sse_astream)} "
        f"SOP_SSE_ASSISTANT_TYPING={int(sse_typing)} "
        f"SOP_HTTPX_TRUST_ENV={httpx_te!r} (若 Connection error 可试设为 0 忽略系统代理)",
        flush=True,
    )
    yield


app = FastAPI(
    title="S&OP 决策中枢 API",
    version="5.0.0",
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
    return {"status": "ok", "service": "sop-hub-v5"}


@app.post("/debug/timing")
async def debug_timing(
    req: ChatRequest,
    include_astream: bool = Query(
        False,
        description="额外跑一遍 astream_events 并统计事件量（多一次完整 Agent 费用）",
    ),
):
    """启用：SOP_DEBUG_TIMING=1。返回 invoke 墙钟与 LLM/工具启动次数；可选对比 astream_events。"""
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
        raise HTTPException(
            status_code=500, detail=_exception_chain_detail(e)
        ) from e
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
                {"input": req.message},
                config=cfg2,
                version="v2",
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
    """非流式：完整回答 + 工具轨迹（便于 Postman / curl）。"""
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
        raise HTTPException(
            status_code=500,
            detail=_exception_chain_detail(e),
        ) from e

    text = _agent_invoke_output_text(out)
    return {
        "session_id": req.session_id,
        "output": text,
        "tool_trace": trace.entries,
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE：token 流 + tool_start/tool_end 事件，供前端打字机与侧边栏。"""
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
    rt.session_store.pop(req.session_id, None)
    return {"ok": True, "session_id": req.session_id}


@app.get("/session/history")
async def session_history(session_id: str = Query(..., min_length=1)):
    """供前端刷新后恢复对话：与 Agent 共用 InMemoryChatMessageHistory。"""
    rt = get_runtime()
    hist = rt.session_store.get(session_id)
    if hist is None:
        return {"session_id": session_id, "messages": []}
    return {
        "session_id": session_id,
        "messages": _ui_messages_from_history(list(hist.messages)),
    }
