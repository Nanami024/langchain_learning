"""
S&OP v5.0 — Streamlit 前端：ChatGPT 式对话 + SSE 打字机 + 工具轨迹。

默认调用本机 FastAPI（见侧边栏配置）。请先启动 w5/backend。
"""

from __future__ import annotations

import json
import os
import time
import uuid

import requests
import streamlit as st

DEFAULT_API = os.getenv("SOP_API_URL", "http://127.0.0.1:8000")


def _fetch_session_history(api_base: str, session_id: str) -> list[dict]:
    try:
        r = requests.get(
            f"{api_base.rstrip('/')}/session/history",
            params={"session_id": session_id},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        msgs = data.get("messages") or []
        return [m for m in msgs if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    except requests.RequestException:
        return []


def _reset_backend_session(api_base: str, session_id: str) -> None:
    try:
        r = requests.post(
            f"{api_base.rstrip('/')}/session/reset",
            json={"session_id": session_id},
            timeout=30,
        )
        r.raise_for_status()
    except requests.RequestException:
        pass


def _chat_non_stream(api_base: str, session_id: str, message: str) -> tuple[str, str]:
    r = requests.post(
        f"{api_base.rstrip('/')}/chat",
        json={"session_id": session_id, "message": message},
        timeout=600,
    )
    r.raise_for_status()
    data = r.json()
    out = data.get("output", "")
    trace = data.get("tool_trace") or []
    trace_md = ""
    for row in trace:
        if row.get("phase") == "start":
            trace_md += f"\n**▶ `{row.get('tool')}`**\n```\n{row.get('inputs', '')}\n```\n"
        elif row.get("phase") == "end":
            trace_md += f"\n**◀ 观测**\n```\n{row.get('observation', '')}\n```\n"
    return out, trace_md


def _chat_stream(
    api_base: str,
    session_id: str,
    message: str,
    placeholder,
    tool_placeholder,
    tool_sidebar_placeholder=None,
) -> tuple[str, str]:
    assistant = ""
    tool_md = ""
    # Streamlit 每次 markdown 重绘很贵；对每个 SSE token 都刷新会把长回答拖到分钟级（w4 只 print 一次）
    _last_flush = 0.0
    _last_flush_len = 0
    try:
        _ui_min_interval = float(os.getenv("SOP_UI_STREAM_MIN_INTERVAL", "0.12"))
    except ValueError:
        _ui_min_interval = 0.12
    try:
        _ui_min_chars = int(os.getenv("SOP_UI_STREAM_MIN_CHARS", "40"))
    except ValueError:
        _ui_min_chars = 40

    def _maybe_flush_tokens(*, force: bool = False) -> None:
        nonlocal _last_flush, _last_flush_len
        now = time.monotonic()
        grown = len(assistant) - _last_flush_len
        if force or grown >= _ui_min_chars or (now - _last_flush >= _ui_min_interval and grown > 0):
            placeholder.markdown(assistant + "▌")
            _last_flush = now
            _last_flush_len = len(assistant)

    url = f"{api_base.rstrip('/')}/chat/stream"
    with requests.post(
        url,
        json={"session_id": session_id, "message": message},
        stream=True,
        timeout=600,
    ) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            try:
                obj = json.loads(raw[6:])
            except json.JSONDecodeError:
                continue
            kind = obj.get("type")
            if kind == "token":
                assistant += obj.get("text", "")
                _maybe_flush_tokens()
            elif kind == "tool_start":
                tool_md += (
                    f"\n**▶ `{obj.get('tool')}`**\n```\n"
                    f"{(obj.get('inputs') or '')[:2000]}\n```\n"
                )
                tool_placeholder.markdown(tool_md)
                if tool_sidebar_placeholder is not None:
                    tool_sidebar_placeholder.markdown(tool_md)
            elif kind == "tool_end":
                tool_md += (
                    f"\n**◀ `{obj.get('tool')}` 观测（截断）**\n```\n"
                    f"{(obj.get('observation') or '')[:1500]}\n```\n"
                )
                tool_placeholder.markdown(tool_md)
                if tool_sidebar_placeholder is not None:
                    tool_sidebar_placeholder.markdown(tool_md)
            elif kind == "error":
                placeholder.error(obj.get("message", "未知错误"))
                break
            elif kind == "done":
                break
    _maybe_flush_tokens(force=True)
    placeholder.markdown(assistant)
    return assistant, tool_md


st.set_page_config(
    page_title="S&OP 决策中枢 v5.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container { padding-top: 1rem; max-width: 900px; }
[data-testid="stChatMessageContent"] { font-size: 1.02rem; line-height: 1.55; }
[data-testid="stSidebar"] h3 { font-size: 0.95rem; }
</style>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# 会话 ID：URL ?sid= 为真源，刷新页面可从后端拉回同一会话（后端未重启时）
if "session_id" not in st.session_state:
    raw = (st.query_params.get("sid") or "").strip()
    st.session_state.session_id = raw if raw else str(uuid.uuid4())

# 仅当地址栏没有 sid 时写入（首次打开）；刷新时已有 sid 则保留链接中的会话
if not (st.query_params.get("sid") or "").strip():
    st.query_params["sid"] = st.session_state.session_id

api_base = st.sidebar.text_input("后端 API 地址", DEFAULT_API)

if st.session_state.get("_history_api_base") != api_base:
    st.session_state.pop("_history_sync_sid", None)
    st.session_state._history_api_base = api_base

if st.session_state.get("_history_sync_sid") != st.session_state.session_id:
    st.session_state.messages = _fetch_session_history(api_base, st.session_state.session_id)
    st.session_state._history_sync_sid = st.session_state.session_id
use_stream = st.sidebar.toggle("流式输出（SSE 打字机）", value=True)
st.sidebar.caption(
    f"会话 ID：`{st.session_state.session_id}`\n\n"
    "刷新页面会保留对话（依赖 URL 中 `sid` 与后端进程未重启）。"
)
st.sidebar.markdown("---")
st.sidebar.subheader("本回合工具轨迹")
tool_sidebar = st.sidebar.empty()

if st.sidebar.button("新对话（清空前后端会话）"):
    _reset_backend_session(api_base, st.session_state.session_id)
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.query_params["sid"] = st.session_state.session_id
    st.session_state.pop("_history_sync_sid", None)
    tool_sidebar.markdown("_等待提问…_")
    st.rerun()

st.title("S&OP 智能决策中枢 v5.0")
st.caption(
    "企业内网风格助手 · 流式回答 · 侧栏与展开区同步展示工具名与入参（对齐 w4 终端 Callback 信息）"
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请输入您的问题…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    tool_md_acc = ""
    with st.chat_message("assistant"):
        ph = st.empty()
        with st.expander("🔧 工具调用详情（本回合，随 SSE 更新）", expanded=True):
            tool_ph = st.empty()
        tool_ph.markdown("_等待模型与工具…_")
        tool_sidebar.markdown("_运行中…_")
        try:
            if use_stream:
                assistant_text, tool_md_acc = _chat_stream(
                    api_base,
                    st.session_state.session_id,
                    prompt,
                    ph,
                    tool_ph,
                    tool_sidebar_placeholder=tool_sidebar,
                )
            else:
                assistant_text, tool_md_acc = _chat_non_stream(
                    api_base, st.session_state.session_id, prompt
                )
                ph.markdown(assistant_text)
                tool_ph.markdown(tool_md_acc or "_本回合无工具记录_")
        except requests.HTTPError as e:
            ph.error(f"HTTP 错误：{e}")
            assistant_text = ""
            tool_md_acc = ""
        except requests.RequestException as e:
            ph.error(f"无法连接后端：{e}\n\n请确认已启动：`uvicorn main:app --port 8000`")
            assistant_text = ""
            tool_md_acc = ""

        if assistant_text:
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_text}
            )

    if tool_md_acc.strip():
        tool_sidebar.markdown(tool_md_acc)
    else:
        tool_sidebar.markdown("_本回合未记录到工具事件（或非流式无轨迹）_")
