"""S&OP v6.0 — Streamlit 前端：MySQL 持久化记忆 + SQL 工具轨迹高亮。

相对 w5 改动：
- 侧边栏标题强调「数据驱动版」，并新增「最近一条 SQL」区，把内层 `sql_db_query` 的入参
  单独高亮成 ```sql 代码块，对应作业要求的「Text-to-SQL 追踪」。
- 其他 SSE / 工具轨迹逻辑直接沿用 w5；本页面同样可以指向 w5 后端使用，但只有 v6 后端
  才能持久化会话与展示 sql_db_* 调用。
"""

from __future__ import annotations

import json
import os
import time
import uuid

import requests
import streamlit as st

DEFAULT_API = os.getenv("SOP_API_URL", "http://127.0.0.1:8000")


def _tool_obs_display_limit() -> int:
    try:
        return max(400, int(os.getenv("SOP_TOOL_OBS_MAX_CHARS", "8000")))
    except ValueError:
        return 8000


def _http_session(api_base: str) -> requests.Session:
    key = "_sop_http_sess"
    base = api_base.rstrip("/")
    if st.session_state.get("_sop_http_base") != base or key not in st.session_state:
        st.session_state[key] = requests.Session()
        st.session_state["_sop_http_base"] = base
    return st.session_state[key]


def _fetch_session_history(api_base: str, session_id: str) -> list[dict]:
    try:
        sess = _http_session(api_base)
        r = sess.get(
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
        sess = _http_session(api_base)
        r = sess.post(
            f"{api_base.rstrip('/')}/session/reset",
            json={"session_id": session_id},
            timeout=30,
        )
        r.raise_for_status()
    except requests.RequestException:
        pass


def _render_inputs_for_tool(tool_name: str, inputs: str) -> str:
    """根据工具名渲染入参；`sql_db_query` 用 ```sql 高亮，其它走纯文本块。"""
    inputs = (inputs or "")[:3000]
    if tool_name == "sql_db_query":
        return f"\n```sql\n{inputs}\n```\n"
    return f"\n```\n{inputs}\n```\n"


def _maybe_extract_sql(tool_name: str, inputs: str) -> str | None:
    """`sql_db_query` 入参形如 `{"query": "SELECT ..."}` 或纯字符串；统一抽出 SQL。"""
    if tool_name != "sql_db_query":
        return None
    s = (inputs or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            for k in ("query", "sql", "input", "__arg1"):
                if isinstance(obj.get(k), str):
                    return obj[k].strip()
        if isinstance(obj, str):
            return obj.strip()
    except json.JSONDecodeError:
        pass
    return s


def _chat_non_stream(api_base: str, session_id: str, message: str) -> tuple[str, str, list[str]]:
    sess = _http_session(api_base)
    r = sess.post(
        f"{api_base.rstrip('/')}/chat",
        json={"session_id": session_id, "message": message},
        timeout=600,
    )
    r.raise_for_status()
    data = r.json()
    out = data.get("output", "")
    trace = data.get("tool_trace") or []
    trace_md = ""
    sqls: list[str] = []
    last_tool: str = ""
    for row in trace:
        if row.get("phase") == "start":
            tool_name = str(row.get("tool") or "")
            last_tool = tool_name
            inputs = str(row.get("inputs", ""))
            trace_md += f"\n**▶ `{tool_name}`**{_render_inputs_for_tool(tool_name, inputs)}"
            sql = _maybe_extract_sql(tool_name, inputs)
            if sql:
                sqls.append(sql)
        elif row.get("phase") == "end":
            obs = str(row.get("observation", ""))
            trace_md += f"\n**◀ `{last_tool}` 观测**\n```\n{obs}\n```\n"
    return out, trace_md, sqls


def _chat_stream(
    api_base: str,
    session_id: str,
    message: str,
    placeholder,
    tool_placeholder,
    sql_placeholder,
    tool_sidebar_placeholder=None,
) -> tuple[str, str, list[str]]:
    assistant = ""
    tool_md = ""
    sqls: list[str] = []
    _last_flush = 0.0
    _last_flush_len = 0
    try:
        _ui_min_interval = float(os.getenv("SOP_UI_STREAM_MIN_INTERVAL", "0.35"))
    except ValueError:
        _ui_min_interval = 0.35
    try:
        _ui_min_chars = int(os.getenv("SOP_UI_STREAM_MIN_CHARS", "160"))
    except ValueError:
        _ui_min_chars = 160

    def _maybe_flush_tokens(*, force: bool = False) -> None:
        nonlocal _last_flush, _last_flush_len
        now = time.monotonic()
        grown = len(assistant) - _last_flush_len
        if force or grown >= _ui_min_chars or (now - _last_flush >= _ui_min_interval and grown > 0):
            placeholder.markdown(assistant + "▌")
            _last_flush = now
            _last_flush_len = len(assistant)

    def _render_sqls() -> None:
        if not sqls:
            return
        md = "\n".join(f"```sql\n{q}\n```" for q in sqls)
        sql_placeholder.markdown(md)

    sess = _http_session(api_base)
    url = f"{api_base.rstrip('/')}/chat/stream"
    last_tool: str = ""
    with sess.post(
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
                tool_name = str(obj.get("tool") or "")
                last_tool = tool_name
                inputs = (obj.get("inputs") or "")[:2000]
                tool_md += f"\n**▶ `{tool_name}`**{_render_inputs_for_tool(tool_name, inputs)}"
                tool_placeholder.markdown(tool_md)
                sql = _maybe_extract_sql(tool_name, inputs)
                if sql:
                    sqls.append(sql)
                    _render_sqls()
                if tool_sidebar_placeholder is not None:
                    tool_sidebar_placeholder.markdown(tool_md)
            elif kind == "tool_end":
                obs_raw = obj.get("observation") or ""
                if not isinstance(obs_raw, str):
                    obs_raw = str(obs_raw)
                lim = _tool_obs_display_limit()
                truncated = len(obs_raw) > lim
                shown = obs_raw[:lim]
                note = (
                    f"\n_（共 {len(obs_raw)} 字，界面展示前 {lim} 字）_\n"
                    if truncated
                    else ""
                )
                tool_md += f"\n**◀ `{last_tool}` 观测**{note}\n```\n{shown}\n```\n"
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
    _render_sqls()
    return assistant, tool_md, sqls


# ----------------------------------------------------------------------------- UI

st.set_page_config(
    page_title="S&OP 决策中枢 v6.0 · 数据驱动版",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container { padding-top: 1rem; max-width: 980px; }
[data-testid="stChatMessageContent"] { font-size: 1.02rem; line-height: 1.55; }
[data-testid="stSidebar"] h3 { font-size: 0.95rem; }
</style>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    raw = (st.query_params.get("sid") or "").strip()
    st.session_state.session_id = raw if raw else str(uuid.uuid4())

if not (st.query_params.get("sid") or "").strip():
    st.query_params["sid"] = st.session_state.session_id

api_base = st.sidebar.text_input("后端 API 地址", DEFAULT_API)

if st.session_state.get("_history_api_base") != api_base:
    st.session_state.pop("_history_sync_sid", None)
    st.session_state._history_api_base = api_base

if st.session_state.get("_history_sync_sid") != st.session_state.session_id:
    st.session_state.messages = _fetch_session_history(api_base, st.session_state.session_id)
    st.session_state._history_sync_sid = st.session_state.session_id

use_stream = st.sidebar.toggle("流式输出（SSE）", value=True)
st.sidebar.caption(
    f"会话 ID：`{st.session_state.session_id}`\n\n"
    "记忆已写入 MySQL，**重启后端 + 刷新页面**仍可续接对话；"
    "URL 中的 `sid` 决定要恢复哪一条。"
)
st.sidebar.markdown("---")
st.sidebar.subheader("本回合工具轨迹")
tool_sidebar = st.sidebar.empty()
st.sidebar.markdown("---")
st.sidebar.subheader("🛢️ Text-to-SQL（最近一次）")
sql_sidebar = st.sidebar.empty()

if st.sidebar.button("新对话（清空 MySQL 中本会话历史）"):
    _reset_backend_session(api_base, st.session_state.session_id)
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.query_params["sid"] = st.session_state.session_id
    st.session_state.pop("_history_sync_sid", None)
    tool_sidebar.markdown("_等待提问…_")
    sql_sidebar.markdown("_等待提问…_")
    st.rerun()

st.title("S&OP 智能决策中枢 v6.0 · 数据驱动版")
st.caption(
    "记忆持久化（MySQL · SQLChatMessageHistory） · "
    "Text-to-SQL（SQLDatabaseToolkit · 仅 SELECT） · 流式回答 · SSE 工具轨迹"
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请输入您的问题…（试试：华东区上个月预测准确率最高的门店是哪个？）"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    tool_md_acc = ""
    sqls_acc: list[str] = []
    with st.chat_message("assistant"):
        ph = st.empty()
        with st.expander("🔧 工具调用详情（本回合，随 SSE 更新）", expanded=False):
            tool_ph = st.empty()
        with st.expander("🛢️ 大模型生成的 SQL（本回合）", expanded=True):
            sql_ph = st.empty()
        tool_ph.markdown("_等待模型与工具…_")
        sql_ph.markdown("_等待 SQL 生成…_")
        tool_sidebar.markdown("_运行中…_")
        sql_sidebar.markdown("_运行中…_")
        try:
            if use_stream:
                assistant_text, tool_md_acc, sqls_acc = _chat_stream(
                    api_base,
                    st.session_state.session_id,
                    prompt,
                    ph,
                    tool_ph,
                    sql_ph,
                    tool_sidebar_placeholder=None,
                )
            else:
                assistant_text, tool_md_acc, sqls_acc = _chat_non_stream(
                    api_base, st.session_state.session_id, prompt
                )
                ph.markdown(assistant_text)
                tool_ph.markdown(tool_md_acc or "_本回合无工具记录_")
                if sqls_acc:
                    sql_ph.markdown("\n".join(f"```sql\n{q}\n```" for q in sqls_acc))
                else:
                    sql_ph.markdown("_本回合未生成 SQL_")
        except requests.HTTPError as e:
            body = ""
            if e.response is not None:
                try:
                    body = (e.response.text or "")[:2500]
                except Exception:
                    pass
            ph.error(f"HTTP 错误：{e}\n\n{body}".strip())
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

    if sqls_acc:
        sql_sidebar.markdown("\n".join(f"```sql\n{q}\n```" for q in sqls_acc))
    else:
        sql_sidebar.markdown("_本回合未生成 SQL_")
