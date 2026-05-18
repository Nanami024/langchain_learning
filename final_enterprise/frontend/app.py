from __future__ import annotations

import os
import time
import uuid

import requests
import streamlit as st

DEFAULT_API = os.getenv("SOP_FINAL_API", "http://127.0.0.1:8100")


def _fetch_history(api: str, sid: str) -> list[dict]:
    try:
        r = requests.get(f"{api}/session/history", params={"session_id": sid}, timeout=20)
        r.raise_for_status()
        return r.json().get("messages") or []
    except requests.RequestException:
        return []


def _fetch_sessions(api: str, limit: int = 30) -> list[dict]:
    try:
        r = requests.get(f"{api}/session/list", params={"limit": limit}, timeout=20)
        r.raise_for_status()
        return r.json().get("items") or []
    except requests.RequestException:
        return []


def _chat(api: str, sid: str, text: str) -> dict:
    r = requests.post(f"{api}/graph/chat", json={"session_id": sid, "message": text}, timeout=180)
    r.raise_for_status()
    return r.json()


def _resume(api: str, sid: str, decision: str, feedback: str = "") -> dict:
    r = requests.post(
        f"{api}/graph/resume",
        json={"session_id": sid, "decision": decision, "feedback": feedback},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


def _graph_state(api: str, sid: str) -> dict | None:
    try:
        r = requests.get(f"{api}/graph/state", params={"session_id": sid}, timeout=20)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def _probe_backend(api: str) -> tuple[str, str]:
    """返回 level: ok | loading | error"""
    base = api.rstrip("/")
    for attempt in range(2):
        try:
            r = requests.get(f"{base}/health", timeout=8)
            r.raise_for_status()
            break
        except requests.ConnectionError:
            if attempt == 0:
                time.sleep(0.6)
                continue
            return "error", "无法连接后端（请先启动 uvicorn 127.0.0.1:8100）"
        except requests.RequestException as e:
            if attempt == 0:
                time.sleep(0.6)
                continue
            return "error", f"后端未响应：{e}"

    try:
        r2 = requests.get(f"{base}/health/ready", timeout=5)
        if r2.status_code == 200:
            return "ok", "后端已就绪"
        detail = r2.json() if "application/json" in (r2.headers.get("content-type") or "") else {}
        err = str(detail.get("error", "") or "").strip()
        if err:
            return "error", f"后端加载失败：{err}"
        return "loading", "后端在线，索引加载中（约 10–40 秒），请稍候…"
    except requests.RequestException:
        return "loading", "后端在线，正在等待索引就绪…"


def _cached_backend_status(api: str) -> tuple[str, str]:
    now = time.time()
    ttl = 2.5
    cached = st.session_state.get("backend_status_cache")
    if cached and now - float(cached.get("ts", 0)) < ttl:
        return str(cached["level"]), str(cached["msg"])
    level, msg = _probe_backend(api)
    st.session_state["backend_status_cache"] = {"level": level, "msg": msg, "ts": now}
    return level, msg


def _fetch_catalog(api: str) -> dict | None:
    try:
        r = requests.get(f"{api.rstrip('/')}/sources/catalog", timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def _render_catalog_sidebar(api: str) -> None:
    level, status_msg = _cached_backend_status(api)
    if level == "ok":
        st.sidebar.success(status_msg)
    elif level == "loading":
        st.sidebar.warning(status_msg)
        if st.sidebar.button("刷新后端状态", key="refresh_backend_status"):
            st.session_state.pop("backend_status_cache", None)
            st.rerun()
    else:
        st.sidebar.error(status_msg)

    catalog = _fetch_catalog(api) if level in ("ok", "loading") else None
    st.sidebar.markdown("---")
    st.sidebar.subheader("可用数据源")

    if not catalog:
        if level == "loading":
            st.sidebar.caption("索引加载完成后即可对话；目录信息可先忽略。")
        else:
            st.sidebar.caption("请先启动后端，再点「刷新后端状态」。")
        return

    tables = catalog.get("tables") or []
    if tables:
        st.sidebar.markdown("**MySQL 表**")
        for name in tables:
            st.sidebar.markdown(f"- `{name}`")
    if "sales_row_count" in catalog:
        st.sidebar.caption(f"sales_performance 行数: {int(catalog.get('sales_row_count') or 0)}")
    if catalog.get("sales_error"):
        st.sidebar.error(f"数据检查失败：{catalog.get('sales_error')}")

    pdfs = catalog.get("pdf_files") or []
    if pdfs:
        st.sidebar.markdown("**制度 PDF**")
        for name in pdfs:
            st.sidebar.markdown(f"- `{name}`")
    else:
        st.sidebar.caption("未找到 PDF（请放入 final_enterprise/data/*.pdf）")

    csvs = catalog.get("csv_files") or []
    if csvs:
        st.sidebar.markdown("**业务 CSV**")
        for name in csvs:
            st.sidebar.markdown(f"- `{name}`")

    sample = catalog.get("sales_sample") or []
    if sample:
        with st.sidebar.expander("样例数据（前2行）", expanded=False):
            st.json(sample)


def _render_trace(meta: dict | None) -> None:
    if not isinstance(meta, dict):
        return
    brief = meta.get("brief") if isinstance(meta.get("brief"), dict) else {}
    trace = meta.get("node_trace") if isinstance(meta.get("node_trace"), list) else []
    if not brief and not trace:
        return
    with st.expander("执行链路", expanded=False):
        if brief:
            st.write(
                f"路由: `{brief.get('route_plan', '-')}` | "
                f"置信度: `{brief.get('route_confidence', '-')}` | "
                f"来源: `{brief.get('route_source', '-')}`"
            )
            if brief.get("route_reason"):
                st.caption(f"判定依据: {brief.get('route_reason')}")
        if trace:
            st.markdown("**节点轨迹**")
            for i, step in enumerate(trace, start=1):
                node = str((step or {}).get("node", "unknown"))
                detail = str((step or {}).get("detail", "")).strip()
                st.markdown(f"{i}. `{node}` - {detail}")


st.set_page_config(page_title="Agentic S&OP Final", page_icon="✅", layout="wide")
st.title("S&OP 智能决策平台")

if "sid" not in st.session_state:
    qp_sid = st.query_params.get("sid")
    st.session_state.sid = str(qp_sid).strip() if qp_sid else str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None

api = st.sidebar.text_input("服务地址", DEFAULT_API)
sid = st.sidebar.text_input("会话 ID", st.session_state.sid)
if sid != st.session_state.sid:
    st.session_state.sid = sid.strip() or st.session_state.sid
st.query_params["sid"] = st.session_state.sid

_render_catalog_sidebar(api)

state = _graph_state(api, st.session_state.sid)
if state and state.get("status") == "interrupted":
    st.session_state.pending_approval = state.get("interrupt")

sessions = _fetch_sessions(api, limit=30)
session_labels: list[str] = []
session_map: dict[str, str] = {}
for row in sessions:
    x_sid = str((row or {}).get("session_id", "")).strip()
    n = int((row or {}).get("messages", 0) or 0)
    if not x_sid:
        continue
    label = f"{x_sid} ({n}条)"
    session_labels.append(label)
    session_map[label] = x_sid
if session_labels:
    chosen = st.sidebar.selectbox("最近会话", session_labels, index=0)
    if st.sidebar.button("切换并加载该会话"):
        st.session_state.sid = session_map.get(chosen, st.session_state.sid)
        st.query_params["sid"] = st.session_state.sid
        st.session_state.messages = _fetch_history(api, st.session_state.sid)
        state = _graph_state(api, st.session_state.sid)
        st.session_state.pending_approval = state.get("interrupt") if state and state.get("status") == "interrupted" else None
        st.rerun()

if st.sidebar.button("加载会话记录"):
    st.session_state.messages = _fetch_history(api, st.session_state.sid)

pending = st.session_state.pending_approval
if pending:
    st.sidebar.subheader("审批处理")
    st.sidebar.write(pending.get("reason", "请完成审批"))
    draft = str(pending.get("draft_answer") or pending.get("recommendation") or "")
    if draft.strip():
        st.sidebar.caption("方案摘要")
        st.sidebar.code(draft[:800], language="markdown")

    feedback_text = st.sidebar.text_area("审批意见（可选）", key="review_feedback")
    if st.sidebar.button("批准执行"):
        resp = _resume(api, st.session_state.sid, "approved")
        st.session_state.messages.append({"role": "assistant", "content": resp.get("output", "")})
        st.session_state.pending_approval = None
        st.rerun()
    if st.sidebar.button("补充意见"):
        resp = _resume(api, st.session_state.sid, "feedback", feedback_text)
        if resp.get("status") == "interrupted":
            st.session_state.pending_approval = resp.get("interrupt")
            st.session_state.messages.append({"role": "assistant", "content": "方案已根据意见更新，请继续审批。"})
        else:
            st.session_state.pending_approval = None
            st.session_state.messages.append({"role": "assistant", "content": resp.get("output", "")})
        st.rerun()
    if st.sidebar.button("驳回方案"):
        resp = _resume(api, st.session_state.sid, "rejected", feedback_text)
        st.session_state.messages.append({"role": "assistant", "content": resp.get("output", "")})
        st.session_state.pending_approval = None
        st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m.get("role", "assistant")):
        st.markdown(m.get("content", ""))
        if m.get("role") == "assistant":
            _render_trace(m.get("meta"))

if prompt := st.chat_input("请输入业务问题"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            resp = _chat(api, st.session_state.sid, prompt)
            if resp.get("status") == "interrupted":
                st.session_state.pending_approval = resp.get("interrupt")
                msg = "该方案需要人工审批。"
                st.info(msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": msg,
                        "meta": {"brief": resp.get("brief"), "node_trace": resp.get("node_trace")},
                    }
                )
                st.rerun()
            else:
                out = resp.get("output", "")
                st.markdown(out)
                _render_trace({"brief": resp.get("brief"), "node_trace": resp.get("node_trace")})
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": out,
                        "meta": {"brief": resp.get("brief"), "node_trace": resp.get("node_trace")},
                    }
                )
            with st.expander("依据来源", expanded=False):
                src = resp.get("sources") or []
                if src:
                    st.json(src)
                else:
                    st.write("暂无可展示来源。")
        except requests.RequestException as e:
            st.error("服务暂时不可用，请稍后重试。")
            detail = str(e)
            if hasattr(e, "response") and e.response is not None:
                try:
                    detail = e.response.json().get("detail", detail)
                except Exception:
                    detail = (e.response.text or detail)[:500]
            st.caption(f"原因：{detail}")
            st.info(
                "请确认：① 后端已启动 `python -m uvicorn main:app --host 127.0.0.1 --port 8100`；"
                "② 左侧「服务地址」与后端一致；③ MySQL / .env 配置正确。"
            )
