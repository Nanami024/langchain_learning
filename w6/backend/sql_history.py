"""会话历史持久化：基于 LangChain `SQLChatMessageHistory` + MySQL。

设计要点：
- 用 SQLAlchemy `Engine`（不是裸字符串）注入，避免每个 session 重复建池。
- `RunnableWithMessageHistory` 期望 `Callable[[str], BaseChatMessageHistory]`，因此对外只暴露
  `get_session_history(session_id)`；其他持久化细节封装在本模块内。
- 提供 `read_messages` / `clear_session` 给 FastAPI 的 `/session/history`、`/session/reset` 复用。
"""

from __future__ import annotations

from typing import Any

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from sqlalchemy import text

from db_config import history_table, rw_engine


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """供 RunnableWithMessageHistory 调用：每次按 session_id 取/建一个持久化历史。"""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=rw_engine(),
        table_name=history_table(),
    )


def read_messages(session_id: str) -> list[BaseMessage]:
    """读取某个会话所有历史消息（用于前端刷新后回灌对话）。"""
    return list(get_session_history(session_id).messages)


def clear_session(session_id: str) -> None:
    """前端「新对话」按钮：清空单个会话历史，但保留会话表本身。"""
    get_session_history(session_id).clear()


def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """运维便利：按最新消息时间倒序列出最近的会话 ID 与消息数。

    SQLChatMessageHistory 默认表 `message_store` 字段：id / session_id / message。
    我们额外统计 `MAX(id)` 近似排序（id 自增，时间序）。
    """
    tbl = history_table()
    sql = text(
        f"SELECT session_id, COUNT(*) AS n, MAX(id) AS last_id "
        f"FROM `{tbl}` GROUP BY session_id ORDER BY last_id DESC LIMIT :lim"
    )
    with rw_engine().connect() as conn:
        rows = conn.execute(sql, {"lim": int(limit)}).all()
    return [{"session_id": r.session_id, "messages": int(r.n), "last_id": int(r.last_id)} for r in rows]
