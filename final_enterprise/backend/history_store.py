from __future__ import annotations

from typing import Any
import json

from sqlalchemy import text

from config import Settings
from database import get_engine


class SessionHistoryStore:
    def __init__(self) -> None:
        self.table = Settings().mysql_history_table
        self.mode = self._detect_mode()

    def _detect_mode(self) -> str:
        settings = Settings()
        with get_engine().connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT COLUMN_NAME
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = :db AND TABLE_NAME = :tbl
                    """
                ),
                {"db": settings.mysql_db, "tbl": self.table},
            ).all()
            cols = {str(r[0]).lower() for r in rows}
        if {"role", "content"}.issubset(cols):
            return "role_content"
        if "message" in cols:
            return "message_json"
        return "unknown"

    def add(self, session_id: str, role: str, content: str) -> None:
        with get_engine().begin() as conn:
            if self.mode == "role_content":
                conn.execute(
                    text(
                        f"""
                        INSERT INTO {self.table} (session_id, role, content)
                        VALUES (:session_id, :role, :content)
                        """
                    ),
                    {"session_id": session_id, "role": role, "content": content},
                )
                return
            # 兼容 SQLChatMessageHistory 的 message JSON 存储结构
            msg_type = "human" if role == "user" else "ai"
            payload = {
                "type": msg_type,
                "data": {
                    "content": content,
                    "additional_kwargs": {},
                    "response_metadata": {},
                    "type": msg_type,
                    "name": None,
                    "id": None,
                },
            }
            conn.execute(
                text(
                    f"""
                    INSERT INTO {self.table} (session_id, message)
                    VALUES (:session_id, :message)
                    """
                ),
                {"session_id": session_id, "message": json.dumps(payload, ensure_ascii=False)},
            )

    def read(self, session_id: str) -> list[dict[str, str]]:
        with get_engine().connect() as conn:
            if self.mode == "role_content":
                rows = conn.execute(
                    text(
                        f"""
                        SELECT role, content
                        FROM {self.table}
                        WHERE session_id = :session_id
                        ORDER BY id ASC
                        """
                    ),
                    {"session_id": session_id},
                ).all()
                return [{"role": str(r.role), "content": str(r.content)} for r in rows]
            rows = conn.execute(
                text(
                    f"""
                    SELECT message
                    FROM {self.table}
                    WHERE session_id = :session_id
                    ORDER BY id ASC
                    """
                ),
                {"session_id": session_id},
            ).all()
        out: list[dict[str, str]] = []
        for r in rows:
            raw = str(r.message)
            try:
                obj = json.loads(raw)
                tp = str(obj.get("type", "ai")).strip().lower()
                content = str((obj.get("data") or {}).get("content", "")).strip()
                role = "user" if tp in ("human", "user") else "assistant"
                out.append({"role": role, "content": content})
            except Exception:
                out.append({"role": "assistant", "content": raw})
        return out

    def clear(self, session_id: str) -> None:
        with get_engine().begin() as conn:
            conn.execute(text(f"DELETE FROM {self.table} WHERE session_id = :session_id"), {"session_id": session_id})

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        with get_engine().connect() as conn:
            rows = conn.execute(
                text(
                    f"""
                    SELECT session_id, COUNT(*) AS n, MAX(id) AS last_id
                    FROM {self.table}
                    GROUP BY session_id
                    ORDER BY last_id DESC
                    LIMIT :lim
                    """
                ),
                {"lim": int(limit)},
            ).all()
        return [
            {"session_id": str(r.session_id), "messages": int(r.n), "last_id": int(r.last_id)}
            for r in rows
        ]
