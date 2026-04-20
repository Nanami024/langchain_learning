"""MySQL 连接相关：URL 拼装与单例 SQLAlchemy Engine。

环境变量（在 `w6/.env` 或系统环境中配置；可参考 `w6/.env.example`）：

| 变量 | 默认 | 说明 |
|------|------|------|
| MYSQL_HOST | 127.0.0.1 | MySQL 主机 |
| MYSQL_PORT | 3306 | 端口 |
| MYSQL_USER | root | 读写账号（导入脚本与会话历史写入用） |
| MYSQL_PASSWORD | （空） | 读写账号密码 |
| MYSQL_DB | sop_ai_system | 业务库 |
| MYSQL_RO_USER | （回退到 MYSQL_USER） | 只读账号；推荐为 SQL Agent 单独建一个仅 SELECT 权限的账号 |
| MYSQL_RO_PASSWORD | （回退到 MYSQL_PASSWORD） | 只读账号密码 |
| MYSQL_HISTORY_TABLE | chat_messages | LangChain `SQLChatMessageHistory` 写入的会话表名 |
| SOP_SALES_TABLE | sales_performance | 业务事实表名（SQL Agent 暴露给大模型） |
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

_W6_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_W6_ROOT / ".env")
load_dotenv()


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def mysql_database() -> str:
    return _env("MYSQL_DB", "sop_ai_system")


def history_table() -> str:
    return _env("MYSQL_HISTORY_TABLE", "chat_messages")


def sales_table() -> str:
    return _env("SOP_SALES_TABLE", "sales_performance")


def _mysql_url(*, read_only: bool, include_db: bool = True) -> str:
    host = _env("MYSQL_HOST", "127.0.0.1")
    port = _env("MYSQL_PORT", "3306")
    if read_only:
        user = _env("MYSQL_RO_USER") or _env("MYSQL_USER", "root")
        password = _env("MYSQL_RO_PASSWORD") or _env("MYSQL_PASSWORD", "")
    else:
        user = _env("MYSQL_USER", "root")
        password = _env("MYSQL_PASSWORD", "")
    db = mysql_database() if include_db else ""
    pwd = quote_plus(password)
    tail = f"/{db}" if include_db else ""
    return f"mysql+pymysql://{user}:{pwd}@{host}:{port}{tail}?charset=utf8mb4"


def server_url() -> str:
    """不带具体库名的 URL，用于 `CREATE DATABASE IF NOT EXISTS`。"""
    return _mysql_url(read_only=False, include_db=False)


@lru_cache(maxsize=1)
def rw_engine() -> Engine:
    """读写引擎：用于会话历史读写、CSV 导入。"""
    return create_engine(
        _mysql_url(read_only=False),
        pool_pre_ping=True,
        pool_recycle=3600,
        future=True,
    )


# 写操作关键字（白名单只允许 SELECT/SHOW/DESCRIBE/EXPLAIN/WITH 等只读语句开头）
_READONLY_PREFIXES = ("SELECT", "WITH", "SHOW", "DESCRIBE", "DESC", "EXPLAIN", "PRAGMA", "USE", "SET")


def _strip_sql(s: str) -> str:
    """去除多语句前导空白与 SQL 注释，便于判断真实首关键字。"""
    text = s.strip()
    if text.startswith("/*"):
        end = text.find("*/")
        if end != -1:
            text = text[end + 2 :].lstrip()
    while text.startswith("--"):
        nl = text.find("\n")
        text = "" if nl == -1 else text[nl + 1 :].lstrip()
    return text


class SQLWriteAttempt(PermissionError):
    """SQL Agent 试图执行非 SELECT 语句时抛出，由 LangChain 捕获并回填到模型。"""


def _install_readonly_guard(engine: Engine) -> None:
    @event.listens_for(engine, "before_cursor_execute")
    def _block_writes(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
        head = _strip_sql(statement).split(None, 1)
        first = head[0].upper() if head else ""
        if first not in _READONLY_PREFIXES:
            raise SQLWriteAttempt(
                f"只读保护：拒绝执行非查询语句（首关键字={first or '?'}）。"
                "本接口仅允许 SELECT / SHOW / DESCRIBE / EXPLAIN / WITH。"
            )


@lru_cache(maxsize=1)
def ro_engine() -> Engine:
    """只读引擎：用于 SQL Agent。即使数据库账号是读写权限，也通过 SQLAlchemy 事件兜底拦截写操作。"""
    eng = create_engine(
        _mysql_url(read_only=True),
        pool_pre_ping=True,
        pool_recycle=3600,
        future=True,
    )
    _install_readonly_guard(eng)
    return eng


def reset_engines() -> None:
    """测试/脚本场景下重置缓存（生产请求路径不要调用）。"""
    rw_engine.cache_clear()
    ro_engine.cache_clear()
