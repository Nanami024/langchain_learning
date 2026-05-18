from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from config import Settings


def _mysql_url(user: str, password: str, settings: Settings) -> str:
    return (
        "mysql+pymysql://"
        f"{user}:{password}"
        f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_db}?charset=utf8mb4"
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = Settings()
    return create_engine(
        _mysql_url(settings.mysql_user, settings.mysql_password, settings),
        future=True,
        pool_pre_ping=True,
    )


@lru_cache(maxsize=1)
def get_readonly_engine() -> Engine:
    settings = Settings()
    if not settings.mysql_ro_user:
        raise RuntimeError("MYSQL_RO_USER is required for AI read-only SQL queries")
    password = settings.mysql_ro_password or settings.mysql_password
    return create_engine(
        _mysql_url(settings.mysql_ro_user, password, settings),
        future=True,
        pool_pre_ping=True,
    )
