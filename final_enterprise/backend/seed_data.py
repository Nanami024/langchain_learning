from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import text

from config import DATA_DIR, Settings
from database import get_engine


def _table_columns(table: str) -> list[str]:
    settings = Settings()
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = :db AND TABLE_NAME = :tbl
                ORDER BY ORDINAL_POSITION
                """
            ),
            {"db": settings.mysql_db, "tbl": table},
        ).all()
    return [str(r[0]) for r in rows]


def _table_schema(table: str) -> list[dict[str, str]]:
    settings = Settings()
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT COLUMN_NAME, IS_NULLABLE, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = :db AND TABLE_NAME = :tbl
                ORDER BY ORDINAL_POSITION
                """
            ),
            {"db": settings.mysql_db, "tbl": table},
        ).mappings().all()
    return [{k.lower(): str(v) for k, v in dict(r).items()} for r in rows]


def _create_tables() -> None:
    settings = Settings()
    sales_table = settings.sales_table
    history_table = settings.mysql_history_table
    ddl_sales = f"""
    CREATE TABLE IF NOT EXISTS {sales_table} (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        biz_date DATE NOT NULL,
        category VARCHAR(64) NULL,
        region VARCHAR(32) NOT NULL,
        store_code VARCHAR(32) NOT NULL,
        sku VARCHAR(64) NULL,
        forecast_qty DOUBLE NULL,
        actual_qty DOUBLE NULL,
        forecast_accuracy DOUBLE NULL,
        oos_rate DOUBLE NULL,
        inventory_turnover_days DOUBLE NULL
    );
    """
    ddl_history = f"""
    CREATE TABLE IF NOT EXISTS {history_table} (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        session_id VARCHAR(128) NOT NULL,
        role VARCHAR(32) NOT NULL,
        content TEXT NOT NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """
    with get_engine().begin() as conn:
        conn.execute(text(ddl_sales))
        conn.execute(text(ddl_history))


def _build_demo_sales() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "biz_date": "2026-03-01",
                "category": "化妆品",
                "region": "华东",
                "store_code": "E-01",
                "sku": "COS-001",
                "forecast_qty": 110,
                "actual_qty": 100,
                "forecast_accuracy": 90.9,
                "oos_rate": 1.8,
                "inventory_turnover_days": 23,
            },
            {
                "biz_date": "2026-03-08",
                "category": "化妆品",
                "region": "华东",
                "store_code": "E-02",
                "sku": "COS-002",
                "forecast_qty": 130,
                "actual_qty": 115,
                "forecast_accuracy": 88.5,
                "oos_rate": 2.2,
                "inventory_turnover_days": 24,
            },
            {
                "biz_date": "2026-03-15",
                "category": "化妆品",
                "region": "华东",
                "store_code": "E-03",
                "sku": "COS-003",
                "forecast_qty": 120,
                "actual_qty": 110,
                "forecast_accuracy": 91.7,
                "oos_rate": 1.6,
                "inventory_turnover_days": 22,
            },
        ]
    )


def _seed_sales_if_empty() -> None:
    settings = Settings()
    sales_table = settings.sales_table
    with get_engine().connect() as conn:
        cnt = int(conn.execute(text(f"SELECT COUNT(*) FROM {sales_table}")).scalar() or 0)
    if cnt > 0:
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = Path(DATA_DIR) / "sales_performance.csv"
    df = pd.read_csv(csv_path) if csv_path.exists() else _build_demo_sales()
    if df.empty:
        return

    schema_rows = _table_schema(sales_table)
    existing_cols = [r["column_name"] for r in schema_rows]
    load_cols = [c for c in df.columns if c in existing_cols and c.lower() != "id"]
    if not load_cols:
        return
    data = df[load_cols].copy()
    required_missing = [
        r
        for r in schema_rows
        if r["column_name"].lower() != "id" and r["is_nullable"].upper() == "NO" and r["column_name"] not in data.columns
    ]
    for r in required_missing:
        col = r["column_name"]
        dtype = r["data_type"].lower()
        if col == "category":
            data[col] = "未分类"
        elif "date" in dtype:
            data[col] = pd.to_datetime(Settings().assumed_today).date()
        elif any(x in dtype for x in ("int", "decimal", "float", "double", "numeric")):
            data[col] = 0
        else:
            data[col] = "N/A"
    if "biz_date" in data.columns:
        data["biz_date"] = pd.to_datetime(data["biz_date"], errors="coerce").dt.date
    ordered_cols = [c for c in existing_cols if c.lower() != "id" and c in data.columns]
    data = data[ordered_cols]
    data = data.dropna(how="all")
    if data.empty:
        return
    data.to_sql(sales_table, con=get_engine(), if_exists="append", index=False, method="multi", chunksize=500)


def _export_sales_csv() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = Path(DATA_DIR) / "sales_performance.csv"
    settings = Settings()
    preferred_cols = [
        "biz_date",
        "category",
        "region",
        "store_code",
        "sku",
        "forecast_qty",
        "actual_qty",
        "forecast_accuracy",
        "oos_rate",
        "inventory_turnover_days",
    ]
    existing_cols = _table_columns(settings.sales_table)
    cols = [c for c in preferred_cols if c in existing_cols] or existing_cols
    if not cols:
        return
    select_cols = ", ".join(cols)
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT {select_cols}
                FROM {settings.sales_table}
                ORDER BY id ASC
                """
            )
        ).all()
    if not rows:
        return
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def bootstrap_data() -> None:
    _create_tables()
    _seed_sales_if_empty()
    _export_sales_csv()


if __name__ == "__main__":
    bootstrap_data()
    print("seed complete")
