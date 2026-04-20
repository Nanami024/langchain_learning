"""一次性脚本：把 `w3/sop_knowledge/data.csv` 导入 MySQL `sales_performance` 表。

执行（在仓库根目录、已激活 .venv 的 PowerShell 中）：

    python -m w6.backend.import_csv_to_mysql

或：

    cd w6/backend
    python import_csv_to_mysql.py --drop

可选参数：
- `--csv <path>`：覆盖默认 CSV 路径（默认走 `w3/sop_knowledge/data.csv`）。
- `--table <name>`：覆盖目标表名（默认读 `SOP_SALES_TABLE` 或 `sales_performance`）。
- `--drop`：导入前 DROP 表重建（演示用；生产请慎用）。
- `--limit N`：只导入前 N 行，便于本机调试。

为何要给原始 CSV 加 `region` / `store_code`？
    第六周作业演示问题「华东区上个月准确率最高的门店是哪个？」要求按区域 + 门店聚合，
    而 `data.csv` 仅有日期/类目/数值列。为复现作业问题，我们用「行序号 % 区域数」的
    确定性映射给每行打上区域与门店编号，保证每次导入得到完全一致的数据，便于复盘。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from db_config import mysql_database, rw_engine, sales_table, server_url  # noqa: E402

_REPO_ROOT = _HERE.parent.parent
_DEFAULT_CSV = _REPO_ROOT / "w3" / "sop_knowledge" / "data.csv"

REGIONS = ["华东", "华北", "华南", "华中", "西南"]
REGION_PREFIX = {"华东": "EC", "华北": "NC", "华南": "SC", "华中": "CC", "西南": "SW"}
STORES_PER_REGION = 12

DDL = """
CREATE TABLE IF NOT EXISTS `{table}` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `biz_date` DATE NOT NULL COMMENT '业务日期（原 CSV 的「日期」）',
  `category` VARCHAR(64) NOT NULL COMMENT '商品类目（家居百货/食品/服装鞋靴/化妆品 等）',
  `region` VARCHAR(32) NOT NULL COMMENT '所属区域（华东/华北/华南/华中/西南，演示合成）',
  `store_code` VARCHAR(32) NOT NULL COMMENT '门店编号（区域前缀+三位序号，如 EC001、NC003）',
  `forecast_qty` INT NOT NULL COMMENT '预测销量',
  `actual_qty` INT NOT NULL COMMENT '实际销量',
  `forecast_accuracy` DECIMAL(5,2) NOT NULL COMMENT '预测准确率（%）',
  `oos_rate` DECIMAL(5,2) NOT NULL COMMENT '缺货率（%）',
  `inventory_turnover_days` INT NOT NULL COMMENT '库存周转天数',
  PRIMARY KEY (`id`),
  KEY `idx_date` (`biz_date`),
  KEY `idx_region` (`region`),
  KEY `idx_store` (`store_code`),
  KEY `idx_category` (`category`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COMMENT='S&OP 门店日度销售与考核明细（v6.0 演示数据，含合成的区域与门店列）';
""".strip()


def _parse_pct(x: object) -> float:
    if x is None:
        return 0.0
    s = str(x).strip().rstrip("%")
    return float(s) if s else 0.0


def _ensure_database() -> None:
    """连接 MySQL 实例（不带 db），如果库不存在就创建之。"""
    from sqlalchemy import create_engine

    url = server_url()
    eng = create_engine(url, pool_pre_ping=True)
    db = mysql_database()
    with eng.begin() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db}` DEFAULT CHARSET utf8mb4"))
    eng.dispose()


def _read_csv(csv_path: Path) -> pd.DataFrame:
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
    if last_err is not None:
        raise last_err
    return pd.read_csv(csv_path)


def _augment(df: pd.DataFrame) -> pd.DataFrame:
    """加入合成的 region / store_code 列；统一字段名为英文便于 SQL Agent。"""
    df = df.copy()
    df.rename(
        columns={
            "日期": "biz_date",
            "商品类目": "category",
            "预测销量": "forecast_qty",
            "实际销量": "actual_qty",
            "预测准确率": "forecast_accuracy",
            "缺货率": "oos_rate",
            "库存周转天数": "inventory_turnover_days",
        },
        inplace=True,
    )
    df["forecast_accuracy"] = df["forecast_accuracy"].map(_parse_pct)
    df["oos_rate"] = df["oos_rate"].map(_parse_pct)
    df["forecast_qty"] = df["forecast_qty"].astype(int)
    df["actual_qty"] = df["actual_qty"].astype(int)
    df["inventory_turnover_days"] = df["inventory_turnover_days"].astype(int)
    df["biz_date"] = pd.to_datetime(df["biz_date"]).dt.strftime("%Y-%m-%d")

    regions: list[str] = []
    stores: list[str] = []
    # 行序确定性映射：第 i 行 → REGIONS[i%5]，门店序号在区域内循环 STORES_PER_REGION 次
    for i in range(len(df)):
        r = REGIONS[i % len(REGIONS)]
        prefix = REGION_PREFIX[r]
        store_idx = ((i // len(REGIONS)) % STORES_PER_REGION) + 1
        regions.append(r)
        stores.append(f"{prefix}{store_idx:03d}")
    df["region"] = regions
    df["store_code"] = stores
    return df[
        [
            "biz_date",
            "category",
            "region",
            "store_code",
            "forecast_qty",
            "actual_qty",
            "forecast_accuracy",
            "oos_rate",
            "inventory_turnover_days",
        ]
    ]


def _create_table(table: str, drop: bool) -> None:
    eng = rw_engine()
    with eng.begin() as conn:
        if drop:
            conn.execute(text(f"DROP TABLE IF EXISTS `{table}`"))
        conn.execute(text(DDL.format(table=table)))


def _insert(df: pd.DataFrame, table: str, batch: int = 500) -> int:
    eng = rw_engine()
    cols = list(df.columns)
    placeholders = ", ".join(f":{c}" for c in cols)
    col_sql = ", ".join(f"`{c}`" for c in cols)
    sql = text(f"INSERT INTO `{table}` ({col_sql}) VALUES ({placeholders})")
    rows = df.to_dict(orient="records")
    written = 0
    with eng.begin() as conn:
        for i in range(0, len(rows), batch):
            chunk = rows[i : i + batch]
            conn.execute(sql, chunk)
            written += len(chunk)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="将 w3/data.csv 导入 MySQL sales_performance 表。")
    parser.add_argument("--csv", default=str(_DEFAULT_CSV), help="CSV 路径（默认 w3/sop_knowledge/data.csv）")
    parser.add_argument("--table", default=os.getenv("SOP_SALES_TABLE", sales_table()))
    parser.add_argument("--drop", action="store_true", help="导入前 DROP 表重建")
    parser.add_argument("--limit", type=int, default=0, help="仅导入前 N 行（0=全部）")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        print(f"[fail] CSV 不存在：{csv_path}", file=sys.stderr)
        return 2

    print(f"[step 1/4] 准备数据库 {mysql_database()} …")
    _ensure_database()

    print(f"[step 2/4] 读取并增强 {csv_path}")
    raw = _read_csv(csv_path)
    if args.limit > 0:
        raw = raw.head(args.limit).copy()
    df = _augment(raw)
    print(f"          行数={len(df)}；区域分布：")
    print(df["region"].value_counts().to_string())

    print(f"[step 3/4] 创建表 `{args.table}`（drop={args.drop}） …")
    _create_table(args.table, drop=args.drop)

    print(f"[step 4/4] 写入数据 …")
    n = _insert(df, args.table)
    print(f"[done] 共写入 {n} 行 → `{mysql_database()}`.`{args.table}`")
    print("        建议 SQL Agent 使用只读账号（MYSQL_RO_USER/MYSQL_RO_PASSWORD），")
    print("        本仓库还会通过 SQLAlchemy 事件兜底拦截 DROP/UPDATE/DELETE 等写操作。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
