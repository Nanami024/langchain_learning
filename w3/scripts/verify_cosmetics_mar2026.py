"""快速验证：2026-03 化妆品 预测准确率 行级算术平均。"""
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parent.parent
csv_path = root / "sop_knowledge" / "data.csv"

df = pd.read_csv(csv_path, encoding="utf-8-sig")
df["dt"] = pd.to_datetime(df["日期"])
sub = df[(df["dt"].dt.year == 2026) & (df["dt"].dt.month == 3) & (df["商品类目"] == "化妆品")]
sub = sub.copy()
sub["acc"] = sub["预测准确率"].str.rstrip("%").astype(float)

mean = sub["acc"].mean()
print("rows:", len(sub))
print("mean:", round(mean, 4), "%")
print("values:", sub["预测准确率"].tolist())
