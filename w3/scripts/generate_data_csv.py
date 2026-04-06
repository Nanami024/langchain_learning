"""生成 sop_knowledge/data.csv（约 1000 行）。

分档阈值严格对齐 PDF《S&OP 供应链绩效考核指标管理规范 (v2023)》§4.3：
  - A 级「优秀」：预测准确率 >= 85%
  - B 级「合格」：75% <= 预测准确率 < 85%   （注意：仍属合格，不是「不合格」）
  - C 级「不合格」：预测准确率 < 75%

缺货率分档对齐同规范 §5.3：<=3% 正常；3%~5% 轻度预警；>5% 严重。

库存周转天数（DOS）红线对齐《企业库存周转健康度与呆滞品（滞销）管理办法》§3：
  食品≤20 天、数码3C≤40 天、化妆品≤60 天、服装鞋包≤90 天、家居建材≤120 天。

《S&OP 异常预警与跨部门协同审批流程管理办法》§3.2：连续两个月预测准确率均「不合格」
（即均 <75%）才升级为红色预警——演示「连续两月」时应使用 <75%，而非 <85%。
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta

random.seed(42)

categories = [
    ("化妆品", "high_oos", 60),
    ("食品", "fresh", 20),
    ("数码3C", "electronics", 40),
    ("服装鞋靴", "slow", 90),
    ("家居百货", "slow2", 120),
    ("母婴", "mixed", 45),
]

# 抽样落在 A/B/C 的比例 (pa, pb, pc)。化妆品略提高 C 档占比，便于演示「不合格 + 连续两月<75%」
tier_bias = {
    "化妆品": (0.22, 0.43, 0.35),
    "食品": (0.55, 0.35, 0.10),
    "数码3C": (0.50, 0.38, 0.12),
    "服装鞋靴": (0.35, 0.40, 0.25),
    "家居百货": (0.30, 0.45, 0.25),
    "母婴": (0.28, 0.42, 0.30),
}


def sample_accuracy(cat: str) -> float:
    r = random.random()
    pa, pb, _pc = tier_bias[cat]
    if r < pa:
        return round(random.uniform(85.0, 99.2), 2)
    if r < pa + pb:
        return round(random.uniform(75.0, 84.99), 2)
    return round(random.uniform(52.0, 74.99), 2)


def sample_oos(cat_key: str) -> float:
    r = random.random()
    if cat_key == "high_oos":
        if r < 0.55:
            return round(random.uniform(0.3, 2.9), 2)
        if r < 0.85:
            return round(random.uniform(3.1, 4.9), 2)
        return round(random.uniform(5.1, 11.0), 2)
    if r < 0.65:
        return round(random.uniform(0.2, 2.9), 2)
    if r < 0.90:
        return round(random.uniform(3.0, 4.9), 2)
    return round(random.uniform(5.0, 9.0), 2)


def accuracy_to_sales(accuracy_pct: float) -> tuple[int, int]:
    pred = int(random.uniform(3000, 80000))
    acc = accuracy_pct / 100.0
    delta = pred * (1 - acc) if acc <= 1 else pred * 0.01
    sign = random.choice([-1, 1])
    act = max(0, int(pred + sign * delta * random.uniform(0.85, 1.15)))
    return pred, act


def dos_for(cat_name: str, oos: float, acc: float) -> int:
    # 基准贴近 PDF 红线中位数略低，便于抖动后仍呈现实务感
    base = {
        "化妆品": 58,
        "食品": 18,
        "数码3C": 38,
        "服装鞋靴": 88,
        "家居百货": 108,
        "母婴": 48,
    }[cat_name]
    jitter = int((100 - acc) * 0.8 + oos * 3 + random.uniform(-6, 8))
    return max(5, min(200, base + jitter))


def main() -> None:
    start = date(2023, 1, 1)
    end = date(2026, 3, 31)
    day_span = (end - start).days

    rows: list[dict] = []
    for _ in range(1000):
        d = start + timedelta(days=random.randint(0, day_span))
        cat, ck, _ = random.choice(categories)
        acc = sample_accuracy(cat)
        pred, act = accuracy_to_sales(acc)
        oos = sample_oos(ck)
        dos = dos_for(cat, oos, acc)
        rows.append(
            {
                "日期": d.isoformat(),
                "商品类目": cat,
                "预测销量": pred,
                "实际销量": act,
                "预测准确率": f"{acc}%",
                "缺货率": f"{oos}%",
                "库存周转天数": dos,
            }
        )

    rows.sort(key=lambda x: (x["日期"], x["商品类目"]))

    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    out_path = root / "sop_knowledge" / "data.csv"

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "日期",
            "商品类目",
            "预测销量",
            "实际销量",
            "预测准确率",
            "缺货率",
            "库存周转天数",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    def parse_pct(s: str) -> float:
        return float(s.rstrip("%"))

    accs = [parse_pct(r["预测准确率"]) for r in rows]
    a_tier = sum(1 for x in accs if x >= 85)
    b_tier = sum(1 for x in accs if 75 <= x < 85)
    c_tier = sum(1 for x in accs if x < 75)
    print("wrote", len(rows), out_path)
    print("PDF §4.3  A优秀>=85%:", a_tier, "  B合格[75,85):", b_tier, "  C不合格<75%:", c_tier)
    for cat in ["化妆品", "食品", "数码3C"]:
        xs = [parse_pct(r["预测准确率"]) for r in rows if r["商品类目"] == cat]
        if xs:
            print(cat, "mean", round(sum(xs) / len(xs), 2), "n", len(xs))


if __name__ == "__main__":
    main()
