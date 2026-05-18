from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from sqlalchemy import text

from config import Settings
from database import get_readonly_engine
from retrieval_core import SimpleHashEmbeddings, hybrid_retrieve, tokenize


@dataclass
class SqlAnalysisResult:
    summary: str
    sql: str
    metrics: dict[str, Any]
    sources: list[dict[str, str]]


class DataAnalyst:
    def __init__(self) -> None:
        self.settings = Settings()
        self.sales_columns = self._get_sales_columns()
        self.embeddings = SimpleHashEmbeddings()
        self.data_docs = self._build_data_docs()
        self.vector = FAISS.from_documents(self.data_docs, self.embeddings)
        self.bm25 = BM25Retriever.from_documents(self.data_docs)
        self.bm25.k = self.settings.policy_bm25_k
        self.llm = self._build_llm()
        self._schema_cache: str | None = None

    def _build_llm(self) -> ChatOpenAI | None:
        if not self.settings.llm_api_key or not self.settings.llm_base_url:
            return None
        try:
            return ChatOpenAI(
                model=self.settings.llm_model,
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url,
                temperature=0,
                request_timeout=self.settings.sql_timeout_seconds,
            )
        except Exception:
            return None

    def _build_data_docs(self) -> list[Document]:
        docs: list[Document] = []
        sales_table = self.settings.sales_table
        alias_map = self._column_aliases()
        docs.append(
            Document(
                page_content=(
                    f"table={sales_table}; domain=s&op 销售预测 需求计划 补货 缺货 库存周转 准确率 "
                    "forecast accuracy inventory stock oos turnover"
                ),
                metadata={"source_type": "domain", "source_id": sales_table},
            )
        )
        for col in self.sales_columns:
            aliases = " ".join(alias_map.get(col.lower(), []))
            docs.append(
                Document(
                    page_content=f"table={sales_table}; column={col}; aliases={aliases}",
                    metadata={"source_type": "schema", "source_id": col},
                )
            )
        try:
            with get_readonly_engine().connect() as conn:
                rows = conn.execute(text(f"SELECT DISTINCT region FROM {sales_table} LIMIT 30")).all()
            for row in rows:
                rg = str(row[0] or "").strip()
                if rg:
                    docs.append(
                        Document(
                            page_content=f"table={sales_table}; dimension=region; value={rg}",
                            metadata={"source_type": "dimension", "source_id": rg},
                        )
                    )
        except Exception:
            pass
        if not docs:
            docs.append(Document(page_content=f"table={sales_table}", metadata={"source_type": "schema"}))
        return docs

    def _column_aliases(self) -> dict[str, list[str]]:
        return {
            "biz_date": ["日期", "时间", "月份", "上个月", "本月", "周"],
            "region": ["区域", "大区", "华东", "华南", "华北", "城市"],
            "store_code": ["门店", "店铺", "仓"],
            "sku": ["商品", "品类", "sku", "单品"],
            "forecast_accuracy": ["预测准确率", "准确率", "偏差", "forecast accuracy"],
            "oos_rate": ["缺货率", "断货率", "oos"],
            "inventory_turnover_days": ["库存周转", "周转天数", "库存天数", "呆滞"],
            "forecast_qty": ["预测量", "预测销量", "需求预测"],
            "actual_qty": ["实际销量", "实际量", "出货量"],
            "category": ["品类", "类别", "分类"],
        }

    def _get_sales_columns(self) -> list[str]:
        sales_table = self.settings.sales_table
        with get_readonly_engine().connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT COLUMN_NAME
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = :db AND TABLE_NAME = :tbl
                    ORDER BY ORDINAL_POSITION
                    """
                ),
                {"db": self.settings.mysql_db, "tbl": sales_table},
            ).all()
            return [str(r[0]) for r in rows]

    def hybrid_relevance(self, question: str, *, final_k: int = 3) -> tuple[float, float, float, list[str]]:
        fused = hybrid_retrieve(
            query=question,
            vector=self.vector,
            bm25=self.bm25,
            faiss_k=self.settings.policy_faiss_k,
            rough_k=self.settings.policy_rough_k,
            final_k=final_k,
            faiss_weight=self.settings.policy_faiss_weight,
            bm25_weight=self.settings.policy_bm25_weight,
        )
        blended = min(1.0, 0.6 * float(fused.score) + 0.4 * float(fused.token_coverage))
        snippets = [str(d.page_content or "")[:280] for d in fused.docs[:final_k]]
        return blended, float(fused.score), float(fused.token_coverage), snippets

    def query_data_relevance(self, question: str) -> float:
        blended, _, _, _ = self.hybrid_relevance(question, final_k=2)
        return blended

    def _is_simple_metric_query(self, question: str) -> bool:
        q = (question or "").strip()
        if not q:
            return False
        metric_words = (
            "准确率",
            "缺货",
            "周转",
            "预测",
            "多少",
            "几",
            "均值",
            "平均",
            "占比",
            "同比",
            "环比",
        )
        if not any(w in q for w in metric_words):
            return False
        if any(w in q for w in ("建议", "怎么做", "怎么办", "方案", "制度", "条款", "审批")):
            return False
        return True

    _NL_SQL_HINTS_TIME = (
        "上个月",
        "上月",
        "本月",
        "本周",
        "上周",
        "本年",
        "去年",
        "前年",
        "biz_date",
        "日期",
        "哪天",
        "几月",
        "季度",
        "财年",
        "最近",
        "近",
        "过去",
        "未来",
        "从20",
        "-20",
        "/20",
    )
    _NL_SQL_HINTS_COMPLEX = (
        "排名",
        "top",
        "前几",
        "第几",
        "最多",
        "最少",
        "最大",
        "最小",
        "分组",
        "拆分",
        "分别统计",
        "分别算",
        "分别属于",
        "各个品类",
        "各品类",
        "每个品类",
        "分品类",
        "按品类",
        "每个门店",
        "各门店",
        "各sku",
        "逐",
    )

    def _question_needs_nl_sql(self, question: str) -> bool:
        """本地快路径未覆盖时间窗、同比环比、多维拆分等口径时需要 NL2SQL。"""
        q = (question or "").strip()
        if not q:
            return False
        ql = q.lower()
        if any(h in q for h in self._NL_SQL_HINTS_TIME):
            return True
        if any(h in ql for h in ("yoy", "mom", "wow", "q1", "q2", "q3", "q4")):
            return True
        if "同比" in q or "环比" in q:
            return True
        if any(h in q for h in self._NL_SQL_HINTS_COMPLEX):
            return True
        if re.search(r"\b20\d{2}-\d{1,2}", q):
            return True
        return False

    def _wants_per_category_breakdown(self, question: str) -> bool:
        """是否要按 category 维度拆行（各品类/分别…准确率或等级），快路径不能只做全表 AVG。"""
        q = (question or "").strip()
        if not q or "category" not in self.sales_columns:
            return False
        by_cat_phrase = bool(
            re.search(r"(各(?:个)?品类|每个品类|分品类|按品类|逐(?:个)?品类|品类维度)", q)
            or (re.search(r"品类", q) and bool(re.search(r"(分别|各自|拆分|分组|对比)", q)))
        )
        if not by_cat_phrase:
            return False
        return bool(re.search(r"(准确率|准确|预测|缺货|等级|评级|[ABC]\s*级|优秀|合格|不合格)", q))

    def _per_category_needs_time_filter(self, question: str) -> bool:
        """按品类拆分时是否要带时间窗（需交给 NL2SQL，本地快捷 GROUP BY 不做日历过滤）。"""
        q = (question or "").strip()
        if not q:
            return False
        ql = q.lower()
        return (
            any(h in q for h in self._NL_SQL_HINTS_TIME)
            or "同比" in q
            or "环比" in q
            or any(h in ql for h in ("yoy", "mom", "wow", "q1", "q2", "q3", "q4"))
            or bool(re.search(r"\b20\d{2}-\d{1,2}", q))
            or bool(re.search(r"\b20\d{2}\b", q))
        )

    @staticmethod
    def _sop_forecast_accuracy_tier(avg_acc: float) -> str:
        """与制度 PDF 口径一致：A≥85，B∈[75,85)，C<75。"""
        if avg_acc >= 85.0:
            return "A（优秀）"
        if avg_acc >= 75.0:
            return "B（合格）"
        return "C（不合格）"

    def _should_use_fast_sql_path(self, question: str) -> bool:
        return self._is_simple_metric_query(question) and not self._question_needs_nl_sql(question)

    def _timing_phase(self, label: str, t0: float) -> None:
        if self.settings.data_analyst_timing:
            logger.info("data_analyst phase=%s elapsed_s=%.3f", label, time.perf_counter() - t0)

    def analyze(self, question: str) -> SqlAnalysisResult:
        q = (question or "").strip()
        if not q:
            return SqlAnalysisResult(summary="问题为空，无法分析。", sql="", metrics={}, sources=[])

        t_all = time.perf_counter()
        cheap_ok = self._should_use_fast_sql_path(q)
        # cheap_ok 且未强制 NL：直接本地 SQL，避免链路上 2 次 Chat（生成 SQL + 总结）
        use_nl = self.llm is not None and (not cheap_ok or self.settings.data_analyst_llm_first)

        if use_nl:
            llm_result = self._try_nl_sql_path(q)
            if llm_result is not None:
                self._timing_phase("analyze_total_nl_ok", t_all)
                return llm_result

        t_fb = time.perf_counter()
        out = self._fallback_overview(q)
        self._timing_phase("fallback_overview", t_fb)
        self._timing_phase("analyze_total_fallback", t_all)
        return out

    def _try_nl_sql_path(self, question: str) -> SqlAnalysisResult | None:
        """语义生成 SQL 并查询；失败则返回 None，交由规则兜底。"""
        if self.llm is None:
            return None
        t0 = time.perf_counter()
        schema = self._schema_context()
        self._timing_phase("nl_sql_schema", t0)
        t1 = time.perf_counter()
        sql_text = self._generate_sql(question, schema)
        self._timing_phase("nl_sql_generate", t1)
        if not sql_text:
            return None
        safe_sql = self._validate_sql(sql_text)
        if not safe_sql:
            return None
        t2 = time.perf_counter()
        try:
            rows = self._execute_query(safe_sql)
        except Exception:
            return None
        self._timing_phase("nl_sql_execute", t2)
        t3 = time.perf_counter()
        summary = self._summarize_rows_heuristic(question, safe_sql, rows)
        if summary is None:
            summary = self._summarize(question, safe_sql, rows)
        self._timing_phase("nl_sql_summarize", t3)
        metrics = {"row_count": len(rows), "preview": rows[: min(3, len(rows))]}
        return SqlAnalysisResult(
            summary=summary,
            sql=safe_sql,
            metrics=metrics,
            sources=[
                {"type": "table", "name": self.settings.sales_table},
                {"type": "csv", "name": "data/sales_performance.csv"},
            ],
        )

    def _schema_context(self) -> str:
        if self._schema_cache:
            return self._schema_cache
        tbl = self.settings.sales_table
        preferred = [
            "biz_date",
            "category",
            "region",
            "store_code",
            "sku",
            "forecast_accuracy",
            "oos_rate",
            "inventory_turnover_days",
        ]
        cols = [c for c in preferred if c in self.sales_columns] or self.sales_columns[: min(7, len(self.sales_columns))]
        if not cols:
            cols = ["*"]
        col_sql = ", ".join(cols)
        with get_readonly_engine().connect() as conn:
            sample_rows = conn.execute(
                text(
                    f"""
                    SELECT {col_sql}
                    FROM {tbl}
                    LIMIT 3
                    """
                )
            ).mappings().all()
        dim_hints: list[str] = []
        for dcol in ("category", "region"):
            if dcol in self.sales_columns:
                vals = self._distinct_values(dcol, limit=24)
                if vals:
                    dim_hints.append(f"{dcol}_examples: {', '.join(vals[:24])}")
        dim_block = ("\n" + "\n".join(dim_hints) + "\n") if dim_hints else ""
        ctx = (
            f"table: {tbl}\n"
            f"columns: {', '.join(self.sales_columns)}\n"
            f"today_assumption: {self.settings.assumed_today}\n"
            f"{dim_block}"
            "dimension_values_note: 上列为库中已出现的枚举示例，仅供对齐用词；"
            "仅在用户明确要求按某品类/区域/门店等口径筛选或对比时加入 WHERE/GROUP BY，"
            "不要仅因问题里出现过某个词就强行筛选。\n"
            f"sample_rows: {json.dumps([dict(r) for r in sample_rows], ensure_ascii=False, default=str)}"
        )
        self._schema_cache = ctx
        return ctx

    def _generate_sql(self, question: str, schema: str) -> str:
        if self.llm is None:
            return ""
        prompt = (
            "你是企业数据库分析助手。根据用户问题生成一条可执行 SQL。\n"
            "必须满足：\n"
            "1) 只允许 SELECT 或 WITH ... SELECT\n"
            "2) 只能查询给定表，不可使用其他表\n"
            "3) 不允许 INSERT/UPDATE/DELETE/DDL\n"
            "4) 结果最多返回 30 行\n"
            "5) 库中枚举值仅作对齐用：只有用户明确要求按某品类/区域等口径筛选、对比、或拆分时，"
            "才在 SQL 中加入相应条件；不要仅因问题正文里出现了某个词就自动 WHERE。\n"
            "6) 若用户问方案/假设（如减少供应），查询应用现有字段给出相关基线指标（准确率、缺货率、周转等），"
            "不要编造不存在的列。\n"
            "只输出 JSON: {\"sql\": \"...\"}，不要输出其他文本。"
        )
        user = f"问题：{question}\n\n数据库信息：\n{schema}"
        try:
            msg = self.llm.invoke([SystemMessage(content=prompt), HumanMessage(content=user)])
            raw = str(msg.content or "").strip()
            obj = json.loads(raw)
            sql_text = str(obj.get("sql", "")).strip()
            return self._strip_sql_fence(sql_text)
        except Exception:
            return ""

    def _strip_sql_fence(self, sql_text: str) -> str:
        out = sql_text.strip()
        if out.startswith("```"):
            out = re.sub(r"^```[a-zA-Z]*\s*", "", out)
            out = re.sub(r"\s*```$", "", out)
        return out.strip().rstrip(";")

    def _validate_sql(self, sql_text: str) -> str:
        q = sql_text.strip().lower()
        if not q or (not q.startswith("select") and not q.startswith("with")):
            return ""
        forbidden = ("insert ", "update ", "delete ", "drop ", "alter ", "truncate ", "create ", "replace ", "grant ", "revoke ")
        if any(x in q for x in forbidden):
            return ""
        tables = set(re.findall(r"\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)|\bjoin\s+([a-zA-Z_][a-zA-Z0-9_]*)", q))
        flat = {x for pair in tables for x in pair if x}
        allowed = {self.settings.sales_table.lower()}
        if flat and not flat.issubset(allowed):
            return ""
        if "limit" not in q:
            return f"{sql_text} LIMIT {self.settings.sql_max_rows}"
        return sql_text

    def _execute_query(self, sql_text: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with get_readonly_engine().connect() as conn:
            rows = conn.execute(text(sql_text), params or {}).mappings().fetchmany(self.settings.sql_max_rows)
        return [dict(r) for r in rows]

    def _summarize_rows_heuristic(self, question: str, sql_text: str, rows: list[dict[str, Any]]) -> str | None:
        """小结果集直接叙述，省掉第二次 LLM（卡顿常来自连续两次 invoke）。"""
        del sql_text
        if not rows:
            return "查询未返回任何行；请放宽条件或检查时间范围。"
        q = (question or "").strip()

        if len(rows) == 1:
            r = rows[0]
            if all(k in r for k in ("avg_acc", "avg_oos", "n")):
                acc = float(r.get("avg_acc") or 0.0)
                oos = float(r.get("avg_oos") or 0.0)
                n = int(r.get("n") or 0)
                if n == 0:
                    return "汇总样本量为 0，暂无可用指标。"
                return f"汇总：预测准确率约 {acc:.2f}%，平均缺货率约 {oos:.2f}%（n={n}）。"
            if len(r) <= 8:
                return "查询结果：" + "，".join(f"{k}={v}" for k, v in r.items())

        dim_key = next((k for k in rows[0] if str(k).lower() == "dim"), None)
        if dim_key and len(rows) <= 15 and all(dim_key in r for r in rows):
            want_oos = any(w in q for w in ("缺货", "断货", "oos")) and "准确" not in q
            parts: list[str] = []
            for r in rows:
                d = str(r.get(dim_key) or "").strip() or "(空)"
                acc = float(r.get("avg_acc") or 0.0)
                oos = float(r.get("avg_oos") or 0.0)
                n = int(r.get("n") or 0)
                if want_oos:
                    parts.append(f"{d} 平均缺货率约 {oos:.2f}%（n={n}）")
                else:
                    parts.append(f"{d} 预测准确率约 {acc:.2f}%（n={n}）")
            return "分组对比：" + "；".join(parts) + "。"

        if len(rows) <= 8:
            brief = json.dumps(rows, ensure_ascii=False, default=str)
            if len(brief) < 1000:
                return f"共 {len(rows)} 行。{brief}"
        return None

    def _summarize(self, question: str, sql_text: str, rows: list[dict[str, Any]]) -> str:
        if self.llm is None:
            return f"已完成数据分析，返回 {len(rows)} 条记录。"
        prompt = (
            "你是企业数据分析顾问。请用自然语言直接回答用户问题（2-4句），先给结论再列关键数字。\n"
            "不得粘贴原始表格或 JSON，不得编造。\n"
            "若用户问评级/等级/档次，只报告当前指标数值；等级判定留给制度解读环节。"
        )
        user = (
            f"用户问题：{question}\n"
            f"SQL：{sql_text}\n"
            f"结果：{json.dumps(rows, ensure_ascii=False, default=str)}"
        )
        try:
            msg = self.llm.invoke([SystemMessage(content=prompt), HumanMessage(content=user)])
            out = str(msg.content or "").strip()
            if out:
                return out
        except Exception:
            pass
        return f"已完成数据分析，返回 {len(rows)} 条记录。"

    def _distinct_values(self, column: str, limit: int = 80) -> list[str]:
        if column not in self.sales_columns:
            return []
        tbl = self.settings.sales_table
        try:
            with get_readonly_engine().connect() as conn:
                rows = conn.execute(
                    text(
                        f"SELECT DISTINCT `{column}` FROM `{tbl}` "
                        f"WHERE `{column}` IS NOT NULL AND `{column}` <> '' LIMIT {int(limit)}"
                    )
                ).all()
            return [str(r[0]).strip() for r in rows if str(r[0] or "").strip()]
        except Exception:
            return []

    def _mention_used_as_data_scope(self, question: str, val: str, column: str) -> bool:
        """用户是否明显把该取值当作数据口径/筛选条件（而非随口提到）。"""
        q = (question or "").strip()
        if not val or val not in q:
            return False
        if f"{val}的" in q:
            pos = q.find(f"{val}的")
            tail = q[pos : pos + len(val) + 24] if pos >= 0 else q
            policy_noise = ("制度", "办法", "规定", "流程", "传闻", "政策", "条例", "标准文书")
            if any(w in tail for w in policy_noise):
                pass
            else:
                return True
        if (
            f"在{val}" in q
            or f"对{val}" in q
            or f"针对{val}" in q
            or f"就{val}" in q
            or f"从{val}" in q
        ):
            return True
        if column == "region" and (f"{val}大区" in q or f"{val}地区" in q):
            return True
        if column == "category":
            for suf in ("品类", "类别", "类目", "大类"):
                if f"{val}{suf}" in q:
                    return True
        metrics = ("准确率", "缺货", "周转", "预测", "库存", "oos", "偏差", "均值", "平均")
        start = 0
        while True:
            pos = q.find(val, start)
            if pos < 0:
                break
            after = q[pos + len(val) : pos + len(val) + 12]
            compact = after.lstrip()
            if any(compact.startswith(m) for m in metrics):
                return True
            start = pos + 1
        return False

    def _dimension_values_mentioned(self, question: str) -> dict[str, list[str]]:
        """All DB dimension values whose literal appears in the user question (per column)."""
        q = str(question or "")
        out: dict[str, list[str]] = {}
        for col in ("category", "region"):
            if col not in self.sales_columns:
                continue
            seen: set[str] = set()
            matched: list[str] = []
            for val in self._distinct_values(col):
                if not val or val not in q or val in seen:
                    continue
                seen.add(val)
                matched.append(val)
            if matched:
                out[col] = matched
        return out

    def _fallback_compare_dimension(
        self, question: str, col: str, filter_vals: list[str], other_filters: list[tuple[str, str]]
    ) -> SqlAnalysisResult:
        """GROUP BY one dimension when the user names 2+ values of that dimension (e.g. 品类对比)."""
        tbl = self.settings.sales_table
        params: dict[str, Any] = {}
        or_parts: list[str] = []
        for i, val in enumerate(filter_vals):
            key = f"cmp_{col}_{i}"
            or_parts.append(f"`{col}` LIKE :{key}")
            params[key] = f"%{val}%"
        and_parts: list[str] = ["(" + " OR ".join(or_parts) + ")"]
        for i, (ocol, oval) in enumerate(other_filters):
            key = f"cmp_other_{i}"
            and_parts.append(f"`{ocol}` LIKE :{key}")
            params[key] = f"%{oval}%"
        where_sql = " AND ".join(and_parts)
        sql = (
            f"SELECT `{col}` AS dim, AVG(forecast_accuracy) AS avg_acc, AVG(oos_rate) AS avg_oos, COUNT(*) AS n "
            f"FROM `{tbl}` WHERE {where_sql} GROUP BY `{col}` ORDER BY avg_acc DESC"
        )
        rows = self._execute_query(sql, params=params)
        q = str(question or "")
        want_acc = any(w in q for w in ("准确率", "准确", "预测准"))
        want_oos = any(w in q for w in ("缺货", "断货", "oos"))
        metric_key = "avg_oos" if want_oos and not want_acc else "avg_acc"
        metric_label = "平均缺货率" if metric_key == "avg_oos" else "平均预测准确率"

        lines: list[str] = []
        total_n = 0
        by_dim: dict[str, dict[str, Any]] = {}
        for r in rows:
            dim = str(r.get("dim") or "").strip() or "(空)"
            acc = float(r.get("avg_acc") or 0.0)
            oos = float(r.get("avg_oos") or 0.0)
            n = int(r.get("n") or 0)
            total_n += n
            by_dim[dim] = {"avg_acc": acc, "avg_oos": oos, "n": n}
            if metric_key == "avg_oos":
                lines.append(f"{dim}：{metric_label}约 {oos:.2f}%（样本行 n={n}）")
            else:
                lines.append(f"{dim}：{metric_label}约 {acc:.2f}%（样本行 n={n}）")

        if not rows:
            scope = "、".join(filter_vals)
            summary = f"未检索到可调用的对比数据（{scope}）。请确认问题中的品类/区域与库中明细一致。"
            return SqlAnalysisResult(
                summary=summary,
                sql=sql,
                metrics={"row_count": 0, "by_dim": {}},
                sources=[
                    {"type": "table", "name": self.settings.sales_table},
                    {"type": "csv", "name": "data/sales_performance.csv"},
                ],
            )

        ordered = sorted(
            by_dim.items(),
            key=lambda kv: float(kv[1].get(metric_key) or 0.0),
            reverse=True,
        )
        best_dim, best_m = ordered[0]
        best_val = float(best_m[metric_key] or 0.0)
        if metric_key == "avg_oos" and len(by_dim) > 1:
            worst_dim, low_m = ordered[-1]
            low_v = float(low_m.get("avg_oos") or 0.0)
            headline = f"就{metric_label}而言，{best_dim}更高（约 {best_val:.2f}%）。相对更低的是 {worst_dim}（约 {low_v:.2f}%）。"
        elif len(by_dim) > 1:
            headline = f"就{metric_label}而言，{best_dim}更高（约 {best_val:.2f}%）。"
        else:
            headline = f"{best_dim}：{metric_label}约 {best_val:.2f}%。"
        missing = [v for v in filter_vals if not any(v == d or v in d or d in v for d in by_dim)]
        miss_note = f" 注意：库中未出现与「{'、'.join(missing)}」完全匹配的{col}明细分组。" if missing else ""
        summary = headline + " " + " ".join(lines) + miss_note
        return SqlAnalysisResult(
            summary=summary.strip(),
            sql=sql,
            metrics={"row_count": total_n, "by_dim": by_dim},
            sources=[
                {"type": "table", "name": self.settings.sales_table},
                {"type": "csv", "name": "data/sales_performance.csv"},
            ],
        )

    def _fallback_all_categories(self, question: str) -> SqlAnalysisResult:
        """按品类 GROUP BY：解决「各品类准确率/等级」但问题里未点出两个具体品类名的情况。"""
        tbl = self.settings.sales_table
        q = str(question or "")
        sql = (
            f"SELECT `category` AS dim, AVG(forecast_accuracy) AS avg_acc, AVG(oos_rate) AS avg_oos, COUNT(*) AS n "
            f"FROM `{tbl}` WHERE `category` IS NOT NULL AND TRIM(`category`) <> '' "
            f"GROUP BY `category` ORDER BY `category` ASC"
        )
        rows = self._execute_query(sql)
        want_tier = bool(re.search(r"(等级|评级|档|档级|[ABC]级|优秀|合格|不合格)", q))
        by_dim: dict[str, dict[str, Any]] = {}
        lines: list[str] = []
        total_n = 0
        for r in rows:
            dim = str(r.get("dim") or "").strip() or "(空)"
            acc = float(r.get("avg_acc") or 0.0)
            oos = float(r.get("avg_oos") or 0.0)
            n = int(r.get("n") or 0)
            total_n += n
            tier = self._sop_forecast_accuracy_tier(acc)
            by_dim[dim] = {"avg_acc": acc, "avg_oos": oos, "n": n, "tier": tier}
            seg = f"{dim}：平均预测准确率约 {acc:.2f}%"
            if want_tier:
                seg += f"，按制度口径预测准确率档位约 {tier}"
            seg += f"；平均缺货率约 {oos:.2f}%（样本行 n={n}）。"
            lines.append(seg)

        if not rows:
            return SqlAnalysisResult(
                summary="库中无可用品类维度数据（category 为空或不存在）。",
                sql=sql,
                metrics={"row_count": 0, "by_dim": {}},
                sources=[
                    {"type": "table", "name": self.settings.sales_table},
                    {"type": "csv", "name": "data/sales_performance.csv"},
                ],
            )

        headline = f"按品类汇总（共 {len(rows)} 个品类，取值库内 biz_date 全量平均）："
        summary = headline + " " + " ".join(lines)
        return SqlAnalysisResult(
            summary=summary.strip(),
            sql=sql,
            metrics={"row_count": total_n, "by_dim": by_dim, "preview": rows[:12]},
            sources=[
                {"type": "table", "name": self.settings.sales_table},
                {"type": "csv", "name": "data/sales_performance.csv"},
            ],
        )

    def _fallback_overview(self, question: str = "") -> SqlAnalysisResult:
        q = str(question or "")
        if self._wants_per_category_breakdown(q) and not self._per_category_needs_time_filter(q):
            return self._fallback_all_categories(q)
        dim_hit = self._dimension_values_mentioned(q)

        compare_col: str | None = None
        compare_vals: list[str] | None = None
        for col in ("category", "region"):
            vals = dim_hit.get(col) or []
            if len(vals) >= 2:
                compare_col = col
                compare_vals = vals
                break

        if compare_col and compare_vals:
            other_filters: list[tuple[str, str]] = []
            for col in ("category", "region"):
                if col == compare_col:
                    continue
                one = dim_hit.get(col)
                if (
                    one
                    and len(one) == 1
                    and self._mention_used_as_data_scope(q, one[0], col)
                ):
                    other_filters.append((col, one[0]))
            return self._fallback_compare_dimension(q, compare_col, compare_vals, other_filters)

        where_sql: list[str] = []
        params: dict[str, Any] = {}
        tags: list[str] = []
        for col in ("category", "region"):
            if col not in self.sales_columns:
                continue
            one = dim_hit.get(col)
            if (
                one
                and len(one) == 1
                and self._mention_used_as_data_scope(q, one[0], col)
            ):
                key = f"dim_{col}"
                where_sql.append(f"`{col}` LIKE :{key}")
                params[key] = f"%{one[0]}%"
                tags.append(one[0])

        where_clause = f" WHERE {' AND '.join(where_sql)}" if where_sql else ""
        sql = (
            f"SELECT AVG(forecast_accuracy) AS avg_acc, AVG(oos_rate) AS avg_oos, COUNT(*) AS n "
            f"FROM {self.settings.sales_table}{where_clause}"
        )
        rows = self._execute_query(sql, params=params)
        avg_acc = float(rows[0].get("avg_acc") or 0.0) if rows else 0.0
        avg_oos = float(rows[0].get("avg_oos") or 0.0) if rows else 0.0
        n = int(rows[0].get("n") or 0) if rows else 0
        scope = "、".join(tags) if tags else "整体"
        if n == 0:
            if tags:
                summary = f"未检索到{scope}相关数据。当前数据库中不存在可用于该筛选条件的记录。"
            else:
                summary = "数据库暂无可用记录。"
            avg_acc = 0.0
            avg_oos = 0.0
        else:
            summary = f"{scope}范围内预测准确率约 {avg_acc:.2f}%，平均缺货率约 {avg_oos:.2f}%。"
        return SqlAnalysisResult(
            summary=summary,
            sql=sql,
            metrics={"avg_accuracy": avg_acc, "avg_oos": avg_oos, "row_count": n},
            sources=[
                {"type": "table", "name": self.settings.sales_table},
                {"type": "csv", "name": "data/sales_performance.csv"},
            ],
        )
