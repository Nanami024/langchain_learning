"""将混合检索与 Pandas 分析封装为 @tool，供 Agent 调用。"""

from __future__ import annotations

from langchain_core.tools import tool

from sop_hub import document_io as dio
from sop_hub.config import Settings
from sop_hub.retriever import HybridRerankPipeline
from sop_hub.table_agent import run_table_agent


def build_sop_tools(pipeline: HybridRerankPipeline, settings: Settings):
    """返回绑定检索管线与配置的两个 LangChain Tool。"""

    @tool
    def sop_document_search(query: str) -> str:
        """从 S&OP 手册与政策 PDF 向量库中检索相关原文片段。

        何时调用：用户询问「手册/ SOP 里怎么规定」「审批流程」「阈值/标准是多少」
        「缺货率超过多少要总监批」「政策条款」「定义」「应当如何做」等——答案必须来自
        正式文档叙述，而不是 CSV 数字表。

        参数 query：面向检索的完整问句；若用户用「这个」「该数值」「上面说的准确率」
        等指代，必须把对话里已出现的具体实体（类目名、指标名、百分比等）写进 query，
        以便命中正确条款。

        返回：带出处的拼接正文；若无命中则说明未检索到相关内容。"""
        q = (query or "").strip()
        if not q:
            return "错误：检索 query 为空。"
        try:
            outcome = pipeline.retrieve(q)
        except Exception as e:
            return f"检索失败：{e}"
        if not outcome.documents:
            return "知识库中未检索到与问题相关的 SOP 片段。"
        ctx = dio.format_docs_for_rag(outcome.documents)
        meta = (
            f"[检索摘要：粗排候选 {outcome.rough_count} 条；"
            f"Rerank={'开启' if outcome.used_rerank else '降级'}]\n\n"
        )
        return meta + ctx

    @tool
    def sop_data_analytics(question: str) -> str:
        """对知识库目录内 CSV 业务表做统计、筛选、聚合与排序（Pandas Agent）。

        何时调用：用户需要「上个月/某区域/某类目」的预测准确率、缺货率、销量汇总、
        同比、TopN、行数统计等——依赖表格中的数值列，而非 PDF 文字说明。

        参数 question：用自然语言描述完整的数据任务（可含时间、类目、区域等过滤条件）；
        若追问依赖上一轮数值，应把具体指标名和类目写清楚。

        返回：基于表格计算的文字结果；若无 CSV 则返回说明。"""
        q = (question or "").strip()
        if not q:
            return "错误：分析问题描述为空。"
        try:
            return run_table_agent(q, settings)
        except Exception as e:
            return f"数据分析失败：{e}"

    return [sop_document_search, sop_data_analytics]
