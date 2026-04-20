"""Text-to-SQL 工具：内嵌 LangChain `create_sql_agent`，对外封装为 `sop_database_query_tool`。

两层防护，保证「零幻觉 + 零写操作」：
1. SQL Agent 系统提示中明确「只允许 SELECT；禁止 DROP/UPDATE/DELETE/INSERT/ALTER/TRUNCATE/CREATE 等任何写操作」。
2. SQLAlchemy `before_cursor_execute` 事件兜底拦截：任何首关键字不在白名单的语句都会抛出
   `SQLWriteAttempt`，由 LangChain 反馈给模型重试，物理上不可能落库。

为什么把 SQL Agent 包成一个 Tool 而不是直接挂到主 Agent？
- 复用 w4 的「主 Agent + Tool Calling」体系，不打断已有的对话历史与提示词；
- 主 Agent 仍然按照「文档 → sop_document_search、表格 → sop_database_query_tool」的语义分流，
  不需要重写 system prompt；
- 内层 SQL Agent 的中间步骤（sql_db_list_tables / sql_db_schema / sql_db_query）会通过
  RunnableConfig 把 callbacks 透传给 `tool_trace_callback.ListToolTraceCallback`，前端侧栏就能
  看到「大模型生成的 SQL」了。
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from sop_hub.config import Settings  # type: ignore  # 由 agent_runtime 注入 sys.path
from sop_hub.openai_http import chat_openai_http_kwargs  # type: ignore

from db_config import ro_engine, sales_table

# 与 w4 保持一致的「今天」语义，便于解析「上个月」等相对时间
_ASSUMED_TODAY = os.getenv("SOP_ASSUMED_TODAY", "2026-04-12")

_FORBIDDEN_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|RENAME|GRANT|REVOKE|MERGE|CALL)\b",
    re.IGNORECASE,
)

SQL_AGENT_PREFIX = f"""你是一名严谨的 MySQL 数据分析师，连接的是企业 S&OP 业务库。

【可用数据】
- 主表：`{{sales_table}}` —— 字段含义见下；除非问题明确要求，请只在该表上工作。
- 时间约定：除非用户另行说明，将「今天」视为 {_ASSUMED_TODAY}，据此解释「上个月」「本季度」等相对时间。
- 百分比列（forecast_accuracy、oos_rate）的取值范围是 0~100，已是百分数，写 SQL 时直接和阈值比较即可（例如 `forecast_accuracy >= 85`）。

【绝对禁令——零写操作】
- 你只允许执行 SELECT / WITH / SHOW / DESCRIBE / EXPLAIN 类只读语句。
- 严禁出现 INSERT / UPDATE / DELETE / DROP / ALTER / TRUNCATE / CREATE / REPLACE / RENAME / GRANT / REVOKE / MERGE / CALL 等任何会修改数据或结构的语句。
- 即便用户请求修改、清理、删除数据，也必须拒绝并提示「本接口仅支持查询」。后端还有 SQLAlchemy 兜底，违规语句会被直接驳回。

【工作流程】
1. 先调用 sql_db_list_tables 与 sql_db_schema 了解 `{{sales_table}}` 的字段与样例数据；
2. 写出语义最贴近问题的 SQL，必要时用 `LIMIT` 控制返回行数（默认 ≤10 行）；
3. 用 sql_db_query 执行；若有报错先用 sql_db_query_checker 自检后重试；
4. 把结果汇总成简洁中文，并附上你最终执行的那条 SQL（用 ``` 包裹），方便核对。

不要凭空编造列名或表名；若问题明显与本表无关，直接说明无法回答。"""


def _build_sql_db() -> SQLDatabase:
    """SQL Agent 看到的库视图：限制 include_tables，避免列出与业务无关的系统表。"""
    sample_rows = int(os.getenv("SOP_SQL_SAMPLE_ROWS", "3"))
    return SQLDatabase(
        engine=ro_engine(),
        include_tables=[sales_table()],
        sample_rows_in_table_info=sample_rows,
    )


@lru_cache(maxsize=1)
def _cached_db() -> SQLDatabase:
    return _build_sql_db()


def _validate_question_for_writes(question: str) -> str | None:
    """轻量预检：如果用户上层问题里就含 DROP/DELETE，提前拒绝并提示。"""
    m = _FORBIDDEN_PATTERN.search(question or "")
    if m:
        return f"本接口只支持查询，已拒绝包含 `{m.group(0).upper()}` 的请求。请改为只读问题。"
    return None


def build_sql_agent(settings: Settings):
    """每次构造一个新的 SQL AgentExecutor：内部状态较少，重建成本可接受，避免线程安全问题。"""
    req_timeout = float(os.getenv("SOP_LLM_REQUEST_TIMEOUT", "120"))
    max_wall = float(os.getenv("SOP_SQL_AGENT_MAX_SECONDS", "180"))
    top_k = int(os.getenv("SOP_SQL_TOP_K", "10"))

    llm = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0,
        request_timeout=req_timeout,
        **chat_openai_http_kwargs(),
    )
    toolkit = SQLDatabaseToolkit(db=_cached_db(), llm=llm)
    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type="tool-calling",
        prefix=SQL_AGENT_PREFIX.format(sales_table=sales_table()),
        verbose=False,
        max_iterations=15,
        max_execution_time=max_wall,
        top_k=top_k,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )


def build_sop_database_query_tool(settings: Settings):
    """返回一个 `@tool`，由主 Agent 通过 tool_calls 调用。"""

    @tool
    def sop_database_query_tool(question: str) -> str:
        """对企业 S&OP 业务库（MySQL）做只读查询并返回中文结论。

        何时调用：用户需要门店/区域/类目维度的销售、预测准确率、缺货率、库存周转等
        统计、筛选、排序、TopN——这些数值来自数据库 `sales_performance` 表，**不要**
        再尝试用 Pandas / 文件读取，也不要凭空给数字。

        参数 question：用自然语言描述完整的数据任务（必要时含时间、区域、类目过滤条件）；
        若用户用「这个准确率」「上面那家门店」等指代，请把上轮对话里出现过的具体实体
        （如门店编号、区域名、类目名、百分比阈值）写进 question 再调用本工具。

        返回：中文结论 + 实际执行的 SQL（包在 ``` 里）；
        若问题包含写操作（DROP/UPDATE/DELETE 等）会被拒绝。
        """
        q = (question or "").strip()
        if not q:
            return "错误：query 为空。"
        guard = _validate_question_for_writes(q)
        if guard:
            return guard
        try:
            agent = build_sql_agent(settings)
            # 不显式传 config：LangChain 会通过 ContextVar 把父链的 callbacks 透传给内层 SQL Agent，
            # 这样 ListToolTraceCallback 同样能捕获 sql_db_list_tables / sql_db_schema / sql_db_query 调用。
            out = agent.invoke({"input": q})
        except Exception as e:  # 包括 SQLWriteAttempt
            return f"SQL Agent 执行失败：{type(e).__name__}: {e}"
        if isinstance(out, dict):
            return str(out.get("output", out))
        return str(out)

    return sop_database_query_tool
