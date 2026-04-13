"""create_tool_calling_agent + AgentExecutor + RunnableWithMessageHistory。"""

from __future__ import annotations

import os
from collections.abc import Callable

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from sop_hub.config import Settings
from sop_hub.openai_http import chat_openai_http_kwargs

# 与作业/数据脚本一致：data.csv 含 2026-03 等样本；可通过环境变量覆盖
_ASSUMED_TODAY = os.getenv("SOP_ASSUMED_TODAY", "2026-04-12")

SYSTEM_PROMPT = f"""你是「S&OP 决策中枢」助手。你本身不能访问数据库或 PDF，必须通过提供的工具完成：
- 需要手册条款、审批规则、阈值与流程时，调用 sop_document_search；
- 需要对业务 CSV 做统计、筛选、上月/类目/区域等指标时，调用 sop_data_analytics。

时间约定：除非用户另行说明，将「今天」视为 {_ASSUMED_TODAY}，据此解释「上个月」「本季度」等相对时间（例如在此日期下「上个月」为 2026 年 3 月）。

规则：
1. 严禁编造表格数值或手册条款；工具返回以外的具体数字一律视为未知。
2. 用户追问若含指代（「这个准确率」「上面说的」），先结合对话历史还原实体，再构造完整 query 调用工具。
3. 回答用户时使用简洁中文，必要时引用工具结果中的要点。"""


def build_agent_with_history(
    settings: Settings,
    tools: list[BaseTool],
    get_session_history: Callable[[str], BaseChatMessageHistory],
    *,
    llm_streaming: bool = False,
) -> RunnableWithMessageHistory:
    req_timeout = float(os.getenv("SOP_LLM_REQUEST_TIMEOUT", "120"))
    llm = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0,
        streaming=llm_streaming,
        request_timeout=req_timeout,
        **chat_openai_http_kwargs(),
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=15,
        handle_parsing_errors=True,
        # 彩色工具链在 app.py 的 invoke(config["callbacks"]) 传入，避免 RunnableWithMessageHistory 丢事件
    )
    return RunnableWithMessageHistory(
        executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="output",
    )
