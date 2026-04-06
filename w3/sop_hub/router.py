"""语义路由：文档检索 vs 表格分析（structured output）。"""

from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .config import Settings


class RouterDecision(BaseModel):
    route: Literal["document", "analytics"] = Field(
        description="document=手册/PDF/政策类查阅；analytics=对表格做统计/聚合/筛选"
    )
    reason: str = Field(default="", description="10 字内力理由")


ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是查询分类器，只输出结构化结果。
规则：
- 若用户要问的是 **手册条款、流程步骤、政策定义、SOP 正文里写了什么**（答案应在 PDF 叙述里）→ route=document。
- 若用户要 **对数据表做数值计算**（平均、求和、筛选、排序、按区域/月份统计、预测准确率等），且依赖 CSV/Excel 里的数字 → route=analytics。
含「上月」「华东区」「平均值」「共多少」等且明显针对业务表 → analytics。
含「流程」「规定」「手册第几章」「应如何审批」→ document。
不确定时偏 document（仍可通过 PDF 解释部分统计概念）。""",
        ),
        ("human", "用户问题：{question}"),
    ]
)


def build_router(settings: Settings):
    llm = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0,
    )
    return ROUTER_PROMPT | llm.with_structured_output(RouterDecision)


def route_question(router, question: str) -> RouterDecision:
    return router.invoke({"question": question})
