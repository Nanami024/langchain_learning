"""表格路径：Pandas DataFrame Agent（不走向量库）。"""

from __future__ import annotations

import os
from typing import List, Union

import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

from .config import Settings
from . import document_io as dio


def load_dataframes(settings: Settings) -> dict:
    paths = dio.list_csv_paths(settings.docs_dir, settings.recursive)
    dfs: dict = {}
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        last_err = None
        for enc in ("utf-8-sig", "utf-8", "gbk"):
            try:
                dfs[name] = pd.read_csv(p, encoding=enc)
                break
            except UnicodeDecodeError as e:
                last_err = e
                continue
        else:
            if last_err:
                raise last_err
            dfs[name] = pd.read_csv(p)
    return dfs


def run_table_agent(question: str, settings: Settings) -> str:
    dfs = load_dataframes(settings)
    if not dfs:
        return "知识库目录中未找到 CSV 文件，无法进行数据分析。"

    # langchain_experimental>=0.4 仅接受 DataFrame 或 list[DataFrame]，不再接受 dict
    names: List[str] = sorted(dfs.keys())
    frames: List[pd.DataFrame] = [dfs[n] for n in names]
    if len(frames) == 1:
        df_arg: Union[pd.DataFrame, List[pd.DataFrame]] = frames[0]
        table_hint = (
            f"（单表：python_repl 中变量名为 df，来自 {names[0]}.csv）\n"
        )
    else:
        df_arg = frames
        mapping = "\n".join(
            f"  df{i + 1} → {names[i]}.csv" for i in range(len(names))
        )
        table_hint = (
            "多表：python_repl 中变量名为 df1, df2, …，对应关系：\n"
            f"{mapping}\n\n"
        )

    llm = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0,
    )
    # 默认 ReAct 文本格式易被国产/聊天模型“直接作答”绕过，触发 OUTPUT_PARSING_FAILURE；
    # tool-calling 走结构化工具调用，更稳；handle_parsing_errors 作兜底。
    agent = create_pandas_dataframe_agent(
        llm,
        df_arg,
        agent_type="tool-calling",
        verbose=False,
        allow_dangerous_code=True,
        max_iterations=20,
        agent_executor_kwargs={
            "handle_parsing_errors": True,
        },
    )
    out = agent.invoke({"input": table_hint + question})
    if isinstance(out, dict):
        return str(out.get("output", out))
    return str(out)
