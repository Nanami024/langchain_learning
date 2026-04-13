"""表格路径：Pandas DataFrame Agent（不走向量库）。"""

from __future__ import annotations

import os
from typing import List, Union

import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

from .config import Settings
from . import document_io as dio

# 进程内缓存：避免每次 sop_data_analytics 都重新 read_csv（大表时很明显）。
# 子 Agent 可能在 REPL 里改 df，故对外每次返回 deep copy，保留缓存中的「干净」副本。
_df_cache_key: tuple | None = None
_df_cache: dict[str, pd.DataFrame] | None = None


def _dfs_fingerprint(settings: Settings) -> tuple:
    paths = sorted(
        dio.list_csv_paths(settings.docs_dir, settings.recursive),
        key=lambda p: str(p),
    )
    return (
        os.path.abspath(str(settings.docs_dir)),
        bool(settings.recursive),
        tuple((str(p), os.path.getmtime(p)) for p in paths),
    )


def load_dataframes(settings: Settings) -> dict[str, pd.DataFrame]:
    global _df_cache_key, _df_cache
    disable_cache = os.getenv("SOP_DISABLE_DF_CACHE", "").lower() in ("1", "true", "yes")
    if disable_cache:
        _df_cache_key, _df_cache = None, None
    key = _dfs_fingerprint(settings)
    if not disable_cache and key == _df_cache_key and _df_cache is not None:
        return {n: df.copy(deep=True) for n, df in _df_cache.items()}

    paths = dio.list_csv_paths(settings.docs_dir, settings.recursive)
    dfs: dict[str, pd.DataFrame] = {}
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
    if not disable_cache:
        _df_cache_key, _df_cache = key, dfs
    return {n: df.copy(deep=True) for n, df in dfs.items()}


def run_table_agent(question: str, settings: Settings) -> str:
    dfs = load_dataframes(settings)
    if not dfs:
        return "知识库目录中未找到 CSV 文件，无法进行数据分析。"

    # 单次 HTTP 超时，避免国产网关/代理挂起导致整段 SSE 永远停在「工具运行中」
    req_timeout = float(os.getenv("SOP_LLM_REQUEST_TIMEOUT", "120"))
    # Pandas 子 Agent 总时长上限（多轮 tool call）；到点返回而非无限阻塞
    max_wall = float(os.getenv("SOP_TABLE_AGENT_MAX_SECONDS", "180"))

    assumed_today = os.getenv("SOP_ASSUMED_TODAY", "2026-04-12")
    time_ctx = (
        f"时间上下文：将「今天」视为 {assumed_today}；"
        "用户若只说「上个月」「上周」等，请按该日期推算具体年月后再筛选 CSV。\n\n"
    )

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
        request_timeout=req_timeout,
    )
    # 默认 ReAct 文本格式易被国产/聊天模型“直接作答”绕过，触发 OUTPUT_PARSING_FAILURE；
    # tool-calling 走结构化工具调用，更稳；handle_parsing_errors 作兜底。
    agent = create_pandas_dataframe_agent(
        llm,
        df_arg,
        agent_type="tool-calling",
        verbose=False,
        allow_dangerous_code=True,
        max_iterations=15,
        max_execution_time=max_wall,
        agent_executor_kwargs={
            "handle_parsing_errors": True,
        },
    )
    out = agent.invoke({"input": time_ctx + table_hint + question})
    if isinstance(out, dict):
        return str(out.get("output", out))
    return str(out)
