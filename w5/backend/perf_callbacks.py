"""轻量回调：统计 LLM / 工具启动次数，用于排查重复调用。"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class LLMInvocationCounter(BaseCallbackHandler):
    def __init__(self) -> None:
        self.llm_starts = 0
        self.tool_starts = 0

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        self.llm_starts += 1

    def on_tool_start(self, *args: Any, **kwargs: Any) -> None:
        self.tool_starts += 1
