"""终端彩色输出：工具名、入参、观测（截断）。"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler


def _short(s: str, limit: int = 1800) -> str:
    s = s.replace("\r\n", "\n").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


class ToolTraceColors:
    TOOL = "\033[96m"  # cyan
    INPUT = "\033[93m"  # yellow
    OBS = "\033[92m"  # green
    DIM = "\033[2m"
    RESET = "\033[0m"


class ColoredToolCallback(BaseCallbackHandler):
    """在工具开始/结束时打印 Action（工具与参数）与 Observation。"""

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        c = ToolTraceColors
        name = (serialized or {}).get("name") or "tool"
        print()
        print(f"{c.TOOL}▶ [Action] 工具: {name}{c.RESET}")
        payload = input_str
        if inputs:
            try:
                payload = json.dumps(inputs, ensure_ascii=False)
            except TypeError:
                payload = str(inputs)
        print(f"{c.INPUT}  入参 (Inputs): {_short(str(payload), 1200)}{c.RESET}")

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        c = ToolTraceColors
        text = output if isinstance(output, str) else str(output)
        print(f"{c.OBS}  [Observation] {_short(text)}{c.RESET}")
        print(f"{c.DIM}  ────────────────────────────────────────────────{c.RESET}")
