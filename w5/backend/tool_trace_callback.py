"""收集工具调用轨迹，供 /chat JSON 响应与日志使用。"""

from __future__ import annotations

import json
import os
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler


def _tool_obs_max_chars() -> int:
    try:
        return max(400, int(os.getenv("SOP_TOOL_OBS_MAX_CHARS", "8000")))
    except ValueError:
        return 8000


def _short(s: str, limit: int = 1200) -> str:
    s = s.replace("\r\n", "\n").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


class ListToolTraceCallback(BaseCallbackHandler):
    """将 on_tool_start / on_tool_end 写入可序列化列表。"""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str = "",
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        name = (serialized or {}).get("name") or "tool"
        payload = input_str
        if inputs:
            try:
                payload = json.dumps(inputs, ensure_ascii=False)
            except TypeError:
                payload = str(inputs)
        self.entries.append(
            {
                "phase": "start",
                "tool": name,
                "inputs": _short(str(payload)),
            }
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        text = output if isinstance(output, str) else str(output)
        self.entries.append(
            {
                "phase": "end",
                "observation": _short(text, _tool_obs_max_chars()),
            }
        )
