"""ChatOpenAI 用的可选 httpx 配置。

在 Cursor / 部分 IDE 下启动的 uvicorn 会继承系统环境里的 HTTP_PROXY，硅基流动等直连 API 可能被错误代理截断，
httpx 只报 APIConnectionError: Connection error.。设 SOP_HTTPX_TRUST_ENV=0 可让 httpx 忽略代理环境变量（与多数终端下
未设代理时的行为一致）。
"""

from __future__ import annotations

import os
from typing import Any


def httpx_trust_system_proxy_env() -> bool:
    """为 True 时沿用 httpx 默认（读取 HTTP_PROXY 等）；为 False 时忽略。"""
    return os.getenv("SOP_HTTPX_TRUST_ENV", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def chat_openai_http_kwargs() -> dict[str, Any]:
    """返回可展开进 ChatOpenAI(...) 的 http_client / http_async_client；未关闭 trust_env 时为空 dict。"""
    if httpx_trust_system_proxy_env():
        return {}
    import httpx

    sec = float(os.getenv("SOP_LLM_REQUEST_TIMEOUT", "120"))
    t = httpx.Timeout(sec, connect=min(60.0, sec))
    return {
        "http_client": httpx.Client(trust_env=False, timeout=t),
        "http_async_client": httpx.AsyncClient(trust_env=False, timeout=t),
    }
