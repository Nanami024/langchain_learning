"""对比 w4 风格 invoke 与 w5 SSE 用的 astream_events：墙钟时间、事件量、LLM 启动次数。

在 w5/backend 目录执行（需已配置 w3/.env 与向量库）：

  python bench_timing.py
  python bench_timing.py --message "你好，简单自我介绍一句即可"
  python bench_timing.py --compare-streaming   # 同一问题，分别 llm_streaming=False / True 各跑一轮

说明：主 Agent 若开启 llm_streaming，部分网关的流式 HTTP 会比非流式明显更慢；astream_events 还会产生大量
Python 事件。Streamlit 前端若对每个 SSE 片段频繁 markdown，也会阻塞 iter_lines 造成反压。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from typing import Any

# 确保可导入同目录模块
_BACKEND = os.path.dirname(os.path.abspath(__file__))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import agent_runtime as ar  # noqa: E402
from perf_callbacks import LLMInvocationCounter  # noqa: E402


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


def _bootstrap_from_env(llm_streaming: bool) -> None:
    ar.reset_runtime()
    docs_dir = os.getenv("SOP_DOCS_DIR") or None
    recursive = _env_flag("SOP_RECURSIVE", "0")
    force_rebuild = _env_flag("SOP_FORCE_REBUILD", "0")
    ar.init_runtime(
        docs_dir=docs_dir,
        recursive=recursive,
        force_rebuild=force_rebuild,
        llm_streaming=llm_streaming,
    )


def _bench_invoke(message: str, session_id: str) -> dict[str, Any]:
    rt = ar.get_runtime()
    cb = LLMInvocationCounter()
    cfg: dict = {"configurable": {"session_id": session_id}, "callbacks": [cb]}
    t0 = time.perf_counter()
    out = rt.agent.invoke({"input": message}, config=cfg)
    elapsed = time.perf_counter() - t0
    text = out.get("output") if isinstance(out, dict) else str(out)
    return {
        "path": "invoke",
        "seconds": round(elapsed, 3),
        "llm_starts": cb.llm_starts,
        "tool_starts": cb.tool_starts,
        "output_chars": len(text or ""),
    }


async def _bench_astream(message: str, session_id: str) -> dict[str, Any]:
    rt = ar.get_runtime()
    cb = LLMInvocationCounter()
    cfg: dict = {"configurable": {"session_id": session_id}, "callbacks": [cb]}
    kinds: Counter[str] = Counter()
    t0 = time.perf_counter()
    async for ev in rt.agent.astream_events(
        {"input": message},
        config=cfg,
        version="v2",
    ):
        k = ev.get("event")
        if isinstance(k, str):
            kinds[k] += 1
    elapsed = time.perf_counter() - t0
    top = dict(kinds.most_common(12))
    stream_ev = kinds.get("on_chat_model_stream", 0)
    return {
        "path": "astream_events",
        "seconds": round(elapsed, 3),
        "llm_starts": cb.llm_starts,
        "tool_starts": cb.tool_starts,
        "event_total": int(sum(kinds.values())),
        "on_chat_model_stream": int(stream_ev),
        "event_top": top,
    }


def _print_row(title: str, row: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(row, ensure_ascii=False, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="S&OP w5 耗时对比（invoke vs astream_events）")
    p.add_argument("--message", default="你好，用一句话自我介绍即可，不要调用工具。", help="测试问题")
    p.add_argument(
        "--compare-streaming",
        action="store_true",
        help="先后以 llm_streaming=False / True 初始化，各跑 invoke + astream_events",
    )
    p.add_argument("--skip-astream", action="store_true", help="只测 invoke，节省一次完整 Agent 运行")
    args = p.parse_args()

    msg = args.message.strip()
    if not msg:
        print("message 为空")
        sys.exit(1)

    if args.compare_streaming:
        for streaming, label in ((False, "llm_streaming=False（对齐 w4 CLI）"), (True, "llm_streaming=True（w5 A 方案默认）")):
            print("\n" + "=" * 60)
            print(label)
            print("=" * 60)
            _bootstrap_from_env(llm_streaming=streaming)
            sid_i = f"bench-inv-{streaming}-{int(time.time())}"
            sid_a = f"bench-as-{streaming}-{int(time.time())}"
            _print_row("invoke", _bench_invoke(msg, sid_i))
            if not args.skip_astream:
                row_a = asyncio.run(_bench_astream(msg, sid_a))
                _print_row("astream_events", row_a)
        return

    # 单次：与 main.py lifespan 默认一致（A 方案）；可用环境变量覆盖
    llm_streaming = _env_flag("SOP_LLM_STREAMING", default="1")
    _bootstrap_from_env(llm_streaming=llm_streaming)
    sid_i = f"bench-inv-{int(time.time())}"
    _print_row("invoke", _bench_invoke(msg, sid_i))
    if not args.skip_astream:
        sid_a = f"bench-as-{int(time.time())}"
        row_a = asyncio.run(_bench_astream(msg, sid_a))
        _print_row("astream_events", row_a)

    print(
        "\n提示：若 astream_events 秒数显著大于 invoke，多为流式网关 + 事件循环开销；"
        "可设 SOP_LLM_STREAMING=0 且 SOP_SSE_ASTREAM_EVENTS=0 与 w4 对齐。"
    )


if __name__ == "__main__":
    main()
