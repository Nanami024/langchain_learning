"""加载 w3 索引与 w4 Agent，供 FastAPI 复用（单例）。"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

# 仓库根目录 → 可 import w4（其 __init__ 会注入 w3 到 sys.path）
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import w4  # noqa: F401, E402 — 副作用：注册 sop_hub

from sop_hub import document_io as dio  # noqa: E402
from sop_hub.config import Settings, load_settings  # noqa: E402
from sop_hub.indexer import build_or_load_vectorstore, chunks_jsonl_path  # noqa: E402
from sop_hub.retriever import HybridRerankPipeline  # noqa: E402

from w4.agent import build_agent_with_history  # noqa: E402
from w4.tools import build_sop_tools  # noqa: E402


@dataclass
class AgentRuntime:
    settings: Settings
    agent: object  # RunnableWithMessageHistory
    session_store: dict[str, InMemoryChatMessageHistory]

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.session_store:
            self.session_store[session_id] = InMemoryChatMessageHistory()
        return self.session_store[session_id]


def _bootstrap(
    *,
    docs_dir: str | None,
    recursive: bool,
    force_rebuild: bool,
    llm_streaming: bool,
) -> AgentRuntime:
    settings = load_settings(docs_dir=docs_dir, recursive=recursive)
    if not settings.api_key or not settings.base_url:
        raise RuntimeError("请在仓库 w3/.env 或环境变量中配置 api_key 与 base_url")

    vs = build_or_load_vectorstore(settings, force_rebuild=force_rebuild)
    if vs is None:
        raise RuntimeError("向量库加载失败（检查索引或先运行 w4 建库）")

    jpath = chunks_jsonl_path(settings.vectorstore_dir)
    dio.export_vectorstore_documents_to_jsonl(vs, jpath)
    bm25_docs = dio.load_documents_from_jsonl(jpath)
    if not bm25_docs:
        raise RuntimeError("BM25 语料为空，请检查索引导出")

    pipeline = HybridRerankPipeline(vs, bm25_docs, settings)
    tools = build_sop_tools(pipeline, settings)
    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    agent = build_agent_with_history(
        settings,
        tools,
        get_session_history,
        llm_streaming=llm_streaming,
    )
    return AgentRuntime(settings=settings, agent=agent, session_store=store)


_runtime: AgentRuntime | None = None


def get_runtime() -> AgentRuntime:
    if _runtime is None:
        raise RuntimeError("Agent 运行时未初始化，请检查应用 lifespan")
    return _runtime


def init_runtime(
    *,
    docs_dir: str | None = None,
    recursive: bool = False,
    force_rebuild: bool = False,
    llm_streaming: bool = False,
) -> AgentRuntime:
    global _runtime
    _runtime = _bootstrap(
        docs_dir=docs_dir,
        recursive=recursive,
        force_rebuild=force_rebuild,
        llm_streaming=llm_streaming,
    )
    return _runtime


def reset_runtime() -> None:
    """清空单例，便于基准脚本用不同 llm_streaming 重建（勿在生产请求路径调用）。"""
    global _runtime
    _runtime = None
