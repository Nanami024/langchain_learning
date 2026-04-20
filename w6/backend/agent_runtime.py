"""复用 w3 索引 + w4 主 Agent 框架；持久化记忆与 SQL 工具替换为 w6 实现。

变化点（相对 w5）：
- 会话历史从 InMemoryChatMessageHistory 换成 SQLChatMessageHistory（MySQL）。
- 数据分析工具从 Pandas Agent 换成 Text-to-SQL（`sop_database_query_tool`）。
- 文档检索工具 `sop_document_search` 与 w4 完全一致，直接复用，不重复造轮子。
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from langchain_core.chat_history import BaseChatMessageHistory

# 仓库根 → 注入 sys.path，便于 `import w4` 触发其 __init__ 把 w3 也加进来
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_module_from_file(alias: str, path: Path) -> ModuleType:
    """按文件路径以唯一别名加载模块，避免污染 sys.path 与 main.py 同名冲突。"""
    spec = importlib.util.spec_from_file_location(alias, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法定位模块文件：{path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


# 复用 w5 的可观测性回调（不在 w6 复制一份），但用唯一别名加载，避免 w5/backend/main.py
# 也被解析成 `main` 而覆盖 w6 自己的 FastAPI 入口
_W5_BACKEND = _REPO_ROOT / "w5" / "backend"
_perf = _load_module_from_file("w6_perf_callbacks", _W5_BACKEND / "perf_callbacks.py")
_trace = _load_module_from_file("w6_tool_trace_callback", _W5_BACKEND / "tool_trace_callback.py")
LLMInvocationCounter = _perf.LLMInvocationCounter
ListToolTraceCallback = _trace.ListToolTraceCallback

import w4  # noqa: F401, E402 — 副作用：注册 sop_hub

from sop_hub import document_io as dio  # noqa: E402
from sop_hub.config import Settings, load_settings  # noqa: E402
from sop_hub.indexer import build_or_load_vectorstore, chunks_jsonl_path  # noqa: E402
from sop_hub.retriever import HybridRerankPipeline  # noqa: E402

from w4.agent import build_agent_with_history  # noqa: E402
from w4.tools import build_sop_tools  # noqa: E402

from sql_agent_tool import build_sop_database_query_tool  # noqa: E402
from sql_history import (  # noqa: E402
    clear_session as _clear_session,
)
from sql_history import (
    get_session_history as _get_session_history,
)
from sql_history import (
    read_messages as _read_messages,
)


@dataclass
class AgentRuntime:
    settings: Settings
    agent: object  # RunnableWithMessageHistory

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        return _get_session_history(session_id)

    def read_messages(self, session_id: str):
        return _read_messages(session_id)

    def clear_session(self, session_id: str) -> None:
        _clear_session(session_id)


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

    # 复用 w4 的 sop_document_search；丢弃 w4 的 sop_data_analytics（Pandas）
    legacy_tools = build_sop_tools(pipeline, settings)
    sop_document_search = next(
        (t for t in legacy_tools if t.name == "sop_document_search"), None
    )
    if sop_document_search is None:
        raise RuntimeError("w4.tools 未提供 sop_document_search，请检查依赖版本")

    sop_database_query_tool = build_sop_database_query_tool(settings)
    tools = [sop_document_search, sop_database_query_tool]

    agent = build_agent_with_history(
        settings,
        tools,
        _get_session_history,
        llm_streaming=llm_streaming,
    )
    return AgentRuntime(settings=settings, agent=agent)


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
    global _runtime
    _runtime = None
