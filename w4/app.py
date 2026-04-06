"""CLI：加载索引与工具链，多轮对话（RunnableWithMessageHistory）。"""

from __future__ import annotations

import argparse
import uuid

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

from sop_hub import document_io as dio
from sop_hub.config import load_settings
from sop_hub.indexer import build_or_load_vectorstore, chunks_jsonl_path
from sop_hub.retriever import HybridRerankPipeline

from .agent import build_agent_with_history
from .callbacks import ColoredToolCallback
from .tools import build_sop_tools


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S&OP 决策中枢 v4.0（Tool Agent + Memory）")
    p.add_argument(
        "--dir",
        default=None,
        help="知识库目录（默认 SOP_DOCS_DIR 或 sop_knowledge）",
    )
    p.add_argument("-r", "--recursive", action="store_true")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument(
        "--session",
        default="default",
        help="会话 ID（多路并行对话时区分状态，默认 default）",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    settings = load_settings(docs_dir=args.dir, recursive=args.recursive)
    if not settings.api_key or not settings.base_url:
        print("请在 .env 配置 api_key 与 base_url。")
        return

    vs = build_or_load_vectorstore(settings, force_rebuild=args.rebuild)
    if vs is None:
        return

    jpath = chunks_jsonl_path(settings.vectorstore_dir)
    dio.export_vectorstore_documents_to_jsonl(vs, jpath)
    bm25_docs = dio.load_documents_from_jsonl(jpath)
    if not bm25_docs:
        print("⚠️ BM25 语料为空，请检查索引。")
        return

    pipeline = HybridRerankPipeline(vs, bm25_docs, settings)
    tools = build_sop_tools(pipeline, settings)

    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    agent = build_agent_with_history(settings, tools, get_session_history)
    session_id = args.session or str(uuid.uuid4())
    # 必须在 invoke 的 config 里传入 callbacks，RunnableWithMessageHistory 才会把工具事件传到终端
    tool_trace = ColoredToolCallback()

    print("=" * 60)
    print("🤖 S&OP 决策中枢 v4.0 — Tool Calling Agent + 记忆")
    print(f"   session_id = {session_id}")
    print("   命令：quit / exit 退出；/new 清空本会话记忆")
    print("=" * 60)

    cfg = {
        "configurable": {"session_id": session_id},
        "callbacks": [tool_trace],
    }

    while True:
        q = input("\n🧑‍💻 请输入: ").strip()
        if q.lower() in ("quit", "exit"):
            print("👋 再见。")
            break
        if not q:
            continue
        if q == "/new":
            store.pop(session_id, None)
            print("♻️ 已清空本会话记忆。")
            continue

        try:
            out = agent.invoke({"input": q}, config=cfg)
        except Exception as e:
            print(f"\n❌ 运行失败：{e}")
            continue

        text = out.get("output") if isinstance(out, dict) else str(out)
        print("\n💡 Agent：")
        print(text)


if __name__ == "__main__":
    main()
