"""S&OP 智能决策中枢 v3.0 — CLI 入口。"""

from __future__ import annotations

import argparse

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import load_settings
from .indexer import build_or_load_vectorstore, chunks_jsonl_path
from .retriever import HybridRerankPipeline
from .router import build_router, route_question
from .table_agent import run_table_agent
from . import document_io as dio

RAG_PROMPT = ChatPromptTemplate.from_template(
    """你是一个助手。请仅根据提供的上下文回答问题，如果上下文中找不到答案，请直接回答“知识库中没有相关信息”，绝对不要瞎编。

上下文：
{context}

用户问题：
{question}

请作答："""
)


def _stream_answer(llm: ChatOpenAI, context: str, question: str) -> None:
    chain = RAG_PROMPT | llm | StrOutputParser()
    for chunk in chain.stream({"context": context, "question": question}):
        print(chunk, end="", flush=True)
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S&OP 智能决策中枢 v3.0")
    p.add_argument(
        "--dir",
        default=None,
        help="知识库目录（默认 SOP_DOCS_DIR 或 sop_knowledge）",
    )
    p.add_argument("-r", "--recursive", action="store_true")
    p.add_argument("--rebuild", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
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
    router = build_router(settings)
    llm_stream = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0,
        streaming=True,
    )

    print("=" * 56)
    print("🤖 S&OP 智能决策中枢 v3.0  （quit / exit 退出）")
    print("=" * 56)

    while True:
        q = input("\n🧑‍💻 请输入问题: ").strip()
        if q.lower() in ("quit", "exit"):
            print("👋 再见。")
            break
        if not q:
            continue

        decision = route_question(router, q)
        if decision.route == "analytics":
            print("\n【命中路由】数据分析（Pandas Agent，不经过向量检索）\n")
            try:
                ans = run_table_agent(q, settings)
            except Exception as e:
                ans = f"表格分析失败：{e}"
            print("\n💡 回答：")
            print(ans)
        else:
            print("\n【命中路由】文档检索（BM25 + FAISS 混合 → Rerank）\n")
            outcome = pipeline.retrieve(q)
            print(f"   （粗排候选 {outcome.rough_count} 条，Rerank={'是' if outcome.used_rerank else '否'}）")
            print("   —— Rerank 最终得分（Top 送入上下文）——")
            for row in outcome.rerank_scores:
                sc = row.get("score")
                scs = f"{sc:.6f}" if isinstance(sc, (int, float)) else "N/A"
                note = row.get("note") or ""
                print(
                    f"   [#{row.get('rank')}] relevance_score={scs} | {row.get('provenance')} {note}"
                )
            ctx = dio.format_docs_for_rag(outcome.documents)
            print("\n💡 回答：")
            _stream_answer(llm_stream, ctx, q)
            print("\n" + "-" * 40)
            print("🔍 参考片段（与上文得分对应）")
            for i, d in enumerate(outcome.documents, 1):
                prev = d.page_content.replace("\n", " ")[:100]
                print(f"  [{i}] {dio.format_provenance_line(d)}")
                print(f"      {prev}…")


if __name__ == "__main__":
    main()
