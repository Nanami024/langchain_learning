"""BM25 + FAISS 混合检索与 Rerank。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import Settings
from . import document_io as dio
from .rerank import rerank_documents

try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:  # 兼容旧版 / 不同分包布局
    try:
        from langchain.retrievers.ensemble import EnsembleRetriever
    except ImportError:  # pragma: no cover
        try:
            from langchain_community.retrievers import EnsembleRetriever
        except ImportError as e:
            raise ImportError(
                "无法导入 EnsembleRetriever。请安装：pip install langchain-classic"
            ) from e


def _dedupe_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    out: List[Document] = []
    for d in docs:
        key = (
            d.metadata.get("source"),
            d.metadata.get("page"),
            d.metadata.get("kb_page_display"),
            d.page_content[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


@dataclass
class RetrievalOutcome:
    documents: List[Document]
    rerank_scores: List[Dict[str, Any]]
    rough_count: int
    used_rerank: bool


class HybridRerankPipeline:
    def __init__(
        self,
        vectorstore: FAISS,
        bm25_corpus: List[Document],
        settings: Settings,
    ):
        self.settings = settings
        self.vectorstore = vectorstore
        k = settings.retriever_each_k
        if settings.use_mmr_faiss:
            try:
                fetch_k = max(
                    k * settings.mmr_fetch_k_mult,
                    settings.mmr_fetch_k_min,
                )
                self.faiss_r = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": k,
                        "fetch_k": int(fetch_k),
                        "lambda_mult": settings.mmr_lambda_mult,
                    },
                )
            except Exception:
                self.faiss_r = vectorstore.as_retriever(search_kwargs={"k": k})
        else:
            self.faiss_r = vectorstore.as_retriever(search_kwargs={"k": k})
        self.bm25_r = BM25Retriever.from_documents(bm25_corpus)
        self.bm25_r.k = k
        self.ensemble = EnsembleRetriever(
            retrievers=[self.faiss_r, self.bm25_r],
            weights=[settings.faiss_weight, settings.bm25_weight],
        )

    def retrieve(self, question: str) -> RetrievalOutcome:
        # Ensemble 默认各 retriever 使用各自的 k；合并后去重截断
        try:
            rough = self.ensemble.invoke(question)
        except TypeError:
            rough = self.ensemble.get_relevant_documents(question)
        if not isinstance(rough, list):
            rough = list(rough) if rough else []
        rough = _dedupe_docs(list(rough))[: self.settings.rough_candidates]
        texts = [d.page_content for d in rough]
        used_rerank = True
        scores_rows: List[Dict[str, Any]] = []
        if not texts:
            return RetrievalOutcome([], [], 0, False)

        try:
            ranked = rerank_documents(
                self.settings,
                query=question,
                documents=texts,
                top_n=self.settings.rerank_top_n,
            )
        except Exception as e:
            print(f"⚠️ Rerank 不可用（{e}），按混合检索顺序截取 Top-{self.settings.final_context_k}。")
            used_rerank = False
            top_docs = rough[: self.settings.final_context_k]
            for i, d in enumerate(top_docs):
                scores_rows.append(
                    {
                        "rank": i + 1,
                        "score": None,
                        "provenance": dio.format_provenance_line(d),
                        "note": "未经过 Rerank（降级）",
                    }
                )
            return RetrievalOutcome(top_docs, scores_rows, len(rough), False)

        top_docs: List[Document] = []
        for rank, it in enumerate(ranked, start=1):
            if 0 <= it.index < len(rough):
                d = rough[it.index]
                top_docs.append(d)
                scores_rows.append(
                    {
                        "rank": rank,
                        "score": it.score,
                        "provenance": dio.format_provenance_line(d),
                        "index_in_candidates": it.index,
                    }
                )
        # 若 rerank 返回异常，补全
        if len(top_docs) < self.settings.final_context_k:
            for d in rough:
                if d in top_docs:
                    continue
                top_docs.append(d)
                scores_rows.append(
                    {
                        "rank": len(scores_rows) + 1,
                        "score": None,
                        "provenance": dio.format_provenance_line(d),
                        "note": "补位",
                    }
                )
                if len(top_docs) >= self.settings.final_context_k:
                    break

        return RetrievalOutcome(
            top_docs[: self.settings.final_context_k],
            scores_rows,
            len(rough),
            used_rerank,
        )
