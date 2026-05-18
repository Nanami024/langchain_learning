from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

_TOKEN_RE = re.compile(r"[A-Za-z][\u4e00-\u9fff]+|[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text or ""):
        tok = m.group(0).lower()
        if re.fullmatch(r"[\u4e00-\u9fff]+", tok):
            if len(tok) <= 2:
                out.append(tok)
            else:
                out.append(tok)
                out.extend(tok[i : i + 2] for i in range(0, len(tok) - 1))
        else:
            out.append(tok)
    return out


class SimpleHashEmbeddings(Embeddings):
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        if not tokens:
            return vec.tolist()
        for tok in tokens:
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:2], "big") % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@dataclass
class HybridResult:
    docs: list[Document]
    score: float
    rough_count: int
    token_coverage: float


def build_hybrid_indexes(
    docs: list[Document],
    embeddings: Embeddings,
    bm25_k: int,
) -> tuple[FAISS, BM25Retriever]:
    vector = FAISS.from_documents(docs, embeddings)
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = bm25_k
    return vector, bm25


def hybrid_retrieve(
    *,
    query: str,
    vector: FAISS,
    bm25: BM25Retriever,
    faiss_k: int,
    rough_k: int,
    final_k: int,
    faiss_weight: float,
    bm25_weight: float,
) -> HybridResult:
    faiss_hits = vector.similarity_search_with_score(query, k=faiss_k)
    bm25_docs = bm25.invoke(query)
    scores: dict[str, float] = {}
    order: dict[str, Document] = {}
    overlaps: dict[str, float] = {}
    q_tokens = set(tokenize(query))
    q_len = float(max(1, len(q_tokens)))

    def _key(d: Document) -> str:
        m = d.metadata or {}
        return f"{m.get('source_file','')}|{m.get('page','')}|{m.get('source_id','')}|{d.page_content[:120]}"

    for rank, (d, dist) in enumerate(faiss_hits, start=1):
        k = _key(d)
        order[k] = d
        d_tokens = set(tokenize(d.page_content))
        overlap = len(q_tokens.intersection(d_tokens)) / q_len
        overlaps[k] = max(overlaps.get(k, 0.0), overlap)
        sim = 1.0 / (1.0 + max(0.0, float(dist)))
        scores[k] = scores.get(k, 0.0) + (faiss_weight * sim * overlap / float(rank))

    for rank, d in enumerate(bm25_docs, start=1):
        k = _key(d)
        order[k] = d
        d_tokens = set(tokenize(d.page_content))
        overlap = len(q_tokens.intersection(d_tokens)) / q_len
        overlaps[k] = max(overlaps.get(k, 0.0), overlap)
        scores[k] = scores.get(k, 0.0) + (bm25_weight * overlap / float(rank))

    ranked = sorted(scores.items(), key=lambda x: (x[1], overlaps.get(x[0], 0.0)), reverse=True)[:rough_k]
    rough_docs = [order[k] for k, _ in ranked]
    top_score = float(ranked[0][1]) if ranked else 0.0
    token_coverage = float(overlaps.get(ranked[0][0], 0.0)) if ranked else 0.0

    rough_docs.sort(
        key=lambda d: (
            len(q_tokens.intersection(set(tokenize(d.page_content)))),
            len(d.page_content),
        ),
        reverse=True,
    )
    return HybridResult(
        docs=rough_docs[:final_k],
        score=top_score,
        rough_count=len(rough_docs),
        token_coverage=token_coverage,
    )
