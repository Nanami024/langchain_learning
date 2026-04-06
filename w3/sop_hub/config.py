"""环境变量与运行时参数集中管理。

默认知识库目录与向量库目录相对于本包所在「w3」根目录，便于整套 w3 拷贝到任意路径独立运行。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# w3/sop_hub/config.py -> w3 根目录
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_W3_ROOT = os.path.abspath(os.path.join(_CONFIG_DIR, ".."))

load_dotenv(os.path.join(_W3_ROOT, ".env"))
load_dotenv()


def _b(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off")


def _default_docs_dir() -> str:
    return os.path.join(_W3_ROOT, "sop_knowledge")


def _default_vectorstore_dir() -> str:
    return os.path.join(_W3_ROOT, "local_faiss_index")


@dataclass
class Settings:
    docs_dir: str
    vectorstore_dir: str
    recursive: bool
    chunk_size: int
    chunk_overlap: int
    incremental: bool
    api_key: str
    base_url: str
    chat_model: str
    embed_model: str
    embed_chunk_size: int
    rerank_model: str
    faiss_weight: float
    bm25_weight: float
    retriever_each_k: int
    rough_candidates: int
    rerank_top_n: int
    final_context_k: int
    use_mmr_faiss: bool
    mmr_lambda_mult: float
    mmr_fetch_k_mult: int
    mmr_fetch_k_min: int


def load_settings(
    docs_dir: str | None = None,
    recursive: bool = False,
) -> Settings:
    raw_docs = docs_dir or os.getenv("SOP_DOCS_DIR") or _default_docs_dir()
    raw_vs = os.getenv("SOP_VECTORSTORE_DIR") or _default_vectorstore_dir()
    return Settings(
        docs_dir=os.path.normpath(os.path.abspath(raw_docs)),
        vectorstore_dir=os.path.normpath(os.path.abspath(raw_vs)),
        recursive=recursive,
        chunk_size=int(os.getenv("SOP_CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("SOP_CHUNK_OVERLAP", "50")),
        incremental=_b("SOP_INCREMENTAL", "1"),
        api_key=os.getenv("api_key", ""),
        base_url=os.getenv("base_url", "").rstrip("/"),
        chat_model=os.getenv("SOP_CHAT_MODEL", "Pro/zai-org/GLM-4.7"),
        embed_model=os.getenv("SOP_EMBED_MODEL", "BAAI/bge-m3"),
        embed_chunk_size=int(os.getenv("SOP_EMBED_CHUNK_SIZE", "32")),
        rerank_model=os.getenv(
            "SOP_RERANK_MODEL",
            "BAAI/bge-reranker-v2-m3",
        ),
        faiss_weight=float(os.getenv("SOP_FAISS_WEIGHT", "0.55")),
        bm25_weight=float(os.getenv("SOP_BM25_WEIGHT", "0.45")),
        retriever_each_k=int(os.getenv("SOP_RETRIEVER_EACH_K", "18")),
        rough_candidates=int(os.getenv("SOP_ROUGH_CANDIDATES", "36")),
        rerank_top_n=int(os.getenv("SOP_RERANK_TOP_N", "6")),
        final_context_k=int(os.getenv("SOP_FINAL_CONTEXT_K", "5")),
        use_mmr_faiss=_b("SOP_USE_MMR", "1"),
        mmr_lambda_mult=float(os.getenv("SOP_MMR_LAMBDA", "0.5")),
        mmr_fetch_k_mult=int(os.getenv("SOP_MMR_FETCH_MULT", "6")),
        mmr_fetch_k_min=int(os.getenv("SOP_MMR_FETCH_MIN", "48")),
    )
