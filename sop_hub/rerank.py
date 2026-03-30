"""SiliconFlow / OpenAPI 兼容 Rerank（POST /v1/rerank）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import httpx

from .config import Settings


@dataclass
class RerankItem:
    index: int
    score: float
    text: str


def rerank_documents(
    settings: Settings,
    query: str,
    documents: List[str],
    top_n: int,
) -> List[RerankItem]:
    if not documents:
        return []
    url = f"{settings.base_url}/rerank"
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": settings.rerank_model,
        "query": query,
        "documents": documents,
        "top_n": min(top_n, len(documents)),
        "return_documents": True,
    }
    with httpx.Client(timeout=120.0) as client:
        r = client.post(url, json=body, headers=headers)
        r.raise_for_status()
        data = r.json()
    results = data.get("results") or []
    out: List[RerankItem] = []
    for row in results:
        idx = int(row.get("index", -1))
        sc = float(row.get("relevance_score", 0.0))
        text = ""
        doc = row.get("document")
        if isinstance(doc, dict):
            text = doc.get("text") or ""
        out.append(RerankItem(index=idx, score=sc, text=text))
    return out
