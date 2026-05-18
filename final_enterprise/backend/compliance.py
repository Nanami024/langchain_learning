from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import DATA_DIR, RUNTIME_DIR, Settings
from corpus_lexicon import CorpusLexicon, get_lexicon
from retrieval_core import HybridResult, SimpleHashEmbeddings, hybrid_retrieve, tokenize


def _doc_tier_definition_boost(page: str) -> int:
    """检索结果重排：含考核评级 A/B/C 定义的片段优先于会议流程等边角料。"""
    t = page or ""
    score = 0
    for needle in (
        "4.3",
        "考核评级标准",
        "预测准确率 ≥",
        "预测准确率 <",
        "75% ≤ 预测准确率",
        "优秀 (A",
        "合格 (B",
        "不合格 (C",
    ):
        if needle in t:
            score += 6
    for needle in ("A 级", "B 级", "C 级", "红黑榜"):
        if needle in t:
            score += 2
    return score


class ComplianceRetriever:
    def __init__(self) -> None:
        self.settings = Settings()
        self.embeddings = SimpleHashEmbeddings()
        self.index_dir = Path(RUNTIME_DIR) / "policy_faiss"
        self.manifest_path = Path(RUNTIME_DIR) / "policy_index_manifest.json"
        self.docs = self._load_source_docs()
        self._lexicon: CorpusLexicon | None = None
        self.classifier_llm = self._build_classifier_llm()
        fingerprint = self._fingerprint_docs(self.docs)

        if self._can_reuse_index(fingerprint):
            self.vector = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            self.vector = FAISS.from_documents(self.docs, self.embeddings)
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.vector.save_local(str(self.index_dir))
            self.manifest_path.write_text(
                json.dumps(
                    {
                        "policy_fingerprint": fingerprint,
                        "chunks": len(self.docs),
                        "files": sorted({str((d.metadata or {}).get("source_file", "")) for d in self.docs}),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        self.bm25 = BM25Retriever.from_documents(self.docs)
        self.bm25.k = self.settings.policy_bm25_k

    def _build_classifier_llm(self) -> ChatOpenAI | None:
        if not self.settings.llm_api_key or not self.settings.llm_base_url:
            return None
        try:
            return ChatOpenAI(
                model=self.settings.llm_model,
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url,
                temperature=0,
                request_timeout=20,
            )
        except Exception:
            return None

    def _load_source_docs(self) -> list[Document]:
        data_dir = Path(DATA_DIR)
        data_dir.mkdir(parents=True, exist_ok=True)
        pdf_files = sorted(data_dir.glob("*.pdf"))
        docs: list[Document] = []
        if pdf_files:
            for fp in pdf_files:
                docs.extend(self._load_pdf_docs(fp))
        else:
            # 若暂未准备 PDF，则使用 markdown 作为制度语料回退。
            md = data_dir / "policies.md"
            if md.exists():
                content = md.read_text(encoding="utf-8")
                docs.extend(self._split_into_docs(content, source_file=md.name, source_type="md"))
        if not docs:
            raise RuntimeError(f"未在 {data_dir} 找到制度文档（支持 PDF，兼容 policies.md）。")
        return docs

    def _load_pdf_docs(self, path: Path) -> list[Document]:
        reader = PdfReader(str(path))
        pages = []
        for idx, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((idx + 1, text))
        docs: list[Document] = []
        for page_no, text in pages:
            docs.extend(self._split_into_docs(text, source_file=path.name, source_type="pdf", page=page_no))
        return docs

    def _split_into_docs(self, text: str, *, source_file: str, source_type: str, page: int | None = None) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        out: list[Document] = []
        for chunk in splitter.split_text(text):
            meta: dict[str, Any] = {"source_file": source_file, "source_type": source_type}
            if page is not None:
                meta["page"] = str(page)
            out.append(Document(page_content=chunk, metadata=meta))
        return out

    def _fingerprint_docs(self, docs: list[Document]) -> str:
        raw = "\n".join(
            f"{(d.metadata or {}).get('source_file','')}|{(d.metadata or {}).get('page','')}|{d.page_content}"
            for d in docs
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _can_reuse_index(self, fingerprint: str) -> bool:
        if not self.manifest_path.exists() or not self.index_dir.exists():
            return False
        if not (self.index_dir / "index.faiss").exists() or not (self.index_dir / "index.pkl").exists():
            return False
        try:
            meta = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return str(meta.get("policy_fingerprint", "")) == fingerprint
        except Exception:
            return False

    def lexicon(self, analyst: Any) -> CorpusLexicon:
        return get_lexicon(analyst, self)

    def _fuse(self, query: str) -> HybridResult:
        return hybrid_retrieve(
            query=query,
            vector=self.vector,
            bm25=self.bm25,
            faiss_k=self.settings.policy_faiss_k,
            rough_k=self.settings.policy_rough_k,
            final_k=self.settings.policy_final_k,
            faiss_weight=self.settings.policy_faiss_weight,
            bm25_weight=self.settings.policy_bm25_weight,
        )

    def hybrid_relevance(self, query: str, *, final_k: int = 2) -> tuple[float, float, float, list[str]]:
        pack, blended = self._search_once(query, final_k=final_k)
        snippets = [s[:120] for s in pack.get("snippets", [])[:final_k]]
        return blended, blended, blended, snippets

    def query_policy_relevance(self, query: str) -> float:
        blended, _, _, _ = self.hybrid_relevance(query, final_k=2)
        return blended

    def should_require_review(
        self, query: str, content: str, snippets: list[str], route_plan: str
    ) -> tuple[bool, str]:
        if route_plan in ("data_only", "out_of_scope"):
            return False, ""
        q = (query or "").strip()
        has_action = bool(re.search(r"(建议|执行|落地|推进|调整|削减|压降|停采|停产|下架|审批|方案)", q))
        if route_plan == "data_then_compliance" and not has_action:
            return False, ""
        if route_plan == "compliance_only":
            joined = f"{query}\n{content}\n" + "\n".join(snippets)
            risk = hybrid_retrieve(
                query=joined,
                vector=self.vector,
                bm25=self.bm25,
                faiss_k=self.settings.policy_faiss_k,
                rough_k=self.settings.policy_rough_k,
                final_k=2,
                faiss_weight=self.settings.policy_faiss_weight,
                bm25_weight=self.settings.policy_bm25_weight,
            )
            if risk.score >= self.settings.risk_min_score * 1.1:
                return True, "制度检索命中高风险管控语义，需人工确认。"
            return False, ""
        return self.is_sensitive_action(query, content, snippets)

    def is_sensitive_action(self, query: str, draft: str, snippets: list[str]) -> tuple[bool, str]:
        joined = f"{query}\n{draft}\n" + "\n".join(snippets[:3])
        risk = hybrid_retrieve(
            query=joined,
            vector=self.vector,
            bm25=self.bm25,
            faiss_k=self.settings.policy_faiss_k,
            rough_k=self.settings.policy_rough_k,
            final_k=2,
            faiss_weight=self.settings.policy_faiss_weight,
            bm25_weight=self.settings.policy_bm25_weight,
        )
        pct = re.search(r"(\d{1,2})\s*%", joined)
        pct_val = int(pct.group(1)) if pct else 0
        if risk.score >= self.settings.risk_min_score or pct_val >= 15:
            reason = "检测到语料中的高风险动作语义或大幅比例调整，需人工审批。"
            return True, reason

        if self.settings.approval_use_llm_classifier and self.classifier_llm is not None:
            prompt = (
                "你是企业风控审查员。请根据问题、草案与制度片段判断是否需要人工审批。"
                "只输出 JSON: {\"sensitive\": true/false, \"reason\": \"...\"}。"
            )
            try:
                msg = self.classifier_llm.invoke([SystemMessage(content=prompt), HumanMessage(content=joined[:2500])])
                obj = json.loads(str(msg.content or "{}"))
                if bool(obj.get("sensitive", False)):
                    return True, str(obj.get("reason", "涉及高风险动作"))
            except Exception:
                pass
        return False, ""

    def _corpus_expand_candidates(
        self,
        query: str,
        lexicon: CorpusLexicon,
        *,
        first_score: float,
        first_pack: dict[str, Any] | None = None,
    ) -> list[str]:
        q = (query or "").strip()
        out = [q]
        if first_score >= self.settings.policy_retrieve_min_score * 1.15:
            return out
        pack = first_pack or {}
        if first_score < self.settings.policy_retrieve_min_score * 1.5 and pack.get("snippets"):
            from collections import Counter

            ctr: Counter[str] = Counter()
            for snip in pack["snippets"][:2]:
                for tok in tokenize(snip):
                    if tok in lexicon.policy_terms:
                        ctr[tok] += 1
            extra = [t for t, _ in ctr.most_common(8) if t not in set(tokenize(q))]
            if extra:
                out.append(f"{q} {' '.join(extra[:6])}")
        q_tokens = set(tokenize(q))
        hint = [t for t in lexicon.policy_top if t in q_tokens]
        if hint:
            out.append(f"{q} {' '.join(sorted(hint)[:6])}")
        return list(dict.fromkeys(out))

    def _search_once(self, query: str, *, final_k: int | None = None) -> tuple[dict[str, Any], float]:
        k = final_k or self.settings.policy_final_k
        fused = hybrid_retrieve(
            query=query,
            vector=self.vector,
            bm25=self.bm25,
            faiss_k=self.settings.policy_faiss_k,
            rough_k=self.settings.policy_rough_k,
            final_k=k,
            faiss_weight=self.settings.policy_faiss_weight,
            bm25_weight=self.settings.policy_bm25_weight,
        )
        blended = min(1.0, 0.6 * float(fused.score) + 0.4 * float(fused.token_coverage))
        docs = list(fused.docs)
        if docs and re.search(r"(等级|评级|分档|考核.*级|A级|B级|C级|优秀|合格|不合格)", query):
            ranked = [(-_doc_tier_definition_boost(d.page_content or ""), i, d) for i, d in enumerate(docs)]
            ranked.sort(key=lambda x: (x[0], x[1]))
            docs = [r[2] for r in ranked]
        if not docs:
            return {"snippets": [], "merged": "", "sources": []}, blended
        snippets = [d.page_content.strip() for d in docs if d.page_content.strip()]
        sources: list[dict[str, str]] = []
        for d in docs:
            src = str((d.metadata or {}).get("source_file", "unknown"))
            stype = str((d.metadata or {}).get("source_type", "text"))
            page = str((d.metadata or {}).get("page", "")).strip()
            name = f"{src}#p{page}" if page else src
            sources.append({"type": stype, "name": name})
        merged = f"已检索 {len(snippets)} 条制度片段（正文由模型归纳，不直接展示原文）。"
        return {"snippets": snippets, "merged": merged, "sources": sources}, blended

    def search(self, query: str, *, context: str = "", analyst: Any = None) -> dict[str, Any]:
        q = (query or "").strip()
        k = self.settings.policy_final_k
        min_score = self.settings.policy_retrieve_min_score

        pack, best_score = self._search_once(q, final_k=k)
        if pack.get("snippets") and best_score >= min_score * 1.15:
            return pack

        best: dict[str, Any] | None = pack if pack.get("snippets") else None
        candidates: list[str] = []
        if re.search(r"(等级|评级标准|考核评级|分档|A级|B级|C级|优秀|合格|不合格)", q):
            candidates.append(
                f"{q} 预测准确率 考核评级标准 4.3 优秀 A级 合格 B级 不合格 C级 75% 85%"
            )
        if analyst is not None and best_score < min_score * 1.15:
            for c in self._corpus_expand_candidates(
                q, self.lexicon(analyst), first_score=best_score, first_pack=pack
            ):
                if c not in candidates:
                    candidates.append(c)
        elif context.strip() and best_score < min_score * 1.15:
            if q not in candidates:
                candidates.append(q)

        if not candidates:
            candidates = [q]
        elif q not in candidates:
            candidates.append(q)

        if context.strip() and best_score < min_score * 1.15:
            candidates.append(f"{q}\n{context.strip()[:200]}")

        for cand in candidates:
            if cand == q and best is not None and best_score >= min_score:
                continue
            pack, score = self._search_once(cand, final_k=k)
            if score > best_score and pack.get("snippets"):
                best = pack
                best_score = score

        if best and best_score >= min_score:
            return best
        if best and best.get("snippets"):
            low = best.copy()
            low["merged"] = (
                "（相关度偏低，以下为最接近的制度片段，请结合业务判断。）\n\n" + str(best.get("merged", ""))
            )
            return low
        return {"snippets": [], "merged": "未检索到与问题直接相关的制度条款。", "sources": []}
