"""PDF/CSV 枚举、加载、切块与溯源元数据（向量库仅使用 PDF）。"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sop_hub.constants import (
    META_KB_PAGE_DISPLAY,
    META_KB_ROW_DISPLAY,
    META_KB_SOURCE_BASENAME,
    META_KB_SOURCE_KIND,
)


def rel_key(abs_path: str, docs_dir_abs: str) -> str:
    return os.path.normpath(os.path.relpath(os.path.abspath(abs_path), docs_dir_abs))


def sha256_file(path: str, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def iter_data_files(
    docs_dir: str, recursive: bool
) -> Iterable[Tuple[str, str]]:
    """(abs_path, 'pdf'|'csv')，按路径排序。"""
    if not os.path.isdir(docs_dir):
        return
    names: List[str] = []
    if recursive:
        for root, _dirs, files in os.walk(docs_dir):
            for fn in files:
                names.append(os.path.join(root, fn))
        names.sort()
    else:
        names = sorted(
            os.path.join(docs_dir, f)
            for f in os.listdir(docs_dir)
            if os.path.isfile(os.path.join(docs_dir, f))
        )
    for path in names:
        low = path.lower()
        if low.endswith(".pdf"):
            yield os.path.abspath(path), "pdf"
        elif low.endswith(".csv"):
            yield os.path.abspath(path), "csv"


def list_pdf_paths(docs_dir_abs: str, recursive: bool) -> List[str]:
    return [p for p, k in iter_data_files(docs_dir_abs, recursive) if k == "pdf"]


def list_csv_paths(docs_dir_abs: str, recursive: bool) -> List[str]:
    return [p for p, k in iter_data_files(docs_dir_abs, recursive) if k == "csv"]


def scan_pdf_sha256(docs_dir_abs: str, recursive: bool) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for abs_path, kind in iter_data_files(docs_dir_abs, recursive):
        if kind != "pdf":
            continue
        rk = rel_key(abs_path, docs_dir_abs)
        out[rk] = sha256_file(abs_path)
    return dict(sorted(out.items()))


def load_pdf_documents(pdf_path: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    for d in docs:
        d.metadata.setdefault("source", pdf_path)
        d.metadata.setdefault("doc_type", "pdf")
    return docs


def apply_kb_provenance_metadata(
    docs: List[Document], docs_dir_abs: str, _recursive: bool = False
) -> None:
    docs_dir_abs = os.path.normpath(os.path.abspath(docs_dir_abs))
    for d in docs:
        src = d.metadata.get("source")
        if not src:
            continue
        abs_src = os.path.abspath(str(src))
        d.metadata[META_KB_SOURCE_BASENAME] = os.path.basename(abs_src)
        low = abs_src.lower()
        if low.endswith(".pdf"):
            d.metadata[META_KB_SOURCE_KIND] = "pdf"
            p = d.metadata.get("page")
            if p is not None:
                try:
                    d.metadata[META_KB_PAGE_DISPLAY] = int(p) + 1
                except (TypeError, ValueError):
                    d.metadata[META_KB_PAGE_DISPLAY] = None
        elif low.endswith(".csv"):
            d.metadata[META_KB_SOURCE_KIND] = "csv"
            row = d.metadata.get("row")
            if row is not None:
                try:
                    d.metadata[META_KB_ROW_DISPLAY] = int(row) + 1
                except (TypeError, ValueError):
                    d.metadata[META_KB_ROW_DISPLAY] = row


def load_all_pdfs(docs_dir: str, recursive: bool) -> List[Document]:
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"知识库目录不存在：{docs_dir!r}")
    docs_dir_abs = os.path.normpath(os.path.abspath(docs_dir))
    docs: List[Document] = []
    n = 0
    for path, kind in iter_data_files(docs_dir_abs, recursive):
        if kind != "pdf":
            continue
        docs.extend(load_pdf_documents(path))
        n += 1
        print(f"  📄 已加载 PDF：{path}")
    if not docs:
        raise FileNotFoundError(
            f"目录 {docs_dir!r} 中未找到 PDF（文本索引仅支持 PDF）。"
        )
    apply_kb_provenance_metadata(docs, docs_dir_abs, recursive)
    print(f"✅ 共载入 {n} 个 PDF，{len(docs)} 条页级片段。")
    return docs


def load_pdfs_for_rel_keys(
    docs_dir_abs: str, rel_keys: Iterable[str], recursive: bool
) -> List[Document]:
    want = set(rel_keys)
    docs: List[Document] = []
    for abs_path, kind in iter_data_files(docs_dir_abs, recursive):
        if kind != "pdf":
            continue
        rk = rel_key(abs_path, docs_dir_abs)
        if rk not in want:
            continue
        docs.extend(load_pdf_documents(abs_path))
    apply_kb_provenance_metadata(docs, docs_dir_abs, recursive)
    return docs


def split_documents(
    docs: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )
    return splitter.split_documents(docs)


def format_provenance_line(doc: Document) -> str:
    src = doc.metadata.get("source", "")
    base = doc.metadata.get(META_KB_SOURCE_BASENAME) or (
        os.path.basename(str(src)) if src else "未知文件"
    )
    kind = doc.metadata.get(META_KB_SOURCE_KIND) or doc.metadata.get("doc_type")
    if kind == "pdf" or str(src).lower().endswith(".pdf"):
        page = doc.metadata.get(META_KB_PAGE_DISPLAY)
        if page is None and doc.metadata.get("page") is not None:
            try:
                page = int(doc.metadata["page"]) + 1
            except (TypeError, ValueError):
                page = None
        if page is not None:
            return f"《{base}》第 {page} 页"
        return f"《{base}》"
    return str(src) if src else "未知来源"


def format_docs_for_rag(docs: List[Document]) -> str:
    parts: List[str] = []
    for d in docs:
        head = format_provenance_line(d)
        parts.append(f"[参考：{head}]\n{d.page_content}")
    return "\n\n".join(parts)


def export_vectorstore_documents_to_jsonl(vectorstore, path: str) -> None:
    mapping = getattr(vectorstore, "index_to_docstore_id", None)
    docstore = getattr(vectorstore, "docstore", None)
    if not mapping or not docstore:
        return
    inner = getattr(docstore, "_dict", None) or getattr(docstore, "dict", None)

    def get_doc(doc_id: str) -> Optional[Document]:
        if inner is not None and doc_id in inner:
            d = inner[doc_id]
            return d if isinstance(d, Document) else None
        search = getattr(docstore, "search", None)
        if callable(search):
            try:
                d = search(doc_id)
                return d if isinstance(d, Document) else None
            except Exception:
                return None
        return None

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _idx, doc_id in mapping.items():
            doc = get_doc(doc_id)
            if doc is None:
                continue
            rec = {"page_content": doc.page_content, "metadata": dict(doc.metadata)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_documents_from_jsonl(path: str) -> List[Document]:
    if not os.path.isfile(path):
        return []
    out: List[Document] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            out.append(
                Document(
                    page_content=o["page_content"],
                    metadata=o.get("metadata") or {},
                )
            )
    return out
