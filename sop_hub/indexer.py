"""PDF 专用 FAISS 索引：清单、增量、指纹（不含 CSV）。

文档同步（Sync / Delete）原理
---------------------------
1. **账本**：`ingest_manifest.json` 中 `files` 以「相对知识库根目录的路径」为键（与 `document_io.rel_key` 一致），
   每个 PDF 对应 `{ sha256, vector_ids }`。`vector_ids` 是该文件所有切块在 FAISS docstore 中的 id 列表，
   全量/增量写盘后由 `group_vector_ids_by_file` 根据 `metadata["source"]` 聚类得到。

2. **每次启动**：对磁盘重新 `scan_pdf_sha256`，得到 `current_sha`（路径 → 当前文件内容哈希）。

3. **.diff**（在 `_incremental` 内）：
   - **removed** = 键在旧清单、不在当前扫描结果 → 认为文件已**删除或重命名走旧名**；
     收集这些键下的全部 `vector_ids`，调用 `FAISS.delete(ids)` **从索引中物理删除**对应向量。
   - **changed** = 键仍在且 `sha256` 变化 → 内容被覆盖修改；**先 delete 旧 id**，再只对该路径重新切块 + `add_documents`。
   - **added** = 键仅在新扫描中 → 新文件或**重命名后的新路径**；只对这批路径 embed，无旧 id 可删。

4. **重命名**：等价于 旧路径 **removed** + 新路径 **added**，旧向量全部删除，新文件重新向量化，不会留下「死向量」。

5. **清单更新**：`merged` 仅包含**当前磁盘仍存在的** PDF 键；removed 的键不再写入清单。
   写盘后从 docstore **导出** `text_chunks.jsonl`，保证 BM25 语料与 FAISS 一致。

6. **知识库中 PDF 全部被删**：检测到无 PDF 且本地仍有索引时，**清空**索引目录内与向量相关的文件，避免检索命中幽灵数据。
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .config import Settings
from .constants import CHUNKS_JSONL, MANIFEST_FILENAME, MANIFEST_VERSION
from . import document_io as dio


def _embed_sig(settings: Settings) -> tuple:
    return (settings.embed_model, settings.chunk_size, settings.chunk_overlap)


def compute_index_fingerprint(
    docs_dir_abs: str,
    recursive: bool,
    pdf_sha: Dict[str, str],
    settings: Settings,
) -> str:
    """仅 PDF 文件 + 嵌入与切块参数。"""
    em, cs, ov = _embed_sig(settings)
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "embed_model": em,
        "chunk_size": cs,
        "chunk_overlap": ov,
        "recursive": recursive,
        "docs_dir": os.path.normcase(os.path.abspath(docs_dir_abs)),
        "pdf_files": list(pdf_sha.items()),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def manifest_path(vectorstore_dir: str) -> str:
    return os.path.join(vectorstore_dir, MANIFEST_FILENAME)


def chunks_jsonl_path(vectorstore_dir: str) -> str:
    return os.path.join(vectorstore_dir, CHUNKS_JSONL)


def purge_text_index_artifacts(vectorstore_dir: str) -> None:
    """
    删除本地文本索引相关文件（FAISS、清单、BM25 语料）。
    在知识库已无任何 PDF 时调用，使磁盘与「空文本库」一致。
    """
    if not os.path.isdir(vectorstore_dir):
        return
    for fn in os.listdir(vectorstore_dir):
        path = os.path.join(vectorstore_dir, fn)
        if not os.path.isfile(path):
            continue
        if fn in (MANIFEST_FILENAME, CHUNKS_JSONL) or fn.startswith("index."):
            try:
                os.remove(path)
            except OSError:
                pass


def load_manifest(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_manifest(path: str, data: dict) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def group_vector_ids_by_file(
    vectorstore: FAISS, docs_dir_abs: str
) -> Dict[str, List[str]]:
    by_file: Dict[str, List[str]] = defaultdict(list)
    mapping = getattr(vectorstore, "index_to_docstore_id", None)
    docstore = getattr(vectorstore, "docstore", None)
    if not mapping or not docstore:
        return {}
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

    for _idx, doc_id in mapping.items():
        doc = get_doc(doc_id)
        if not doc:
            continue
        src = doc.metadata.get("source")
        if not src:
            continue
        rk = dio.rel_key(str(src), docs_dir_abs)
        by_file[rk].append(str(doc_id))
    return dict(by_file)


def _delete_ids(vectorstore: FAISS, ids: List[str]) -> None:
    if not ids:
        return
    delete = getattr(vectorstore, "delete", None)
    if delete is None:
        raise RuntimeError("FAISS 不支持 delete，请升级 langchain-community")
    delete(ids)


def _manifest_config_matches(old: dict, docs_dir_abs: str, recursive: bool, s: Settings) -> bool:
    em, cs, ov = _embed_sig(s)
    if old.get("manifest_version") != MANIFEST_VERSION:
        return False
    if (
        old.get("embed_model") != em
        or old.get("chunk_size") != cs
        or old.get("chunk_overlap") != ov
    ):
        return False
    if bool(old.get("recursive")) != bool(recursive):
        return False
    od = old.get("docs_dir")
    if not od:
        return False
    return os.path.normcase(od) == os.path.normcase(os.path.abspath(docs_dir_abs))


def _full_rebuild(
    settings: Settings,
    docs_dir_abs: str,
    pdf_sha: Dict[str, str],
    fingerprint: str,
    embeddings: OpenAIEmbeddings,
) -> FAISS:
    print("⏳ 全量建库（仅 PDF）…")
    docs = dio.load_all_pdfs(docs_dir_abs, settings.recursive)
    splits = dio.split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    print(f"✅ 切块 {len(splits)}，正在 Embedding…")
    vs = FAISS.from_documents(splits, embeddings)
    os.makedirs(settings.vectorstore_dir, exist_ok=True)
    vs.save_local(settings.vectorstore_dir)
    by_file = group_vector_ids_by_file(vs, docs_dir_abs)
    em, cs, ov = _embed_sig(settings)
    mpath = manifest_path(settings.vectorstore_dir)
    save_manifest(
        mpath,
        {
            "manifest_version": MANIFEST_VERSION,
            "content_fingerprint": fingerprint,
            "embed_model": em,
            "chunk_size": cs,
            "chunk_overlap": ov,
            "recursive": settings.recursive,
            "docs_dir": os.path.normcase(os.path.abspath(docs_dir_abs)),
            "files": {
                rel: {"sha256": pdf_sha[rel], "vector_ids": by_file.get(rel, [])}
                for rel in pdf_sha
            },
        },
    )
    jpath = chunks_jsonl_path(settings.vectorstore_dir)
    dio.export_vectorstore_documents_to_jsonl(vs, jpath)
    print(f"✅ 已保存 FAISS、清单与 {CHUNKS_JSONL}\n")
    return vs


def _incremental(
    old: dict,
    settings: Settings,
    docs_dir_abs: str,
    current_sha: Dict[str, str],
    fingerprint: str,
    embeddings: OpenAIEmbeddings,
) -> FAISS:
    old_files: Dict[str, Any] = old.get("files") or {}
    old_rels, new_rels = set(old_files.keys()), set(current_sha.keys())
    removed, added = old_rels - new_rels, new_rels - old_rels
    changed = {
        r
        for r in old_rels & new_rels
        if old_files[r].get("sha256") != current_sha[r]
    }
    to_resync = added | changed
    print("⏳ 增量更新 PDF 索引…")
    print(
        f"   删除 {len(removed)} 新增 {len(added)} 变更 {len(changed)}；重嵌入 {len(to_resync)} 个文件。"
    )
    vs = FAISS.load_local(
        settings.vectorstore_dir, embeddings, allow_dangerous_deserialization=True
    )
    drop: List[str] = []
    for r in removed | changed:
        drop.extend(old_files.get(r, {}).get("vector_ids") or [])
    if drop:
        print(f"⏳ 删除旧向量 {len(drop)} 条…")
        _delete_ids(vs, drop)
    new_ids_by_rel: Dict[str, List[str]] = defaultdict(list)
    if to_resync:
        new_splits = dio.split_documents(
            dio.load_pdfs_for_rel_keys(docs_dir_abs, sorted(to_resync), settings.recursive),
            settings.chunk_size,
            settings.chunk_overlap,
        )
        print(f"⏳ 追加 {len(new_splits)} 块…")
        new_ids = vs.add_documents(new_splits)
        for doc, vid in zip(new_splits, new_ids):
            rk = dio.rel_key(str(doc.metadata.get("source", "")), docs_dir_abs)
            new_ids_by_rel[rk].append(str(vid))
    merged = {}
    for rel, sha in current_sha.items():
        if rel in to_resync:
            merged[rel] = {
                "sha256": sha,
                "vector_ids": list(new_ids_by_rel.get(rel, [])),
            }
        else:
            merged[rel] = {
                "sha256": sha,
                "vector_ids": list((old_files.get(rel) or {}).get("vector_ids") or []),
            }
    em, cs, ov = _embed_sig(settings)
    save_manifest(
        manifest_path(settings.vectorstore_dir),
        {
            "manifest_version": MANIFEST_VERSION,
            "content_fingerprint": fingerprint,
            "embed_model": em,
            "chunk_size": cs,
            "chunk_overlap": ov,
            "recursive": settings.recursive,
            "docs_dir": os.path.normcase(os.path.abspath(docs_dir_abs)),
            "files": merged,
        },
    )
    vs.save_local(settings.vectorstore_dir)
    dio.export_vectorstore_documents_to_jsonl(
        vs, chunks_jsonl_path(settings.vectorstore_dir)
    )
    print("✅ 增量完成，已刷新 jsonl。\n")
    return vs


def build_or_load_vectorstore(
    settings: Settings, force_rebuild: bool = False
) -> Optional[FAISS]:
    docs_dir_abs = settings.docs_dir
    idx = os.path.join(settings.vectorstore_dir, "index.faiss")
    mpath = manifest_path(settings.vectorstore_dir)

    if not os.path.isdir(docs_dir_abs):
        print(f"❌ 知识库目录不存在：{docs_dir_abs}")
        return None

    pdf_sha = dio.scan_pdf_sha256(docs_dir_abs, settings.recursive)
    if not pdf_sha:
        if os.path.isfile(idx) or os.path.isfile(mpath):
            print(
                "⚠️ 知识库中已无任何 PDF；将清空本地文本索引文件，避免检索到已删除文件的幽灵向量。"
            )
            purge_text_index_artifacts(settings.vectorstore_dir)
        print(f"❌ {docs_dir_abs} 下没有 PDF，无法建立文本索引。")
        return None

    fp = compute_index_fingerprint(
        docs_dir_abs, settings.recursive, pdf_sha, settings
    )
    old = load_manifest(mpath) if os.path.isfile(mpath) else None
    stored_fp = (old or {}).get("content_fingerprint")

    print("📋 PDF 索引指纹")
    print(f"   当前：{fp}")
    print(f"   存储：{stored_fp or '（无）'}")

    embeddings = OpenAIEmbeddings(
        model=settings.embed_model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        chunk_size=settings.embed_chunk_size,
    )

    fast = (
        not force_rebuild
        and os.path.isfile(idx)
        and old is not None
        and stored_fp == fp
        and _manifest_config_matches(old, docs_dir_abs, settings.recursive, settings)
    )
    if fast:
        print("   → 一致，直接加载 FAISS。\n")
        return FAISS.load_local(
            settings.vectorstore_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    print("   → 不一致或强制重建，更新索引…")
    if force_rebuild or not settings.incremental:
        vs = _full_rebuild(settings, docs_dir_abs, pdf_sha, fp, embeddings)
        print(f"   已写入存储指纹：{fp}\n")
        return vs
    if (
        old
        and os.path.isfile(idx)
        and _manifest_config_matches(old, docs_dir_abs, settings.recursive, settings)
        and isinstance(old.get("files"), dict)
    ):
        try:
            vs = _incremental(old, settings, docs_dir_abs, pdf_sha, fp, embeddings)
            print(f"   已写入存储指纹：{fp}\n")
            return vs
        except Exception as e:
            print(f"⚠️ 增量失败（{e}），全量重建…")
    vs = _full_rebuild(settings, docs_dir_abs, pdf_sha, fp, embeddings)
    print(f"   已写入存储指纹：{fp}\n")
    return vs
