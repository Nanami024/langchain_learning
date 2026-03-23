"""
基于私有知识库的智能 S&OP 业务助手（LangChain + FAISS + LCEL）。

启动时从指定目录批量加载 PDF/CSV，构建或加载 FAISS；终端流式问答，并打印 Top-K 参考来源。

说明：使用 os.getenv("SOP_CHAT_MODEL", "Pro/zai-org/GLM-4.7") 是为了在 .env 里切换对话模型
（不同厂商/版本/实验模型）而无需改代码；默认仍为你当前使用的 GLM 路由名。
知识库采用「单一 FAISS 索引」混合存放 PDF 与 CSV 切块，metadata 中 doc_type/source
可区分来源；若拆成两个向量库需维护两次检索与合并排序，对本助手场景收益不大。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- 知识文档目录：目录内所有 .pdf / .csv 均会入库 ---
DEFAULT_DOCS_DIR = os.getenv("SOP_DOCS_DIR", "sop_knowledge")
DB_DIR = os.getenv("SOP_VECTORSTORE_DIR", "local_faiss_index")
CHUNK_SIZE = int(os.getenv("SOP_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("SOP_CHUNK_OVERLAP", "50"))

# 与向量库同目录保存：内容指纹 + 每个源文件对应的向量 id，用于「未变更则直接加载 / 变更则增量更新」
MANIFEST_FILENAME = "ingest_manifest.json"
MANIFEST_VERSION = 2
# 指纹不一致时是否尝试增量（删旧向量 + 仅为新增/变更文件请求 Embedding）；关闭则始终全量重建
INCREMENTAL_ENABLED = os.getenv("SOP_INCREMENTAL", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

# ---------------------------------------------------------------------------
# 切块 / 溯源元数据键（集中定义，便于维护与文档对照）
# ---------------------------------------------------------------------------
# kb_source_kind: "pdf" | "csv"
# kb_page_display: PDF 页码（人类可读，从 1 开始）；由 PyPDF 的 page(0-based)+1 得到
# kb_source_basename: 文件名，溯源展示用（《文件名》第 n 页）
# kb_row_display: CSV 行号（若 Loader 提供 row 元数据则填）
META_KB_SOURCE_KIND = "kb_source_kind"
META_KB_PAGE_DISPLAY = "kb_page_display"
META_KB_SOURCE_BASENAME = "kb_source_basename"
META_KB_ROW_DISPLAY = "kb_row_display"

# --- 模型（SOP_CHAT_MODEL：仅便于环境切换模型，见模块文档说明）---
llm = ChatOpenAI(
    model=os.getenv("SOP_CHAT_MODEL", "Pro/zai-org/GLM-4.7"),
    api_key=os.getenv("api_key"),
    base_url=os.getenv("base_url"),
    temperature=0,
    streaming=True,
)

embeddings = OpenAIEmbeddings(
    model=os.getenv("SOP_EMBED_MODEL", "BAAI/bge-m3"),
    api_key=os.getenv("api_key"),
    base_url=os.getenv("base_url"),
    chunk_size=int(os.getenv("SOP_EMBED_CHUNK_SIZE", "32")),
)


def _embed_config_signature() -> Tuple[str, int, int]:
    """参与指纹的嵌入与切块配置（与建库逻辑保持一致）。"""
    return (
        os.getenv("SOP_EMBED_MODEL", "BAAI/bge-m3"),
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )


def sha256_file(path: str, chunk_bytes: int = 1024 * 1024) -> str:
    """对文件内容计算 SHA256（分块读，适合较大 PDF）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _rel_key(abs_path: str, docs_dir_abs: str) -> str:
    """用于清单中的相对路径键（随系统使用 os.path 规范）。"""
    return os.path.normpath(os.path.relpath(os.path.abspath(abs_path), docs_dir_abs))


def apply_kb_provenance_metadata(
    docs: List[Document], docs_dir_abs: str, _recursive: bool = False
) -> None:
    """
    为 Loader 产出的每条 Document 写入溯源字段；切块时 RecursiveCharacterTextSplitter 会带到子 chunk。
    PDF：文件名 + 页码（1-based）；CSV：文件名 + 可选行号。
    第三参数保留与调用方一致（目录递归扫描开关），当前仅按每条文档的 source 路径区分文件。
    """
    docs_dir_abs = os.path.normpath(os.path.abspath(docs_dir_abs))

    for d in docs:
        src = d.metadata.get("source")
        if not src:
            continue
        abs_src = os.path.abspath(str(src))
        d.metadata[META_KB_SOURCE_BASENAME] = os.path.basename(abs_src)

        lower = abs_src.lower()
        if lower.endswith(".pdf"):
            d.metadata[META_KB_SOURCE_KIND] = "pdf"
            p = d.metadata.get("page")
            if p is not None:
                try:
                    d.metadata[META_KB_PAGE_DISPLAY] = int(p) + 1
                except (TypeError, ValueError):
                    d.metadata[META_KB_PAGE_DISPLAY] = None
        elif lower.endswith(".csv"):
            d.metadata[META_KB_SOURCE_KIND] = "csv"
            row = d.metadata.get("row")
            if row is not None:
                try:
                    # LangChain CSVLoader 通常 row 为从 0 起的数据行下标
                    d.metadata[META_KB_ROW_DISPLAY] = int(row) + 1
                except (TypeError, ValueError):
                    d.metadata[META_KB_ROW_DISPLAY] = row


def format_provenance_line(doc: Document) -> str:
    """
    将单条检索切块格式化为可读的溯源一行：PDF 为《文件名》第 n 页；CSV 为《文件名》第 n 行。
    缺省字段时回退到路径与原始 page 元数据，兼容旧索引。
    """
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

    if kind == "csv" or str(src).lower().endswith(".csv"):
        row = doc.metadata.get(META_KB_ROW_DISPLAY)
        if row is None and doc.metadata.get("row") is not None:
            try:
                row = int(doc.metadata["row"]) + 1
            except (TypeError, ValueError):
                row = doc.metadata.get("row")
        if row is not None:
            return f"《{base}》第 {row} 行"
        return f"《{base}》"

    return str(src) if src else "未知来源"


def scan_files_sha256(docs_dir_abs: str, recursive: bool) -> Dict[str, str]:
    """返回 {相对路径: 文件内容 sha256}，路径排序保证指纹稳定。"""
    out: Dict[str, str] = {}
    for abs_path, _kind in _iter_knowledge_files(docs_dir_abs, recursive=recursive):
        rk = _rel_key(abs_path, docs_dir_abs)
        out[rk] = sha256_file(abs_path)
    return dict(sorted(out.items()))


def compute_content_fingerprint(
    docs_dir_abs: str,
    recursive: bool,
    files_sha: Dict[str, str],
) -> str:
    """
    内容指纹：嵌入模型名、切块参数、目录绝对路径、是否递归、各文件内容哈希。
    任一文件改动或配置变动都会导致指纹变化。
    """
    embed_model, cs, ov = _embed_config_signature()
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "embed_model": embed_model,
        "chunk_size": cs,
        "chunk_overlap": ov,
        "recursive": recursive,
        "docs_dir": os.path.normcase(os.path.abspath(docs_dir_abs)),
        "files": list(files_sha.items()),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def manifest_path() -> str:
    return os.path.join(DB_DIR, MANIFEST_FILENAME)


def load_ingest_manifest() -> Optional[dict]:
    p = manifest_path()
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_ingest_manifest(data: dict) -> None:
    os.makedirs(DB_DIR, exist_ok=True)
    path = manifest_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def group_vector_ids_by_file(
    vectorstore: FAISS, docs_dir_abs: str
) -> Dict[str, List[str]]:
    """
    从已构建的 FAISS 中按 metadata['source'] 聚合每个源文件对应的向量 id。
    用于全量建库后写入清单。
    """
    by_file: Dict[str, List[str]] = defaultdict(list)
    mapping = getattr(vectorstore, "index_to_docstore_id", None)
    docstore = getattr(vectorstore, "docstore", None)
    if not mapping or not docstore:
        return {}
    inner = getattr(docstore, "_dict", None)
    if inner is None:
        inner = getattr(docstore, "dict", None)

    def _get_doc(doc_id: str) -> Optional[Document]:
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
        doc = _get_doc(doc_id)
        if doc is None:
            continue
        src = doc.metadata.get("source")
        if not src:
            continue
        rk = _rel_key(str(src), docs_dir_abs)
        by_file[rk].append(str(doc_id))
    return dict(by_file)


def _vectorstore_delete_ids(vectorstore: FAISS, ids: List[str]) -> None:
    if not ids:
        return
    delete = getattr(vectorstore, "delete", None)
    if delete is None:
        raise RuntimeError("当前 FAISS 实现不支持 delete，请全量重建或升级 langchain-community")
    delete(ids)


def load_documents_for_rel_keys(
    docs_dir_abs: str,
    rel_keys: Iterable[str],
    recursive: bool,
) -> List[Document]:
    """仅加载给定相对路径对应的文件（用于增量向量化）。"""
    want = set(rel_keys)
    docs: List[Document] = []
    for abs_path, kind in _iter_knowledge_files(docs_dir_abs, recursive=recursive):
        rk = _rel_key(abs_path, docs_dir_abs)
        if rk not in want:
            continue
        if kind == "pdf":
            docs.extend(load_pdf_documents(abs_path))
        else:
            docs.extend(load_csv_documents(abs_path))
    apply_kb_provenance_metadata(docs, docs_dir_abs, recursive)
    return docs


def load_pdf_documents(pdf_path: str) -> List[Document]:
    """使用 PyPDFLoader 将 PDF 读入为 LangChain Document 列表。"""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    for d in docs:
        d.metadata.setdefault("source", pdf_path)
        d.metadata.setdefault("doc_type", "pdf")
    return docs


def load_csv_documents(csv_path: str) -> List[Document]:
    """加载 CSV，每行一条 Document；依次尝试常见中文/Excel 导出编码。"""
    last_err: Optional[Exception] = None
    docs: List[Document] = []
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            loader = CSVLoader(csv_path, encoding=enc)
            docs = loader.load()
            break
        except UnicodeDecodeError as e:
            last_err = e
            continue
    else:
        if last_err:
            raise last_err
        raise OSError(f"无法读取 CSV：{csv_path}")
    for d in docs:
        d.metadata.setdefault("source", csv_path)
        d.metadata.setdefault("doc_type", "csv")
    return docs


def _iter_knowledge_files(
    docs_dir: str, recursive: bool = False
) -> Iterable[Tuple[str, str]]:
    """遍历目录下待加载文件，返回 (绝对路径, 'pdf'|'csv')，路径排序保证可复现。"""
    if not os.path.isdir(docs_dir):
        return
    names = []
    if recursive:
        for root, _dirs, files in os.walk(docs_dir):
            for f in files:
                names.append(os.path.join(root, f))
        names.sort()
    else:
        names = sorted(
            os.path.join(docs_dir, f)
            for f in os.listdir(docs_dir)
            if os.path.isfile(os.path.join(docs_dir, f))
        )
    for path in names:
        lower = path.lower()
        if lower.endswith(".pdf"):
            yield (os.path.abspath(path), "pdf")
        elif lower.endswith(".csv"):
            yield (os.path.abspath(path), "csv")


def load_documents_from_dir(docs_dir: str, recursive: bool = False) -> List[Document]:
    """
    读取目录内全部 PDF 与 CSV（可选递归子目录），合并为一份 Document 列表。
    统一写入同一向量库即可同时检索叙述型手册与表格型数据；无需按类型拆两个索引。
    """
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(
            f"知识库目录不存在：{docs_dir!r}。请创建目录并放入 PDF/CSV，或设置 SOP_DOCS_DIR。"
        )
    docs_dir_abs = os.path.normpath(os.path.abspath(docs_dir))
    docs: List[Document] = []
    seen_pdf = 0
    seen_csv = 0
    for path, kind in _iter_knowledge_files(docs_dir, recursive=recursive):
        if kind == "pdf":
            docs.extend(load_pdf_documents(path))
            seen_pdf += 1
            print(f"  📄 已加载 PDF：{path}")
        else:
            docs.extend(load_csv_documents(path))
            seen_csv += 1
            print(f"  📊 已加载 CSV：{path}")
    if not docs:
        raise FileNotFoundError(
            f"目录 {docs_dir!r} 中未找到任何 .pdf 或 .csv 文件"
            + ("（含子目录）" if recursive else "")
            + "。"
        )
    apply_kb_provenance_metadata(docs, docs_dir_abs, recursive)
    print(f"✅ 共载入 {seen_pdf} 个 PDF、{seen_csv} 个 CSV，合计 {len(docs)} 条原始片段（已写入溯源元数据）。")
    return docs


def split_documents(
    docs: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    RecursiveCharacterTextSplitter：默认约 500 字一块、50 字重叠（见 CHUNK_SIZE / CHUNK_OVERLAP）。
    重叠可减少切块边界处语义被切断，提高检索召回稳定性。
    """
    cs = CHUNK_SIZE if chunk_size is None else chunk_size
    ov = CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cs,
        chunk_overlap=ov,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )
    return splitter.split_documents(docs)


def _manifest_config_matches(old: dict, docs_dir_abs: str, recursive: bool) -> bool:
    em, cs, ov = _embed_config_signature()
    if old.get("manifest_version") != MANIFEST_VERSION:
        return False
    if old.get("embed_model") != em or old.get("chunk_size") != cs or old.get("chunk_overlap") != ov:
        return False
    if bool(old.get("recursive")) != bool(recursive):
        return False
    old_dir = old.get("docs_dir")
    if not old_dir:
        return False
    return os.path.normcase(old_dir) == os.path.normcase(os.path.abspath(docs_dir_abs))


def _full_rebuild_and_save(
    docs_dir_abs: str,
    recursive: bool,
    files_sha: Dict[str, str],
    fingerprint: str,
) -> FAISS:
    print("⏳ 全量建库：扫描目录并读取文档…")
    docs = load_documents_from_dir(docs_dir_abs, recursive=recursive)
    print("⏳ 文本切块中…")
    splits = split_documents(docs)
    print(f"✅ 切块完成，共 {len(splits)} 块。")
    print("⏳ 调用 Embedding API 写入 FAISS…")
    vectorstore = FAISS.from_documents(splits, embeddings)
    os.makedirs(DB_DIR, exist_ok=True)
    vectorstore.save_local(DB_DIR)
    by_file = group_vector_ids_by_file(vectorstore, docs_dir_abs)
    em, cs, ov = _embed_config_signature()
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "content_fingerprint": fingerprint,
        "embed_model": em,
        "chunk_size": cs,
        "chunk_overlap": ov,
        "recursive": recursive,
        "docs_dir": os.path.normcase(os.path.abspath(docs_dir_abs)),
        "files": {
            rel: {"sha256": files_sha[rel], "vector_ids": by_file.get(rel, [])}
            for rel in files_sha
        },
    }
    save_ingest_manifest(manifest)
    print(f"✅ 全量建库完成，已写入 {DB_DIR!r} 与清单。\n")
    return vectorstore


def _incremental_update_and_save(
    old: dict,
    docs_dir_abs: str,
    recursive: bool,
    current_sha: Dict[str, str],
    fingerprint: str,
) -> FAISS:
    old_files: Dict[str, Any] = old.get("files") or {}
    old_rels = set(old_files.keys())
    new_rels = set(current_sha.keys())

    removed = old_rels - new_rels
    added = new_rels - old_rels
    changed = {
        rel
        for rel in old_rels & new_rels
        if old_files[rel].get("sha256") != current_sha[rel]
    }
    to_resync = added | changed

    print("⏳ 增量更新：")
    print(f"   删除源文件 {len(removed)} 个，新增 {len(added)} 个，内容变更 {len(changed)} 个。")
    print(f"   将重新向量化 {len(to_resync)} 个源文件对应的切块（其余文件复用已有向量）。")

    vectorstore = FAISS.load_local(
        DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    ids_to_drop: List[str] = []
    for rel in removed:
        ids_to_drop.extend(old_files[rel].get("vector_ids") or [])
    for rel in changed:
        ids_to_drop.extend(old_files[rel].get("vector_ids") or [])

    if ids_to_drop:
        print(f"⏳ 从索引中移除旧向量 {len(ids_to_drop)} 条…")
        _vectorstore_delete_ids(vectorstore, ids_to_drop)

    new_splits: List[Document] = []
    if to_resync:
        print("⏳ 加载并切块变更/新增文件…")
        new_splits = split_documents(
            load_documents_for_rel_keys(docs_dir_abs, sorted(to_resync), recursive)
        )
        print(f"✅ 待写入新向量切块数：{len(new_splits)}")
    new_ids_by_rel: Dict[str, List[str]] = defaultdict(list)
    if new_splits:
        print("⏳ 调用 Embedding API 追加向量…")
        new_ids = vectorstore.add_documents(new_splits)
        if len(new_ids) != len(new_splits):
            raise RuntimeError("add_documents 返回的 id 数量与文档块数量不一致")
        for doc, vid in zip(new_splits, new_ids):
            src = doc.metadata.get("source", "")
            rk = _rel_key(str(src), docs_dir_abs)
            new_ids_by_rel[rk].append(str(vid))

    merged_files: Dict[str, dict] = {}
    for rel, sha in current_sha.items():
        if rel in to_resync:
            merged_files[rel] = {
                "sha256": sha,
                "vector_ids": list(new_ids_by_rel.get(rel, [])),
            }
        else:
            merged_files[rel] = {
                "sha256": sha,
                "vector_ids": list((old_files.get(rel) or {}).get("vector_ids") or []),
            }

    em, cs, ov = _embed_config_signature()
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "content_fingerprint": fingerprint,
        "embed_model": em,
        "chunk_size": cs,
        "chunk_overlap": ov,
        "recursive": recursive,
        "docs_dir": os.path.normcase(os.path.abspath(docs_dir_abs)),
        "files": merged_files,
    }
    save_ingest_manifest(manifest)
    vectorstore.save_local(DB_DIR)
    print(f"✅ 增量更新完成，已保存 {DB_DIR!r} 与清单。\n")
    return vectorstore


def build_vector_database(
    docs_dir: str = DEFAULT_DOCS_DIR,
    recursive: bool = False,
    force_rebuild: bool = False,
) -> Optional[FAISS]:
    """
    1) 计算各文件 SHA256 与总指纹；与 ingest_manifest.json 一致且存在索引则直接加载。
    2) 指纹不一致：默认增量更新；失败或 SOP_INCREMENTAL=0 或 --rebuild 则全量重建。
    """
    docs_dir_abs = os.path.normpath(os.path.abspath(docs_dir))
    index_faiss = os.path.join(DB_DIR, "index.faiss")

    try:
        current_sha = scan_files_sha256(docs_dir_abs, recursive)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return None

    if not current_sha:
        print(f"❌ 目录 {docs_dir_abs!r} 中没有任何可入库的 PDF/CSV。")
        return None

    fingerprint = compute_content_fingerprint(docs_dir_abs, recursive, current_sha)
    stored_manifest = load_ingest_manifest() if os.path.isfile(manifest_path()) else None
    stored_fp = (stored_manifest or {}).get("content_fingerprint")

    print("📋 内容指纹比对")
    print(f"   当前知识库哈希码：{fingerprint}")
    print(f"   本地存储哈希码：  {stored_fp or '（尚无清单或无法读取）'}")

    can_fast_load = (
        not force_rebuild
        and os.path.isfile(index_faiss)
        and stored_manifest is not None
        and stored_fp == fingerprint
        and _manifest_config_matches(stored_manifest, docs_dir_abs, recursive)
    )

    if can_fast_load:
        print("   判定：一致 → 直接加载向量库，跳过建库。")
        vs = FAISS.load_local(
            DB_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"✅ 向量库已就绪（{DB_DIR!r}）。\n")
        return vs

    print("   判定：不一致或需要重建 → 将更新向量库。")

    if force_rebuild or not INCREMENTAL_ENABLED:
        vs = _full_rebuild_and_save(
            docs_dir_abs, recursive, current_sha, fingerprint
        )
        print(f"   更新后本地存储哈希码：{fingerprint}（已与当前知识库一致）\n")
        return vs

    old = stored_manifest
    if (
        old
        and os.path.isfile(index_faiss)
        and _manifest_config_matches(old, docs_dir_abs, recursive)
        and isinstance(old.get("files"), dict)
    ):
        try:
            vs = _incremental_update_and_save(
                old, docs_dir_abs, recursive, current_sha, fingerprint
            )
            print(f"   更新后本地存储哈希码：{fingerprint}（已与当前知识库一致）\n")
            return vs
        except Exception as e:
            print(f"⚠️ 增量更新失败（{e}），改为全量重建…")

    vs = _full_rebuild_and_save(docs_dir_abs, recursive, current_sha, fingerprint)
    print(f"   更新后本地存储哈希码：{fingerprint}（已与当前知识库一致）\n")
    return vs


def format_docs(docs: List[Document]) -> str:
    """将检索到的切块拼成上下文字符串；每条前附可读溯源，便于模型对齐出处。"""
    parts: List[str] = []
    for d in docs:
        head = format_provenance_line(d)
        parts.append(f"[参考：{head}]\n{d.page_content}")
    return "\n\n".join(parts)


RAG_PROMPT = ChatPromptTemplate.from_template(
    """你是一个助手。请仅根据提供的上下文回答问题，如果上下文中找不到答案，请直接回答“知识库中没有相关信息”，绝对不要瞎编。

上下文：
{context}

用户问题：
{question}

请作答："""
)


def build_retrieval_runnable(retriever: Any):
    """
    LCEL：RunnableParallel 得到检索文档与问题；再 assign 出 context 字符串。
    生成阶段在 main 中流式调用，便于终端逐字输出。
    """
    setup = RunnableParallel(
        retrieved_docs=retriever,
        question=RunnablePassthrough(),
    )
    return setup | RunnablePassthrough.assign(
        context=lambda x: format_docs(x["retrieved_docs"]),
    )


def stream_rag_answer(state: dict) -> str:
    """对已有 context / question 流式生成回答，返回完整字符串。"""
    gen = RAG_PROMPT | llm | StrOutputParser()
    parts: List[str] = []
    for chunk in gen.stream(
        {"context": state["context"], "question": state["question"]}
    ):
        print(chunk, end="", flush=True)
        parts.append(chunk)
    return "".join(parts)


def print_sources(retrieved_docs: List[Document], top_k: int) -> None:
    """在回答末尾打印 Top-K 参考切块：溯源一行 + 路径 + 内容摘要。"""
    print("\n" + "-" * 30)
    print(f"🔍 参考来源（Top-{top_k}）")
    for i, doc in enumerate(retrieved_docs[:top_k]):
        prov = format_provenance_line(doc)
        raw_src = doc.metadata.get("source", "")
        preview = doc.page_content.replace("\n", " ")[:120]
        if len(doc.page_content) > 120:
            preview += "…"
        print(f"[{i + 1}] 溯源：{prov}")
        if raw_src:
            print(f"    文件路径：{raw_src}")
        print(f"    片段：{preview}")
    print("-" * 30)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S&OP 私有知识库问答（FAISS + LCEL）")
    p.add_argument(
        "--dir",
        dest="docs_dir",
        default=DEFAULT_DOCS_DIR,
        help=f"知识库目录，加载其中全部 .pdf 与 .csv（默认 {DEFAULT_DOCS_DIR}，可用环境变量 SOP_DOCS_DIR）",
    )
    p.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="递归扫描子目录中的 PDF/CSV",
    )
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="忽略清单指纹与增量逻辑，强制全量重新切块并向量化（环境变量 SOP_INCREMENTAL=0 可关闭增量）",
    )
    p.add_argument(
        "--k",
        type=int,
        default=int(os.getenv("SOP_RETRIEVER_K", "8")),
        help="送入模型的检索块数量（默认 8，可用环境变量 SOP_RETRIEVER_K；复杂问题建议 ≥6）",
    )
    p.add_argument(
        "--similarity",
        action="store_true",
        help="仅用相似度 Top-K，关闭 MMR（默认开启 MMR 以减轻「块太像但不互补」导致的漏答）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    vs = build_vector_database(
        docs_dir=args.docs_dir,
        recursive=args.recursive,
        force_rebuild=args.rebuild,
    )
    if vs is None:
        return

    use_mmr = os.getenv("SOP_USE_MMR", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    retriever: Any
    if use_mmr and not args.similarity:
        _fk_env = os.getenv("SOP_RETRIEVER_FETCH_K")
        if _fk_env is not None and str(_fk_env).strip() != "":
            try:
                fetch_k = max(int(_fk_env), args.k)
            except ValueError:
                fetch_k = max(args.k * 5, 30)
        else:
            fetch_k = max(args.k * 5, 30)
        lam = float(os.getenv("SOP_MMR_LAMBDA", "0.55"))
        try:
            retriever = vs.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": args.k,
                    "fetch_k": fetch_k,
                    "lambda_mult": lam,
                },
            )
        except Exception as e:
            print(f"⚠️ MMR 检索不可用（{e}），已退回普通相似度 Top-K。")
            retriever = vs.as_retriever(search_kwargs={"k": args.k})
    else:
        retriever = vs.as_retriever(search_kwargs={"k": args.k})
    retrieval = build_retrieval_runnable(retriever)

    print("=" * 50)
    print("🤖 基于私有知识库的智能 S&OP 业务助手（流式输出；输入 quit / exit 退出）")
    print("=" * 50)

    while True:
        query = input("\n🧑‍💻 请输入问题: ").strip()
        if query.lower() in ("quit", "exit"):
            print("👋 再见。")
            break
        if not query:
            continue

        print("\n⏳ 检索并生成中…\n")
        out = retrieval.invoke(query)
        print("💡 回答：")
        stream_rag_answer(out)
        print()
        print_sources(out["retrieved_docs"], args.k)


if __name__ == "__main__":
    main()
