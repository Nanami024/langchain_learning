# S&OP 智能决策中枢 v3.0 — 实现与任务对照说明

本文说明 **OKR / 日程任务** 如何在仓库中落地，便于答辩或复现。

## Day 1–2：工程重构与向量同步

### 任务 1：拆分 `w2.py`

| 原职责 | 新模块 |
|--------|--------|
| 环境、路径、权重 | `sop_hub/config.py` |
| PDF 加载、切块、溯源元数据 | `sop_hub/document_io.py` |
| FAISS 构建、清单、增量、指纹 | `sop_hub/indexer.py` |
| 混合检索 + Rerank | `sop_hub/retriever.py` |
| Rerank HTTP | `sop_hub/rerank.py` |
| CLI 交互 | `sop_hub/main.py` |

单体 `w2.py` **保留**作参考，生产入口为 **`python -m sop_hub.main`**。

### 任务 2：同步与删除（Sync/Delete）

- **账本**：`ingest_manifest.json` 的 `files`：**键 = PDF 相对路径**；值含 `sha256`、`vector_ids`。  
- **删除文件**：路径在旧清单、不在磁盘 → 收集 `vector_ids` → `FAISS.delete(ids)`。  
- **修改文件**：`sha256` 变化 → 删旧 id → 仅对该 PDF 重新切块 + `add_documents`。  
- **重命名**：旧路径 → `removed`；新路径 → `added`；向量侧表现即为删旧向量化新，与「精准剔除」一致。  
- **BM25 语料**：每次索引写盘后 **`export_chunks_jsonl`** 从 **FAISS docstore 全量导出**，保证与向量切片一致，无需单独维护两套 diff。

## Day 3–4：语义路由与表格急救

### 任务 3：LCEL 路由

- `sop_hub/router.py`：`ChatOpenAI + with_structured_output(RouterDecision)`。  
- 输入用户问题，输出 `document` / `analytics`（及可选 `reason`）。  
- `main` 在生成前 **打印命中路由**（中文标签：文档检索 / 数据分析）。

### 任务 4：Pandas Agent，禁止表格走 FAISS

- `sop_hub/table_agent.py`：`create_pandas_dataframe_agent`。  
- 仅当路由为 `analytics` 时调用；**不**加载 `vectorstore.get_relevant_documents`。  
- 数据来自知识库目录内实时枚举的 **CSV**（`pandas.read_csv`）。

## Day 5–7：混合检索与 Rerank

### 任务 5：BM25 + FAISS + Ensemble

- `langchain_community.retrievers.BM25Retriever` + `FAISS.as_retriever`。  
- `EnsembleRetriever(retrievers=[...], weights=[SOP_FAISS_WEIGHT, SOP_BM25_WEIGHT])`。  
- 合并后再 **按 `source/page/content` 前缀去重**，截断为 `SOP_ROUGH_CANDIDATES` 条（默认 20）。

### 任务 6：Rerank API

- `sop_hub/rerank.py`：`POST {base_url}/rerank`，body 与 SiliconFlow OpenAPI 一致（`model`、`query`、`documents`、`top_n`）。  
- 返回的 `results[].relevance_score` **在终端逐条打印**（作业强制要求）。  
- 最终 **Top-3** 拼进原有「仅依据上下文」的 RAG Prompt（与 `w2` 约束一致）。

## 考核用例与预期行为

| 问题 | 路由 | 路径 |
|------|------|------|
| A：S&OP 手册缺货率>5% SKU 第三季度应急调拨审批流程 | document | 混合检索 → Rerank → Top-3 → LLM |
| B：上个月华东区门店预测准确率平均值 | analytics | Pandas Agent 对 CSV 聚合 |

## 提交物检查清单

- [x] 多文件工程 + `requirements.txt`  
- [x] 终端打印路由 + 文本路径打印 Rerank 得分  
- [x] 技术说明（本文档 + `SOP_HUB_ARCHITECTURE.md`）  

## 本地运行

```bash
cd langchain_learnning
pip install -r requirements.txt
# .env：api_key, base_url（须含 /v1，供 Chat、Embedding、POST /v1/rerank）
# 可选：SOP_RERANK_MODEL=BAAI/bge-reranker-v2-m3
python -m sop_hub --rebuild   # 首次、升 v3、或改 PDF 后
python -m sop_hub             # 日常
```

### 从旧版 w2 / manifest v2 迁移

v3 清单 `manifest_version=3` 且**仅索引 PDF**。旧索引若混有 CSV 向量或 v2 清单，首次运行会**自动全量重建** PDF 索引；CSV 仅在 `analytics` 路由下由 Pandas 读取。

### 依赖说明

- `langchain` 元包需提供 `langchain.retrievers.ensemble`；若报错可 `pip install -U langchain`。  
- `rank-bm25` 供 BM25Retriever 使用。  
- `langchain-experimental` 供 `create_pandas_dataframe_agent`。

---

*与 `sop_hub` 源码一一对应；若改模块名或环境变量，请同步更新本节。*
