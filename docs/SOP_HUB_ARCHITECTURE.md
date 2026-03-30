# S&OP 智能决策中枢 v3.0 — 技术架构说明

## 1. 目标与边界

| 能力 | 实现方式 |
|------|----------|
| 非结构化（PDF 手册等） | **仅 PDF** 进入向量索引；**BM25 + FAISS** 混合检索 → **Top-N 粗排 → Rerank API → Top-3** → LLM 生成 |
| 结构化（CSV 统计） | **严禁**走 FAISS；**语义路由**命中「数据分析」后，**Pandas DataFrame Agent** 对磁盘 CSV 执行真实计算 |
| 向量生命周期 | `ingest_manifest.json` 按 **PDF 相对路径 + 内容 SHA256** 跟踪；删改文件 **`delete` 向量 id** 后增量 `add_documents`；**重命名 = 旧路径删除 + 新路径新增** |

CSV **不参与**文本向量库，避免「表格数值题」误走语义检索；与作业要求一致。

## 2. 目录结构

```text
langchain_learnning/
├── docs/
│   ├── SOP_HUB_ARCHITECTURE.md      # 本文
│   └── SOP_HUB_IMPLEMENTATION_GUIDE.md
├── sop_hub/
│   ├── __init__.py
│   ├── config.py        # 环境变量、路径、权重、模型名
│   ├── constants.py     # 溯源 metadata 键名
│   ├── document_io.py   # PDF/CSV 枚举、加载、切块、溯源元数据
│   ├── indexer.py       # FAISS 构建/增量/清单/导出 text_chunks.jsonl
│   ├── rerank.py        # SiliconFlow /v1/rerank 客户端
│   ├── retriever.py     # BM25 + FAISS Ensemble → 粗排 → Rerank → Top-k
│   ├── router.py        # LCEL + structured output：document | analytics
│   ├── table_agent.py   # create_pandas_dataframe_agent
│   └── main.py          # CLI 入口、交互、路由与得分打印
├── requirements.txt
└── w2.py                # 历史单文件脚本（保留）
```

## 3. 数据流

### 3.1 索引（indexer）

1. 扫描知识库目录下全部 **`.pdf`**（可选递归），计算 `pdf_rel_path → sha256`。  
2. **指纹**仅由：嵌入模型、切块参数、`docs_dir`、recursive、**各 PDF 文件哈希** 构成（**不含 CSV**，避免加表触发整库重建）。  
3. 指纹命中且存在 `index.faiss` + 清单 → **直接加载**。  
4. 否则：增量或全量更新 FAISS；清单中记录每个 PDF 的 `vector_ids`；**从 docstore 导出** `text_chunks.jsonl` 供 BM25 加载（与 FAISS 切块一致）。  
5. **删除/重命名 PDF**：旧相对路径不在磁盘上 → `removed` → `delete` 该路径全部 id；新路径 → `added` → 仅新文件 embed。

### 3.2 文本检索（retriever）

1. **BM25Retriever**（`rank_bm25`）与 **FAISS 相似度** 各取一批块（`settings.retriever_each_k`）。  
2. **EnsembleRetriever** 按权重（默认 **FAISS 0.6 : BM25 0.4**）合并、去重，得到最多 **`rough_limit`（默认 20）** 条候选。  
3. 调用 **`POST {base_url}/rerank`**（与 SiliconFlow OpenAPI 一致）：`query`、`documents` 为候选文本列表，`top_n=3`。  
4. 将返回的 **`relevance_score`** 写回展示结构；**终端强制打印**每条 Top-3 的得分与溯源。

### 3.3 路由（router）

- 使用 **Chat + `with_structured_output(Pydantic)`**，输出 `route ∈ {document, analytics}`。  
- **document**：手册条款、流程、政策表述、概念解释等。  
- **analytics**：求平均/汇总/筛选/排序、「上个月/区域」等**表格计算**。

### 3.4 表格 Agent（table_agent）

- 扫描目录下 **`.csv`**（可选 `.xlsx` 若扩展），读入 `dict[name, DataFrame]`。  
- `create_pandas_dataframe_agent(..., allow_dangerous_code=True)` 执行 Python 计算，**不调用**向量库。

## 4. 配置项（环境变量摘要）

| 变量 | 含义 |
|------|------|
| `api_key` / `base_url` | SiliconFlow 兼容 Chat / Embedding / Rerank |
| `SOP_DOCS_DIR` | 知识库根目录 |
| `SOP_VECTORSTORE_DIR` | FAISS 与清单、jsonl 目录 |
| `SOP_FAISS_WEIGHT` / `SOP_BM25_WEIGHT` | Ensemble 权重 |
| `SOP_ROUGH_CANDIDATES` | 送入 Rerank 的最大候选数（默认 20） |
| `SOP_RERANK_MODEL` | 如 `BAAI/bge-reranker-v2-m3` |
| `SOP_RETRIEVER_EACH_K` | 子检索器各取条数 |

详见 `sop_hub/config.py` 内注释。

## 5. 依赖

见根目录 `requirements.txt`：`langchain`、`langchain-community`、`langchain-openai`、`langchain-experimental`、`faiss-cpu`、`rank_bm25`、`pandas`、`httpx`、`pydantic` 等。

## 6. 小巧思

- **PDF 与 CSV 指纹分离**：索引只跟 PDF 变动绑定，业务表可频繁替换而不必重建向量。  
- **Rerank 失败降级**：HTTP 错误时退回混合检索分数顺序截断 Top-3，并标注「未经过 Rerank」。  
- **路由可解释**：结构化输出带 `reason`，调试时可打印（默认关闭Verbose）。  
- **得分统一展示**：`relevance_score` 原样打印，便于对比 SiliconFlow 文档。

---

*版本：v3.0 · 与 `sop_hub/` 代码同步。*
