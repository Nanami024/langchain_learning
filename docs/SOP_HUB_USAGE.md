# S&OP 智能决策中枢 v3.0 — 使用说明

本文面向**使用者**：从安装、配置知识库到日常提问、看懂终端输出。设计细节见 `SOP_HUB_ARCHITECTURE.md`。

---

## 1. 系统能做什么

| 能力 | 何时使用 | 技术路径 |
|------|----------|----------|
| **查手册 / 政策 / 流程** | 问题要在 **PDF** 叙述里找答案 | 语义路由 → **BM25 + FAISS 混合检索** → **Rerank** → 大模型作答 |
| **算表格 / 统计** | 问题要对 **CSV** 做平均、筛选、汇总等 | 语义路由 → **Pandas Agent**（**不走向量库**） |

程序会**自动二选一**。终端会先打印 **「命中路由」**，再出答案。

---

## 2. 环境准备

### 2.1 Python

建议 **Python 3.10+**。

### 2.2 安装依赖

在项目根目录（与 `sop_hub` 同级）执行：

```bash
cd c:\Users\ROG\Desktop\langchain_learnning
pip install -r requirements.txt
```

### 2.3 配置 `.env`（必配）

在项目根目录放置 `.env`（与 `w2.py` 同级即可），**至少**包含硅基流动（或其它 OpenAI 兼容网关）的密钥与地址：

```env
api_key=你的API密钥
base_url=https://api.siliconflow.com/v1
```

说明：

- **`base_url`** 请带 **`/v1`**。程序会调用同一套密钥下的：对话、Embedding、以及 **`POST {base_url}/rerank`** 重排接口，**一般不需要单独再申请「Rerank 专用 Key」**。
- 若 Rerank 报错，请到控制台确认该 Key 已开通对应模型、余额充足。

---

## 3. 准备知识库目录

默认目录名：**`sop_knowledge`**（与项目根同级），或通过环境变量 / 命令行指定。

| 放什么 | 用途 |
|--------|------|
| **`.pdf`** | 参与 **向量化索引**（FAISS + BM25 语料来自同一批切块） |
| **`.csv`** | **不**进向量库；仅当问题被路由为「数据分析」时，由 **Pandas** 读取计算 |

可选：在子目录里放 PDF/CSV，启动时加 **`-r` / `--recursive`** 递归扫描。

**注意**：当前版本的**文本索引只处理 PDF**。若目录里只有 CSV、没有 PDF，启动建索引会提示无法建文本索引（表格问答仍可在有 CSV 时使用）。

---

## 4. 如何启动

在**项目根目录**下执行（确保当前目录能 `import sop_hub`）：

```bash
# 首次使用、更换/批量更新 PDF、或想清空旧索引重来
python -m sop_hub --rebuild

# 日常使用（会按清单做增量或直载）
python -m sop_hub
```

等价写法：

```bash
python -m sop_hub.main --rebuild
python -m sop_hub.main
```

### 4.1 命令行参数

| 参数 | 含义 |
|------|------|
| `--dir 路径` | 知识库根目录；不配则用环境变量 `SOP_DOCS_DIR`，再默认 `sop_knowledge` |
| `-r` / `--recursive` | 递归子目录中的 PDF/CSV |
| `--rebuild` | 强制重建 PDF 向量索引（会重新 Embedding，耗时与文档量有关） |

示例：

```bash
python -m sop_hub --dir D:\data\sop_pack -r --rebuild
```

### 4.2 启动后终端交互

- 按提示输入问题，**流式**输出回答。
- 输入 **`quit`** 或 **`exit`** 退出。

---

## 5. 一次提问时终端会显示什么

### 5.1 路由为「文档检索」

你会依次看到大致如下信息（表述以实际运行为准）：

1. **`【命中路由】文档检索（BM25 + FAISS 混合 → Rerank）`**
2. 粗排候选条数、是否经过 Rerank
3. **`Rerank 最终得分`**：每条有 **`relevance_score`**（服务器返回的交叉打分）、以及**溯源**（如《某.pdf》第 n 页）
4. **`💡 回答：`** 模型正文（仅依据检索到的上下文）
5. 参考片段摘要

若 Rerank 接口失败，会提示降级，得分处可能为 **`N/A`** 并注明未经过 Rerank。

### 5.2 路由为「数据分析」

1. **`【命中路由】数据分析（Pandas Agent，不经过向量检索）`**
2. **`💡 回答：`** Agent 根据 **CSV** 执行代码后的结果（数值以表为准）

---

## 6. 索引与文件变动（你需要知道的）

- 向量与清单默认在 **`local_faiss_index`**（可用 `SOP_VECTORSTORE_DIR` 改）。
- 内含：`index.faiss`、`index.pkl`、`ingest_manifest.json`、`text_chunks.jsonl` 等。
- **只改 CSV**：一般不触发 PDF 索引重建；表格路由下 Agent 每次从磁盘读最新 CSV。
- **增删改 PDF**：多数情况会**增量**更新；若想彻底重来，用 **`--rebuild`**。
- **重命名 PDF**：旧文件名对应的向量会从索引中剔除，新文件名会当新文件嵌入。
- **删光所有 PDF**：下次启动时会**自动删除**上述索引文件，避免还能搜到已删文档（幽灵向量）。

### 6.1 「同步 / 精准剔除」是怎么做到的？（原理）

系统不靠人工对 FAISS「猜 id」，而是维护一份 **清单账本** + **按文件粒度记录向量 id**：

1. **键是什么**  
   每个入库的 PDF 用 **相对知识库根目录的路径** 做键（例如 `手册\A.pdf`），与磁盘扫描结果一一对应。路径变了（重命名 / 移动），旧键消失、新键出现。

2. **每个键记什么**  
   - **`sha256`**：该 PDF **当前内容**的哈希。内容一改，哈希就变。  
   - **`vector_ids`**：这个文件切出来的 **每一块**在 LangChain FAISS 里的 **文档 id 列表**。建库/增量结束后，用「遍历 docstore、读 `metadata['source']`」聚类得到。

3. **每次启动**  
   重新扫磁盘，得到 **当前** `{ 路径 → sha256 }`，与清单里的旧 `files` 做集合运算：  
   - **removed**：旧里有、当前扫描没有 → 文件删了或重命名掉了旧路径 → 把清单里该路径下 **所有 `vector_ids` 收集起来，调用 FAISS 的 `delete`**，从向量库里删掉这些向量。  
   - **changed**：路径还在，但 `sha256` 变了 → 先 **delete 旧 id**，再 **只对这个文件**重新切块、`add_documents`。  
   - **added**：只有新扫描里有 → 新文件或重命名后的新路径 → **只嵌入这些文件**，没有旧 id 可删。

4. **重命名为什么也算精准**  
   重命名 = 旧相对路径 **removed**（整文件向量一并删） + 新相对路径 **added**（重新向量化），不会在索引里留下旧文件名的向量。

5. **BM25 怎么跟上**  
   每次索引写盘后，从 **当前 FAISS docstore** 导出 `text_chunks.jsonl`，BM25 与向量块 **同源**，不会出现「向量删了 BM25 还在」的长期不一致。

实现代码主要在 `sop_hub/indexer.py`（模块顶部有英文注释版说明；`_incremental`、`group_vector_ids_by_file`、`purge_text_index_artifacts`）。

更宏观的设计见 `SOP_HUB_ARCHITECTURE.md` 与 `SOP_HUB_IMPLEMENTATION_GUIDE.md`。

---

## 7. 常用环境变量（可选调参）

不配则用代码里默认值。名称与 `sop_hub/config.py` 一致。

| 变量 | 作用 | 默认示意 |
|------|------|----------|
| `SOP_DOCS_DIR` | 知识库目录 | `sop_knowledge` |
| `SOP_VECTORSTORE_DIR` | 索引输出目录 | `local_faiss_index` |
| `SOP_CHAT_MODEL` | 对话模型 | `Pro/zai-org/GLM-4.7` |
| `SOP_EMBED_MODEL` | 嵌入模型 | `BAAI/bge-m3` |
| `SOP_EMBED_CHUNK_SIZE` | 嵌入 API 单批条数上限 | `32` |
| `SOP_RERANK_MODEL` | 重排模型 | `BAAI/bge-reranker-v2-m3` |
| `SOP_FAISS_WEIGHT` / `SOP_BM25_WEIGHT` | 混合检索权重 | `0.6` / `0.4` |
| `SOP_RETRIEVER_EACH_K` | 各子检索器先取多少条 | `12` |
| `SOP_ROUGH_CANDIDATES` | 送入 Rerank 的最大候选数 | `20` |
| `SOP_RERANK_TOP_N` | Rerank 后取几条 | `3` |
| `SOP_FINAL_CONTEXT_K` | 最终写入上下文的条数 | `3` |
| `SOP_CHUNK_SIZE` / `SOP_CHUNK_OVERLAP` | PDF 切块 | `500` / `50` |
| `SOP_INCREMENTAL` | 是否启用索引增量 | `1` |
| `SOP_USE_MMR` | FAISS 子路是否 MMR | `0` |

---

## 8. 常见问题

**Q：提示没有 `api_key` / `base_url`？**  
A：检查 `.env` 是否在项目根、`python-dotenv` 是否安装，变量名是否为 `api_key`、`base_url`。

**Q：Rerank 403/404？**  
A：核对 `base_url` 是否含 `/v1`；控制台是否支持所选 `SOP_RERANK_MODEL`；必要时换官方文档列出的 Rerank 模型名。

**Q：只有表格问题，没有 PDF？**  
A：需至少能保证「文档路由」不报错——若完全无 PDF，当前版本在建索引阶段会失败；可放一个占位小 PDF，或后续再扩展「无 PDF 仅表格」模式。

**Q：回答和参考看起来不一致？**  
A：先看是否走了**数据分析**路由却期望查手册，或反之；再等 Rerank/混合检索调参（增大 `SOP_ROUGH_CANDIDATES` 等）。

---

## 9. 相关文档索引

| 文档 | 内容 |
|------|------|
| `docs/SOP_HUB_ARCHITECTURE.md` | 架构与模块 |
| `docs/SOP_HUB_IMPLEMENTATION_GUIDE.md` | 与作业 OKR 的对照 |
| `SOP_RAG_TECH_DOC.md` | 旧版单脚本 `w2.py` 的技术说明（可选） |

---

*版本：v3.0 · 与 `sop_hub` 当前行为一致；若改入口或环境变量名，请同步更新本文。*
