# S&OP 智能决策中枢 v5.0 — Web 版

前后端分离：**FastAPI** 提供 `/chat` 与 **SSE** 流式 `/chat/stream`；**Streamlit** 提供类 ChatGPT 对话页（`st.chat_message` / `st.chat_input`），并在**侧边栏**与**展开面板**中展示工具调用轨迹。

## 第五周作业要求对照（自检）

| 要求 | 实现位置 |
|------|-----------|
| `backend/`、`frontend/` 分目录 + 根目录本 README | `w5/backend/`、`w5/frontend/`、`w5/README.md` |
| FastAPI 封装 Agent、REST 风格接口 | `backend/main.py`（路由、Pydantic 请求/响应体） |
| `POST /chat` JSON，可用 Postman / curl | `ChatRequest` → 返回 `output`、`tool_trace` |
| 复用 w4 Agent 核心（与终端一致） | `backend/agent_runtime.py` 调用 `w4.agent` / `w4.tools` + w3 索引 |
| Streamlit 网页、`st.chat_message` / `st.chat_input` | `frontend/app.py` |
| `st.session_state` 维护会话 | `messages`、`session_id` |
| 前端用 `requests` 调后端 | 流式与非流式请求均使用 `requests` |
| SSE 流式 + 网页打字机效果 | 后端 `StreamingResponse` + 前端循环解析 `data:` 行并刷新占位符 |
| 侧栏或折叠区展示工具调用 | 侧栏「本回合工具轨迹」+ 助手消息内 `st.expander` 同步更新 |
| 浏览器访问 `localhost:8501` 即可对话 | 见下方启动命令（须先起后端 8000） |

**关于课表中的 LangServe**：第五周日程提到「LangServe 基础」属于扩展知识。本仓库为完整控制 **会话记忆（`session_id`）** 与 **SSE 事件格式（token / tool_*）**，采用 **FastAPI 原生** 挂载 Runnable，未再叠一层 LangServe；你在课堂对比「LangServe 一键挂载链」与「手写 `/chat/stream`」即可。

## 环境准备

1. Python 3.10+（推荐 3.11/3.12）。
2. 在仓库 **`w3/`** 配置好 `.env`（与 w4 相同：`api_key`、`base_url`、`SOP_CHAT_MODEL` 等），并确保 **`w3/local_faiss_index`** 已建索引（若未建库，可先运行一次 `python -m w4 --rebuild` 或 w3 主程序）。

### 为什么说「要用装过依赖的同一个 Python」？

Windows 上常会同时存在多个解释器，例如：微软商店版 `python`、`py launcher`、你自己装的 `Python 3.14` 等。  
若出现下面两种情况之一，就会 **装包和运行不是同一套环境**，从而 `ModuleNotFoundError: No module named 'fastapi'`：

- 用 **`pip install`** 时，实际调用的是 **A 解释器** 的 pip（包装进了 A 的 `site-packages`）；  
- 用 **`python`** / **`uvicorn`** 启动时，终端找到的却是 **B 解释器**（里面没装 FastAPI）。

**推荐做法：在本仓库根目录建一个虚拟环境**（例如 `.venv`），**先激活，再 pip、再跑后端/前端**。这样「装依赖」和「运行」一定指向同一条 `python.exe`，不会串环境。

在仓库根目录 `langchain_learnning/`（PowerShell 示例）：

```powershell
cd C:\Users\你的用户名\Desktop\langchain_learnning
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
pip install -r w5/backend/requirements.txt
pip install -r w5/frontend/requirements.txt
```

之后在**已激活**的同一终端里启动后端、再开新终端同样 `Activate.ps1` 后启动 Streamlit 即可。（两个终端都要激活同一个 `.venv`，或全程只用一个终端先后跑。）

3. 若不用虚拟环境，也务必保证：**执行 `pip install` 时显示的 pip 路径** 与 **`where python` / `Get-Command python` 指向的解释器** 一致（否则仍会缺包）。

4. 安装依赖（**在已选定的环境中**，于仓库根目录执行）：

```bash
pip install -r requirements.txt
pip install -r w5/backend/requirements.txt
pip install -r w5/frontend/requirements.txt
```

## 启动顺序（重要）

1. **先启动后端**（加载向量库与 Agent，首次较慢）。  
2. **再启动前端**（Streamlit 通过 `requests` 访问 `http://127.0.0.1:8000`）。

API 交互式文档：<http://127.0.0.1:8000/docs>（Swagger）。

## 启动后端（FastAPI）

在目录 **`w5/backend/`** 下（**先激活 `.venv`**，与上面 `pip install` 同一环境）：

```bash
cd w5/backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

若未将 `uvicorn` 加入 PATH，可用（同样会走当前激活环境中的解释器）：

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

可选环境变量：

| 变量 | 说明 |
|------|------|
| `SOP_DOCS_DIR` | 知识库目录，默认走 w3 的 `sop_knowledge` |
| `SOP_FORCE_REBUILD` | 设为 `1` 时启动时强制重建向量索引 |
| `SOP_CORS_ORIGINS` | 逗号分隔的允许源，默认 `*` |
| `SOP_ASSUMED_TODAY` | 与 w4 一致，解析「上个月」等相对时间 |
| `SOP_LLM_STREAMING` | 默认 `1`（A 方案：`ChatOpenAI` 流式）；设 `0` 可与 w4 CLI 非流式对齐、常更快 |
| `SOP_SSE_ASTREAM_EVENTS` | 默认 `1`（A 方案：`astream_events` 真 token SSE）；设 `0` 则 `invoke` 后再 SSE 回放 |
| `SOP_SSE_ASSISTANT_TYPING` | 默认 `0`；设 `1` 时在 invoke 回放路径下将正文切成多段 `token`（更像打字机，事件更多） |

健康检查：<http://127.0.0.1:8000/health>

### 用 curl 测非流式接口

```bash
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"session_id\":\"demo1\",\"message\":\"你好，简要自我介绍\"}"
```

（Linux/macOS 将 `^` 换为 `\`。）

## 启动前端（Streamlit）

**另开一个终端**，在目录 **`w5/frontend/`** 下：

```bash
cd w5/frontend
streamlit run app.py --server.port 8501
```

浏览器打开：**<http://localhost:8501>**

- 侧边栏可改后端地址（默认 `http://127.0.0.1:8000`）。
- **流式输出**开关：默认**开启**，走 `/chat/stream`（SSE）；后端默认 **A 方案**（`astream_events` + 模型流式）。关闭时走 `/chat` 一次性返回（仍带 `tool_trace`）。
- **新对话**会调用 `/session/reset` 并生成新的 `session_id`。

也可通过环境变量指定 API：

```bash
set SOP_API_URL=http://127.0.0.1:8000
streamlit run app.py --server.port 8501
```

## 接口说明（摘要）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 探活 |
| POST | `/chat` | JSON 请求体 `{"session_id","message"}`，响应含 `output` 与 `tool_trace` |
| POST | `/chat/stream` | 同上，响应为 `text/event-stream`（SSE），`data:` 行为 JSON，`type` 为 `token` / `tool_start` / `tool_end` / `done` / `error` |
| POST | `/session/reset` | 请求体 `{"session_id"}`，清空该会话在后端的聊天记忆 |

## 目录结构

```
w5/
├── README.md           # 本说明
├── backend/            # FastAPI
│   ├── main.py
│   ├── agent_runtime.py
│   ├── tool_trace_callback.py
│   └── requirements.txt
└── frontend/           # Streamlit
    ├── app.py
    ├── .streamlit/
    │   └── config.toml # 主题与基础配置
    └── requirements.txt
```

Agent 逻辑复用 **`w4`**（`build_agent_with_history`、`build_sop_tools`、w3 检索与表格管线），与终端版行为一致；v5 仅增加 Web 层与 SSE 事件桥接。
