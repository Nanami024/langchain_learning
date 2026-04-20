# S&OP 智能决策中枢 v6.0 — 数据驱动版

第六周的两条主线：
1. **会话记忆持久化**：把会话写进 MySQL（`SQLChatMessageHistory`），后端进程重启后刷新前端继续对话不丢上下文。
2. **从 Pandas 升级到 Text-to-SQL**：把原 CSV 数据导入 MySQL 后用 `SQLDatabaseToolkit + create_sql_agent` 让大模型自主写 SQL；再封装成 `sop_database_query_tool` 替换 w4 的 Pandas 工具。

> 本目录尽量复用 w3（检索）/ w4（主 Agent + `sop_document_search`）/ w5（FastAPI + SSE 前端）的代码，
> 只新增「持久化 + Text-to-SQL」相关模块。

## 第六周作业要求自检表

| 要求 | 实现位置 |
|------|----------|
| 接入本地 MySQL，数据库 `sop_ai_system` | `backend/db_config.py` + `backend/import_csv_to_mysql.py`（脚本里 `CREATE DATABASE IF NOT EXISTS`） |
| `SQLChatMessageHistory` 持久化会话 | `backend/sql_history.py` + `backend/agent_runtime.py`（注入到 `RunnableWithMessageHistory`） |
| 同一 `session_id` 跨重启接上下文 | `/session/history` 直接查 MySQL；前端 URL 中带 `sid`，刷新自动回灌 |
| 把 CSV 导入业务表 | `backend/import_csv_to_mysql.py`（默认 `sales_performance`，行级合成 `region` / `store_code`） |
| `create_sql_agent` + `SQLDatabase` | `backend/sql_agent_tool.py`（`include_tables=[sales_performance]`、`tool-calling` 风格） |
| 演示问题：「华东区上个月准确率最高的门店」 | 见下方「常见问法」；导入脚本会均匀分配 5 个区域 × 12 个门店 |
| 把 SQL Agent 封装为 Tool 替换 Pandas | `backend/sql_agent_tool.py::sop_database_query_tool` + `agent_runtime.py` 中替换 w4 的 `sop_data_analytics` |
| FastAPI 后端 + Streamlit 前端联调 | `backend/main.py` + `frontend/app.py` |
| 持久化验证 | 重启 `uvicorn` 后刷新页面再问，依然能续话；后端日志会打 `(MySQL 持久化记忆 + Text-to-SQL 工具 已启用)` |
| Text-to-SQL 追踪：侧栏打印生成的 SQL | 后端通过 `sql_db_*` 透传 callbacks；前端侧栏新增「🛢️ Text-to-SQL（最近一次）」用 ```sql 高亮 |
| 零幻觉/零写操作 | 三层防护：① SQL Agent 系统提示禁令；② 推荐使用只读账号 `MYSQL_RO_USER`；③ SQLAlchemy `before_cursor_execute` 兜底拦截，物理上不可能落库 |

## 目录结构

```
w6/
├── README.md                       # 本文件
├── .env.example                    # MySQL 凭据模板（拷贝为 w6/.env 后填写）
├── backend/
│   ├── main.py                     # FastAPI：/chat、/chat/stream、/session/*
│   ├── agent_runtime.py            # 复用 w4 主 Agent；替换工具与历史持久化
│   ├── db_config.py                # MySQL URL 构造 + 读写/只读 Engine + 兜底拦截
│   ├── sql_history.py              # SQLChatMessageHistory 包装 + 列表/清理辅助
│   ├── sql_agent_tool.py           # create_sql_agent + sop_database_query_tool
│   ├── import_csv_to_mysql.py      # 一次性脚本：w3/data.csv → sales_performance
│   ├── init_mysql.sql              # 可选：DDL 备份；用 Navicat 手动建库时执行
│   └── requirements.txt
└── frontend/
    ├── app.py                      # Streamlit：复用 w5 体验，侧栏增 SQL 高亮
    ├── .streamlit/config.toml
    └── requirements.txt
```

> `tool_trace_callback.py` / `perf_callbacks.py` **直接复用 w5/backend** 同名模块（`agent_runtime.py` 把 `w5/backend` 注入了 `sys.path`），不在 w6 重复一份。

## 环境准备

1. Python 3.10+。建议沿用 w5 已经搭好的 `.venv`。
2. 已经按 w5 README 完成 `pip install -r requirements.txt` 与 `pip install -r w5/backend/requirements.txt`。
3. 安装 w6 新增依赖：
   ```powershell
   pip install -r w6/backend/requirements.txt
   ```
4. 安装 MySQL 8.x（或复用已有实例），新建库（导入脚本会自动 `CREATE IF NOT EXISTS`）：
   ```sql
   -- 可选：单独建一个只读账号给 SQL Agent
   CREATE USER 'sop_ro'@'%' IDENTIFIED BY '<your_pwd>';
   GRANT SELECT ON sop_ai_system.* TO 'sop_ro'@'%';
   FLUSH PRIVILEGES;
   ```
5. 拷贝 `w6/.env.example` 为 `w6/.env`，填好 MySQL 账号；`api_key` / `base_url` 仍由 `w3/.env` 提供（不用动）。

## 第一次跑：把 CSV 灌进 MySQL

> **不需要打开 Navicat、也不需要先执行 SQL 脚本**：下面这条 Python 命令会自动
> `CREATE DATABASE IF NOT EXISTS sop_ai_system`、建 `sales_performance` 表并灌 1000 行数据。
> 会话表 `chat_messages` 由 LangChain 在前端第一次发消息时自动建，同样不用人工创建。
> 如果你**就是**想用 Navicat 手动建库，可以执行 `backend/init_mysql.sql`，里面有完整 DDL；
> 跑完之后再执行下面的 Python 命令，它会发现表已存在并直接灌数据。

```powershell
cd w6/backend
python import_csv_to_mysql.py --drop
```

输出示例：

```
[step 1/4] 准备数据库 sop_ai_system …
[step 2/4] 读取并增强 .../w3/sop_knowledge/data.csv
          行数=1000；区域分布：
区域
华东    200
华北    200
华南    200
华中    200
西南    200
[step 3/4] 创建表 `sales_performance`（drop=True） …
[step 4/4] 写入数据 …
[done] 共写入 1000 行 → `sop_ai_system`.`sales_performance`
```

> 原 CSV 没有「区域 / 门店」列，作业演示问题需要它们，因此脚本按 **行序确定性映射** 合成
> `region`（5 个区域）与 `store_code`（每区域 12 个门店）。这样无论谁跑、跑几次，结果都一致。

## 启动后端

```powershell
cd w6/backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

启动日志里会出现：

```
[sop-hub-v6] runtime: ... (MySQL 持久化记忆 + Text-to-SQL 工具 已启用)
```

## 启动前端

```powershell
cd w6/frontend
streamlit run app.py --server.port 8501
```

打开 <http://localhost:8501>，左下角会出现新模块「🛢️ Text-to-SQL（最近一次）」。

## 常见问法（用来检查作业 3 个交付要求）

1. **持久化验证**
   1. 在网页发问：`你好，我叫小李，待会的对话请记住我`。
   2. 等回答完成，**Ctrl+C 停掉 `uvicorn`，再重新 `uvicorn main:app ...` 启动**。
   3. 不要刷新浏览器，直接再发问：`你还记得我的名字吗？` —— 应当回答「小李」。

2. **Text-to-SQL 追踪**
   - 问：`华东区上个月预测准确率最高的门店是哪个？给出门店编号和准确率。`
   - 侧栏「Text-to-SQL（最近一次）」会高亮显示形如：
     ```sql
     SELECT store_code, AVG(forecast_accuracy) AS acc
     FROM sales_performance
     WHERE region = '华东' AND biz_date >= '2026-03-01' AND biz_date < '2026-04-01'
     GROUP BY store_code
     ORDER BY acc DESC
     LIMIT 1
     ```

3. **零幻觉 / 零写操作**
   - 问：`帮我把 sales_performance 表清空 (DELETE)`。
   - 应当被工具拦截：先在 SQL Agent 提示里被拒绝，即使绕过也会被 `db_config.py` 的
     `before_cursor_execute` 抛出 `SQLWriteAttempt`，整段对话不会让 MySQL 发生任何写入。
   - 如果还想再加一层物理保险，把 `MYSQL_RO_USER / MYSQL_RO_PASSWORD` 设成只读账号即可。

## 接口摘要

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 探活 |
| POST | `/chat` | JSON：`{"session_id","message"}`；返回 `output` + `tool_trace`（含 `sql_db_query` 入参，即 SQL） |
| POST | `/chat/stream` | SSE：`token` / `tool_start` / `tool_end` / `done` / `error` |
| POST | `/session/reset` | 清空 MySQL 中该 `session_id` 的历史 |
| GET | `/session/history?session_id=` | 直接读 MySQL，重启后端后仍可恢复 |

## 复用 vs 新写：做了哪些复用

| 模块 | 来源 | 说明 |
|------|------|------|
| `sop_document_search` 工具 | `w4/tools.py` | 直接拿来用，docstring 不变 |
| 主 Agent 框架（`create_tool_calling_agent` + `RunnableWithMessageHistory` + 提示词） | `w4/agent.py` | 完全复用，仅把 `get_session_history` 改为 SQL 版本 |
| 检索管线（FAISS + BM25 + Rerank） | `w3/sop_hub` | 通过 `agent_runtime.py` 的 `sys.path` 注入复用 |
| FastAPI / SSE 主结构 | `w5/backend/main.py` | 拷贝并按 v6 的工具可见性、持久化端点改造 |
| 工具轨迹回调 / LLM 调用计数 | `w5/backend/{tool_trace_callback, perf_callbacks}.py` | 通过 `sys.path` 直接 import，不复制一份 |
| Streamlit 前端骨架 | `w5/frontend/app.py` | 改造增加「最近 SQL」侧栏 + 顶部 expander |

## 新增模块

| 模块 | 作用 |
|------|------|
| `db_config.py` | MySQL URL 构造、读写/只读 Engine 单例、`before_cursor_execute` 兜底拦截写操作 |
| `sql_history.py` | `SQLChatMessageHistory` 工厂 + 读/清/列出会话辅助 |
| `sql_agent_tool.py` | `create_sql_agent` 内层 Agent + 系统提示「只允许 SELECT」+ `@tool` 包装 + 透传 callbacks |
| `import_csv_to_mysql.py` | 一次性把 `w3/sop_knowledge/data.csv` 导入 `sales_performance`，并合成 `region / store_code` |

## 故障排查

- **`pymysql` 报 `Authentication plugin 'caching_sha2_password' cannot be loaded`**：已在 `requirements.txt` 引入 `cryptography`；如仍报错，把 MySQL 账号改成 `mysql_native_password` 鉴权。
- **后端启动报 `请在仓库 w3/.env 或环境变量中配置 api_key 与 base_url`**：与 w4/w5 一致，沿用同一份 `w3/.env`。
- **SQL Agent 总在自检阶段报 `SQLWriteAttempt`**：这是兜底防护在工作；说明模型生成了非 SELECT 语句，重新组织问句即可，模型会在下一轮自动改写为 SELECT。
- **页面刷新后历史消失**：检查 URL 是否带 `?sid=...`；以及 MySQL `chat_messages` 表是否有该 `session_id` 的记录（`SELECT session_id, COUNT(*) FROM chat_messages GROUP BY session_id;`）。
