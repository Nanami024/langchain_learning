from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env.example", override=True)
load_dotenv(PROJECT_ROOT / ".env", override=True)
load_dotenv()


def _env(name: str, default: str = "") -> str:
    raw = os.getenv(name, default) or ""
    return raw.strip().strip('"').strip("'")


@dataclass(frozen=True)
class Settings:
    mysql_host: str = os.getenv("MYSQL_HOST", "127.0.0.1")
    mysql_port: int = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_db: str = os.getenv("MYSQL_DB", "sop_ai_system")
    mysql_user: str = os.getenv("MYSQL_USER", "root")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "")
    mysql_ro_user: str = os.getenv("MYSQL_RO_USER", "")
    mysql_ro_password: str = os.getenv("MYSQL_RO_PASSWORD", "")
    mysql_history_table: str = os.getenv("MYSQL_HISTORY_TABLE", "chat_history")
    sales_table: str = os.getenv("SOP_SALES_TABLE", "sales_performance")
    auto_seed: bool = os.getenv("SOP_AUTO_SEED", "1").strip().lower() not in ("0", "false", "no", "off")
    assumed_today: str = os.getenv("SOP_ASSUMED_TODAY", "2026-04-12")
    seed_days: int = int(os.getenv("SOP_SEED_DAYS", "120"))
    seed_regions: int = int(os.getenv("SOP_SEED_REGIONS", "5"))
    stores_per_region: int = int(os.getenv("SOP_STORES_PER_REGION", "12"))
    llm_api_key: str = _env("OPENAI_API_KEY")
    llm_base_url: str = _env("OPENAI_BASE_URL")
    llm_model: str = _env("OPENAI_MODEL", "gpt-4o-mini")
    sql_max_rows: int = int(os.getenv("SOP_SQL_MAX_ROWS", "30"))
    sql_timeout_seconds: int = int(os.getenv("SOP_SQL_TIMEOUT_SECONDS", "25"))
    scope_min_score: float = float(os.getenv("SCOPE_MIN_SCORE", "0.18"))
    route_min_data_score: float = float(os.getenv("ROUTE_MIN_DATA_SCORE", "0.14"))
    route_min_policy_score: float = float(os.getenv("ROUTE_MIN_POLICY_SCORE", "0.14"))
    policy_faiss_k: int = int(os.getenv("POLICY_FAISS_K", "8"))
    policy_bm25_k: int = int(os.getenv("POLICY_BM25_K", "8"))
    policy_rough_k: int = int(os.getenv("POLICY_ROUGH_K", "12"))
    policy_final_k: int = int(os.getenv("POLICY_FINAL_K", "4"))
    policy_faiss_weight: float = float(os.getenv("POLICY_FAISS_WEIGHT", "0.55"))
    policy_bm25_weight: float = float(os.getenv("POLICY_BM25_WEIGHT", "0.45"))
    risk_min_score: float = float(os.getenv("RISK_MIN_SCORE", "0.32"))
    route_use_llm: bool = os.getenv("ROUTE_USE_LLM", "0").strip().lower() not in ("0", "false", "no", "off")
    route_llm_min_confidence: float = float(os.getenv("ROUTE_LLM_MIN_CONFIDENCE", "0.72"))
    policy_retrieve_min_score: float = float(os.getenv("POLICY_RETRIEVE_MIN_SCORE", "0.07"))
    route_skip_llm_when_clear: bool = os.getenv("ROUTE_SKIP_LLM_WHEN_CLEAR", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    approval_use_llm_classifier: bool = os.getenv("APPROVAL_USE_LLM_CLASSIFIER", "0").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    # 1=默认可用快路径时也先走 NL2SQL（由模型写 SQL）；0=简单问法可走本地快路径（省调用）
    data_analyst_llm_first: bool = os.getenv("SOP_DATA_ANALYST_LLM_FIRST", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    # 1=uvicorn 日志里打印 data_analyst 各阶段耗时（与 .env.example 中 SOP_DATA_ANALYST_TIMING 一致）
    data_analyst_timing: bool = os.getenv("SOP_DATA_ANALYST_TIMING", "0").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    # 1=data_then_compliance 等长报告：先自然语言草稿再转 JSON 定版（更易解析），0=仅一阶段直接用草稿
    synthesis_two_stage_json: bool = os.getenv("SOP_SYNTHESIS_TWO_STAGE_JSON", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    # 合成/总结用 LLM 的 HTTP 超时（秒）。长上下文+慢网关若过短会 invoke 失败→「模型未返回有效正文」
    synthesis_timeout_seconds: int = int(os.getenv("SOP_SYNTHESIS_TIMEOUT_SECONDS", "120"))
    # 单次合成请求里 HumanMessage 最大字符；旧版硬编码 8000 会截断 JSON+RAG 导致非法 JSON 与空回复
    synthesis_max_user_chars: int = int(os.getenv("SOP_SYNTHESIS_MAX_USER_CHARS", "28000"))

    def llm_ready(self) -> bool:
        return bool(self.llm_api_key and self.llm_base_url)
