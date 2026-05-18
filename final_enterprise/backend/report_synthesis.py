from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import Settings

ResponseMode = Literal["data_only", "policy_only", "full"]

_SYNTH_RULES = (
    "你是企业 S&OP 助手。\n"
    "【硬性要求】\n"
    "1) 必须先直接回答用户问题（结论写在最前，1-4句）。\n"
    "2) 参考材料仅供你归纳，禁止大段复制粘贴原文，禁止把「规则片段」标题或整段语料当作正文。\n"
    "3) 制度/数据引用用你自己的话概括，必要时括号注明来源文件名。\n"
    "4) 若材料不足以定论，明确说明缺什么，不要堆砌语料。\n"
)


def build_synthesis_llm(settings: Settings) -> ChatOpenAI | None:
    if not settings.llm_ready():
        return None
    try:
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=0.2,
            request_timeout=max(18, min(300, int(settings.synthesis_timeout_seconds or 120))),
            max_retries=0,
        )
    except Exception:
        return None


def revise_report_draft_with_feedback(
    llm: ChatOpenAI | None,
    *,
    draft_markdown: str,
    human_feedback: str,
    user_question: str = "",
    settings: Settings | None = None,
) -> str:
    """人工审批「给修改意见」后：由 LLM 按意见重写报告；失败返回空串由调用方兜底。"""
    if llm is None:
        return ""
    system = (
        "你是企业 S&OP 报告编辑。下面是已生成的 Markdown 报告草稿及人工修改意见。\n"
        "请**根据意见重写整份报告**（保留清晰的 Markdown 结构；若原文有 ## 回答、## 数据支撑、"
        "## 制度依据、## 执行与审批 等标题，请尽量沿用同级标题组织内容）。\n"
        "必须实质修改正文以体现人工要求，禁止只写一句「已收到意见」或泛泛的套话。\n"
        "禁止编造原文没有出现过的数字、表名或制度条款；若意见要求的论据在草稿中不存在，请明确说明需补充数据或检索。\n"
        "输出自然语言 Markdown，不要整篇包在代码围栏里，不要输出 JSON。"
    )
    st = settings or Settings()
    cap = max(8000, int(st.synthesis_max_user_chars or 28000))
    q_part = (user_question or "")[:2000]
    fb_part = (human_feedback or "")[:4000]
    draft_cap = max(4000, cap - len(q_part) - len(fb_part) - 80)
    user = (
        f"用户原始问题（供对照）：{q_part}\n\n"
        f"---\n当前报告草稿：\n{(draft_markdown or '')[:draft_cap]}\n\n"
        f"---\n人工意见：\n{fb_part}"
    )
    try:
        msg = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)],
        )
        text = _strip_outer_code_fence(_lc_message_content(getattr(msg, "content", None)).strip())
        if text:
            return text
    except Exception as exc:
        logger.warning("revise_report_draft_with_feedback failed: %s: %s", type(exc).__name__, exc, exc_info=True)
    return ""


def llm_status(settings: Settings) -> dict[str, Any]:
    if not settings.llm_ready():
        return {
            "configured": False,
            "reachable": False,
            "detail": "缺少 OPENAI_API_KEY 或 OPENAI_BASE_URL（请写入 final_enterprise/.env 并重启 backend）",
        }
    model = build_synthesis_llm(settings)
    if model is None:
        return {
            "configured": True,
            "reachable": False,
            "detail": "已读取环境变量，但 ChatOpenAI 初始化失败，请检查模型名与 base_url",
            "model": settings.llm_model,
            "base_url": settings.llm_base_url,
        }
    try:
        msg = model.invoke([HumanMessage(content="只回复一个词：pong")])
        return {
            "configured": True,
            "reachable": True,
            "detail": "LLM 可用",
            "model": settings.llm_model,
            "base_url": settings.llm_base_url,
            "probe": _lc_message_content(getattr(msg, "content", None))[:80],
        }
    except Exception as exc:
        return {
            "configured": True,
            "reachable": False,
            "detail": f"LLM 调用失败：{type(exc).__name__}: {exc}",
            "model": settings.llm_model,
            "base_url": settings.llm_base_url,
        }


def _strip_outer_code_fence(text: str) -> str:
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _lc_message_content(content: Any) -> str:
    """LangChain AIMessage 在部分网关下 content 为 str 或 content block 列表。"""
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _invoke_synthesis_plain(
    model: ChatOpenAI,
    system: str,
    user: str,
    *,
    max_user_chars: int = 28000,
) -> str:
    """w6 风格：单次对话，不要求 JSON，避免网关/模型包一层说明导致解析失败。"""
    cap = max(2000, int(max_user_chars or 28000))
    raw_user = user or ""
    if len(raw_user) > cap:
        logger.warning(
            "synthesis user payload truncated: chars=%d -> cap=%d (raise SOP_SYNTHESIS_MAX_USER_CHARS if needed)",
            len(raw_user),
            cap,
        )
    clipped = raw_user[:cap]
    try:
        msg = model.invoke(
            [SystemMessage(content=system), HumanMessage(content=clipped)],
        )
        out = _strip_outer_code_fence(_lc_message_content(getattr(msg, "content", None)))
        if not out.strip():
            logger.warning("synthesis_llm returned empty content (model=%s)", getattr(model, "model_name", ""))
        return out
    except Exception as exc:
        logger.warning(
            "synthesis_llm invoke failed: %s: %s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return ""


def _parse_json_relaxed(raw: str) -> dict[str, Any] | None:
    """第二阶段专用：从模型输出中抠 JSON 对象。"""
    text = _strip_outer_code_fence((raw or "").strip())
    if not text:
        return None
    try:
        o = json.loads(text)
        if isinstance(o, dict):
            return o
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
        if isinstance(o, dict):
            return o
    except json.JSONDecodeError:
        return None
    return None


def _draft_to_report_json(
    model: ChatOpenAI,
    draft_markdown: str,
    *,
    max_user_chars: int = 28000,
) -> dict[str, Any] | None:
    """第二路模型：只做「长文 → 结构化 JSON」，输入短、任务单一，成功率高于一步 JSON。"""
    system = (
        "你是 JSON 转换器。下面用户消息里是一篇已定稿的中文 Markdown 草稿（含 ## 回答 等小节）。\n"
        "你的任务：**只输出一个 JSON 对象**，键名必须完全一致且值为字符串：\n"
        "direct_answer, data_support, policy_support, action, approval_status, risk_note。\n"
        "从草稿对应小节摘入；某键在草稿中不明显则从全文归纳，不要用 null。\n"
        "字符串内禁止未转义的双引号；换行用 \\n。\n"
        "禁止编造草稿中不存在的数字。\n"
        "不要输出任何 JSON 以外的字符（不要代码围栏、不要前言后语）。"
    )
    cap = max(4000, int(max_user_chars or 28000) - 200)
    user = f"草稿如下：\n{(draft_markdown or '')[:cap]}"
    raw = _invoke_synthesis_plain(model, system, user, max_user_chars=max_user_chars)
    return _parse_json_relaxed(raw)


def _render_full_report_from_json(
    obj: dict[str, Any],
    *,
    sql_summary: str,
    approval_line: str,
    risk_line: str,
) -> str:
    direct = str(obj.get("direct_answer", "") or "").strip()
    data_sup = str(obj.get("data_support", "") or "").strip() or sql_summary
    policy_sup = str(obj.get("policy_support", "") or "").strip() or "（细则见文末制度摘录。）"
    action = str(obj.get("action", "") or "").strip()
    appr = str(obj.get("approval_status", "") or "").strip() or approval_line
    risk = str(obj.get("risk_note", "") or "").strip() or risk_line
    parts: list[str] = [
        "## 回答",
        "",
        direct or data_sup,
        "",
        "## 数据支撑",
        "",
        data_sup,
        "",
        "## 制度依据",
        "",
        policy_sup,
        "",
        "## 执行与审批",
        "",
    ]
    if action:
        parts.append(f"执行建议：{action}\n")
    parts.append(f"审批：{appr}")
    parts.append(f"风险：{risk}")
    return "\n".join(parts).strip()


def _refs_block(snippets: list[str], sources: list[dict[str, str]] | None = None) -> str:
    if not snippets and not sources:
        return ""
    lines = ["", "---", "**参考来源（佐证，非正文）**"]
    for i, snip in enumerate(snippets[:4], start=1):
        preview = re.sub(r"\s+", " ", (snip or "")).strip()[:100]
        if len(snip or "") > 100:
            preview += "…"
        src_name = ""
        if sources and i - 1 < len(sources):
            src_name = str(sources[i - 1].get("name", "")).strip()
        cite = f"（{src_name}）" if src_name else ""
        lines.append(f"- 摘录{i}{cite}：{preview}")
    return "\n".join(lines)


def _pack_reference(question: str, snippets: list[str], extra: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {
        "question": question,
        "reference_material": snippets[:8],
        "note": "以下内容仅供归纳，不得原文粘贴到回答字段",
    }
    if extra:
        payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def format_data_only(
    *,
    question: str,
    sql_summary: str,
    sql_text: str,
    row_count: int,
    preview: list[Any] | None,
    settings: Settings,
    llm: ChatOpenAI | None = None,
    sources: list[dict[str, str]] | None = None,
    rewrite_with_llm: bool = False,
) -> str:
    model = llm if llm is not None else (build_synthesis_llm(settings) if rewrite_with_llm else None)
    summary = (sql_summary or "未查询到有效数据。").strip()
    body = summary
    if model is not None:
        plain = _invoke_synthesis_plain(
            model,
            _SYNTH_RULES + "用户关注的是数据库查询结论。请根据材料用 2～5 句中文概括，不要输出 JSON。",
            _pack_reference(question, [], {"sql_summary": summary, "row_count": row_count, "sql": sql_text}),
            max_user_chars=settings.synthesis_max_user_chars,
        )
        if plain:
            body = plain

    lines = ["## 数据查询结果", "", body]
    if row_count >= 0:
        lines.append(f"\n命中行数：{row_count}")
    if sql_text.strip():
        lines.extend(["", "执行SQL：", f"```sql\n{sql_text.strip()}\n```"])
    if preview:
        lines.extend(["", "样例数据（最多3行）："])
        for idx, row in enumerate(preview[:3], start=1):
            lines.append(f"- 行{idx}: {row}")
    lines.append(_refs_block([], sources))
    return "\n".join(lines).strip()


def format_policy_only(
    *,
    question: str,
    compliance_snippets: list[str],
    settings: Settings,
    llm: ChatOpenAI | None = None,
    sources: list[dict[str, str]] | None = None,
) -> str:
    model = llm or build_synthesis_llm(settings)
    if model is not None and compliance_snippets:
        pack = _pack_reference(question, compliance_snippets)
        body = _invoke_synthesis_plain(
            model,
            _SYNTH_RULES
            + "用户只问制度/标准/评级分档等。请用 2～8 句中文直接解读，可分段但不要 JSON、不要用代码围栏。\n"
            "若涉及分级，请写明各档名称与阈值（如有材料依据）。",
            pack,
            max_user_chars=settings.synthesis_max_user_chars,
        )
        if body:
            return "## 制度解读\n\n" + body + _refs_block(compliance_snippets, sources)
        degrade = _degraded_policy_section(
            compliance_snippets,
            note="模型调用未返回有效文本（超时或网关错误）。以下为检索摘录。",
        )
        return f"## 制度解读\n\n{degrade}" + _refs_block(compliance_snippets, sources)

    if not compliance_snippets:
        return "## 制度解读\n\n未检索到可用的制度片段，请换种问法或检查 PDF 是否已入库。"
    if model is None:
        return (
            "## 制度解读\n\n"
            "未加载 LLM：请在 `final_enterprise/.env` 配置 OPENAI_API_KEY、OPENAI_BASE_URL 后重启后端；"
            "当前仅展示检索摘录。"
            + _refs_block(compliance_snippets, sources)
        )
    return (
        "## 制度解读\n\n"
        "已检索到制度材料，但未能生成解读（请检查网络与模型网关）。"
        + _refs_block(compliance_snippets, sources)
    )


def _degraded_policy_section(compliance_snippets: list[str], *, note: str) -> str:
    lines = [
        note,
        "",
        "（以下为检索命中的制度原文摘录，可直接对照；完整内容见文末参考来源。）",
    ]
    for i, snip in enumerate(compliance_snippets[:5], start=1):
        short = re.sub(r"\s+", " ", (snip or "").strip())
        if len(short) > 480:
            short = short[:480] + "…"
        lines.append(f"- **摘录{i}** {short}")
    return "\n".join(lines)


def format_full_report(
    *,
    question: str,
    sql_summary: str,
    compliance_snippets: list[str],
    requires_review: bool,
    settings: Settings,
    llm: ChatOpenAI | None = None,
    sources: list[dict[str, str]] | None = None,
    snippet_sources: list[dict[str, str]] | None = None,
) -> str:
    model = llm or build_synthesis_llm(settings)
    sql_summary = (sql_summary or "无数据分析结果").strip()
    approval_line = "待人工确认。" if requires_review else "无需审批（非敏感问题）。"
    risk_line = "需人工审批后执行。" if requires_review else "无需审批（非高风险问题）。"
    cite_sources = snippet_sources if snippet_sources else sources

    if model is None:
        hint = (
            "未加载 LLM：请在 `final_enterprise/.env` 配置 OPENAI_API_KEY、OPENAI_BASE_URL、OPENAI_MODEL，"
            "保存后重启 uvicorn；浏览器打开 http://127.0.0.1:8100/health/llm 自检。"
        )
        if not settings.llm_ready():
            hint = (
                "环境变量未生效：backend 进程未读到 OPENAI_API_KEY / OPENAI_BASE_URL。"
                "请确认 .env 在 final_enterprise 目录、修改后已重启 backend。"
            )
        return (
            "## 回答\n\n"
            f"{sql_summary}\n\n"
            "## 数据支撑\n\n"
            f"{sql_summary}\n\n"
            "## 制度依据\n\n"
            f"{hint}\n\n"
            "## 执行与审批\n\n"
            f"审批：{approval_line}"
            + _refs_block(compliance_snippets, cite_sources)
        )

    user_payload = _pack_reference(
        question,
        compliance_snippets,
        {"data_summary": sql_summary, "requires_human_approval": requires_review},
    )
    system = (
        _SYNTH_RULES
        + "你是企业 S&OP 总结官（与 w6 synthesis_node 相同策略：自然语言输出，不输出 JSON）。\n"
        "请阅读 user 消息中的 JSON：内含 question、reference_material、data_summary、requires_human_approval。\n"
        "用 Markdown 输出，且必须按顺序包含以下二级标题（标题文字须一致）：## 回答 / ## 数据支撑 / ## 制度依据 / ## 执行与审批\n"
        "各节 2～6 句；结论节先直接答用户问题；制度节概括摘录要点；执行与审批节说明是否需审批及风险一句。\n"
        "禁止编造材料中不存在的数字或条款。"
    )
    body = _invoke_synthesis_plain(
        model,
        system,
        user_payload,
        max_user_chars=settings.synthesis_max_user_chars,
    )
    if body and settings.synthesis_two_stage_json:
        structured = _draft_to_report_json(
            model,
            body,
            max_user_chars=settings.synthesis_max_user_chars,
        )
        if structured:
            body = _render_full_report_from_json(
                structured,
                sql_summary=sql_summary,
                approval_line=approval_line,
                risk_line=risk_line,
            )
    if body:
        return body.strip() + _refs_block(compliance_snippets, cite_sources)

    degrade_note = "模型未返回有效正文（超时或网关错误）。数据摘要如下，可对照摘录。"
    policy_fallback = _degraded_policy_section(compliance_snippets, note=degrade_note)
    return (
        "## 回答\n\n"
        f"{sql_summary}\n\n"
        "## 数据支撑\n\n"
        f"{sql_summary}\n\n"
        "## 制度依据\n\n"
        f"{policy_fallback}\n\n"
        "## 执行与审批\n\n"
        f"审批：{approval_line}\n\n风险：{risk_line}"
        + _refs_block(compliance_snippets, cite_sources)
    )


def render_by_route(
    route_plan: str,
    *,
    question: str,
    sql_summary: str = "",
    sql_text: str = "",
    row_count: int = 0,
    preview: list[Any] | None = None,
    compliance_snippets: list[str] | None = None,
    requires_review: bool = False,
    settings: Settings,
    llm: ChatOpenAI | None = None,
    sources: list[dict[str, str]] | None = None,
    snippet_sources: list[dict[str, str]] | None = None,
) -> str:
    snippets = compliance_snippets or []
    cite = snippet_sources if snippet_sources else sources
    if route_plan == "data_only":
        return format_data_only(
            question=question,
            sql_summary=sql_summary,
            sql_text=sql_text,
            row_count=row_count,
            preview=preview,
            settings=settings,
            llm=llm,
            sources=sources,
        )
    if route_plan == "compliance_only":
        return format_policy_only(
            question=question,
            compliance_snippets=snippets,
            settings=settings,
            llm=llm,
            sources=cite,
        )
    return format_full_report(
        question=question,
        sql_summary=sql_summary,
        compliance_snippets=snippets,
        requires_review=requires_review,
        settings=settings,
        llm=llm,
        sources=sources,
        snippet_sources=snippet_sources,
    )
