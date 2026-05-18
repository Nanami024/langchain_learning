from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from corpus_lexicon import CorpusLexicon, get_lexicon
from retrieval_core import tokenize

if TYPE_CHECKING:
    from analytics import DataAnalyst
    from compliance import ComplianceRetriever
    from config import Settings

RoutePlan = Literal["out_of_scope", "data_only", "compliance_only", "data_then_compliance"]

_DATA_ASK_RE = re.compile(
    r"(多少|均值|平均|占比|同比|环比|统计|查询|趋势|是多少|"
    r"几(?!个等级|档位|种等级|级标准|级\)|档\))|"
    r"几个(?!等级|档|种等级))"
)

_POLICY_ASK_RE = re.compile(r"(制度|标准|条款|流程|审批|依据|评级|等级|红线)")

_ROUTE_TO_INTENT: dict[RoutePlan, str] = {
    "out_of_scope": "out_of_scope",
    "data_only": "data",
    "compliance_only": "policy",
    "data_then_compliance": "data+policy",
}

@dataclass(frozen=True)
class RouteDecision:
    intent: str
    route_plan: RoutePlan
    confidence: float
    reason: str
    needs_data: bool
    needs_policy: bool
    needs_guidance: bool
    in_domain: bool
    data_score: float
    policy_score: float
    route_source: str = "hybrid+corpus"
    slot_route: RoutePlan | None = None
    llm_route: RoutePlan | None = None
    llm_used: bool = False


@dataclass(frozen=True)
class _SlotSignals:
    needs_data: bool
    needs_policy: bool
    needs_guidance: bool
    in_domain: bool
    data_overlap: float
    policy_overlap: float


@dataclass(frozen=True)
class _HybridSignals:
    data_score: float
    policy_score: float
    data_snippets: tuple[str, ...]
    policy_snippets: tuple[str, ...]


def _policy_structure_definition_query(query: str) -> bool:
    """纯问制度里分几档、有哪些等级等，不应跑业务库聚合。"""
    q = (query or "").strip()
    if not q:
        return False
    return bool(
        re.search(
            r"(几个等级|哪几[级档]|分几[级档]|共有几[级档]|都有哪些等级|一共几级|"
            r"等级划分|如何分级|分级标准|评级标准|考核.*几档|分档规则|档位|"
            r"介绍.{0,80}等级|说明.{0,80}等级|简述.{0,80}等级|讲讲.{0,80}等级|描述.{0,80}等级|"
            r"什么是.{0,40}等级|等级是什么|有哪些等级|"
            r"预测准确率.{0,60}等级|准确率.{0,40}等级|指标.{0,40}等级)",
            q,
        )
    )


def _explicit_business_database_lookup(query: str) -> bool:
    """用户明确要求从业务库/数据表取数（与「数据库概论」类问题区分）。"""
    q = (query or "").strip()
    if not q:
        return False
    tail = r"(?:数据库|数据表|mysql)(?!范式|原理|概念|设计|理论|基础教程)"
    if re.search(
        rf"(?:查看|查询|统计|导出|拉取|读取|从).{{0,8}}(?:业务)?{tail}",
        q,
        re.I,
    ):
        return True
    if re.search(
        rf"(?:业务)?{tail}.{{0,14}}(?:里|中|的).{{0,8}}(?:记录|数据|各|每|分别|哪个|多少|列表)",
        q,
        re.I,
    ):
        return True
    return False


def _token_overlap_ratio(query: str, vocabulary: frozenset[str]) -> float:
    q_tokens = set(tokenize(query))
    if not q_tokens or not vocabulary:
        return 0.0
    return len(q_tokens.intersection(vocabulary)) / len(q_tokens)


def _hybrid_signals(query: str, *, analyst: DataAnalyst, retriever: ComplianceRetriever) -> _HybridSignals:
    data_score, _, _, _ = analyst.hybrid_relevance(query, final_k=1)
    policy_score, _, _, _ = retriever.hybrid_relevance(query, final_k=1)
    return _HybridSignals(
        data_score=data_score,
        policy_score=policy_score,
        data_snippets=(),
        policy_snippets=(),
    )


def _slot_signals(
    query: str,
    *,
    hybrid: _HybridSignals,
    settings: Settings,
    lexicon: CorpusLexicon,
) -> _SlotSignals:
    data_overlap = _token_overlap_ratio(query, lexicon.data_terms)
    policy_overlap = _token_overlap_ratio(query, lexicon.policy_terms)
    asks_data_fact = bool(_DATA_ASK_RE.search(query))
    asks_policy_fact = bool(_POLICY_ASK_RE.search(query))

    needs_data = (
        hybrid.data_score >= settings.route_min_data_score * 0.4
        or data_overlap >= 0.06
    )
    policy_dominant = hybrid.policy_score >= hybrid.data_score + 0.08
    needs_policy = (
        (policy_overlap >= 0.12 and policy_overlap >= data_overlap + 0.03)
        or (hybrid.policy_score >= settings.route_min_policy_score * 1.2 and policy_dominant)
    )
    if asks_policy_fact and hybrid.policy_score >= 0.08:
        needs_policy = True
    needs_guidance = needs_data and needs_policy and hybrid.policy_score >= hybrid.data_score * 0.9
    if asks_data_fact and not asks_policy_fact:
        needs_policy = False
        needs_guidance = False

    in_domain = (
        hybrid.data_score >= settings.scope_min_score * 0.4
        or hybrid.policy_score >= settings.scope_min_score * 0.4
        or data_overlap >= 0.04
        or policy_overlap >= 0.04
    )
    if _policy_structure_definition_query(query):
        needs_data = False
        needs_policy = True
        needs_guidance = False

    if _explicit_business_database_lookup(query):
        needs_data = True
        if needs_policy:
            needs_guidance = hybrid.policy_score >= hybrid.data_score * 0.9

    return _SlotSignals(
        needs_data=needs_data,
        needs_policy=needs_policy,
        needs_guidance=needs_guidance,
        in_domain=in_domain,
        data_overlap=data_overlap,
        policy_overlap=policy_overlap,
    )


def _route_from_slots(slots: _SlotSignals, hybrid: _HybridSignals) -> RoutePlan:
    if not slots.in_domain:
        return "out_of_scope"
    if slots.needs_data and slots.needs_policy:
        return "data_then_compliance"
    if slots.needs_data and not slots.needs_policy:
        return "data_only"
    if slots.needs_policy and not slots.needs_data:
        return "compliance_only"
    if hybrid.data_score >= hybrid.policy_score:
        return "data_only" if hybrid.data_score >= 0.06 else "out_of_scope"
    return "compliance_only" if hybrid.policy_score >= 0.06 else "out_of_scope"


def _build_route_llm(settings: Settings) -> ChatOpenAI | None:
    if not settings.route_use_llm or not settings.llm_api_key or not settings.llm_base_url:
        return None
    try:
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=0,
            request_timeout=settings.sql_timeout_seconds,
        )
    except Exception:
        return None


def _parse_llm_route(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _llm_route_judge(
    query: str,
    *,
    hybrid: _HybridSignals,
    slots: _SlotSignals,
    slot_route: RoutePlan,
    settings: Settings,
    llm: ChatOpenAI,
) -> tuple[RoutePlan | None, float, str]:
    prompt = (
        "你是 S&OP 企业助手的路由裁判。已做 BM25+向量混合检索。\n"
        "输出路径：out_of_scope | data_only | compliance_only | data_then_compliance\n"
        "- 只查数 → data_only\n"
        "- 只问制度/标准/条款 → compliance_only\n"
        "- 既要数据又要分析建议 → data_then_compliance\n"
        "只输出 JSON："
        '{"route_plan":"...","confidence":0-1,"reason":"..."}'
    )
    user = json.dumps(
        {
            "query": query,
            "hybrid_data_score": hybrid.data_score,
            "hybrid_policy_score": hybrid.policy_score,
            "data_snippets": [],
            "policy_snippets": [],
            "corpus_slot_route": slot_route,
            "corpus_needs_data": slots.needs_data,
            "corpus_needs_policy": slots.needs_policy,
        },
        ensure_ascii=False,
    )
    try:
        msg = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=user[:6000])])
        obj = _parse_llm_route(str(msg.content or ""))
        if not obj:
            return None, 0.0, "llm parse failed"
        route = str(obj.get("route_plan", "")).strip()
        if route not in _ROUTE_TO_INTENT:
            return None, 0.0, "llm invalid route_plan"
        conf = float(obj.get("confidence", 0.0) or 0.0)
        return route, min(1.0, max(0.0, conf)), str(obj.get("reason", "") or "llm route judge")  # type: ignore[return-value]
    except Exception as exc:
        return None, 0.0, f"llm error: {exc}"


def _fuse_routes(
    *,
    slot_route: RoutePlan,
    llm_route: RoutePlan | None,
    llm_conf: float,
    llm_reason: str,
    slots: _SlotSignals,
    hybrid: _HybridSignals,
    settings: Settings,
    llm_used: bool,
) -> tuple[RoutePlan, float, str, str]:
    if not llm_used or llm_route is None:
        conf = max(hybrid.data_score, hybrid.policy_score, 0.12)
        return slot_route, conf, f"hybrid+corpus → {slot_route}", "hybrid+corpus"

    # 已标「要查业务库」，即使槽位或 LLM 偏制度，也不能只走 compliance_only
    if slots.needs_data and llm_route == "compliance_only" and slot_route == "compliance_only":
        if slots.needs_policy:
            conf = min(1.0, max(llm_conf, hybrid.data_score, hybrid.policy_score, 0.55))
            return (
                "data_then_compliance",
                conf,
                "hybrid+corpus+llm (guard: 明确查库 + 等级/制度口径 → data_then_compliance)",
                "hybrid+corpus+llm",
            )
        conf = min(1.0, max(llm_conf, hybrid.data_score, 0.55))
        return (
            "data_only",
            conf,
            "hybrid+corpus+llm (guard: 明确查库 → data_only)",
            "hybrid+corpus+llm",
        )

    if slot_route == "data_then_compliance" and llm_route == "compliance_only" and slots.needs_data:
        conf = min(1.0, max(llm_conf, hybrid.data_score) + 0.08)
        return (
            "data_then_compliance",
            conf,
            f"hybrid+corpus+llm (guard mixed route; llm wanted {llm_route})",
            "hybrid+corpus+llm",
        )

    if slot_route == llm_route:
        conf = min(1.0, max(llm_conf, hybrid.data_score, hybrid.policy_score) + 0.1)
        return slot_route, conf, f"hybrid+corpus+llm agree → {slot_route}; {llm_reason}", "hybrid+corpus+llm"

    if llm_conf >= settings.route_llm_min_confidence:
        return llm_route, llm_conf, f"hybrid+corpus+llm override → {llm_route}; {llm_reason}", "hybrid+corpus+llm"

    conf = max(hybrid.data_score, hybrid.policy_score, 0.1)
    return slot_route, conf, f"hybrid+corpus (llm low-conf); {llm_reason}", "hybrid+corpus+llm"


def _skip_route_llm(
    slot_route: RoutePlan,
    hybrid: _HybridSignals,
    slots: _SlotSignals,
    settings: Settings,
) -> bool:
    if not settings.route_use_llm or not settings.route_skip_llm_when_clear:
        return not settings.route_use_llm
    top = max(hybrid.data_score, hybrid.policy_score)
    if top < 0.08:
        return False
    if slot_route == "data_only" and not slots.needs_policy:
        return True
    if slot_route == "compliance_only" and not slots.needs_data:
        return True
    if slot_route == "data_then_compliance" and slots.needs_data and slots.needs_policy:
        return True
    if slot_route == "out_of_scope" and top < settings.scope_min_score * 0.55:
        return True
    return False


def resolve_route(
    query: str,
    *,
    analyst: DataAnalyst,
    retriever: ComplianceRetriever,
    settings: Settings,
) -> RouteDecision:
    q = (query or "").strip()
    lexicon = get_lexicon(analyst, retriever)
    hybrid = _hybrid_signals(q, analyst=analyst, retriever=retriever)
    slots = _slot_signals(q, hybrid=hybrid, settings=settings, lexicon=lexicon)
    slot_route = _route_from_slots(slots, hybrid)

    llm_route: RoutePlan | None = None
    llm_conf = 0.0
    llm_reason = ""
    llm_used = False
    if not _skip_route_llm(slot_route, hybrid, slots, settings):
        llm = _build_route_llm(settings)
        if llm is not None:
            llm_route, llm_conf, llm_reason = _llm_route_judge(
                q, hybrid=hybrid, slots=slots, slot_route=slot_route, settings=settings, llm=llm
            )
            llm_used = llm_route is not None

    final_route, confidence, reason, route_source = _fuse_routes(
        slot_route=slot_route,
        llm_route=llm_route,
        llm_conf=llm_conf,
        llm_reason=llm_reason,
        slots=slots,
        hybrid=hybrid,
        settings=settings,
        llm_used=llm_used,
    )

    return RouteDecision(
        intent=_ROUTE_TO_INTENT[final_route],
        route_plan=final_route,
        confidence=confidence,
        reason=reason,
        needs_data=slots.needs_data,
        needs_policy=slots.needs_policy,
        needs_guidance=slots.needs_guidance,
        in_domain=slots.in_domain and final_route != "out_of_scope",
        data_score=hybrid.data_score,
        policy_score=hybrid.policy_score,
        route_source=route_source,
        slot_route=slot_route,
        llm_route=llm_route,
        llm_used=llm_used,
    )
