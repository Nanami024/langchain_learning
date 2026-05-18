from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from analytics import DataAnalyst
from compliance import ComplianceRetriever
from config import Settings
from query_router import RouteDecision, RoutePlan, resolve_route
from report_synthesis import build_synthesis_llm, render_by_route, revise_report_draft_with_feedback


class SopState(TypedDict, total=False):
    session_id: str
    user_query: str
    context_query: str
    resolved_query: str
    context_used: bool
    intent: str
    route_plan: RoutePlan
    route_reason: str
    route_confidence: float
    route_source: str
    scope_score: float
    sql_summary: str
    sql_result: dict[str, Any]
    sql_text: str
    compliance_summary: str
    compliance_snippets: list[str]
    compliance_sources: list[dict[str, str]]
    requires_human_approval: bool
    approval_reason: str
    draft_answer: str
    human_feedback: str
    review_round: int
    review_action: str
    approval_decision: str
    final_answer: str
    sources: list[dict[str, str]]
    node_trace: list[dict[str, str]]


def _trace(state: SopState, node: str, detail: str) -> list[dict[str, str]]:
    rows = list(state.get("node_trace", []))
    rows.append({"node": node, "detail": detail})
    return rows


def _normalize_review_input(payload: Any) -> tuple[str, str]:
    if isinstance(payload, dict):
        action = str(payload.get("action") or "").strip().lower()
        feedback = str(payload.get("feedback") or "").strip()
    else:
        raw = str(payload or "").strip()
        if raw.lower().startswith("feedback:"):
            action = "feedback"
            feedback = raw.split(":", 1)[1].strip()
        else:
            action = raw.lower()
            feedback = ""

    if action in ("approved", "approve", "同意", "通过"):
        return "approved", feedback
    if action in ("rejected", "reject", "驳回", "不用", "拒绝"):
        return "rejected", feedback
    if action in ("feedback", "revise", "修改", "意见", "继续"):
        return "feedback", feedback
    # 默认按“继续给意见”处理，避免误完成。
    return "feedback", feedback


def _merge_sources(existing: list[dict[str, str]] | None, incoming: list[dict[str, str]] | None) -> list[dict[str, str]]:
    rows = list(existing or []) + list(incoming or [])
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        tp = str((row or {}).get("type", "")).strip()
        nm = str((row or {}).get("name", "")).strip()
        if not tp and not nm:
            continue
        key = (tp, nm)
        if key in seen:
            continue
        seen.add(key)
        out.append({"type": tp or "unknown", "name": nm or "unknown"})
    return out


@dataclass
class GraphRuntime:
    graph: Any
    checkpoint_path: str = "memory"

    def invoke(self, session_id: str, message: str, context_query: str = "") -> dict[str, Any]:
        cfg = {"configurable": {"thread_id": session_id}}
        out = self.graph.invoke(
            {
                "session_id": session_id,
                "user_query": message,
                "context_query": context_query,
                "node_trace": [],
            },
            config=cfg,
        )
        return self._normalize(out)

    def resume(self, session_id: str, decision: Any) -> dict[str, Any]:
        cfg = {"configurable": {"thread_id": session_id}}
        out = self.graph.invoke(Command(resume=decision), config=cfg)
        return self._normalize(out)

    def get_state(self, session_id: str) -> dict[str, Any]:
        cfg = {"configurable": {"thread_id": session_id}}
        snap = self.graph.get_state(cfg)
        vals = dict(snap.values) if snap and snap.values else {}
        return self._normalize(vals)

    def close(self) -> None:
        return None

    def _normalize(self, state: dict[str, Any]) -> dict[str, Any]:
        interrupts = state.get("__interrupt__") or []
        if interrupts:
            intr = interrupts[0]
            payload = getattr(intr, "value", intr)
            return {
                "status": "interrupted",
                "output": "",
                "interrupt": payload,
                "node_trace": state.get("node_trace", []),
                "state": state,
            }
        return {
            "status": "completed",
            "output": str(state.get("final_answer", "")),
            "interrupt": None,
            "node_trace": state.get("node_trace", []),
            "state": state,
        }


def build_graph_runtime() -> GraphRuntime:
    settings = Settings()
    analyst = DataAnalyst()
    retriever = ComplianceRetriever()
    synth_llm = build_synthesis_llm(settings)

    def _latest_user_from_context(ctx: str) -> str:
        for line in reversed(str(ctx or "").splitlines()):
            if line.startswith("用户:"):
                return line.split("用户:", 1)[1].strip()
        return ""

    def _query_needs_prior_turn(query: str) -> bool:
        """短追问或含指代词时，需拼接 context_query 才能解析实体（如「他们两个」→ 上文品类）。"""
        q = (query or "").strip()
        if not q:
            return False
        if len(q) <= 10:
            return True
        anaphora = (
            "他们",
            "二者",
            "这两",
            "两边",
            "他俩",
            "这俩",
            "那俩",
            "上面两个",
            "刚才说的",
            "前述",
            "这两种",
            "这两类",
            "两个品类",
        )
        if any(a in q for a in anaphora):
            return True
        if ("两个" in q or "俩" in q) and any(
            w in q for w in ("级别", "评级", "等级", "谁", "哪", "对比", "差异", "还是", "多少")
        ):
            return True
        return False

    def _needs_context(query: str) -> bool:
        return _query_needs_prior_turn(query)

    def _is_followup_action_query(query: str) -> bool:
        q = (query or "").strip()
        return len(q) <= 12 or _query_needs_prior_turn(q)

    def supervisor_node(state: SopState) -> SopState:
        query = str(state.get("user_query", "") or "")
        ctx = str(state.get("context_query", "") or "")
        use_ctx = bool(ctx.strip()) and _needs_context(query)
        latest_user_ctx = _latest_user_from_context(ctx)
        base_ctx = latest_user_ctx or ctx
        resolved_query = f"{base_ctx}\n当前追问：{query}" if use_ctx else query
        decision = resolve_route(resolved_query, analyst=analyst, retriever=retriever, settings=settings)
        prev_route = str(state.get("route_plan", "") or "").strip()

        # 追问继承：短句在域内延续上一轮能力组合，而不是重新用阈值判域外。
        if use_ctx and decision.route_plan == "out_of_scope" and _is_followup_action_query(query):
            if prev_route in ("data_only", "data_then_compliance"):
                decision = RouteDecision(
                    intent="data+policy",
                    route_plan="data_then_compliance",
                    confidence=max(decision.confidence, 0.2),
                    reason="follow-up inherit previous data route",
                    needs_data=True,
                    needs_policy=True,
                    needs_guidance=True,
                    in_domain=True,
                    data_score=decision.data_score,
                    policy_score=decision.policy_score,
                    route_source="follow-up-inherit",
                )
            elif prev_route == "compliance_only":
                decision = RouteDecision(
                    intent="policy",
                    route_plan="compliance_only",
                    confidence=max(decision.confidence, 0.2),
                    reason="follow-up inherit previous compliance route",
                    needs_data=False,
                    needs_policy=True,
                    needs_guidance=False,
                    in_domain=True,
                    data_score=decision.data_score,
                    policy_score=decision.policy_score,
                    route_source="follow-up-inherit",
                )

        return {
            "intent": decision.intent,
            "route_plan": decision.route_plan,
            "route_reason": decision.reason,
            "route_confidence": decision.confidence,
            "scope_score": max(decision.data_score, decision.policy_score),
            "resolved_query": resolved_query,
            "context_used": use_ctx,
            "route_source": decision.route_source,
            "node_trace": _trace(
                state,
                "supervisor_node",
                (
                    f"route={decision.route_plan}; source={decision.route_source}; "
                    f"hybrid(data={decision.data_score:.2f},policy={decision.policy_score:.2f}); "
                    f"slots=data={decision.needs_data},policy={decision.needs_policy},guidance={decision.needs_guidance}; "
                    f"slot_route={decision.slot_route}; llm_route={decision.llm_route}; llm_used={decision.llm_used}; "
                    f"conf={decision.confidence:.2f}; context_used={use_ctx}"
                ),
            ),
        }

    def data_analyst_node(state: SopState) -> SopState:
        rq = str(state.get("resolved_query") or state.get("user_query", ""))
        res = analyst.analyze(rq)
        return {
            "sql_summary": res.summary,
            "sql_result": res.metrics,
            "sql_text": res.sql,
            "sources": _merge_sources(state.get("sources"), res.sources),
            "node_trace": _trace(state, "data_analyst_node", "sql analyzed"),
        }

    def compliance_expert_node(state: SopState) -> SopState:
        q = str(state.get("resolved_query") or state.get("user_query", ""))
        sql_summary = str(state.get("sql_summary", "") or "")
        r = retriever.search(q, context=sql_summary, analyst=analyst)
        merged = r["merged"]
        comp_srcs = list(r.get("sources") or [])
        return {
            "compliance_summary": merged,
            "compliance_snippets": r["snippets"],
            "compliance_sources": comp_srcs,
            "requires_human_approval": False,
            "approval_reason": "",
            "sources": _merge_sources(state.get("sources"), comp_srcs),
            "node_trace": _trace(state, "compliance_expert_node", "compliance retrieved"),
        }

    def out_of_scope_node(state: SopState) -> SopState:
        msg = (
            "当前问题不在知识库覆盖范围内，暂不回答。"
            "本系统只基于企业数据库中的两类数据：PDF 制度文档与 CSV 业务数据。"
            "请补充与这些材料直接相关的问题后重试。"
        )
        return {
            "approval_decision": "out_of_scope",
            "final_answer": msg,
            "node_trace": _trace(state, "out_of_scope_node", "blocked by corpus scope"),
        }

    def synthesis_draft_node(state: SopState) -> SopState:
        route = str(state.get("route_plan", "data_then_compliance"))
        sql_summary = str(state.get("sql_summary", "") or "无数据分析结果")
        comp = str(state.get("compliance_summary", "") or "未检索到与问题直接相关的制度条款。")
        q = str(state.get("resolved_query") or state.get("user_query", ""))
        snippets = list(state.get("compliance_snippets", []))
        need_review, reason = retriever.should_require_review(q, sql_summary, snippets, route)
        result = state.get("sql_result") or {}
        preview = result.get("preview") if isinstance(result, dict) else None
        row_count = int(result.get("row_count") or 0) if isinstance(result, dict) else 0
        draft = render_by_route(
            route,
            question=q,
            sql_summary=sql_summary,
            sql_text=str(state.get("sql_text", "") or ""),
            row_count=row_count,
            preview=preview if isinstance(preview, list) else None,
            compliance_snippets=snippets,
            requires_review=need_review,
            settings=settings,
            llm=synth_llm,
            sources=list(state.get("sources") or []),
            snippet_sources=list(state.get("compliance_sources") or []),
        )
        return {
            "draft_answer": draft,
            "review_round": int(state.get("review_round", 0)),
            "requires_human_approval": need_review,
            "approval_reason": reason,
            "node_trace": _trace(
                state,
                "synthesis_draft_node",
                f"route={route}; review={need_review}; llm={synth_llm is not None}",
            ),
        }

    def data_only_finalize_node(state: SopState) -> SopState:
        q = str(state.get("resolved_query") or state.get("user_query", ""))
        summary = str(state.get("sql_summary", "")).strip() or "未查询到有效数据。"
        result = state.get("sql_result") or {}
        preview = result.get("preview") if isinstance(result, dict) else None
        row_count = int(result.get("row_count") or 0) if isinstance(result, dict) else 0
        sql_text = str(state.get("sql_text", "")).strip()
        final = render_by_route(
            "data_only",
            question=q,
            sql_summary=summary,
            sql_text=sql_text,
            row_count=row_count,
            preview=preview if isinstance(preview, list) else None,
            settings=settings,
            llm=None,
            sources=list(state.get("sources") or []),
        )
        return {
            "approval_decision": "not_required",
            "requires_human_approval": False,
            "final_answer": final,
            "node_trace": _trace(state, "data_only_finalize_node", "finalized data-only response"),
        }

    def compliance_only_finalize_node(state: SopState) -> SopState:
        q = str(state.get("resolved_query") or state.get("user_query", ""))
        snippets = list(state.get("compliance_snippets", []))
        need_review, reason = retriever.should_require_review(q, "", snippets, "compliance_only")
        final = render_by_route(
            "compliance_only",
            question=q,
            compliance_snippets=snippets,
            settings=settings,
            llm=synth_llm,
            sources=list(state.get("sources") or []),
            snippet_sources=list(state.get("compliance_sources") or []),
        )
        out: SopState = {
            "requires_human_approval": need_review,
            "approval_reason": reason,
            "node_trace": _trace(state, "compliance_only_finalize_node", f"review={need_review}"),
        }
        if need_review:
            out["draft_answer"] = final
        else:
            out["approval_decision"] = "not_required"
            out["final_answer"] = final
        return out

    def route_after_compliance_only_finalize(state: SopState) -> str:
        return "review_gate_node" if bool(state.get("requires_human_approval", False)) else "__end__"

    def review_gate_node(state: SopState) -> SopState:
        round_n = int(state.get("review_round", 0)) + 1
        payload = {
            "type": "approval_required",
            "round": round_n,
            "reason": state.get("approval_reason", "触发审批"),
            "draft_answer": state.get("draft_answer", "")[:2500],
        }
        action, feedback = _normalize_review_input(interrupt(payload))
        return {
            "review_round": round_n,
            "review_action": action,
            "human_feedback": feedback,
            "node_trace": _trace(state, "review_gate_node", f"action={action}; round={round_n}"),
        }

    def revise_with_feedback_node(state: SopState) -> SopState:
        draft = str(state.get("draft_answer", "") or "")
        fb = str(state.get("human_feedback", "") or "").strip() or "请进一步细化执行范围与风险控制。"
        q = str(state.get("resolved_query") or state.get("user_query", "") or "")
        revised = revise_report_draft_with_feedback(
            synth_llm,
            draft_markdown=draft,
            human_feedback=fb,
            user_question=q,
            settings=settings,
        )
        if not revised.strip():
            revised = (
                f"{draft}\n\n"
                "## 人工反馈后的修订\n"
                f"- 人工意见：{fb}\n"
                "- AI修订：（模型不可用或未返回正文）请缩小执行范围、补充风险与复核节奏；可稍后重试或检查 OPENAI 配置。"
            )
        return {
            "draft_answer": revised,
            "node_trace": _trace(state, "revise_with_feedback_node", "draft revised via llm or fallback"),
        }

    def finalize_node(state: SopState) -> SopState:
        final = str(state.get("draft_answer", "")).strip()
        if state.get("requires_human_approval", False) and "审批：同意执行" not in final:
            final += "\n\n（已通过人工审批流程）"
        return {"approval_decision": "approved", "final_answer": final, "node_trace": _trace(state, "finalize_node", "approved/finalized")}

    def reject_finalize_node(state: SopState) -> SopState:
        final = (
            "## S&OP 决策报告（已驳回）\n\n"
            "审批结论：驳回。\n"
            "建议：本轮方案不执行，请补充更保守的范围、明确风险上限后再提交。"
        )
        return {"approval_decision": "rejected", "final_answer": final, "node_trace": _trace(state, "reject_finalize_node", "rejected/finalized")}

    def route_supervisor(state: SopState) -> str:
        route = state.get("route_plan", "out_of_scope")
        if route == "out_of_scope":
            return "out_of_scope_node"
        if route == "data_only":
            return "data_analyst_node"
        if route == "compliance_only":
            return "compliance_expert_node"
        if route == "data_then_compliance":
            return "data_analyst_node"
        return "out_of_scope_node"

    def route_after_data(state: SopState) -> str:
        return "compliance_expert_node" if state.get("route_plan") == "data_then_compliance" else "data_only_finalize_node"

    def route_after_compliance(state: SopState) -> str:
        if state.get("route_plan") == "compliance_only":
            return "compliance_only_finalize_node"
        return "synthesis_draft_node"

    def route_after_draft(state: SopState) -> str:
        return "review_gate_node" if bool(state.get("requires_human_approval", False)) else "finalize_node"

    def route_after_review(state: SopState) -> str:
        action = str(state.get("review_action", "feedback"))
        if action == "approved":
            return "finalize_node"
        if action == "rejected":
            return "reject_finalize_node"
        return "revise_with_feedback_node"

    g = StateGraph(SopState)
    g.add_node("supervisor_node", supervisor_node)
    g.add_node("out_of_scope_node", out_of_scope_node)
    g.add_node("data_analyst_node", data_analyst_node)
    g.add_node("compliance_expert_node", compliance_expert_node)
    g.add_node("synthesis_draft_node", synthesis_draft_node)
    g.add_node("data_only_finalize_node", data_only_finalize_node)
    g.add_node("compliance_only_finalize_node", compliance_only_finalize_node)
    g.add_node("review_gate_node", review_gate_node)
    g.add_node("revise_with_feedback_node", revise_with_feedback_node)
    g.add_node("finalize_node", finalize_node)
    g.add_node("reject_finalize_node", reject_finalize_node)
    g.add_edge(START, "supervisor_node")
    g.add_conditional_edges(
        "supervisor_node",
        route_supervisor,
        {
            "out_of_scope_node": "out_of_scope_node",
            "data_analyst_node": "data_analyst_node",
            "compliance_expert_node": "compliance_expert_node",
        },
    )
    g.add_conditional_edges(
        "data_analyst_node",
        route_after_data,
        {"compliance_expert_node": "compliance_expert_node", "data_only_finalize_node": "data_only_finalize_node"},
    )
    g.add_conditional_edges(
        "compliance_expert_node",
        route_after_compliance,
        {
            "synthesis_draft_node": "synthesis_draft_node",
            "compliance_only_finalize_node": "compliance_only_finalize_node",
        },
    )
    g.add_conditional_edges(
        "synthesis_draft_node",
        route_after_draft,
        {"review_gate_node": "review_gate_node", "finalize_node": "finalize_node"},
    )
    g.add_conditional_edges(
        "review_gate_node",
        route_after_review,
        {
            "finalize_node": "finalize_node",
            "reject_finalize_node": "reject_finalize_node",
            "revise_with_feedback_node": "revise_with_feedback_node",
        },
    )
    g.add_edge("revise_with_feedback_node", "review_gate_node")
    g.add_edge("out_of_scope_node", END)
    g.add_edge("data_only_finalize_node", END)
    g.add_conditional_edges(
        "compliance_only_finalize_node",
        route_after_compliance_only_finalize,
        {"review_gate_node": "review_gate_node", "__end__": END},
    )
    g.add_edge("finalize_node", END)
    g.add_edge("reject_finalize_node", END)

    saver = MemorySaver()
    app = g.compile(checkpointer=saver)
    return GraphRuntime(graph=app, checkpoint_path="memory")
