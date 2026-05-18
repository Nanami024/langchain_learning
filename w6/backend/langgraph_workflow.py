"""LangGraph 多智能体工作流：数据分析 + 合规检索 + 人类审批。

目标：
1) 用 StateGraph 显式表达节点与条件路由，替代黑盒单体 Agent。
2) 支持“接力跑”：先查数据，再结合数据检索制度建议。
3) 在高风险建议场景触发 human-in-the-loop 中断，等待外部审批再继续。
"""

from __future__ import annotations

import json
import os
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from sop_hub.config import Settings
from sop_hub.openai_http import chat_openai_http_kwargs

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except Exception:  # pragma: no cover - 环境未安装时走内存降级
    SqliteSaver = None  # type: ignore[assignment]


RoutePlan = Literal["direct", "data_only", "compliance_only", "data_then_compliance"]
ApprovalDecision = Literal["approved", "rejected", "revise"]


class WorkflowState(TypedDict, total=False):
    session_id: str
    user_query: str
    intent: str
    route_plan: RoutePlan
    sql_result: str
    compliance_result: str
    requires_human_approval: bool
    approval_reason: str
    approval_decision: ApprovalDecision
    final_answer: str
    error: str
    node_trace: list[dict[str, str]]


def _llm(settings: Settings) -> ChatOpenAI:
    req_timeout = float(os.getenv("SOP_LLM_REQUEST_TIMEOUT", "120"))
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0,
        request_timeout=req_timeout,
        **chat_openai_http_kwargs(),
    )


def _append_trace(state: WorkflowState, node: str, detail: str) -> list[dict[str, str]]:
    rows = list(state.get("node_trace", []))
    rows.append({"node": node, "detail": detail})
    return rows


def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _risk_keywords() -> tuple[str, ...]:
    return (
        "削减",
        "砍掉",
        "停产",
        "下架",
        "关闭",
        "裁员",
        "高风险",
        "强制",
    )


def _is_high_risk(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    return any(k in s for k in _risk_keywords())


def _force_approval(query: str) -> bool:
    q = (query or "").strip().lower()
    return "[force_approval]" in q or "force_approval" in q


def _parse_route_json(raw: str, query: str) -> tuple[str, RoutePlan]:
    try:
        obj = json.loads(raw)
        intent = _safe_text(obj.get("intent")).strip() or "general"
        route = _safe_text(obj.get("route")).strip()
        if route in ("data_only", "compliance_only", "data_then_compliance", "direct"):
            return intent, route  # type: ignore[return-value]
    except json.JSONDecodeError:
        pass

    q = query.lower()
    has_data = any(k in q for k in ("准确率", "缺货率", "销量", "库存", "top", "上个月", "本季度"))
    has_policy = any(
        k in q
        for k in ("措施", "制度", "规范", "审批", "惩罚", "sop", "规则", "怎么办", "怎么做")
    )
    if has_data and has_policy:
        return "data+policy", "data_then_compliance"
    if has_data:
        return "data", "data_only"
    if has_policy:
        return "policy", "compliance_only"
    return "general", "direct"


@dataclass
class GraphRuntime:
    graph: Any
    checkpointer_mode: str
    checkpoint_path: str
    _exit_stack: ExitStack

    def invoke(self, session_id: str, message: str) -> dict[str, Any]:
        cfg = {"configurable": {"thread_id": session_id}}
        out = self.graph.invoke(
            {
                "session_id": session_id,
                "user_query": message,
                "node_trace": [],
            },
            config=cfg,
        )
        return _normalize_graph_result(out)

    def resume(self, session_id: str, decision: str) -> dict[str, Any]:
        cfg = {"configurable": {"thread_id": session_id}}
        out = self.graph.invoke(Command(resume=decision), config=cfg)
        return _normalize_graph_result(out)

    def get_state(self, session_id: str) -> dict[str, Any]:
        cfg = {"configurable": {"thread_id": session_id}}
        snapshot = self.graph.get_state(cfg)
        values = dict(snapshot.values) if snapshot and snapshot.values else {}
        return _normalize_graph_result(values)

    def close(self) -> None:
        self._exit_stack.close()


def _normalize_graph_result(state: dict[str, Any]) -> dict[str, Any]:
    interrupts = state.get("__interrupt__") or []
    if interrupts:
        first = interrupts[0]
        payload = getattr(first, "value", first)
        return {
            "status": "interrupted",
            "interrupt": payload,
            "output": "",
            "node_trace": state.get("node_trace", []),
            "state": state,
        }
    return {
        "status": "completed",
        "interrupt": None,
        "output": _safe_text(state.get("final_answer", "")).strip(),
        "node_trace": state.get("node_trace", []),
        "state": state,
    }


def build_graph_runtime(
    settings: Settings,
    *,
    sop_database_query_tool: Any,
    sop_document_search_tool: Any,
) -> GraphRuntime:
    planner_llm = _llm(settings)
    writer_llm = _llm(settings)

    def supervisor_node(state: WorkflowState) -> WorkflowState:
        query = _safe_text(state.get("user_query", "")).strip()
        if not query:
            return {
                "error": "用户问题为空",
                "final_answer": "错误：问题为空，请重新输入。",
                "node_trace": _append_trace(state, "supervisor_node", "empty query"),
            }

        prompt = (
            "你是 S&OP 流程主管。请只输出 JSON，字段包含：\n"
            '1) "intent": 字符串\n'
            '2) "route": 只能是 direct/data_only/compliance_only/data_then_compliance\n'
            "路由规则：\n"
            "- 仅问数据指标 => data_only\n"
            "- 仅问制度、条款、惩罚、流程 => compliance_only\n"
            "- 既问指标又问措施/怎么办 => data_then_compliance\n"
            "- 都不明显 => direct\n"
        )
        msg = planner_llm.invoke([SystemMessage(content=prompt), HumanMessage(content=query)])
        intent, route = _parse_route_json(_safe_text(msg.content), query)
        return {
            "intent": intent,
            "route_plan": route,
            "node_trace": _append_trace(state, "supervisor_node", f"route={route}, intent={intent}"),
        }

    def data_analyst_node(state: WorkflowState) -> WorkflowState:
        query = _safe_text(state.get("user_query", "")).strip()
        try:
            sql_out = sop_database_query_tool.invoke({"question": query})
        except Exception as e:
            sql_out = f"数据节点失败：{type(e).__name__}: {e}"
        sql_text = _safe_text(sql_out)
        return {
            "sql_result": sql_text,
            "node_trace": _append_trace(state, "data_analyst_node", "sql done"),
        }

    def compliance_expert_node(state: WorkflowState) -> WorkflowState:
        query = _safe_text(state.get("user_query", "")).strip()
        sql_result = _safe_text(state.get("sql_result", "")).strip()
        if sql_result:
            comp_query = (
                f"用户问题：{query}\n\n"
                f"已知数据结论：{sql_result}\n\n"
                "请检索与该数据异常/表现相关的 S&OP 规范、审批阈值与建议措施。"
            )
        else:
            comp_query = query
        try:
            comp_out = sop_document_search_tool.invoke({"query": comp_query})
        except Exception as e:
            comp_out = f"合规节点失败：{type(e).__name__}: {e}"
        comp_text = _safe_text(comp_out)
        need_approval = _is_high_risk(comp_text) or _force_approval(query)
        reason = "检测到潜在高风险建议，需人工审批后执行。"
        if _force_approval(query):
            reason = "检测到测试标记 [force_approval]，强制进入人工审批。"
        return {
            "compliance_result": comp_text,
            "requires_human_approval": need_approval,
            "approval_reason": reason if need_approval else "",
            "node_trace": _append_trace(
                state,
                "compliance_expert_node",
                "compliance done; approval="
                + ("required" if need_approval else "not_required"),
            ),
        }

    def approval_gate_node(state: WorkflowState) -> WorkflowState:
        payload = {
            "type": "approval_required",
            "reason": _safe_text(state.get("approval_reason", "")).strip(),
            "recommendation": _safe_text(state.get("compliance_result", "")).strip()[:2000],
            "hint": "请输入 approved/rejected/revise，或中文：同意/驳回/修改",
        }
        decision = _safe_text(interrupt(payload)).strip().lower()
        if decision in ("approved", "approve", "同意", "通过", "yes"):
            normalized: ApprovalDecision = "approved"
        elif decision in ("revise", "修改", "调整"):
            normalized = "revise"
        else:
            normalized = "rejected"
        return {
            "approval_decision": normalized,
            "node_trace": _append_trace(state, "approval_gate_node", f"decision={normalized}"),
        }

    def synthesis_node(state: WorkflowState) -> WorkflowState:
        if state.get("error"):
            return {
                "final_answer": _safe_text(state.get("final_answer", "")),
                "node_trace": _append_trace(state, "synthesis_node", "skip due to error"),
            }

        approval = _safe_text(state.get("approval_decision", "")).strip()
        if approval == "rejected":
            return {
                "final_answer": (
                    "已收到人工审批结果：驳回。建议暂不执行高风险调整。"
                    "请给出你希望的修改方向（例如调整幅度、区域范围或审批条件），我将重新生成方案。"
                ),
                "node_trace": _append_trace(state, "synthesis_node", "rejected response"),
            }

        query = _safe_text(state.get("user_query", "")).strip()
        sql_result = _safe_text(state.get("sql_result", "")).strip()
        comp_result = _safe_text(state.get("compliance_result", "")).strip()
        route = _safe_text(state.get("route_plan", "direct")).strip()
        approval_note = (
            "人工审批结果：同意/修改后继续执行。"
            if approval in ("approved", "revise")
            else "人工审批：本轮未触发。"
        )
        prompt = (
            "你是企业 S&OP 总结官。请输出结构化中文结论，包含：\n"
            "1) 关键事实（来自数据）\n"
            "2) 规则依据（来自制度检索）\n"
            "3) 建议动作（可执行且可追踪）\n"
            "4) 风险与审批说明\n"
            "禁止编造不存在的数据或条款。"
        )
        human = (
            f"用户问题：{query}\n\n"
            f"路由：{route}\n"
            f"数据节点结果：{sql_result or '（无）'}\n\n"
            f"合规节点结果：{comp_result or '（无）'}\n\n"
            f"{approval_note}"
        )
        out = writer_llm.invoke([SystemMessage(content=prompt), HumanMessage(content=human)])
        return {
            "final_answer": _safe_text(out.content).strip(),
            "node_trace": _append_trace(state, "synthesis_node", "final answer generated"),
        }

    def route_from_supervisor(state: WorkflowState) -> str:
        route = _safe_text(state.get("route_plan", "direct"))
        if route == "data_only":
            return "data_analyst_node"
        if route == "compliance_only":
            return "compliance_expert_node"
        if route == "data_then_compliance":
            return "data_analyst_node"
        return "synthesis_node"

    def route_after_data(state: WorkflowState) -> str:
        route = _safe_text(state.get("route_plan", "direct"))
        if route == "data_then_compliance":
            return "compliance_expert_node"
        return "synthesis_node"

    def route_after_compliance(state: WorkflowState) -> str:
        need_approval = bool(state.get("requires_human_approval", False))
        return "approval_gate_node" if need_approval else "synthesis_node"

    g = StateGraph(WorkflowState)
    g.add_node("supervisor_node", supervisor_node)
    g.add_node("data_analyst_node", data_analyst_node)
    g.add_node("compliance_expert_node", compliance_expert_node)
    g.add_node("approval_gate_node", approval_gate_node)
    g.add_node("synthesis_node", synthesis_node)

    g.add_edge(START, "supervisor_node")
    g.add_conditional_edges(
        "supervisor_node",
        route_from_supervisor,
        {
            "data_analyst_node": "data_analyst_node",
            "compliance_expert_node": "compliance_expert_node",
            "synthesis_node": "synthesis_node",
        },
    )
    g.add_conditional_edges(
        "data_analyst_node",
        route_after_data,
        {
            "compliance_expert_node": "compliance_expert_node",
            "synthesis_node": "synthesis_node",
        },
    )
    g.add_conditional_edges(
        "compliance_expert_node",
        route_after_compliance,
        {
            "approval_gate_node": "approval_gate_node",
            "synthesis_node": "synthesis_node",
        },
    )
    g.add_edge("approval_gate_node", "synthesis_node")
    g.add_edge("synthesis_node", END)

    checkpointer_mode = "memory"
    checkpoint_path = ""
    exit_stack = ExitStack()
    checkpointer: Any = MemorySaver()

    raw_path = os.getenv("SOP_LANGGRAPH_CHECKPOINT_PATH", "").strip()
    if raw_path and SqliteSaver is not None:
        db_path = Path(raw_path)
        if not db_path.is_absolute():
            db_path = (Path(__file__).resolve().parent / db_path).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(db_path)
        checkpointer = exit_stack.enter_context(SqliteSaver.from_conn_string(checkpoint_path))
        checkpointer_mode = "sqlite"
    elif SqliteSaver is not None:
        default_path = (Path(__file__).resolve().parent / "runtime" / "langgraph_checkpoints.sqlite").resolve()
        default_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(default_path)
        checkpointer = exit_stack.enter_context(SqliteSaver.from_conn_string(checkpoint_path))
        checkpointer_mode = "sqlite"

    app = g.compile(checkpointer=checkpointer)
    return GraphRuntime(
        graph=app,
        checkpointer_mode=checkpointer_mode,
        checkpoint_path=checkpoint_path,
        _exit_stack=exit_stack,
    )
