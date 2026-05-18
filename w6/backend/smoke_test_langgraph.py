"""W6 LangGraph + HITL 一键冒烟测试。

使用方式：
    python smoke_test_langgraph.py --api http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

import httpx


def _pp(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def run(api: str) -> int:
    sid = f"smoke-{uuid.uuid4().hex[:8]}"
    base = api.rstrip("/")
    print(f"[smoke] session_id={sid}")
    with httpx.Client(timeout=300) as c:
        r = c.get(f"{base}/health")
        r.raise_for_status()
        print("[1/5] health ok:", _pp(r.json()))

        normal_q = "请先给出华东区上个月的预测准确率概览，再给我优化建议。"
        r = c.post(f"{base}/graph/chat", json={"session_id": sid, "message": normal_q})
        r.raise_for_status()
        normal = r.json()
        print("[2/5] graph chat(normal):", _pp({"status": normal.get("status"), "brief": normal.get("brief")}))

        force_q = "[force_approval] 请基于SOP规范给出库存调整建议，并等待人工审批。"
        r = c.post(f"{base}/graph/chat", json={"session_id": sid, "message": force_q})
        r.raise_for_status()
        paused = r.json()
        print("[3/5] graph chat(force approval):", _pp({"status": paused.get("status"), "interrupt": paused.get("interrupt")}))
        if paused.get("status") != "interrupted":
            print("[x] expected interrupted status but got:", paused.get("status"))
            return 2

        r = c.post(f"{base}/graph/resume", json={"session_id": sid, "decision": "approved"})
        r.raise_for_status()
        resumed = r.json()
        print("[4/5] graph resume(approved):", _pp({"status": resumed.get("status"), "output_head": str(resumed.get("output") or "")[:160]}))
        if resumed.get("status") != "completed":
            print("[x] expected completed status after resume")
            return 3

        r = c.get(f"{base}/graph/state", params={"session_id": sid})
        r.raise_for_status()
        state = r.json()
        print("[5/5] graph state:", _pp({"status": state.get("status"), "brief": state.get("brief")}))

    print("[done] smoke test passed")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://127.0.0.1:8000", help="FastAPI base url")
    args = p.parse_args()
    try:
        return run(args.api)
    except httpx.HTTPError as e:
        print(f"[x] HTTP error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
