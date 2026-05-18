from __future__ import annotations

"""全链路自检：需在 final_enterprise/backend 下使用与 uvicorn 相同的 venv 运行：
   pip install -r requirements.txt
   python smoke_test.py
"""

import time
import uuid

from fastapi.testclient import TestClient

from main import app


def _wait_ready(c: TestClient, *, timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    last_status: int | None = None
    while time.time() < deadline:
        r = c.get("/health/ready")
        last_status = r.status_code
        if r.status_code == 200:
            return
        time.sleep(0.75)
    raise RuntimeError(f"runtime not ready after {timeout_s}s (last HTTP {last_status})")


def run() -> None:
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200

        _wait_ready(c, timeout_s=120.0)

        llm_probe = c.get("/health/llm").json()
        assert isinstance(llm_probe, dict)

        cat = c.get("/sources/catalog")
        assert cat.status_code == 200
        cj = cat.json()
        assert "pdf_files" in cj and "csv_files" in cj

        sid_data = f"smoke-data-{uuid.uuid4().hex[:8]}"
        r = c.post("/graph/chat", json={"session_id": sid_data, "message": "预测准确率大约多少"})
        assert r.status_code == 200, r.text
        b = r.json()
        assert b["status"] in ("completed", "interrupted")
        rp = (b.get("brief") or {}).get("route_plan")
        assert rp in ("data_only", "data_then_compliance", "compliance_only", "out_of_scope")

        sid_policy = f"smoke-pol-{uuid.uuid4().hex[:8]}"
        r = c.post(
            "/graph/chat",
            json={
                "session_id": sid_policy,
                "message": "预测准确率考核 A B C 等级阈值是什么",
            },
        )
        assert r.status_code == 200
        assert r.json()["status"] in ("completed", "interrupted")

        sid_oos = f"smoke-oos-{uuid.uuid4().hex[:8]}"
        r = c.post("/graph/chat", json={"session_id": sid_oos, "message": "给我写一个vue代码"})
        assert r.status_code == 200
        assert (r.json().get("brief") or {}).get("route_plan") == "out_of_scope"

        sid_mix = f"smoke-mix-{uuid.uuid4().hex[:8]}"
        r = c.post(
            "/graph/chat",
            json={
                "session_id": sid_mix,
                "message": "化妆品品类预测准确率和制度上的评级关系是什么",
            },
        )
        assert r.status_code == 200
        assert r.json()["status"] in ("completed", "interrupted")
        rp = (r.json().get("brief") or {}).get("route_plan")
        assert rp in ("data_then_compliance", "compliance_only", "data_only")

        sid_ap = f"smoke-ap-{uuid.uuid4().hex[:8]}"
        r = c.post(
            "/graph/chat",
            json={
                "session_id": sid_ap,
                "message": "根据制度建议将库存周转天数压降20%，并给出执行方案",
            },
        )
        assert r.status_code == 200
        body = r.json()
        if body["status"] == "interrupted":
            r2 = c.post(
                "/graph/resume",
                json={"session_id": sid_ap, "decision": "approved", "feedback": ""},
            )
            assert r2.status_code == 200
            assert r2.json()["status"] == "completed"
        else:
            assert body["status"] == "completed"

        hist = c.get("/session/history", params={"session_id": sid_data})
        assert hist.status_code == 200
        assert isinstance(hist.json().get("messages"), list)

    print("smoke_test passed: health, ready, catalog, data/policy/mix routes, oos, optional approve, history")


if __name__ == "__main__":
    run()
