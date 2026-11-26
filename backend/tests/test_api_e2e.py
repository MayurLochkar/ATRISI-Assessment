"""End-to-end FastAPI TestClient verification suite."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base
from backend.main import app, get_db
from backend.services.sample_syllabi import ensure_sample_pdf


@pytest.fixture()
def client():
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health_and_samples(client: TestClient):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["pytorch_model"]["ready"] is True

    sr = client.get("/api/samples")
    assert sr.status_code == 200
    samples = sr.json()
    assert len(samples) == 3

    pdf_r = client.get("/api/samples/cs301_ml_dl/pdf")
    assert pdf_r.status_code == 200
    assert pdf_r.headers["content-type"] == "application/pdf"
    assert pdf_r.content.startswith(b"%PDF")


def test_full_pdf_upload_lifecycle_replan_rag_and_exports(client: TestClient):
    sample_pdf = ensure_sample_pdf("cs301_ml_dl")

    with open(sample_pdf, "rb") as f:
        gen = client.post(
            "/api/plans/generate",
            files={"file": ("cs301_ml_dl.pdf", f, "application/pdf")},
            data={
                "total_weeks": "10",
                "weekly_hours": "10",
                "preferred_days": '["Mon","Tue","Thu","Sat"]',
                "learning_intensity": "balanced",
                "include_spaced_repetition": "true",
                "engine": "local_hybrid",
            },
        )
    assert gen.status_code == 200, gen.text
    plan = gen.json()
    plan_id = plan["plan_id"]
    assert len(plan["topics"]) >= 6
    assert len(plan["weeks"]) == 10
    assert all(len(w["tasks"]) >= 1 for w in plan["weeks"])
    assert plan["evaluation"]["prerequisite_compliance_rate"] == 100.0

    # List saved plans
    lst = client.get("/api/plans")
    assert lst.status_code == 200
    assert any(item["plan_id"] == plan_id for item in lst.json())

    # Complete a task & submit FSRS rating
    first_task = next(tk for w in plan["weeks"] for tk in w["tasks"])
    patch_r = client.patch(
        f"/api/plans/{plan_id}/tasks/{first_task['id']}",
        json={"is_completed": True, "confidence_rating": "easy"},
    )
    assert patch_r.status_code == 200
    patched_plan = patch_r.json()
    updated_tk = next(
        tk for w in patched_plan["weeks"] for tk in w["tasks"] if tk["id"] == first_task["id"]
    )
    assert updated_tk["is_completed"] is True
    assert updated_tk["confidence_rating"] == "easy"
    assert updated_tk["fsrs_stability_days"] > first_task["fsrs_stability_days"]

    # Adaptive Re-Plan from Week 3 with adjusted capacity
    replan_r = client.post(
        f"/api/plans/{plan_id}/replan",
        json={"current_week": 3, "new_weekly_hours": 12.0, "include_spaced_repetition": True},
    )
    assert replan_r.status_code == 200
    replanned = replan_r.json()
    assert replanned["constraints"]["weekly_hours"] == 12.0
    assert any("Adaptive Re-Plan" in s["stage"] for s in replanned["pipeline_telemetry"])

    # Hybrid RAG query
    rag_r = client.post(
        f"/api/plans/{plan_id}/rag",
        json={"question": "When is the midterm exam and what is its weight?", "top_k": 3},
    )
    assert rag_r.status_code == 200
    rag_data = rag_r.json()
    assert len(rag_data["citations"]) >= 1
    assert "Midterm" in rag_data["answer"] or "25%" in rag_data["answer"]

    # Exports: ICS, Markdown, JSON
    ics_r = client.get(f"/api/plans/{plan_id}/export/ics")
    assert ics_r.status_code == 200
    assert "BEGIN:VCALENDAR" in ics_r.text
    assert "BEGIN:VEVENT" in ics_r.text

    md_r = client.get(f"/api/plans/{plan_id}/export/markdown")
    assert md_r.status_code == 200
    assert "# CS301:" in md_r.text

    json_r = client.get(f"/api/plans/{plan_id}/export/json")
    assert json_r.status_code == 200
    assert json_r.json()["plan_id"] == plan_id

    # Delete plan
    del_r = client.delete(f"/api/plans/{plan_id}")
    assert del_r.status_code == 200
    assert client.get(f"/api/plans/{plan_id}").status_code == 404


def test_generate_from_text_and_invalid_upload(client: TestClient):
    bad = client.post(
        "/api/plans/generate",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert bad.status_code == 400

    txt_r = client.post(
        "/api/plans/generate-text",
        json={
            "syllabus_text": """
CS410: Distributed Cloud Systems
Instructor: Prof. Lin | 3 Credits

Module 1: Networking & RPC Foundations
- TCP/IP internals, serialization, gRPC design

Module 2: Consensus & Replication
- Paxos, Raft leader election, log replication
- Midterm Exam in Week 5 (30%)

Module 3: Distributed Storage & Consistency
- CAP theorem, Dynamo consistent hashing, Spanner TrueTime
- Final Project in Week 8 (40%)
""",
            "total_weeks": 8,
            "weekly_hours": 8.0,
        },
    )
    assert txt_r.status_code == 200
    data = txt_r.json()
    assert data["metadata"]["course_code"] == "CS410"
    assert len(data["weeks"]) == 8
