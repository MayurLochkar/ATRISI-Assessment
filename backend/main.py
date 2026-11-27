"""FastAPI entrypoint exposing SyllabAI REST API for PDF parsing, planning, RAG & exports."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Generator, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from sqlalchemy.orm import Session

from backend.ai.cognitive_model import get_cognitive_engine
from backend.ai.pipeline import replan_study_schedule, run_study_plan_pipeline
from backend.ai.rag_engine import answer_syllabus_question
from backend.ai.spaced_repetition import update_task_fsrs
from backend.config import settings
from backend.database import (
    SessionLocal,
    delete_plan,
    get_plan_and_chunks,
    init_db,
    list_plans,
    save_plan,
)
from backend.schemas import (
    PlannerConstraints,
    RagQueryRequest,
    RagQueryResponse,
    ReplanRequest,
    StudyEngineMode,
    StudyPlanListItem,
    StudyPlanResponse,
    TaskUpdateRequest,
    TextGenerateRequest,
)
from backend.services.export_service import export_plan_to_ics, export_plan_to_markdown
from backend.services.sample_syllabi import SAMPLES_CATALOG, ensure_sample_pdf, list_sample_catalog


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    init_db()
    # Warm up PyTorch CognitiveLoadNet once at boot so first request is instant
    get_cognitive_engine()
    for sid in SAMPLES_CATALOG:
        ensure_sample_pdf(sid)
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Multi-modal Syllabus PDF to Weekly Study Planner powered by PyTorch, NetworkX, SciPy MILP, FSRS & Hybrid RAG.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.api_route("/", methods=["GET", "HEAD"])
def root_health() -> dict:
    return {"status": "ok", "service": settings.APP_NAME, "docs": "/docs", "health": "/api/health"}


@app.get("/api/health")
def health_check() -> dict:
    cog = get_cognitive_engine()
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "pytorch_model": {
            "ready": True,
            "val_accuracy_pct": cog.val_accuracy,
            "val_load_mae": cog.val_load_mae,
        },
        "engines": {
            "local_hybrid": True,
            "openai_configured": bool(settings.OPENAI_API_KEY),
            "gemini_configured": bool(settings.GEMINI_API_KEY),
            "ollama_url": settings.OLLAMA_BASE_URL,
        },
    }


@app.get("/api/samples")
def get_samples() -> list[dict]:
    return list_sample_catalog()


@app.get("/api/samples/{sample_id}/pdf")
def download_sample_pdf(sample_id: str):
    if sample_id not in SAMPLES_CATALOG:
        raise HTTPException(status_code=404, detail="Sample syllabus not found")
    path = ensure_sample_pdf(sample_id)
    return FileResponse(
        path=str(path),
        media_type="application/pdf",
        filename=f"{sample_id}.pdf",
    )


@app.post("/api/samples/{sample_id}/generate", response_model=StudyPlanResponse)
def generate_from_sample(
    sample_id: str,
    total_weeks: Optional[int] = Form(default=None),
    weekly_hours: Optional[float] = Form(default=None),
    preferred_days: str = Form(default='["Mon","Tue","Wed","Thu","Sat"]'),
    start_date: str = Form(default=""),
    learning_intensity: str = Form(default="balanced"),
    include_spaced_repetition: bool = Form(default=True),
    engine: StudyEngineMode = Form(default="local_hybrid"),
    api_key: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
) -> StudyPlanResponse:
    if sample_id not in SAMPLES_CATALOG:
        raise HTTPException(status_code=404, detail="Sample syllabus not found")
    pdf_path = ensure_sample_pdf(sample_id)
    meta = SAMPLES_CATALOG[sample_id]["meta"]

    try:
        days_list = json.loads(preferred_days)
        if not isinstance(days_list, list) or not days_list:
            days_list = ["Mon", "Tue", "Wed", "Thu", "Sat"]
    except json.JSONDecodeError:
        days_list = [d.strip() for d in preferred_days.split(",") if d.strip()] or ["Mon", "Wed", "Fri"]

    constraints = PlannerConstraints(
        total_weeks=max(2, min(24, total_weeks or meta.recommended_weeks)),
        weekly_hours=max(2.0, min(60.0, weekly_hours or meta.recommended_hours)),
        preferred_days=days_list,
        start_date=start_date,
        learning_intensity=learning_intensity if learning_intensity in ("balanced", "front_loaded", "exam_focused") else "balanced",  # type: ignore[arg-type]
        include_spaced_repetition=include_spaced_repetition,
        engine=engine,
    )
    plan, chunks = run_study_plan_pipeline(
        constraints=constraints,
        pdf_bytes=pdf_path.read_bytes(),
        api_key=api_key,
    )
    save_plan(db, plan, chunks)
    return plan


@app.post("/api/plans/generate", response_model=StudyPlanResponse)
async def generate_plan_from_pdf(
    file: UploadFile = File(...),
    total_weeks: int = Form(default=12),
    weekly_hours: float = Form(default=10.0),
    preferred_days: str = Form(default='["Mon","Tue","Wed","Thu","Sat"]'),
    start_date: str = Form(default=""),
    learning_intensity: str = Form(default="balanced"),
    include_spaced_repetition: bool = Form(default=True),
    engine: StudyEngineMode = Form(default="local_hybrid"),
    api_key: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
) -> StudyPlanResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a valid .pdf syllabus file.")

    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")
    if len(pdf_bytes) > settings.MAX_PDF_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"PDF exceeds {settings.MAX_PDF_SIZE_MB}MB limit.",
        )

    try:
        days_list = json.loads(preferred_days)
        if not isinstance(days_list, list) or not days_list:
            days_list = ["Mon", "Tue", "Wed", "Thu", "Sat"]
    except json.JSONDecodeError:
        days_list = [d.strip() for d in preferred_days.split(",") if d.strip()] or ["Mon", "Wed", "Fri"]

    constraints = PlannerConstraints(
        total_weeks=max(2, min(24, total_weeks)),
        weekly_hours=max(2.0, min(60.0, weekly_hours)),
        preferred_days=days_list,
        start_date=start_date,
        learning_intensity=learning_intensity if learning_intensity in ("balanced", "front_loaded", "exam_focused") else "balanced",  # type: ignore[arg-type]
        include_spaced_repetition=include_spaced_repetition,
        engine=engine,
    )

    try:
        plan, chunks = run_study_plan_pipeline(
            constraints=constraints,
            pdf_bytes=pdf_bytes,
            api_key=api_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {exc}") from exc

    save_plan(db, plan, chunks)
    return plan


@app.post("/api/plans/generate-text", response_model=StudyPlanResponse)
def generate_plan_from_text(
    req: TextGenerateRequest, db: Session = Depends(get_db)
) -> StudyPlanResponse:
    constraints = PlannerConstraints(
        total_weeks=req.total_weeks,
        weekly_hours=req.weekly_hours,
        preferred_days=req.preferred_days,
        start_date=req.start_date,
        learning_intensity=req.learning_intensity,
        include_spaced_repetition=req.include_spaced_repetition,
        engine=req.engine,
    )
    try:
        plan, chunks = run_study_plan_pipeline(
            constraints=constraints,
            raw_text=req.syllabus_text,
            api_key=req.api_key,
            title_hint=req.course_title_hint,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    save_plan(db, plan, chunks)
    return plan


@app.get("/api/plans", response_model=list[StudyPlanListItem])
def get_all_plans(db: Session = Depends(get_db)) -> list[StudyPlanListItem]:
    return list_plans(db)


@app.get("/api/plans/{plan_id}", response_model=StudyPlanResponse)
def get_plan(plan_id: str, db: Session = Depends(get_db)) -> StudyPlanResponse:
    res = get_plan_and_chunks(db, plan_id)
    if res is None:
        raise HTTPException(status_code=404, detail="Study plan not found")
    return res[0]


@app.delete("/api/plans/{plan_id}")
def remove_plan(plan_id: str, db: Session = Depends(get_db)) -> dict:
    ok = delete_plan(db, plan_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Study plan not found")
    return {"deleted": True, "plan_id": plan_id}


@app.patch("/api/plans/{plan_id}/tasks/{task_id}", response_model=StudyPlanResponse)
def update_task(
    plan_id: str,
    task_id: str,
    payload: TaskUpdateRequest,
    db: Session = Depends(get_db),
) -> StudyPlanResponse:
    res = get_plan_and_chunks(db, plan_id)
    if res is None:
        raise HTTPException(status_code=404, detail="Study plan not found")
    plan, chunks = res

    found = False
    for week in plan.weeks:
        for idx, tk in enumerate(week.tasks):
            if tk.id == task_id:
                if payload.is_completed is not None:
                    tk.is_completed = payload.is_completed
                if payload.confidence_rating is not None:
                    tk = update_task_fsrs(tk, payload.confidence_rating)
                    week.tasks[idx] = tk
                found = True
                break
        if found:
            break

    if not found:
        raise HTTPException(status_code=404, detail="Task not found in study plan")

    save_plan(db, plan, chunks)
    return plan


@app.post("/api/plans/{plan_id}/replan", response_model=StudyPlanResponse)
def adaptive_replan(
    plan_id: str,
    req: ReplanRequest,
    db: Session = Depends(get_db),
) -> StudyPlanResponse:
    res = get_plan_and_chunks(db, plan_id)
    if res is None:
        raise HTTPException(status_code=404, detail="Study plan not found")
    plan, chunks = res
    updated = replan_study_schedule(plan, req, chunks)
    save_plan(db, updated, chunks)
    return updated


@app.post("/api/plans/{plan_id}/rag", response_model=RagQueryResponse)
def query_syllabus_rag(
    plan_id: str,
    req: RagQueryRequest,
    db: Session = Depends(get_db),
) -> RagQueryResponse:
    from backend.ai.pdf_ingest import chunk_pages, pages_from_raw_text

    res = get_plan_and_chunks(db, plan_id)
    if res is not None:
        plan, chunks = res
        engine_to_use: StudyEngineMode = req.engine or plan.constraints.engine
    elif req.fallback_context:
        chunks = chunk_pages(pages_from_raw_text(req.fallback_context), target_tokens=200, overlap_tokens=30)
        engine_to_use = req.engine or "local_hybrid"
    else:
        raise HTTPException(status_code=404, detail="Study plan not found")

    return answer_syllabus_question(
        chunks=chunks,
        question=req.question,
        top_k=req.top_k,
        engine=engine_to_use,
        api_key=req.api_key,
    )


@app.get("/api/plans/{plan_id}/export/ics")
def download_ics(plan_id: str, db: Session = Depends(get_db)) -> Response:
    res = get_plan_and_chunks(db, plan_id)
    if res is None:
        raise HTTPException(status_code=404, detail="Study plan not found")
    plan, _ = res
    ics_content = export_plan_to_ics(plan)
    filename = f"{plan.metadata.course_code.lower()}_study_plan.ics"
    return Response(
        content=ics_content,
        media_type="text/calendar; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/plans/{plan_id}/export/markdown")
def download_markdown(plan_id: str, db: Session = Depends(get_db)) -> PlainTextResponse:
    res = get_plan_and_chunks(db, plan_id)
    if res is None:
        raise HTTPException(status_code=404, detail="Study plan not found")
    plan, _ = res
    md = export_plan_to_markdown(plan)
    filename = f"{plan.metadata.course_code.lower()}_study_guide.md"
    return PlainTextResponse(
        content=md,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/plans/{plan_id}/export/json")
def download_json(plan_id: str, db: Session = Depends(get_db)) -> Response:
    res = get_plan_and_chunks(db, plan_id)
    if res is None:
        raise HTTPException(status_code=404, detail="Study plan not found")
    plan, _ = res
    filename = f"{plan.metadata.course_code.lower()}_study_plan.json"
    return Response(
        content=plan.model_dump_json(indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
