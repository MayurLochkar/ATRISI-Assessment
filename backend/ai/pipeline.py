"""End-to-end Neuro-Symbolic & Agentic AI Pipeline Orchestrator + Adaptive Re-Planner."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np

from backend.ai.cognitive_model import get_cognitive_engine
from backend.ai.knowledge_graph import build_prerequisite_dag
from backend.ai.optimizer import optimize_study_schedule
from backend.ai.pdf_ingest import (
    chunk_pages,
    extract_pages_from_pdf,
    pages_from_raw_text,
)
from backend.ai.spaced_repetition import inject_spaced_repetition
from backend.ai.syllabus_parser import parse_syllabus
from backend.schemas import (
    EvaluationMetrics,
    PipelineStageMetric,
    PlannerConstraints,
    PrerequisiteEdge,
    ReplanRequest,
    StudyPlanResponse,
    SyllabusChunk,
    TopicNode,
    WeeklySummarySchema,
)


def _expand_topics_for_horizon(
    topics: list[TopicNode], total_weeks: int
) -> list[TopicNode]:
    """
    If a syllabus has fewer units than total_weeks (e.g. a 5-unit syllabus over 12 weeks),
    split dense units into coherent Part A / Part B sub-modules so every week receives
    meaningful, balanced study material instead of leaving W2/W3 empty.
    """
    expanded = [t.model_copy() for t in topics]
    target = min(max(total_weeks, len(expanded)), 24)

    while len(expanded) < target:
        # Pick the unit with the most subtopics (>= 4) to split
        candidates = [
            (idx, t) for idx, t in enumerate(expanded) if len(t.subtopics) >= 4
        ]
        if not candidates:
            break
        split_idx, parent = max(candidates, key=lambda item: len(item[1].subtopics))
        mid = len(parent.subtopics) // 2
        part_a_subs = parent.subtopics[:mid]
        part_b_subs = parent.subtopics[mid:]

        base_title = parent.title.split(" — Part")[0].strip()
        node_a = TopicNode(
            id="tmp_a",
            module_number=parent.module_number,
            module_title=parent.module_title,
            title=f"{base_title} — Part A (Core Concepts)",
            subtopics=part_a_subs,
            page_refs=parent.page_refs,
            keywords=parent.keywords[:4],
        )
        node_b = TopicNode(
            id="tmp_b",
            module_number=parent.module_number,
            module_title=parent.module_title,
            title=f"{base_title} — Part B (Applied & Advanced)",
            subtopics=part_b_subs,
            page_refs=parent.page_refs,
            keywords=parent.keywords[4:] or parent.keywords[:4],
        )
        expanded = expanded[:split_idx] + [node_a, node_b] + expanded[split_idx + 1 :]

    for idx, t in enumerate(expanded, start=1):
        t.id = f"topic_{idx:02d}"
    return expanded


def _compute_evaluation_metrics(
    topics: list[TopicNode],
    edges: list[PrerequisiteEdge],
    weeks: list[WeeklySummarySchema],
    chunks: list[SyllabusChunk],
    retention_projection: list[dict[str, float | int]],
    solver_status: str,
    feasibility_warning: Optional[str],
    recommended_hours: Optional[float],
    val_accuracy: float,
    val_load_mae: float,
) -> EvaluationMetrics:
    # Map topic_id -> earliest week where primary tasks appear
    topic_week: dict[str, int] = {}
    for w in weeks:
        for tk in w.tasks:
            if tk.topic_id and tk.task_type != "active_recall":
                if tk.topic_id not in topic_week:
                    topic_week[tk.topic_id] = w.week_number

    if edges:
        compliant = sum(
            1
            for e in edges
            if topic_week.get(e.source, 1) <= topic_week.get(e.target, 999)
        )
        compliance_rate = round((compliant / len(edges)) * 100.0, 1)
    else:
        compliance_rate = 100.0

    week_hours = [w.planned_hours for w in weeks]
    active_hours = [h for h in week_hours if h > 0]
    if len(active_hours) >= 2 and np.mean(active_hours) > 0:
        cv = float(np.std(active_hours) / np.mean(active_hours))
        load_balance = round(max(0.0, min(100.0, (1.0 - cv) * 100.0)), 1)
    else:
        load_balance = 100.0

    utilizations = [
        (w.planned_hours / w.capacity_hours) * 100.0
        for w in weeks
        if w.capacity_hours > 0
    ]
    avg_util = round(float(np.mean(utilizations)) if utilizations else 0.0, 1)

    bloom_dist: dict[str, int] = {
        "Remember": 0,
        "Understand": 0,
        "Apply": 0,
        "Analyze": 0,
        "Evaluate": 0,
        "Create": 0,
    }
    for t in topics:
        bloom_dist[t.bloom_level] = bloom_dist.get(t.bloom_level, 0) + 1

    return EvaluationMetrics(
        prerequisite_compliance_rate=compliance_rate,
        load_balance_score=load_balance,
        avg_weekly_utilization=avg_util,
        critical_path_length=sum(1 for t in topics if t.is_critical_path),
        total_study_hours=round(sum(w.planned_hours for w in weeks), 2),
        solver_status=solver_status,
        feasibility_warning=feasibility_warning,
        recommended_weekly_hours=recommended_hours,
        bloom_distribution=bloom_dist,
        retention_projection=retention_projection,
        rag_chunk_count=len(chunks),
        pytorch_model_val_accuracy=val_accuracy,
        pytorch_model_load_mae=val_load_mae,
    )


def run_study_plan_pipeline(
    constraints: PlannerConstraints,
    pdf_bytes: Optional[bytes] = None,
    raw_text: Optional[str] = None,
    api_key: Optional[str] = None,
    title_hint: Optional[str] = None,
) -> Tuple[StudyPlanResponse, list[SyllabusChunk]]:
    """Execute the full 6-stage AI/ML study planning pipeline with telemetry."""
    telemetry: list[PipelineStageMetric] = []

    # Stage 1: Multi-modal PDF / Text Extraction + Tiktoken Chunking
    t0 = time.perf_counter()
    if pdf_bytes is not None:
        pages, extraction_method = extract_pages_from_pdf(pdf_bytes, api_key=api_key)
    elif raw_text is not None:
        pages = pages_from_raw_text(raw_text)
        extraction_method = "raw_text_segmented"
    else:
        raise ValueError("Either pdf_bytes or raw_text must be provided.")

    if not any(txt.strip() for _, txt in pages):
        raise ValueError(
            "No readable text could be extracted from this scanned/image PDF. "
            "Try the 'Paste Syllabus Text' tab, upload a text-selectable PDF, "
            "or add a Gemini API key in Settings for Vision OCR."
        )

    chunks = chunk_pages(pages)
    d1 = round((time.perf_counter() - t0) * 1000.0, 2)
    telemetry.append(
        PipelineStageMetric(
            stage="1. Multi-Modal Document Ingestion",
            model_or_algorithm=f"{extraction_method} + tiktoken",
            duration_ms=d1,
            status="success",
            details=f"Extracted {len(pages)} page(s) into {len(chunks)} overlapping RAG chunks.",
        )
    )

    # Stage 2: Structural NLP / LLM Syllabus Parsing
    t1 = time.perf_counter()
    metadata, topics, assessments, parse_status = parse_syllabus(
        pages=pages,
        extraction_method=extraction_method,
        total_weeks=constraints.total_weeks,
        engine=constraints.engine,
        api_key=api_key,
        title_hint=title_hint,
    )
    topics = _expand_topics_for_horizon(topics, constraints.total_weeks)
    d2 = round((time.perf_counter() - t1) * 1000.0, 2)
    telemetry.append(
        PipelineStageMetric(
            stage="2. Curriculum & Assessment Parser",
            model_or_algorithm=(
                f"{constraints.engine} JSON Agent"
                if parse_status == "success" and constraints.engine != "local_hybrid"
                else "Structural NLP + Regex Grammar"
            ),
            duration_ms=d2,
            status=parse_status,  # type: ignore[arg-type]
            details=f"Parsed {len(topics)} study topics and {len(assessments)} milestone assessments.",
        )
    )

    # Stage 3: Prerequisite Knowledge Graph (DAG + PageRank + Critical Path)
    t2 = time.perf_counter()
    topics, edges, topo_order = build_prerequisite_dag(topics)
    order_lookup = {tid: idx for idx, tid in enumerate(topo_order)}
    topics.sort(key=lambda t: order_lookup.get(t.id, 999))
    d3 = round((time.perf_counter() - t2) * 1000.0, 2)
    telemetry.append(
        PipelineStageMetric(
            stage="3. Prerequisite Knowledge Graph",
            model_or_algorithm="TF-IDF Cosine + NetworkX DAG + PageRank",
            duration_ms=d3,
            status="success",
            details=f"Built verified DAG with {len(edges)} dependency edges; Critical Path length = {sum(1 for t in topics if t.is_critical_path)}.",
        )
    )

    # Stage 4: PyTorch Multi-Task Cognitive Load & Bloom's Taxonomy Classifier (Batched)
    t3 = time.perf_counter()
    cog_engine = get_cognitive_engine()
    total_topics = len(topics)
    batch_inputs = [
        (t.title, t.subtopics, idx / max(1, total_topics - 1), len(t.prerequisites))
        for idx, t in enumerate(topics)
    ]
    preds = cog_engine.predict_batch(
        batch_inputs, weekly_hours_budget=constraints.weekly_hours
    )
    for t, (bloom, conf, load, hrs) in zip(topics, preds):
        t.bloom_level = bloom
        t.bloom_confidence = conf
        t.cognitive_load = load
        t.estimated_hours = hrs
    d4 = round((time.perf_counter() - t3) * 1000.0, 2)
    telemetry.append(
        PipelineStageMetric(
            stage="4. Cognitive Load & Bloom Classifier",
            model_or_algorithm="PyTorch CognitiveLoadNet (Multi-Task MLP)",
            duration_ms=d4,
            status="success",
            details=f"Classified {len(topics)} topics (Val Acc {cog_engine.val_accuracy}%, Load MAE {cog_engine.val_load_mae}).",
        )
    )

    # Stage 5: SciPy MILP Multi-Objective Schedule Optimizer
    t4 = time.perf_counter()
    weeks, solver_status, warning, rec_hours = optimize_study_schedule(
        topics=topics,
        edges=edges,
        assessments=assessments,
        constraints=constraints,
    )
    d5 = round((time.perf_counter() - t4) * 1000.0, 2)
    telemetry.append(
        PipelineStageMetric(
            stage="5. MILP Constraint Schedule Optimizer",
            model_or_algorithm=f"SciPy MILP ({solver_status})",
            duration_ms=d5,
            status="warning" if warning else "success",
            details=warning or f"Optimized weekly capacity & DAG precedence across {constraints.total_weeks} weeks.",
        )
    )

    # Stage 6: FSRS Exponential Memory Decay & Active Recall Injection
    t5 = time.perf_counter()
    weeks, retention_proj = inject_spaced_repetition(
        weeks=weeks, topics=topics, constraints=constraints
    )
    d6 = round((time.perf_counter() - t5) * 1000.0, 2)
    recall_count = sum(
        1 for w in weeks for tk in w.tasks if tk.task_type == "active_recall"
    )
    telemetry.append(
        PipelineStageMetric(
            stage="6. FSRS Spaced Repetition Engine",
            model_or_algorithm="Exponential Decay R(t)=exp(-t/S)",
            duration_ms=d6,
            status="success",
            details=f"Injected {recall_count} active recall checkpoints across {constraints.total_weeks} weeks.",
        )
    )

    evaluation = _compute_evaluation_metrics(
        topics=topics,
        edges=edges,
        weeks=weeks,
        chunks=chunks,
        retention_projection=retention_proj,
        solver_status=solver_status,
        feasibility_warning=warning,
        recommended_hours=rec_hours,
        val_accuracy=cog_engine.val_accuracy,
        val_load_mae=cog_engine.val_load_mae,
    )

    effective_engine = (
        constraints.engine
        if parse_status == "success" and constraints.engine != "local_hybrid"
        else "local_hybrid"
    )

    plan = StudyPlanResponse(
        plan_id=f"plan_{uuid.uuid4().hex[:10]}",
        created_at=datetime.now(timezone.utc).isoformat(),
        engine_used=effective_engine,
        metadata=metadata,
        constraints=constraints,
        topics=topics,
        edges=edges,
        assessments=assessments,
        weeks=weeks,
        evaluation=evaluation,
        pipeline_telemetry=telemetry,
    )
    return plan, chunks


def replan_study_schedule(
    plan: StudyPlanResponse, req: ReplanRequest, chunks: list[SyllabusChunk]
) -> StudyPlanResponse:
    """
    Adaptive Re-Planner:
    Locks completed tasks and past weeks (< current_week), gathers incomplete topics,
    and re-solves SciPy MILP + FSRS across [current_week..total_weeks].
    """
    t0 = time.perf_counter()
    new_constraints = plan.constraints.model_copy()
    if req.new_weekly_hours is not None:
        new_constraints.weekly_hours = req.new_weekly_hours
    new_constraints.include_spaced_repetition = req.include_spaced_repetition

    start_wk = min(req.current_week, new_constraints.total_weeks)

    # Identify topics that are already completely done before start_wk
    completed_topic_ids: set[str] = set()
    topic_primary_tasks: dict[str, list[bool]] = {}
    for w in plan.weeks:
        for tk in w.tasks:
            if tk.topic_id and tk.task_type != "active_recall":
                topic_primary_tasks.setdefault(tk.topic_id, []).append(tk.is_completed)
    for tid, flags in topic_primary_tasks.items():
        if flags and all(flags):
            completed_topic_ids.add(tid)

    remaining_topics = [t for t in plan.topics if t.id not in completed_topic_ids]
    remaining_ids = {t.id for t in remaining_topics}
    remaining_edges = [
        e for e in plan.edges if e.source in remaining_ids and e.target in remaining_ids
    ]
    remaining_assessments = [
        a for a in plan.assessments if (a.target_week or start_wk) >= start_wk
    ]

    # Mark prior weeks' completed state intact; keep only completed tasks in weeks < start_wk
    preserved_weeks: list[WeeklySummarySchema] = []
    for w in plan.weeks:
        if w.week_number < start_wk:
            kept_tasks = [tk for tk in w.tasks if tk.is_completed]
            w_copy = w.model_copy()
            w_copy.tasks = kept_tasks
            w_copy.planned_hours = round(
                sum(tk.estimated_minutes for tk in kept_tasks) / 60.0, 2
            )
            preserved_weeks.append(w_copy)

    new_weeks, solver_status, warning, rec_hours = optimize_study_schedule(
        topics=remaining_topics,
        edges=remaining_edges,
        assessments=remaining_assessments,
        constraints=new_constraints,
        start_week=start_wk,
        existing_weeks=preserved_weeks,
    )
    new_weeks, retention_proj = inject_spaced_repetition(
        weeks=new_weeks,
        topics=plan.topics,
        constraints=new_constraints,
        start_week=start_wk,
    )

    cog_engine = get_cognitive_engine()
    new_eval = _compute_evaluation_metrics(
        topics=plan.topics,
        edges=plan.edges,
        weeks=new_weeks,
        chunks=chunks,
        retention_projection=retention_proj,
        solver_status=f"replan_{solver_status}",
        feasibility_warning=warning,
        recommended_hours=rec_hours,
        val_accuracy=cog_engine.val_accuracy,
        val_load_mae=cog_engine.val_load_mae,
    )

    dur = round((time.perf_counter() - t0) * 1000.0, 2)
    plan.constraints = new_constraints
    plan.weeks = new_weeks
    plan.evaluation = new_eval
    plan.pipeline_telemetry.append(
        PipelineStageMetric(
            stage=f"Adaptive Re-Plan (From Week {start_wk})",
            model_or_algorithm=f"SciPy MILP + FSRS ({solver_status})",
            duration_ms=dur,
            status="warning" if warning else "success",
            details=(
                f"Re-scheduled {len(remaining_topics)} incomplete topic(s) across "
                f"Weeks {start_wk}–{new_constraints.total_weeks} at {new_constraints.weekly_hours}h/wk."
            ),
        )
    )
    return plan
