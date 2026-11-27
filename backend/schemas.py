"""Pydantic v2 data contracts for the SyllabAI multi-stage pipeline and API."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

StudyEngineMode = Literal["local_hybrid", "openai", "gemini", "ollama"]
BloomLevel = Literal["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
TaskType = Literal[
    "concept_deep_dive",
    "hands_on_practice",
    "active_recall",
    "assessment_prep",
    "project_milestone",
]
ConfidenceRating = Literal["hard", "good", "easy"]


class SyllabusChunk(BaseModel):
    chunk_id: str
    page_number: int
    section_title: str = "General"
    text: str
    token_count: int


class ExtractedAssessment(BaseModel):
    id: str
    title: str
    assessment_type: Literal["quiz", "midterm", "final", "assignment", "project"] = "assignment"
    target_week: Optional[int] = None
    weight_percent: Optional[float] = None
    topics_covered: list[str] = Field(default_factory=list)
    description: str = ""


class TopicNode(BaseModel):
    id: str
    module_number: int
    module_title: str
    title: str
    subtopics: list[str] = Field(default_factory=list)
    bloom_level: BloomLevel = "Understand"
    bloom_confidence: float = 0.85
    cognitive_load: float = Field(default=5.0, ge=1.0, le=10.0)
    estimated_hours: float = Field(default=2.5, ge=0.5, le=40.0)
    pagerank_centrality: float = 0.0
    is_critical_path: bool = False
    prerequisites: list[str] = Field(default_factory=list)
    page_refs: list[int] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class PrerequisiteEdge(BaseModel):
    source: str
    target: str
    weight: float = Field(ge=0.0, le=1.0)
    reason: str
    is_critical: bool = False


class CourseMetadata(BaseModel):
    course_code: str = "COURSE-101"
    course_title: str = "Untitled Course"
    instructor: Optional[str] = None
    credits: Optional[float] = None
    semester_weeks: int = 12
    total_pages: int = 1
    extraction_method: str = "pymupdf_native"


class PlannerConstraints(BaseModel):
    total_weeks: int = Field(default=12, ge=2, le=24)
    weekly_hours: float = Field(default=10.0, ge=2.0, le=60.0)
    preferred_days: list[str] = Field(
        default_factory=lambda: ["Mon", "Tue", "Wed", "Thu", "Sat"]
    )
    start_date: str = ""
    learning_intensity: Literal["balanced", "front_loaded", "exam_focused"] = "balanced"
    include_spaced_repetition: bool = True
    engine: StudyEngineMode = "local_hybrid"


class StudyTaskSchema(BaseModel):
    id: str
    topic_id: Optional[str] = None
    week_number: int
    scheduled_date: str
    day_label: str
    title: str
    description: str
    task_type: TaskType
    bloom_level: BloomLevel = "Apply"
    estimated_minutes: int = Field(ge=15, le=480)
    priority: Literal["critical", "high", "medium", "low"] = "medium"
    is_completed: bool = False
    confidence_rating: Optional[ConfidenceRating] = None
    fsrs_stability_days: float = 4.0
    predicted_retention: float = 0.92
    deliverable: str = ""
    substeps: list[str] = Field(default_factory=list)
    page_refs: list[int] = Field(default_factory=list)


class WeeklySummarySchema(BaseModel):
    week_number: int
    week_label: str
    date_range: str
    focus_theme: str
    planned_hours: float
    capacity_hours: float
    cognitive_load_avg: float
    is_exam_week: bool = False
    milestone_alert: Optional[str] = None
    tasks: list[StudyTaskSchema] = Field(default_factory=list)


class PipelineStageMetric(BaseModel):
    stage: str
    model_or_algorithm: str
    duration_ms: float
    status: Literal["success", "fallback", "warning"] = "success"
    details: str


class EvaluationMetrics(BaseModel):
    prerequisite_compliance_rate: float
    load_balance_score: float
    avg_weekly_utilization: float
    critical_path_length: int
    total_study_hours: float
    solver_status: str
    feasibility_warning: Optional[str] = None
    recommended_weekly_hours: Optional[float] = None
    bloom_distribution: dict[str, int]
    retention_projection: list[dict[str, float | int]]
    rag_chunk_count: int
    pytorch_model_val_accuracy: float
    pytorch_model_load_mae: float


class StudyPlanResponse(BaseModel):
    plan_id: str
    created_at: str
    engine_used: str
    metadata: CourseMetadata
    constraints: PlannerConstraints
    topics: list[TopicNode]
    edges: list[PrerequisiteEdge]
    assessments: list[ExtractedAssessment]
    weeks: list[WeeklySummarySchema]
    evaluation: EvaluationMetrics
    pipeline_telemetry: list[PipelineStageMetric]


class StudyPlanListItem(BaseModel):
    plan_id: str
    created_at: str
    course_code: str
    course_title: str
    total_weeks: int
    weekly_hours: float
    total_tasks: int
    completed_tasks: int
    engine_used: str


class TaskUpdateRequest(BaseModel):
    is_completed: Optional[bool] = None
    confidence_rating: Optional[ConfidenceRating] = None


class ReplanRequest(BaseModel):
    current_week: int = Field(default=1, ge=1, le=24)
    new_weekly_hours: Optional[float] = Field(default=None, ge=2.0, le=60.0)
    include_spaced_repetition: bool = True


class RagQueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)
    engine: Optional[StudyEngineMode] = None
    api_key: Optional[str] = None
    fallback_context: Optional[str] = Field(default=None, max_length=60000)


class RagCitation(BaseModel):
    chunk_id: str
    page_number: int
    section_title: str
    score: float
    snippet: str


class RagQueryResponse(BaseModel):
    answer: str
    engine_used: str
    citations: list[RagCitation]
    latency_ms: float


class TextGenerateRequest(BaseModel):
    syllabus_text: str = Field(min_length=40, max_length=150000)
    course_title_hint: Optional[str] = None
    total_weeks: int = Field(default=12, ge=2, le=24)
    weekly_hours: float = Field(default=10.0, ge=2.0, le=60.0)
    preferred_days: list[str] = Field(
        default_factory=lambda: ["Mon", "Tue", "Wed", "Thu", "Sat"]
    )
    start_date: str = ""
    learning_intensity: Literal["balanced", "front_loaded", "exam_focused"] = "balanced"
    include_spaced_repetition: bool = True
    engine: StudyEngineMode = "local_hybrid"
    api_key: Optional[str] = None
