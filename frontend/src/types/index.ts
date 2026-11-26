export type StudyEngineMode = 'local_hybrid' | 'openai' | 'gemini' | 'ollama';

export type BloomLevel =
  | 'Remember'
  | 'Understand'
  | 'Apply'
  | 'Analyze'
  | 'Evaluate'
  | 'Create';

export type TaskType =
  | 'concept_deep_dive'
  | 'hands_on_practice'
  | 'active_recall'
  | 'assessment_prep'
  | 'project_milestone';

export type ConfidenceRating = 'hard' | 'good' | 'easy';

export interface CourseMetadata {
  course_code: string;
  course_title: string;
  instructor: string | null;
  credits: number | null;
  semester_weeks: number;
  total_pages: number;
  extraction_method: string;
}

export interface PlannerConstraints {
  total_weeks: number;
  weekly_hours: number;
  preferred_days: string[];
  start_date: string;
  learning_intensity: 'balanced' | 'front_loaded' | 'exam_focused';
  include_spaced_repetition: boolean;
  engine: StudyEngineMode;
}

export interface TopicNode {
  id: string;
  module_number: number;
  module_title: string;
  title: string;
  subtopics: string[];
  bloom_level: BloomLevel;
  bloom_confidence: number;
  cognitive_load: number;
  estimated_hours: number;
  pagerank_centrality: number;
  is_critical_path: boolean;
  prerequisites: string[];
  page_refs: number[];
  keywords: string[];
}

export interface PrerequisiteEdge {
  source: string;
  target: string;
  weight: number;
  reason: string;
  is_critical: boolean;
}

export interface ExtractedAssessment {
  id: string;
  title: string;
  assessment_type: 'quiz' | 'midterm' | 'final' | 'assignment' | 'project';
  target_week: number | null;
  weight_percent: number | null;
  topics_covered: string[];
  description: string;
}

export interface StudyTask {
  id: string;
  topic_id: string | null;
  week_number: number;
  scheduled_date: string;
  day_label: string;
  title: string;
  description: string;
  task_type: TaskType;
  bloom_level: BloomLevel;
  estimated_minutes: number;
  priority: 'critical' | 'high' | 'medium' | 'low';
  is_completed: boolean;
  confidence_rating: ConfidenceRating | null;
  fsrs_stability_days: number;
  predicted_retention: number;
  deliverable: string;
  substeps: string[];
  page_refs: number[];
}

export interface WeeklySummary {
  week_number: number;
  week_label: string;
  date_range: string;
  focus_theme: string;
  planned_hours: number;
  capacity_hours: number;
  cognitive_load_avg: number;
  is_exam_week: boolean;
  milestone_alert: string | null;
  tasks: StudyTask[];
}

export interface PipelineStageMetric {
  stage: string;
  model_or_algorithm: string;
  duration_ms: number;
  status: 'success' | 'fallback' | 'warning';
  details: string;
}

export interface RetentionPoint {
  week: number;
  with_fsrs_pct: number;
  without_fsrs_pct: number;
}

export interface EvaluationMetrics {
  prerequisite_compliance_rate: number;
  load_balance_score: number;
  avg_weekly_utilization: number;
  critical_path_length: number;
  total_study_hours: number;
  solver_status: string;
  feasibility_warning: string | null;
  recommended_weekly_hours: number | null;
  bloom_distribution: Record<string, number>;
  retention_projection: RetentionPoint[];
  rag_chunk_count: number;
  pytorch_model_val_accuracy: number;
  pytorch_model_load_mae: number;
}

export interface StudyPlanResponse {
  plan_id: string;
  created_at: string;
  engine_used: string;
  metadata: CourseMetadata;
  constraints: PlannerConstraints;
  topics: TopicNode[];
  edges: PrerequisiteEdge[];
  assessments: ExtractedAssessment[];
  weeks: WeeklySummary[];
  evaluation: EvaluationMetrics;
  pipeline_telemetry: PipelineStageMetric[];
}

export interface StudyPlanListItem {
  plan_id: string;
  created_at: string;
  course_code: string;
  course_title: string;
  total_weeks: number;
  weekly_hours: number;
  total_tasks: number;
  completed_tasks: number;
  engine_used: string;
}

export interface RagCitation {
  chunk_id: string;
  page_number: number;
  section_title: string;
  score: number;
  snippet: string;
}

export interface RagQueryResponse {
  answer: string;
  engine_used: string;
  citations: RagCitation[];
  latency_ms: number;
}

export interface SampleSyllabus {
  id: string;
  course_code: string;
  title: string;
  subtitle: string;
  recommended_weeks: number;
  recommended_hours: number;
}

export interface HealthStatus {
  status: string;
  app: string;
  version: string;
  pytorch_model: {
    ready: boolean;
    val_accuracy_pct: number;
    val_load_mae: number;
  };
  engines: {
    local_hybrid: boolean;
    openai_configured: boolean;
    gemini_configured: boolean;
    ollama_url: string;
  };
}
