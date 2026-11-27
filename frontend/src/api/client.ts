import type {
  ConfidenceRating,
  HealthStatus,
  RagQueryResponse,
  SampleSyllabus,
  StudyEngineMode,
  StudyPlanListItem,
  StudyPlanResponse,
} from '../types';

const rawApiUrl = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');
const API_BASE = rawApiUrl ? `${rawApiUrl}/api` : '/api';

async function fetchWithTimeout(
  url: string,
  init: RequestInit = {},
  timeoutMs = 38000
): Promise<Response> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new Error(
        'Request timed out at 38s (Render free-tier cold start). Please click Generate again — the server is now awake and will finish in ~2s!'
      );
    }
    throw err;
  } finally {
    window.clearTimeout(timer);
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      if (data?.detail) {
        message = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
      }
    } catch {
      const txt = await res.text().catch(() => '');
      if (txt) message = txt;
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export interface GenerateOptions {
  total_weeks: number;
  weekly_hours: number;
  preferred_days: string[];
  start_date: string;
  learning_intensity: 'balanced' | 'front_loaded' | 'exam_focused';
  include_spaced_repetition: boolean;
  engine: StudyEngineMode;
  api_key?: string;
}

export const apiClient = {
  async getHealth(): Promise<HealthStatus> {
    const res = await fetch(`${API_BASE}/health`);
    return handleResponse<HealthStatus>(res);
  },

  async getSamples(): Promise<SampleSyllabus[]> {
    const res = await fetch(`${API_BASE}/samples`);
    return handleResponse<SampleSyllabus[]>(res);
  },

  async listPlans(): Promise<StudyPlanListItem[]> {
    const res = await fetch(`${API_BASE}/plans`);
    return handleResponse<StudyPlanListItem[]>(res);
  },

  async getPlan(planId: string): Promise<StudyPlanResponse> {
    const res = await fetch(`${API_BASE}/plans/${encodeURIComponent(planId)}`);
    return handleResponse<StudyPlanResponse>(res);
  },

  async deletePlan(planId: string): Promise<void> {
    const res = await fetch(`${API_BASE}/plans/${encodeURIComponent(planId)}`, {
      method: 'DELETE',
    });
    await handleResponse(res);
  },

  async generateFromPdf(file: File, opts: GenerateOptions): Promise<StudyPlanResponse> {
    const form = new FormData();
    form.append('file', file);
    form.append('total_weeks', String(opts.total_weeks));
    form.append('weekly_hours', String(opts.weekly_hours));
    form.append('preferred_days', JSON.stringify(opts.preferred_days));
    form.append('start_date', opts.start_date);
    form.append('learning_intensity', opts.learning_intensity);
    form.append('include_spaced_repetition', String(opts.include_spaced_repetition));
    form.append('engine', opts.engine);
    if (opts.api_key) form.append('api_key', opts.api_key);

    const res = await fetchWithTimeout(`${API_BASE}/plans/generate`, {
      method: 'POST',
      body: form,
    });
    return handleResponse<StudyPlanResponse>(res);
  },

  async generateFromSample(sampleId: string, opts: GenerateOptions): Promise<StudyPlanResponse> {
    const form = new FormData();
    form.append('total_weeks', String(opts.total_weeks));
    form.append('weekly_hours', String(opts.weekly_hours));
    form.append('preferred_days', JSON.stringify(opts.preferred_days));
    form.append('start_date', opts.start_date);
    form.append('learning_intensity', opts.learning_intensity);
    form.append('include_spaced_repetition', String(opts.include_spaced_repetition));
    form.append('engine', opts.engine);
    if (opts.api_key) form.append('api_key', opts.api_key);

    const res = await fetchWithTimeout(
      `${API_BASE}/samples/${encodeURIComponent(sampleId)}/generate`,
      {
        method: 'POST',
        body: form,
      }
    );
    return handleResponse<StudyPlanResponse>(res);
  },

  async generateFromText(
    syllabusText: string,
    opts: GenerateOptions,
    courseTitleHint?: string
  ): Promise<StudyPlanResponse> {
    const res = await fetchWithTimeout(`${API_BASE}/plans/generate-text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        syllabus_text: syllabusText,
        course_title_hint: courseTitleHint || null,
        ...opts,
        api_key: opts.api_key || null,
      }),
    });
    return handleResponse<StudyPlanResponse>(res);
  },

  async updateTask(
    planId: string,
    taskId: string,
    payload: { is_completed?: boolean; confidence_rating?: ConfidenceRating }
  ): Promise<StudyPlanResponse> {
    const res = await fetch(
      `${API_BASE}/plans/${encodeURIComponent(planId)}/tasks/${encodeURIComponent(taskId)}`,
      {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }
    );
    return handleResponse<StudyPlanResponse>(res);
  },

  async replanSchedule(
    planId: string,
    payload: {
      current_week: number;
      new_weekly_hours?: number;
      include_spaced_repetition: boolean;
    }
  ): Promise<StudyPlanResponse> {
    const res = await fetch(`${API_BASE}/plans/${encodeURIComponent(planId)}/replan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleResponse<StudyPlanResponse>(res);
  },

  async queryRag(
    planId: string,
    question: string,
    engine?: StudyEngineMode,
    apiKey?: string
  ): Promise<RagQueryResponse> {
    const res = await fetch(`${API_BASE}/plans/${encodeURIComponent(planId)}/rag`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        top_k: 4,
        engine: engine || null,
        api_key: apiKey || null,
      }),
    });
    return handleResponse<RagQueryResponse>(res);
  },

  getExportUrl(planId: string, kind: 'ics' | 'markdown' | 'json'): string {
    return `${API_BASE}/plans/${encodeURIComponent(planId)}/export/${kind}`;
  },

  getSamplePdfUrl(sampleId: string): string {
    return `${API_BASE}/samples/${encodeURIComponent(sampleId)}/pdf`;
  },
};
