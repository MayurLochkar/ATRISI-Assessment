import { useCallback, useEffect, useState } from 'react';
import {
  AlertCircle,
  BarChart2,
  CalendarDays,
  GitBranch,
  MessageSquareText,
} from 'lucide-react';
import { apiClient, type GenerateOptions } from './api/client';
import type {
  ConfidenceRating,
  HealthStatus,
  SampleSyllabus,
  StudyEngineMode,
  StudyPlanListItem,
  StudyPlanResponse,
} from './types';
import { Header } from './components/Header';
import { UploadWizard } from './components/UploadWizard';
import { WeeklyPlannerView } from './components/WeeklyPlannerView';
import { KnowledgeGraphView } from './components/KnowledgeGraphView';
import { AnalyticsView } from './components/AnalyticsView';
import { SyllabusRagChat } from './components/SyllabusRagChat';
import { PipelineTraceModal } from './components/PipelineTraceModal';
import { SettingsModal } from './components/SettingsModal';

type DashboardTab = 'planner' | 'graph' | 'analytics' | 'rag';

export function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [samples, setSamples] = useState<SampleSyllabus[]>([]);
  const [savedPlans, setSavedPlans] = useState<StudyPlanListItem[]>([]);
  const [activePlan, setActivePlan] = useState<StudyPlanResponse | null>(null);

  const [activeTab, setActiveTab] = useState<DashboardTab>('planner');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isReplanning, setIsReplanning] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [telemetryOpen, setTelemetryOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const [engine, setEngine] = useState<StudyEngineMode>(() => {
    const saved = localStorage.getItem('syllabai_engine');
    return (saved as StudyEngineMode) || 'local_hybrid';
  });
  const [apiKey, setApiKey] = useState<string>(
    () => localStorage.getItem('syllabai_apikey') || ''
  );

  const refreshMetadata = useCallback(async () => {
    try {
      const [h, s, p] = await Promise.all([
        apiClient.getHealth(),
        apiClient.getSamples(),
        apiClient.listPlans(),
      ]);
      setHealth(h);
      setSamples(s);
      setSavedPlans(p);
      setErrorMsg((prev) =>
        prev && prev.startsWith('Backend connection warning') ? null : prev
      );
    } catch (err) {
      setErrorMsg(
        err instanceof Error
          ? `Backend connection warning: ${err.message}. Ensure FastAPI is running on port 8000.`
          : 'Unable to reach backend API.'
      );
    }
  }, []);

  useEffect(() => {
    refreshMetadata();
    const timer = window.setInterval(() => {
      if (!health) {
        refreshMetadata();
      }
    }, 5000);
    return () => window.clearInterval(timer);
  }, [health, refreshMetadata]);

  const handlePlanCreated = async (plan: StudyPlanResponse) => {
    setActivePlan(plan);
    setActiveTab('planner');
    setErrorMsg(null);
    const plans = await apiClient.listPlans().catch(() => savedPlans);
    setSavedPlans(plans);
  };

  const handleGeneratePdf = async (file: File, opts: GenerateOptions) => {
    setIsGenerating(true);
    setErrorMsg(null);
    try {
      const plan = await apiClient.generateFromPdf(file, opts);
      await handlePlanCreated(plan);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'PDF generation failed');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleGenerateSample = async (sampleId: string, opts: GenerateOptions) => {
    setIsGenerating(true);
    setErrorMsg(null);
    try {
      const plan = await apiClient.generateFromSample(sampleId, opts);
      await handlePlanCreated(plan);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Sample generation failed');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleGenerateText = async (
    text: string,
    opts: GenerateOptions,
    titleHint?: string
  ) => {
    setIsGenerating(true);
    setErrorMsg(null);
    try {
      const plan = await apiClient.generateFromText(text, opts, titleHint);
      await handlePlanCreated(plan);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Text generation failed');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSelectPlan = async (planId: string) => {
    setErrorMsg(null);
    try {
      const plan = await apiClient.getPlan(planId);
      setActivePlan(plan);
      setActiveTab('planner');
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Failed to load plan');
    }
  };

  const handleDeletePlan = async (planId: string) => {
    try {
      await apiClient.deletePlan(planId);
      if (activePlan?.plan_id === planId) {
        setActivePlan(null);
      }
      const plans = await apiClient.listPlans();
      setSavedPlans(plans);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Failed to delete plan');
    }
  };

  const handleUpdateTask = async (
    taskId: string,
    payload: { is_completed?: boolean; confidence_rating?: ConfidenceRating }
  ) => {
    if (!activePlan) return;
    try {
      const updated = await apiClient.updateTask(activePlan.plan_id, taskId, payload);
      setActivePlan(updated);
      const plans = await apiClient.listPlans().catch(() => savedPlans);
      setSavedPlans(plans);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Failed to update task');
    }
  };

  const handleReplan = async (currentWeek: number, newWeeklyHours: number) => {
    if (!activePlan) return;
    setIsReplanning(true);
    setErrorMsg(null);
    try {
      const updated = await apiClient.replanSchedule(activePlan.plan_id, {
        current_week: currentWeek,
        new_weekly_hours: newWeeklyHours,
        include_spaced_repetition: activePlan.constraints.include_spaced_repetition,
      });
      setActivePlan(updated);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Adaptive replan failed');
    } finally {
      setIsReplanning(false);
    }
  };

  return (
    <div className="min-h-screen text-[#2A1810]">
      <Header
        health={health}
        savedPlans={savedPlans}
        activePlan={activePlan}
        onSelectPlan={handleSelectPlan}
        onDeletePlan={handleDeletePlan}
        onNewPlan={() => setActivePlan(null)}
        onOpenTelemetry={() => setTelemetryOpen(true)}
        onOpenSettings={() => setSettingsOpen(true)}
      />

      {errorMsg && (
        <div className="mx-auto mt-5 max-w-7xl px-5 sm:px-8">
          <div className="brutal-card flex items-center justify-between gap-3 bg-[#FFE4E6] px-5 py-3.5 text-sm font-bold text-[#9F1239]">
            <div className="flex items-center gap-2.5">
              <AlertCircle className="h-5 w-5 shrink-0" />
              <span>{errorMsg}</span>
            </div>
            <button
              onClick={() => setErrorMsg(null)}
              className="brutal-btn rounded-lg bg-[#FFFFFF] px-3 py-1 text-xs font-bold text-[#2A1810]"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {!activePlan ? (
        <UploadWizard
          samples={samples}
          engine={engine}
          apiKey={apiKey}
          isGenerating={isGenerating}
          onGeneratePdf={handleGeneratePdf}
          onGenerateSample={handleGenerateSample}
          onGenerateText={handleGenerateText}
        />
      ) : (
        <main className="mx-auto max-w-7xl px-5 py-8 sm:px-8 space-y-8">
          {/* Spacious Course Header & Tab Navigation */}
          <div className="brutal-card flex flex-wrap items-center justify-between gap-5 p-6">
            <div>
              <div className="flex flex-wrap items-center gap-2.5">
                <span className="rounded-lg border-2 border-[#2A1810] bg-[#4A2C1D] px-3 py-1 font-mono text-xs font-bold text-[#FDFBF7]">
                  {activePlan.metadata.course_code}
                </span>
                <span className="rounded-lg border-[1.5px] border-[#2A1810] bg-[#FEF3C7] px-2.5 py-1 font-mono text-xs font-bold text-[#2A1810]">
                  {activePlan.metadata.extraction_method}
                </span>
                {activePlan.metadata.instructor && (
                  <span className="text-xs font-bold text-[#6F4E37]">
                    {activePlan.metadata.instructor}
                  </span>
                )}
              </div>
              <h1 className="mt-3 font-display text-2xl font-bold tracking-tight text-[#2A1810] sm:text-3xl">
                {activePlan.metadata.course_title}
              </h1>
            </div>

            <nav className="flex flex-wrap gap-2 rounded-2xl border-2 border-[#2A1810] bg-[#F6EFE6] p-1.5">
              {[
                { id: 'planner', label: 'Weekly Planner', icon: CalendarDays },
                { id: 'graph', label: 'Prerequisite Graph', icon: GitBranch },
                { id: 'analytics', label: 'AI Metrics & FSRS', icon: BarChart2 },
                { id: 'rag', label: 'Ask Syllabus (RAG)', icon: MessageSquareText },
              ].map((tab) => {
                const Icon = tab.icon;
                const active = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as DashboardTab)}
                    className={`flex items-center gap-2 rounded-xl px-4 py-2 text-xs font-bold transition ${
                      active
                        ? 'border-2 border-[#2A1810] bg-[#4A2C1D] text-[#FDFBF7] shadow-[2px_2px_0px_#2A1810]'
                        : 'text-[#6F4E37] hover:bg-[#FFFFFF] hover:text-[#2A1810]'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </nav>
          </div>

          {activeTab === 'planner' && (
            <WeeklyPlannerView
              plan={activePlan}
              onUpdateTask={handleUpdateTask}
              onReplan={handleReplan}
              isReplanning={isReplanning}
            />
          )}
          {activeTab === 'graph' && <KnowledgeGraphView plan={activePlan} />}
          {activeTab === 'analytics' && <AnalyticsView plan={activePlan} />}
          {activeTab === 'rag' && (
            <SyllabusRagChat
              planId={activePlan.plan_id}
              courseCode={activePlan.metadata.course_code}
              engine={engine}
              apiKey={apiKey}
            />
          )}
        </main>
      )}

      {activePlan && (
        <PipelineTraceModal
          isOpen={telemetryOpen}
          onClose={() => setTelemetryOpen(false)}
          telemetry={activePlan.pipeline_telemetry}
          engineUsed={activePlan.engine_used}
        />
      )}

      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        engine={engine}
        onChangeEngine={(eng) => {
          setEngine(eng);
          localStorage.setItem('syllabai_engine', eng);
        }}
        apiKey={apiKey}
        onSaveApiKey={(k) => {
          setApiKey(k);
          localStorage.setItem('syllabai_apikey', k);
        }}
        health={health}
      />
    </div>
  );
}

export default App;
