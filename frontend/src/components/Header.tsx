import React, { useState } from 'react';
import {
  Activity,
  BrainCircuit,
  CalendarPlus,
  ChevronDown,
  Download,
  FileCode2,
  FileText,
  FolderOpen,
  Plus,
  Settings,
  Trash2,
} from 'lucide-react';
import type { HealthStatus, StudyPlanListItem, StudyPlanResponse } from '../types';
import { apiClient } from '../api/client';

interface HeaderProps {
  health: HealthStatus | null;
  savedPlans: StudyPlanListItem[];
  activePlan: StudyPlanResponse | null;
  onSelectPlan: (planId: string) => void;
  onDeletePlan: (planId: string) => void;
  onNewPlan: () => void;
  onOpenTelemetry: () => void;
  onOpenSettings: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  savedPlans,
  activePlan,
  onSelectPlan,
  onDeletePlan,
  onNewPlan,
  onOpenTelemetry,
  onOpenSettings,
}) => {
  const [plansOpen, setPlansOpen] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);

  return (
    <header className="sticky top-0 z-40 border-b-[2.5px] border-[#2A1810] bg-[#FFFFFF]/95 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-5 py-4 sm:px-8">
        {/* Brand */}
        <div className="flex items-center gap-3.5">
          <div className="brutal-pill flex h-11 w-11 items-center justify-center rounded-xl bg-[#4A2C1D] text-[#FDFBF7]">
            <BrainCircuit className="h-6 w-6" />
          </div>
          <div>
            <div className="flex items-center gap-2.5">
              <span className="font-display text-xl font-bold tracking-tight text-[#2A1810]">
                Syllab<span className="text-[#B45309]">AI</span>
              </span>
              <span className="hidden rounded-md border-[1.5px] border-[#2A1810] bg-[#FEF3C7] px-2 py-0.5 font-mono text-[11px] font-bold text-[#2A1810] sm:inline-block">
                NEURO-SYMBOLIC + MILP
              </span>
            </div>
            <p className="text-xs font-medium text-[#6F4E37]">
              {health?.pytorch_model.ready
                ? `PyTorch CognitiveNet Active (${health.pytorch_model.val_accuracy_pct}% Val Acc)`
                : 'Connecting to AI Engine...'}
            </p>
          </div>
        </div>

        {/* Actions */}
        <div className="flex flex-wrap items-center gap-2.5">
          {/* Saved Plans Dropdown */}
          <div className="relative">
            <button
              onClick={() => {
                setPlansOpen(!plansOpen);
                setExportOpen(false);
              }}
              className="brutal-btn flex items-center gap-2 rounded-xl bg-[#F6EFE6] px-3.5 py-2 text-xs font-bold text-[#2A1810]"
            >
              <FolderOpen className="h-4 w-4 text-[#8B4513]" />
              <span>Saved Plans ({savedPlans.length})</span>
              <ChevronDown className="h-3.5 w-3.5" />
            </button>

            {plansOpen && (
              <div className="brutal-card absolute right-0 mt-3 w-84 p-3 z-50 animate-fade-up">
                <div className="mb-2 border-b border-[#E8DDD4] pb-1.5 font-mono text-[11px] font-bold uppercase tracking-wider text-[#6F4E37]">
                  Persisted Study Plans (SQLite)
                </div>
                {savedPlans.length === 0 ? (
                  <div className="px-3 py-5 text-center text-xs font-medium text-[#6F4E37]">
                    No saved plans yet. Upload a syllabus or run a sample!
                  </div>
                ) : (
                  <div className="max-h-72 space-y-2 overflow-y-auto pr-1">
                    {savedPlans.map((item) => (
                      <div
                        key={item.plan_id}
                        className={`group flex items-center justify-between rounded-xl border-2 p-2.5 transition ${
                          activePlan?.plan_id === item.plan_id
                            ? 'border-[#2A1810] bg-[#FEF3C7]'
                            : 'border-transparent bg-[#FDFBF7] hover:border-[#2A1810]'
                        }`}
                      >
                        <button
                          onClick={() => {
                            onSelectPlan(item.plan_id);
                            setPlansOpen(false);
                          }}
                          className="flex-1 text-left"
                        >
                          <div className="flex items-center gap-2">
                            <span className="rounded bg-[#4A2C1D] px-1.5 py-0.5 font-mono text-[11px] font-bold text-[#FDFBF7]">
                              {item.course_code}
                            </span>
                            <span className="truncate text-xs font-bold text-[#2A1810] max-w-[165px]">
                              {item.course_title}
                            </span>
                          </div>
                          <div className="mt-1 text-[11px] font-medium text-[#6F4E37]">
                            {item.total_weeks}w · {item.weekly_hours}h/wk · {item.completed_tasks}/{item.total_tasks} tasks
                          </div>
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeletePlan(item.plan_id);
                          }}
                          title="Delete plan"
                          className="rounded-lg p-1.5 text-[#6F4E37] opacity-70 transition hover:bg-[#FFE4E6] hover:text-[#9F1239] hover:opacity-100"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {activePlan && (
            <>
              <button
                onClick={onOpenTelemetry}
                className="brutal-btn flex items-center gap-1.5 rounded-xl bg-[#D1FAE5] px-3.5 py-2 text-xs font-bold text-[#2A1810]"
              >
                <Activity className="h-4 w-4 text-[#065F46]" />
                <span className="hidden sm:inline">AI Telemetry</span>
              </button>

              <div className="relative">
                <button
                  onClick={() => {
                    setExportOpen(!exportOpen);
                    setPlansOpen(false);
                  }}
                  className="brutal-btn flex items-center gap-1.5 rounded-xl bg-[#FFFFFF] px-3.5 py-2 text-xs font-bold text-[#2A1810]"
                >
                  <Download className="h-4 w-4 text-[#8B4513]" />
                  <span>Export</span>
                  <ChevronDown className="h-3.5 w-3.5" />
                </button>
                {exportOpen && (
                  <div className="brutal-card absolute right-0 mt-3 w-60 p-2 z-50 animate-fade-up">
                    <a
                      href={apiClient.getExportUrl(activePlan.plan_id, 'ics')}
                      onClick={() => setExportOpen(false)}
                      className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-xs text-[#2A1810] transition hover:bg-[#F6EFE6]"
                    >
                      <CalendarPlus className="h-4 w-4 text-[#065F46]" />
                      <div>
                        <div className="font-bold">Calendar (.ics)</div>
                        <div className="text-[11px] text-[#6F4E37]">Google / Apple / Outlook</div>
                      </div>
                    </a>
                    <a
                      href={apiClient.getExportUrl(activePlan.plan_id, 'markdown')}
                      onClick={() => setExportOpen(false)}
                      className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-xs text-[#2A1810] transition hover:bg-[#F6EFE6]"
                    >
                      <FileText className="h-4 w-4 text-[#8B4513]" />
                      <div>
                        <div className="font-bold">Study Guide (.md)</div>
                        <div className="text-[11px] text-[#6F4E37]">Notion / GitHub checklist</div>
                      </div>
                    </a>
                    <a
                      href={apiClient.getExportUrl(activePlan.plan_id, 'json')}
                      onClick={() => setExportOpen(false)}
                      className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-xs text-[#2A1810] transition hover:bg-[#F6EFE6]"
                    >
                      <FileCode2 className="h-4 w-4 text-[#B45309]" />
                      <div>
                        <div className="font-bold">Structured JSON</div>
                        <div className="text-[11px] text-[#6F4E37]">Full DAG &amp; metrics</div>
                      </div>
                    </a>
                  </div>
                )}
              </div>
            </>
          )}

          <button
            onClick={onOpenSettings}
            title="AI Engine & API Keys"
            className="brutal-btn rounded-xl bg-[#FFFFFF] p-2 text-[#2A1810]"
          >
            <Settings className="h-4 w-4" />
          </button>

          <button
            onClick={onNewPlan}
            className="brutal-btn flex items-center gap-1.5 rounded-xl bg-[#4A2C1D] px-4 py-2 text-xs font-bold text-[#FDFBF7] hover:bg-[#3A2115]"
          >
            <Plus className="h-4 w-4" />
            <span>New Syllabus</span>
          </button>
        </div>
      </div>
    </header>
  );
};
