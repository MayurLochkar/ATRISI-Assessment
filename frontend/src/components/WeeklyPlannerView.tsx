import React, { useEffect, useState } from 'react';
import {
  AlertTriangle,
  BookOpenCheck,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Circle,
  Clock,
  Flame,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  Search,
  Sparkles,
  Timer,
} from 'lucide-react';
import type { ConfidenceRating, StudyPlanResponse, StudyTask, TaskType } from '../types';

interface WeeklyPlannerViewProps {
  plan: StudyPlanResponse;
  onUpdateTask: (
    taskId: string,
    payload: { is_completed?: boolean; confidence_rating?: ConfidenceRating }
  ) => Promise<void>;
  onReplan: (currentWeek: number, newWeeklyHours: number) => Promise<void>;
  isReplanning: boolean;
}

const BLOOM_COLORS: Record<string, string> = {
  Remember: 'bg-[#F6EFE6] text-[#2A1810]',
  Understand: 'bg-[#E0F2FE] text-[#075985]',
  Apply: 'bg-[#D1FAE5] text-[#065F46]',
  Analyze: 'bg-[#FEF3C7] text-[#92400E]',
  Evaluate: 'bg-[#FFEDD5] text-[#9A3412]',
  Create: 'bg-[#FCE7F3] text-[#9D174D]',
};

const TYPE_META: Record<TaskType, { label: string; bg: string }> = {
  concept_deep_dive: { label: 'Concept Deep-Dive', bg: 'bg-[#E0F2FE] text-[#2A1810]' },
  hands_on_practice: { label: 'Hands-On Practice', bg: 'bg-[#D1FAE5] text-[#2A1810]' },
  active_recall: { label: 'FSRS Active Recall', bg: 'bg-[#EDE9FE] text-[#2A1810]' },
  assessment_prep: { label: 'Assessment Prep', bg: 'bg-[#FEF3C7] text-[#2A1810]' },
  project_milestone: { label: 'Project Milestone', bg: 'bg-[#FCE7F3] text-[#2A1810]' },
};

export const WeeklyPlannerView: React.FC<WeeklyPlannerViewProps> = ({
  plan,
  onUpdateTask,
  onReplan,
  isReplanning,
}) => {
  const [selectedWeek, setSelectedWeek] = useState<number | 'all'>(
    plan.weeks[0]?.week_number ?? 1
  );
  const [typeFilter, setTypeFilter] = useState<TaskType | 'all'>('all');
  const [statusFilter, setStatusFilter] = useState<'all' | 'pending' | 'completed'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedTaskId, setExpandedTaskId] = useState<string | null>(null);

  // Adaptive Re-Planner controls
  const [replanWeek, setReplanWeek] = useState<number>(1);
  const [replanHours, setReplanHours] = useState<number>(plan.constraints.weekly_hours);

  // Embedded Focus Pomodoro Timer state
  const [activeTimerTaskId, setActiveTimerTaskId] = useState<string | null>(null);
  const [secondsLeft, setSecondsLeft] = useState<number>(25 * 60);
  const [timerRunning, setTimerRunning] = useState<boolean>(false);

  useEffect(() => {
    setReplanHours(plan.constraints.weekly_hours);
    if (selectedWeek !== 'all' && !plan.weeks.some((w) => w.week_number === selectedWeek)) {
      setSelectedWeek(plan.weeks[0]?.week_number ?? 'all');
    }
  }, [plan, selectedWeek]);

  useEffect(() => {
    if (!timerRunning) return;
    const id = window.setInterval(() => {
      setSecondsLeft((prev) => {
        if (prev <= 1) {
          setTimerRunning(false);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => window.clearInterval(id);
  }, [timerRunning]);

  const allTasks = plan.weeks.flatMap((w) => w.tasks);
  const completedCount = allTasks.filter((t) => t.is_completed).length;
  const progressPct = allTasks.length > 0 ? Math.round((completedCount / allTasks.length) * 100) : 0;

  const visibleWeeks =
    selectedWeek === 'all'
      ? plan.weeks
      : plan.weeks.filter((w) => w.week_number === selectedWeek);

  const startFocusTimer = (task: StudyTask) => {
    if (activeTimerTaskId === task.id) {
      setTimerRunning(!timerRunning);
    } else {
      setActiveTimerTaskId(task.id);
      setSecondsLeft(Math.min(50, Math.max(15, task.estimated_minutes)) * 60);
      setTimerRunning(true);
    }
  };

  const formatTime = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  };

  return (
    <div className="space-y-8 animate-fade-up">
      {/* Feasibility Warning Banner if over capacity */}
      {plan.evaluation.feasibility_warning && (
        <div className="brutal-card flex items-start gap-3 bg-[#FEF3C7] p-5 text-sm text-[#2A1810]">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-[#B45309]" />
          <div>
            <span className="font-bold">MILP Capacity Notice: </span>
            {plan.evaluation.feasibility_warning}
          </div>
        </div>
      )}

      {/* Spacious KPI Summary Strip */}
      <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
        <div className="brutal-card brutal-card-hover p-5">
          <div className="flex items-center justify-between text-xs font-bold uppercase tracking-wider text-[#6F4E37]">
            <span>Completion</span>
            <BookOpenCheck className="h-4 w-4 text-[#065F46]" />
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="font-mono text-3xl font-bold text-[#2A1810]">{progressPct}%</span>
            <span className="text-xs font-semibold text-[#6F4E37]">
              ({completedCount}/{allTasks.length} tasks)
            </span>
          </div>
          <div className="mt-3 h-2.5 w-full overflow-hidden rounded-full border-[1.5px] border-[#2A1810] bg-[#F6EFE6]">
            <div
              className="h-full bg-[#065F46] transition-all duration-300"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        </div>

        <div className="brutal-card brutal-card-hover p-5">
          <div className="flex items-center justify-between text-xs font-bold uppercase tracking-wider text-[#6F4E37]">
            <span>DAG Compliance</span>
            <Sparkles className="h-4 w-4 text-[#B45309]" />
          </div>
          <div className="mt-3 font-mono text-3xl font-bold text-[#4A2C1D]">
            {plan.evaluation.prerequisite_compliance_rate}%
          </div>
          <p className="mt-2 text-xs font-medium text-[#6F4E37]">
            Critical Path: {plan.evaluation.critical_path_length} core topics
          </p>
        </div>

        <div className="brutal-card brutal-card-hover p-5">
          <div className="flex items-center justify-between text-xs font-bold uppercase tracking-wider text-[#6F4E37]">
            <span>Load Balance</span>
            <Flame className="h-4 w-4 text-[#C86D2D]" />
          </div>
          <div className="mt-3 font-mono text-3xl font-bold text-[#2A1810]">
            {plan.evaluation.load_balance_score}%
          </div>
          <p className="mt-2 text-xs font-medium text-[#6F4E37]">
            {plan.evaluation.total_study_hours}h total · {plan.evaluation.avg_weekly_utilization}% avg util
          </p>
        </div>

        {/* Adaptive Re-Planner Card */}
        <div className="brutal-card bg-[#F6EFE6] p-5">
          <div className="flex items-center justify-between text-xs font-bold uppercase tracking-wider text-[#2A1810]">
            <span>Adaptive Re-Planner</span>
            <RefreshCw className={`h-4 w-4 text-[#4A2C1D] ${isReplanning ? 'animate-spin' : ''}`} />
          </div>
          <div className="mt-3 flex items-center gap-2">
            <select
              value={replanWeek}
              onChange={(e) => setReplanWeek(Number(e.target.value))}
              className="rounded-lg border-2 border-[#2A1810] bg-[#FFFFFF] px-2.5 py-1.5 text-xs font-bold text-[#2A1810]"
            >
              {plan.weeks.map((w) => (
                <option key={w.week_number} value={w.week_number}>
                  From W{w.week_number}
                </option>
              ))}
            </select>
            <input
              type="number"
              min={2}
              max={60}
              step={0.5}
              value={replanHours}
              onChange={(e) => setReplanHours(Number(e.target.value))}
              className="w-16 rounded-lg border-2 border-[#2A1810] bg-[#FFFFFF] px-2 py-1.5 font-mono text-xs font-bold text-[#2A1810]"
              title="New weekly hours"
            />
            <span className="text-xs font-bold text-[#6F4E37]">h/wk</span>
          </div>
          <button
            disabled={isReplanning}
            onClick={() => onReplan(replanWeek, replanHours)}
            className="brutal-btn mt-3 w-full rounded-xl bg-[#4A2C1D] py-2 text-xs font-bold text-[#FDFBF7] disabled:opacity-50"
          >
            {isReplanning ? 'Re-Solving MILP...' : 'Re-Balance Remaining'}
          </button>
        </div>
      </div>

      {/* Week Selector & Search / Filter Bar */}
      <div className="brutal-card p-5 space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => setSelectedWeek('all')}
            className={`brutal-btn rounded-xl px-3.5 py-2 text-xs font-bold ${
              selectedWeek === 'all'
                ? 'bg-[#4A2C1D] text-[#FDFBF7]'
                : 'bg-[#FDFBF7] text-[#2A1810]'
            }`}
          >
            All Weeks
          </button>
          {plan.weeks.map((w) => {
            const done = w.tasks.filter((t) => t.is_completed).length;
            const total = w.tasks.length;
            const active = selectedWeek === w.week_number;
            return (
              <button
                key={w.week_number}
                onClick={() => setSelectedWeek(w.week_number)}
                className={`brutal-btn flex items-center gap-1.5 rounded-xl px-3 py-2 text-xs font-bold ${
                  active
                    ? 'bg-[#4A2C1D] text-[#FDFBF7]'
                    : w.is_exam_week
                    ? 'bg-[#FEF3C7] text-[#2A1810]'
                    : 'bg-[#FFFFFF] text-[#2A1810]'
                }`}
              >
                <span>W{w.week_number}</span>
                {total > 0 && (
                  <span className="font-mono text-[11px] opacity-80">
                    {done}/{total}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        <div className="flex flex-wrap items-center gap-3 border-t-2 border-[#E8DDD4] pt-4">
          <div className="relative flex-1 min-w-[220px]">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[#6F4E37]" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Filter tasks by topic, module, or keyword..."
              className="w-full rounded-xl border-2 border-[#2A1810] bg-[#FDFBF7] py-2 pl-9 pr-3 text-xs font-medium text-[#2A1810] placeholder-[#9C7A65] focus:bg-white focus:outline-none"
            />
          </div>

          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value as TaskType | 'all')}
            className="rounded-xl border-2 border-[#2A1810] bg-[#FDFBF7] px-3 py-2 text-xs font-bold text-[#2A1810] focus:outline-none"
          >
            <option value="all">All Task Modes</option>
            <option value="concept_deep_dive">Concept Deep-Dive</option>
            <option value="hands_on_practice">Hands-On Practice</option>
            <option value="active_recall">FSRS Active Recall</option>
            <option value="assessment_prep">Assessment Prep</option>
            <option value="project_milestone">Project Milestone</option>
          </select>

          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as 'all' | 'pending' | 'completed')}
            className="rounded-xl border-2 border-[#2A1810] bg-[#FDFBF7] px-3 py-2 text-xs font-bold text-[#2A1810] focus:outline-none"
          >
            <option value="all">All Statuses</option>
            <option value="pending">Pending Only</option>
            <option value="completed">Completed Only</option>
          </select>
        </div>
      </div>

      {/* Active Focus Timer Floating Bar */}
      {activeTimerTaskId && (
        <div className="brutal-card flex flex-wrap items-center justify-between gap-4 bg-[#FEF3C7] px-6 py-4 animate-fade-up">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl border-2 border-[#2A1810] bg-[#FFFFFF]">
              <Timer className="h-5 w-5 text-[#B45309] animate-pulse" />
            </div>
            <div>
              <div className="font-mono text-xs font-bold uppercase tracking-wider text-[#92400E]">
                Deep Focus Pomodoro Active
              </div>
              <div className="text-sm font-bold text-[#2A1810]">
                {allTasks.find((t) => t.id === activeTimerTaskId)?.title}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="rounded-xl border-2 border-[#2A1810] bg-[#FFFFFF] px-4 py-1.5 font-mono text-xl font-bold text-[#2A1810]">
              {formatTime(secondsLeft)}
            </span>
            <button
              onClick={() => setTimerRunning(!timerRunning)}
              className="brutal-btn rounded-xl bg-[#4A2C1D] p-2.5 text-[#FDFBF7]"
            >
              {timerRunning ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </button>
            <button
              onClick={() => {
                setTimerRunning(false);
                setSecondsLeft(25 * 60);
              }}
              className="brutal-btn rounded-xl bg-[#FFFFFF] p-2.5 text-[#2A1810]"
            >
              <RotateCcw className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      {/* Weeks & Tasks List */}
      <div className="space-y-8">
        {visibleWeeks.map((week) => {
          const q = searchQuery.trim().toLowerCase();
          const filteredTasks = week.tasks.filter((t) => {
            if (typeFilter !== 'all' && t.task_type !== typeFilter) return false;
            if (statusFilter === 'pending' && t.is_completed) return false;
            if (statusFilter === 'completed' && !t.is_completed) return false;
            if (
              q &&
              !t.title.toLowerCase().includes(q) &&
              !t.description.toLowerCase().includes(q)
            ) {
              return false;
            }
            return true;
          });

          return (
            <div key={week.week_number} className="brutal-card p-6 sm:p-8">
              <div className="flex flex-wrap items-center justify-between gap-4 border-b-2 border-[#2A1810] pb-5">
                <div>
                  <div className="flex flex-wrap items-center gap-2.5">
                    <span className="rounded-lg border-2 border-[#2A1810] bg-[#4A2C1D] px-3 py-1 font-mono text-xs font-bold text-[#FDFBF7]">
                      {week.week_label}
                    </span>
                    <span className="font-mono text-xs font-semibold text-[#6F4E37]">
                      {week.date_range}
                    </span>
                    {week.milestone_alert && (
                      <span className="rounded-lg border-[1.5px] border-[#2A1810] bg-[#FEF3C7] px-3 py-1 text-xs font-bold text-[#92400E]">
                        {week.milestone_alert}
                      </span>
                    )}
                  </div>
                  <h3 className="mt-2.5 font-display text-lg font-bold text-[#2A1810]">
                    {week.focus_theme}
                  </h3>
                </div>
                <div className="rounded-xl border-2 border-[#2A1810] bg-[#F6EFE6] px-4 py-2 text-right">
                  <div className="font-mono text-sm font-bold text-[#2A1810]">
                    {week.planned_hours}h / {week.capacity_hours}h
                  </div>
                  <div className="text-[11px] font-semibold text-[#6F4E37]">
                    Avg Load: {week.cognitive_load_avg}/10
                  </div>
                </div>
              </div>

              {filteredTasks.length === 0 ? (
                <div className="py-10 text-center text-sm font-medium text-[#6F4E37]">
                  No tasks matching current filters in this week.
                </div>
              ) : (
                <div className="mt-6 space-y-4">
                  {filteredTasks.map((task) => {
                    const isExpanded = expandedTaskId === task.id;
                    const typeInfo = TYPE_META[task.task_type];
                    return (
                      <div
                        key={task.id}
                        className={`rounded-2xl border-2 border-[#2A1810] p-5 transition-all duration-150 ${
                          task.is_completed
                            ? 'bg-[#D1FAE5]/45 opacity-85'
                            : 'bg-[#FDFBF7] shadow-[3px_3px_0px_#2A1810] hover:-translate-x-0.5 hover:-translate-y-0.5 hover:bg-[#FFFFFF] hover:shadow-[5px_5px_0px_#2A1810]'
                        }`}
                      >
                        <div className="flex items-start gap-4">
                          <button
                            onClick={() =>
                              onUpdateTask(task.id, { is_completed: !task.is_completed })
                            }
                            className="mt-0.5 shrink-0 text-[#4A2C1D] transition hover:scale-110"
                            title="Mark complete"
                          >
                            {task.is_completed ? (
                              <CheckCircle2 className="h-6 w-6 text-[#065F46]" />
                            ) : (
                              <Circle className="h-6 w-6" />
                            )}
                          </button>

                          <div className="min-w-0 flex-1">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="font-mono text-xs font-bold text-[#8B4513]">
                                {task.day_label} · {task.scheduled_date}
                              </span>
                              <span
                                className={`rounded-md border border-[#2A1810] px-2 py-0.5 text-[11px] font-bold ${typeInfo.bg}`}
                              >
                                {typeInfo.label}
                              </span>
                              <span
                                className={`rounded-md border border-[#2A1810] px-2 py-0.5 text-[11px] font-bold ${
                                  BLOOM_COLORS[task.bloom_level] || ''
                                }`}
                              >
                                {task.bloom_level}
                              </span>
                              {task.priority === 'critical' && (
                                <span className="rounded-md border border-[#2A1810] bg-[#FFE4E6] px-2 py-0.5 text-[11px] font-bold text-[#9F1239]">
                                  Critical Path
                                </span>
                              )}
                              {task.page_refs.length > 0 && (
                                <span className="font-mono text-[11px] font-semibold text-[#6F4E37]">
                                  p.{task.page_refs.join(',')}
                                </span>
                              )}
                            </div>

                            <button
                              onClick={() => setExpandedTaskId(isExpanded ? null : task.id)}
                              className="mt-2 flex items-center gap-2 text-left group"
                            >
                              <h4
                                className={`font-display text-base font-bold ${
                                  task.is_completed
                                    ? 'line-through text-[#6F4E37]'
                                    : 'text-[#2A1810] group-hover:text-[#8B4513]'
                                }`}
                              >
                                {task.title}
                              </h4>
                              {isExpanded ? (
                                <ChevronUp className="h-4 w-4 text-[#6F4E37]" />
                              ) : (
                                <ChevronDown className="h-4 w-4 text-[#6F4E37]" />
                              )}
                            </button>
                            <p className="mt-1 text-sm text-[#6F4E37] leading-relaxed">
                              {task.description}
                            </p>

                            {isExpanded && (
                              <div className="mt-4 space-y-3 rounded-xl border-2 border-[#2A1810] bg-[#FFFFFF] p-4 text-xs animate-fade-up">
                                {task.deliverable && (
                                  <div className="text-[#2A1810]">
                                    <span className="font-bold text-[#8B4513]">Deliverable: </span>
                                    <span className="font-medium">{task.deliverable}</span>
                                  </div>
                                )}
                                {task.substeps.length > 0 && (
                                  <div>
                                    <div className="font-bold text-[#2A1810]">Actionable Substeps:</div>
                                    <ul className="mt-1.5 list-disc space-y-1.5 pl-5 text-[#4A2C1D]">
                                      {task.substeps.map((st, idx) => (
                                        <li key={idx}>{st}</li>
                                      ))}
                                    </ul>
                                  </div>
                                )}
                                <div className="flex flex-wrap items-center justify-between gap-3 border-t-2 border-[#E8DDD4] pt-3">
                                  <div className="font-mono text-xs text-[#4A2C1D]">
                                    FSRS Stability: <strong>{task.fsrs_stability_days}d</strong> ·
                                    Recall Prob: <strong className="text-[#065F46]">{Math.round(task.predicted_retention * 100)}%</strong>
                                  </div>
                                  <div className="flex items-center gap-2">
                                    <span className="text-xs font-bold text-[#6F4E37]">Rate Recall:</span>
                                    {(['hard', 'good', 'easy'] as ConfidenceRating[]).map((rating) => (
                                      <button
                                        key={rating}
                                        onClick={() =>
                                          onUpdateTask(task.id, {
                                            is_completed: true,
                                            confidence_rating: rating,
                                          })
                                        }
                                        className={`brutal-btn rounded-lg px-3 py-1 text-xs font-bold capitalize ${
                                          task.confidence_rating === rating
                                            ? 'bg-[#4A2C1D] text-[#FDFBF7]'
                                            : 'bg-[#F6EFE6] text-[#2A1810]'
                                        }`}
                                      >
                                        {rating}
                                      </button>
                                    ))}
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>

                          <div className="flex shrink-0 flex-col items-end gap-2.5">
                            <span className="flex items-center gap-1.5 rounded-lg border border-[#2A1810] bg-[#FFFFFF] px-2.5 py-1 font-mono text-xs font-bold text-[#2A1810]">
                              <Clock className="h-3.5 w-3.5 text-[#8B4513]" />
                              {task.estimated_minutes}m
                            </span>
                            <button
                              onClick={() => startFocusTimer(task)}
                              title="Start focus timer"
                              className={`brutal-btn rounded-lg px-3 py-1.5 text-xs font-bold ${
                                activeTimerTaskId === task.id && timerRunning
                                  ? 'bg-[#FEF3C7] text-[#2A1810]'
                                  : 'bg-[#FFFFFF] text-[#2A1810]'
                              }`}
                            >
                              {activeTimerTaskId === task.id && timerRunning ? 'Pause' : 'Focus'}
                            </button>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
