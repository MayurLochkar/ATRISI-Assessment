import React from 'react';
import { Award, BarChart3, Brain, TrendingUp } from 'lucide-react';
import type { StudyPlanResponse } from '../types';

interface AnalyticsViewProps {
  plan: StudyPlanResponse;
}

export const AnalyticsView: React.FC<AnalyticsViewProps> = ({ plan }) => {
  const ev = plan.evaluation;
  const maxHrs = Math.max(
    plan.constraints.weekly_hours * 1.15,
    ...plan.weeks.map((w) => w.planned_hours),
    1
  );

  const bloomEntries = Object.entries(ev.bloom_distribution);
  const maxBloom = Math.max(1, ...bloomEntries.map(([, count]) => count));

  const retPoints = ev.retention_projection;
  const chartW = 540;
  const chartH = 200;
  const padL = 40;
  const padR = 18;
  const padT = 18;
  const padB = 28;
  const innerW = chartW - padL - padR;
  const innerH = chartH - padT - padB;

  const toX = (idx: number) =>
    retPoints.length <= 1
      ? padL
      : padL + (idx / (retPoints.length - 1)) * innerW;
  const toY = (pct: number) => padT + ((100 - pct) / 100) * innerH;

  const fsrsPath = retPoints
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(1)} ${toY(p.with_fsrs_pct).toFixed(1)}`)
    .join(' ');
  const noFsrsPath = retPoints
    .map(
      (p, i) =>
        `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(1)} ${toY(p.without_fsrs_pct).toFixed(1)}`
    )
    .join(' ');

  return (
    <div className="space-y-8 animate-fade-up">
      {/* Top Quantitative Model & Solver Scorecard */}
      <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
        <div className="brutal-card brutal-card-hover bg-[#D1FAE5] p-5">
          <div className="text-xs font-bold uppercase tracking-wider text-[#065F46]">
            PyTorch CognitiveLoadNet
          </div>
          <div className="mt-2 font-mono text-2xl font-bold text-[#2A1810]">
            {ev.pytorch_model_val_accuracy}% Val Acc
          </div>
          <div className="mt-1 font-mono text-xs font-semibold text-[#14532D]">
            Load Regression MAE: {ev.pytorch_model_load_mae}
          </div>
        </div>
        <div className="brutal-card brutal-card-hover bg-[#FEF3C7] p-5">
          <div className="text-xs font-bold uppercase tracking-wider text-[#92400E]">
            MILP Solver Convergence
          </div>
          <div className="mt-2 font-mono text-2xl font-bold text-[#2A1810]">
            {ev.solver_status}
          </div>
          <div className="mt-1 text-xs font-semibold text-[#78350F]">
            Balance: {ev.load_balance_score}% · Util: {ev.avg_weekly_utilization}%
          </div>
        </div>
        <div className="brutal-card brutal-card-hover bg-[#F6EFE6] p-5">
          <div className="text-xs font-bold uppercase tracking-wider text-[#6F4E37]">
            DAG Prerequisite Compliance
          </div>
          <div className="mt-2 font-mono text-2xl font-bold text-[#2A1810]">
            {ev.prerequisite_compliance_rate}%
          </div>
          <div className="mt-1 text-xs font-semibold text-[#4A2C1D]">
            {plan.edges.length} edges · {ev.critical_path_length} critical nodes
          </div>
        </div>
        <div className="brutal-card brutal-card-hover bg-[#FFFFFF] p-5">
          <div className="text-xs font-bold uppercase tracking-wider text-[#6F4E37]">
            Hybrid RAG Knowledge Index
          </div>
          <div className="mt-2 font-mono text-2xl font-bold text-[#2A1810]">
            {ev.rag_chunk_count} Chunks
          </div>
          <div className="mt-1 text-xs font-semibold text-[#6F4E37]">
            Sparse TF-IDF + Dense PyTorch RRF
          </div>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid gap-8 lg:grid-cols-2">
        {/* Weekly Hours vs Capacity */}
        <div className="brutal-card p-6 sm:p-7">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b-2 border-[#E8DDD4] pb-4">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-[#8B4513]" />
              <h3 className="font-display text-base font-bold text-[#2A1810]">
                Weekly Planned Hours vs. Capacity
              </h3>
            </div>
            <span className="rounded-md border border-[#2A1810] bg-[#F6EFE6] px-2.5 py-0.5 font-mono text-xs font-bold text-[#2A1810]">
              Budget: {plan.constraints.weekly_hours}h/wk
            </span>
          </div>

          <div className="mt-6 flex items-end gap-2.5 h-52 border-b-2 border-[#2A1810] pb-2 pt-4">
            {plan.weeks.map((w) => {
              const hPct = Math.min(100, (w.planned_hours / maxHrs) * 100);
              const over = w.planned_hours > w.capacity_hours * 1.05;
              return (
                <div
                  key={w.week_number}
                  className="group flex flex-1 flex-col items-center gap-1.5 h-full justify-end"
                >
                  <span className="font-mono text-[10px] font-bold text-[#2A1810] opacity-0 transition group-hover:opacity-100">
                    {w.planned_hours}h
                  </span>
                  <div className="relative flex w-full max-w-[36px] items-end justify-center h-full">
                    <div
                      className={`w-full rounded-t-lg border-2 border-[#2A1810] transition-all group-hover:-translate-y-0.5 ${
                        over
                          ? 'bg-[#F59E0B]'
                          : w.is_exam_week
                          ? 'bg-[#C86D2D]'
                          : 'bg-[#4A2C1D]'
                      }`}
                      style={{ height: `${Math.max(6, hPct)}%` }}
                      title={`${w.week_label}: ${w.planned_hours}h / ${w.capacity_hours}h`}
                    />
                  </div>
                  <span className="font-mono text-[11px] font-bold text-[#6F4E37]">
                    W{w.week_number}
                  </span>
                </div>
              );
            })}
          </div>
          <div className="mt-4 flex flex-wrap gap-5 text-xs font-semibold text-[#6F4E37]">
            <span className="flex items-center gap-2">
              <span className="h-3 w-3 rounded border border-[#2A1810] bg-[#4A2C1D]" /> Standard Week
            </span>
            <span className="flex items-center gap-2">
              <span className="h-3 w-3 rounded border border-[#2A1810] bg-[#C86D2D]" /> Exam / Milestone Week
            </span>
            <span className="flex items-center gap-2">
              <span className="h-3 w-3 rounded border border-[#2A1810] bg-[#F59E0B]" /> Capacity Exceeded
            </span>
          </div>
        </div>

        {/* FSRS Memory Retention Curve */}
        <div className="brutal-card p-6 sm:p-7">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b-2 border-[#E8DDD4] pb-4">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-[#065F46]" />
              <h3 className="font-display text-base font-bold text-[#2A1810]">
                FSRS Memory Retention Projection
              </h3>
            </div>
            <span className="rounded-md border border-[#2A1810] bg-[#D1FAE5] px-2.5 py-0.5 font-mono text-xs font-bold text-[#065F46]">
              R(t) = exp(-t/S)
            </span>
          </div>

          <div className="mt-5 overflow-x-auto">
            <svg viewBox={`0 0 ${chartW} ${chartH}`} className="w-full h-52">
              {[25, 50, 75, 100].map((tick) => (
                <g key={tick}>
                  <line
                    x1={padL}
                    x2={chartW - padR}
                    y1={toY(tick)}
                    y2={toY(tick)}
                    stroke="#E8DDD4"
                    strokeDasharray="4 4"
                  />
                  <text
                    x={padL - 8}
                    y={toY(tick) + 4}
                    textAnchor="end"
                    fill="#6F4E37"
                    fontSize="10.5"
                    fontFamily="JetBrains Mono, monospace"
                    fontWeight="600"
                  >
                    {tick}%
                  </text>
                </g>
              ))}

              <path
                d={noFsrsPath}
                fill="none"
                stroke="#BE123C"
                strokeWidth="2.5"
                strokeDasharray="6 4"
              />
              <path d={fsrsPath} fill="none" stroke="#065F46" strokeWidth="3.2" />

              {retPoints.map((pt, i) => (
                <g key={pt.week}>
                  <circle
                    cx={toX(i)}
                    cy={toY(pt.with_fsrs_pct)}
                    r="4.2"
                    fill="#065F46"
                    stroke="#FFFFFF"
                    strokeWidth="1.5"
                  />
                  <circle cx={toX(i)} cy={toY(pt.without_fsrs_pct)} r="3.2" fill="#BE123C" />
                  <text
                    x={toX(i)}
                    y={chartH - 6}
                    textAnchor="middle"
                    fill="#6F4E37"
                    fontSize="10"
                    fontFamily="JetBrains Mono, monospace"
                    fontWeight="700"
                  >
                    W{pt.week}
                  </text>
                </g>
              ))}
            </svg>
          </div>
          <div className="mt-3 flex flex-wrap gap-6 text-xs font-bold">
            <span className="flex items-center gap-2 text-[#065F46]">
              <span className="h-1 w-5 bg-[#065F46] inline-block" /> With FSRS Active Recall
            </span>
            <span className="flex items-center gap-2 text-[#BE123C]">
              <span className="h-1 w-5 border-t-[3px] border-dashed border-[#BE123C] inline-block" /> Without Review (Decay)
            </span>
          </div>
        </div>
      </div>

      {/* Bloom Distribution & Extracted Milestones */}
      <div className="grid gap-8 lg:grid-cols-2">
        <div className="brutal-card p-6 sm:p-7">
          <div className="flex items-center gap-2 border-b-2 border-[#E8DDD4] pb-4">
            <Brain className="h-5 w-5 text-[#8B4513]" />
            <h3 className="font-display text-base font-bold text-[#2A1810]">
              Bloom&apos;s Taxonomy Distribution (PyTorch)
            </h3>
          </div>
          <div className="mt-5 space-y-4">
            {bloomEntries.map(([level, count]) => (
              <div key={level} className="space-y-1.5">
                <div className="flex justify-between text-xs font-bold">
                  <span className="text-[#2A1810]">{level}</span>
                  <span className="font-mono text-[#6F4E37]">{count} topic(s)</span>
                </div>
                <div className="h-3 w-full overflow-hidden rounded-full border-[1.5px] border-[#2A1810] bg-[#FDFBF7]">
                  <div
                    className="h-full bg-[#4A2C1D]"
                    style={{ width: `${(count / maxBloom) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="brutal-card p-6 sm:p-7">
          <div className="flex items-center gap-2 border-b-2 border-[#E8DDD4] pb-4">
            <Award className="h-5 w-5 text-[#B45309]" />
            <h3 className="font-display text-base font-bold text-[#2A1810]">
              Extracted Assessments &amp; Deadlines
            </h3>
          </div>
          <div className="mt-5 space-y-3">
            {plan.assessments.map((a) => (
              <div
                key={a.id}
                className="flex items-center justify-between rounded-xl border-2 border-[#2A1810] bg-[#FDFBF7] px-4 py-3"
              >
                <div>
                  <div className="text-sm font-bold text-[#2A1810]">{a.title}</div>
                  <div className="mt-0.5 font-mono text-xs font-semibold uppercase text-[#6F4E37]">
                    {a.assessment_type}
                    {a.target_week ? ` · Target Week ${a.target_week}` : ''}
                  </div>
                </div>
                {a.weight_percent !== null && (
                  <span className="rounded-lg border-[1.5px] border-[#2A1810] bg-[#FEF3C7] px-3 py-1 font-mono text-xs font-bold text-[#2A1810]">
                    {a.weight_percent}%
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
