import React from 'react';
import { Activity, AlertCircle, CheckCircle2, X } from 'lucide-react';
import type { PipelineStageMetric } from '../types';

interface PipelineTraceModalProps {
  isOpen: boolean;
  onClose: () => void;
  telemetry: PipelineStageMetric[];
  engineUsed: string;
}

export const PipelineTraceModal: React.FC<PipelineTraceModalProps> = ({
  isOpen,
  onClose,
  telemetry,
  engineUsed,
}) => {
  if (!isOpen) return null;

  const totalMs = telemetry.reduce((acc, s) => acc + s.duration_ms, 0);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#2A1810]/50 p-4 backdrop-blur-sm">
      <div className="brutal-card w-full max-w-2xl p-6 animate-fade-up">
        <div className="flex items-center justify-between border-b-2 border-[#2A1810] pb-4">
          <div className="flex items-center gap-2.5">
            <Activity className="h-5 w-5 text-[#065F46]" />
            <div>
              <h2 className="font-display text-lg font-bold text-[#2A1810]">
                AI/ML Pipeline Execution Telemetry
              </h2>
              <p className="font-mono text-xs font-semibold text-[#6F4E37]">
                Engine: {engineUsed} · Total Latency: {totalMs.toFixed(1)} ms
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="brutal-btn rounded-lg bg-[#F6EFE6] p-1.5 text-[#2A1810]"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="mt-5 max-h-[60vh] space-y-3 overflow-y-auto pr-1">
          {telemetry.map((st, idx) => (
            <div
              key={idx}
              className="rounded-xl border-2 border-[#2A1810] bg-[#FDFBF7] p-4"
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  {st.status === 'success' ? (
                    <CheckCircle2 className="h-4 w-4 text-[#065F46]" />
                  ) : (
                    <AlertCircle className="h-4 w-4 text-[#B45309]" />
                  )}
                  <span className="text-sm font-bold text-[#2A1810]">{st.stage}</span>
                </div>
                <span className="rounded-md border border-[#2A1810] bg-[#FEF3C7] px-2.5 py-0.5 font-mono text-xs font-bold text-[#2A1810]">
                  {st.duration_ms} ms
                </span>
              </div>
              <div className="mt-1.5 font-mono text-xs font-bold text-[#8B4513]">
                {st.model_or_algorithm}
              </div>
              <p className="mt-1 text-xs font-medium text-[#6F4E37]">{st.details}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
