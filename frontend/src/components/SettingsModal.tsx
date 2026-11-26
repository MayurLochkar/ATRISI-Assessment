import React, { useState } from 'react';
import { KeyRound, ShieldCheck, X } from 'lucide-react';
import type { HealthStatus, StudyEngineMode } from '../types';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  engine: StudyEngineMode;
  onChangeEngine: (engine: StudyEngineMode) => void;
  apiKey: string;
  onSaveApiKey: (key: string) => void;
  health: HealthStatus | null;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  engine,
  onChangeEngine,
  apiKey,
  onSaveApiKey,
  health,
}) => {
  const [draftKey, setDraftKey] = useState(apiKey);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#2A1810]/50 p-4 backdrop-blur-sm">
      <div className="brutal-card w-full max-w-lg p-6 animate-fade-up">
        <div className="flex items-center justify-between border-b-2 border-[#2A1810] pb-4">
          <div className="flex items-center gap-2.5">
            <KeyRound className="h-5 w-5 text-[#8B4513]" />
            <h2 className="font-display text-lg font-bold text-[#2A1810]">
              AI Engine &amp; Optional LLM Keys
            </h2>
          </div>
          <button
            onClick={onClose}
            className="brutal-btn rounded-lg bg-[#F6EFE6] p-1.5 text-[#2A1810]"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="mt-5 space-y-5">
          <div className="rounded-xl border-2 border-[#2A1810] bg-[#D1FAE5] p-4 text-xs text-[#2A1810]">
            <div className="flex items-center gap-2 font-bold">
              <ShieldCheck className="h-4 w-4 text-[#065F46]" />
              Zero-Key Offline Mode Included
            </div>
            <p className="mt-1.5 leading-relaxed text-[#14532D]">
              <strong>Local Hybrid Neuro-Symbolic ML</strong> (PyTorch CognitiveLoadNet + NetworkX DAG + SciPy MILP + FSRS + Hybrid RRF RAG) runs 100% locally with zero API costs.
            </p>
          </div>

          <div>
            <label className="block font-mono text-xs font-bold uppercase tracking-wider text-[#6F4E37]">
              Active AI Orchestration Engine
            </label>
            <div className="mt-2.5 grid grid-cols-2 gap-3">
              {[
                { id: 'local_hybrid', label: 'Local Hybrid AI/ML', sub: 'Zero-key PyTorch + MILP' },
                {
                  id: 'openai',
                  label: 'OpenAI Structured Agent',
                  sub: health?.engines.openai_configured ? 'Env Key Detected' : 'Requires Key',
                },
                {
                  id: 'gemini',
                  label: 'Google Gemini Agent',
                  sub: health?.engines.gemini_configured ? 'Env Key Detected' : 'Requires Key',
                },
                { id: 'ollama', label: 'Local Ollama', sub: health?.engines.ollama_url || 'localhost:11434' },
              ].map((opt) => (
                <button
                  key={opt.id}
                  onClick={() => onChangeEngine(opt.id as StudyEngineMode)}
                  className={`brutal-btn rounded-xl p-3 text-left ${
                    engine === opt.id
                      ? 'bg-[#FEF3C7] text-[#2A1810]'
                      : 'bg-[#FDFBF7] text-[#2A1810]'
                  }`}
                >
                  <div className="text-xs font-bold">{opt.label}</div>
                  <div className="mt-0.5 text-[11px] font-medium text-[#6F4E37]">{opt.sub}</div>
                </button>
              ))}
            </div>
          </div>

          {(engine === 'openai' || engine === 'gemini') && (
            <div>
              <label className="block text-xs font-bold text-[#2A1810]">
                Optional Runtime {engine === 'openai' ? 'OpenAI' : 'Gemini'} API Key
              </label>
              <input
                type="password"
                value={draftKey}
                onChange={(e) => setDraftKey(e.target.value)}
                placeholder={engine === 'openai' ? 'sk-...' : 'AIza...'}
                className="mt-1.5 w-full rounded-xl border-2 border-[#2A1810] bg-[#FDFBF7] px-3.5 py-2.5 font-mono text-xs text-[#2A1810] placeholder-[#9C7A65] focus:bg-white focus:outline-none"
              />
              <p className="mt-1.5 text-[11px] font-medium text-[#6F4E37]">
                Stored only in browser localStorage and sent per-request; falls back to Local Hybrid ML if empty.
              </p>
            </div>
          )}
        </div>

        <div className="mt-6 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="brutal-btn rounded-xl bg-[#F6EFE6] px-4 py-2 text-xs font-bold text-[#2A1810]"
          >
            Cancel
          </button>
          <button
            onClick={() => {
              onSaveApiKey(draftKey.trim());
              onClose();
            }}
            className="brutal-btn rounded-xl bg-[#4A2C1D] px-5 py-2 text-xs font-bold text-[#FDFBF7]"
          >
            Save Preferences
          </button>
        </div>
      </div>
    </div>
  );
};
