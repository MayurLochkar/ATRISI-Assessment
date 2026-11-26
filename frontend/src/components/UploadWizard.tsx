import React, { useRef, useState } from 'react';
import {
  BookOpen,
  Calendar,
  Clock,
  Cpu,
  FileDown,
  FileText,
  Layers,
  Loader2,
  Sparkles,
  UploadCloud,
  Zap,
} from 'lucide-react';
import type { SampleSyllabus, StudyEngineMode } from '../types';
import { apiClient, type GenerateOptions } from '../api/client';

interface UploadWizardProps {
  samples: SampleSyllabus[];
  engine: StudyEngineMode;
  apiKey: string;
  isGenerating: boolean;
  onGeneratePdf: (file: File, opts: GenerateOptions) => Promise<void>;
  onGenerateSample: (sampleId: string, opts: GenerateOptions) => Promise<void>;
  onGenerateText: (text: string, opts: GenerateOptions, titleHint?: string) => Promise<void>;
}

const ALL_DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

const PIPELINE_STAGES = [
  '1. PyMuPDF + EasyOCR Layout Extraction & tiktoken Chunking',
  '2. Curriculum & Assessment NLP/LLM Extraction',
  '3. NetworkX Prerequisite DAG & Critical Path Computation',
  '4. PyTorch CognitiveLoadNet Bloom & Effort Prediction',
  '5. SciPy MILP Multi-Objective Schedule Optimization',
  '6. FSRS Exponential Decay Active Recall Injection',
];

export const UploadWizard: React.FC<UploadWizardProps> = ({
  samples,
  engine,
  apiKey,
  isGenerating,
  onGeneratePdf,
  onGenerateSample,
  onGenerateText,
}) => {
  const [inputTab, setInputTab] = useState<'pdf' | 'text'>('pdf');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [rawText, setRawText] = useState('');
  const [titleHint, setTitleHint] = useState('');

  const [totalWeeks, setTotalWeeks] = useState(12);
  const [weeklyHours, setWeeklyHours] = useState(10);
  const [preferredDays, setPreferredDays] = useState<string[]>(['Mon', 'Tue', 'Wed', 'Thu', 'Sat']);
  const [startDate, setStartDate] = useState('');
  const [learningIntensity, setLearningIntensity] = useState<
    'balanced' | 'front_loaded' | 'exam_focused'
  >('balanced');
  const [includeSpacedRepetition, setIncludeSpacedRepetition] = useState(true);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const buildOptions = (overrideWeeks?: number, overrideHours?: number): GenerateOptions => ({
    total_weeks: overrideWeeks ?? totalWeeks,
    weekly_hours: overrideHours ?? weeklyHours,
    preferred_days: preferredDays.length > 0 ? preferredDays : ['Mon', 'Wed', 'Fri'],
    start_date: startDate,
    learning_intensity: learningIntensity,
    include_spaced_repetition: includeSpacedRepetition,
    engine,
    api_key: apiKey || undefined,
  });

  const toggleDay = (day: string) => {
    setPreferredDays((prev) =>
      prev.includes(day) ? (prev.length > 1 ? prev.filter((d) => d !== day) : prev) : [...prev, day]
    );
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f && f.name.toLowerCase().endsWith('.pdf')) {
      setSelectedFile(f);
      setInputTab('pdf');
    }
  };

  const handleSubmit = async () => {
    if (inputTab === 'pdf' && selectedFile) {
      await onGeneratePdf(selectedFile, buildOptions());
    } else if (inputTab === 'text' && rawText.trim().length >= 40) {
      await onGenerateText(rawText, buildOptions(), titleHint);
    }
  };

  return (
    <div className="mx-auto max-w-7xl px-5 py-10 sm:px-8 space-y-10 animate-fade-up">
      {/* Spacious Neo-Brutalist Hero Banner */}
      <div className="brutal-card bg-[#FFFFFF] p-7 sm:p-10">
        <div className="inline-flex items-center gap-2 rounded-lg border-2 border-[#2A1810] bg-[#FEF3C7] px-3 py-1 font-mono text-xs font-bold text-[#2A1810]">
          <Sparkles className="h-4 w-4 text-[#B45309]" />
          <span>PyMuPDF + PyTorch CognitiveNet + NetworkX DAG + SciPy MILP + FSRS + RAG</span>
        </div>
        <h1 className="mt-4 font-display text-3xl font-bold tracking-tight text-[#2A1810] sm:text-5xl leading-tight">
          Turn Any Syllabus PDF into an{' '}
          <span className="underline decoration-[#C86D2D] decoration-4 underline-offset-4">
            Optimized Weekly Plan
          </span>
        </h1>
        <p className="mt-4 max-w-3xl text-base leading-relaxed text-[#6F4E37]">
          Upload your 5-unit semester syllabus PDF or test with a built-in university syllabus below.
          SyllabAI extracts topics &amp; exams, builds a prerequisite DAG, predicts cognitive load with PyTorch,
          and solves an optimal weekly schedule using SciPy MILP.
        </p>
      </div>

      {/* Main Grid: Generous Gap */}
      <div className="grid gap-8 lg:grid-cols-12 items-start">
        {/* Left: Upload / Paste + Samples */}
        <div className="lg:col-span-7 space-y-8">
          <div className="brutal-card p-6 sm:p-8 space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="inline-flex gap-2 rounded-xl border-2 border-[#2A1810] bg-[#F6EFE6] p-1.5">
                <button
                  onClick={() => setInputTab('pdf')}
                  className={`flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-bold transition ${
                    inputTab === 'pdf'
                      ? 'bg-[#4A2C1D] text-[#FDFBF7] shadow-[2px_2px_0px_#2A1810]'
                      : 'text-[#6F4E37] hover:text-[#2A1810]'
                  }`}
                >
                  <UploadCloud className="h-4 w-4" />
                  <span>Upload Syllabus PDF</span>
                </button>
                <button
                  onClick={() => setInputTab('text')}
                  className={`flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-bold transition ${
                    inputTab === 'text'
                      ? 'bg-[#4A2C1D] text-[#FDFBF7] shadow-[2px_2px_0px_#2A1810]'
                      : 'text-[#6F4E37] hover:text-[#2A1810]'
                  }`}
                >
                  <FileText className="h-4 w-4" />
                  <span>Paste Syllabus Text</span>
                </button>
              </div>
              <span className="rounded-lg border-[1.5px] border-[#2A1810] bg-[#FDFBF7] px-3 py-1 font-mono text-xs font-bold text-[#4A2C1D]">
                Engine: {engine}
              </span>
            </div>

            {inputTab === 'pdf' ? (
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`cursor-pointer rounded-2xl border-[2.5px] border-dashed p-10 text-center transition-all duration-200 ${
                  isDragging
                    ? 'border-[#2A1810] bg-[#FEF3C7] scale-[1.01]'
                    : selectedFile
                    ? 'border-[#2A1810] bg-[#D1FAE5]'
                    : 'border-[#6F4E37] bg-[#FDFBF7] hover:border-[#2A1810] hover:bg-[#F6EFE6]'
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,application/pdf"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) setSelectedFile(f);
                  }}
                />
                <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl border-2 border-[#2A1810] bg-[#FFFFFF] shadow-[3px_3px_0px_#2A1810]">
                  <UploadCloud
                    className={`h-7 w-7 ${
                      selectedFile ? 'text-[#065F46]' : 'text-[#8B4513]'
                    }`}
                  />
                </div>
                {selectedFile ? (
                  <div className="mt-4">
                    <p className="text-base font-bold text-[#2A1810]">{selectedFile.name}</p>
                    <p className="mt-1 font-mono text-xs font-medium text-[#065F46]">
                      {(selectedFile.size / 1024).toFixed(1)} KB · Ready for multi-modal parsing
                    </p>
                  </div>
                ) : (
                  <div className="mt-4">
                    <p className="text-base font-bold text-[#2A1810]">
                      Drag &amp; drop your syllabus PDF here, or click to browse
                    </p>
                    <p className="mt-1.5 text-sm text-[#6F4E37]">
                      Supports native text PDFs (PyMuPDF) &amp; scanned PDFs (EasyOCR fallback)
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-4">
                <input
                  type="text"
                  value={titleHint}
                  onChange={(e) => setTitleHint(e.target.value)}
                  placeholder="Optional Course Title (e.g., CS401: Natural Language Processing)"
                  className="w-full rounded-xl border-2 border-[#2A1810] bg-[#FDFBF7] px-4 py-3 text-sm font-medium text-[#2A1810] placeholder-[#9C7A65] focus:bg-white focus:outline-none"
                />
                <textarea
                  rows={8}
                  value={rawText}
                  onChange={(e) => setRawText(e.target.value)}
                  placeholder="Paste your 5-unit syllabus modules, topics, and exam dates here (min 40 characters)..."
                  className="w-full rounded-xl border-2 border-[#2A1810] bg-[#FDFBF7] p-4 font-mono text-xs text-[#2A1810] placeholder-[#9C7A65] focus:bg-white focus:outline-none"
                />
              </div>
            )}

            <button
              disabled={
                isGenerating ||
                (inputTab === 'pdf' ? !selectedFile : rawText.trim().length < 40)
              }
              onClick={handleSubmit}
              className="brutal-btn flex w-full items-center justify-center gap-2.5 rounded-xl bg-[#4A2C1D] py-4 text-sm font-bold text-[#FDFBF7] hover:bg-[#3A2115] disabled:cursor-not-allowed disabled:opacity-45"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span>Running 6-Stage AI/ML Pipeline...</span>
                </>
              ) : (
                <>
                  <Zap className="h-5 w-5 text-[#FBBF24]" />
                  <span>Generate Optimized Study Plan</span>
                </>
              )}
            </button>

            {isGenerating && (
              <div className="space-y-2 rounded-xl border-2 border-[#2A1810] bg-[#FEF3C7] p-4">
                {PIPELINE_STAGES.map((st) => (
                  <div key={st} className="flex items-center gap-2.5 text-xs font-semibold text-[#2A1810]">
                    <span className="h-2 w-2 animate-ping rounded-full bg-[#B45309]" />
                    <span>{st}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* 1-Click Built-in University Syllabi */}
          <div className="brutal-card p-6 sm:p-8">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b-2 border-[#E8DDD4] pb-4">
              <div className="flex items-center gap-2.5">
                <BookOpen className="h-5 w-5 text-[#8B4513]" />
                <h2 className="font-display text-lg font-bold text-[#2A1810]">
                  1-Click Sample University Syllabus PDFs
                </h2>
              </div>
              <span className="font-mono text-xs font-semibold text-[#6F4E37]">
                Generated via ReportLab
              </span>
            </div>
            <div className="mt-5 grid gap-5 sm:grid-cols-3">
              {samples.map((s) => (
                <div
                  key={s.id}
                  className="brutal-card brutal-card-hover flex flex-col justify-between bg-[#FDFBF7] p-4"
                >
                  <div>
                    <span className="inline-block rounded-md border-[1.5px] border-[#2A1810] bg-[#FEF3C7] px-2 py-0.5 font-mono text-[11px] font-bold text-[#2A1810]">
                      {s.course_code}
                    </span>
                    <h3 className="mt-2.5 font-display text-sm font-bold text-[#2A1810] leading-snug">
                      {s.title}
                    </h3>
                    <p className="mt-1.5 text-xs leading-relaxed text-[#6F4E37]">{s.subtitle}</p>
                  </div>
                  <div className="mt-4 flex items-center gap-2 pt-3 border-t-2 border-[#E8DDD4]">
                    <button
                      disabled={isGenerating}
                      onClick={() =>
                        onGenerateSample(
                          s.id,
                          buildOptions(s.recommended_weeks, s.recommended_hours)
                        )
                      }
                      className="brutal-btn flex-1 rounded-lg bg-[#4A2C1D] px-3 py-2 text-xs font-bold text-[#FDFBF7] disabled:opacity-40"
                    >
                      Run Pipeline
                    </button>
                    <a
                      href={apiClient.getSamplePdfUrl(s.id)}
                      target="_blank"
                      rel="noreferrer"
                      title="Download Sample PDF"
                      className="brutal-btn rounded-lg bg-[#FFFFFF] p-2 text-[#2A1810]"
                    >
                      <FileDown className="h-4 w-4" />
                    </a>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right: Study Constraints Configurator */}
        <div className="lg:col-span-5">
          <div className="brutal-card p-6 sm:p-8 space-y-6">
            <div className="flex items-center gap-2.5 border-b-2 border-[#E8DDD4] pb-4">
              <Layers className="h-5 w-5 text-[#8B4513]" />
              <h2 className="font-display text-lg font-bold text-[#2A1810]">
                MILP Optimizer Constraints
              </h2>
            </div>

            {/* Total Weeks */}
            <div className="rounded-xl border-2 border-[#2A1810] bg-[#FDFBF7] p-4">
              <div className="flex items-center justify-between text-sm">
                <span className="flex items-center gap-2 font-bold text-[#2A1810]">
                  <Calendar className="h-4 w-4 text-[#8B4513]" />
                  Study Horizon
                </span>
                <span className="rounded-md bg-[#4A2C1D] px-2.5 py-0.5 font-mono text-xs font-bold text-[#FDFBF7]">
                  {totalWeeks} Weeks
                </span>
              </div>
              <input
                type="range"
                min={2}
                max={20}
                value={totalWeeks}
                onChange={(e) => setTotalWeeks(Number(e.target.value))}
                className="mt-3 w-full accent-[#4A2C1D]"
              />
            </div>

            {/* Weekly Hours */}
            <div className="rounded-xl border-2 border-[#2A1810] bg-[#FDFBF7] p-4">
              <div className="flex items-center justify-between text-sm">
                <span className="flex items-center gap-2 font-bold text-[#2A1810]">
                  <Clock className="h-4 w-4 text-[#065F46]" />
                  Weekly Capacity Budget
                </span>
                <span className="rounded-md bg-[#D1FAE5] border border-[#2A1810] px-2.5 py-0.5 font-mono text-xs font-bold text-[#2A1810]">
                  {weeklyHours} hrs / wk
                </span>
              </div>
              <input
                type="range"
                min={3}
                max={35}
                step={0.5}
                value={weeklyHours}
                onChange={(e) => setWeeklyHours(Number(e.target.value))}
                className="mt-3 w-full accent-[#065F46]"
              />
            </div>

            {/* Start Date */}
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-[#6F4E37]">
                Start Date (defaults to next Monday if blank)
              </label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="mt-2 w-full rounded-xl border-2 border-[#2A1810] bg-[#FDFBF7] px-3.5 py-2.5 text-sm font-medium text-[#2A1810] focus:bg-white focus:outline-none"
              />
            </div>

            {/* Preferred Study Days */}
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-[#6F4E37]">
                Preferred Study Days
              </label>
              <div className="mt-2.5 flex flex-wrap gap-2">
                {ALL_DAYS.map((day) => {
                  const active = preferredDays.includes(day);
                  return (
                    <button
                      key={day}
                      type="button"
                      onClick={() => toggleDay(day)}
                      className={`brutal-btn rounded-lg px-3 py-1.5 text-xs font-bold ${
                        active
                          ? 'bg-[#4A2C1D] text-[#FDFBF7]'
                          : 'bg-[#FFFFFF] text-[#2A1810]'
                      }`}
                    >
                      {day}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Learning Intensity */}
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-[#6F4E37]">
                Scheduling Objective Profile
              </label>
              <div className="mt-2.5 grid grid-cols-3 gap-2.5">
                {[
                  { id: 'balanced', label: 'Balanced', desc: 'Min-max load' },
                  { id: 'front_loaded', label: 'Front-Loaded', desc: 'Early buffer' },
                  { id: 'exam_focused', label: 'Exam-Focused', desc: 'Exam prep' },
                ].map((mode) => (
                  <button
                    key={mode.id}
                    type="button"
                    onClick={() =>
                      setLearningIntensity(mode.id as 'balanced' | 'front_loaded' | 'exam_focused')
                    }
                    className={`brutal-btn rounded-xl p-3 text-left ${
                      learningIntensity === mode.id
                        ? 'bg-[#FEF3C7] text-[#2A1810]'
                        : 'bg-[#FFFFFF] text-[#2A1810]'
                    }`}
                  >
                    <div className="text-xs font-bold">{mode.label}</div>
                    <div className="mt-0.5 text-[11px] font-medium text-[#6F4E37]">{mode.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Spaced Repetition Toggle */}
            <label className="flex cursor-pointer items-center justify-between rounded-xl border-2 border-[#2A1810] bg-[#F6EFE6] p-4 transition hover:bg-[#FEF3C7]/60">
              <div>
                <div className="flex items-center gap-2 text-sm font-bold text-[#2A1810]">
                  <Cpu className="h-4 w-4 text-[#8B4513]" />
                  <span>FSRS Spaced Repetition Recall</span>
                </div>
                <p className="mt-0.5 text-xs text-[#6F4E37]">
                  Inject exponential memory-decay active recall checkpoints
                </p>
              </div>
              <input
                type="checkbox"
                checked={includeSpacedRepetition}
                onChange={(e) => setIncludeSpacedRepetition(e.target.checked)}
                className="h-5 w-5 accent-[#4A2C1D]"
              />
            </label>
          </div>
        </div>
      </div>
    </div>
  );
};
