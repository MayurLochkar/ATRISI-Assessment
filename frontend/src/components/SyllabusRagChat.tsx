import React, { useState } from 'react';
import { Bot, FileSearch, Loader2, Send, Sparkles, User } from 'lucide-react';
import type { RagCitation, StudyEngineMode, StudyPlanResponse } from '../types';
import { apiClient } from '../api/client';

interface SyllabusRagChatProps {
  plan: StudyPlanResponse;
  engine: StudyEngineMode;
  apiKey: string;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  text: string;
  citations?: RagCitation[];
  meta?: string;
}

const QUICK_PROMPTS = [
  'What are the exams, assignments, and grading weights?',
  'Which topics are foundational prerequisites for the end of the course?',
  'Summarize the core concepts covered in the middle modules.',
  'Are there any coding labs or capstone projects mentioned?',
];

export const SyllabusRagChat: React.FC<SyllabusRagChatProps> = ({
  plan,
  engine,
  apiKey,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      text: `Hello! I'm your ${plan.metadata.course_code} Hybrid RAG Study Coach. Ask any question about your uploaded syllabus — I retrieve answers using Sparse TF-IDF + Dense PyTorch Subword Embeddings fused via Reciprocal Rank Fusion (RRF) with exact page citations.`,
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const buildFallbackContext = () => {
    const topicLines = plan.topics.map(
      (t) =>
        `Module ${t.module_number}: ${t.title} (Bloom: ${t.bloom_level}, Load: ${t.cognitive_load}/10)\nSubtopics: ${t.subtopics.join(', ')}`
    );
    const assessLines = plan.assessments.map(
      (a) =>
        `Assessment: ${a.title} (${a.assessment_type}${a.target_week ? `, Week ${a.target_week}` : ''}${a.weight_percent ? `, ${a.weight_percent}%` : ''}) - ${a.description}`
    );
    return `${plan.metadata.course_code}: ${plan.metadata.course_title}\n\n${topicLines.join('\n\n')}\n\n${assessLines.join('\n')}`;
  };

  const sendQuestion = async (q: string) => {
    const trimmed = q.trim();
    if (!trimmed || loading) return;

    setMessages((prev) => [...prev, { role: 'user', text: trimmed }]);
    setInput('');
    setLoading(true);

    try {
      const res = await apiClient.queryRag(
        plan.plan_id,
        trimmed,
        engine,
        apiKey,
        buildFallbackContext()
      );
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          text: res.answer,
          citations: res.citations,
          meta: `${res.engine_used} · ${res.latency_ms}ms`,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          text: `Error querying syllabus RAG index: ${
            err instanceof Error ? err.message : 'Unknown error'
          }`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid gap-8 lg:grid-cols-12 animate-fade-up">
      <div className="lg:col-span-8 brutal-card flex flex-col overflow-hidden h-[600px]">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b-2 border-[#2A1810] bg-[#F6EFE6] px-6 py-4">
          <div className="flex items-center gap-2.5">
            <Sparkles className="h-5 w-5 text-[#8B4513]" />
            <h2 className="font-display text-base font-bold text-[#2A1810]">
              Ask Your Syllabus — Hybrid Sparse+Dense RAG
            </h2>
          </div>
          <span className="rounded-md border border-[#2A1810] bg-[#FFFFFF] px-2.5 py-0.5 font-mono text-xs font-bold text-[#2A1810]">
            RRF Top-4 Retrieval
          </span>
        </div>

        <div className="flex-1 space-y-5 overflow-y-auto bg-[#FDFBF7] p-6">
          {messages.map((m, idx) => (
            <div
              key={idx}
              className={`flex gap-3.5 ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {m.role === 'assistant' && (
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border-2 border-[#2A1810] bg-[#FEF3C7] text-[#2A1810]">
                  <Bot className="h-4 w-4" />
                </div>
              )}
              <div
                className={`max-w-[82%] rounded-2xl border-2 border-[#2A1810] px-5 py-3.5 text-sm leading-relaxed shadow-[3px_3px_0px_#2A1810] ${
                  m.role === 'user'
                    ? 'bg-[#4A2C1D] text-[#FDFBF7]'
                    : 'bg-[#FFFFFF] text-[#2A1810]'
                }`}
              >
                <div className="whitespace-pre-wrap">{m.text}</div>
                {m.meta && (
                  <div className="mt-2 font-mono text-[11px] font-semibold text-[#6F4E37]">
                    {m.meta}
                  </div>
                )}
                {m.citations && m.citations.length > 0 && (
                  <div className="mt-4 space-y-2 border-t-2 border-[#E8DDD4] pt-3">
                    <div className="flex items-center gap-1.5 font-mono text-[11px] font-bold uppercase tracking-wider text-[#8B4513]">
                      <FileSearch className="h-3.5 w-3.5" />
                      Retrieved Syllabus Citations
                    </div>
                    {m.citations.map((c) => (
                      <div
                        key={c.chunk_id}
                        className="rounded-xl border border-[#2A1810] bg-[#F6EFE6] p-3 text-xs"
                      >
                        <div className="flex justify-between font-mono text-[11px] font-bold text-[#4A2C1D]">
                          <span>
                            Page {c.page_number} · {c.section_title}
                          </span>
                          <span>RRF {c.score.toFixed(4)}</span>
                        </div>
                        <p className="mt-1 text-[#6F4E37] line-clamp-2">{c.snippet}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              {m.role === 'user' && (
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border-2 border-[#2A1810] bg-[#D1FAE5] text-[#2A1810]">
                  <User className="h-4 w-4" />
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div className="flex items-center gap-2 text-xs font-bold text-[#6F4E37]">
              <Loader2 className="h-4 w-4 animate-spin text-[#8B4513]" />
              <span>Running Hybrid Sparse + Dense RRF retrieval...</span>
            </div>
          )}
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            sendQuestion(input);
          }}
          className="flex items-center gap-3 border-t-2 border-[#2A1810] bg-[#FFFFFF] p-4"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about grading, exam dates, units, or prerequisites..."
            className="flex-1 rounded-xl border-2 border-[#2A1810] bg-[#FDFBF7] px-4 py-2.5 text-sm font-medium text-[#2A1810] placeholder-[#9C7A65] focus:bg-white focus:outline-none"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="brutal-btn flex items-center gap-2 rounded-xl bg-[#4A2C1D] px-5 py-2.5 text-xs font-bold text-[#FDFBF7] disabled:opacity-40"
          >
            <Send className="h-4 w-4" />
            <span>Ask</span>
          </button>
        </form>
      </div>

      {/* Quick Suggested Queries */}
      <div className="lg:col-span-4 brutal-card p-6 h-fit">
        <h3 className="font-display text-lg font-bold text-[#2A1810]">
          Suggested Syllabus Prompts
        </h3>
        <p className="mt-1.5 text-xs font-medium text-[#6F4E37]">
          Click any prompt to test hybrid retrieval &amp; page citation grounding:
        </p>
        <div className="mt-5 space-y-3">
          {QUICK_PROMPTS.map((q) => (
            <button
              key={q}
              onClick={() => sendQuestion(q)}
              className="brutal-btn w-full rounded-xl bg-[#FDFBF7] p-3.5 text-left text-xs font-bold text-[#2A1810] hover:bg-[#FEF3C7]"
            >
              {q}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
