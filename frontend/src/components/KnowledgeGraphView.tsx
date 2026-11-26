import React, { useMemo, useState } from 'react';
import { GitBranch, Info, Sparkles, ZoomIn, ZoomOut } from 'lucide-react';
import type { StudyPlanResponse, TopicNode } from '../types';

interface KnowledgeGraphViewProps {
  plan: StudyPlanResponse;
}

interface PositionedNode {
  topic: TopicNode;
  layer: number;
  x: number;
  y: number;
}

export const KnowledgeGraphView: React.FC<KnowledgeGraphViewProps> = ({ plan }) => {
  const [selectedTopicId, setSelectedTopicId] = useState<string>(
    plan.topics[0]?.id ?? ''
  );
  const [criticalOnly, setCriticalOnly] = useState(false);
  const [zoom, setZoom] = useState(1);

  const layout = useMemo(() => {
    const topics = plan.topics;
    const edges = plan.edges;
    const depthMap: Record<string, number> = {};
    for (const t of topics) depthMap[t.id] = 0;

    for (let pass = 0; pass < topics.length; pass++) {
      for (const e of edges) {
        if ((depthMap[e.source] ?? 0) + 1 > (depthMap[e.target] ?? 0)) {
          depthMap[e.target] = (depthMap[e.source] ?? 0) + 1;
        }
      }
    }

    const byLayer: Record<number, TopicNode[]> = {};
    for (const t of topics) {
      const l = depthMap[t.id] ?? 0;
      if (!byLayer[l]) byLayer[l] = [];
      byLayer[l].push(t);
    }

    const maxLayer = Math.max(0, ...Object.keys(byLayer).map(Number));
    const maxRows = Math.max(1, ...Object.values(byLayer).map((arr) => arr.length));

    const colWidth = 260;
    const rowHeight = 122;
    const padX = 125;
    const padY = 85;

    const nodes: PositionedNode[] = [];
    const nodeMap: Record<string, PositionedNode> = {};

    for (let l = 0; l <= maxLayer; l++) {
      const group = byLayer[l] ?? [];
      const totalH = (group.length - 1) * rowHeight;
      const startY = padY + ((maxRows - 1) * rowHeight - totalH) / 2;
      group.forEach((t, idx) => {
        const p: PositionedNode = {
          topic: t,
          layer: l,
          x: padX + l * colWidth,
          y: startY + idx * rowHeight,
        };
        nodes.push(p);
        nodeMap[t.id] = p;
      });
    }

    const svgWidth = Math.max(780, padX * 2 + maxLayer * colWidth + 90);
    const svgHeight = Math.max(440, padY * 2 + (maxRows - 1) * rowHeight + 50);

    return { nodes, nodeMap, svgWidth, svgHeight };
  }, [plan]);

  const selectedNode =
    plan.topics.find((t) => t.id === selectedTopicId) ?? plan.topics[0] ?? null;

  const incomingEdges = plan.edges.filter((e) => e.target === selectedNode?.id);
  const outgoingEdges = plan.edges.filter((e) => e.source === selectedNode?.id);

  return (
    <div className="grid gap-8 lg:grid-cols-12 animate-fade-up">
      {/* SVG Prerequisite DAG Canvas */}
      <div className="lg:col-span-8 brutal-card p-6">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b-2 border-[#2A1810] pb-4">
          <div>
            <div className="flex items-center gap-2">
              <GitBranch className="h-5 w-5 text-[#8B4513]" />
              <h2 className="font-display text-lg font-bold text-[#2A1810]">
                NetworkX Prerequisite Knowledge Graph
              </h2>
            </div>
            <p className="mt-1 text-xs font-medium text-[#6F4E37]">
              Click any topic node to inspect Bloom score, PageRank centrality &amp; dependencies
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2.5">
            <div className="flex items-center gap-1 rounded-xl border-2 border-[#2A1810] bg-[#F6EFE6] p-1">
              <button
                onClick={() => setZoom((z) => Math.max(0.65, Number((z - 0.15).toFixed(2))))}
                className="rounded-lg p-1.5 text-[#2A1810] hover:bg-white"
                title="Zoom Out"
              >
                <ZoomOut className="h-4 w-4" />
              </button>
              <span className="px-2 font-mono text-xs font-bold text-[#2A1810]">
                {Math.round(zoom * 100)}%
              </span>
              <button
                onClick={() => setZoom((z) => Math.min(1.45, Number((z + 0.15).toFixed(2))))}
                className="rounded-lg p-1.5 text-[#2A1810] hover:bg-white"
                title="Zoom In"
              >
                <ZoomIn className="h-4 w-4" />
              </button>
            </div>
            <label className="flex cursor-pointer items-center gap-2 rounded-xl border-2 border-[#2A1810] bg-[#FEF3C7] px-3 py-2 text-xs font-bold text-[#2A1810]">
              <input
                type="checkbox"
                checked={criticalOnly}
                onChange={(e) => setCriticalOnly(e.target.checked)}
                className="accent-[#4A2C1D]"
              />
              <span>Critical Path Only</span>
            </label>
          </div>
        </div>

        <div className="mt-5 overflow-auto rounded-2xl border-2 border-[#2A1810] bg-[#FDFBF7] p-4 max-h-[540px]">
          <svg
            width={layout.svgWidth * zoom}
            height={layout.svgHeight * zoom}
            viewBox={`0 0 ${layout.svgWidth} ${layout.svgHeight}`}
            className="block select-none"
          >
            <defs>
              <marker
                id="arrow-normal"
                viewBox="0 0 10 10"
                refX="8"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto"
              >
                <path d="M 0 1 L 10 5 L 0 9 z" fill="#6F4E37" />
              </marker>
              <marker
                id="arrow-critical"
                viewBox="0 0 10 10"
                refX="8"
                refY="5"
                markerWidth="7"
                markerHeight="7"
                orient="auto"
              >
                <path d="M 0 1 L 10 5 L 0 9 z" fill="#C86D2D" />
              </marker>
            </defs>

            {plan.edges.map((edge, idx) => {
              const src = layout.nodeMap[edge.source];
              const dst = layout.nodeMap[edge.target];
              if (!src || !dst) return null;

              const x1 = src.x + 92;
              const y1 = src.y;
              const x2 = dst.x - 96;
              const y2 = dst.y;
              const cx = (x1 + x2) / 2;
              const d = `M ${x1} ${y1} C ${cx} ${y1}, ${cx} ${y2}, ${x2} ${y2}`;

              const dimmed = criticalOnly && !edge.is_critical;
              return (
                <path
                  key={`${edge.source}-${edge.target}-${idx}`}
                  d={d}
                  fill="none"
                  stroke={edge.is_critical ? '#C86D2D' : '#6F4E37'}
                  strokeWidth={edge.is_critical ? 3 : 1.8}
                  strokeOpacity={dimmed ? 0.14 : edge.is_critical ? 0.95 : 0.7}
                  markerEnd={
                    edge.is_critical ? 'url(#arrow-critical)' : 'url(#arrow-normal)'
                  }
                />
              );
            })}

            {layout.nodes.map(({ topic, x, y }) => {
              const isSelected = topic.id === selectedNode?.id;
              const dimmed = criticalOnly && !topic.is_critical_path;
              const fillColor = isSelected
                ? '#4A2C1D'
                : topic.is_critical_path
                ? '#FEF3C7'
                : '#FFFFFF';

              return (
                <g
                  key={topic.id}
                  transform={`translate(${x - 92}, ${y - 38})`}
                  onClick={() => setSelectedTopicId(topic.id)}
                  className="cursor-pointer"
                  opacity={dimmed ? 0.25 : 1}
                >
                  {/* Hard Neo-Brutalist Offset Shadow */}
                  <rect x={4} y={4} width={184} height={76} rx={12} fill="#2A1810" />
                  <rect
                    width={184}
                    height={76}
                    rx={12}
                    fill={fillColor}
                    stroke="#2A1810"
                    strokeWidth={2.2}
                  />
                  <text
                    x={12}
                    y={20}
                    fill={isSelected ? '#FBBF24' : '#8B4513'}
                    fontSize="10"
                    fontFamily="JetBrains Mono, monospace"
                    fontWeight="700"
                  >
                    {`M${topic.module_number} · ${topic.bloom_level} · ${topic.estimated_hours}h`}
                  </text>
                  <text
                    x={12}
                    y={41}
                    fill={isSelected ? '#FDFBF7' : '#2A1810'}
                    fontSize="12"
                    fontWeight="700"
                  >
                    {topic.title.length > 22 ? `${topic.title.slice(0, 21)}…` : topic.title}
                  </text>
                  <text
                    x={12}
                    y={61}
                    fill={isSelected ? '#DFD1C3' : '#6F4E37'}
                    fontSize="10"
                    fontFamily="JetBrains Mono, monospace"
                    fontWeight="600"
                  >
                    {`Load ${topic.cognitive_load}/10 · PR ${topic.pagerank_centrality.toFixed(2)}`}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-5 text-xs font-semibold text-[#6F4E37]">
          <span className="flex items-center gap-2">
            <span className="h-3.5 w-3.5 rounded border-[1.5px] border-[#2A1810] bg-[#FEF3C7]" />
            Critical Path (Longest Chain)
          </span>
          <span className="flex items-center gap-2">
            <span className="h-3.5 w-3.5 rounded border-[1.5px] border-[#2A1810] bg-[#4A2C1D]" />
            Selected Node
          </span>
          <span className="flex items-center gap-2">
            <span className="h-3.5 w-3.5 rounded border-[1.5px] border-[#2A1810] bg-[#FFFFFF]" />
            Standard Topic (PR = PageRank)
          </span>
        </div>
      </div>

      {/* Right: Node Inspector */}
      <div className="lg:col-span-4 brutal-card p-6 h-fit">
        <div className="flex items-center gap-2 border-b-2 border-[#2A1810] pb-3.5">
          <Info className="h-5 w-5 text-[#8B4513]" />
          <h3 className="font-display text-lg font-bold text-[#2A1810]">Topic Node Inspector</h3>
        </div>

        {selectedNode ? (
          <div className="mt-5 space-y-5">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <span className="rounded-md border-[1.5px] border-[#2A1810] bg-[#4A2C1D] px-2.5 py-0.5 font-mono text-xs font-bold text-[#FDFBF7]">
                  Module {selectedNode.module_number}
                </span>
                {selectedNode.is_critical_path && (
                  <span className="flex items-center gap-1 rounded-md border-[1.5px] border-[#2A1810] bg-[#FEF3C7] px-2.5 py-0.5 text-xs font-bold text-[#92400E]">
                    <Sparkles className="h-3.5 w-3.5" />
                    Critical Path
                  </span>
                )}
              </div>
              <h4 className="mt-3 font-display text-lg font-bold text-[#2A1810]">
                {selectedNode.title}
              </h4>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-xl border-2 border-[#2A1810] bg-[#FDFBF7] p-3.5">
                <div className="text-xs font-bold text-[#6F4E37]">PyTorch Bloom</div>
                <div className="mt-1 text-base font-bold text-[#2A1810]">
                  {selectedNode.bloom_level}
                </div>
                <div className="font-mono text-[11px] font-semibold text-[#8B4513]">
                  Conf: {Math.round(selectedNode.bloom_confidence * 100)}%
                </div>
              </div>
              <div className="rounded-xl border-2 border-[#2A1810] bg-[#FDFBF7] p-3.5">
                <div className="text-xs font-bold text-[#6F4E37]">Cognitive Load</div>
                <div className="mt-1 font-mono text-base font-bold text-[#2A1810]">
                  {selectedNode.cognitive_load} / 10
                </div>
                <div className="font-mono text-[11px] font-semibold text-[#065F46]">
                  Est. {selectedNode.estimated_hours} hrs
                </div>
              </div>
            </div>

            {selectedNode.subtopics.length > 0 && (
              <div>
                <div className="text-xs font-bold uppercase tracking-wider text-[#6F4E37]">
                  Subtopics Covered
                </div>
                <ul className="mt-2 space-y-1.5 text-sm text-[#2A1810] list-disc pl-5">
                  {selectedNode.subtopics.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ul>
              </div>
            )}

            <div>
              <div className="text-xs font-bold uppercase tracking-wider text-[#6F4E37]">
                Prerequisites ({incomingEdges.length})
              </div>
              {incomingEdges.length === 0 ? (
                <p className="mt-1.5 text-xs font-medium text-[#6F4E37]">
                  Entry-level foundational topic.
                </p>
              ) : (
                <div className="mt-2 space-y-2">
                  {incomingEdges.map((e, i) => {
                    const srcTopic = plan.topics.find((t) => t.id === e.source);
                    return (
                      <button
                        key={i}
                        onClick={() => setSelectedTopicId(e.source)}
                        className="brutal-btn w-full rounded-xl bg-[#FDFBF7] p-3 text-left"
                      >
                        <div className="text-xs font-bold text-[#2A1810]">
                          {srcTopic?.title || e.source}
                        </div>
                        <div className="mt-0.5 text-[11px] font-medium text-[#6F4E37]">
                          {e.reason} (w={e.weight})
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>

            <div>
              <div className="text-xs font-bold uppercase tracking-wider text-[#6F4E37]">
                Unlocks Next ({outgoingEdges.length})
              </div>
              {outgoingEdges.length === 0 ? (
                <p className="mt-1.5 text-xs font-medium text-[#6F4E37]">
                  Terminal synthesis topic.
                </p>
              ) : (
                <div className="mt-2 space-y-2">
                  {outgoingEdges.map((e, i) => {
                    const dstTopic = plan.topics.find((t) => t.id === e.target);
                    return (
                      <button
                        key={i}
                        onClick={() => setSelectedTopicId(e.target)}
                        className="brutal-btn w-full rounded-xl bg-[#F6EFE6] p-3 text-left"
                      >
                        <div className="text-xs font-bold text-[#2A1810]">
                          {dstTopic?.title || e.target}
                        </div>
                        <div className="mt-0.5 text-[11px] font-medium text-[#6F4E37]">
                          {e.reason}
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        ) : (
          <p className="mt-4 text-sm text-[#6F4E37]">No topics available.</p>
        )}
      </div>
    </div>
  );
};
