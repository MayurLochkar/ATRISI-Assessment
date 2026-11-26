"""NetworkX Prerequisite Knowledge Graph (DAG), Critical Path & PageRank."""

from __future__ import annotations

import re
from typing import Tuple

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.schemas import PrerequisiteEdge, TopicNode

# Domain concept dependency rules (prerequisite keyword -> dependent keyword)
CURATED_CONCEPT_CHAINS: tuple[tuple[str, str, str], ...] = (
    ("linear algebra", "optimization", "Matrix calculus underpins optimization"),
    ("linear algebra", "regression", "Linear systems formulate regression"),
    ("linear algebra", "neural", "Tensor & matrix operations power neural nets"),
    ("probability", "bayesian", "Probability axioms foundation for Bayesian inference"),
    ("probability", "statistics", "Probability precedes statistical estimation"),
    ("calculus", "gradient", "Derivatives enable gradient optimization"),
    ("gradient", "backpropagation", "Gradient descent extends to backpropagation"),
    ("regression", "classification", "Linear models generalize to classification"),
    ("neural", "transformer", "Feedforward nets precede attention architectures"),
    ("backpropagation", "deep learning", "Backprop drives deep network training"),
    ("attention", "llm", "Self-attention underpins large language models"),
    ("complexity", "dynamic programming", "Asymptotic analysis guides DP design"),
    ("recursion", "tree", "Recursive thinking enables tree traversals"),
    ("recursion", "dynamic programming", "Recursion + memoization forms DP"),
    ("tree", "graph", "Trees generalize into general graphs"),
    ("array", "linked list", "Contiguous structures contrast with pointers"),
    ("sorting", "binary search", "Sorted order enables logarithmic search"),
    ("python", "pandas", "Core Python precedes data manipulation"),
    ("sql", "normalization", "Query basics precede schema normalization"),
    ("foundations", "capstone", "Core theory precedes capstone synthesis"),
)


def _topic_text(t: TopicNode) -> str:
    return f"{t.module_title} {t.title} {' '.join(t.subtopics)} {' '.join(t.keywords)}".lower()


def build_prerequisite_dag(
    topics: list[TopicNode],
) -> Tuple[list[TopicNode], list[PrerequisiteEdge], list[str]]:
    """
    Construct a verified Directed Acyclic Graph (DAG) of topic prerequisites,
    compute PageRank centrality, Critical Path, and Topological Ordering.
    """
    if not topics:
        return [], [], []

    n = len(topics)
    G = nx.DiGraph()
    for idx, topic in enumerate(topics):
        G.add_node(topic.id, order=idx, module=topic.module_number)

    if n == 1:
        topics[0].pagerank_centrality = 1.0
        topics[0].is_critical_path = True
        return topics, [], [topics[0].id]

    texts = [_topic_text(t) for t in topics]
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        tfidf = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf)
    except ValueError:
        sim_matrix = np.zeros((n, n), dtype=float)

    candidate_edges: list[PrerequisiteEdge] = []

    for i in range(n):
        t_i = topics[i]
        txt_i = texts[i]
        for j in range(i + 1, n):
            t_j = topics[j]
            txt_j = texts[j]
            sim = float(sim_matrix[i, j])
            weight = 0.0
            reasons: list[str] = []

            # 1. Explicit prerequisite wording in later topic referencing earlier topic keywords
            title_words = [w for w in re.findall(r"[a-z]{4,}", t_i.title.lower()) if w not in {"introduction", "module", "week", "unit", "basics"}]
            overlap_words = [w for w in title_words if w in txt_j]
            if overlap_words:
                weight += min(0.45, 0.18 * len(overlap_words))
                reasons.append(f"Concept continuation ({', '.join(overlap_words[:2])})")

            # 2. Curated academic dependency pairs
            for pre_kw, post_kw, desc in CURATED_CONCEPT_CHAINS:
                if pre_kw in txt_i and post_kw in txt_j:
                    weight += 0.42
                    reasons.append(desc)
                    break

            # 3. TF-IDF semantic bridge across sequential modules
            mod_gap = t_j.module_number - t_i.module_number
            if sim >= 0.14 and mod_gap >= 1:
                weight += min(0.35, sim * 0.9)
                reasons.append(f"Semantic similarity ({sim:.2f})")

            # 4. Sequential spine between consecutive modules (keeps curriculum connected)
            if j == i + 1 and (mod_gap == 1 or weight == 0.0):
                spine_w = 0.28 if mod_gap >= 1 else 0.20
                if weight < spine_w:
                    weight = max(weight, spine_w)
                    reasons.append("Sequential curriculum progression")

            # 5. Foundational first topic links to capstone/project topics
            if i == 0 and any(k in txt_j for k in ("capstone", "project", "synthesis", "final")):
                weight = max(weight, 0.36)
                reasons.append("Foundational dependency for project synthesis")

            if weight >= 0.24:
                candidate_edges.append(
                    PrerequisiteEdge(
                        source=t_i.id,
                        target=t_j.id,
                        weight=round(min(0.98, weight), 3),
                        reason="; ".join(dict.fromkeys(reasons)) or "Curriculum dependency",
                    )
                )

    # Limit fan-in per node to <= 3 strongest edges for visual clarity and MILP tractability
    by_target: dict[str, list[PrerequisiteEdge]] = {}
    for edge in candidate_edges:
        by_target.setdefault(edge.target, []).append(edge)

    selected_edges: list[PrerequisiteEdge] = []
    for target_id, group in by_target.items():
        group.sort(key=lambda e: e.weight, reverse=True)
        for e in group[:3]:
            selected_edges.append(e)
            G.add_edge(e.source, e.target, weight=e.weight)

    # Guarantee strict DAG (break cycles if any ever arise)
    while not nx.is_directed_acyclic_graph(G):
        cycle = nx.find_cycle(G, orientation="original")
        weakest = min(cycle, key=lambda c: G[c[0]][c[1]].get("weight", 0.1))
        G.remove_edge(weakest[0], weakest[1])
        selected_edges = [
            e
            for e in selected_edges
            if not (e.source == weakest[0] and e.target == weakest[1])
        ]

    # PageRank centrality (computed on reversed graph so foundational prerequisites get higher rank!)
    try:
        pr = nx.pagerank(G.reverse(copy=True), alpha=0.85, weight="weight")
    except Exception:
        pr = {t.id: round(1.0 / n, 4) for t in topics}

    # Longest path (Critical Path) weighted by edge weight + topic cognitive load
    longest_path: list[str] = []
    try:
        longest_path = nx.dag_longest_path(G, weight="weight")
    except Exception:
        longest_path = [t.id for t in topics]

    crit_nodes = set(longest_path)
    crit_edge_pairs = set(zip(longest_path, longest_path[1:]))

    prereq_map: dict[str, list[str]] = {t.id: [] for t in topics}
    for e in selected_edges:
        e.is_critical = (e.source, e.target) in crit_edge_pairs
        prereq_map[e.target].append(e.source)

    for t in topics:
        t.pagerank_centrality = round(float(pr.get(t.id, 0.0)), 4)
        t.is_critical_path = t.id in crit_nodes
        t.prerequisites = prereq_map.get(t.id, [])

    topo_order = list(nx.topological_sort(G))
    return topics, selected_edges, topo_order
