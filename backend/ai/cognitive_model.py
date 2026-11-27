"""PyTorch Multi-Task Neural Classifier for Bloom's Taxonomy & Cognitive Load."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from backend.schemas import BloomLevel

BLOOM_LEVELS: tuple[BloomLevel, ...] = (
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
)

BLOOM_KEYWORDS: dict[BloomLevel, tuple[str, ...]] = {
    "Remember": (
        "introduction",
        "overview",
        "history",
        "terminology",
        "definitions",
        "basics",
        "fundamentals",
        "syntax",
        "notation",
        "recall",
        "survey",
        "glossary",
        "concepts",
    ),
    "Understand": (
        "explain",
        "principles",
        "theory",
        "interpret",
        "probability",
        "distributions",
        "representation",
        "properties",
        "foundations",
        "mechanics",
        "intuition",
        "classification",
        "taxonomy",
    ),
    "Apply": (
        "implement",
        "solve",
        "programming",
        "practice",
        "regression",
        "trees",
        "graphs",
        "traversal",
        "sorting",
        "queries",
        "numerical",
        "compute",
        "simulation",
        "training",
        "lab",
        "coding",
    ),
    "Analyze": (
        "complexity",
        "asymptotic",
        "dynamic programming",
        "backpropagation",
        "decomposition",
        "eigenvalues",
        "svd",
        "bayesian",
        "convergence",
        "regularization",
        "bias-variance",
        "debugging",
        "profiling",
        "derivation",
        "proof",
    ),
    "Evaluate": (
        "benchmark",
        "critique",
        "ablation",
        "validation",
        "metrics",
        "hypothesis",
        "tradeoffs",
        "interpretability",
        "fairness",
        "alignment",
        "robustness",
        "audit",
        "comparison",
        "testing",
    ),
    "Create": (
        "capstone",
        "design",
        "architecture",
        "build",
        "synthesize",
        "generative",
        "transformers",
        "attention",
        "diffusion",
        "rlhf",
        "multi-agent",
        "system design",
        "research project",
        "deployment",
        "production",
    ),
}

HIGH_COMPLEXITY_TOKENS = (
    "theorem",
    "proof",
    "stochastic",
    "optimization",
    "gradient",
    "backpropagation",
    "transformer",
    "attention",
    "eigen",
    "jacobian",
    "hessian",
    "lagrangian",
    "convex",
    "markov",
    "bayesian",
    "variational",
    "dynamic programming",
    "np-complete",
    "amortized",
    "distributed",
    "concurrency",
    "compiler",
    "kernel",
    "neural",
    "reinforcement",
    "calculus",
    "differential",
    "manifold",
    "fourier",
)

FEATURE_DIM = 32


def _stable_hash_bucket(token: str, num_buckets: int = 16) -> int:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % num_buckets


def featurize_topic(
    title: str,
    subtopics: list[str],
    module_ratio: float,
    prereq_count: int = 0,
) -> np.ndarray:
    """Convert topic text and structural syllabus signals into a 32-D normalized vector."""
    combined = f"{title} {' '.join(subtopics)}".lower()
    tokens = re.findall(r"[a-z0-9\-]+", combined)
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    # Dims 0..5: Bloom keyword hits
    for idx, level in enumerate(BLOOM_LEVELS):
        hits = sum(1 for kw in BLOOM_KEYWORDS[level] if kw in combined)
        vec[idx] = min(1.0, hits / 2.5)

    # Dims 6..15: Structural & pedagogical complexity signals
    high_hits = sum(1 for kw in HIGH_COMPLEXITY_TOKENS if kw in combined)
    vec[6] = min(1.0, high_hits / 4.0)
    vec[7] = min(1.0, len(subtopics) / 8.0)
    vec[8] = min(1.0, len(tokens) / 35.0)
    vec[9] = float(np.clip(module_ratio, 0.0, 1.0))
    vec[10] = min(1.0, prereq_count / 4.0)
    vec[11] = 1.0 if any(w in combined for w in ("project", "capstone", "system", "build")) else 0.0
    vec[12] = 1.0 if any(w in combined for w in ("math", "linear", "calculus", "probab", "stat", "proof")) else 0.0
    vec[13] = 1.0 if any(w in combined for w in ("code", "python", "implement", "lab", "algorithm", "api")) else 0.0
    vec[14] = 1.0 if any(w in combined for w in ("intro", "overview", "setup", "history", "basics")) else 0.0
    vec[15] = 1.0 if any(w in combined for w in ("review", "exam", "midterm", "final", "recap")) else 0.0

    # Dims 16..31: Hashed n-gram semantic fingerprint (L2 normalized)
    hash_vec = np.zeros(16, dtype=np.float32)
    for tok in tokens:
        if len(tok) > 2:
            hash_vec[_stable_hash_bucket(tok, 16)] += 1.0
    norm = float(np.linalg.norm(hash_vec))
    if norm > 1e-6:
        hash_vec /= norm
    vec[16:32] = hash_vec
    return vec


class CognitiveLoadNet(nn.Module):
    """Multi-task PyTorch MLP predicting Bloom's Taxonomy class & Cognitive Load (1-10)."""

    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
        )
        self.bloom_head = nn.Linear(32, len(BLOOM_LEVELS))
        self.load_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        bloom_logits = self.bloom_head(h)
        # Scale sigmoid output [0, 1] -> [1.0, 10.0]
        load_score = 1.0 + 9.0 * self.load_head(h).squeeze(-1)
        return bloom_logits, load_score


def _build_training_corpus() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Curated, domain-diverse curriculum training dataset across CS, Math, AI, Systems."""
    seed_samples: list[tuple[str, list[str], float, int, int, float]] = [
        # (title, subtopics, module_ratio, prereq_count, bloom_idx, cognitive_load)
        ("Course Overview & Python Environment Setup", ["syllabus", "git", "jupyter basics", "terminology"], 0.05, 0, 0, 2.1),
        ("Introduction to Computing History & Definitions", ["binary", "ascii", "hardware basics", "glossary"], 0.08, 0, 0, 2.3),
        ("Basic SQL Syntax & Relational Terminology", ["tables", "select", "keys", "definitions"], 0.12, 0, 0, 2.6),
        ("Probability Spaces & Axioms Intuition", ["sample space", "events", "conditional probability", "bayes intuition"], 0.18, 1, 1, 4.4),
        ("Vector Spaces & Geometric Interpretation", ["span", "basis", "linear independence", "dot product"], 0.22, 1, 1, 5.1),
        ("Principles of Object-Oriented Programming", ["encapsulation", "inheritance", "polymorphism", "principles"], 0.20, 1, 1, 3.8),
        ("Supervised Linear & Logistic Regression", ["gradient descent", "mse", "cross entropy", "python lab"], 0.35, 1, 2, 5.8),
        ("Binary Search Trees & Tree Traversals", ["inorder", "avl rotations", "implement bst", "coding practice"], 0.38, 1, 2, 6.2),
        ("Graph Algorithms: BFS, DFS & Shortest Paths", ["dijkstra", "adjacency list", "implement traversal", "coding"], 0.45, 2, 2, 6.8),
        ("Relational Algebra & Query Optimization Lab", ["joins", "indexing", "b-trees", "sql practice"], 0.42, 1, 2, 5.5),
        ("Eigenvalues, Eigenvectors & SVD Decomposition", ["spectral theorem", "pca", "matrix factorization", "proof"], 0.50, 2, 3, 7.6),
        ("Dynamic Programming & Memoization Analysis", ["optimal substructure", "knapsack", "complexity proof", "recurrence"], 0.58, 2, 3, 8.2),
        ("Backpropagation & Automatic Differentiation", ["chain rule", "computational graph", "vanishing gradients", "derivation"], 0.62, 2, 3, 8.4),
        ("Asymptotic Complexity & Amortized Analysis", ["master theorem", "big-o proof", "potential method", "np-complete"], 0.65, 2, 3, 8.1),
        ("Bayesian Inference & Variational Methods", ["posterior", "elbo derivation", "markov chain monte carlo", "math"], 0.72, 3, 3, 8.9),
        ("Model Evaluation, Ablation & Bias-Variance Tradeoff", ["cross validation", "roc-auc", "error analysis", "calibration"], 0.70, 2, 4, 6.7),
        ("LLM Alignment, Fairness & Safety Auditing", ["rlhf tradeoffs", "red teaming", "robustness evaluation", "critique"], 0.82, 3, 4, 7.4),
        ("Distributed Systems Consensus & Fault Tolerance", ["raft tradeoffs", "cap theorem", "benchmark comparison", "testing"], 0.78, 3, 4, 8.5),
        ("Transformers, Multi-Head Attention & Generative AI", ["self-attention", "positional encoding", "build mini-gpt", "architecture"], 0.86, 3, 5, 9.2),
        ("Capstone Multi-Agent RAG System Architecture", ["design pipeline", "vector index", "production deployment", "capstone"], 0.95, 3, 5, 9.0),
        ("Diffusion Models & Generative Synthesis Project", ["score matching", "denoising", "create generator", "research project"], 0.92, 3, 5, 9.4),
        ("Midterm Review & Consolidation", ["recap", "practice exam", "review concepts"], 0.50, 1, 1, 4.0),
    ]

    rng = np.random.default_rng(42)
    features: list[np.ndarray] = []
    blooms: list[int] = []
    loads: list[float] = []

    # Expand with controlled stochastic curriculum augmentations (12x -> 264 samples)
    extra_prefixes = ["Applied", "Foundations of", "Advanced", "Core", "Modern", "Computational"]
    extra_suffixes = ["Workshop", "Theory", "Methods", "Seminar", "Module", "Practice"]
    for _ in range(12):
        for title, subs, ratio, pre, b_idx, load in seed_samples:
            jitter_ratio = float(np.clip(ratio + rng.normal(0.0, 0.05), 0.02, 0.99))
            jitter_pre = int(np.clip(pre + rng.integers(-1, 2), 0, 4))
            aug_title = title
            aug_subs = list(subs)
            if rng.random() > 0.55:
                aug_title = f"{rng.choice(extra_prefixes)} {title}"
            if rng.random() > 0.55:
                aug_subs.append(str(rng.choice(extra_suffixes)).lower())
            feat = featurize_topic(aug_title, aug_subs, jitter_ratio, jitter_pre)
            noise = rng.normal(0.0, 0.015, size=feat.shape).astype(np.float32)
            features.append(np.clip(feat + noise, 0.0, 1.0))
            blooms.append(b_idx)
            loads.append(float(np.clip(load + rng.normal(0.0, 0.25), 1.2, 9.8)))

    return (
        np.stack(features),
        np.array(blooms, dtype=np.int64),
        np.array(loads, dtype=np.float32),
    )


@dataclass
class CognitiveEngine:
    model: CognitiveLoadNet
    val_accuracy: float
    val_load_mae: float

    def predict_batch(
        self,
        items: list[tuple[str, list[str], float, int]],
        weekly_hours_budget: float = 10.0,
    ) -> list[tuple[BloomLevel, float, float, float]]:
        """Vectorized batch inference over all syllabus topics in a single forward pass."""
        if not items:
            return []
        feats = np.stack([featurize_topic(t, subs, r, pre) for t, subs, r, pre in items])
        x = torch.from_numpy(feats)
        self.model.eval()
        with torch.no_grad():
            logits, load_tensor = self.model(x)
            probs_batch = torch.softmax(logits, dim=-1).numpy()
            nn_loads = load_tensor.numpy()

        total_topics = len(items)
        results: list[tuple[BloomLevel, float, float, float]] = []
        for idx, (title, subtopics, _, prereq_count) in enumerate(items):
            probs = probs_batch[idx]
            nn_load = float(nn_loads[idx])
            combined = f"{title} {' '.join(subtopics)}".lower()
            sym_scores = np.zeros(len(BLOOM_LEVELS), dtype=np.float32)
            for b_idx, lvl in enumerate(BLOOM_LEVELS):
                sym_scores[b_idx] = sum(1.0 for kw in BLOOM_KEYWORDS[lvl] if kw in combined)
            if sym_scores.sum() > 0:
                blended_probs = 0.72 * probs + 0.28 * (sym_scores / sym_scores.sum())
            else:
                blended_probs = probs

            best_idx = int(np.argmax(blended_probs))
            bloom_level = BLOOM_LEVELS[best_idx]
            bloom_conf = float(np.clip(blended_probs[best_idx], 0.45, 0.99))
            structural_adj = 0.25 * min(len(subtopics), 6) + 0.35 * min(prereq_count, 3)
            cognitive_load = float(np.clip(0.8 * nn_load + 0.2 * (3.2 + structural_adj), 1.5, 9.8))
            raw_hours = 1.4 + (cognitive_load / 10.0) * 3.8 + 0.22 * min(len(subtopics), 8)
            if total_topics > 0 and weekly_hours_budget > 0:
                raw_hours = max(1.5, min(12.0, round(raw_hours * 2.0) / 2.0))
            results.append(
                (bloom_level, round(bloom_conf, 3), round(cognitive_load, 2), round(raw_hours, 2))
            )
        return results

    def predict_topic(
        self,
        title: str,
        subtopics: list[str],
        module_ratio: float,
        prereq_count: int = 0,
        weekly_hours_budget: float = 10.0,
        total_topics: int = 12,
    ) -> tuple[BloomLevel, float, float, float]:
        """Return (bloom_level, bloom_confidence, cognitive_load, estimated_hours)."""
        feat = featurize_topic(title, subtopics, module_ratio, prereq_count)
        x = torch.from_numpy(feat).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            logits, load_tensor = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
            nn_load = float(load_tensor.item())

        # Symbolic prior calibration for exact Bloom verb hits
        combined = f"{title} {' '.join(subtopics)}".lower()
        sym_scores = np.zeros(len(BLOOM_LEVELS), dtype=np.float32)
        for idx, lvl in enumerate(BLOOM_LEVELS):
            sym_scores[idx] = sum(1.0 for kw in BLOOM_KEYWORDS[lvl] if kw in combined)
        if sym_scores.sum() > 0:
            sym_probs = sym_scores / sym_scores.sum()
            blended_probs = 0.72 * probs + 0.28 * sym_probs
        else:
            blended_probs = probs

        best_idx = int(np.argmax(blended_probs))
        bloom_level = BLOOM_LEVELS[best_idx]
        bloom_conf = float(np.clip(blended_probs[best_idx], 0.45, 0.99))

        # Calibrate cognitive load with structural complexity
        structural_adj = 0.25 * min(len(subtopics), 6) + 0.35 * min(prereq_count, 3)
        cognitive_load = float(np.clip(0.8 * nn_load + 0.2 * (3.2 + structural_adj), 1.5, 9.8))

        # Base study hours scaled by load and subtopic richness
        raw_hours = 1.4 + (cognitive_load / 10.0) * 3.8 + 0.22 * min(len(subtopics), 8)
        if total_topics > 0 and weekly_hours_budget > 0:
            # Soft scale gently toward course scope without collapsing complexity ratios
            raw_hours = max(1.5, min(12.0, round(raw_hours * 2.0) / 2.0))

        return bloom_level, round(bloom_conf, 3), round(cognitive_load, 2), round(raw_hours, 2)


_ENGINE_INSTANCE: CognitiveEngine | None = None


def get_cognitive_engine() -> CognitiveEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is not None:
        return _ENGINE_INSTANCE

    torch.manual_seed(42)
    np.random.seed(42)

    X, y_bloom, y_load = _build_training_corpus()
    perm = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = perm[:split], perm[split:]

    X_train = torch.from_numpy(X[train_idx])
    yb_train = torch.from_numpy(y_bloom[train_idx])
    yl_train = torch.from_numpy(y_load[train_idx])

    X_val = torch.from_numpy(X[val_idx])
    yb_val = torch.from_numpy(y_bloom[val_idx])
    yl_val = torch.from_numpy(y_load[val_idx])

    model = CognitiveLoadNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.025, weight_decay=1e-3)
    ce_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.SmoothL1Loss()

    model.train()
    for _ in range(45):
        optimizer.zero_grad()
        logits, pred_load = model(X_train)
        loss = ce_loss_fn(logits, yb_train) + 0.35 * reg_loss_fn(pred_load, yl_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits, val_pred_load = model(X_val)
        val_acc = float((val_logits.argmax(dim=-1) == yb_val).float().mean().item())
        val_mae = float(torch.abs(val_pred_load - yl_val).mean().item())

    _ENGINE_INSTANCE = CognitiveEngine(
        model=model,
        val_accuracy=round(val_acc * 100.0, 1),
        val_load_mae=round(val_mae, 3),
    )
    import gc
    del X, y_bloom, y_load, X_train, yb_train, yl_train, X_val, yb_val, yl_val
    gc.collect()
    return _ENGINE_INSTANCE
