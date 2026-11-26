"""Hybrid Sparse (TF-IDF) + Dense Neural Projection RAG Engine with Reciprocal Rank Fusion."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.ai.llm_providers import call_llm_text, resolve_effective_engine
from backend.schemas import RagCitation, RagQueryResponse, StudyEngineMode, SyllabusChunk


def _char_ngram_dense_embeddings(texts: list[str], dim: int = 128) -> torch.Tensor:
    """
    Deterministic PyTorch dense semantic projection over character-trigram TF-IDF
    using a fixed orthogonal projection matrix (Johnson-Lindenstrauss).
    """
    if not texts:
        return torch.zeros((0, dim), dtype=torch.float32)

    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=2048)
    mat = char_vec.fit_transform(texts).toarray().astype(np.float32)

    rng = np.random.default_rng(2026)
    proj = rng.normal(0.0, 1.0 / np.sqrt(dim), size=(mat.shape[1], dim)).astype(np.float32)
    dense = torch.from_numpy(mat @ proj)
    dense = torch.nn.functional.normalize(dense, p=2, dim=1)
    return dense


def _query_dense_vector(
    query: str,
    char_vec: TfidfVectorizer,
    proj_matrix: np.ndarray,
) -> torch.Tensor:
    q_mat = char_vec.transform([query]).toarray().astype(np.float32)
    q_dense = torch.from_numpy(q_mat @ proj_matrix)
    return torch.nn.functional.normalize(q_dense, p=2, dim=1)


def retrieve_hybrid_rrf(
    chunks: list[SyllabusChunk], question: str, top_k: int = 4
) -> list[tuple[SyllabusChunk, float]]:
    """Retrieve top_k syllabus chunks via Sparse TF-IDF + Dense PyTorch RRF fusion."""
    if not chunks:
        return []

    docs = [f"{c.section_title}\n{c.text}" for c in chunks]
    n = len(docs)

    # 1. Sparse word-level TF-IDF ranking
    try:
        word_vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        doc_sparse = word_vec.fit_transform(docs)
        q_sparse = word_vec.transform([question])
        sparse_sims = cosine_similarity(q_sparse, doc_sparse).ravel()
    except ValueError:
        sparse_sims = np.zeros(n, dtype=float)

    sparse_order = np.argsort(-sparse_sims)
    sparse_rank = {int(idx): rank + 1 for rank, idx in enumerate(sparse_order)}

    # 2. Dense PyTorch character-subword semantic projection
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=2048)
    mat = char_vec.fit_transform(docs).toarray().astype(np.float32)
    dim = 128
    rng = np.random.default_rng(2026)
    proj = rng.normal(0.0, 1.0 / np.sqrt(dim), size=(mat.shape[1], dim)).astype(np.float32)
    doc_dense = torch.nn.functional.normalize(torch.from_numpy(mat @ proj), p=2, dim=1)
    q_dense = _query_dense_vector(question, char_vec, proj)
    dense_sims = torch.matmul(doc_dense, q_dense.squeeze(0)).numpy()

    dense_order = np.argsort(-dense_sims)
    dense_rank = {int(idx): rank + 1 for rank, idx in enumerate(dense_order)}

    # 3. Reciprocal Rank Fusion (k = 60) + lexical similarity boost
    k_rrf = 60.0
    scored: list[tuple[int, float]] = []
    for idx in range(n):
        rrf = 1.0 / (k_rrf + sparse_rank[idx]) + 1.0 / (k_rrf + dense_rank[idx])
        combined = rrf + 0.03 * float(sparse_sims[idx]) + 0.015 * float(max(0.0, dense_sims[idx]))
        scored.append((idx, combined))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [(chunks[idx], round(score, 4)) for idx, score in scored[:top_k]]


def _synthesize_local_answer(
    question: str, retrieved: list[tuple[SyllabusChunk, float]]
) -> str:
    """Create a structured, citation-grounded local answer when running offline."""
    if not retrieved:
        return "No syllabus content is indexed for this study plan yet."

    q_words = set(w.lower() for w in question.split() if len(w) > 3)
    highlighted_lines: list[str] = []
    for chunk, _ in retrieved[:3]:
        lines = [ln.strip() for ln in chunk.text.splitlines() if ln.strip()]
        # Rank lines inside the chunk by keyword overlap with question
        ranked = sorted(
            lines,
            key=lambda ln: sum(1 for w in q_words if w in ln.lower()),
            reverse=True,
        )
        best_excerpt = " ".join(ranked[:3])
        if len(best_excerpt) > 320:
            best_excerpt = best_excerpt[:317] + "..."
        highlighted_lines.append(
            f"- **[Page {chunk.page_number} · {chunk.section_title}]**: {best_excerpt}"
        )

    return (
        "Based on Hybrid Sparse+Dense Retrieval (RRF) across your indexed syllabus:\n\n"
        + "\n".join(highlighted_lines)
        + "\n\n*Tip: Connect an OpenAI/Gemini key in Settings if you'd like conversational re-phrasing on top of these retrieved citations.*"
    )


def answer_syllabus_question(
    chunks: list[SyllabusChunk],
    question: str,
    top_k: int = 4,
    engine: StudyEngineMode = "local_hybrid",
    api_key: Optional[str] = None,
) -> RagQueryResponse:
    t0 = time.perf_counter()
    retrieved = retrieve_hybrid_rrf(chunks, question, top_k=top_k)
    citations = [
        RagCitation(
            chunk_id=c.chunk_id,
            page_number=c.page_number,
            section_title=c.section_title,
            score=score,
            snippet=(c.text[:260] + "...") if len(c.text) > 260 else c.text,
        )
        for c, score in retrieved
    ]

    effective, key, _ = resolve_effective_engine(engine, api_key)
    answer_text: Optional[str] = None
    used_engine = "local_hybrid_rrf"

    if effective != "local_hybrid" and retrieved:
        context_block = "\n\n".join(
            f"[Page {c.page_number} | {c.section_title}]\n{c.text}"
            for c, _ in retrieved
        )
        sys_prompt = (
            "You are SyllabAI Study Coach. Answer the student's question strictly using the "
            "provided syllabus excerpts. Always cite page numbers like [Page 2] inline."
        )
        usr_prompt = f"Context:\n{context_block}\n\nQuestion: {question}"
        llm_reply = call_llm_text(sys_prompt, usr_prompt, effective, key)
        if llm_reply:
            answer_text = llm_reply
            used_engine = f"{effective}_rag"

    if not answer_text:
        answer_text = _synthesize_local_answer(question, retrieved)

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return RagQueryResponse(
        answer=answer_text,
        engine_used=used_engine,
        citations=citations,
        latency_ms=latency_ms,
    )
