"""Unit tests for all 6 AI/ML pipeline engines."""

from __future__ import annotations

from backend.ai.cognitive_model import get_cognitive_engine
from backend.ai.knowledge_graph import build_prerequisite_dag
from backend.ai.optimizer import optimize_study_schedule
from backend.ai.pdf_ingest import chunk_pages, extract_pages_from_pdf, pages_from_raw_text
from backend.ai.rag_engine import answer_syllabus_question, retrieve_hybrid_rrf
from backend.ai.spaced_repetition import inject_spaced_repetition, retention_probability, update_task_fsrs
from backend.schemas import ExtractedAssessment, PlannerConstraints, StudyTaskSchema, TopicNode
from backend.services.sample_syllabi import ensure_sample_pdf


def test_pdf_ingest_and_chunking():
    pdf_path = ensure_sample_pdf("cs301_ml_dl")
    pages, method = extract_pages_from_pdf(pdf_path.read_bytes())
    assert len(pages) >= 1
    assert method == "pymupdf_native"
    chunks = chunk_pages(pages, target_tokens=160, overlap_tokens=30)
    assert len(chunks) >= 3
    assert all(c.token_count > 0 and c.page_number >= 1 for c in chunks)


def test_pytorch_cognitive_model_metrics_and_inference():
    engine = get_cognitive_engine()
    assert engine.val_accuracy >= 70.0
    assert engine.val_load_mae <= 2.5

    bloom_intro, _, load_intro, _ = engine.predict_topic(
        "Introduction & Course Overview", ["terminology", "basics", "setup"], 0.05, 0
    )
    bloom_hard, _, load_hard, hrs_hard = engine.predict_topic(
        "Transformers, Multi-Head Attention & Capstone Architecture",
        ["self-attention", "positional encoding", "build mini-gpt", "derivation"],
        0.9,
        3,
    )
    assert bloom_intro in ("Remember", "Understand")
    assert bloom_hard in ("Analyze", "Evaluate", "Create")
    assert load_hard > load_intro
    assert hrs_hard >= 2.0


def test_knowledge_graph_dag_and_critical_path():
    topics = [
        TopicNode(id="t1", module_number=1, module_title="Linear Algebra", title="Linear Algebra & Matrix Calculus", subtopics=["eigenvalues", "svd"]),
        TopicNode(id="t2", module_number=2, module_title="Optimization", title="Gradient Descent & Optimization", subtopics=["convexity", "learning rate"]),
        TopicNode(id="t3", module_number=3, module_title="Deep Learning", title="Neural Networks & Backpropagation", subtopics=["chain rule", "mlp"]),
        TopicNode(id="t4", module_number=4, module_title="Transformers", title="Attention & Transformer Architectures", subtopics=["self-attention", "llm"]),
    ]
    out_topics, edges, topo = build_prerequisite_dag(topics)
    assert len(topo) == 4
    assert len(edges) >= 3
    assert any(t.is_critical_path for t in out_topics)
    assert all(t.pagerank_centrality > 0 for t in out_topics)
    # Ensure strict forward topological ordering
    pos = {tid: i for i, tid in enumerate(topo)}
    for e in edges:
        assert pos[e.source] < pos[e.target]


def test_milp_optimizer_respects_prerequisites_and_load():
    topics = [
        TopicNode(id="t1", module_number=1, module_title="M1", title="Linear Algebra", estimated_hours=3.0, cognitive_load=5.0, pagerank_centrality=0.4),
        TopicNode(id="t2", module_number=2, module_title="M2", title="Gradient Optimization", estimated_hours=4.0, cognitive_load=6.5, prerequisites=["t1"], pagerank_centrality=0.3),
        TopicNode(id="t3", module_number=3, module_title="M3", title="Backpropagation", estimated_hours=5.0, cognitive_load=8.0, prerequisites=["t2"], is_critical_path=True, pagerank_centrality=0.2),
        TopicNode(id="t4", module_number=4, module_title="M4", title="Transformers Capstone", estimated_hours=5.0, cognitive_load=9.0, prerequisites=["t3"], is_critical_path=True, bloom_level="Create", pagerank_centrality=0.1),
    ]
    topics, edges, _ = build_prerequisite_dag(topics)
    assessments = [
        ExtractedAssessment(id="a1", title="Midterm", assessment_type="midterm", target_week=3, weight_percent=30.0),
        ExtractedAssessment(id="a2", title="Final Exam", assessment_type="final", target_week=6, weight_percent=40.0),
    ]
    constraints = PlannerConstraints(
        total_weeks=6,
        weekly_hours=8.0,
        preferred_days=["Mon", "Wed", "Fri"],
        start_date="2026-12-07",
    )
    weeks, status, warning, _ = optimize_study_schedule(topics, edges, assessments, constraints)
    assert len(weeks) == 6
    assert "milp" in status
    assert warning is None

    topic_week = {}
    for w in weeks:
        for tk in w.tasks:
            if tk.topic_id and tk.task_type != "assessment_prep":
                topic_week.setdefault(tk.topic_id, w.week_number)
    for e in edges:
        assert topic_week[e.source] <= topic_week[e.target]


def test_fsrs_spaced_repetition_and_update():
    r1 = retention_probability(2.0, 6.0)
    r2 = retention_probability(14.0, 6.0)
    assert 0 < r2 < r1 < 1.0

    tk = StudyTaskSchema(
        id="task_001",
        topic_id="t1",
        week_number=1,
        scheduled_date="2026-12-07",
        day_label="Mon",
        title="Test Task",
        description="Desc",
        task_type="concept_deep_dive",
        estimated_minutes=60,
        fsrs_stability_days=5.0,
    )
    hard_tk = update_task_fsrs(tk.model_copy(), "hard")
    easy_tk = update_task_fsrs(tk.model_copy(), "easy")
    assert hard_tk.fsrs_stability_days < tk.fsrs_stability_days
    assert easy_tk.fsrs_stability_days > tk.fsrs_stability_days

    topics = [TopicNode(id="t1", module_number=1, module_title="M1", title="Core Math", cognitive_load=8.0, is_critical_path=True, estimated_hours=4.0)]
    constraints = PlannerConstraints(total_weeks=6, weekly_hours=8.0, start_date="2026-12-07", include_spaced_repetition=True)
    weeks, _, _, _ = optimize_study_schedule(topics, [], [], constraints)
    weeks, proj = inject_spaced_repetition(weeks, topics, constraints)
    assert len(proj) == 6
    assert any(tk.task_type == "active_recall" for w in weeks for tk in w.tasks)
    assert proj[-1]["with_fsrs_pct"] >= proj[-1]["without_fsrs_pct"]


def test_hybrid_rag_retrieval():
    raw = """Module 1: Linear Algebra
Eigenvalues, SVD, and matrix norms.

Module 2: Grading Policy
Midterm exam is worth 25% in Week 6. Final project is worth 35%."""
    pages = pages_from_raw_text(raw)
    chunks = chunk_pages(pages, target_tokens=80, overlap_tokens=10)
    top = retrieve_hybrid_rrf(chunks, "What is the midterm exam weight?", top_k=2)
    assert len(top) >= 1
    resp = answer_syllabus_question(chunks, "What is the midterm exam weight?", top_k=2)
    assert "25%" in resp.answer or "Midterm" in resp.answer
    assert len(resp.citations) >= 1


def test_parse_five_unit_roman_and_ocr_inline_syllabus():
    from backend.ai.syllabus_parser import parse_syllabus

    ocr_text = (
        "MADHAV INSTITUTE OF TECHNOLOGY & SCIENCE, GWALIOR\n"
        "Centre for Artificial Intelligence EVOLUTIONARY ALGORITHMS AND OPTIMIZATION (31243103) "
        "COURSE OBJECTIVES To understand EA. "
        "UNIT I: Foundation of Optimization and Evolutionary Algorithms (EA) "
        "Introduction to Optimization, Search space representation, Objective function and fitness landscape. "
        "UNIT MI: Genetic Algorithms and Differential Evolution "
        "Chromosome encoding techniques, Selection methods, Crossover operators and mutation operators. "
        "UNIT M: Swarm Intelligence and Nature-Inspired Optimization "
        "Particle Swarm Optimization (PSO), Ant Colony Optimization (ACO), Artificial Bee Colony. "
        "UNIT IV: Multi-objective and Hybrid Optimization "
        "Concept of Dominance and Pareto-optimality, Weighted-Sum Method, Constraint Method. "
        "UNIT V: Multi-Objective Evolutionary Algorithms (MOEAs) "
        "NSGA-II, SPEA2, Neuro-evolution, AutoML and Neural Architecture Search. "
        "TEXT & REFERENCE BOOKS Introduction to Evolutionary Computing."
    )
    meta, topics, _, _ = parse_syllabus([(1, ocr_text)], "pymupdf_plus_easyocr", 10)
    assert "EVOLUTIONARY ALGORITHMS" in meta.course_title.upper()
    assert len(topics) == 5
    assert [t.module_number for t in topics] == [1, 2, 3, 4, 5]
    assert all(len(t.subtopics) >= 2 for t in topics)


def test_parse_multiline_section_gate_style_syllabus():
    from backend.ai.syllabus_parser import parse_syllabus

    gate_text = (
        "GATE 2027\n"
        "IIT Madras | Organizing Institute\n"
        "Data Science and Artificial Intelligence\n"
        "Section 1:\nProbability and Statistics\n"
        "Counting (permutation and combinations), probability axioms, Bayes Theorem, random variables.\n"
        "Section 2:\nLinear Algebra\n"
        "Vector space, subspaces, eigenvalues and eigenvectors, LU decomposition, singular value decomposition.\n"
        "Section 3:\nCalculus and Optimization\n"
        "Functions of a single variable, limit, continuity and differentiability, Taylor series.\n"
        "Section 4:\nMachine Learning\n"
        "Supervised Learning: regression and classification, ridge regression, logistic regression, SVM, PCA.\n"
    )
    meta, topics, _, _ = parse_syllabus([(1, gate_text)], "pymupdf_native", 12)
    assert len(topics) == 4
    assert topics[0].title == "Probability and Statistics"
    assert topics[1].title == "Linear Algebra"
    assert "Bayes Theorem" in topics[0].subtopics

