"""Generates realistic multi-page University Syllabus PDFs via ReportLab for 1-click testing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.config import settings


@dataclass(frozen=True)
class SampleSyllabusMeta:
    id: str
    course_code: str
    title: str
    subtitle: str
    recommended_weeks: int
    recommended_hours: float


SAMPLES_CATALOG: Dict[str, dict] = {
    "cs301_ml_dl": {
        "meta": SampleSyllabusMeta(
            id="cs301_ml_dl",
            course_code="CS301",
            title="Machine Learning & Deep Learning Systems",
            subtitle="12-Week University AI/ML Core Syllabus (10 Modules + Midterm + Capstone)",
            recommended_weeks=12,
            recommended_hours=10.0,
        ),
        "instructor": "Dr. Ananya Rao",
        "credits": "4 Credits",
        "description": (
            "This upper-division computer science syllabus covers statistical learning theory, "
            "optimization, classical supervised and unsupervised methods, deep neural architectures, "
            "transformers, and production MLOps deployment."
        ),
        "modules": [
            (
                "Module 1: Linear Algebra & Probability Foundations for ML",
                [
                    "Vector spaces, matrix decompositions, eigenvalues, and singular value decomposition (SVD)",
                    "Probability axioms, Bayes theorem, Gaussian distributions, and maximum likelihood intuition",
                    "Python NumPy vectorization and numerical stability exercises",
                ],
            ),
            (
                "Module 2: Supervised Linear & Logistic Regression",
                [
                    "Ordinary least squares, normal equations, and gradient descent convergence",
                    "Ridge (L2) and Lasso (L1) regularization and bias-variance tradeoff",
                    "Logistic regression, cross-entropy loss, and multiclass softmax implementation",
                ],
            ),
            (
                "Module 3: Model Evaluation, Validation & Decision Trees",
                [
                    "K-fold cross-validation, ROC-AUC, precision-recall, and calibration benchmarks",
                    "Decision trees, Gini impurity, random forests, and gradient boosted trees (XGBoost)",
                    "Ablation study and hyperparameter tuning lab",
                ],
            ),
            (
                "Module 4: Unsupervised Learning & Dimensionality Reduction",
                [
                    "K-means clustering, Gaussian Mixture Models, and Expectation-Maximization",
                    "Principal Component Analysis (PCA) derivation via SVD",
                    "UMAP/t-SNE visualization and anomaly detection",
                ],
            ),
            (
                "Module 5: Neural Networks & Backpropagation Mechanics",
                [
                    "Multilayer perceptrons, non-linear activations, and universal approximation principles",
                    "Computational graphs, chain rule derivation, and automatic differentiation",
                    "Batch normalization, AdamW optimization, and vanishing gradient diagnostics",
                ],
            ),
            (
                "Module 6: Convolutional Neural Networks & Computer Vision",
                [
                    "2D convolutions, receptive fields, residual connections (ResNet), and transfer learning",
                    "Object detection and semantic segmentation architectures",
                    "PyTorch image classifier training and GPU profiling lab",
                ],
            ),
            (
                "Module 7: Sequence Modeling, Attention & Transformers",
                [
                    "Recurrent networks (LSTM/GRU) limitations and sequence-to-sequence bottlenecks",
                    "Scaled dot-product self-attention, multi-head attention, and positional encodings",
                    "Decoder-only transformer architecture and build mini-GPT from scratch",
                ],
            ),
            (
                "Module 8: Large Language Models, RAG & Alignment",
                [
                    "Pretraining scaling laws, LoRA parameter-efficient fine-tuning, and RLHF tradeoffs",
                    "Retrieval-Augmented Generation (RAG), hybrid vector + BM25 search, and evaluation",
                    "Safety auditing, hallucination mitigation, and faithfulness metrics",
                ],
            ),
            (
                "Module 9: Capstone AI System Design & Production MLOps",
                [
                    "Design end-to-end multi-agent RAG + ML microservice architecture",
                    "Quantization, FastAPI serving latency optimization, and drift monitoring",
                    "Final capstone synthesis project presentation and technical report",
                ],
            ),
        ],
        "assessments": [
            ("Quiz 1: Linear Algebra & Regression Check", "Week 3", "10%"),
            ("Midterm Exam: Classical ML & Optimization Theory", "Week 6", "25%"),
            ("Lab Assignment 3: PyTorch CNN & Attention Implementation", "Week 9", "20%"),
            ("Capstone AI System Design Project", "Week 11", "25%"),
            ("Final Exam: Comprehensive Deep Learning & LLMs", "Week 12", "20%"),
        ],
    },
    "cs204_dsa": {
        "meta": SampleSyllabusMeta(
            id="cs204_dsa",
            course_code="CS204",
            title="Data Structures & Algorithms",
            subtitle="12-Week Systems & Coding Interview Core Syllabus (8 Modules)",
            recommended_weeks=12,
            recommended_hours=9.0,
        ),
        "instructor": "Prof. Marcus Chen",
        "credits": "4 Credits",
        "description": (
            "Rigorous study of fundamental data structures, asymptotic complexity proofs, "
            "graph algorithms, greedy strategies, dynamic programming, and NP-completeness."
        ),
        "modules": [
            (
                "Module 1: Asymptotic Complexity & Recurrence Analysis",
                [
                    "Big-O, Omega, Theta definitions and amortized potential method analysis",
                    "Master theorem derivations and recursion tree proofs",
                ],
            ),
            (
                "Module 2: Arrays, Linked Lists, Stacks & Queues",
                [
                    "Dynamic array resizing amortized analysis and cache locality",
                    "Monotonic stacks, sliding window deques, and coding lab",
                ],
            ),
            (
                "Module 3: Divide & Conquer, Sorting & Hashing",
                [
                    "Quicksort randomized analysis, mergesort, and lower-bound sorting proofs",
                    "Open addressing vs chaining hash tables and universal hashing",
                ],
            ),
            (
                "Module 4: Trees, Heaps & Self-Balancing BSTs",
                [
                    "Binary tree traversals, binary heaps, and priority queue applications",
                    "AVL tree rotations, Red-Black invariants, and B-tree indexing",
                ],
            ),
            (
                "Module 5: Graph Traversal & Topological Sorting",
                [
                    "BFS, DFS, strongly connected components (Tarjan/Kosaraju)",
                    "DAG topological ordering and cycle detection algorithms",
                ],
            ),
            (
                "Module 6: Shortest Paths & Minimum Spanning Trees",
                [
                    "Dijkstra, Bellman-Ford, and Floyd-Warshall dynamic shortest paths",
                    "Union-Find (DSU) with path compression, Kruskal, and Prim MST",
                ],
            ),
            (
                "Module 7: Dynamic Programming & Memoization",
                [
                    "Optimal substructure, overlapping subproblems, and state transitions",
                    "Knapsack, edit distance, DAG longest path, and bitmask DP",
                ],
            ),
            (
                "Module 8: Greedy Algorithms & NP-Completeness",
                [
                    "Exchange arguments, interval scheduling, and Huffman coding",
                    "Polynomial-time reductions, P vs NP, and approximation algorithms",
                ],
            ),
        ],
        "assessments": [
            ("Assignment 1: Complexity & Linear Structures Lab", "Week 3", "15%"),
            ("Midterm Exam: Trees, Hashing & Graph Traversal", "Week 6", "25%"),
            ("Assignment 2: Shortest Paths & Dynamic Programming", "Week 9", "20%"),
            ("Final Exam: Comprehensive Algorithms & Complexity", "Week 12", "40%"),
        ],
    },
    "math210_linalg_prob": {
        "meta": SampleSyllabusMeta(
            id="math210_linalg_prob",
            course_code="MATH210",
            title="Linear Algebra & Probability for Data Science",
            subtitle="10-Week Quantitative Mathematics Foundation Syllabus",
            recommended_weeks=10,
            recommended_hours=8.0,
        ),
        "instructor": "Dr. Elena Vance",
        "credits": "3 Credits",
        "description": (
            "Mathematical foundations bridging linear algebra, matrix decompositions, "
            "probability theory, stochastic processes, and Bayesian statistical inference."
        ),
        "modules": [
            (
                "Module 1: Vectors, Linear Independence & Basis",
                [
                    "Vector spaces, span, subspaces, and geometric projections",
                    "Orthogonality, Gram-Schmidt process, and QR factorization",
                ],
            ),
            (
                "Module 2: Systems of Equations & Matrix Factorizations",
                [
                    "Gaussian elimination, LU decomposition, rank-nullity theorem",
                    "Least squares approximation and pseudoinverse derivations",
                ],
            ),
            (
                "Module 3: Eigenvalues, Spectral Theorem & SVD",
                [
                    "Characteristic polynomials, diagonalization, and symmetric matrices",
                    "Singular Value Decomposition (SVD) and low-rank matrix approximation",
                ],
            ),
            (
                "Module 4: Probability Spaces & Conditional Independence",
                [
                    "Kolmogorov axioms, combinatorics, and Bayes rule",
                    "Independence, conditional expectation, and law of total probability",
                ],
            ),
            (
                "Module 5: Random Variables & Multivariate Distributions",
                [
                    "Binomial, Poisson, Exponential, and Multivariate Gaussian distributions",
                    "Covariance matrices, Central Limit Theorem, and concentration inequalities",
                ],
            ),
            (
                "Module 6: Markov Chains & Bayesian Statistical Inference",
                [
                    "Transition matrices, stationary distributions, and PageRank eigenvalue interpretation",
                    "Prior/posterior conjugate updates, MAP vs MLE estimation, and credible intervals",
                ],
            ),
        ],
        "assessments": [
            ("Quiz 1: Vector Spaces & QR Proof", "Week 3", "15%"),
            ("Midterm Exam: Eigenvalues, SVD & Probability", "Week 5", "30%"),
            ("Assignment 2: Markov Chain & Bayesian Simulation", "Week 8", "20%"),
            ("Final Exam: Comprehensive Linear Algebra & Probability", "Week 10", "35%"),
        ],
    },
}


def ensure_sample_pdf(sample_id: str) -> Path:
    """Create (or return cached) realistic university syllabus PDF on disk."""
    if sample_id not in SAMPLES_CATALOG:
        raise KeyError(f"Unknown sample syllabus: {sample_id}")

    settings.ensure_dirs()
    pdf_path = settings.SAMPLES_DIR / f"{sample_id}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 1000:
        return pdf_path

    data = SAMPLES_CATALOG[sample_id]
    meta: SampleSyllabusMeta = data["meta"]

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=LETTER,
        leftMargin=54,
        rightMargin=54,
        topMargin=52,
        bottomMargin=52,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SyllabusTitle",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=6,
        textColor=colors.HexColor("#0f172a"),
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=12,
        spaceAfter=4,
        textColor=colors.HexColor("#1e293b"),
    )
    mod_style = ParagraphStyle(
        "ModH3",
        parent=styles["Heading3"],
        fontSize=11.5,
        spaceBefore=8,
        spaceAfter=3,
        textColor=colors.HexColor("#312e81"),
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=4,
    )

    story = [
        Paragraph(f"{meta.course_code}: {meta.title}", title_style),
        Paragraph(
            f"Instructor: {data['instructor']} | {data['credits']} | Semester Duration: {meta.recommended_weeks} Weeks",
            body_style,
        ),
        Spacer(1, 6),
        Paragraph("Course Description", h2_style),
        Paragraph(data["description"], body_style),
        Spacer(1, 6),
        Paragraph("Course Modules & Weekly Syllabus Topics", h2_style),
    ]

    for mod_title, bullets in data["modules"]:
        story.append(Paragraph(mod_title, mod_style))
        for b in bullets:
            story.append(Paragraph(f"- {b}", body_style))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Grading & Assessment Schedule", h2_style))

    table_data = [["Assessment", "Target Timeline", "Weight"]]
    for name, wk, wt in data["assessments"]:
        table_data.append([name, wk, wt])
        story.append(Paragraph(f"- {name} ({wk}) — {wt}", body_style))

    story.append(Spacer(1, 8))
    tbl = Table(table_data, colWidths=[280, 110, 80])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#312e81")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(tbl)
    doc.build(story)
    return pdf_path


def list_sample_catalog() -> list[dict]:
    items = []
    for sid, payload in SAMPLES_CATALOG.items():
        meta: SampleSyllabusMeta = payload["meta"]
        items.append(
            {
                "id": meta.id,
                "course_code": meta.course_code,
                "title": meta.title,
                "subtitle": meta.subtitle,
                "recommended_weeks": meta.recommended_weeks,
                "recommended_hours": meta.recommended_hours,
            }
        )
    return items
