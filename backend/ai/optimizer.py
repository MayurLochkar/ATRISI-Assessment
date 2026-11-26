"""SciPy MILP Multi-Objective Study Schedule Optimizer & Task Synthesizer."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from backend.schemas import (
    ExtractedAssessment,
    PlannerConstraints,
    PrerequisiteEdge,
    StudyTaskSchema,
    TopicNode,
    WeeklySummarySchema,
)

DAY_INDEX = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}


def _parse_start_date(start_date_str: str) -> date:
    if start_date_str:
        try:
            return date.fromisoformat(start_date_str)
        except ValueError:
            pass
    today = date.today()
    # Next Monday (or today if Monday)
    days_ahead = (0 - today.weekday()) % 7
    return today + timedelta(days=days_ahead)


def _week_day_dates(
    plan_start: date, week_num: int, preferred_days: list[str]
) -> list[tuple[str, str]]:
    """Return ordered list of (day_label, YYYY-MM-DD) within week_num."""
    valid_days = [d for d in preferred_days if d in DAY_INDEX]
    if not valid_days:
        valid_days = ["Mon", "Wed", "Fri", "Sat"]
    valid_days.sort(key=lambda d: DAY_INDEX[d])

    # Align week 1 start to Monday of plan_start's week
    monday_w1 = plan_start - timedelta(days=plan_start.weekday())
    week_monday = monday_w1 + timedelta(weeks=week_num - 1)
    result: list[tuple[str, str]] = []
    for d in valid_days:
        dt = week_monday + timedelta(days=DAY_INDEX[d])
        result.append((d, dt.isoformat()))
    return result


def _week_date_range(plan_start: date, week_num: int) -> str:
    monday_w1 = plan_start - timedelta(days=plan_start.weekday())
    start_w = monday_w1 + timedelta(weeks=week_num - 1)
    end_w = start_w + timedelta(days=6)
    return f"{start_w.strftime('%b %d')} – {end_w.strftime('%b %d')}"


def _decompose_topic_into_tasks(
    topic: TopicNode,
    week_num: int,
    day_slots: list[tuple[str, str]],
    slot_cursor: int,
    task_counter: int,
) -> tuple[list[StudyTaskSchema], int, int]:
    """Decompose a curriculum topic into pedagogical micro-tasks (Deep Dive, Practice, Project)."""
    total_mins = max(45, int(round(topic.estimated_hours * 60 / 15.0) * 15))
    subtopics = topic.subtopics or [f"Core principles of {topic.title}", "Worked examples", "Self-check problems"]

    # Determine number of study blocks (1 to 3) based on effort
    if total_mins <= 110:
        splits = [("concept_deep_dive", 1.0, "Concept Mastery & Guided Notes")]
    elif total_mins <= 210:
        splits = [
            ("concept_deep_dive", 0.50, "Concept Deep-Dive & Structural Mapping"),
            ("hands_on_practice", 0.50, "Applied Problem Solving & Implementation"),
        ]
    else:
        third_type = "project_milestone" if topic.bloom_level in ("Evaluate", "Create") else "hands_on_practice"
        third_label = "Synthesis & Architecture Lab" if third_type == "project_milestone" else "Advanced Drill & Edge-Case Analysis"
        splits = [
            ("concept_deep_dive", 0.38, "Core Theory & Conceptual Deep-Dive"),
            ("hands_on_practice", 0.34, "Guided Implementation & Exercises"),
            (third_type, 0.28, third_label),
        ]

    priority = "critical" if topic.is_critical_path else ("high" if topic.cognitive_load >= 7.0 else "medium")
    tasks: list[StudyTaskSchema] = []

    for idx, (ttype, ratio, subtitle) in enumerate(splits):
        day_label, sched_date = day_slots[slot_cursor % len(day_slots)]
        slot_cursor += 1
        task_counter += 1
        mins = max(30, int(round((total_mins * ratio) / 15.0) * 15))

        # Partition subtopics cleanly across micro-tasks
        chunk_size = max(1, (len(subtopics) + len(splits) - 1) // len(splits))
        slice_subs = subtopics[idx * chunk_size : (idx + 1) * chunk_size] or subtopics[:2]

        if ttype == "concept_deep_dive":
            deliverable = f"1-page structured concept map & summary notes for {topic.title}"
            steps = [
                f"Read syllabus sections covering: {', '.join(slice_subs[:3])}",
                "Annotate key definitions, assumptions, and mathematical/system invariants",
                "Write a 3-bullet 'teach-back' explanation without looking at notes",
            ]
        elif ttype == "project_milestone":
            deliverable = f"Working prototype / design critique addressing {topic.title}"
            steps = [
                f"Design and build component for: {', '.join(slice_subs[:3])}",
                "Run verification tests / ablation checks against baseline",
                "Document trade-offs and edge cases in study log",
            ]
        else:
            deliverable = f"Completed problem set / code lab for {topic.title}"
            steps = [
                f"Solve practice exercises on: {', '.join(slice_subs[:3])}",
                "Trace failure cases and verify time/accuracy behavior",
                "Flag any weak sub-concepts for spaced repetition review",
            ]

        initial_stability = round(max(2.5, 7.5 - 0.45 * topic.cognitive_load), 2)
        tasks.append(
            StudyTaskSchema(
                id=f"task_{task_counter:03d}",
                topic_id=topic.id,
                week_number=week_num,
                scheduled_date=sched_date,
                day_label=day_label,
                title=f"{topic.title}: {subtitle}",
                description=(
                    f"Module {topic.module_number} ({topic.module_title}) • "
                    f"Focus: {', '.join(slice_subs[:4])}"
                ),
                task_type=ttype,  # type: ignore[arg-type]
                bloom_level=topic.bloom_level,
                estimated_minutes=mins,
                priority=priority,  # type: ignore[arg-type]
                is_completed=False,
                fsrs_stability_days=initial_stability,
                predicted_retention=0.94,
                deliverable=deliverable,
                substeps=steps,
                page_refs=topic.page_refs,
            )
        )

    return tasks, slot_cursor, task_counter


def optimize_study_schedule(
    topics: list[TopicNode],
    edges: list[PrerequisiteEdge],
    assessments: list[ExtractedAssessment],
    constraints: PlannerConstraints,
    start_week: int = 1,
    existing_weeks: Optional[list[WeeklySummarySchema]] = None,
) -> Tuple[list[WeeklySummarySchema], str, Optional[str], Optional[float]]:
    """
    Solve Mixed-Integer Linear Program (MILP) assigning topics to weeks [start_week..W]
    subject to Prerequisite DAG ordering, Assessment deadlines, and Weekly Hour Capacity.
    Returns (weeks, solver_status, feasibility_warning, recommended_weekly_hours).
    """
    W_total = constraints.total_weeks
    active_weeks = list(range(start_week, W_total + 1))
    W = len(active_weeks)
    N = len(topics)

    plan_start = _parse_start_date(constraints.start_date)

    # Preserve completed weeks when replanning from start_week > 1
    preserved_weeks: list[WeeklySummarySchema] = []
    if existing_weeks and start_week > 1:
        preserved_weeks = [w for w in existing_weeks if w.week_number < start_week]

    if N == 0 or W == 0:
        empty_weeks = preserved_weeks + [
            WeeklySummarySchema(
                week_number=wk,
                week_label=f"Week {wk}",
                date_range=_week_date_range(plan_start, wk),
                focus_theme="Review & Consolidation",
                planned_hours=0.0,
                capacity_hours=constraints.weekly_hours,
                cognitive_load_avg=0.0,
                is_exam_week=False,
                tasks=[],
            )
            for wk in active_weeks
        ]
        return empty_weeks, "optimal_trivial", None, None

    # Map topic ID -> index
    id_to_idx = {t.id: idx for idx, t in enumerate(topics)}
    hours = np.array([t.estimated_hours for t in topics], dtype=float)
    total_topic_hours = float(hours.sum())
    total_capacity = W * float(constraints.weekly_hours)

    feasibility_warning: Optional[str] = None
    recommended_hours: Optional[float] = None
    if total_topic_hours > total_capacity * 1.05:
        rec = round((total_topic_hours * 1.1) / max(1, W), 1)
        recommended_hours = rec
        feasibility_warning = (
            f"Estimated syllabus workload ({total_topic_hours:.1f}h) exceeds your "
            f"{W}-week budget ({total_capacity:.1f}h). MILP relaxed capacity smoothly; "
            f"consider {rec:.1f}h/week or extending timeline."
        )

    # Identify exam weeks for capacity buffer / deadline precedence
    exam_weeks = {
        a.target_week
        for a in assessments
        if a.target_week and start_week <= a.target_week <= W_total and a.assessment_type in ("midterm", "final", "quiz")
    }

    # Weekly effective capacities C_w
    cap = np.full(W, float(constraints.weekly_hours), dtype=float)
    for w_idx, wk_num in enumerate(active_weeks):
        if wk_num in exam_weeks and constraints.learning_intensity == "exam_focused":
            cap[w_idx] *= 0.85  # Reserve 15% buffer for exam prep tasks

    # MILP Decision Vector Layout:
    # - x[i, w] for i in 0..N-1, w in 0..W-1 -> N*W binary variables
    # - o[w] for w in 0..W-1 -> W continuous overtime variables >= 0
    # - L_max -> 1 continuous peak utilization variable >= 0
    num_x = N * W
    idx_L = num_x + W
    total_vars = num_x + W + 1

    def x_idx(i: int, w: int) -> int:
        return i * W + w

    # Objective coefficients c
    c = np.zeros(total_vars, dtype=float)
    for i, t in enumerate(topics):
        for w in range(W):
            wk_num = active_weeks[w]
            pos_ratio = w / max(1, W - 1)
            # Gentle temporal pacing cost
            if constraints.learning_intensity == "front_loaded":
                pacing = 0.6 * pos_ratio
            elif constraints.learning_intensity == "exam_focused":
                pacing = 0.25 * (1.0 - pos_ratio)
            else:
                # Prefer smooth alignment with natural curriculum progression
                target_ratio = i / max(1, N - 1)
                pacing = 0.45 * abs(pos_ratio - target_ratio)

            # High-PageRank foundational topics earn small bonus for earlier placement
            prereq_bonus = 0.35 * t.pagerank_centrality * pos_ratio
            # Exam week slight deterrence for introducing brand-new heavy topics
            exam_pen = 0.8 if (wk_num in exam_weeks and t.cognitive_load >= 7.0) else 0.0
            c[x_idx(i, w)] = pacing + prereq_bonus + exam_pen

    # Penalize overtime heavily and minimize peak utilization L_max for load balance
    for w in range(W):
        c[num_x + w] = 80.0
    c[idx_L] = 20.0

    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []

    # 1. Each topic assigned to exactly one week: sum_w x[i, w] == 1
    for i in range(N):
        row = np.zeros(total_vars, dtype=float)
        for w in range(W):
            row[x_idx(i, w)] = 1.0
        rows.append(row)
        lb.append(1.0)
        ub.append(1.0)

    # 2. Prerequisite DAG ordering: sum_w w*x[u, w] - sum_w w*x[v, w] <= 0
    for edge in edges:
        if edge.source in id_to_idx and edge.target in id_to_idx:
            u = id_to_idx[edge.source]
            v = id_to_idx[edge.target]
            if u == v:
                continue
            row = np.zeros(total_vars, dtype=float)
            for w in range(W):
                row[x_idx(u, w)] = float(w)
                row[x_idx(v, w)] = -float(w)
            rows.append(row)
            lb.append(-np.inf)
            ub.append(0.0)

    # 3. Capacity + Overtime: sum_i h_i x[i, w] - o_w <= cap[w]
    for w in range(W):
        row = np.zeros(total_vars, dtype=float)
        for i in range(N):
            row[x_idx(i, w)] = hours[i]
        row[num_x + w] = -1.0
        rows.append(row)
        lb.append(-np.inf)
        ub.append(cap[w])

    # 4. Min-Max Load Leveling: sum_i h_i x[i, w] - cap[w]*L_max <= 0
    for w in range(W):
        row = np.zeros(total_vars, dtype=float)
        for i in range(N):
            row[x_idx(i, w)] = hours[i]
        row[idx_L] = -cap[w]
        rows.append(row)
        lb.append(-np.inf)
        ub.append(0.0)

    # 4b. When N >= W + 2, require at least 1 topic in every week so no week is left empty
    if N >= W + 2:
        for w in range(W):
            row = np.zeros(total_vars, dtype=float)
            for i in range(N):
                row[x_idx(i, w)] = 1.0
            rows.append(row)
            lb.append(1.0)
            ub.append(float(N))

    # 5. Soft deadline guidance for midterm/quiz assessments
    mid_assess = [a for a in assessments if a.assessment_type == "midterm" and a.target_week]
    if mid_assess and N >= 4:
        mid_wk = mid_assess[0].target_week
        if mid_wk and start_week <= mid_wk <= W_total:
            rel_mid = mid_wk - start_week
            # First half of topics should finish by midterm week
            for i in range(N // 2):
                row = np.zeros(total_vars, dtype=float)
                for w in range(W):
                    row[x_idx(i, w)] = float(w)
                rows.append(row)
                lb.append(-np.inf)
                ub.append(float(rel_mid))

    A = np.vstack(rows)
    constraints_milp = LinearConstraint(A, lb, ub)

    integrality = np.zeros(total_vars, dtype=int)
    integrality[:num_x] = 1

    lower_b = np.zeros(total_vars, dtype=float)
    upper_b = np.full(total_vars, np.inf, dtype=float)
    upper_b[:num_x] = 1.0
    bounds = Bounds(lower_b, upper_b)

    res = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints_milp,
        options={"time_limit": 4.0},
    )

    topic_week_map: dict[int, int] = {}
    if res.success and res.x is not None:
        solver_status = "milp_optimal" if res.status == 0 else "milp_feasible"
        for i in range(N):
            best_w = int(np.argmax([res.x[x_idx(i, w)] for w in range(W)]))
            topic_week_map[i] = active_weeks[best_w]
    else:
        # Topological greedy capacity-aware fallback (guaranteed valid!)
        solver_status = "topological_greedy_fallback"
        week_loads = [0.0] * W
        min_w = 0
        for i in range(N):
            # Respect any already-placed prerequisites
            prereq_weeks = [
                topic_week_map[id_to_idx[p]] - start_week
                for p in topics[i].prerequisites
                if p in id_to_idx and id_to_idx[p] in topic_week_map
            ]
            earliest = max(prereq_weeks) if prereq_weeks else min_w
            chosen = min(
                range(earliest, W),
                key=lambda w: week_loads[w] / max(0.5, cap[w]),
            )
            topic_week_map[i] = active_weeks[chosen]
            week_loads[chosen] += hours[i]
            min_w = max(min_w, earliest)

    # Group topics by week and synthesize concrete daily tasks
    topics_by_week: dict[int, list[TopicNode]] = {wk: [] for wk in active_weeks}
    for i, t in enumerate(topics):
        topics_by_week[topic_week_map[i]].append(t)

    task_counter = sum(len(w.tasks) for w in preserved_weeks)
    built_weeks: list[WeeklySummarySchema] = list(preserved_weeks)

    for wk_num in active_weeks:
        wk_topics = topics_by_week[wk_num]
        day_slots = _week_day_dates(plan_start, wk_num, constraints.preferred_days)
        slot_cursor = 0
        wk_tasks: list[StudyTaskSchema] = []

        for t in wk_topics:
            new_tasks, slot_cursor, task_counter = _decompose_topic_into_tasks(
                t, wk_num, day_slots, slot_cursor, task_counter
            )
            wk_tasks.extend(new_tasks)

        # Inject explicit assessment prep tasks for assessments falling in this week
        wk_assessments = [a for a in assessments if a.target_week == wk_num]
        is_exam = any(a.assessment_type in ("midterm", "final", "quiz") for a in wk_assessments)
        milestone_alert = (
            " · ".join(
                f"{a.title} ({a.weight_percent:.0f}%)" if a.weight_percent else a.title
                for a in wk_assessments
            )
            if wk_assessments
            else None
        )

        for a in wk_assessments:
            task_counter += 1
            # Place prep task toward end of preferred study days in that week
            d_label, s_date = day_slots[-1]
            prep_mins = 90 if a.assessment_type in ("midterm", "final") else 60
            wk_tasks.append(
                StudyTaskSchema(
                    id=f"task_{task_counter:03d}",
                    topic_id=wk_topics[-1].id if wk_topics else None,
                    week_number=wk_num,
                    scheduled_date=s_date,
                    day_label=d_label,
                    title=f"Assessment Prep: {a.title}",
                    description=a.description or f"Timed mock practice & rubric check for {a.title}",
                    task_type="assessment_prep",
                    bloom_level="Evaluate",
                    estimated_minutes=prep_mins,
                    priority="critical",
                    is_completed=False,
                    fsrs_stability_days=5.0,
                    predicted_retention=0.95,
                    deliverable=f"Timed self-test & error log before {a.title}",
                    substeps=[
                        "Simulate exam / rubric conditions without notes",
                        "Review mistakes and drill weakest 2 subtopics",
                        "Verify submission / exam checklist",
                    ],
                    page_refs=[1],
                )
            )

        # Sort weekly tasks chronologically by date
        wk_tasks.sort(key=lambda tk: tk.scheduled_date)
        planned_hrs = round(sum(tk.estimated_minutes for tk in wk_tasks) / 60.0, 2)
        load_avg = (
            round(float(np.mean([t.cognitive_load for t in wk_topics])), 2)
            if wk_topics
            else 2.5
        )
        theme = (
            " & ".join(dict.fromkeys(t.module_title for t in wk_topics[:2]))
            if wk_topics
            else ("Exam Synthesis & Active Recall" if is_exam else "Spaced Consolidation & Catch-Up")
        )

        built_weeks.append(
            WeeklySummarySchema(
                week_number=wk_num,
                week_label=f"Week {wk_num}",
                date_range=_week_date_range(plan_start, wk_num),
                focus_theme=theme[:100],
                planned_hours=planned_hrs,
                capacity_hours=float(constraints.weekly_hours),
                cognitive_load_avg=load_avg,
                is_exam_week=is_exam,
                milestone_alert=milestone_alert,
                tasks=wk_tasks,
            )
        )

    # Guarantee no week is completely empty (e.g. if N < W even after expansion)
    last_topic = topics[-1] if topics else None
    for idx_w, w_obj in enumerate(built_weeks):
        if w_obj.week_number < start_week:
            continue
        if len(w_obj.tasks) == 0 and last_topic is not None:
            ref_topic = topics[min(len(topics) - 1, idx_w)]
            day_slots = _week_day_dates(plan_start, w_obj.week_number, constraints.preferred_days)
            d_label, s_date = day_slots[0]
            task_counter += 1
            w_obj.tasks.append(
                StudyTaskSchema(
                    id=f"task_{task_counter:03d}",
                    topic_id=ref_topic.id,
                    week_number=w_obj.week_number,
                    scheduled_date=s_date,
                    day_label=d_label,
                    title=f"Synthesis & Practice: {ref_topic.title}",
                    description=f"Consolidated problem-solving & active synthesis for {ref_topic.module_title}",
                    task_type="hands_on_practice",
                    bloom_level="Apply",
                    estimated_minutes=60,
                    priority="medium",
                    is_completed=False,
                    fsrs_stability_days=5.5,
                    predicted_retention=0.92,
                    deliverable=f"Solved practice set & summary sheet for {ref_topic.title}",
                    substeps=[
                        f"Review key concepts from {ref_topic.title}",
                        "Complete 3 timed practice problems without notes",
                        "Update personal formula/concept cheat-sheet",
                    ],
                    page_refs=ref_topic.page_refs,
                )
            )
            w_obj.planned_hours = round(sum(tk.estimated_minutes for tk in w_obj.tasks) / 60.0, 2)
            w_obj.focus_theme = f"Synthesis: {ref_topic.module_title}"[:100]

    return built_weeks, solver_status, feasibility_warning, recommended_hours
