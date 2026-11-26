"""FSRS-inspired Exponential Memory Decay & Spaced Repetition Scheduler."""

from __future__ import annotations

import math
from typing import Tuple

from backend.schemas import (
    ConfidenceRating,
    PlannerConstraints,
    StudyTaskSchema,
    TopicNode,
    WeeklySummarySchema,
)


def retention_probability(days_elapsed: float, stability_days: float) -> float:
    """Exponential forgetting curve R(t) = exp(-t / S)."""
    s = max(1.0, stability_days)
    t = max(0.0, days_elapsed)
    return round(float(math.exp(-t / s)), 4)


def update_task_fsrs(task: StudyTaskSchema, rating: ConfidenceRating) -> StudyTaskSchema:
    """Update memory stability S and predicted retention upon student confidence feedback."""
    s = max(1.5, task.fsrs_stability_days)
    if rating == "hard":
        new_s = round(max(1.5, s * 0.72), 2)
        new_ret = retention_probability(2.0, new_s)
    elif rating == "good":
        new_s = round(min(45.0, s * 1.65), 2)
        new_ret = retention_probability(2.0, new_s)
    else:  # easy
        new_s = round(min(60.0, s * 2.40), 2)
        new_ret = retention_probability(2.0, new_s)

    task.confidence_rating = rating
    task.fsrs_stability_days = new_s
    task.predicted_retention = new_ret
    return task


def inject_spaced_repetition(
    weeks: list[WeeklySummarySchema],
    topics: list[TopicNode],
    constraints: PlannerConstraints,
    start_week: int = 1,
) -> Tuple[list[WeeklySummarySchema], list[dict[str, float | int]]]:
    """
    Inject targeted FSRS Active Recall micro-tasks for high-load / Critical Path topics
    and generate week-by-week memory retention projection curves (With FSRS vs Without).
    """
    topic_map = {t.id: t for t in topics}
    total_weeks = constraints.total_weeks

    if constraints.include_spaced_repetition and total_weeks >= 3:
        max_task_id = 0
        for w in weeks:
            for tk in w.tasks:
                if tk.id.startswith("task_"):
                    try:
                        max_task_id = max(max_task_id, int(tk.id.split("_")[1]))
                    except ValueError:
                        pass

        # Find week where each topic is initially studied
        first_week_studied: dict[str, int] = {}
        first_date_by_week: dict[int, tuple[str, str]] = {}
        for w in weeks:
            if w.tasks:
                first_date_by_week[w.week_number] = (
                    w.tasks[-1].day_label,
                    w.tasks[-1].scheduled_date,
                )
            for tk in w.tasks:
                if tk.topic_id and tk.topic_id not in first_week_studied:
                    first_week_studied[tk.topic_id] = w.week_number

        # Select high-leverage topics (Critical Path or cognitive_load >= 6.0)
        priority_topics = sorted(
            [
                t
                for t in topics
                if t.id in first_week_studied
                and first_week_studied[t.id] >= start_week
                and (t.is_critical_path or t.cognitive_load >= 5.8)
            ],
            key=lambda x: (x.is_critical_path, x.cognitive_load),
            reverse=True,
        )[:10]

        week_lookup = {w.week_number: w for w in weeks}

        for t in priority_topics:
            w0 = first_week_studied[t.id]
            # Schedule 1st recall (+1 or +2 weeks) and optional 2nd recall (+4 weeks)
            intervals = [2] if w0 + 2 <= total_weeks else ([1] if w0 + 1 <= total_weeks else [])
            if t.is_critical_path and w0 + 4 <= total_weeks:
                intervals.append(4)

            for step_idx, delta_w in enumerate(intervals, start=1):
                target_wk = w0 + delta_w
                w_obj = week_lookup.get(target_wk)
                if w_obj is None:
                    continue
                # Avoid overloading already over-capacity weeks unless spare room or light 20m review
                if w_obj.planned_hours > w_obj.capacity_hours * 1.15 and step_idx > 1:
                    continue

                max_task_id += 1
                day_lbl, sched_dt = first_date_by_week.get(
                    target_wk,
                    (constraints.preferred_days[-1] if constraints.preferred_days else "Sat", w_obj.date_range.split("–")[0].strip()),
                )
                # Ensure valid ISO date if week had no prior tasks
                if len(sched_dt) != 10 or "-" not in sched_dt:
                    from backend.ai.optimizer import _parse_start_date, _week_day_dates

                    slots = _week_day_dates(
                        _parse_start_date(constraints.start_date),
                        target_wk,
                        constraints.preferred_days,
                    )
                    day_lbl, sched_dt = slots[-1]

                stability_after = round(4.5 * (1.75 ** step_idx), 2)
                w_obj.tasks.append(
                    StudyTaskSchema(
                        id=f"task_{max_task_id:03d}",
                        topic_id=t.id,
                        week_number=target_wk,
                        scheduled_date=sched_dt,
                        day_label=day_lbl,
                        title=f"FSRS Recall #{step_idx}: {t.title}",
                        description=(
                            f"Spaced active-recall checkpoint (+{delta_w}w retention boost) for "
                            f"{t.module_title}"
                        ),
                        task_type="active_recall",
                        bloom_level="Analyze" if t.cognitive_load >= 6.5 else "Apply",
                        estimated_minutes=25 if step_idx == 1 else 20,
                        priority="high" if t.is_critical_path else "medium",
                        is_completed=False,
                        fsrs_stability_days=stability_after,
                        predicted_retention=retention_probability(delta_w * 7.0, stability_after),
                        deliverable=f"Closed-book 3-question recall check for {t.title}",
                        substeps=[
                            "Close notes and write core formulas/concepts from memory",
                            f"Quiz yourself on: {', '.join((t.subtopics or [t.title])[:2])}",
                            "Rate recall confidence (Hard / Good / Easy) to update FSRS curve",
                        ],
                        page_refs=t.page_refs,
                    )
                )
                w_obj.tasks.sort(key=lambda tk: tk.scheduled_date)
                w_obj.planned_hours = round(
                    sum(tk.estimated_minutes for tk in w_obj.tasks) / 60.0, 2
                )

    # Build retention projection curve across weeks 1..total_weeks
    projection: list[dict[str, float | int]] = []
    # Track cumulative studied topics & review boosts
    studied_by_week: dict[str, int] = {}
    review_weeks_by_topic: dict[str, list[int]] = {}
    for w in weeks:
        for tk in w.tasks:
            if tk.topic_id:
                if tk.task_type == "active_recall":
                    review_weeks_by_topic.setdefault(tk.topic_id, []).append(w.week_number)
                elif tk.topic_id not in studied_by_week:
                    studied_by_week[tk.topic_id] = w.week_number

    for wk in range(1, total_weeks + 1):
        active_topics = [tid for tid, sw in studied_by_week.items() if sw <= wk]
        if not active_topics:
            projection.append(
                {
                    "week": wk,
                    "with_fsrs_pct": 95.0,
                    "without_fsrs_pct": 95.0,
                }
            )
            continue

        fsrs_scores: list[float] = []
        raw_scores: list[float] = []
        for tid in active_topics:
            t_obj = topic_map.get(tid)
            base_s = 6.5 - 0.35 * (t_obj.cognitive_load if t_obj else 5.0)
            base_s = max(2.8, base_s)

            # Without FSRS: decays monotonically from week studied
            days_no_review = max(1.0, (wk - studied_by_week[tid]) * 7.0)
            raw_scores.append(retention_probability(days_no_review, base_s) * 100.0)

            # With FSRS: each recall event resets elapsed clock and scales stability x1.8
            revs = [rw for rw in review_weeks_by_topic.get(tid, []) if rw <= wk]
            last_touch = max([studied_by_week[tid]] + revs)
            boosted_s = base_s * (1.85 ** len(revs))
            days_since_touch = max(1.0, (wk - last_touch) * 7.0)
            fsrs_scores.append(retention_probability(days_since_touch, boosted_s) * 100.0)

        projection.append(
            {
                "week": wk,
                "with_fsrs_pct": round(sum(fsrs_scores) / len(fsrs_scores), 1),
                "without_fsrs_pct": round(sum(raw_scores) / len(raw_scores), 1),
            }
        )

    return weeks, projection
