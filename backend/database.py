"""SQLAlchemy 2.0 SQLite persistence for study plans and indexed RAG chunks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from backend.config import settings
from backend.schemas import StudyPlanListItem, StudyPlanResponse, SyllabusChunk


class Base(DeclarativeBase):
    pass


class StudyPlanRecord(Base):
    __tablename__ = "study_plans"

    plan_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    course_code: Mapped[str] = mapped_column(String(64), nullable=False)
    course_title: Mapped[str] = mapped_column(String(255), nullable=False)
    engine_used: Mapped[str] = mapped_column(String(64), nullable=False)
    plan_json: Mapped[str] = mapped_column(Text, nullable=False)
    chunks_json: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False}
    if settings.DATABASE_URL.startswith("sqlite")
    else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def save_plan(
    db: Session, plan: StudyPlanResponse, chunks: list[SyllabusChunk]
) -> None:
    record = db.get(StudyPlanRecord, plan.plan_id)
    serialized_plan = plan.model_dump_json()
    serialized_chunks = json.dumps([c.model_dump() for c in chunks])
    if record is None:
        record = StudyPlanRecord(
            plan_id=plan.plan_id,
            created_at=plan.created_at,
            course_code=plan.metadata.course_code,
            course_title=plan.metadata.course_title,
            engine_used=plan.engine_used,
            plan_json=serialized_plan,
            chunks_json=serialized_chunks,
        )
        db.add(record)
    else:
        record.course_code = plan.metadata.course_code
        record.course_title = plan.metadata.course_title
        record.engine_used = plan.engine_used
        record.plan_json = serialized_plan
        record.chunks_json = serialized_chunks
    db.commit()


def get_plan_and_chunks(
    db: Session, plan_id: str
) -> Optional[tuple[StudyPlanResponse, list[SyllabusChunk]]]:
    record = db.get(StudyPlanRecord, plan_id)
    if record is None:
        return None
    plan = StudyPlanResponse.model_validate_json(record.plan_json)
    raw_chunks = json.loads(record.chunks_json)
    chunks = [SyllabusChunk.model_validate(c) for c in raw_chunks]
    return plan, chunks


def list_plans(db: Session) -> list[StudyPlanListItem]:
    records = (
        db.query(StudyPlanRecord)
        .order_by(StudyPlanRecord.created_at.desc())
        .all()
    )
    items: list[StudyPlanListItem] = []
    for rec in records:
        plan = StudyPlanResponse.model_validate_json(rec.plan_json)
        all_tasks = [t for w in plan.weeks for t in w.tasks]
        completed = sum(1 for t in all_tasks if t.is_completed)
        items.append(
            StudyPlanListItem(
                plan_id=plan.plan_id,
                created_at=plan.created_at,
                course_code=plan.metadata.course_code,
                course_title=plan.metadata.course_title,
                total_weeks=plan.constraints.total_weeks,
                weekly_hours=plan.constraints.weekly_hours,
                total_tasks=len(all_tasks),
                completed_tasks=completed,
                engine_used=plan.engine_used,
            )
        )
    return items


def delete_plan(db: Session, plan_id: str) -> bool:
    record = db.get(StudyPlanRecord, plan_id)
    if record is None:
        return False
    db.delete(record)
    db.commit()
    return True
