"""Structural NLP + Hybrid LLM Syllabus Parser for metadata, modules, topics & assessments."""

from __future__ import annotations

import re
from typing import Optional, Tuple

from backend.ai.llm_providers import call_llm_json, resolve_effective_engine
from backend.schemas import (
    CourseMetadata,
    ExtractedAssessment,
    StudyEngineMode,
    TopicNode,
)

# Matches UNIT I:, UNIT - II, Section 1:, Module 3 -, Week 4 |, plus common OCR variants (UNIT MI:, UNIT M:)
UNIT_TOKEN_RE = re.compile(
    r"(?:^|(?<=\s))(?:module|unit|week|chapter|part|section|topic)\s*(?:[-–—:]\s*)?"
    r"(\d{1,2}|VIII|VII|VI|IV|V|III|II|I|IX|X|MI|M)\b(?:\s*[:.\-\|\u2013\u2014]\s*|\s*$)",
    re.IGNORECASE,
)

MODULE_LINE_RE = re.compile(
    r"^(?:module|unit|week|chapter|part|section|topic)\s*(?:[-–—:]\s*)?"
    r"(\d{1,2}|VIII|VII|VI|IV|V|III|II|I|IX|X|MI|M)\b\s*[:.\-\|\u2013\u2014]?\s*(.*)$",
    re.IGNORECASE,
)

NUMBERED_TOPIC_RE = re.compile(r"^(\d{1,2})\.\s+([A-Z][^\n]{4,95})$")
BULLET_RE = re.compile(r"^(?:[-*\u2022\u25cf\u25e6]|\d+[.)]|[a-z][.)])\s+(.+)$")
COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,5}[-\s]?\d{3,4}[A-Z]?)\b")
NUMERIC_CODE_RE = re.compile(r"\((\d{6,9})(?:\s*/\s*\d{6,9})*\)")
CREDITS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:credit\s*hours?|credits?|units?)\b", re.IGNORECASE)
INSTRUCTOR_RE = re.compile(
    r"(?:instructor|professor|lecturer|faculty|taught by)\s*[:\-]\s*([^\n,|]{3,60})",
    re.IGNORECASE,
)
ASSESSMENT_RE = re.compile(
    r"\b(midterm|mid-?sem(?:ester)?|end-?sem(?:ester)?|final\s*exam|quiz\s*\d*|assignment\s*\d*|homework\s*\d*|lab\s*\d*|project|capstone)\b",
    re.IGNORECASE,
)
WEIGHT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
WEEK_REF_RE = re.compile(r"\bweek\s*(\d{1,2})\b", re.IGNORECASE)

STOP_HEADINGS = (
    "text & reference",
    "text and reference",
    "textbooks",
    "reference books",
    "course outcomes",
    "grading",
    "evaluation",
    "academic integrity",
    "attendance",
    "office hours",
    "policies",
    "disability",
    "plagiarism",
)

ROMAN_MAP = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "MI": 2,  # Common EasyOCR misread of "II"
    "M": 3,   # Common EasyOCR misread of "III"
}


def _parse_unit_number(raw: str, prev_num: int) -> int:
    up = raw.strip().upper()
    if up.isdigit():
        return int(up)
    if up in ROMAN_MAP:
        val = ROMAN_MAP[up]
        # Disambiguate OCR "MI"/"M" using sequence context
        if up in ("MI", "M") and prev_num >= 1:
            return prev_num + 1
        return val
    return prev_num + 1


def _normalize_syllabus_lines(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """
    Break inline unit/section headers and policy headings onto their own lines
    so both single-line OCR dumps and multi-line native PDFs parse uniformly.
    """
    normalized: list[tuple[int, str]] = []
    inline_breaks = re.compile(
        r"(?<!^)\s+(?=(?:UNIT|MODULE|SECTION|CHAPTER)\s*(?:[-–—:]\s*)?(?:\d{1,2}|VIII|VII|VI|IV|V|III|II|I|IX|X|MI|M)\s*[:.\-\|\u2013\u2014])",
        re.IGNORECASE,
    )
    section_breaks = re.compile(
        r"\s+(?=(?:COURSE OBJECTIVES|COURSE OUTCOMES|TEXT\s*(?:&|AND)\s*REFERENCE BOOKS|REFERENCE BOOKS|GRADING|EVALUATION SCHEME)\b)",
        re.IGNORECASE,
    )

    for page_num, text in pages:
        expanded = inline_breaks.sub("\n", text)
        expanded = section_breaks.sub("\n", expanded)
        for ln in expanded.splitlines():
            clean = ln.strip()
            if clean:
                normalized.append((page_num, clean))
    return normalized


def _split_prose_subtopics(text: str) -> list[str]:
    """Split comma/semicolon-separated syllabus topic lists without breaking inside parentheses."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch in (",", ";") and depth == 0:
            item = "".join(buf).strip(" .:-")
            if len(item) >= 3:
                parts.append(item)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip(" .:-")
    if len(tail) >= 3:
        parts.append(tail)
    return parts


def _extract_metadata(
    pages: list[tuple[int, str]],
    extraction_method: str,
    title_hint: Optional[str] = None,
) -> CourseMetadata:
    first_text = "\n".join(txt for _, txt in pages[:2])
    norm_lines = _normalize_syllabus_lines(pages[:2])
    lines = [ln for _, ln in norm_lines]

    code_match = COURSE_CODE_RE.search(first_text)
    num_code_match = NUMERIC_CODE_RE.search(first_text)
    if code_match:
        course_code = code_match.group(1).replace(" ", "")
    elif num_code_match:
        course_code = f"CS-{num_code_match.group(1)}"
    else:
        course_code = "SYL-101"

    course_title = title_hint.strip() if title_hint and title_hint.strip() else ""
    if not course_title:
        # Match explicit ALL-CAPS subject title right before numeric course code e.g. "EVOLUTIONARY ALGORITHMS AND OPTIMIZATION (31243103)"
        caps_code_m = re.search(
            r"\b([A-Z][A-Z0-9\s&,\-]{6,80}?)\s*\(\d{5,9}",
            first_text,
        )
        if caps_code_m:
            candidate = caps_code_m.group(1).strip(" :-|")
            # Strip leading "Centre for X" or "Department of X" if captured
            candidate = re.sub(
                r"^(?:centre\s+for\s+[a-z\s]+|department\s+of\s+[a-z\s]+|naac[^\n]*grade)\s+",
                "",
                candidate,
                flags=re.IGNORECASE,
            ).strip()
            if len(candidate) >= 5 and "university" not in candidate.lower():
                course_title = candidate

    if not course_title:
        skip_words = (
            "university",
            "institute",
            "deemed",
            "ministry",
            "government",
            "naac",
            "accredited",
            "department of",
            "centre for",
            "syllabus",
            "semester",
            "instructor",
            "email",
            "office hours",
            "course objectives",
            "course outcomes",
        )
        for ln in lines[:12]:
            if len(ln) < 5 or len(ln) > 105:
                continue
            low = ln.lower()
            if MODULE_LINE_RE.match(ln):
                break
            if any(skip in low for skip in skip_words):
                if code_match and code_match.group(1) in ln and ":" in ln:
                    course_title = ln.split(":", 1)[1].strip()
                    break
                continue
            if code_match and code_match.group(1) in ln:
                cleaned = ln.replace(code_match.group(1), "").lstrip(":-\u2013\u2014 ").strip()
                if len(cleaned) >= 4:
                    course_title = cleaned
                    break
            course_title = ln
            break

    if not course_title:
        course_title = "Academic Course Syllabus"

    instr_match = INSTRUCTOR_RE.search(first_text)
    instructor = instr_match.group(1).strip() if instr_match else None

    cred_match = CREDITS_RE.search(first_text)
    credits = float(cred_match.group(1)) if cred_match else 3.0

    week_nums = [int(m.group(1)) for _, txt in pages for m in WEEK_REF_RE.finditer(txt)]
    semester_weeks = max([w for w in week_nums if 4 <= w <= 24], default=12)

    return CourseMetadata(
        course_code=course_code,
        course_title=course_title[:120],
        instructor=instructor,
        credits=credits,
        semester_weeks=semester_weeks,
        total_pages=max(1, len(pages)),
        extraction_method=extraction_method,
    )


def _extract_assessments(pages: list[tuple[int, str]], total_weeks: int) -> list[ExtractedAssessment]:
    assessments: list[ExtractedAssessment] = []
    seen_titles: set[str] = set()
    full_lines = _normalize_syllabus_lines(pages)

    for _, ln in full_lines:
        m = ASSESSMENT_RE.search(ln)
        if not m or len(ln) > 180:
            continue
        raw_kind = m.group(1).lower()
        if "midterm" in raw_kind or "mid-sem" in raw_kind or "midsem" in raw_kind:
            atype = "midterm"
        elif "final" in raw_kind or "end-sem" in raw_kind or "endsem" in raw_kind:
            atype = "final"
        elif "quiz" in raw_kind:
            atype = "quiz"
        elif "project" in raw_kind or "capstone" in raw_kind:
            atype = "project"
        else:
            atype = "assignment"

        weight_m = WEIGHT_RE.search(ln)
        weight = float(weight_m.group(1)) if weight_m else None
        if weight is not None and weight > 100:
            weight = None

        wk_m = WEEK_REF_RE.search(ln)
        target_wk = int(wk_m.group(1)) if wk_m else None
        if target_wk is None:
            if atype == "midterm":
                target_wk = max(2, total_weeks // 2)
            elif atype == "final":
                target_wk = total_weeks
            elif atype == "project":
                target_wk = max(2, total_weeks - 1)

        if target_wk is not None:
            target_wk = max(1, min(total_weeks, target_wk))

        title = re.sub(r"\s+", " ", ln).strip(" -:\u2022*")
        if len(title) > 68:
            title = m.group(1).title()
        key = title.lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)

        assessments.append(
            ExtractedAssessment(
                id=f"assess_{len(assessments) + 1:02d}",
                title=title,
                assessment_type=atype,
                target_week=target_wk,
                weight_percent=weight,
                topics_covered=[],
                description=ln[:200],
            )
        )
        if len(assessments) >= 12:
            break

    if not assessments:
        assessments = [
            ExtractedAssessment(
                id="assess_01",
                title="Midterm Checkpoint Exam",
                assessment_type="midterm",
                target_week=max(2, total_weeks // 2),
                weight_percent=30.0,
                description="Cumulative midpoint mastery check",
            ),
            ExtractedAssessment(
                id="assess_02",
                title="Final Comprehensive Assessment",
                assessment_type="final",
                target_week=total_weeks,
                weight_percent=40.0,
                description="End-of-course comprehensive exam / synthesis",
            ),
        ]
    return assessments


def _extract_keywords(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][a-zA-Z0-9\-]{3,}", text.lower())
    stop = {
        "with", "from", "that", "this", "into", "over", "using", "based", "introduction",
        "module", "week", "unit", "chapter", "topic", "course", "study", "hours", "topics",
        "overview", "part", "section", "and", "the", "for", "your",
    }
    uniq: list[str] = []
    for w in words:
        if w not in stop and w not in uniq:
            uniq.append(w)
        if len(uniq) >= 8:
            break
    return uniq


def _parse_topics_structural(pages: list[tuple[int, str]]) -> list[TopicNode]:
    """Extract structured units/modules/sections and subtopics from syllabus lines."""
    norm_lines = _normalize_syllabus_lines(pages)
    topics: list[TopicNode] = []
    current_mod_num = 0
    current_topic: Optional[dict] = None
    in_policy_section = False
    expecting_title_for_current = False

    def flush_topic():
        nonlocal current_topic, expecting_title_for_current
        expecting_title_for_current = False
        if not current_topic:
            return
        title = current_topic["title"].strip(" :-–—|")
        if len(title) < 2:
            current_topic = None
            return

        raw_subs: list[str] = []
        for s in current_topic["subtopics"]:
            s_clean = s.strip()
            if len(s_clean) >= 3:
                if len(s_clean) > 75 and ("," in s_clean or ";" in s_clean):
                    raw_subs.extend(_split_prose_subtopics(s_clean))
                else:
                    raw_subs.append(s_clean)

        # Deduplicate while preserving order
        subtopics: list[str] = []
        seen_sub: set[str] = set()
        for s in raw_subs:
            low_s = s.lower()
            if low_s not in seen_sub:
                seen_sub.add(low_s)
                subtopics.append(s[:120])

        # If title itself contains a colon followed by a long topic list (e.g. from OCR inline unit)
        if ":" in title:
            left, right = title.split(":", 1)
            if len(left.strip()) >= 4 and ("," in right or ";" in right):
                title = left.strip()
                subtopics = _split_prose_subtopics(right)[:10] + subtopics

        # If title is very long (OCR merged Unit title + first few topics), split cleanly
        if len(title) > 95 and not subtopics:
            split_items = _split_prose_subtopics(title)
            if len(split_items) >= 2:
                title = split_items[0][:90]
                subtopics = split_items[1:10]

        tid = f"topic_{len(topics) + 1:02d}"
        kws = _extract_keywords(f"{title} {' '.join(subtopics)}")
        topics.append(
            TopicNode(
                id=tid,
                module_number=current_topic["module_number"],
                module_title=title[:80],
                title=title[:110],
                subtopics=subtopics[:10],
                page_refs=sorted(set(current_topic["pages"])),
                keywords=kws,
            )
        )
        current_topic = None

    idx = 0
    while idx < len(norm_lines):
        page_num, ln = norm_lines[idx]
        low = ln.lower()

        if any(low.startswith(h) for h in STOP_HEADINGS):
            flush_topic()
            in_policy_section = True
            idx += 1
            continue

        mod_m = MODULE_LINE_RE.match(ln)
        if mod_m:
            in_policy_section = False
            flush_topic()
            mod_num = _parse_unit_number(mod_m.group(1), current_mod_num)
            mod_rest = mod_m.group(2).strip(" :-–—")
            current_mod_num = mod_num

            if not mod_rest and idx + 1 < len(norm_lines):
                # Multi-line header: "Section 1:" followed on next line by "Probability and Statistics"
                next_pg, next_ln = norm_lines[idx + 1]
                if len(next_ln) <= 95 and not MODULE_LINE_RE.match(next_ln):
                    mod_rest = next_ln.strip(" :-–—")
                    page_num = next_pg
                    idx += 1

            # Extract concise title if OCR jammed title + body together (e.g. "Foundation of Optimization (EA) Introduction to...")
            title_part = mod_rest
            extra_sub_text = ""
            if len(mod_rest) > 85:
                # Try splitting at first "Introduction" or comma list
                m_split = re.search(r"\s+(?=Introduction\b|Principles\b|Overview\b)", mod_rest)
                if m_split and m_split.start() >= 12:
                    title_part = mod_rest[: m_split.start()].strip()
                    extra_sub_text = mod_rest[m_split.start() :].strip()
                elif "," in mod_rest:
                    first_c = mod_rest.find(",")
                    if 12 <= first_c <= 80:
                        title_part = mod_rest[:first_c].strip()
                        extra_sub_text = mod_rest[first_c + 1 :].strip()

            current_topic = {
                "module_number": mod_num,
                "title": title_part or f"Unit {mod_num}",
                "subtopics": _split_prose_subtopics(extra_sub_text) if extra_sub_text else [],
                "pages": [page_num],
            }
            idx += 1
            continue

        if in_policy_section:
            if "syllabus" in low or "course schedule" in low or "course content" in low:
                in_policy_section = False
            else:
                idx += 1
                continue

        num_m = NUMBERED_TOPIC_RE.match(ln)
        if num_m and not any(w in low for w in ("points", "attendance", "late policy", "academic honesty")):
            flush_topic()
            idx_num = int(num_m.group(1))
            title_txt = num_m.group(2).strip()
            current_mod_num = idx_num
            current_topic = {
                "module_number": idx_num,
                "title": title_txt,
                "subtopics": [],
                "pages": [page_num],
            }
            idx += 1
            continue

        bul_m = BULLET_RE.match(ln)
        if bul_m and current_topic is not None:
            sub = bul_m.group(1).strip()
            if len(sub) < 180 and not ASSESSMENT_RE.search(sub.lower()[:20]):
                if ";" in sub:
                    current_topic["subtopics"].extend(x.strip() for x in sub.split(";") if x.strip())
                else:
                    current_topic["subtopics"].append(sub)
                if page_num not in current_topic["pages"]:
                    current_topic["pages"].append(page_num)
            idx += 1
            continue

        # Pipe-separated schedule row
        if "|" in ln and len(ln.split("|")) >= 2:
            parts = [p.strip() for p in ln.split("|") if p.strip()]
            wk_m = WEEK_REF_RE.search(parts[0]) if parts else None
            if wk_m:
                flush_topic()
                m_num = int(wk_m.group(1))
                topic_col = parts[1] if len(parts) > 1 else parts[0]
                subs = [p for p in parts[2:] if len(p) > 2]
                current_mod_num = m_num
                current_topic = {
                    "module_number": m_num,
                    "title": topic_col,
                    "subtopics": subs,
                    "pages": [page_num],
                }
                idx += 1
                continue

        # Prose lines belonging to the active Unit / Section
        if current_topic is not None:
            if page_num not in current_topic["pages"]:
                current_topic["pages"].append(page_num)
            if "," in ln or ";" in ln:
                current_topic["subtopics"].extend(_split_prose_subtopics(ln))
            elif len(ln) >= 6:
                current_topic["subtopics"].append(ln)

        idx += 1

    flush_topic()

    if len(topics) < 2:
        topics = _fallback_paragraph_segmentation(pages)

    # Split overly dense units ONLY if total topics < 5
    if len(topics) < 5:
        expanded: list[TopicNode] = []
        for t in topics:
            if len(t.subtopics) >= 6:
                mid = len(t.subtopics) // 2
                part1 = t.subtopics[:mid]
                part2 = t.subtopics[mid:]
                expanded.append(
                    TopicNode(
                        id=f"topic_{len(expanded) + 1:02d}",
                        module_number=t.module_number,
                        module_title=t.module_title,
                        title=f"{t.title} — Part I",
                        subtopics=part1,
                        page_refs=t.page_refs,
                        keywords=_extract_keywords(f"{t.title} {' '.join(part1)}"),
                    )
                )
                expanded.append(
                    TopicNode(
                        id=f"topic_{len(expanded) + 1:02d}",
                        module_number=t.module_number,
                        module_title=t.module_title,
                        title=f"{t.title} — Part II",
                        subtopics=part2,
                        page_refs=t.page_refs,
                        keywords=_extract_keywords(f"{t.title} {' '.join(part2)}"),
                    )
                )
            else:
                t.id = f"topic_{len(expanded) + 1:02d}"
                expanded.append(t)
        topics = expanded

    return topics[:32]


def _fallback_paragraph_segmentation(pages: list[tuple[int, str]]) -> list[TopicNode]:
    """Segment prose-only syllabi into coherent study topics via sentence/line heuristics."""
    topics: list[TopicNode] = []
    candidate_lines: list[tuple[int, str]] = []
    for pg, txt in pages:
        for ln in txt.splitlines():
            clean = ln.strip(" -\u2022*\t")
            if 12 <= len(clean) <= 130 and not any(s in clean.lower() for s in STOP_HEADINGS):
                candidate_lines.append((pg, clean))

    if not candidate_lines:
        return [
            TopicNode(
                id="topic_01",
                module_number=1,
                module_title="Core Course Study",
                title="Core Syllabus Foundations & Review",
                subtopics=["Key terminology", "Problem solving", "Exam synthesis"],
                page_refs=[1],
                keywords=["syllabus", "foundations", "review"],
            )
        ]

    batch_size = max(2, min(4, len(candidate_lines) // 5 or 2))
    mod_idx = 1
    for i in range(0, len(candidate_lines), batch_size):
        chunk = candidate_lines[i : i + batch_size]
        head_pg, head_txt = chunk[0]
        subs = [txt for _, txt in chunk[1:]]
        pgs = sorted({pg for pg, _ in chunk})
        topics.append(
            TopicNode(
                id=f"topic_{len(topics) + 1:02d}",
                module_number=mod_idx,
                module_title=head_txt[:60],
                title=head_txt[:100],
                subtopics=subs,
                page_refs=pgs,
                keywords=_extract_keywords(f"{head_txt} {' '.join(subs)}"),
            )
        )
        mod_idx += 1
        if len(topics) >= 16:
            break
    return topics


def parse_syllabus(
    pages: list[tuple[int, str]],
    extraction_method: str,
    total_weeks: int,
    engine: StudyEngineMode = "local_hybrid",
    api_key: Optional[str] = None,
    title_hint: Optional[str] = None,
) -> Tuple[CourseMetadata, list[TopicNode], list[ExtractedAssessment], str]:
    """
    Hybrid Parser Agent:
    1. Runs universal structural NLP parser as ground-truth anchor.
    2. If an LLM engine (OpenAI/Gemini/Ollama) is configured, refines topics & metadata via structured JSON.
    """
    metadata = _extract_metadata(pages, extraction_method, title_hint)
    topics = _parse_topics_structural(pages)
    assessments = _extract_assessments(pages, total_weeks)

    effective, key, status = resolve_effective_engine(engine, api_key)
    if effective == "local_hybrid":
        return metadata, topics, assessments, status

    joined = "\n\n".join(f"[Page {pg}]\n{txt}" for pg, txt in pages)[:14000]
    system_prompt = (
        "You are an expert Academic Curriculum Parser Agent. "
        "Extract course metadata, ordered study modules/topics, and assessments as strict JSON."
    )
    user_prompt = f"""Parse the following syllabus into JSON with this exact schema:
{{
  "course_code": "string",
  "course_title": "string",
  "instructor": "string or null",
  "credits": 3.0,
  "topics": [
    {{
      "module_number": 1,
      "module_title": "string",
      "title": "string",
      "subtopics": ["string"],
      "page_refs": [1]
    }}
  ],
  "assessments": [
    {{
      "title": "string",
      "assessment_type": "quiz|midterm|final|assignment|project",
      "target_week": 6,
      "weight_percent": 25.0,
      "description": "string"
    }}
  ]
}}

Syllabus text:
{joined}
"""
    llm_data = call_llm_json(system_prompt, user_prompt, effective, key)
    if not llm_data or not isinstance(llm_data.get("topics"), list) or not llm_data["topics"]:
        return metadata, topics, assessments, "fallback"

    try:
        if llm_data.get("course_title"):
            metadata.course_title = str(llm_data["course_title"])[:120]
        if llm_data.get("course_code"):
            metadata.course_code = str(llm_data["course_code"])[:30]
        if llm_data.get("instructor"):
            metadata.instructor = str(llm_data["instructor"])[:80]
        if llm_data.get("credits") is not None:
            metadata.credits = float(llm_data["credits"])

        parsed_topics: list[TopicNode] = []
        for idx, raw_t in enumerate(llm_data["topics"][:28], start=1):
            t_title = str(raw_t.get("title", "")).strip()
            if not t_title:
                continue
            subs = [str(s).strip() for s in raw_t.get("subtopics", []) if str(s).strip()][:10]
            pg_refs = [int(p) for p in raw_t.get("page_refs", [1]) if isinstance(p, (int, float))] or [1]
            parsed_topics.append(
                TopicNode(
                    id=f"topic_{idx:02d}",
                    module_number=int(raw_t.get("module_number", idx)),
                    module_title=str(raw_t.get("module_title", t_title))[:80],
                    title=t_title[:110],
                    subtopics=subs,
                    page_refs=pg_refs,
                    keywords=_extract_keywords(f"{t_title} {' '.join(subs)}"),
                )
            )
        if parsed_topics:
            topics = parsed_topics

        if isinstance(llm_data.get("assessments"), list) and llm_data["assessments"]:
            parsed_assess: list[ExtractedAssessment] = []
            for idx, a in enumerate(llm_data["assessments"][:12], start=1):
                atype = str(a.get("assessment_type", "assignment")).lower()
                if atype not in {"quiz", "midterm", "final", "assignment", "project"}:
                    atype = "assignment"
                wk = a.get("target_week")
                wk_int = max(1, min(total_weeks, int(wk))) if isinstance(wk, (int, float)) else None
                parsed_assess.append(
                    ExtractedAssessment(
                        id=f"assess_{idx:02d}",
                        title=str(a.get("title", f"Assessment {idx}"))[:70],
                        assessment_type=atype,  # type: ignore[arg-type]
                        target_week=wk_int,
                        weight_percent=float(a["weight_percent"]) if isinstance(a.get("weight_percent"), (int, float)) else None,
                        description=str(a.get("description", ""))[:200],
                    )
                )
            if parsed_assess:
                assessments = parsed_assess
        return metadata, topics, assessments, "success"
    except Exception:
        return metadata, topics, assessments, "fallback"
