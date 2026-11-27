"""Multi-modal PDF ingestion, OCR fallback, and token-aware chunking."""

from __future__ import annotations

import re
from typing import Tuple

import pymupdf as fitz
from pypdf import PdfReader

from backend.schemas import SyllabusChunk

_EASYOCR_READER = None
_TIKTOKEN_ENC = None


def _get_tokenizer():
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is not None:
        return _TIKTOKEN_ENC
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        # Trigger tiny encode to verify encoding data is available offline
        enc.encode("test")
        _TIKTOKEN_ENC = enc
    except Exception:
        _TIKTOKEN_ENC = False
    return _TIKTOKEN_ENC


def count_tokens(text: str) -> int:
    enc = _get_tokenizer()
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # Deterministic BPE-approximated fallback (~1.3 tokens per word)
    words = re.findall(r"\S+", text)
    return max(1, int(round(len(words) * 1.3)))


def _get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is not None:
        return _EASYOCR_READER
    try:
        import easyocr

        _EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception:
        _EASYOCR_READER = False
    return _EASYOCR_READER


def _extract_page_text_with_blocks(page: fitz.Page) -> str:
    """Sort layout blocks in reading order to handle multi-column academic syllabi."""
    blocks = page.get_text("blocks") or []
    # Keep text blocks (type == 0) sorted top-to-bottom, left-to-right (quantized y)
    text_blocks = [b for b in blocks if len(b) >= 5 and (len(b) < 7 or b[6] == 0)]
    text_blocks.sort(key=lambda b: (round(b[1] / 14.0) * 14.0, b[0]))
    cleaned = []
    for b in text_blocks:
        txt = str(b[4]).strip()
        if txt:
            cleaned.append(txt)
    if cleaned:
        return "\n".join(cleaned)
    return (page.get_text("text") or "").strip()


def _page_to_pil_rgb(page: fitz.Page, dpi: int = 135):
    """Render PyMuPDF page directly to PIL RGB without PNG encode/decode overhead."""
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _ocr_page_with_easyocr(page: fitz.Page) -> str:
    """Render PDF page via PyMuPDF (no Poppler required) and OCR with EasyOCR preserving lines."""
    reader = _get_easyocr_reader()
    if not reader:
        return ""
    try:
        import numpy as np

        img = _page_to_pil_rgb(page, dpi=150)
        arr = np.array(img)
        results = reader.readtext(arr, detail=1, paragraph=False)
        if not results:
            return ""
        # Group OCR boxes into visual lines by vertical center y
        items = []
        for bbox, text, conf in results:
            txt = str(text).strip()
            if not txt:
                continue
            ys = [pt[1] for pt in bbox]
            xs = [pt[0] for pt in bbox]
            y_mid = sum(ys) / len(ys)
            x_min = min(xs)
            items.append((y_mid, x_min, txt))
        items.sort(key=lambda it: (round(it[0] / 18.0) * 18.0, it[1]))
        lines: list[str] = []
        cur_y = None
        cur_words: list[str] = []
        for y_mid, _, txt in items:
            if cur_y is None or abs(y_mid - cur_y) <= 18.0:
                cur_words.append(txt)
                cur_y = y_mid if cur_y is None else (cur_y + y_mid) / 2.0
            else:
                lines.append(" ".join(cur_words))
                cur_words = [txt]
                cur_y = y_mid
        if cur_words:
            lines.append(" ".join(cur_words))
        return "\n".join(lines)
    except Exception:
        return ""


def _compress_pil_under_limit(img, limit_bytes: int = 750_000) -> bytes:
    """Fast JPEG compression under OCR.space 1MB limit."""
    import io

    img.thumbnail((1250, 1250))
    for max_dim, qual in ((1250, 70), (1050, 50), (850, 35)):
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=qual, optimize=True)
        data = buf.getvalue()
        if len(data) <= limit_bytes:
            return data
    return data


def _ocr_pil_with_cloud_fallback(img, api_key: str | None = None) -> str:
    """
    Zero-RAM OCR fallback for lightweight cloud deployments (e.g. Render 512Mi)
    where local EasyOCR is not installed:
    1) Gemini Vision OCR if GEMINI_API_KEY (or runtime key) is available.
    2) Free OCR.space REST API (compressed JPEG < 880KB).
    """
    try:
        import base64
        import httpx
        from backend.config import settings

        jpeg_bytes = _compress_pil_under_limit(img)

        # 1. Try Gemini Vision if a Gemini key is available
        gem_key = (api_key if api_key and api_key.startswith("AIza") else "") or settings.GEMINI_API_KEY
        if gem_key:
            b64 = base64.b64encode(jpeg_bytes).decode("ascii")
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{settings.GEMINI_MODEL}:generateContent?key={gem_key}"
            )
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": "Transcribe all readable syllabus text from this page verbatim, preserving line breaks."},
                            {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
                        ]
                    }
                ]
            }
            with httpx.Client(timeout=15.0) as client:
                r = client.post(url, json=payload)
                if r.status_code == 200:
                    txt = (
                        r.json()
                        .get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", "")
                    ).strip()
                    if txt:
                        return txt

        # 2. Try OCR.space free tier (single fast 9s call)
        if jpeg_bytes and len(jpeg_bytes) <= 950_000:
            with httpx.Client(timeout=9.0) as client:
                resp = client.post(
                    "https://api.ocr.space/parse/image",
                    data={
                        "apikey": "helloworld",
                        "language": "eng",
                        "isOverlayRequired": "false",
                        "OCREngine": "2",
                    },
                    files={"file": ("page.jpg", jpeg_bytes, "image/jpeg")},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    parsed = data.get("ParsedResults") or []
                    if parsed:
                        text = str(parsed[0].get("ParsedText", "")).strip()
                        if text:
                            return text
    except Exception:
        pass
    return ""


def _detect_section_heading(line: str, current_section: str) -> str:
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return current_section
    heading_patterns = [
        r"^(?:module|unit|week|chapter|part|section|lecture)\s*\d+[:.\-\s].*",
        r"^(?:course\s+(?:description|objectives|outline|schedule|policies)|grading|assessments?|prerequisites|textbooks?|exams?)\b.*",
    ]
    for pat in heading_patterns:
        if re.match(pat, stripped, flags=re.IGNORECASE):
            return stripped[:80]
    if stripped.isupper() and 4 <= len(stripped) <= 60:
        return stripped.title()
    return current_section


def extract_pages_from_pdf(
    pdf_bytes: bytes, api_key: str | None = None
) -> Tuple[list[tuple[int, str]], str]:
    """
    Extract (page_number, text) pairs from PDF bytes using PyMuPDF layout blocks,
    with EasyOCR + Cloud Vision OCR fallback for scanned pages and pypdf backup.
    """
    pages: list[tuple[int, str]] = []
    used_ocr = False

    try:
        from concurrent.futures import ThreadPoolExecutor

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        cloud_ocr_jobs: list[tuple[int, object]] = []

        for idx, page in enumerate(doc, start=1):
            text = _extract_page_text_with_blocks(page)
            if len(re.sub(r"\s+", "", text)) < 35:
                ocr_text = _ocr_page_with_easyocr(page)
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    used_ocr = True
                elif idx <= 3:
                    # Render PIL image on main thread (PyMuPDF doc is not thread-safe)
                    cloud_ocr_jobs.append((idx - 1, _page_to_pil_rgb(page, dpi=115)))
            pages.append((idx, text.strip()))
        doc.close()

        # Run Cloud OCR calls in parallel across scanned pages
        if cloud_ocr_jobs:
            with ThreadPoolExecutor(max_workers=min(4, len(cloud_ocr_jobs))) as pool:
                futures = [
                    (page_pos, pool.submit(_ocr_pil_with_cloud_fallback, pil_img, api_key))
                    for page_pos, pil_img in cloud_ocr_jobs
                ]
                for page_pos, fut in futures:
                    ocr_res = fut.result()
                    if len(ocr_res.strip()) > len(pages[page_pos][1].strip()):
                        pages[page_pos] = (page_pos + 1, ocr_res.strip())
                        used_ocr = True

        method = "pymupdf_plus_ocr" if used_ocr else "pymupdf_native"
        if any(txt for _, txt in pages):
            return pages, method
    except Exception:
        pass

    # Secondary fallback via pypdf
    import io

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        txt = (page.extract_text() or "").strip()
        pages.append((idx, txt))
    return pages, "pypdf_fallback"


def chunk_pages(
    pages: list[tuple[int, str]], target_tokens: int = 240, overlap_tokens: int = 40
) -> list[SyllabusChunk]:
    """Segment extracted pages into overlapping section-aware chunks for RAG."""
    chunks: list[SyllabusChunk] = []
    current_section = "Course Overview"
    chunk_idx = 0

    for page_num, page_text in pages:
        if not page_text.strip():
            continue
        lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
        buffer_lines: list[str] = []
        buffer_tokens = 0
        buffer_section = current_section

        for ln in lines:
            detected = _detect_section_heading(ln, current_section)
            if detected != current_section and buffer_lines and buffer_tokens > 60:
                text_block = "\n".join(buffer_lines)
                chunk_idx += 1
                chunks.append(
                    SyllabusChunk(
                        chunk_id=f"chunk_{chunk_idx:03d}",
                        page_number=page_num,
                        section_title=buffer_section,
                        text=text_block,
                        token_count=count_tokens(text_block),
                    )
                )
                # Keep tail overlap
                overlap_lines = buffer_lines[-2:] if len(buffer_lines) >= 2 else []
                buffer_lines = list(overlap_lines)
                buffer_tokens = count_tokens("\n".join(buffer_lines)) if buffer_lines else 0
                buffer_section = detected
            current_section = detected
            if not buffer_lines:
                buffer_section = current_section

            ln_tokens = count_tokens(ln)
            buffer_lines.append(ln)
            buffer_tokens += ln_tokens

            if buffer_tokens >= target_tokens:
                text_block = "\n".join(buffer_lines)
                chunk_idx += 1
                chunks.append(
                    SyllabusChunk(
                        chunk_id=f"chunk_{chunk_idx:03d}",
                        page_number=page_num,
                        section_title=buffer_section,
                        text=text_block,
                        token_count=count_tokens(text_block),
                    )
                )
                # Build overlap window from trailing lines
                tail: list[str] = []
                tail_toks = 0
                for prev in reversed(buffer_lines):
                    t = count_tokens(prev)
                    if tail_toks + t > overlap_tokens and tail:
                        break
                    tail.insert(0, prev)
                    tail_toks += t
                buffer_lines = tail
                buffer_tokens = tail_toks
                buffer_section = current_section

        if buffer_lines:
            text_block = "\n".join(buffer_lines)
            chunk_idx += 1
            chunks.append(
                SyllabusChunk(
                    chunk_id=f"chunk_{chunk_idx:03d}",
                    page_number=page_num,
                    section_title=buffer_section,
                    text=text_block,
                    token_count=count_tokens(text_block),
                )
            )

    return chunks


def pages_from_raw_text(raw_text: str) -> list[tuple[int, str]]:
    """Convert raw pasted syllabus text into pseudo-pages for uniform processing."""
    cleaned = raw_text.strip()
    if not cleaned:
        return []
    paragraphs = re.split(r"\n\s*\n", cleaned)
    pages: list[tuple[int, str]] = []
    buf: list[str] = []
    char_count = 0
    page_num = 1
    for p in paragraphs:
        buf.append(p.strip())
        char_count += len(p)
        if char_count >= 1800:
            pages.append((page_num, "\n\n".join(buf)))
            page_num += 1
            buf = []
            char_count = 0
    if buf:
        pages.append((page_num, "\n\n".join(buf)))
    return pages
