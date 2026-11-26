"""Unified LLM provider adapter (OpenAI, Gemini, Ollama) with zero-key local fallback."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import httpx

from backend.config import settings
from backend.schemas import StudyEngineMode


def resolve_effective_engine(
    requested: StudyEngineMode, api_key_override: Optional[str] = None
) -> tuple[StudyEngineMode, Optional[str], str]:
    """Determine which engine can actually execute given available credentials."""
    if requested == "local_hybrid":
        return "local_hybrid", None, "success"

    if requested == "openai":
        key = (api_key_override or settings.OPENAI_API_KEY).strip()
        if key:
            return "openai", key, "success"
        return "local_hybrid", None, "fallback"

    if requested == "gemini":
        key = (api_key_override or settings.GEMINI_API_KEY).strip()
        if key:
            return "gemini", key, "success"
        return "local_hybrid", None, "fallback"

    if requested == "ollama":
        return "ollama", None, "success"

    return "local_hybrid", None, "fallback"


def _strip_json_fences(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    engine: StudyEngineMode,
    api_key: Optional[str] = None,
    timeout: float = 25.0,
) -> Optional[dict[str, Any]]:
    """Call external LLM provider expecting structured JSON; return None on failure."""
    effective, key, _ = resolve_effective_engine(engine, api_key)
    if effective == "local_hybrid":
        return None

    try:
        if effective == "openai" and key:
            from openai import OpenAI

            client = OpenAI(api_key=key, timeout=timeout)
            resp = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = resp.choices[0].message.content or ""
            return json.loads(_strip_json_fences(text))

        if effective == "gemini" and key:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{settings.GEMINI_MODEL}:generateContent?key={key}"
            )
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}],
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "responseMimeType": "application/json",
                },
            }
            with httpx.Client(timeout=timeout) as client:
                r = client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            return json.loads(_strip_json_fences(text))

        if effective == "ollama":
            url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
            payload = {
                "model": settings.OLLAMA_MODEL,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            with httpx.Client(timeout=timeout) as client:
                r = client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
            text = data.get("message", {}).get("content", "")
            return json.loads(_strip_json_fences(text))
    except Exception:
        return None
    return None


def call_llm_text(
    system_prompt: str,
    user_prompt: str,
    engine: StudyEngineMode,
    api_key: Optional[str] = None,
    timeout: float = 25.0,
) -> Optional[str]:
    """Call external LLM provider for grounded natural-language text generation."""
    effective, key, _ = resolve_effective_engine(engine, api_key)
    if effective == "local_hybrid":
        return None

    try:
        if effective == "openai" and key:
            from openai import OpenAI

            client = OpenAI(api_key=key, timeout=timeout)
            resp = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return (resp.choices[0].message.content or "").strip()

        if effective == "gemini" and key:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{settings.GEMINI_MODEL}:generateContent?key={key}"
            )
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}],
                    }
                ],
                "generationConfig": {"temperature": 0.2},
            }
            with httpx.Client(timeout=timeout) as client:
                r = client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
            return (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )

        if effective == "ollama":
            url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
            payload = {
                "model": settings.OLLAMA_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            with httpx.Client(timeout=timeout) as client:
                r = client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
            return str(data.get("message", {}).get("content", "")).strip()
    except Exception:
        return None
    return None
