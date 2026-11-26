"""Application configuration loaded from environment variables."""

from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    """Centralized runtime settings for SyllabAI."""

    APP_NAME: str = "SyllabAI — Neuro-Symbolic AI Study Planner"
    APP_VERSION: str = "1.0.0"

    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{(BASE_DIR / 'syllabai.db').as_posix()}"
    )

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2").strip()

    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    SAMPLES_DIR: Path = DATA_DIR / "samples"
    MAX_PDF_SIZE_MB: int = 25

    def ensure_dirs(self) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
