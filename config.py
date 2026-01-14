"""
Centralized configuration for the Universal Bank Statement Parser.
Editable via environment variables. Uses safe defaults for production.
"""

import os
from typing import Optional
from pathlib import Path

# optional dotenv support for local development
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# Load .env from repo root if present (local dev only)
if load_dotenv:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


class Config:
    """
    Application configuration. Values are read from environment variables.
    Keep secrets (API keys) out of the repository; set them as env vars.
    """

    # --- API Settings ---
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Universal Bank Statement Parser")
    VERSION: str = os.getenv("VERSION", "2.0")
    DESCRIPTION: str = os.getenv(
        "DESCRIPTION", "Hybrid extraction + AI cleanup using Groq LLaMA models."
    )

    # --- PDF Constraints ---
    MAX_PDF_SIZE_MB: float = float(os.getenv("MAX_PDF_SIZE_MB", "25"))
    OCR_MAX_PAGES: int = int(os.getenv("OCR_MAX_PAGES", "200"))

    # --- AI Settings ---
    # NOTE: Do NOT hardcode API keys. Set GROQ_API_KEY in the environment.
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    # Allow overriding the Groq endpoint and model via env for flexibility
    GROQ_URL: str = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "groq/compound")
    # Maximum characters to send to the AI in the assembled prompt
    AI_MAX_PAYLOAD_CHARS: int = int(os.getenv("AI_MAX_PAYLOAD_CHARS", "15000"))

    # Rate limiting config (slowapi style, e.g. "4/minute")
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() in ("1", "true", "yes")
    RATE_LIMIT: str = os.getenv("RATE_LIMIT", "4/minute")
    AI_RETRY_COUNT: int = int(os.getenv("AI_RETRY_COUNT", "1"))  # EMERGENCY: Reduced from 3 to 1 to prevent token exhaustion
    AI_TIMEOUT_SECONDS: int = int(os.getenv("AI_TIMEOUT_SECONDS", "45"))

    # If false (default), the server will refuse requests whose assembled
    # AI payload exceeds `AI_MAX_PAYLOAD_CHARS` and return a simple 413 error
    # instructing users to upload a smaller/resampled PDF. Set to true to
    # allow automatic truncation logic to proceed (not recommended for large PDFs).
    ALLOW_AI_PAYLOAD_TRUNCATION: bool = os.getenv("ALLOW_AI_PAYLOAD_TRUNCATION", "false").lower() in ("1", "true", "yes")

    # --- Logging ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


config = Config()
