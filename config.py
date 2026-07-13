"""
Configuration constants and environment variable loading for RAG Document Assistant.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
EMBEDDING_MODEL = "fastembed" # Local fastembed model
LLM_MODEL = "stepfun/step-3.7-flash:free"
LLM_TEMPERATURE = 0.2

# ── Text Chunking ──────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# ── Qdrant Cloud ───────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL")          # e.g. https://xxx.cloud.qdrant.io:6333
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "documents"

# ── Retrieval ──────────────────────────────────────────
RETRIEVAL_K = 4

# ── Upload / Security limits ───────────────────────────
ALLOWED_EXTENSIONS = {".pdf", ".txt"}

MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def validate_config() -> list[str]:
    """Return a list of missing required secrets (empty list means valid)."""
    problems: list[str] = []

    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-your"):
        problems.append("OPENAI_API_KEY is not set (LLM endpoint credentials).")
    if not QDRANT_URL:
        problems.append("QDRANT_URL is not set (vector database endpoint).")
    if not QDRANT_API_KEY:
        problems.append("QDRANT_API_KEY is not set (vector database credentials).")

    return problems
