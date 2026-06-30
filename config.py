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
