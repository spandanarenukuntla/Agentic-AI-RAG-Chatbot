from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(ROOT_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
PDF_PATH = Path(os.getenv("PDF_PATH", str(ROOT_DIR / "data" / "Ebook-Agentic-AI.pdf"))).expanduser()
TEXT_FALLBACK_PATH = ROOT_DIR / "data" / "agentic-ai-reference.txt"
