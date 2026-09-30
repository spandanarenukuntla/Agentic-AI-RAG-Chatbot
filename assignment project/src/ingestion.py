from __future__ import annotations

import re
from pathlib import Path
from typing import List

from src.config import PDF_PATH, PINECONE_API_KEY, PINECONE_INDEX_NAME, TEXT_FALLBACK_PATH


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def load_document_text(pdf_path: str | Path | None = None) -> str:
    """Read a PDF if it exists; otherwise fall back to a text file."""
    target = Path(pdf_path) if pdf_path else PDF_PATH

    if target.exists() and target.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(target))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)
        except Exception:
            return ""

    if target.exists() and target.is_file():
        try:
            return target.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    if TEXT_FALLBACK_PATH.exists():
        try:
            return TEXT_FALLBACK_PATH.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    return ""


def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    if not text:
        return []

    cleaned = normalize_text(text)
    if not cleaned:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    chunks: List[str] = []
    current: List[str] = []
    current_length = 0

    for sentence in sentences:
        if not sentence:
            continue
        sentence_length = len(sentence)
        if current and current_length + sentence_length > chunk_size:
            joined = " ".join(current).strip()
            if joined:
                chunks.append(joined)
            current = [sentence]
            current_length = sentence_length
        else:
            current.append(sentence)
            current_length += sentence_length

    if current:
        joined = " ".join(current).strip()
        if joined:
            chunks.append(joined)

    final_chunks: List[str] = []
    for index, chunk in enumerate(chunks):
        if not chunk:
            continue
        if index == 0:
            final_chunks.append(chunk)
            continue
        if final_chunks and len(final_chunks[-1]) + len(chunk) <= chunk_size + chunk_overlap:
            final_chunks[-1] = f"{final_chunks[-1]} {chunk}"
        else:
            final_chunks.append(chunk)

    return final_chunks


def compute_chunk_score(question: str, chunk: str) -> float:
    q_terms = set(re.findall(r"[a-zA-Z0-9]+", question.lower()))
    c_terms = set(re.findall(r"[a-zA-Z0-9]+", chunk.lower()))
    if not q_terms or not c_terms:
        return 0.0

    overlap = len(q_terms & c_terms)
    return (overlap / len(q_terms)) + (overlap / max(len(c_terms), 1))


def retrieve_relevant_chunks(question: str, text: str, top_k: int = 3) -> List[str]:
    if not question or not text:
        return []

    chunks = split_text_into_chunks(text)
    ranked_chunks = sorted(
        chunks,
        key=lambda chunk: compute_chunk_score(question, chunk),
        reverse=True,
    )
    return ranked_chunks[:top_k]


def run_ingestion(pdf_path: str | Path | None = None, index_name: str | None = None):
    """Index a local PDF into Pinecone if credentials are available."""
    target = Path(pdf_path) if pdf_path else PDF_PATH
    if not target.exists() or not PINECONE_API_KEY:
        return None

    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_openai import OpenAIEmbeddings
        from langchain_pinecone import PineconeVectorStore
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        return None

    index = index_name or PINECONE_INDEX_NAME
    try:
        loader = PyPDFLoader(str(target))
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=index,
        )
        return vector_store
    except Exception:
        return None
