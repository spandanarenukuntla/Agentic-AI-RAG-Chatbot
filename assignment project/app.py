from __future__ import annotations

import os
from typing import Any, Dict

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    FastAPI = None
    BaseModel = object  # type: ignore[misc]

from src.graph import build_rag_graph


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    final_answer: str
    retrieved_context: list[str]
    confidence_score: float


if FastAPI is not None:
    app = FastAPI(title="Agentic AI RAG API")

    graph = build_rag_graph(pdf_path=os.getenv("PDF_PATH"))

    @app.get("/health")
    async def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", response_model=QueryResponse)
    async def chat_endpoint(request: QueryRequest) -> QueryResponse:
        initial_state = {"question": request.query, "context": [], "answer": "", "score": 0.0}
        result = graph.invoke(initial_state)
        return QueryResponse(
            final_answer=result.get("answer", ""),
            retrieved_context=result.get("context", []),
            confidence_score=float(result.get("score", 0.0)),
        )
else:  # pragma: no cover
    app = None


def run_streamlit() -> None:
    import streamlit as st

    st.title("Agentic AI RAG Chatbot")
    question = st.text_input("Ask a question about the eBook")
    if question:
        result = build_rag_graph(pdf_path=os.getenv("PDF_PATH")).invoke(
            {"question": question, "context": [], "answer": "", "score": 0.0}
        )
        st.write("### Answer")
        st.write(result.get("answer", ""))
        st.write("### Context")
        for item in result.get("context", []):
            st.write(item)
        st.write(f"### Confidence: {result.get('score', 0.0)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
