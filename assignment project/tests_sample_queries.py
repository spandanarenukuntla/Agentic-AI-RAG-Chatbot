from __future__ import annotations

from src.graph import run_chat_query

SAMPLE_QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "Who won the 2022 FIFA World Cup?",
    "What are the main goals of the eBook?",
]


def run_tests() -> None:
    for index, question in enumerate(SAMPLE_QUERIES, start=1):
        result = run_chat_query(question)
        print(f"\nQuery {index}: {question}")
        print(f"Answer: {result.get('answer', '')}")
        print(f"Context count: {len(result.get('context', []))}")
        print(f"Confidence: {result.get('score', 0.0)}")


if __name__ == "__main__":
    run_tests()
