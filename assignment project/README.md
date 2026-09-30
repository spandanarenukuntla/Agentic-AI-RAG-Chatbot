# Agentic AI RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot for the Agentic AI eBook. It stores document chunks for retrieval, uses a stateful LangGraph workflow, and exposes a FastAPI API plus an optional Streamlit UI.

## Project structure

- `src/config.py` — environment variables and paths.
- `src/ingestion.py` — PDF/text loading, chunking, and retrieval scoring.
- `src/graph.py` — graph state, retrieval node, generation node, and local fallback answer logic.
- `app.py` — FastAPI server and optional Streamlit UI.
- `tests_sample_queries.py` — benchmark queries for validation.
- `data/` — source PDF or a local fallback text file.

## Setup

1. Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   # Windows: .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:

   ```bash
   copy .env.example .env
   ```

   Then update the values with your API credentials:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=agentic-ai-index
   PDF_PATH=./data/Ebook-Agentic-AI.pdf
   ```

4. Download the Agentic AI eBook PDF and save it in `data/Ebook-Agentic-AI.pdf`.

   If the PDF is unavailable, the project automatically falls back to `data/agentic-ai-reference.txt` so it can still run in a local/offline environment.

## Run the API

```bash
python app.py
```

Then call the endpoint with:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Agentic AI according to the eBook?"}'
```

Example response:

```json
{
  "final_answer": "Agentic AI refers to AI systems that can pursue goals, reason through tasks, use tools, and maintain context while operating with autonomy.",
  "retrieved_context": ["..."],
  "confidence_score": 0.95
}
```

## Run the Streamlit UI (optional)

```bash
streamlit run app.py
```

## Test benchmark queries

```bash
python tests_sample_queries.py
```

The script includes the following checks:

- What is Agentic AI according to the eBook?
- How do AI agents differ from traditional automation systems?
- What are the core components of an Agentic Architecture?
- What role does memory play in Agentic AI workflows?
- Who won the 2022 FIFA World Cup?

The final query is designed to validate that the system refuses to answer when context is absent.

## Notes

- The project is designed to work with Pinecone and OpenAI when credentials are available.
- If the credentials are missing, the project still answers from the local fallback text and returns a grounded, confidence-based response.
- This assignment follows the requested pattern for an end-to-end RAG chatbot workflow, complete with retrieval, grounding, and response generation.
