# Agro QA Chatbot
# AGRO QA Chatbot

RAG-based Q&A system for farmers. Upload PDFs and ask natural-language questions.

This repository was restructured to use a modern stack:

- FastAPI backend (`src/api/app.py`)
- Modular services in `src/services` and `src/core`
- React + Vite frontend skeleton in `frontend/`
- SQLite persistent conversation history (`database/`)
- Docker and docker-compose for local development

## Features

- PDF upload and chunking
- TF-IDF retrieval-based context (vector store)
- RAG-style pipeline with optional OpenAI support
- FastAPI backend and React frontend
- SQLite persistent conversation history
- Docker + docker-compose for local dev

## Quickstart

1. Create a Python venv and activate it

```bash
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
# or bash
source venv/bin/activate
```

2. Install Python dependencies

```bash
pip install -r requirements.txt
```

3. Run backend

```bash
uvicorn src.api.app:app --reload --port 8000
```

4. Run frontend (from `frontend` folder)

```bash
npm install
npm run dev
```

5. Open frontend at `http://localhost:5173` and backend at `http://localhost:8000`

## Project Layout

See the `src/`, `frontend/`, `database/`, and `docker/` folders for implementation.

## Notes

- The project currently uses TF-IDF for local embeddings and retrieval as a lightweight RAG approach.
- If you want higher-quality LLM responses, set `OPENAI_API_KEY` in a `.env` file and configure `OPENAI_MODEL` in `src/core/config.py`.


# Agro-Crop-Q-A
