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

- The project currently uses TF-IDF and a local summarization fallback for free PDF Q&A.
- A stronger local retrieval pipeline is now supported using SentenceTransformer embeddings and FAISS.
- The PDF extractor now also tries OCR on embedded images when Tesseract and an image backend are available.
- If you want higher-quality LLM responses, set `OPENAI_API_KEY` in a `.env` file and configure `OPENAI_MODEL` in `src/core/config.py`.

## OCR Setup

To enable image OCR inside PDFs, install the following dependencies and a Tesseract runtime:

```bash
pip install -r requirements.txt
```

On Windows, install Tesseract OCR:

1. Download the installer from https://github.com/tesseract-ocr/tesseract.
2. Install it and add the Tesseract installation directory to your PATH.

If Tesseract is not on PATH, configure it in Python by setting:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

The app will use OCR only when the required libraries are installed and the Tesseract binary is available.

# Agro-Crop-Q-A
