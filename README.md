# AGRO — Intelligent Agricultural RAG System

An advanced Retrieval-Augmented Generation (RAG) system designed for intelligent question-answering over agricultural PDF documents. Upload manuals, guides, research papers, and farming reports to build a searchable knowledge base, then ask natural-language questions. The system retrieves relevant document chunks and synthesizes accurate, context-aware answers using configurable LLMs.

**Features • Architecture • Installation • Usage • API • Evaluation Metrics**

## Features

**Intelligent RAG System**: Leverages FAISS vector search and embeddings for context-aware responses  
**Multi-Model Support**: Support for local models (Ollama) and cloud providers (OpenAI API)  
**Semantic Search**: Advanced vector retrieval with configurable chunk size and top-k retrieval  
**Memory System**: Session memory for short-term context and SQLite for persistent conversation history  
**Interactive Frontend**: Modern React + TypeScript interface with real-time response streaming  
**Knowledge Base**: Support for multiple document formats (PDF, TXT) with intelligent chunking  
**Performance Optimized**: Efficient vector indexing with FAISS for sub-second retrieval  

## Architecture

```
┌─────────────────────────┐
│   React Frontend        │  (Port 5173 or 8080)
│   (TypeScript + Vite)   │
└────────┬────────────────┘
         │ HTTP/REST
         ▼
┌─────────────────────────┐
│   FastAPI Backend       │  (Port 8000)
│   REST API Server       │
└────────┬────────────────┘
         │
    ┌────┴──────┐
    │            │
    ▼            ▼
┌─────────┐  ┌──────────────┐
│   RAG   │  │    Memory    │
│ Engine  │  │    System    │
└────┬────┘  └──────────────┘
     │
┌────┴──────────────────────┐
│ FAISS Vector Search        │
│ + Embedding Models         │
│ + LLM (Ollama/OpenAI)      │
└────────────────────────────┘
```

## Core Components

- **[src/core/rag_engine.py](src/core/rag_engine.py)**: Handles embedding generation, vector retrieval, and LLM response generation
- **[src/services/retrieval_service.py](src/services/retrieval_service.py)**: Document retrieval and ranking pipeline
- **[src/services/embedding_service.py](src/services/embedding_service.py)**: Embedding generation and vector management
- **[src/services/pdf_service.py](src/services/pdf_service.py)**: PDF document processing and chunking
- **[src/core/memory.py](src/core/memory.py)**: Dual memory system (session + persistent SQLite)
- **[src/api/app.py](src/api/app.py)**: FastAPI REST API server
- **[frontend/src/pages/Chat.tsx](frontend/src/pages/Chat.tsx)**: React chat interface

## Tech Stack

### Backend
- **Python 3.12+**
- **FastAPI** - Modern async web framework
- **FAISS** - Efficient similarity search and clustering
- **Ollama** - Local LLM inference (configurable models)
- **SQLite** - Persistent conversation memory
- **NumPy & Pandas** - Data processing
- **LangChain** - LLM orchestration (optional)

### Frontend
- **React 18.3** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Styling
- **Radix UI** - Accessible components

## Installation

### Prerequisites
- Python 3.12+ installed
- Node.js 18+ and npm/pnpm installed
- Ollama installed and running locally (or OpenAI API key for cloud models)

### Step 1: Install Ollama Models (Optional for Local Models)

```bash
# Visit https://ollama.ai/download to install Ollama

# Pull required models
ollama pull llama2  # or your preferred model
ollama pull nomic-embed-text  # or your preferred embedding model

# Verify Ollama is running
ollama list
```

### Step 2: Clone Repository

```bash
git clone <repo-url> && cd AGRO
```

### Step 3: Install Python Dependencies

**Option A: Using venv (recommended)**

On Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS / Linux:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Option B: Using uv**
```bash
uv sync
```

### Step 4: Configure Environment Variables

Copy `.env.example` to `.env` and edit with your settings:

```bash
cp .env.example .env
```

Key configuration variables:
- `USE_LOCAL_MODEL` - Set to `true` for Ollama, `false` for OpenAI
- `OLLAMA_URL` - Local Ollama endpoint (default: http://localhost:11434)
- `LOCAL_LLM_MODEL` - Model name for Ollama (e.g., `llama2`, `neural-chat`)
- `EMBEDDING_MODEL` - Embedding model (e.g., `nomic-embed-text`)
- `OPENAI_API_KEY` - Your OpenAI API key (if using cloud models)
- `OPENAI_MODEL` - OpenAI model name (e.g., `gpt-3.5-turbo`)

### Step 5: Install Frontend Dependencies

```bash
cd frontend
npm install
# or
pnpm install
```

### Step 6: Download/Prepare Knowledge Base

Place your agricultural PDF documents in `data/documents/`:

```bash
data/
├── documents/
│   ├── farming_guide.pdf
│   ├── crop_manual.pdf
│   └── research_paper.pdf
└── vector_store/
```

## Usage

### Starting the Backend Server

From project root:

```bash
python app.py
# or
python -m uvicorn src.api.app:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

API docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Starting the Frontend

```bash
cd frontend
npm run dev
# or
pnpm dev
```

The frontend will be available at `http://localhost:5173` (or the URL shown in your terminal)

### Using the Application

1. Open the frontend in your browser
2. Upload agricultural PDF documents using the upload interface
3. Ask natural-language questions about your documents
4. View responses with retrieved source documents and confidence scores
5. Access conversation history and previous queries

## API Endpoints

### POST /chat
Send a chat message and receive an AI response with source document references.

**Request:**
```json
{
  "query": "What are the best practices for rice cultivation?",
  "use_conversation_history": true
}
```

**Response:**
```json
{
  "response": "Based on the agricultural documents in our knowledge base, here are the best practices for rice cultivation...",
  "sources": [
    {
      "document": "rice_farming_guide.pdf",
      "chunk": "Chapter 3: Cultivation Methods",
      "relevance_score": 0.89
    }
  ],
  "tokens_used": 150,
  "latency_ms": 2340
}
```

### POST /documents/upload
Upload PDF documents to build the knowledge base.

**Request:**
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "filename": "document.pdf",
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "chunks_created": 45,
  "embedding_status": "completed"
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "AGRO RAG System is running"
}
```

### GET /documents
List all uploaded documents in the knowledge base.

**Response:**
```json
{
  "documents": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "filename": "rice_farming_guide.pdf",
      "uploaded_at": "2024-01-15T10:30:00Z",
      "chunks": 45,
      "size_bytes": 2500000
    }
  ],
  "total_documents": 3
}
```

## Evaluation Metrics

AGRO tracks three metric groups for research and engineering trade-offs:

### Retrieval Metrics
- **Precision@k**: Fraction of top-k retrieved chunks that are relevant (averaged across queries)
- **Recall@k**: Fraction of query's relevant chunks appearing in top-k results
- **Hit Rate@k**: Fraction of queries with at least one relevant item in top-k
- **MRR@k** (Mean Reciprocal Rank): Average reciprocal rank of first relevant item

### Answer Quality
- **Semantic Similarity**: Average semantic similarity between generated answer and reference answers
- **Hallucination Rate**: Fraction of answers containing unsupported or incorrect information
- **Answer Relevance**: Whether answers directly address the user's question

### System Performance
- **Avg Latency**: Average end-to-end response time (seconds)
- **Avg Context Used**: Average number of document chunks used per query
- **RAG Efficiency**: Precision / Avg Latency (higher is better)

### Evaluation Charts

The repository includes evaluation scripts that produce visual metrics. See [compute_and_plot_metrics.py](compute_and_plot_metrics.py) for implementations.

Retrieval Metrics:

![Retrieval Metrics](images/evaluation_metrics/retrieval_metrics.png)

Answer Quality Metrics:

![Answer Quality Metrics](images/evaluation_metrics/answer_quality_metrics.png)

System Performance Metrics:

![System Performance Metrics](images/evaluation_metrics/system_performance_metrics.png)

## Project Structure

```
AGRO/
├── src/
│   ├── __init__.py
│   ├── api/
│   │   └── app.py                 # FastAPI server and routes
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration management
│   │   ├── memory.py              # Session + persistent memory
│   │   └── rag_engine.py          # RAG core logic
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embedding_service.py   # Embedding generation
│   │   ├── llm_service.py         # LLM inference
│   │   ├── pdf_service.py         # PDF processing
│   │   └── retrieval_service.py   # Document retrieval
│   └── utils/
│       ├── __init__.py
│       ├── pdf_utils.py           # PDF utilities
│       ├── text_processing.py     # Text processing
│       └── validators.py          # Input validation
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   ├── index.css
│   │   ├── components/            # Reusable React components
│   │   ├── pages/
│   │   │   ├── Chat.tsx           # Main chat interface
│   │   │   └── Chat.css
│   │   └── utils/
│   ├── package.json
│   ├── vite.config.js
│   └── public/
├── database/
│   ├── __init__.py
│   ├── models.py                  # SQLAlchemy models
│   └── repository.py              # Data access layer
├── data/
│   ├── documents/                 # Uploaded PDF documents
│   └── vector_store/              # FAISS indices and embeddings
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/
│   ├── test_api.py
│   ├── test_pdf_service.py
│   └── test_retrieval.py
├── app.py                         # Development launcher
├── compute_and_plot_metrics.py   # Evaluation metrics script
├── requirements.txt               # Python dependencies
├── pyproject.toml                # Project metadata
├── README.md                      # This file
└── .env.example                   # Environment template
```

## Configuration

### Backend Configuration

Edit [src/core/config.py](src/core/config.py) or environment variables:

- **Model Selection**: Switch between Ollama (local) and OpenAI (cloud)
- **Vector Settings**: Configure chunk size, embedding dimensions, top-k retrieval
- **API Settings**: CORS origins, host, port (default: 8000)
- **Database**: SQLite path for conversation memory

### Frontend Configuration

Edit [frontend/src/pages/Chat.tsx](frontend/src/pages/Chat.tsx):

- **API Endpoint**: Set backend URL (default: http://localhost:8000)
- **UI Customization**: Colors, themes, layout

Edit [frontend/vite.config.js](frontend/vite.config.js):

- **Frontend Port**: Default 5173
- **Build optimization**

### Ollama Configuration

Ensure Ollama is running on `http://localhost:11434` (default). If using a different port, update:

- [src/services/embedding_service.py](src/services/embedding_service.py)
- [src/services/llm_service.py](src/services/llm_service.py)

## Key Features Explained

### Document Processing
Uploaded PDF documents are automatically:
1. Split into semantic chunks (configurable size)
2. Converted to embeddings using your selected embedding model
3. Indexed in FAISS for fast similarity search
4. Stored with metadata for traceability

### Vector Retrieval
When you ask a question:
1. Query is embedded using the same embedding model
2. FAISS performs nearest-neighbor search to find top-k most relevant chunks
3. Retrieved chunks are scored by relevance
4. Top results are sent to the LLM as context

### Response Generation
The LLM receives:
- User's original question
- Retrieved context from documents
- Conversation history (if enabled)
- System prompt for agricultural expertise

The LLM synthesizes an answer grounded in your documents.

### Memory System

**Session Memory**: Maintains context of the current conversation (in-memory)  
**Persistent Memory**: SQLite stores all conversations for long-term reference and personalization

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start Ollama service
ollama serve
```

### Port Already in Use

If port 8000 is already in use:

```bash
# Use a different port
uvicorn src.api.app:app --reload --port 8001

# Update frontend in frontend/src/pages/Chat.tsx
# Change API_ENDPOINT to "http://localhost:8001/chat"
```

### Missing or Empty Vector Store

If you see errors about missing embeddings:

1. Ensure PDF documents are in `data/documents/`
2. Upload documents through the frontend or API
3. Wait for indexing to complete (check logs)

### CORS Errors

Ensure the frontend URL in [src/api/app.py](src/api/app.py) matches your actual frontend URL.

### Slow Responses

Performance tips:
- Use quantized embedding models (smaller, faster)
- Reduce top-k retrieval value (fewer chunks to embed)
- Use a smaller LLM model if available
- Enable response caching for common queries

### PDF Upload Failures

Check that:
- PDF is not corrupted
- PDF has readable text (not image-only scans)
- File size is within limits
- You have write permissions in `data/documents/`

## Performance Optimization

### Latency Breakdown
Typical response time (~2-5 seconds):
- **Embedding**: 100-300ms
- **FAISS Retrieval**: 50-200ms
- **LLM Inference**: 1500-4000ms

LLM inference is the primary bottleneck. Options:
- Use smaller/quantized models
- Implement streaming responses
- Add response caching
- Use GPU acceleration if available

## Testing

Run unit tests:

```bash
# Run all tests
python -m pytest -v

# Run specific test file
python -m pytest tests/test_retrieval.py -v

# Run with coverage
python -m pytest --cov=src
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Add tests for new functionality
4. Submit a pull request with clear description

**Open issues** for bugs and feature requests.

## Dependencies

### Python Packages (see [requirements.txt](requirements.txt))

- fastapi >= 0.95.0
- uvicorn >= 0.21.0
- faiss-cpu >= 1.7.4
- pydantic >= 2.0.0
- python-multipart >= 0.0.5
- ollama >= 0.0.10
- requests >= 2.31.0
- sqlalchemy >= 2.0.0
- numpy >= 1.24.0
- pandas >= 2.0.0

### Frontend Packages (see [frontend/package.json](frontend/package.json))

- react >= 18.2.0
- typescript >= 5.0.0
- vite >= 4.0.0
- tailwindcss >= 3.0.0
- radix-ui components

## License

This project is licensed under the MIT License — see LICENSE file for details.

## Acknowledgments

- **Ollama** for providing local LLM inference
- **Meta AI FAISS** for efficient similarity search
- **FastAPI** for the modern Python web framework
- **React & Vite** for the frontend framework
- **Open-source community** for invaluable libraries and tools

--- 
