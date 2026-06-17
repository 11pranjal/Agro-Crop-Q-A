"""FastAPI application for AGRO QA Chatbot"""
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
import shutil
import os
from pathlib import Path

from src.core.config import settings
from src.services.pdf_service import PDFService
from src.core.rag_engine import RAGEngine
from src.core.memory import HybridMemory
from database.models import init_db, get_db

app = FastAPI(title="AGRO QA Chatbot")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_db()

# Services
pdf_service = PDFService()
rag = RAGEngine()
memory = HybridMemory()


@app.get("/health")
async def health_check():
    return {"message": "AGRO QA Chatbot API is running."}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Validate filename
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    saved_name, path = pdf_service.save_uploaded_file(await file.read(), file.filename)

    # Re-open saved file for processing
    with open(path, 'rb') as f:
        result = pdf_service.process_pdf(f)

    if not result.get('success'):
        return JSONResponse(status_code=400, content={"error": result.get('error')})

    # Add chunks to RAG engine
    chunks = result.get('chunks', [])
    if chunks:
        rag.add_documents(chunks)

    return {
        "success": True,
        "filename": saved_name,
        "total_chunks": result.get('total_chunks')
    }


@app.post("/chat")
async def chat(query: str = Form(...), session_id: Optional[str] = Form(None)):
    # Basic validation
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Create session if none
    if not session_id:
        session_id = memory.session_memory.create_session()

    # Get context from hybrid memory
    conversation_context = memory.get_context(session_id)

    # Generate answer via RAG
    qa = rag.generate_answer(query)
    answer = qa.get('answer')

    # Save conversation
    memory.add_conversation(session_id, query, answer)

    return {
        "response": {
            "response": answer,
            "context_used": qa.get('context_used'),
            "sources": qa.get('sources'),
            "session_id": session_id
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.app:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
