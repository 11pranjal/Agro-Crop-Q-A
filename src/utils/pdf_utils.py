"""PDF processing and text extraction utilities"""
from typing import Optional
import PyPDF2
from io import BytesIO


def extract_text_from_pdf(file_stream) -> str:
    """
    Extract text from PDF file stream
    
    Args:
        file_stream: File stream object
        
    Returns:
        Extracted text from PDF
    """
    try:
        pdf_reader = PyPDF2.PdfReader(file_stream)
        text = ""
        
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error extracting PDF text: {str(e)}")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Split text into sentence-based chunks
    
    Args:
        text: Text to chunk
        chunk_size: Approximate size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    import re

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = (current + " " + sentence).strip() if current else sentence
        else:
            if current:
                chunks.append(current)
            current = sentence
    
    if current:
        chunks.append(current)

    return chunks


def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    import re
    
    # Remove common PDF noise strings
    text = re.sub(r'(?i)back to contents', ' ', text)
    text = re.sub(r'(?i)photo[^.\n]*', ' ', text)
    text = re.sub(r'(?i)(figure|image|source|copyright)[^.\n]*', ' ', text)
    text = re.sub(r'(?i)bugwood\.org', ' ', text)
    text = re.sub(r'(?i)all [A-Z][a-z]+ [A-Z][a-z]+ [^\n]*', ' ', text)

    # Normalize whitespace and remove stray special characters
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.!?,()-]', '', text)
    
    return text.strip()
