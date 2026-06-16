from io import BytesIO
from typing import List
import PyPDF2


def extract_text_from_pdf(file_stream) -> str:
    """Extract text from a PDF file-like object."""
    reader = PyPDF2.PdfReader(file_stream)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n\n".join(texts)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks (by characters)."""
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= length:
            break
    return chunks
