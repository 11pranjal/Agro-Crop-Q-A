from io import BytesIO
from typing import List
import PyPDF2


def extract_text_from_pdf(file_stream) -> str:
    """Extract text from a PDF file-like object."""
    try:
        file_stream.seek(0)  # Reset stream position to beginning
    except (AttributeError, OSError):
        pass  # Stream might not be seekable, continue anyway
    
    try:
        reader = PyPDF2.PdfReader(file_stream)
        texts = []
        num_pages = len(reader.pages)
        print(f"DEBUG: PDF has {num_pages} pages")
        
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text:
                    texts.append(text)
                    print(f"DEBUG: Page {i} extracted {len(text)} characters")
                else:
                    texts.append("")
                    print(f"DEBUG: Page {i} is empty")
            except Exception as e:
                print(f"Error extracting text from page {i}: {e}")
                texts.append("")
        
        result = "\n\n".join(texts)
        print(f"DEBUG: Total extracted text: {len(result)} characters")
        return result
    except Exception as e:
        print(f"Error reading PDF: {e}")
        raise ValueError(f"Failed to read PDF: {str(e)}")


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
