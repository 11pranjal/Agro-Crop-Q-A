"""PDF processing and text extraction utilities"""
from typing import Optional
import PyPDF2
from io import BytesIO

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except ImportError:
    pytesseract = None


def extract_text_from_pdf(file_stream) -> str:
    """
    Extract text from PDF file stream, including OCR on images when available.

    Args:
        file_stream: File stream object

    Returns:
        Extracted text from PDF
    """
    try:
        pdf_bytes = _read_file_bytes(file_stream)
        text = _extract_pdf_text(pdf_bytes)
        ocr_text = _extract_image_text_from_pdf(pdf_bytes)

        if ocr_text:
            text = f"{text}\n{ocr_text}" if text else ocr_text

        return text.strip()
    except Exception as e:
        raise ValueError(f"Error extracting PDF text: {str(e)}")


def _read_file_bytes(file_stream) -> bytes:
    if isinstance(file_stream, (bytes, bytearray)):
        return bytes(file_stream)

    if hasattr(file_stream, "read"):
        data = file_stream.read()
        try:
            file_stream.seek(0)
        except Exception:
            pass
        return data

    raise ValueError("Unsupported PDF input type")


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()


def _extract_image_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""

    ocr_texts = []

    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            pix = page.get_pixmap(alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text = pytesseract.image_to_string(image, lang="eng")
            if page_text:
                ocr_texts.append(page_text)
        return "\n".join(ocr_texts).strip()
    except Exception:
        pass

    try:
        from pdf2image import convert_from_bytes
        pages = convert_from_bytes(pdf_bytes, dpi=300)
        for image in pages:
            page_text = pytesseract.image_to_string(image, lang="eng")
            if page_text:
                ocr_texts.append(page_text)
        return "\n".join(ocr_texts).strip()
    except Exception:
        return ""


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Split text into sentence-based chunks with configurable overlap.

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

        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            if overlap > 0:
                overlap_text = current[-overlap:]
                current = (overlap_text + " " + sentence).strip()
            else:
                current = sentence
        else:
            # Sentence itself is longer than chunk_size, keep it as a single chunk.
            chunks.append(sentence)
            current = ""

        if len(current) > chunk_size:
            chunks.append(current)
            current = ""

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

    # Normalize hyphenated line breaks and bad spaces around dashes
    text = re.sub(r'\s*-\s*\n\s*', '', text)
    text = re.sub(r'\s*-\s*', '-', text)

    # Remove stray spaces before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.!?,()-]', '', text)

    return text.strip()
