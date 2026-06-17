"""PDF processing service"""
from pathlib import Path
from typing import Optional
from src.utils.pdf_utils import extract_text_from_pdf, chunk_text, clean_text
from src.utils.validators import sanitize_filename
from src.core.config import settings
import shutil


class PDFService:
    """Service for handling PDF uploads and processing"""
    
    def __init__(self):
        self.documents_path = settings.DOCUMENTS_PATH
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
    
    def save_uploaded_file(self, file_bytes, original_filename: str) -> tuple[str, str]:
        """
        Save uploaded PDF file
        
        Args:
            file_bytes: File bytes/stream
            original_filename: Original filename
            
        Returns:
            Tuple of (saved_filename, file_path)
        """
        safe_filename = sanitize_filename(original_filename)
        file_path = self.documents_path / safe_filename
        
        # Save file
        with open(file_path, 'wb') as f:
            # Handle both bytes and file-like objects
            if isinstance(file_bytes, bytes):
                f.write(file_bytes)
            else:
                f.write(file_bytes.read())
        
        return safe_filename, str(file_path)
    
    def process_pdf(self, file_stream) -> dict:
        """
        Process PDF file and extract chunks
        
        Args:
            file_stream: File stream
            
        Returns:
            Dictionary with extracted text and chunks
        """
        try:
            # Extract text
            text = extract_text_from_pdf(file_stream)
            
            if not text.strip():
                raise ValueError("PDF contains no extractable text")
            
            # Clean text
            cleaned_text = clean_text(text)
            
            # Create chunks
            chunks = chunk_text(cleaned_text, self.chunk_size, self.chunk_overlap)
            
            return {
                "success": True,
                "raw_text": text,
                "cleaned_text": cleaned_text,
                "chunks": chunks,
                "total_chunks": len(chunks),
                "char_count": len(cleaned_text)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes"""
        return Path(file_path).stat().st_size
    
    def delete_file(self, file_path: str) -> bool:
        """Delete uploaded file"""
        try:
            Path(file_path).unlink()
            return True
        except Exception:
            return False
