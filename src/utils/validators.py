"""Input validation utilities"""
import re
from typing import Optional


def validate_text_input(text: str, min_length: int = 3, max_length: int = 5000) -> tuple[bool, str]:
    """
    Validate user text input
    
    Args:
        text: Text to validate
        min_length: Minimum length
        max_length: Maximum length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Input cannot be empty"
    
    text = text.strip()
    
    if len(text) < min_length:
        return False, f"Input must be at least {min_length} characters"
    
    if len(text) > max_length:
        return False, f"Input cannot exceed {max_length} characters"
    
    return True, ""


def validate_filename(filename: str) -> tuple[bool, str]:
    """
    Validate uploaded filename
    
    Args:
        filename: Filename to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not filename:
        return False, "Filename cannot be empty"
    
    # Check if it's a PDF
    if not filename.lower().endswith('.pdf'):
        return False, "Only PDF files are allowed"
    
    # Check for invalid characters
    if re.search(r'[<>:"|?*]', filename):
        return False, "Filename contains invalid characters"
    
    # Check filename length
    if len(filename) > 255:
        return False, "Filename is too long"
    
    return True, ""


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    import uuid
    import os
    
    # Get file extension
    ext = os.path.splitext(filename)[1]
    
    # Create safe filename with UUID
    safe_name = f"{uuid.uuid4().hex}{ext}"
    
    return safe_name
