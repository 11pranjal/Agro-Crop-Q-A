"""Text chunking and processing utilities"""


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    if not text or chunk_size <= 0:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('. ')
            if last_period > chunk_size // 2:
                end = start + last_period + 2
                chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start = end - overlap
    
    return chunks


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
    """
    import re
    
    # Split on period, question mark, exclamation mark
    sentences = re.split(r'[.!?]+', text)
    
    return [s.strip() for s in sentences if s.strip()]


def remove_duplicates(chunks: list[str]) -> list[str]:
    """
    Remove duplicate chunks while preserving order
    
    Args:
        chunks: List of text chunks
        
    Returns:
        List of unique chunks
    """
    seen = set()
    unique_chunks = []
    
    for chunk in chunks:
        if chunk not in seen:
            seen.add(chunk)
            unique_chunks.append(chunk)
    
    return unique_chunks
