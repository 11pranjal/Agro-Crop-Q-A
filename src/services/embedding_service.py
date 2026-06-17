"""Embedding service for vector generation"""
import numpy as np
from typing import Optional
from src.core.config import settings


class EmbeddingService:
    """Service for generating embeddings from text"""
    
    def __init__(self):
        self.use_local = settings.USE_LOCAL_MODEL
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_EMBEDDING_MODEL
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector or None
        """
        if not text or not text.strip():
            return None
        
        if self.use_local or not self.api_key:
            return self._get_local_embedding(text)
        else:
            return self._get_openai_embedding(text)
    
    def _get_local_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding using local method (TF-IDF based)
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(max_features=1000)
        embedding = vectorizer.fit_transform([text]).toarray()[0]
        
        return embedding
    
    def _get_openai_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding using OpenAI API
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector or None
        """
        try:
            import openai
            
            openai.api_key = self.api_key
            
            response = openai.Embedding.create(
                input=text,
                model=self.model
            )
            
            return np.array(response['data'][0]['embedding'])
        except Exception as e:
            print(f"Error getting OpenAI embedding: {e}")
            return None
    
    def get_batch_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """
        Get embeddings for multiple texts
        
        Args:
            texts: List of texts
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            if embedding is not None:
                embeddings.append(embedding)
        
        return embeddings
