"""Embedding service for vector generation"""
import numpy as np
from typing import Optional
from src.core.config import settings


class EmbeddingService:
    """Service for generating embeddings from text"""

    def __init__(self):
        self.use_local = settings.USE_LOCAL_MODEL or not settings.OPENAI_API_KEY
        self.api_key = settings.OPENAI_API_KEY
        self.openai_model = settings.OPENAI_EMBEDDING_MODEL
        self.local_model_name = settings.LOCAL_EMBEDDING_MODEL
        self.local_model = None

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

        if self.use_local:
            return self._get_local_embedding(text)
        return self._get_openai_embedding(text)

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm == 0 or np.isnan(norm):
            return embedding.astype('float32')
        return (embedding / norm).astype('float32')

    def _get_local_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding using a local sentence-transformers model.
        """
        try:
            if self.local_model is None:
                from sentence_transformers import SentenceTransformer
                self.local_model = SentenceTransformer(self.local_model_name)

            embedding = self.local_model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return self._normalize_embedding(np.array(embedding, dtype='float32'))
        except Exception as e:
            print(f"Local embedding error: {e}")
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=1000)
            embedding = vectorizer.fit_transform([text]).toarray()[0]
            return self._normalize_embedding(np.array(embedding, dtype='float32'))

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
            client = openai.OpenAI(api_key=self.api_key)

            response = client.embeddings.create(
                input=text,
                model=self.openai_model
            )
            embedding = np.array(response['data'][0]['embedding'], dtype='float32')
            return self._normalize_embedding(embedding)
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
