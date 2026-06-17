"""Vector store and retrieval service"""
import pickle
from pathlib import Path
from typing import Optional
import numpy as np
from src.core.config import settings
from src.services.embedding_service import EmbeddingService


class VectorStore:
    """Vector store for document retrieval using embeddings and optional FAISS."""

    def __init__(self, storage_path: str = None, embedding_service=None):
        self.storage_path = Path(storage_path or settings.VECTOR_STORE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.documents_path = self.storage_path / "documents.pkl"
        self.vectors_path = self.storage_path / "vectors.npy"
        self.index_path = self.storage_path / "faiss.index"

        self.embedding_service = embedding_service or EmbeddingService()
        self.documents: list[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.index = None

        self._load()

    def _load(self):
        """Load existing documents, vectors, and rebuild index."""
        if self.documents_path.exists() and self.vectors_path.exists():
            try:
                with open(self.documents_path, 'rb') as f:
                    self.documents = pickle.load(f)

                self.vectors = np.load(self.vectors_path)
                self._build_index()
            except Exception as e:
                print(f"Error loading vector store: {e}")
                self._reset()
        else:
            self._reset()

    def _reset(self):
        """Reset vector store."""
        self.documents = []
        self.vectors = None
        self.index = None

    def _save(self):
        """Save documents and vectors."""
        with open(self.documents_path, 'wb') as f:
            pickle.dump(self.documents, f)

        if self.vectors is not None:
            np.save(self.vectors_path, self.vectors)

    def _build_index(self):
        """Build or rebuild the FAISS index."""
        try:
            import faiss

            if self.vectors is None or len(self.vectors) == 0:
                self.index = None
                return

            dim = self.vectors.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(self.vectors.astype('float32'))
            self.index = index
        except Exception as e:
            print(f"FAISS unavailable, falling back to TF-IDF search: {e}")
            self.index = None

    def add_documents(self, documents: list[str]):
        """Add documents to vector store."""
        if not documents or self.embedding_service is None:
            return

        new_embeddings = []
        new_texts = []
        for document in documents:
            if document not in self.documents:
                embedding = self.embedding_service.get_embedding(document)
                if embedding is None:
                    continue
                new_texts.append(document)
                new_embeddings.append(embedding)

        if not new_texts:
            return

        self.documents.extend(new_texts)
        if self.vectors is None:
            self.vectors = np.vstack([np.array(new_embeddings, dtype='float32')])
        else:
            self.vectors = np.vstack([self.vectors, np.array(new_embeddings, dtype='float32')])

        self._build_index()
        self._save()

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Search for similar documents."""
        if not query or not self.documents:
            return []

        if self.index is not None and self.embedding_service is not None:
            try:
                query_vec = self.embedding_service.get_embedding(query)
                if query_vec is None:
                    return []

                if query_vec.ndim == 1:
                    query_vec = np.expand_dims(query_vec, axis=0)

                query_vec = query_vec.astype('float32')
                distances, indices = self.index.search(query_vec, top_k)
                results = []
                for score, idx in zip(distances[0], indices[0]):
                    if idx < 0 or idx >= len(self.documents):
                        continue
                    results.append({
                        "document": self.documents[idx],
                        "score": float(score),
                        "index": int(idx)
                    })
                return results
            except Exception as e:
                print(f"Error during FAISS search: {e}")

        return self._search_tfidf(query, top_k)

    def _search_tfidf(self, query: str, top_k: int = 3) -> list[dict]:
        """Fallback TF-IDF search when FAISS is unavailable."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            document_vectors = vectorizer.fit_transform(self.documents).toarray()
            query_vector = vectorizer.transform([query]).toarray()
            similarities = cosine_similarity(query_vector, document_vectors)[0]

            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:
                    results.append({
                        "document": self.documents[idx],
                        "score": float(similarities[idx]),
                        "index": int(idx)
                    })
            return results
        except Exception as e:
            print(f"Error during TF-IDF fallback search: {e}")
            return []

    def clear(self):
        """Clear vector store."""
        self._reset()
        self._save()

    def get_document_count(self) -> int:
        """Get number of documents in store."""
        return len(self.documents)
