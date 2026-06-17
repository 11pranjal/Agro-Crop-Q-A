"""Vector store and retrieval service"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path
from typing import Optional
from src.core.config import settings


class VectorStore:
    """Vector store for document retrieval using TF-IDF"""
    
    def __init__(self, storage_path: str = None):
        self.storage_path = Path(storage_path or settings.VECTOR_STORE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.vectorizer_path = self.storage_path / "vectorizer.pkl"
        self.vectors_path = self.storage_path / "vectors.npy"
        self.docs_path = self.storage_path / "documents.pkl"
        
        self.vectorizer = None
        self.vectors = None
        self.documents = []
        
        self._load()
    
    def _load(self):
        """Load existing vectorizer and documents"""
        if self.vectorizer_path.exists() and self.vectors_path.exists():
            try:
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                self.vectors = np.load(self.vectors_path)
                
                with open(self.docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
            except Exception as e:
                print(f"Error loading vector store: {e}")
                self._reset()
        else:
            self._reset()
    
    def _reset(self):
        """Reset vector store"""
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.vectors = None
        self.documents = []
    
    def _save(self):
        """Save vectorizer and documents"""
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        if self.vectors is not None:
            np.save(self.vectors_path, self.vectors)
        
        with open(self.docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
    
    def add_documents(self, documents: list[str]):
        """
        Add documents to vector store
        
        Args:
            documents: List of document strings
        """
        self.documents.extend(documents)
        
        # Re-fit vectorizer with all documents
        self.vectors = self.vectorizer.fit_transform(self.documents).toarray()
        
        self._save()
    
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Search for similar documents
        
        Args:
            query: Query string
            top_k: Number of top results
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.vectors is None or len(self.documents) == 0:
            return []
        
        try:
            query_vector = self.vectorizer.transform([query]).toarray()
            similarities = cosine_similarity(query_vector, self.vectors)[0]
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only return if there's some similarity
                    results.append({
                        "document": self.documents[idx],
                        "score": float(similarities[idx]),
                        "index": int(idx)
                    })
            
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def clear(self):
        """Clear vector store"""
        self._reset()
        self._save()
    
    def get_document_count(self) -> int:
        """Get number of documents in store"""
        return len(self.documents)
