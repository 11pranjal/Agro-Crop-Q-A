import os
import json
from typing import List, Dict
import pickle

class VectorStore:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)
        self.meta_path = os.path.join(self.storage_path, "meta.jsonl")
        self._meta: List[Dict] = []
        self._texts: List[str] = []
        
        if os.path.exists(self.meta_path):
            self._load_meta()
        print(f"VectorStore initialized. Loaded {len(self._meta)} documents from {self.meta_path}")

    def _load_meta(self):
        self._meta = []
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    self._meta.append(json.loads(line))
        self._texts = [record["text"] for record in self._meta]

    def _save_meta(self, record: Dict):
        try:
            with open(self.meta_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()  # Ensure data is written to disk
        except IOError as e:
            raise IOError(f"Failed to save document metadata to {self.meta_path}: {e}")

    def add_documents(self, texts: List[str], source: str = "uploaded"):
        if not texts:
            return
        initial_count = len(self._meta)
        for text in texts:
            record = {"source": source, "text": text}
            self._meta.append(record)
            self._save_meta(record)
        self._texts = [record["text"] for record in self._meta]
        print(f"Added {len(texts)} documents from {source}. Total documents: {len(self._meta)}")

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Simple keyword-based search without TF-IDF for better performance"""
        if not self._meta:
            return []
        
        # Simple keyword matching - much faster than TF-IDF
        query_words = set(query_text.lower().split())
        results_with_score = []
        
        for idx, record in enumerate(self._meta):
            text_words = set(record["text"].lower().split())
            # Calculate overlap score
            overlap = len(query_words & text_words)
            if overlap > 0:
                results_with_score.append((overlap, idx, record))
        
        # Sort by score (overlap count) and return top k
        results_with_score.sort(reverse=True, key=lambda x: x[0])
        return [record for _, _, record in results_with_score[:top_k]]
