import os
import json
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class VectorStore:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)
        self.meta_path = os.path.join(self.storage_path, "meta.jsonl")
        self._meta: List[Dict] = []
        self._texts: List[str] = []
        self._vectorizer = None
        self._doc_matrix = None

        if os.path.exists(self.meta_path):
            self._load_meta()
            self._fit_index()

    def _load_meta(self):
        self._meta = []
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    self._meta.append(json.loads(line))
        self._texts = [record["text"] for record in self._meta]

    def _save_meta(self, record: Dict):
        with open(self.meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _fit_index(self):
        if not self._texts:
            self._vectorizer = None
            self._doc_matrix = None
            return
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._doc_matrix = self._vectorizer.fit_transform(self._texts)

    def add_documents(self, texts: List[str], source: str = "uploaded"):
        if not texts:
            return
        for text in texts:
            record = {"source": source, "text": text}
            self._meta.append(record)
            self._save_meta(record)
        self._texts = [record["text"] for record in self._meta]
        self._fit_index()

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        if not self._meta or self._vectorizer is None:
            return []
        query_vec = self._vectorizer.transform([query_text])
        similarities = linear_kernel(query_vec, self._doc_matrix).flatten()
        best_indices = similarities.argsort()[::-1][:top_k]
        results = []
        for idx in best_indices:
            if similarities[idx] <= 0:
                continue
            results.append(self._meta[idx])
        return results
