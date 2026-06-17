"""RAG (Retrieval-Augmented Generation) Engine"""
from pathlib import Path
from typing import Optional
from src.services.retrieval_service import VectorStore
from src.services.embedding_service import EmbeddingService
from src.core.config import settings


class RAGEngine:
    """Main RAG engine for question answering"""
    
    def __init__(self):
        self.base_store_path = Path(settings.VECTOR_STORE_PATH)
        self.base_store_path.mkdir(parents=True, exist_ok=True)
        self.embedding_service = EmbeddingService()
        self.top_k = settings.TOP_K_RESULTS
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        self.session_stores: dict[str, VectorStore] = {}

    def _get_vector_store(self, session_id: Optional[str] = None) -> VectorStore:
        if not session_id:
            session_id = "global"

        if session_id not in self.session_stores:
            store_path = self.base_store_path / session_id
            self.session_stores[session_id] = VectorStore(str(store_path))

        return self.session_stores[session_id]
    
    def add_documents(self, documents: list[str], session_id: Optional[str] = None):
        """
        Add documents to the session-specific vector store
        
        Args:
            documents: List of document chunks
            session_id: Optional session identifier
        """
        if documents:
            self._get_vector_store(session_id).add_documents(documents)
    
    def retrieve_context(self, query: str, session_id: Optional[str] = None) -> list[dict]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query
            session_id: Session identifier to scope search
            
        Returns:
            List of relevant documents with scores
        """
        results = self._get_vector_store(session_id).search(query, top_k=self.top_k)
        
        # Filter by similarity threshold; if nothing is sufficiently similar, return no context
        filtered_results = [r for r in results if r['score'] >= self.similarity_threshold]
        if not filtered_results:
            return []

        # Remove duplicate document texts
        unique = []
        seen = set()
        for result in filtered_results:
            doc_text = result['document'].strip()
            if doc_text not in seen:
                seen.add(doc_text)
                unique.append(result)
        return unique
    
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate response using OpenAI or local LLM
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Generated response
        """
        if settings.OPENAI_API_KEY:
            if context.strip():
                return self._generate_with_openai(query, context)
            return self._generate_general_openai_response(query)
        else:
            return self._generate_with_local(query, context)

    def _generate_general_openai_response(self, query: str) -> str:
        """Generate a general OpenAI response for non-PDF queries"""
        try:
            import openai

            openai.api_key = settings.OPENAI_API_KEY
            system_prompt = (
                "You are a helpful agriculture assistant. "
                "If the user asks about agriculture, answer clearly and simply. "
                "If the user asks a general or casual question, answer naturally like a friendly chat assistant. "
                "Do not mention that you need a PDF if the question is general. "
                "Keep the response polite and easy to understand."
            )

            user_prompt = (
                f"Question: {query}\n\n"
                "Answer naturally and helpfully."
            )

            response = openai.ChatCompletion.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=250,
                n=1,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )

            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _generate_with_openai(self, query: str, context: str) -> str:
        """Generate response using OpenAI API"""
        if not context.strip():
            return "I couldn't find relevant information to answer your question. Please upload a PDF with agricultural information or ask a question covered by the uploaded document."

        try:
            import openai

            openai.api_key = settings.OPENAI_API_KEY

            system_prompt = (
                "You are an agricultural expert helping farmers. "
                "Answer using only the context provided from the uploaded PDF. "
                "Do not invent facts, do not reference outside knowledge, and do not search the web. "
                "Do not repeat the context verbatim. Instead, summarize the answer naturally in your own words. "
                "If the answer is not contained in the PDF context, reply exactly: "
                "I couldn't find relevant information to answer your question."
            )

            user_prompt = (
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer concisely, directly, and in a way a farmer can understand. "
                "Do not quote the context verbatim; instead, respond in a general, natural way. "
                "If the context does not contain the answer, say exactly: "
                "I couldn't find relevant information to answer your question."
            )

            response = openai.ChatCompletion.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=250,
                n=1,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )

            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _generate_with_local(self, query: str, context: str) -> str:
        """Generate response using local summarization"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import re
        
        if not context.strip():
            return "I couldn't find relevant information to answer your question. Please upload a PDF with agricultural information."
        
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context) if s.strip()]
        if not sentences:
            return "I couldn't find relevant information to answer your question. Please upload a PDF with agricultural information."
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            sentence_vectors = vectorizer.fit_transform(sentences)
            query_vector = vectorizer.transform([query])
            scores = cosine_similarity(query_vector, sentence_vectors)[0]
            top_indices = scores.argsort()[::-1]
            top_sentences = []
            seen_sentences = set()
            for idx in top_indices:
                if scores[idx] <= 0:
                    break
                sentence = sentences[idx]
                if sentence not in seen_sentences:
                    seen_sentences.add(sentence)
                    top_sentences.append(sentence)
                if len(top_sentences) >= 3:
                    break
            if top_sentences:
                return ' '.join(top_sentences)
        except Exception:
            pass
        
        if len(sentences) > 3:
            return '. '.join(sentences[:3]) + '.'
        return context    

    def _is_small_talk(self, query: str) -> bool:
        import re

        q = query.strip().lower()
        if not q:
            return False

        small_talk_patterns = [
            r"\b(hi|hello|hey|greetings)\b",
            r"\b(how are you|how are you doing|how is it going|how is your day|how was your day|what's up|whats up)\b",
            r"\b(thanks|thank you)\b",
            r"\b(bye|goodbye|see you)\b",
            r"\b(nice to meet you|pleased to meet you|what can you do|who are you|what are you|are you a bot|are you a machine)\b",
            r"\b(ok|okay|oke)\b.*\b(begin|start)\b",
            r"\b(let's begin|lets begin|let's start|lets start)\b",
        ]

        if len(q.split()) <= 14 and any(re.search(pattern, q) for pattern in small_talk_patterns):
            return True
        return False

    def _generate_small_talk_response(self, query: str) -> str:
        import re

        q = query.strip().lower()
        if re.search(r"\b(hi|hello|hey|greetings)\b", q):
            return "Hello! Upload a PDF and ask me questions about agriculture. I can answer from the document."
        if re.search(r"\b(how are you|how are you doing|how is it going|how is your day|how was your day)\b", q):
            return "I'm doing well, thank you. I'm ready to help with agricultural questions from your PDF."
        if re.search(r"\b(are you a bot|are you a machine|who are you|what are you|what can you do)\b", q):
            return "Yes, I'm an agriculture assistant bot. Upload a PDF and ask me questions from the document."
        if re.search(r"\b(ok|okay|oke)\b.*\b(begin|start)\b", q) or re.search(r"\b(let's begin|lets begin|let's start|lets start)\b", q):
            return "Great! Upload a PDF and then ask me your agriculture questions. I'm ready to help."
        if re.search(r"\b(thanks|thank you)\b", q):
            return "You're welcome! Ask another agriculture question whenever you're ready."
        if re.search(r"\b(bye|goodbye|see you)\b", q):
            return "Goodbye! Bring another PDF when you want more agricultural answers."
        return "I can help answer questions from your uploaded PDF. Please upload a document first."

    def _is_summary_request(self, query: str) -> bool:
        import re

        q = query.strip().lower()
        if not q:
            return False

        summary_patterns = [
            r"\b(summarize|summary|summarise)\b",
            r"\b(whole pdf|whole document|entire pdf|entire document|document overview|overview of the document|full document)\b",
            r"\b(what is this document about|what is this pdf about|what does this document (talk|talks) about|what does this pdf (talk|talks) about|tell me about this document|tell me about this pdf|what does this document say|what does this pdf say)\b",
        ]

        return any(re.search(pattern, q) for pattern in summary_patterns)

    def _generate_document_summary(self, query: str, session_id: Optional[str] = None) -> str:
        documents = self._get_vector_store(session_id).documents
        if not documents:
            return "I couldn't find relevant information to answer your question. Please upload a PDF with agricultural information."

        context = " ".join(documents)
        if len(context) > 20000:
            context = " ".join(documents[:20])

        if settings.OPENAI_API_KEY:
            return self._generate_openai_summary(context)
        return self._generate_local_summary(context)

    def _generate_openai_summary(self, context: str) -> str:
        try:
            import openai

            openai.api_key = settings.OPENAI_API_KEY
            system_prompt = (
                "You are a concise agricultural assistant. "
                "Read the following PDF content and summarize the main topics and findings in a few clear bullet points. "
                "Do not repeat long text verbatim; paraphrase and keep the summary short and easy to read."
            )
            user_prompt = (
                f"PDF content:\n{context}\n\n"
                "Please summarize the document as 3-5 bullet points."
            )

            response = openai.ChatCompletion.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=250,
                n=1,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def _generate_local_summary(self, context: str) -> str:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import re

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context) if s.strip()]
        if not sentences:
            return "I couldn't find relevant information to answer your question. Please upload a PDF with agricultural information."

        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            sentence_vectors = vectorizer.fit_transform(sentences)
            doc_vector = sentence_vectors.mean(axis=0)
            scores = cosine_similarity(doc_vector, sentence_vectors)[0]
            top_indices = scores.argsort()[::-1][:6]
            top_sentences = [sentences[i] for i in top_indices]
            bullets = [f"- {s}" for s in top_sentences]
            return '\n'.join(bullets)
        except Exception:
            if len(sentences) > 6:
                bullets = [f"- {s}" for s in sentences[:6]]
                return '\n'.join(bullets)
            return '\n'.join([f"- {s}" for s in sentences])

    def generate_answer(self, query: str, session_id: Optional[str] = None) -> dict:
        """
        Complete QA pipeline: retrieve + generate
        
        Args:
            query: User query
            session_id: Optional session identifier
            
        Returns:
            Dictionary with answer and metadata
        """
        if self._is_small_talk(query):
            answer = self._generate_small_talk_response(query)
            return {
                "answer": answer,
                "context_used": 0,
                "sources": [],
                "query": query
            }

        if self._is_summary_request(query):
            answer = self._generate_document_summary(query, session_id)
            context_count = len(self._get_vector_store(session_id).documents)
            return {
                "answer": answer,
                "context_used": context_count,
                "sources": [],
                "query": query
            }

        # Retrieve relevant documents
        retrieved = self.retrieve_context(query, session_id)
        
        # Build context from retrieved documents
        context = " ".join([r['document'] for r in retrieved])
        
        # Generate response
        answer = self.generate_response(query, context)
        
        return {
            "answer": answer,
            "context_used": len(retrieved),
            "sources": retrieved,
            "query": query
        }
