"""RAG (Retrieval-Augmented Generation) Engine"""
from typing import Optional
from src.services.retrieval_service import VectorStore
from src.services.embedding_service import EmbeddingService
from src.core.config import settings


class RAGEngine:
    """Main RAG engine for question answering"""
    
    def __init__(self):
        self.vector_store = VectorStore(str(settings.VECTOR_STORE_PATH))
        self.embedding_service = EmbeddingService()
        self.top_k = settings.TOP_K_RESULTS
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
    
    def add_documents(self, documents: list[str]):
        """
        Add documents to the vector store
        
        Args:
            documents: List of document chunks
        """
        if documents:
            self.vector_store.add_documents(documents)
    
    def retrieve_context(self, query: str) -> list[dict]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query
            
        Returns:
            List of relevant documents with scores
        """
        results = self.vector_store.search(query, top_k=self.top_k)
        
        # Filter by similarity threshold
        filtered_results = [r for r in results if r['score'] >= self.similarity_threshold]
        
        return filtered_results if filtered_results else results[:self.top_k]
    
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
            return self._generate_with_openai(query, context)
        else:
            return self._generate_with_local(query, context)
    
    def _generate_with_openai(self, query: str, context: str) -> str:
        """Generate response using OpenAI API"""
        try:
            import openai
            
            openai.api_key = settings.OPENAI_API_KEY
            
            prompt = f"""You are an agricultural expert helping farmers. 
            
Based on the following context, provide a clear and helpful answer to the farmer's question.

Context:
{context}

Question: {query}

Answer:"""
            
            response = openai.ChatCompletion.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an agricultural expert helping farmers with their questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _generate_with_local(self, query: str, context: str) -> str:
        """Generate response using local summarization"""
        # Simple local response generation
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        
        if not context.strip():
            return "I couldn't find relevant information to answer your question. Please upload a PDF with agricultural information."
        
        # Extract key sentences from context
        sentences = context.split('. ')
        
        if len(sentences) > 3:
            # Return top 3 most relevant sentences
            response = '. '.join(sentences[:3]) + '.'
        else:
            response = context
        
        return response
    
    def generate_answer(self, query: str) -> dict:
        """
        Complete QA pipeline: retrieve + generate
        
        Args:
            query: User query
            
        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve relevant documents
        retrieved = self.retrieve_context(query)
        
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
