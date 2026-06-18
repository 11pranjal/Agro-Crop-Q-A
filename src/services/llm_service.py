"""Local and remote LLM service abstraction."""
import json
import urllib.request
import urllib.error
from typing import Optional

from src.core.config import settings


class LLMService:
    """Service for calling OpenAI, Ollama, or local fallback."""

    def __init__(self):
        self.openai_key = settings.OPENAI_API_KEY
        self.openai_model = settings.OPENAI_MODEL
        self.ollama_url = settings.OLLAMA_URL.rstrip("/") if settings.OLLAMA_URL else ""
        self.local_llm_model = settings.LOCAL_LLM_MODEL

    def can_generate(self) -> bool:
        return bool(self.openai_key or self.ollama_url)

    def generate_with_context(self, query: str, context: str) -> Optional[str]:
        # Prioritize Ollama (free, local) over OpenAI
        if self.ollama_url:
            return self._generate_with_ollama_with_context(query, context)
        if self.openai_key:
            return self._generate_with_openai(query, context)
        return None

    def generate_general(self, query: str) -> Optional[str]:
        # Prioritize Ollama (free, local) over OpenAI
        if self.ollama_url:
            return self._generate_general_ollama_response(query)
        if self.openai_key:
            return self._generate_general_openai_response(query)
        return None

    def _generate_with_openai(self, query: str, context: str) -> Optional[str]:
        try:
            import openai

            client = openai.OpenAI(api_key=self.openai_key)
            system_prompt = (
                "You are an agricultural expert helping farmers. "
                "Use ONLY the context provided from the uploaded PDF to answer. "
                "Do not invent facts or reference outside knowledge. "
                "If the answer is not in the context, reply exactly: 'I couldn't find relevant information to answer your question.'"
            )
            user_prompt = (
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer concisely and return the response in Q&A format exactly like this:\n"
                "Question: <repeat the question>\nAnswer: <one to three sentence concise answer>\n"
            )
            response = client.chat.completions.create(
                model=self.openai_model,
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
            print(f"OpenAI context generation error: {e}")
            return None

    def _generate_general_openai_response(self, query: str) -> Optional[str]:
        try:
            import openai

            client = openai.OpenAI(api_key=self.openai_key)
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
            response = client.chat.completions.create(
                model=self.openai_model,
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
            print(f"OpenAI general generation error: {e}")
            return None

    def _generate_with_ollama_with_context(self, query: str, context: str) -> Optional[str]:
        prompt = (
            "You are an agricultural expert helping farmers with practical knowledge.\n\n"
            "Using ONLY the provided context, answer the user's question.\n"
            "Requirements:\n"
            "- Write a natural, clear answer (1-3 sentences).\n"
            "- Summarize the information, don't copy verbatim.\n"
            "- Convert lists into readable bullet points if applicable.\n"
            "- Ignore page numbers, document titles, and section headings.\n"
            "- If the answer is not in the context, say: 'I couldn't find relevant information to answer your question.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        return self._ollama_generate(prompt, temperature=0.2, max_tokens=300)

    def _generate_general_ollama_response(self, query: str) -> Optional[str]:
        prompt = (
            "You are a helpful agriculture assistant.\n\n"
            "Answer the user's question clearly and helpfully.\n"
            "If it's about agriculture, provide practical advice.\n"
            "If it's a general question, answer naturally and friendly.\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        return self._ollama_generate(prompt, temperature=0.5, max_tokens=300)

    def _ollama_generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 300) -> Optional[str]:
        if not self.ollama_url:
            return None

        body = {
            "model": f"{self.local_llm_model}:latest",
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": False
        }
        try:
            request = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
                result = payload.get("response", "").strip()
                if result:
                    return result
                return None
        except urllib.error.HTTPError as e:
            print(f"Ollama HTTP error: {e.code} {e.reason}")
            if e.code == 404:
                print(f"Model '{self.local_llm_model}' not found. Install with: ollama pull {self.local_llm_model}")
        except urllib.error.URLError as e:
            print(f"Ollama connection error: {e.reason}")
            print(f"Make sure Ollama is running at {self.ollama_url}")
        except Exception as e:
            print(f"Ollama generation error: {e}")
        return None

    def _extract_ollama_text(self, payload: dict) -> Optional[str]:
        """Legacy method - kept for backward compatibility"""
        if not payload:
            return None
        return payload.get("response", "").strip() or None
