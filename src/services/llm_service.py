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
        if self.openai_key:
            return self._generate_with_openai(query, context)
        if self.ollama_url:
            return self._generate_with_ollama_with_context(query, context)
        return None

    def generate_general(self, query: str) -> Optional[str]:
        if self.openai_key:
            return self._generate_general_openai_response(query)
        if self.ollama_url:
            return self._generate_general_ollama_response(query)
        return None

    def _generate_with_openai(self, query: str, context: str) -> Optional[str]:
        try:
            import openai

            client = openai.OpenAI(api_key=self.openai_key)
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
            "You are an agricultural expert helping farmers. "
            "Answer using only the context provided from the uploaded PDF. "
            "Do not invent facts, do not reference outside knowledge, and do not search the web. "
            "Do not repeat the context verbatim. Instead, summarize the answer naturally in your own words. "
            "If the answer is not contained in the PDF context, reply: "
            "I couldn't find relevant information to answer your question.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\n"
            "Answer concisely, directly, and in a way a farmer can understand."
        )
        return self._ollama_generate(prompt, temperature=0.0, max_tokens=250)

    def _generate_general_ollama_response(self, query: str) -> Optional[str]:
        prompt = (
            "You are a helpful agriculture assistant. "
            "If the user asks about agriculture, answer clearly and simply. "
            "If the user asks a general or casual question, answer naturally like a friendly chat assistant. "
            "Do not mention that you need a PDF if the question is general. "
            "Keep the response polite and easy to understand.\n\n"
            f"Question: {query}\n\n"
            "Answer naturally and helpfully."
        )
        return self._ollama_generate(prompt, temperature=0.7, max_tokens=250)

    def _ollama_generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 250) -> Optional[str]:
        if not self.ollama_url:
            return None

        body = {
            "model": self.local_llm_model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_output_tokens": max_tokens
        }
        try:
            request = urllib.request.Request(
                f"{self.ollama_url}/v1/generate",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
                return self._extract_ollama_text(payload)
        except urllib.error.HTTPError as e:
            print(f"Ollama HTTP error: {e.code} {e.reason}")
        except urllib.error.URLError as e:
            print(f"Ollama URL error: {e.reason}")
        except Exception as e:
            print(f"Ollama generation error: {e}")
        return None

    def _extract_ollama_text(self, payload: dict) -> Optional[str]:
        if not payload:
            return None

        if "results" in payload and isinstance(payload["results"], list):
            texts = []
            for result in payload["results"]:
                if isinstance(result, dict):
                    if "output" in result:
                        output = result["output"]
                        if isinstance(output, list):
                            for entry in output:
                                if isinstance(entry, dict) and "content" in entry:
                                    texts.append(entry["content"])
                    elif "content" in result:
                        texts.append(result["content"])
            return "".join(texts).strip() if texts else None

        if "output" in payload:
            output = payload["output"]
            if isinstance(output, list):
                return "".join(str(entry.get("content", "")) for entry in output if isinstance(entry, dict)).strip()

        if "choices" in payload and isinstance(payload["choices"], list):
            for choice in payload["choices"]:
                if isinstance(choice, dict):
                    message = choice.get("message")
                    if isinstance(message, dict) and "content" in message:
                        return message["content"].strip()
                    if "text" in choice:
                        return choice["text"].strip()
        return None
