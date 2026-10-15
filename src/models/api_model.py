"""OpenAI-compatible / vLLM API model runner."""

import os
import json
import logging
import urllib.request
from typing import Optional

from src.models.base_model import BaseLLM

logger = logging.getLogger(__name__)


class APIModel(BaseLLM):
    """Client for OpenAI-compatible inference servers (vLLM, Ollama, OpenAI, etc.)."""

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None,
    ):
        super().__init__(model_name)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")

    def count_tokens(self, text: str) -> int:
        """Approximation when tokenizer is not hosted locally."""
        words = text.split()
        return max(1, int(len(words) * 1.3))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """Send completion request to API server."""
        url = f"{self.base_url}/completions"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "stop": ["\n\n", "Review:", "Input:", "Question:", "Article:"],
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
                return resp_json["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"API generation error: {e}")
            raise
