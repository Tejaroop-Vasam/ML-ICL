"""Causal LM runner loaded dynamically from Hugging Face Hub (Apple Silicon MPS / CUDA / CPU)."""

import logging
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.models.base_model import BaseLLM

logger = logging.getLogger(__name__)


class LocalLLM(BaseLLM):
    """Model runner loading directly from Hugging Face Hub on the fly."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(model_name)

        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if dtype is None:
            # float16 on any GPU (MPS or CUDA) for speed and memory efficiency
            dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32

        logger.info(f"Loading '{model_name}' on device '{self.device}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",   # HF automatically places layers on GPU/CPU as needed
        )
        # device_map="auto" handles placement — no manual .to(device) needed

        self.model.eval()
        logger.info(f"Successfully loaded '{model_name}'.")

    def count_tokens(self, text: str) -> int:
        """Count exact tokens using the local model's tokenizer."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """Generate response tokens locally."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": temperature > 0.0,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = outputs[0][prompt_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
