"""HuggingFace local causal LM runner with MPS / CUDA acceleration."""

import logging
from typing import Optional
import torch

from src.models.base_model import BaseLLM

logger = logging.getLogger(__name__)


class HuggingFaceLLM(BaseLLM):
    """Local Hugging Face model runner supporting Llama-3-8B, Llama-3.2-1B, and other causal LMs."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        device: Optional[str] = None,
        torch_dtype: Optional[str] = "auto",
        trust_remote_code: bool = True,
        load_in_8bit: bool = False,
    ):
        super().__init__(model_name)
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Loading HuggingFace model '{model_name}' on device '{self.device}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"trust_remote_code": trust_remote_code}
        if torch_dtype == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif torch_dtype == "float16":
            model_kwargs["torch_dtype"] = torch.float16

        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
        if not load_in_8bit and self.device != "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info(f"Successfully loaded '{model_name}'.")

    def count_tokens(self, text: str) -> int:
        """Count exact tokens using tokenizer."""
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        return len(encoded)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """Generate text from prompt with greedy or sampling decoding."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = outputs[0][prompt_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
