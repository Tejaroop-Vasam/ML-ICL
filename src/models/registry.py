"""Hugging Face Hub model registry and factory."""

from typing import Optional
from src.models.base_model import BaseLLM
from src.models.local_model import LocalLLM

LOCAL_MODELS = {
    "0.6b": "Qwen/Qwen3-0.6B",
    "1.7b": "Qwen/Qwen3-1.7B",
    "4b": "Qwen/Qwen3-4B",
    "8b": "Qwen/Qwen3-8B",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "default": "Qwen/Qwen3-0.6B",
}


def get_model(
    model_name: str = "0.6b",
    device: Optional[str] = None,
    **kwargs,
) -> BaseLLM:
    """Instantiate an LLM runner loaded directly from Hugging Face Hub.

    Args:
        model_name: Model identifier or alias ('0.6b', '1.7b', '4b', '8b', or repo id)
        device: 'mps', 'cuda', or 'cpu' (auto-detected if None)
    """
    clean_name = model_name.strip()
    resolved_name = LOCAL_MODELS.get(clean_name.lower(), clean_name)

    return LocalLLM(model_name=resolved_name, device=device, **kwargs)
