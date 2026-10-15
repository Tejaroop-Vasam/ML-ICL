"""Model registry and factory."""

from typing import Optional
from src.models.base_model import BaseLLM
from src.models.mock_model import MockLLM


def get_model(
    model_name: str = "mock",
    backend: str = "auto",
    device: Optional[str] = None,
    **kwargs,
) -> BaseLLM:
    """Instantiate LLM runner based on backend and model name.

    Args:
        model_name: Model identifier (e.g., 'meta-llama/Llama-3.2-1B-Instruct', 'mock')
        backend: 'hf' (HuggingFace local), 'api' (OpenAI/vLLM), 'mock', or 'auto'
    """
    if backend == "mock" or model_name.lower().startswith("mock"):
        return MockLLM(model_name=model_name, **kwargs)

    if backend == "api":
        from src.models.api_model import APIModel
        return APIModel(model_name=model_name, **kwargs)

    # Hugging Face local
    try:
        from src.models.hf_model import HuggingFaceLLM
        return HuggingFaceLLM(model_name=model_name, device=device, **kwargs)
    except Exception as e:
        if backend == "auto":
            import logging
            logging.warning(f"Could not load HuggingFace model '{model_name}': {e}. Falling back to MockLLM.")
            return MockLLM(model_name="mock-fallback")
        raise
