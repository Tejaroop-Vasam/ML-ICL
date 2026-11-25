"""Shared dense-embedding configuration for the experiment runners."""

from functools import lru_cache

import numpy as np

BGE_M3_ID = "BAAI/bge-m3"


@lru_cache(maxsize=1)
def load_dense_encoder(model_id=BGE_M3_ID, *, device=None, revision=None):
    import torch
    from sentence_transformers import SentenceTransformer

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    options = {"model_kwargs": {"torch_dtype": torch.float16}} if device.startswith("cuda") else {}
    return SentenceTransformer(model_id, device=device, revision=revision, **options)


def dense_embeddings(encoder, texts, *, show_progress=False):
    """Return normalized vectors for cosine ranking with a modest GPU batch size."""
    vectors = encoder.encode(
        texts, batch_size=4, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=show_progress,
    )
    return np.asarray(vectors, dtype=np.float32)
