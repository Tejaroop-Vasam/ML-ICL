"""BGE-M3 embeddings, token lengths, and similarity ranking."""

import numpy as np

from src.review_experiments.common import hub_revision

BGE_M3_ID = "BAAI/bge-m3"


class Retriever:
    def __init__(self, device):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_id = BGE_M3_ID
        self.revision = hub_revision(self.model_id, "model")
        options = {"model_kwargs": {"torch_dtype": torch.float16}} if device.startswith("cuda") else {}
        self.model = SentenceTransformer(self.model_id, device=device, revision=self.revision, **options)
        self.limit = int(self.model.max_seq_length)

    def encode(self, texts):
        vectors = self.model.encode(
            texts, batch_size=4, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def lengths(self, texts):
        lengths = self.model.tokenizer(
            texts, truncation=False, add_special_tokens=True, return_length=True, verbose=False,
        )["length"]
        return list(map(int, lengths))

    @staticmethod
    def rank(matrix, query):
        scores = matrix @ query
        return np.argsort(-scores, kind="stable").tolist(), scores
