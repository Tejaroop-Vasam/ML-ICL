"""Similarity-based demonstration selection using dense sentence embeddings."""

import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from src.data.example import ICLExample
from src.selection.base_selector import BaseSelector
from src.selection.ordering import order_demonstrations

logger = logging.getLogger(__name__)


class SimilaritySelector(BaseSelector):
    """Retrieves k demonstrations with highest semantic cosine similarity to test query."""

    def __init__(
        self,
        candidate_pool: List[ICLExample],
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = "data/cache/embeddings",
        seed: int = 42,
    ):
        super().__init__(candidate_pool, seed)
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None
        self._candidate_embeddings: Optional[np.ndarray] = None
        self._query_cache: dict = {}

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            logger.info(f"Loading SentenceTransformer '{self.model_name}' on device '{device}'...")
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def get_candidate_embeddings(self) -> np.ndarray:
        """Compute or load cached normalized embeddings for all candidates."""
        if self._candidate_embeddings is not None:
            return self._candidate_embeddings

        # Check cache if available
        cache_file = None
        if self.cache_dir and self.candidate_pool:
            dataset_name = self.candidate_pool[0].dataset
            clean_model = self.model_name.replace("/", "_")
            pool_hash = f"{dataset_name}_{len(self.candidate_pool)}_{clean_model}.npy"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / pool_hash

            if cache_file.exists():
                try:
                    logger.info(f"Loading cached embeddings from {cache_file}...")
                    self._candidate_embeddings = np.load(cache_file)
                    if len(self._candidate_embeddings) == len(self.candidate_pool):
                        return self._candidate_embeddings
                except Exception as e:
                    logger.warning(f"Failed to load cached embeddings: {e}. Recomputing.")

        # Compute embeddings
        model = self._get_model()
        texts = [c.text for c in self.candidate_pool]
        logger.info(f"Encoding {len(texts)} candidate demonstrations with {self.model_name}...")
        embeddings = model.encode(
            texts,
            show_progress_bar=len(texts) > 200,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        self._candidate_embeddings = embeddings

        if cache_file:
            np.save(cache_file, embeddings)
            logger.info(f"Cached {len(embeddings)} embeddings to {cache_file}")

        return self._candidate_embeddings

    def select_demonstrations(
        self,
        test_example: ICLExample,
        k: int,
        ordering: str = "similar_last",
        **kwargs,
    ) -> List[ICLExample]:
        """Retrieve top-k most semantically similar demonstrations.

        Args:
            test_example: Query sample.
            k: Number of demonstrations to retrieve.
            ordering: Ordering strategy ('similar_last', 'similar_first', 'random', etc.).
        """
        if k <= 0 or not self.candidate_pool:
            return []

        if test_example.id in self._query_cache:
            query_emb = self._query_cache[test_example.id]
        else:
            model = self._get_model()
            query_emb = model.encode(
                [test_example.text],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0]
            self._query_cache[test_example.id] = query_emb

        candidate_embs = self.get_candidate_embeddings()
        # Cosine similarity is simple dot product since vectors are L2-normalized
        sim_scores = np.dot(candidate_embs, query_emb)

        # Exclude exact test query match if in candidate pool
        for idx, c in enumerate(self.candidate_pool):
            if c.id == test_example.id:
                sim_scores[idx] = -1e9

        top_indices = np.argsort(sim_scores)[::-1][:k]
        selected_demos = [self.candidate_pool[i] for i in top_indices]
        selected_scores = [float(sim_scores[i]) for i in top_indices]

        # Apply requested ordering
        return order_demonstrations(
            demonstrations=selected_demos,
            ordering=ordering,
            similarity_scores=selected_scores,
            seed=self.seed,
        )

    def compute_similarity_score(self, demo: ICLExample, test_example: ICLExample) -> float:
        """Compute cosine similarity between a single demo and test example."""
        model = self._get_model()
        embs = model.encode(
            [demo.text, test_example.text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return float(np.dot(embs[0], embs[1]))
