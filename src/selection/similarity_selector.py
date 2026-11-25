"""Similarity-based demonstration selection using dense sentence embeddings."""

import logging
import hashlib
import json
from pathlib import Path
from typing import List, Optional
import numpy as np

from src.data.example import ICLExample
from src.retrieval import BGE_M3_ID, dense_embeddings, load_dense_encoder
from src.selection.base_selector import BaseSelector
from src.selection.ordering import order_demonstrations

logger = logging.getLogger(__name__)


class SimilaritySelector(BaseSelector):
    """Retrieves k demonstrations with highest semantic cosine similarity to test query."""

    def __init__(
        self,
        candidate_pool: List[ICLExample],
        model_name: str = BGE_M3_ID,
        cache_dir: Optional[str] = "data/cache/embeddings",
        seed: int = 42,
    ):
        super().__init__(candidate_pool, seed)
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None
        self._candidate_embeddings: Optional[np.ndarray] = None
        self._query_cache: dict = {}
        self._pool_indices = {(item.id, item.text): index for index, item in enumerate(candidate_pool)}

    def _get_model(self):
        if self._model is None:
            self._model = load_dense_encoder(self.model_name)
        return self._model

    def _query_embedding(self, example):
        key = (example.id, example.text)
        if key not in self._query_cache:
            self._query_cache[key] = dense_embeddings(self._get_model(), [example.text])[0]
        return self._query_cache[key]

    def get_candidate_embeddings(self) -> np.ndarray:
        """Compute or load cached normalized embeddings for all candidates."""
        if self._candidate_embeddings is not None:
            return self._candidate_embeddings

        # Check cache if available
        cache_file = None
        if self.cache_dir and self.candidate_pool:
            dataset_name = self.candidate_pool[0].dataset
            clean_model = self.model_name.replace("/", "_")
            pool_data = json.dumps([(item.id, item.text) for item in self.candidate_pool])
            fingerprint = hashlib.sha256(pool_data.encode("utf-8")).hexdigest()[:16]
            pool_hash = f"{dataset_name}_{clean_model}_{fingerprint}.npy"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / pool_hash

            if cache_file.exists():
                try:
                    logger.info(f"Loading cached embeddings from {cache_file}...")
                    cached = np.load(cache_file)
                    if cached.ndim == 2 and len(cached) == len(self.candidate_pool):
                        self._candidate_embeddings = cached
                        return cached
                except Exception as e:
                    logger.warning(f"Failed to load cached embeddings: {e}. Recomputing.")

        # Compute embeddings
        model = self._get_model()
        texts = [c.text for c in self.candidate_pool]
        logger.info(f"Encoding {len(texts)} candidate demonstrations with {self.model_name}...")
        embeddings = dense_embeddings(model, texts, show_progress=len(texts) > 200)
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

        query_emb = self._query_embedding(test_example)

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

    def compute_similarity_scores(self, test_example: ICLExample, candidates: List[ICLExample]) -> np.ndarray:
        """Compute cosine similarity scores between test example and a candidate subset."""
        if not candidates:
            return np.array([])
        query_emb = self._query_embedding(test_example)
        candidate_keys = [(item.id, item.text) for item in candidates]
        if all(key in self._pool_indices for key in candidate_keys):
            cand_embs = self.get_candidate_embeddings()[[self._pool_indices[key] for key in candidate_keys]]
        else:
            cand_embs = dense_embeddings(self._get_model(), [item.text for item in candidates])
        return np.dot(cand_embs, query_emb)

    def compute_similarity_score(self, demo: ICLExample, test_example: ICLExample) -> float:
        """Compute cosine similarity between a single demo and test example."""
        return float(self.compute_similarity_scores(test_example, [demo])[0])
