"""Shared records, deterministic sampling and BGE-M3 retrieval."""

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.retrieval import BGE_M3_ID, dense_embeddings, load_dense_encoder


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def rng_for(seed, *parts):
    return random.Random(int(digest(json_text([seed, *parts]))[:16], 16))


def file_sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def clean_key(text):
    return " ".join(text.split()).casefold()


@lru_cache(maxsize=None)
def hub_revision(repo_id, kind="dataset"):
    from huggingface_hub import HfApi

    info = HfApi().repo_info(repo_id, repo_type=kind)
    if not info.sha:
        raise RuntimeError(f"Missing pinned revision for {repo_id}")
    return info.sha


@dataclass
class Job:
    task: str
    dataset: str
    sample_id: str
    condition: str
    text: str
    query: str
    reference: object
    max_new: int
    mode: str = "chat"
    details: dict = field(default_factory=dict)
    labels: tuple = ()
    db_path: Path | None = None
    score_only: bool = False


class Retriever:
    def __init__(self, device):
        self.model_id = BGE_M3_ID
        self.revision = hub_revision(self.model_id, "model")
        self.model = load_dense_encoder(self.model_id, device=device, revision=self.revision)
        self.limit = int(self.model.max_seq_length)

    def encode(self, texts):
        return dense_embeddings(self.model, texts)

    def lengths(self, texts):
        lengths = self.model.tokenizer(
            texts, truncation=False, add_special_tokens=True, return_length=True,
            verbose=False,
        )["length"]
        return [int(n) for n in lengths]

    @staticmethod
    def rank(matrix, query):
        scores = matrix @ query
        return np.argsort(-scores, kind="stable").tolist(), scores


def choose(rows, n, seed, name):
    if n is None or n >= len(rows):
        return list(rows)
    return rng_for(seed, name).sample(rows, n)


def round_robin_sample(rows, n, group, seed, name):
    """Draw seeded, approximately balanced groups, refilling when groups run out."""
    if n is None or n >= len(rows):
        return list(rows)
    rng = rng_for(seed, name)
    groups = defaultdict(list)
    for row in rows:
        groups[group(row)].append(row)
    order = sorted(groups)
    rng.shuffle(order)
    for values in groups.values():
        rng.shuffle(values)
    sample = []
    while len(sample) < n:
        for key in order:
            if groups[key]:
                sample.append(groups[key].pop())
                if len(sample) == n:
                    break
    return sample
