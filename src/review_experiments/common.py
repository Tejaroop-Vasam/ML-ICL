"""Shared records, dataset revisions, and deterministic sampling."""

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import chain, islice, zip_longest
from pathlib import Path


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


def choose(rows, n, seed, name):
    if n is None or n >= len(rows):
        return list(rows)
    return rng_for(seed, name).sample(rows, n)


def take_round_robin(groups, n):
    """Take one item from each group per round, skipping exhausted groups."""
    missing = object()
    rounds = zip_longest(*groups, fillvalue=missing)
    items = (row for row in chain.from_iterable(rounds) if row is not missing)
    return list(islice(items, max(0, n)))


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
    return take_round_robin((reversed(groups[key]) for key in order), n)
