"""Dataset registry and factory functions."""

from typing import Dict, Type, List, Optional
from src.data.base_loader import BaseDataLoader
from src.data.sst2_loader import SST2DataLoader
from src.data.agnews_loader import AGNewsDataLoader
from src.data.trec_loader import TRECDataLoader
from src.data.imdb_loader import IMDbDataLoader

DATASET_REGISTRY: Dict[str, Type[BaseDataLoader]] = {
    "sst2": SST2DataLoader,
    "ag_news": AGNewsDataLoader,
    "trec": TRECDataLoader,
    "imdb": IMDbDataLoader,
}


def list_datasets() -> List[str]:
    """Return list of supported dataset identifiers."""
    return list(DATASET_REGISTRY.keys())


def get_dataset(
    dataset_name: str,
    cache_dir: Optional[str] = "data/processed",
    seed: int = 42,
) -> BaseDataLoader:
    """Factory function to get a dataset loader instance by name.

    Args:
        dataset_name: One of 'sst2', 'ag_news', 'trec', 'imdb'.
        cache_dir: Directory where processed JSONL splits are cached.
        seed: Random seed for reproducible splits and sampling.
    """
    key = dataset_name.lower().strip()
    if key not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available: {list_datasets()}"
        )
    loader_cls = DATASET_REGISTRY[key]
    return loader_cls(cache_dir=cache_dir, seed=seed)


def get_all_datasets(
    cache_dir: Optional[str] = "data/processed",
    seed: int = 42,
) -> Dict[str, BaseDataLoader]:
    """Instantiate loaders for all registered benchmark datasets."""
    return {
        name: get_dataset(name, cache_dir=cache_dir, seed=seed)
        for name in DATASET_REGISTRY
    }
