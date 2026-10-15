"""Data loading package for In-Context Learning benchmarks."""

from src.data.example import ICLExample
from src.data.base_loader import BaseDataLoader
from src.data.sst2_loader import SST2DataLoader
from src.data.agnews_loader import AGNewsDataLoader
from src.data.trec_loader import TRECDataLoader
from src.data.imdb_loader import IMDbDataLoader
from src.data.registry import get_dataset, list_datasets, get_all_datasets

__all__ = [
    "ICLExample",
    "BaseDataLoader",
    "SST2DataLoader",
    "AGNewsDataLoader",
    "TRECDataLoader",
    "IMDbDataLoader",
    "get_dataset",
    "list_datasets",
    "get_all_datasets",
]
