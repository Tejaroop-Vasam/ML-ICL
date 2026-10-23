#!/usr/bin/env python3
"""Minimal script to download Qwen3 models locally via HuggingFace Hub."""

from huggingface_hub import snapshot_download

MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]
def download_models():
    for model_id in MODELS:
        print(f"Downloading {model_id}...")
        path = snapshot_download(repo_id=model_id)
        print(f"Successfully downloaded {model_id} to:\n  {path}\n")


if __name__ == "__main__":
    download_models()
