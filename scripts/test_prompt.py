#!/usr/bin/env python3
"""Minimal example to run a prompt through a downloaded Qwen3 model."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# You can change this to Qwen/Qwen3-1.7B, Qwen/Qwen3-4B, etc.
# MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_NAME = "Qwen/Qwen3-1.7B"

# Choose device: Mac Apple Silicon GPU ("mps") if available, else "cpu"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    print(f"1. Loading tokenizer and model '{MODEL_NAME}' on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "mps" else torch.float32,
    ).to(DEVICE)

    # 2. Define your prompt
    prompt = "Give me java code for binary search"
    print(f"\n2. Prompt:\n{prompt}")

    # 3. Convert text into input token IDs
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # 4. Generate new tokens
    print("\n3. Generating response...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,  # greedy decoding (deterministic)
        )
    # print(output_ids.length)

    # print(output_ids[0])
    # print(output_ids[1])
    # 5. Convert token IDs back to human-readable text
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("\n4. Model Output:")
    print("-" * 40)
    print(full_output)
    print("-" * 40)


if __name__ == "__main__":
    main()
