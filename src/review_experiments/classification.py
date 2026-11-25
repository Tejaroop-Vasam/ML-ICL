"""Classification prompt construction and paired IMDb diagnostics."""

import re
from functools import lru_cache

from datasets import load_dataset

from src.data import ICLExample, IMDbDataLoader, get_dataset
from src.label_parsing import extract_legacy_label

from .common import Job, clean_key, hub_revision, rng_for, round_robin_sample
from .config import CLASS_MAX_NEW

CLASS_SOURCES = {
    "imdb": "stanfordnlp/imdb",
    "yelp_polarity": "fancyzhx/yelp_polarity",
    "trec": "SetFit/TREC-QC",
    "yahoo_answers": "community-datasets/yahoo_answers_topics",
}
LABEL_FIELDS = {"imdb": "label", "yelp_polarity": "label", "trec": "label_coarse", "yahoo_answers": "topic"}


@lru_cache(maxsize=None)
def prompt_loader(dataset):
    return get_dataset(dataset)


def classification_prompt(dataset, query, demos=(), instruction=None, structured=False):
    example = ICLExample(id="query", dataset=dataset, split="test", text=query, label=-1, label_name="")
    return prompt_loader(dataset).format_prompt(
        example, demonstrations=demos, format_type="structured" if structured else "plain",
        custom_instruction=instruction,
    )


def classification_data(config, dataset, n_eval):
    loader = prompt_loader(dataset)
    label_field = LABEL_FIELDS[dataset]
    repo = CLASS_SOURCES[dataset]
    revision = hub_revision(repo)
    data = load_dataset(repo, revision=revision)

    def example(split, idx, row):
        if dataset == "yahoo_answers":
            title = (row["question_title"] or "").strip()
            content = (row["question_content"] or "").strip()
            answer = (row["best_answer"] or "").strip()
            parts = [title, content if content != title else "", f"Answer: {answer}" if answer else ""]
            text = " ".join(part for part in parts if part)
        else:
            text = row["text"].strip()
        if dataset == "imdb":
            text = IMDbDataLoader._clean_review_text(text)
        label = int(row[label_field])
        return ICLExample(
            id=f"{dataset}_{split}_{idx}", dataset=dataset, split=split,
            text=text, label=label, label_name=loader.classes[label],
        )

    # Select by the label column first, so very large train splits are not materialized as Python text rows.
    def subset(split, limit):
        table = data[split]
        indexed = list(enumerate(table[label_field]))
        indices = round_robin_sample(indexed, limit, lambda pair: pair[1], config.seed, f"{dataset}/{split}")
        ids = [idx for idx, _ in indices]
        selected = table.select(ids)
        return [example(split, idx, row) for idx, row in zip(ids, selected, strict=True)]

    queries = subset("test", n_eval)
    # Oversample the small pool to permit deduplication and query-specific exact-text exclusions.
    train = subset("train", min(config.pool_size * 2, len(data["train"])))
    unique_train = list({clean_key(row.text): row for row in reversed(train) if row.text}.values())
    pool = round_robin_sample(unique_train, config.pool_size, lambda row: row.label_name, config.seed, dataset + "/pool")
    return queries, pool, revision


def label_paths(output, labels):
    clean = output.strip()
    stripped = clean.strip("`*\"'").rstrip(".").strip("`*\"' ")
    exact = next((label for label in labels if stripped.casefold() == label.casefold()), None)
    patterns = {label: rf"\b{re.escape(label)}\b" for label in labels}
    mentions = [label for label in labels if re.search(patterns[label], clean, re.I)]
    legacy, route = extract_legacy_label(output, labels)
    return {
        "strict_prediction": exact,
        "unique_mention_prediction": mentions[0] if len(mentions) == 1 else None,
        "mentions": mentions,
        "legacy_prediction": legacy,
        "legacy_parser_route": route,
        "parse_status": ("exact" if exact else "unique_mention" if len(mentions) == 1
                         else "ambiguous" if mentions else "no_label" if clean else "empty"),
    }


def class_jobs(config, dataset, n_eval, retriever, codec):
    queries, pool, revision = classification_data(config, dataset, n_eval)
    loader = prompt_loader(dataset)
    labels = tuple(loader.classes)
    pool_text = [row.text for row in pool]
    pool_vectors = retriever.encode(pool_text)
    pool_lengths = retriever.lengths(pool_text)
    query_batch_size = 128
    max_k = max((config.reference_k, *config.k_sweep))
    for start in range(0, len(queries), query_batch_size):
        batch = queries[start:start + query_batch_size]
        query_texts = [row.text for row in batch]
        vectors = retriever.encode(query_texts)
        lengths = retriever.lengths(query_texts)

        for row, query_vector, query_length in zip(batch, vectors, lengths, strict=True):
            query, gold, sample_id = row.text, row.label_name, row.id
            ranked, scores = retriever.rank(pool_vectors, query_vector)
            key = clean_key(query)
            available = [i for i in ranked if clean_key(pool[i].text) != key]
            if len(available) < max_k:
                raise ValueError(f"Too few distinct training demonstrations for {sample_id}")

            def job(condition, chosen=(), *, mode="chat", instruction=None, limit=CLASS_MAX_NEW,
                    structured=False, score=False, extras=None, query_view=None):
                visible_query = query if query_view is None else query_view
                demos = [pool[i] for i in chosen]
                return Job(
                    "classification", dataset, sample_id, condition,
                    classification_prompt(dataset, visible_query, demos, instruction, structured),
                    visible_query, gold, limit, mode=mode, labels=labels, score_only=score,
                    details={
                        "source_revision": revision, "split": "test",
                        "retriever_query_tokens": query_length, "retriever_limit": retriever.limit,
                        "retriever_query_truncated": query_length > retriever.limit,
                        "demonstrations": [
                            {"id": pool[i].id, "label": pool[i].label_name, "score": float(scores[i]),
                             "retriever_tokens": pool_lengths[i], "retriever_truncated": pool_lengths[i] > retriever.limit}
                            for i in chosen
                        ],
                        **(extras or {}),
                    },
                )

            yield job("legacy_raw_16", mode="raw")
            yield job("chat_same_instruction_16")
            strict = loader.task_instruction + "\nReturn only the exact label."
            yield job("chat_label_only_16", instruction=strict)
            if dataset == "imdb":
                yield job("legacy_raw_64", mode="raw", limit=64)
                yield job("chat_label_only_scores", instruction=strict, score=True)
                reversed_instruction = "Classify the sentiment of each movie review as either 'negative' or 'positive'."
                yield job("chat_label_only_reversed_scores",
                          instruction=reversed_instruction + "\nReturn only the exact label.", score=True)
                review_tokens = codec.encode(query) if config.imdb_slice_tokens else []
                if len(review_tokens) > config.imdb_slice_tokens:
                    views = {"head": review_tokens[:config.imdb_slice_tokens], "tail": review_tokens[-config.imdb_slice_tokens:]}
                    for view, token_ids in views.items():
                        text = codec.tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
                        yield job(f"chat_label_only_{view}", instruction=strict, query_view=text,
                                  extras={"review_view": view, "full_review_tokens": len(review_tokens)})

            for k in sorted(set(config.k_sweep) | {config.reference_k}):
                selected = available[:k]
                yield job(f"bge_m3_k{k}", selected[::-1], instruction=strict)
            for seed in config.random_seeds:
                chosen = rng_for(config.seed, dataset, sample_id, "random", seed).sample(available, config.reference_k)
                yield job(f"random_k{config.reference_k}_seed{seed}", chosen, instruction=strict,
                          extras={"random_seed": seed})

            # Equal-quota retrieval uses training labels. Rotate remainder without the held-out label.
            label_order = list(labels)
            rng_for(config.seed, dataset, sample_id, "class order").shuffle(label_order)
            by_label = {label: [i for i in available if pool[i].label_name == label] for label in label_order}
            balanced = []
            while len(balanced) < config.reference_k:
                for candidates in by_label.values():
                    if candidates:
                        balanced.append(candidates.pop(0))
                        if len(balanced) == config.reference_k:
                            break
            yield job(f"balanced_bge_m3_k{config.reference_k}", sorted(balanced, key=lambda i: scores[i]),
                      instruction=strict)
            if dataset == "imdb":
                oracle = [i for i in available if pool[i].label_name == gold][:config.reference_k]
                if len(oracle) == config.reference_k:
                    yield job(f"label_informed_oracle_k{config.reference_k}", oracle[::-1], instruction=strict,
                              extras={"uses_query_gold": True})
                chosen = available[:config.reference_k]
                yield job(f"bge_m3_similar_first_k{config.reference_k}", chosen, instruction=strict)
                yield job(f"bge_m3_structured_k{config.reference_k}", chosen[::-1], instruction=strict, structured=True)
