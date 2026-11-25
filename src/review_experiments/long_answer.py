"""CLAPNQ retrieval, long-context interventions and answer overlap scoring."""

import json
import re

import numpy as np
from huggingface_hub import hf_hub_download
from rouge_score.rouge_scorer import RougeScorer

from .common import Job, choose, clean_key, hub_revision, rng_for
from .config import QA_MAX_NEW

CLAPNQ_ID = "PrimeQA/clapnq"
QA_INSTRUCTION = (
    "Answer the question in several cohesive sentences and cover all requested parts. "
    "When passages are provided, ground your answer in them. "
    "If the provided passages cannot support an answer, return 'Insufficient evidence.'"
)
PASSAGE_SEPARATOR = "\n\n[Passage]\n"
ROUGE = RougeScorer(["rougeL", "rougeLsum"], use_stemmer=True)


def qa_data(split):
    revision = hub_revision(CLAPNQ_ID)
    path = hf_hub_download(
        CLAPNQ_ID, f"clapnq_{split}_answerable.jsonl",
        revision=revision, repo_type="dataset",
    )
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            annotations = [answer for answer in item["output"] if (answer.get("answer") or "").strip()]
            passages = [passage for passage in item["passages"] if (passage.get("text") or "").strip()]
            if not annotations or len(passages) != 1:
                continue
            passage = passages[0]
            rows.append({
                "id": str(item["id"]), "question": item["input"].strip(),
                "title": passage["title"].strip(),
                "passage": f'Title: {passage["title"].strip()}\n{passage["text"].strip()}',
                "answers": list(dict.fromkeys(answer["answer"].strip() for answer in annotations)),
                "support_sentences": [answer.get("selected_sentences", []) for answer in annotations],
            })
    return rows, revision


def qa_prompt(question, passages, demos=()):
    blocks = [QA_INSTRUCTION]
    for demo in demos:
        blocks.append(
            f"Example\n{PASSAGE_SEPARATOR.lstrip()}{demo['passage']}\n"
            f"Question: {demo['question']}\nAnswer: {demo['answers'][0]}"
        )
    if passages:
        blocks.append(PASSAGE_SEPARATOR.lstrip() + PASSAGE_SEPARATOR.join(passages))
    blocks.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(blocks)


def stress_contexts(codec, question, gold, distractors, targets):
    positions = ("front", "middle", "end")
    cumulative = np.cumsum([codec.count(PASSAGE_SEPARATOR + row["passage"]) for row in distractors])
    base = codec.count(codec.render(qa_prompt(question, [gold["passage"]]), "chat"))

    def arrangements(count):
        insertion = (0, count // 2, count)
        variants = []
        for position, slot in zip(positions, insertion, strict=True):
            ordered = distractors[:slot] + [gold] + distractors[slot:count]
            text = qa_prompt(question, [row["passage"] for row in ordered])
            variants.append((position, ordered, text, codec.count(codec.render(text, "chat"))))
        return variants

    for target in sorted(set(targets)):
        if target < base:
            continue
        count = int(np.searchsorted(cumulative, target - base, side="right"))
        variants = arrangements(count)
        while count and max(variant[3] for variant in variants) > target:
            count -= 1
            variants = arrangements(count)
        while count < len(distractors):
            trial = arrangements(count + 1)
            if max(variant[3] for variant in trial) > target:
                break
            count += 1
            variants = trial
        if count > 1:
            for position, ordered, text, actual in variants:
                yield target, position, ordered, text, actual


def qa_jobs(config, n_eval, retriever, codec):
    train, revision = qa_data("train")
    dev, dev_revision = qa_data("dev")
    if revision != dev_revision:
        raise ValueError("CLAPNQ train and dev revisions differ.")
    eligible = [
        row for row in dev
        if max(len(answer.split()) for answer in row["answers"]) >= config.min_qa_reference_words
    ]
    if not eligible:
        raise ValueError("No eligible CLAPNQ dev examples with the configured reference length.")
    cases = choose(eligible, n_eval, config.seed, "CLAPNQ/eligible dev")
    stress_cases = {row["id"] for row in choose(cases, config.stress_n, config.seed, "CLAPNQ/stress")}
    # Deduplicate passages and reserve dev titles and near-identical question keys from training use.
    dev_titles = {clean_key(row["title"]) for row in dev}
    dev_questions = {clean_key(row["question"]) for row in dev}
    dev_ids = {row["id"] for row in dev}
    train = [
        row for row in train
        if clean_key(row["title"]) not in dev_titles
        and clean_key(row["question"]) not in dev_questions
        and row["id"] not in dev_ids
    ]
    distractors = list({clean_key(row["title"]): row for row in reversed(train)}.values())
    if len(distractors) < config.qa_candidates - 1 or len(train) < config.qa_demo_k:
        raise ValueError("Too few distinct CLAPNQ training items for the requested study.")
    demo_vectors = retriever.encode([row["question"] for row in train]) if config.qa_demo_k else None

    for row in cases:
        sample_id, question = row["id"], row["question"]
        rng = rng_for(config.seed, "clapnq", sample_id, "candidate order")
        pool = rng.sample(distractors, config.qa_candidates - 1) + [row]
        rng.shuffle(pool)
        query_vector = retriever.encode([question])[0]
        texts = [entry["passage"] for entry in pool]
        ranked, scores = retriever.rank(retriever.encode(texts), query_vector)
        lengths = retriever.lengths(texts)
        gold_index = next(i for i, entry in enumerate(pool) if entry is row)
        random_index = rng.randrange(len(pool))
        demo_ranking = retriever.rank(demo_vectors, query_vector)[0] if config.qa_demo_k else []
        demos = [train[i] for i in demo_ranking[:config.qa_demo_k][::-1]]
        candidate_rows = [
            {"id": pool[i]["id"], "score": float(scores[i]), "retriever_tokens": lengths[i]}
            for i in ranked
        ]

        def job(condition, chosen, *, demos=(), custom_text=None, extras=None):
            passages = [entry["passage"] for entry in chosen]
            has_gold = any(entry is row for entry in chosen)
            return Job(
                "qa", "clapnq", f"clapnq_dev_{sample_id}", condition,
                custom_text or qa_prompt(question, passages, demos),
                question, row["answers"], QA_MAX_NEW,
                details={
                    "split": "dev_answerable", "source_revision": revision,
                    "reference_word_counts": [len(answer.split()) for answer in row["answers"]],
                    "gold_support_sentences": row["support_sentences"],
                    "gold_evidence_id": sample_id, "gold_evidence_title": row["title"],
                    "retrieved_passage_ids": [entry["id"] for entry in chosen],
                    "gold_passage_available": has_gold, "gold_rank": ranked.index(gold_index) + 1,
                    "gold_evidence_marker": row["passage"].split("\n", 1)[0] if has_gold else None,
                    "retriever_limit": retriever.limit, "demonstration_ids": [entry["id"] for entry in demos],
                    **(extras or {}),
                },
            )

        yield job("no_passage", [])
        yield job("minilm_passage", [pool[ranked[0]]], extras={"candidate_ranking": candidate_rows})
        yield job("random_passage", [pool[random_index]])
        yield job("gold_passage", [row])
        if demos:
            yield job(f"minilm_passage_icl_k{config.qa_demo_k}", [pool[ranked[0]]], demos=demos)
            yield job(f"gold_passage_icl_k{config.qa_demo_k}", [row], demos=demos)
        if sample_id in stress_cases:
            ordered = rng_for(config.seed, "clapnq", sample_id, "stress distractors").sample(distractors, len(distractors))
            clipped = [target for target in config.stress_prompt_targets if target <= config.max_prompt_tokens]
            for target, position, passages, text, actual in stress_contexts(codec, question, row, ordered, clipped):
                yield job(
                    f"stress_{target}_{position}", passages, custom_text=text,
                    extras={"target_prompt_tokens": target, "sizing_prompt_tokens": actual,
                            "evidence_position": position, "num_distractors": len(passages) - 1},
                )


def qa_score(references, output):
    def sentence_lines(text):
        return "\n".join(re.split(r"(?<=[.!?])\s+", text.strip()))

    scores = []
    for reference in references:
        l = ROUGE.score(reference, output)["rougeL"]
        lsum = ROUGE.score(sentence_lines(reference), sentence_lines(output))["rougeLsum"]
        scores.append({"rouge_l": dict(l._asdict()), "rouge_lsum": dict(lsum._asdict())})
    return {
        "prediction": output.strip(), "prediction_words": len(output.split()),
        "abstained": output.strip().rstrip(".").casefold() == "insufficient evidence",
        "reference_overlap": scores,
        "max_rouge_l_f1": max(score["rouge_l"]["fmeasure"] for score in scores),
        "max_rouge_lsum_f1": max(score["rouge_lsum"]["fmeasure"] for score in scores),
    }
