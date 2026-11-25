"""Qwen3 GPU inference and exact per-trial prompt accounting."""

import gc
import importlib.metadata
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .artifacts import Checkpoint
from .classification import CLASS_SOURCES, class_jobs, classification_prompt, label_paths
from .common import digest, hub_revision, json_text
from .config import CLASS_DATASETS, CLASS_MAX_NEW, NATIVE_WINDOW, QA_MAX_NEW, SQL_DATASETS, SQL_MAX_NEW, WEIGHT_MODES
from .long_answer import CLAPNQ_ID, qa_jobs, qa_score
from .text_to_sql import SQL_ROW_LIMIT, SQL_TIMEOUT, score_sql, sql_jobs


class QwenInput:
    def __init__(self, model_id, revision):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
        if not self.tokenizer.is_fast or not self.tokenizer.chat_template:
            raise ValueError("A fast Qwen3 tokenizer with the official chat template is required.")

    def render(self, text, mode):
        if mode == "raw":
            return text
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )

    def encode(self, rendered, mode="chat"):
        return self.tokenizer.encode(
            rendered, add_special_tokens=mode == "raw", truncation=False, verbose=False,
        )

    def count(self, text, mode="chat"):
        return len(self.encode(text, mode))

    def evidence_start(self, rendered, marker):
        start = rendered.find(marker)
        if start < 0:
            raise ValueError("Gold evidence is missing from the rendered prompt.")
        offsets = self.tokenizer(
            rendered, return_offsets_mapping=True, add_special_tokens=False,
            truncation=False, verbose=False,
        )["offset_mapping"]
        return next(index for index, (_, end) in enumerate(offsets) if end > start)


class QwenRunner:
    def __init__(self, model_id, revision, codec, weights):
        self.codec = codec
        self.priors = {}
        if weights not in WEIGHT_MODES:
            raise ValueError(f"Unsupported load mode: {weights}")
        if weights == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("Choose fp16 or nf4 on this GPU.")
        dtype = torch.bfloat16 if weights == "bf16" else torch.float16
        options = {
            "revision": revision, "dtype": dtype,
            "device_map": {"": 0}, "attn_implementation": "sdpa",
        }
        if weights == "nf4":
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **options).eval()
        self.device = self.model.get_input_embeddings().weight.device

    @torch.inference_mode()
    def generate(self, ids, limit, mode):
        tokenizer = self.codec.tokenizer
        tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        eos = tokenizer.eos_token_id if mode == "raw" else self.model.generation_config.eos_token_id
        if eos is None:
            eos = tokenizer.eos_token_id
        output = self.model.generate(
            input_ids=tensor, attention_mask=torch.ones_like(tensor), max_new_tokens=limit,
            do_sample=False, temperature=None, top_p=None, top_k=None, min_p=None,
            eos_token_id=eos, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        suffix = output[0, len(ids):].tolist()
        stops = eos if isinstance(eos, (list, tuple)) else [eos]
        ended = bool(suffix and suffix[-1] in stops)
        return {
            "raw_output": tokenizer.decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            "output_token_ids": suffix, "completion_tokens": len(suffix),
            "stop_reason": "eos" if ended else "generation_cap" if len(suffix) == limit else "other",
        }

    @torch.inference_mode()
    def log_scores(self, ids, labels):
        values = {}
        for label in labels:
            suffix = self.codec.encode(label)
            if not suffix:
                raise ValueError(f"Label has no tokens: {label!r}")
            tensor = torch.tensor([ids + suffix], dtype=torch.long, device=self.device)
            logits = self.model(input_ids=tensor, use_cache=False, logits_to_keep=len(suffix) + 1).logits
            log_probs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
            total = log_probs.gather(1, tensor[0, -len(suffix):].unsqueeze(-1)).sum()
            values[label] = float(total.item())
        return values

    def score_imdb(self, job, ids):
        labels = job.labels
        instruction = job.text.split("\n\n", 1)[0]
        if instruction not in self.priors:
            prior_probabilities = []
            for content_free_input in ("N/A", "[no review]", ""):
                prompt = classification_prompt("imdb", content_free_input, instruction=instruction)
                prior_scores = self.log_scores(self.codec.encode(self.codec.render(prompt, job.mode)), labels)
                probs = torch.softmax(torch.tensor([prior_scores[label] for label in labels]), dim=0).tolist()
                prior_probabilities.append(dict(zip(labels, probs, strict=True)))
            self.priors[instruction] = {
                label: float(np.log(max(np.mean([row[label] for row in prior_probabilities]), 1e-30)))
                for label in labels
            }
        scores = self.log_scores(ids, labels)
        prior = self.priors[instruction]
        calibrated = {label: scores[label] - prior[label] for label in labels}
        pred = max(scores, key=scores.get)
        cal_pred = max(calibrated, key=calibrated.get)
        return {
            "prediction": pred, "is_correct": pred == job.reference,
            "label_log_likelihood": scores, "content_free_label_log_prior": prior,
            "calibrated_scores": calibrated, "calibrated_prediction": cal_pred,
            "calibrated_correct": cal_pred == job.reference,
        }


def classification_score(job, output):
    result = label_paths(output, job.labels)
    return {
        **result, "prediction": result["strict_prediction"],
        "is_correct": result["strict_prediction"] == job.reference,
        "legacy_correct": result["legacy_prediction"] == job.reference,
        "unique_mention_correct": result["unique_mention_prediction"] == job.reference,
    }


def manifest_for(config, model_id, weights, revision, retriever):
    params = asdict(config)
    for name in ("retry_oom", "output_root", "cache_root"):
        params.pop(name)
    params = {key: str(value) if isinstance(value, Path) else value for key, value in params.items()}
    source_ids = [CLASS_SOURCES[name] for name in CLASS_DATASETS if config.samples[name] != 0]
    if config.samples["clapnq"] != 0:
        source_ids.append(CLAPNQ_ID)
    return {
        **params,
        "experiment_version": "1.2",
        "model_id": model_id, "model_revision": revision, "weights": weights,
        "native_window": NATIVE_WINDOW, "chat_thinking": False,
        "fixed_data_revisions": {repo: hub_revision(repo) for repo in source_ids},
        "retriever_id": retriever.model_id, "retriever_revision": retriever.revision,
        "retriever_max_seq_length": retriever.limit,
        "sql_row_limit": SQL_ROW_LIMIT, "sql_timeout_seconds": SQL_TIMEOUT,
        "generation_budgets": {"classification": CLASS_MAX_NEW, "sql": SQL_MAX_NEW, "qa": QA_MAX_NEW},
        "gpu": torch.cuda.get_device_name(0) if not config.profile_only else None,
        "runtime_versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "datasets", "sentence-transformers", "sqlglot", "bitsandbytes")
        },
        "sqlite_version": sqlite3.sqlite_version,
    }


def evaluate_trial(config, job, codec, runner, checkpoint, manifest):
    rendered = codec.render(job.text, job.mode)
    prompt_sha = digest(rendered)
    # Include data provenance so changed SQL inputs cannot reuse a stale trial.
    trial_id = digest(json_text([
        job.dataset, job.sample_id, job.condition, job.mode, job.max_new, prompt_sha,
        job.reference, job.details.get("source_revision"), job.details.get("dataset_json_sha256"),
        job.details.get("db_sha256"), job.details.get("train_json_sha256"),
    ]))
    if checkpoint.has(trial_id):
        return
    ids = codec.encode(rendered, job.mode)
    prompt_tokens = len(ids)
    reserved = job.max_new
    if job.score_only:
        reserved = max(len(codec.encode(label)) for label in job.labels)
    details = dict(job.details)
    if details.get("gold_evidence_marker"):
        details["gold_evidence_start_token"] = codec.evidence_start(rendered, details["gold_evidence_marker"])
    row = {
        "trial_id": trial_id, "dataset": job.dataset, "task": job.task, "sample_id": job.sample_id,
        "condition": job.condition, "model_id": manifest["model_id"], "model_revision": manifest["model_revision"],
        "weight_mode": manifest["weights"], "prompt_mode": job.mode,
        "retriever_id": manifest["retriever_id"],
        "query": job.query, "query_tokens": codec.count(job.query), "reference": job.reference,
        "prompt_sha256": prompt_sha, "prompt_tokens": prompt_tokens,
        "native_context_tokens": NATIVE_WINDOW, "prompt_pct_of_native": 100 * prompt_tokens / NATIVE_WINDOW,
        "output_tokens_reserved": reserved, "planned_total_tokens": prompt_tokens + reserved,
        "native_headroom_after_reserve": NATIVE_WINDOW - prompt_tokens - reserved,
        "configured_prompt_limit": config.max_prompt_tokens,
        "completion_tokens": None, "observed_total_tokens": None, "raw_output": None,
        "details": details, "status": "ok",
    }
    if prompt_tokens + reserved > NATIVE_WINDOW:
        row["status"] = "native_window_overflow"
    elif prompt_tokens > config.max_prompt_tokens:
        row["status"] = "configured_prompt_limit"
    elif runner is None:
        row["status"] = "profile_only"
    else:
        started = time.perf_counter()
        try:
            if job.score_only:
                row.update(runner.score_imdb(job, ids))
            else:
                generation = runner.generate(ids, job.max_new, job.mode)
                row.update(generation)
                row["observed_total_tokens"] = prompt_tokens + generation["completion_tokens"]
                output = generation["raw_output"]
                if job.task == "classification":
                    row.update(classification_score(job, output))
                elif job.task == "sql":
                    row.update(score_sql(job, output))
                else:
                    row.update(qa_score(job.reference, output))
        except torch.cuda.OutOfMemoryError as exc:
            row["status"] = "cuda_oom"
            row["runtime_error"] = str(exc)
        row["elapsed_seconds"] = time.perf_counter() - started
        if row["status"] == "cuda_oom":
            gc.collect()
            torch.cuda.empty_cache()
    checkpoint.save(row, rendered)


def run_jobs(config, model_id, weights, retriever):
    revision = hub_revision(model_id, "model")
    codec = QwenInput(model_id, revision)
    manifest = manifest_for(config, model_id, weights, revision, retriever)
    run_name = model_id.replace("/", "_") + "_" + digest(json_text(manifest))[:12]
    run_dir = config.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Checkpoint(run_dir / "checkpoint.sqlite", manifest, config.retry_oom)
    runner = None
    try:
        if not config.profile_only:
            runner = QwenRunner(model_id, revision, codec, weights)
        groups = [
            (dataset, class_jobs(config, dataset, config.samples[dataset], retriever, codec))
            for dataset in CLASS_DATASETS if config.samples[dataset] != 0
        ]
        groups.extend(
            (dataset, sql_jobs(config, dataset, config.samples[dataset], retriever))
            for dataset in SQL_DATASETS if config.samples[dataset] != 0
        )
        if config.samples["clapnq"] != 0:
            groups.append(("clapnq", qa_jobs(config, config.samples["clapnq"], retriever, codec)))

        for name, jobs in groups:
            for job in tqdm(jobs, desc=name, unit="prompt"):
                evaluate_trial(config, job, codec, runner, checkpoint, manifest)
        checkpoint.mark_finished()
    finally:
        checkpoint.db.close()
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return run_dir
