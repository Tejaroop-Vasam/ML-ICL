"""Resumable trial ledgers and raw evidence exports."""

import csv
import gzip
import json
import sqlite3
import zipfile
from collections import defaultdict
from contextlib import closing

from .common import choose, digest, json_text, rng_for, round_robin_sample


class Checkpoint:
    """Commit complete per-example records and deduplicated prompts in one transaction."""
    def __init__(self, path, manifest, retry_oom=False):
        self.retry_oom = retry_oom
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS manifest (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS prompts (sha TEXT PRIMARY KEY, gzip_text BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS trials (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, prompt_sha TEXT NOT NULL, payload TEXT NOT NULL
            );
        """)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO manifest VALUES (?, ?)", ("configuration", json_text(manifest)))
        previous = self.db.execute("SELECT value FROM manifest WHERE key = 'configuration'").fetchone()[0]
        if previous != json_text(manifest):
            self.db.close()
            raise ValueError("Checkpoint configuration mismatch.")

    def has(self, trial_id):
        result = self.db.execute("SELECT status FROM trials WHERE id = ?", (trial_id,)).fetchone()
        return result is not None and not (self.retry_oom and result[0] == "cuda_oom")

    def save(self, row, rendered):
        with self.db:
            prompt_sha = row["prompt_sha256"]
            if self.db.execute("SELECT 1 FROM prompts WHERE sha = ?", (prompt_sha,)).fetchone() is None:
                self.db.execute(
                    "INSERT INTO prompts VALUES (?, ?)",
                    (prompt_sha, gzip.compress(rendered.encode("utf-8"), compresslevel=3, mtime=0)),
                )
            self.db.execute(
                "INSERT OR REPLACE INTO trials VALUES (?, ?, ?, ?)",
                (row["trial_id"], row["status"], prompt_sha, json_text(row)),
            )

    def mark_finished(self):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO manifest VALUES ('loop_completed', 'true')")

ANNOTATION_RANDOM_PER_DATASET = 6
ANNOTATION_TARGETED_PER_DATASET = 6


def stream_trials(connection):
    for (payload,) in connection.execute("SELECT payload FROM trials ORDER BY id"):
        yield json.loads(payload)


def data_anchors(row, reference_k, dense_prefix):
    condition = row["condition"]
    if row["task"] == "classification":
        return condition in {
            "legacy_raw_16", "chat_label_only_16", "chat_label_only_scores",
            f"{dense_prefix}_k{reference_k}", f"balanced_{dense_prefix}_k{reference_k}",
            f"label_informed_oracle_k{reference_k}",
        }
    if row["task"] == "sql":
        return condition in {"full_schema", f"{dense_prefix}_schema_matched_k", "oracle_schema"}
    return condition in {f"{dense_prefix}_passage", "gold_passage"}


def case_stratum(variants, reference_k, dense_prefix):
    rows = {row["condition"]: row for row in variants}
    first = variants[0]
    if first["task"] == "classification":
        legacy = rows.get("legacy_raw_16", {})
        label_only = rows.get("chat_label_only_16", {})
        dense = rows.get(f"{dense_prefix}_k{reference_k}", {})
        balanced = rows.get(f"balanced_{dense_prefix}_k{reference_k}", {})
        if legacy.get("legacy_parser_route") == "defaulted_to_first_label":
            return "legacy_parser_default"
        if legacy.get("legacy_correct") != label_only.get("is_correct"):
            return "legacy_vs_label_only_disagreement"
        if dense.get("is_correct") != balanced.get("is_correct"):
            return "demo_selection_disagreement"
        return "agreement"
    if first["task"] == "sql":
        dense = rows.get(f"{dense_prefix}_schema_matched_k", {})
        oracle = rows.get("oracle_schema", {})
        if dense and oracle and dense.get("is_correct") != oracle.get("is_correct"):
            return "schema_arm_disagreement"
        if oracle.get("is_correct") is False:
            return "oracle_sql_failure"
        return "agreement"
    if rows.get(f"{dense_prefix}_passage", {}).get("gold_passage_available") is False:
        return "gold_passage_missed"
    return "passage_recovered"


def write_annotation_packets(connection, run_dir, configuration):
    seed, reference_k = configuration["seed"], configuration["reference_k"]
    # Match the stored condition names when exporting an earlier checkpoint.
    dense_prefix = "minilm" if configuration.get("experiment_version") == "1.1" else "bge_m3"
    grouped = defaultdict(lambda: defaultdict(list))
    for row in stream_trials(connection):
        if row["status"] == "ok" and data_anchors(row, reference_k, dense_prefix):
            grouped[row["dataset"]][row["sample_id"]].append(row)

    annotations_path = run_dir / "blind_annotations.jsonl"
    key_path = run_dir / "annotation_key.jsonl"
    rubric_path = run_dir / "annotation_labels.csv"
    with annotations_path.open("w", encoding="utf-8") as packets, key_path.open("w", encoding="utf-8") as key, \
            rubric_path.open("w", encoding="utf-8", newline="") as fields:
        writer = csv.writer(fields)
        writer.writerow([
            "annotation_id", "rater_id", "pass", "verdict", "primary_error_code",
            "secondary_error_codes", "coverage_0_to_2", "faithfulness_0_to_2",
            "input_excerpt", "output_excerpt", "reason", "confidence",
        ])
        for dataset, cases in sorted(grouped.items()):
            cases = {sample_id: variants for sample_id, variants in cases.items() if len(variants) >= 2}
            ids = sorted(cases)
            random_ids = choose(ids, ANNOTATION_RANDOM_PER_DATASET, seed, dataset + "/annotation random")
            from_other = [sample_id for sample_id in ids if sample_id not in set(random_ids)]
            targeted = round_robin_sample(
                from_other, ANNOTATION_TARGETED_PER_DATASET,
                lambda sample_id: case_stratum(cases[sample_id], reference_k, dense_prefix),
                seed, dataset + "/annotation targeted",
            )
            for sample_id in random_ids + targeted:
                variants = cases[sample_id]
                stratum = case_stratum(variants, reference_k, dense_prefix) if sample_id in targeted else None
                case_id = digest(json_text([run_dir.name, dataset, sample_id]))[:20]
                order = sorted(variants, key=lambda row: row["trial_id"])
                rng_for(seed, case_id).shuffle(order)
                for number, row in enumerate(order):
                    annotation_id = case_id + f"_{number + 1}"
                    packet = {
                        "annotation_id": annotation_id, "task": row["task"], "dataset": dataset,
                        "input": row["query"], "reference": row["reference"],
                        "response": row["raw_output"] if row["raw_output"] is not None else row.get("prediction"),
                    }
                    if row["task"] == "qa":
                        packet["reference_support"] = row["details"]["gold_support_sentences"]
                    packets.write(json_text(packet) + "\n")
                    key.write(json_text({
                        "annotation_id": annotation_id, "trial_id": row["trial_id"],
                        "sample_id": sample_id, "condition": row["condition"], "prompt_sha256": row["prompt_sha256"],
                        "sampling_cohort": "random" if sample_id in random_ids else "targeted",
                        "targeted_stratum": stratum,
                    }) + "\n")
                    writer.writerow([annotation_id] + [""] * 11)


def export_raw(run_dir):
    checkpoint_path = run_dir / "checkpoint.sqlite"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    archive_path = run_dir.with_name(run_dir.name + ".zip")
    with closing(sqlite3.connect(checkpoint_path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        configuration = dict(connection.execute("SELECT key, value FROM manifest"))
        manifest = {key: json.loads(value) for key, value in configuration.items()}
        with (run_dir / "manifest.json").open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, ensure_ascii=False)
        with (run_dir / "trials.jsonl").open("w", encoding="utf-8") as file:
            for (payload,) in connection.execute("SELECT payload FROM trials ORDER BY id"):
                file.write(payload + "\n")
        with gzip.open(run_dir / "prompts.jsonl.gz", "wt", encoding="utf-8") as file:
            for sha, blob in connection.execute("SELECT sha, gzip_text FROM prompts ORDER BY sha"):
                file.write(json_text({"prompt_sha256": sha, "text": gzip.decompress(blob).decode("utf-8")}) + "\n")
        write_annotation_packets(connection, run_dir, manifest["configuration"])
    files = ("manifest.json", "trials.jsonl", "prompts.jsonl.gz", "blind_annotations.jsonl",
             "annotation_key.jsonl", "annotation_labels.csv", "checkpoint.sqlite")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for name in files:
            archive.write(run_dir / name, arcname=name)
    return archive_path
