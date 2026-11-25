"""Official Spider and BIRD benchmark loading, prompting and SQL execution."""

import json
import re
import shutil
import sqlite3
import time
import zipfile
from collections import Counter
from contextlib import closing
from functools import lru_cache
from pathlib import Path

import gdown
import requests
from huggingface_hub import hf_hub_download
from sqlglot import exp, parse
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import traverse_scope

from .common import Job, clean_key, file_sha256, hub_revision, rng_for, round_robin_sample
from .config import SQL_MAX_NEW

SPIDER_FILE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
BIRD_DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
BIRD_DEV_MIRROR = "HAL-9001/bird-dev"
SQL_ROW_LIMIT = 100_000
SQL_TIMEOUT = 30


def transfer(url, destination):
    tmp = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with tmp.open("wb") as file:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    file.write(block)
        tmp.replace(destination)
    finally:
        tmp.unlink(missing_ok=True)


def extract_zip(path, root):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            if not (root / item.filename).resolve().is_relative_to(root):
                raise ValueError(f"Invalid archive path: {item.filename}")
        archive.extractall(root)


def find_file(root, names):
    for name in names:
        paths = sorted(
            path for path in root.rglob(name)
            if "__MACOSX" not in path.parts and not path.name.startswith("._")
        )
        if paths:
            return paths[0]
    return None


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sql_data_root(config, dataset, split):
    configured = {
        ("spider", "dev"): config.spider_path, ("spider", "train"): config.spider_path,
        ("bird", "dev"): config.bird_dev_path, ("bird", "train"): config.bird_train_path,
    }[dataset, split]
    root = config.cache_root / (dataset if dataset == "spider" else f"bird_{split}")
    root.mkdir(parents=True, exist_ok=True)
    names = ("train_spider.json",) if dataset == "spider" and split == "train" else (f"{split}.json",)
    if configured is not None:
        path = Path(configured).expanduser()
        if path.is_dir():
            root = path
        elif path.is_file():
            root = root / file_sha256(path)[:16]
            if not (root / ".archive-extracted").exists():
                extract_zip(path, root)
                (root / ".archive-extracted").touch()
        else:
            raise FileNotFoundError(path)
    elif not find_file(root, names) or (
        (root / ("spider_data.zip" if dataset == "spider" else "dev.zip")).exists()
        and not (root / ".archive-extracted").exists()
    ):
        if dataset == "spider":
            archive = root / "spider_data.zip"
            if not archive.exists():
                temp = root / "spider_data.part"
                try:
                    result = gdown.download(id=SPIDER_FILE_ID, output=str(temp), quiet=False)
                    if not result or not zipfile.is_zipfile(temp):
                        raise RuntimeError("Provide the official Spider ZIP with --spider-path.")
                    temp.replace(archive)
                finally:
                    temp.unlink(missing_ok=True)
            extract_zip(archive, root)
            (root / ".archive-extracted").touch()
        elif split == "dev":
            archive = root / "dev.zip"
            try:
                if not archive.exists():
                    transfer(BIRD_DEV_URL, archive)
                extract_zip(archive, root)
                (root / ".archive-extracted").touch()
            except (requests.RequestException, zipfile.BadZipFile) as exc:
                print(f"BIRD official download unavailable ({exc}). Trying the archived dev release on Hugging Face.")
                revision = hub_revision(BIRD_DEV_MIRROR)
                cached = hf_hub_download(
                    BIRD_DEV_MIRROR, "dev.zip", repo_type="dataset", revision=revision,
                )
                expected_sha = "cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630"
                if file_sha256(cached) != expected_sha:
                    raise ValueError("The mirrored BIRD archive failed its published checksum.")
                shutil.copyfile(cached, archive)
                extract_zip(archive, root)
                (root / ".archive-extracted").touch()
        else:
            raise ValueError("Set --bird-train-path to the official BIRD training release for BIRD ICL.")

    questions = find_file(root, names)
    if questions is None:
        raise FileNotFoundError(f"Expected {names} inside {root}")

    # BIRD commonly distributes the databases inside a nested ZIP.
    preferred_dirs = ("database",) if dataset == "spider" else (f"{split}_databases", "databases")
    db_roots = [questions.parent / name for name in preferred_dirs]
    db_root = next((folder for folder in db_roots if folder.is_dir()), None)
    if dataset == "bird" and db_root is None:
        nested = find_file(root, (f"{split}_databases.zip", "databases.zip"))
        if nested is not None:
            db_root = config.cache_root / f"bird_{split}_sqlite_{file_sha256(nested)[:16]}"
            if not (db_root / ".archive-extracted").exists():
                extract_zip(nested, db_root)
                (db_root / ".archive-extracted").touch()
    # Scope database lookup to the data split when possible.
    db_root = db_root or questions.parent
    db_paths = {
        path.stem: path for path in sorted(db_root.rglob("*.sqlite"))
        if "__MACOSX" not in path.parts and path.parent.name == path.stem
    }
    if not db_paths:
        raise FileNotFoundError(f"No benchmark SQLite databases found near {questions}.")
    return questions, db_paths


@lru_cache(maxsize=None)
def read_only_uri(db_path):
    return db_path.resolve().as_uri() + "?mode=ro"


@lru_cache(maxsize=256)
def db_schema(db_path):
    with closing(sqlite3.connect(read_only_uri(db_path), uri=True)) as connection:
        rows = connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type IN ('table', 'view') "
            "AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY lower(name), name"
        ).fetchall()
    if not rows:
        raise ValueError(f"Database has no visible tables or views: {db_path}")
    return dict(rows)


def parse_select(sql):
    statements = [statement for statement in parse(sql, read="sqlite") if statement is not None]
    if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.SetOperation)):
        raise ValueError("Expected a single SELECT or set query.")
    return statements[0]


def physical_tables(tree):
    return {
        source.name.casefold()
        for scope in traverse_scope(tree)
        for source in scope.sources.values()
        if isinstance(source, exp.Table)
    }


def gold_table_names(reference, schema):
    try:
        tables = physical_tables(parse_select(reference))
    except (SqlglotError, ValueError):
        return None
    lookup = {name.casefold(): name for name in schema}
    if not tables or not tables.issubset(lookup):
        return None
    return [name for name in schema if name.casefold() in tables]


def sql_block(db_id, schema, question, evidence, answer=None):
    result = [
        f"Database: {db_id}", "SQLite schema:",
        "\n".join(ddl.rstrip(";") + ";" for ddl in schema.values()),
    ]
    if evidence:
        result.append("Domain evidence: " + evidence)
    result.extend(["Question: " + question, "SQL:" if answer is None else "SQL: " + answer])
    return "\n".join(result)


def sql_jobs(config, dataset, n_eval, retriever):
    dev_file, databases = sql_data_root(config, dataset, "dev")
    dev_rows = list(enumerate(read_json(dev_file)))
    dev_sha = file_sha256(dev_file)
    cases = round_robin_sample(dev_rows, n_eval, lambda pair: pair[1]["db_id"], config.seed, dataset + "/dev")
    reference_key = "query" if dataset == "spider" else "SQL"
    instruction = (
        "Write one SQLite SELECT query for the final question. Use the schema for that question. "
        "Return only SQL."
    )
    train_pool, train_vectors = [], None
    train_sha = None
    if config.sql_demo_k and (dataset == "spider" or config.bird_train_path is not None):
        train_file, train_databases = sql_data_root(config, dataset, "train")
        train_sha = file_sha256(train_file)
        dev_db_ids = {row["db_id"] for _, row in dev_rows}
        train_rows = [
            (idx, item) for idx, item in enumerate(read_json(train_file))
            if item["db_id"] not in dev_db_ids
            and item["db_id"] in train_databases
        ]
        train_pool = round_robin_sample(train_rows, config.pool_size, lambda pair: pair[1]["db_id"], config.seed, dataset + "/train")
        if len(train_pool) < config.sql_demo_k:
            raise ValueError(f"Too few official cross-database {dataset} demonstrations.")
        train_vectors = retriever.encode([row["question"] for _, row in train_pool])

    schema_vectors = {}
    database_hash = {}
    for idx, row in cases:
        sample_id = f"{dataset}_dev_{idx}"
        db_id = row["db_id"]
        db_path = databases.get(db_id)
        if db_path is None:
            raise FileNotFoundError(f"No database for {sample_id} ({db_id}).")
        schema = db_schema(db_path)
        if db_id not in schema_vectors:
            schema_vectors[db_id] = retriever.encode([f"{name}\n{ddl}" for name, ddl in schema.items()])
            database_hash[db_id] = file_sha256(db_path)
        question, reference = row["question"], row[reference_key]
        evidence = (row.get("evidence") or "").strip()
        retrieval_query = question + ("\n" + evidence if evidence else "")
        query_vector = retriever.encode([retrieval_query])[0]
        names = list(schema)
        oracle = gold_table_names(reference, schema)
        table_count = len(oracle) if oracle else min(3, len(names))
        ranked, scores = retriever.rank(schema_vectors[db_id], query_vector)
        retrieved_tables = [names[i] for i in ranked[:table_count]]
        random_tables = rng_for(config.seed, sample_id, "tables").sample(names, table_count)
        # Canonical schema ordering in every arm keeps the comparison focused on table choice.
        chosen_modes = [
            ("full_schema", names, evidence, []),
            ("bge_m3_schema_matched_k", retrieved_tables, evidence, []),
            ("random_schema_matched_k", random_tables, evidence, []),
        ]
        if oracle:
            chosen_modes.append(("oracle_schema", oracle, evidence, []))
        if dataset == "bird" and evidence:
            chosen_modes.append(("full_schema_without_evidence", names, "", []))
        if train_pool:
            demo_ranking, _ = retriever.rank(train_vectors, retriever.encode([question])[0])
            eligible = [i for i in demo_ranking if clean_key(train_pool[i][1]["question"]) != clean_key(question)]
            if len(eligible) >= config.sql_demo_k:
                demos = [train_pool[i] for i in eligible[:config.sql_demo_k][::-1]]
                chosen_modes.append((f"full_schema_bge_m3_icl_k{config.sql_demo_k}", names, evidence, demos))

        for condition, chosen, cue, demos in chosen_modes:
            prompt_parts = [instruction]
            for _, demo in demos:
                prompt_parts.append(sql_block(
                    demo["db_id"], db_schema(train_databases[demo["db_id"]]),
                    demo["question"], (demo.get("evidence") or "").strip(), demo[reference_key],
                ))
            selected = {name: ddl for name, ddl in schema.items() if name in chosen}
            prompt_parts.append(sql_block(db_id, selected, question, cue))
            yield Job(
                "sql", dataset, sample_id, condition, "\n\n".join(prompt_parts),
                question, reference, SQL_MAX_NEW, db_path=db_path,
                details={
                    "split": "dev", "dataset_json_sha256": dev_sha, "db_sha256": database_hash[db_id],
                    "db_id": db_id, "difficulty": row.get("difficulty"),
                    "benchmark_evidence": evidence, "evidence_in_prompt": bool(cue),
                    "total_schema_tables": len(names), "schema_tables": list(selected),
                    "oracle_tables": oracle, "gold_table_count_used": oracle is not None and "matched_k" in condition,
                    "oracle_table_recall": len(set(chosen) & set(oracle)) / len(oracle) if oracle else None,
                    "table_scores": {name: float(score) for name, score in zip(names, scores, strict=True)},
                    "retriever_limit": retriever.limit,
                    "retriever_query_tokens": retriever.lengths([retrieval_query])[0],
                    "demonstration_ids": [f"{dataset}_train_{i}" for i, _ in demos],
                    "train_json_sha256": train_sha if demos else None,
                },
            )


def sql_text(output):
    output = output.strip()
    fenced = re.fullmatch(r"```(?:sql|sqlite)?\s*\n?(.*?)\s*```", output, re.I | re.S)
    return fenced.group(1).strip() if fenced else output


def authorize_sql(action, arg1, arg2, db_name, trigger):
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").casefold() in {"load_extension", "readfile", "writefile"}:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def execute_sql(db_path, query):
    deadline = time.monotonic() + SQL_TIMEOUT
    with closing(sqlite3.connect(read_only_uri(db_path), uri=True, timeout=2)) as con:
        con.execute("PRAGMA query_only=ON")
        con.execute("PRAGMA trusted_schema=OFF")
        con.set_authorizer(authorize_sql)
        con.set_progress_handler(lambda: int(time.monotonic() >= deadline), 2000)
        try:
            cursor = con.execute(query)
            rows = cursor.fetchmany(SQL_ROW_LIMIT + 1)
            if len(rows) > SQL_ROW_LIMIT:
                return "row_limit", None, None
            return "ok", rows, len(cursor.description)
        except sqlite3.Error as exc:
            message = str(exc)
            return ("timeout" if time.monotonic() >= deadline else message), None, None


@lru_cache(maxsize=1)
def execute_reference(db_path, sql):
    return execute_sql(db_path, sql)


def score_sql(job, output):
    prediction = sql_text(output)
    gold_status, gold_rows, gold_columns = execute_reference(job.db_path, job.reference)
    if gold_status != "ok":
        return {"prediction": prediction, "sql_status": "reference_error", "gold_error": gold_status, "is_correct": None}
    reason = None
    try:
        tree = parse_select(prediction)
    except (SqlglotError, ValueError) as exc:
        tree, reason = None, str(exc)
    if tree is None:
        return {"prediction": prediction, "sql_status": "invalid_sql", "sql_error": reason, "is_correct": False}

    status, predicted_rows, predicted_columns = execute_sql(job.db_path, prediction)
    result = {
        "prediction": prediction, "sql_status": "execution_error" if status != "ok" else "executed",
        "sql_error": status if status != "ok" else None,
        "predicted_tables": sorted(physical_tables(tree)),
        "gold_result_row_count": len(gold_rows), "predicted_result_row_count": None,
        "gold_result_column_count": gold_columns, "predicted_result_column_count": predicted_columns,
        "is_correct": False,
    }
    if predicted_rows is None:
        return result

    if job.dataset == "bird":
        correct = set(predicted_rows) == set(gold_rows)
        comparison = "bird_row_set"
    else:
        try:
            ordered = parse_select(job.reference).args.get("order") is not None
        except (SqlglotError, ValueError):
            return {**result, "sql_status": "reference_parse_error", "is_correct": None}
        matching_rows = predicted_rows == gold_rows if ordered else Counter(predicted_rows) == Counter(gold_rows)
        correct = predicted_columns == gold_columns and matching_rows
        comparison = "single_db_ordered_rows" if ordered else "single_db_row_multiset"
    return {
        **result, "is_correct": correct, "execution_comparison": comparison,
        "predicted_result_row_count": len(predicted_rows),
    }
