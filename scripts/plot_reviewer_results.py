#!/usr/bin/env python3
"""Plot exported reviewer trials locally. Requires pandas and matplotlib."""

import argparse
import hashlib
import html
import json
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.label_parsing import rescore_classification_trial


def exports(paths):
    for path in paths:
        if path.is_file() and path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                for member in sorted(names):
                    name = PurePosixPath(member)
                    manifest = str(name.with_name("manifest.json"))
                    if name.name == "trials.jsonl" and manifest in names:
                        yield path.stem, json.loads(archive.read(manifest)), archive.read(member).splitlines()
        elif path.is_dir():
            for trial_file in sorted(path.rglob("trials.jsonl")):
                manifest = trial_file.with_name("manifest.json")
                if manifest.is_file():
                    yield trial_file.parent.name, json.loads(manifest.read_text()), trial_file.read_bytes().splitlines()
        else:
            raise ValueError(f"Expected a results directory or ZIP: {path}")


def load_runs(paths):
    seen = set()
    for name, manifest, lines in exports(paths):
        config = manifest.get("configuration", manifest)
        records = [rescore_classification_trial(json.loads(line)) for line in lines if line.strip()]
        key = hashlib.sha256(b"\n".join(lines)).digest()
        if not records or key in seen:
            continue
        seen.add(key)
        rows = pd.json_normalize(records)
        rows["run"] = name
        rows["model"] = config.get("model_id", name).rsplit("/", 1)[-1]
        rows["retriever"] = config.get("retriever_id", "unknown")
        rows["reference_key"] = rows["reference"].map(json.dumps)
        if "parse_status" in rows:
            rows["label_found"] = rows["parse_status"].map({
                "unique_mention": 1.0, "ambiguous": 0.0, "no_label": 0.0, "empty": 0.0,
            })
        rows["scored"] = np.nan
        for task, metric in (("classification", "is_correct"), ("sql", "is_correct"), ("qa", "max_rouge_l_f1")):
            mask = rows["task"].eq(task) & rows["status"].eq("ok")
            if metric in rows:
                rows.loc[mask, "scored"] = pd.to_numeric(rows.loc[mask, metric], errors="coerce").astype(float)
        yield name, config, rows


def slug(value):
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value))


def scores_label(task):
    return {"classification": "Classification accuracy (%)", "sql": "Execution accuracy (%)",
            "qa": "ROUGE-L F1 (%)"}[task]


def save(fig, path):
    fig.savefig(path, dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bars(values, title, xlabel, out, *, percentage=False):
    values = values.dropna(how="all") if isinstance(values, pd.DataFrame) else values.dropna()
    if values.empty:
        return
    bars_per_condition = len(values.columns) if isinstance(values, pd.DataFrame) else 1
    row_height = max(.42, .2 * bars_per_condition)
    fig, ax = plt.subplots(figsize=(11, max(3, row_height * len(values) + 1.4)), layout="constrained")
    values.rename(index=lambda label: str(label).replace("_", " ")).iloc[::-1].plot.barh(
        ax=ax, color="#285f83" if isinstance(values, pd.Series) else None,
    )
    ax.set(title=title, xlabel=xlabel, ylabel="")
    if percentage:
        ax.set_xlim(0, 105)
        for group in ax.containers:
            ax.bar_label(group, labels=[f"{value:.1f}" if pd.notna(value) else "" for value in group.datavalues],
                         padding=3, fontsize=8)
    if isinstance(values, pd.DataFrame):
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    save(fig, out)


def heatmap(frame, title, out, *, xlabel="", ylabel="", vmax=None):
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, .9 * len(frame.columns) + 3), max(3.5, .48 * len(frame) + 2)),
                           layout="constrained")
    matrix = frame.to_numpy(dtype=float)
    cmap = matplotlib.colormaps["Blues"].with_extremes(bad="#eeeeee")
    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.grid(False)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.set_xticks(range(len(frame.columns)), [str(c).replace("_", " ") for c in frame.columns], rotation=32, ha="right")
    ax.set_yticks(range(len(frame)), [str(i).replace("_", " ") for i in frame.index])
    upper = image.norm.vmax or 1
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:.0f}" if np.isfinite(value) else "–", ha="center", va="center",
                color="white" if value > upper * .62 else "#162d3b", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=.8)
    save(fig, out)


def scatter(rows, x, y, title, x_label, y_label, out, *, scale=1):
    if not {x, y}.issubset(rows.columns):
        return
    points = rows.dropna(subset=[x, y])
    if points.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    ax.scatter(points[x], points[y] * scale, s=23, color="#286f86", alpha=.55)
    ax.set(title=title, xlabel=x_label, ylabel=y_label)
    save(fig, out)


def paired_delta(rows, baseline, title, out):
    table = rows.pivot_table(index="sample_id", columns="condition", values="scored")
    if baseline not in table or len(table.columns) < 2:
        return
    diffs = table.drop(columns=baseline).sub(table[baseline], axis=0)
    counts = diffs.count()
    means = diffs.mean().dropna() * 100
    if not means.empty:
        means.index = [f"{condition} (n={counts[condition]})" for condition in means.index]
        bars(means, title, "Paired difference (points)", out)


def classification(rows, name, out):
    valid = rows[rows["status"].eq("ok")].copy()
    if "prediction" not in valid or valid.empty:
        return
    valid["prediction"] = valid["prediction"].fillna("Invalid output")
    f1_by_condition = {}
    for condition, group in valid.groupby("condition"):
        matrix = pd.crosstab(group["reference"], group["prediction"])
        predicted = matrix.sum(axis=0).reindex(matrix.index, fill_value=0)
        correct = np.diag(matrix.reindex(columns=matrix.index, fill_value=0).to_numpy())
        f1_by_condition[condition] = 100 * np.mean(2 * correct / (matrix.sum(axis=1) + predicted))
        heatmap(matrix, f"{name} | {condition}", out / f"confusion_{slug(condition)}.png",
                xlabel="Prediction", ylabel="Reference")
    bars(pd.Series(f1_by_condition), name, "Macro F1 over observed reference classes (%)",
         out / "macro_f1.png", percentage=True)
    if "parse_status" in rows:
        status = pd.crosstab(rows["condition"], rows["parse_status"], normalize="index") * 100
        bars(status, name, "Responses (%)", out / "parser_status.png")
    if "label_found" in rows:
        generated = rows.dropna(subset=["label_found"])
        scores = generated.groupby("condition")[["scored", "label_found"]].mean() * 100
        scores = scores.rename(columns={"scored": "Classification accuracy", "label_found": "Label coverage"})
        bars(scores, name, "Accuracy and label coverage (%)", out / "parser_accuracy.png", percentage=True)
    if "calibrated_correct" in rows:
        calibration = rows.dropna(subset=["calibrated_correct"]).groupby("condition")[["is_correct", "calibrated_correct"]]
        bars(calibration.mean().rename(columns={"is_correct": "Raw likelihood", "calibrated_correct": "Calibrated"}) * 100,
             name, "Accuracy (%)", out / "label_calibration.png", percentage=True)

    sweep = rows[rows["condition"].str.fullmatch(r"(?:bge_m3|minilm)_k\d+|chat_label_only_16")].copy()
    if not sweep.empty:
        sweep["shots"] = sweep["condition"].str.extract(r"_k(\d+)$")[0].fillna(0).astype(int)
        series = sweep.groupby("shots")["scored"].mean().dropna() * 100
        if len(series) > 1:
            fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
            ax.plot(series.index, series, "o-", color="#285f83")
            ax.set(title=name, xlabel="Retrieved demonstrations", ylabel=scores_label("classification"), ylim=(0, 105))
            ax.set_xticks(series.index)
            save(fig, out / "shot_count.png")
    key = "details.retriever_query_tokens"
    if key in rows:
        unique_queries = rows[rows["condition"].eq("chat_label_only_16")].drop_duplicates("sample_id")
        if not unique_queries.empty:
            fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
            ax.hist(unique_queries[key].dropna(), bins=16, color="#285f83")
            limit = unique_queries["details.retriever_limit"].dropna()
            if not limit.empty:
                ax.axvline(limit.iloc[0], color="#a3492b", linestyle="--", label=f"Limit ({limit.iloc[0]:,})")
                ax.set_xscale("log")
                ax.legend()
            ax.set(title=name, xlabel="Retriever input tokens", ylabel="Queries")
            save(fig, out / "retriever_input_lengths.png")


def sql(rows, name, out):
    if "sql_status" in rows:
        bars(pd.crosstab(rows["condition"], rows["sql_status"]), name, "Trials", out / "sql_outcomes.png")
    key = "details.oracle_table_recall"
    if key in rows:
        bars(rows.groupby("condition")[key].mean() * 100, name, "Mean reference-table recall (%)",
             out / "schema_table_recall.png", percentage=True)
        recall = rows[rows["condition"].str.endswith("_schema_matched_k")].dropna(subset=[key, "scored"])
        if not recall.empty:
            heatmap(pd.crosstab(recall[key].map(lambda x: f"{x:.0%}"), recall["scored"].map({0: "Wrong", 1: "Correct"})),
                    name, out / "table_recall_vs_execution.png", ylabel="Reference tables retrieved")
    for key, label in (("details.difficulty", "Difficulty"), ("details.db_id", "Database")):
        if key in rows and rows[key].notna().any():
            table = rows.pivot_table(index=key, columns="condition", values="scored") * 100
            heatmap(table, name, out / f"execution_by_{label.lower()}.png", ylabel=label, vmax=100)


def qa(rows, name, out):
    regular = rows[~rows["condition"].str.startswith("stress_")]
    for key, axis, fmt in (("max_rouge_lsum_f1", "ROUGE-Lsum F1 (%)", True),
                           ("abstained", "Abstention (%)", True),
                           ("prediction_words", "Median generated words", False)):
        if key in regular:
            by_condition = regular.groupby("condition")[key]
            values = by_condition.mean() * 100 if fmt else by_condition.median()
            bars(values, name, axis, out / f"{key}.png", percentage=fmt)
    key = "details.gold_rank"
    if key in regular:
        ranks = regular[regular["condition"].str.endswith("_passage") & regular["condition"].str.match("bge_m3|minilm")]
        ranks = ranks.drop_duplicates("sample_id")[key].dropna()
        bars(ranks.astype(int).value_counts().sort_index(), name, "Questions", out / "gold_passage_rank.png")
        if not ranks.empty:
            recall = pd.Series({f"Top {k}": 100 * ranks.le(k).mean() for k in (1, 3, 5, 10)})
            bars(recall, name, "Gold passage recall (%)", out / "passage_recall_at_k.png", percentage=True)
    stress = rows[rows["condition"].str.startswith("stress_")].copy()
    if not stress.empty:
        length, position = "details.target_prompt_tokens", "details.evidence_position"
        stress["success"] = stress["status"].eq("ok")
        stress[length] = stress[length].astype(int)
        coverage = stress.pivot_table(index=position, columns=length, values="success").reindex(
            [value for value in ("front", "middle", "end") if value in set(stress[position])],
        ) * 100
        heatmap(coverage, "Stress test | completion rate (%)", out / "stress_completion.png",
                xlabel="Target prompt tokens", ylabel="Evidence position", vmax=100)
        performance = stress.pivot_table(index=position, columns=length, values="scored") * 100
        heatmap(performance.reindex(index=coverage.index, columns=coverage.columns),
                "Stress test | ROUGE-L on completed trials (%)", out / "stress_quality.png",
                xlabel="Target prompt tokens", ylabel="Evidence position", vmax=100)
        scatter(stress, "details.gold_evidence_start_token", "scored", "Stress test | evidence location",
                "Evidence start token", "ROUGE-L F1 (%)", out / "stress_evidence_location.png", scale=100)


def plot_run(name, rows, out):
    status = pd.crosstab(rows["dataset"], rows["status"])
    bars(status, name, "Trials", out / "run_status.png")
    for task, task_rows in rows.groupby("task"):
        if task_rows["dataset"].nunique() > 1:
            table = task_rows.pivot_table(index="condition", columns="dataset", values="scored") * 100
            heatmap(table, f"{name} | {scores_label(task)}", out / f"{task}_overview.png", vmax=100)
    for dataset, group in rows.groupby("dataset"):
        target = out / slug(dataset)
        target.mkdir(exist_ok=True)
        task = group["task"].iloc[0]
        title = f"{rows['model'].iloc[0]} | {dataset.replace('_', ' ')}"
        grouped = group.groupby("condition")
        bars(grouped["scored"].mean() * 100, title, scores_label(task), target / "performance.png", percentage=True)
        bars(pd.crosstab(group["condition"], group["status"]), title, "Trials", target / "trial_status.png")
        for column, label in (("prompt_tokens", "Median prompt tokens"), ("completion_tokens", "Median completion tokens"),
                              ("elapsed_seconds", "Median elapsed seconds"), ("prompt_pct_of_native", "Median context used (%)")):
            if column in group:
                source = group[group["status"].eq("ok")] if column == "elapsed_seconds" else group
                bars(source.groupby("condition")[column].median(), title, label, target / f"{column}.png")
        if "stop_reason" in group:
            bars(pd.crosstab(group["condition"], group["stop_reason"]), title, "Trials", target / "stop_reasons.png")
        scatter(group[group["status"].eq("ok")], "prompt_tokens", "elapsed_seconds", title,
                "Prompt tokens", "Elapsed seconds", target / "prompt_vs_latency.png")
        scatter(group, "prompt_tokens", "completion_tokens", title,
                "Prompt tokens", "Completion tokens", target / "prompt_vs_completion.png")
        scored = group.dropna(subset=["scored"]).groupby("condition")[["prompt_tokens", "scored"]].mean()
        scatter(scored, "prompt_tokens", "scored", title,
                "Mean prompt tokens per condition", scores_label(task), target / "prompt_vs_quality.png", scale=100)
        baseline = {"classification": "chat_label_only_16", "sql": "full_schema", "qa": "gold_passage"}[task]
        paired_delta(group, baseline, f"{title} | relative to {baseline.replace('_', ' ')}", target / "paired_differences.png")
        {"classification": classification, "sql": sql, "qa": qa}[task](group, title, target)


def mean_matched_scores(trials):
    conditions = trials.groupby(level="condition")
    scores = conditions.mean() * 100
    scores.index = [f"{condition} (n={count})" for condition, count in conditions.size().items()]
    return scores


def comparisons(runs, out):
    # Pair only trials with identical rendered prompts within the same retrieval/weight setup.
    cohorts = {}
    for _, config, rows in runs:
        key = tuple(config.get(field, "unknown") for field in ("retriever_id", "retriever_revision", "weights"))
        cohorts.setdefault(key, []).append(rows)
    for number, (key, frames) in enumerate(cohorts.items(), 1):
        if len({frame["model"].iloc[0] for frame in frames}) < 2:
            continue
        folder = out / f"cross_model_{number}_{slug(key[0])}"
        folder.mkdir(exist_ok=True)
        combined = pd.concat(frames, ignore_index=True)
        model_count = combined["model"].nunique()
        for dataset, group in combined.groupby("dataset"):
            match_keys = ["condition", "sample_id", "prompt_sha256", "reference_key"]
            common = group.pivot_table(index=match_keys, columns="model", values="scored").dropna()
            if common.empty or len(common.columns) != model_count:
                continue
            bars(mean_matched_scores(common), f"{dataset.replace('_', ' ')} | matched prompts", scores_label(group["task"].iloc[0]),
                 folder / f"{slug(dataset)}.png", percentage=True)

            parser_scores = {"scored": "Accuracy", "label_found": "Label coverage"}
            zero_shot = ["legacy_raw_16", "chat_same_instruction_16", "chat_label_only_16"]
            if group["task"].iloc[0] == "classification" and set(parser_scores).issubset(group.columns):
                diagnostic = group[group["condition"].isin(zero_shot)].pivot_table(
                    index=match_keys, columns="model", values=list(parser_scores),
                ).reindex(columns=pd.MultiIndex.from_product([parser_scores, common.columns])).dropna()
                if diagnostic.empty:
                    continue
                by_parser = mean_matched_scores(diagnostic)
                by_parser = by_parser.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0, sort_remaining=False)
                by_parser.columns = [f"{model} · {parser_scores[metric]}" for model, metric in by_parser.columns]
                bars(by_parser, f"{dataset.replace('_', ' ')} | zero-shot label extraction", "Accuracy and coverage (%)",
                     folder / f"{slug(dataset)}_zero_shot_parsers.png", percentage=True)


def index_page(out):
    items = []
    for path in sorted(out.rglob("*.png")):
        relative = path.relative_to(out)
        label = html.escape(str(relative.with_suffix("")).replace("_", " "))
        src = html.escape(relative.as_posix(), quote=True)
        items.append(f'<li><a href="{src}">{label}<br><img src="{src}" loading="lazy" alt="{label}"></a></li>')
    (out / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Reviewer experiment plots</title>'
        '<style>body{font:16px system-ui;margin:2rem;color:#162d3b}ul{list-style:none;padding:0;display:grid;'
        'grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:1.4rem}img{width:100%}a{color:#285f83}'
        'li{border:1px solid #ddd;padding:.7rem}</style><h1>Reviewer experiment plots</h1>'
        f"<p>{len(items)} plots. Open a figure for its full resolution.</p><ul>{''.join(items)}</ul></html>",
        encoding="utf-8",
    )
    return len(items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="Export directories, parent directories, or result ZIPs.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/reviewer_plots"))
    args = parser.parse_args()
    runs = list(load_runs(args.inputs or [Path(".")]))
    if not runs:
        parser.error("No exported trials found. Give a directory or ZIP containing trials.jsonl and manifest.json.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    summary = []
    classification_trials = []
    for number, (name, _, rows) in enumerate(runs, 1):
        out = args.output_dir / f"{number:02d}_{slug(name)}"
        out.mkdir(exist_ok=True)
        plot_run(name, rows, out)
        grouped = rows.groupby(["run", "model", "retriever", "dataset", "task", "condition"], dropna=False)
        summary.append(grouped.agg(attempted=("status", "size"), completed=("status", lambda x: x.eq("ok").sum()),
                                   scored_n=("scored", "count"), mean_score=("scored", "mean"),
                                   median_prompt_tokens=("prompt_tokens", "median")).reset_index())
        classified = rows.loc[rows["task"].eq("classification")]
        if not classified.empty:
            columns = ["run", "model", "dataset", "sample_id", "condition", "status", "score_method",
                       "reference", "prediction", "is_correct", "parse_status", "raw_output"]
            classification_trials.append(classified.reindex(columns=columns))
    comparisons(runs, args.output_dir)
    pd.concat(summary, ignore_index=True).to_csv(args.output_dir / "condition_summary.csv", index=False)
    if classification_trials:
        pd.concat(classification_trials, ignore_index=True).to_csv(
            args.output_dir / "classification_trials.csv", index=False,
        )
    total = index_page(args.output_dir)
    print(f"Saved {total} graphs from {len(runs)} runs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
