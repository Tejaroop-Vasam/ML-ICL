"""Command-line launcher used by Colab and other GPU environments."""

import argparse
from dataclasses import fields
from pathlib import Path

from .config import DEFAULT_MODEL, DEFAULT_SAMPLES, WEIGHT_MODES, ReviewConfig


def sample_count(value):
    if value.casefold() == "all":
        return None
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use a nonnegative integer or 'all'.") from exc
    if count < 0:
        raise argparse.ArgumentTypeError("Use a nonnegative integer or 'all'.")
    return count


def sample_setting(value):
    dataset, separator, count = value.partition("=")
    if not separator or dataset not in DEFAULT_SAMPLES:
        raise argparse.ArgumentTypeError("Use DATASET=COUNT with a supported dataset.")
    return dataset, sample_count(count)


def build_parser():
    defaults = ReviewConfig()
    parser = argparse.ArgumentParser(description="Run the Qwen3 reviewer experiments and save raw trial evidence.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run or resume selected experiments.")
    run.add_argument("--datasets", nargs="+", choices=tuple(DEFAULT_SAMPLES),
                     help="Datasets to include. Defaults to all.")
    run.add_argument("--samples", action="append", type=sample_setting, metavar="DATASET=COUNT",
                     help="Override a pilot sample count. Repeat as needed or use 'all'.")
    run.add_argument("--model", action="append", help=f"Qwen3 Hub ID. Repeat for multiple models. Default: {DEFAULT_MODEL}.")
    run.add_argument("--weights", choices=WEIGHT_MODES, default="nf4", help="Model precision. Default: nf4.")
    run.add_argument("--profile-only", action="store_true", help="Build and count prompts without loading Qwen3 weights.")
    run.add_argument("--retry-oom", action="store_true", help="Retry previously recorded GPU out-of-memory trials.")
    run.add_argument("--skip-export", action="store_true", help="Keep the checkpoint and skip packaging raw files.")
    run.add_argument("--stress-n", type=sample_count, default=defaults.stress_n,
                     help="Number of CLAPNQ questions for the long-context sweep. Use 0 to skip.")
    run.add_argument("--stress-prompt-targets", nargs="+", type=int, default=defaults.stress_prompt_targets)
    run.add_argument("--k-sweep", nargs="+", type=int, default=defaults.k_sweep)
    run.add_argument("--random-seeds", nargs="+", type=int, default=defaults.random_seeds)
    for name in ("seed", "pool_size", "reference_k", "imdb_slice_tokens", "sql_demo_k",
                 "qa_demo_k", "qa_candidates", "min_qa_reference_words", "max_prompt_tokens"):
        run.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("cache_root", "output_root", "spider_path", "bird_dev_path", "bird_train_path"):
        run.add_argument("--" + name.replace("_", "-"), type=Path, default=getattr(defaults, name))

    export = commands.add_parser("export", help="Export raw evidence from existing checkpoint directories.")
    export.add_argument("run_dirs", nargs="+", type=Path, metavar="RUN_DIR")
    return parser


def config_from_args(args, parser):
    active = set(args.datasets) if args.datasets is not None else set(DEFAULT_SAMPLES)
    samples = {name: count if name in active else 0 for name, count in DEFAULT_SAMPLES.items()}
    for dataset, count in args.samples or ():
        if dataset not in active:
            parser.error(f"Sample override for unselected dataset: {dataset}")
        samples[dataset] = count

    settings = {item.name: getattr(args, item.name) for item in fields(ReviewConfig) if item.name != "samples"}
    for name in ("stress_prompt_targets", "random_seeds", "k_sweep"):
        settings[name] = tuple(settings[name])
    try:
        return ReviewConfig(samples=samples, **settings)
    except ValueError as exc:
        parser.error(str(exc))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "export":
        from .artifacts import export_raw

        for run_dir in args.run_dirs:
            print(f"Raw evidence: {export_raw(run_dir)}")
        return

    config = config_from_args(args, parser)
    import torch

    if not config.profile_only and not torch.cuda.is_available():
        parser.error("A CUDA runtime is required for inference. Use --profile-only for prompt accounting.")

    from .artifacts import export_raw
    from .common import Retriever
    from .runner import run_jobs

    retriever = Retriever(device="cpu" if config.profile_only else "cuda")
    for model_id in dict.fromkeys(args.model or (DEFAULT_MODEL,)):
        run_dir = run_jobs(config, model_id, args.weights, retriever)
        print(f"Checkpoint: {run_dir / 'checkpoint.sqlite'}")
        if not args.skip_export:
            print(f"Raw evidence: {export_raw(run_dir)}")


if __name__ == "__main__":
    main()
