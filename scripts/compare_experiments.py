from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _parse_json_stream(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    rows: list[dict] = []
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        row, next_index = decoder.raw_decode(text, index)
        rows.append(row)
        index = next_index
    return rows


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.extend(_parse_json_stream(line))
    return rows


def select_last_run(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    runs: list[list[dict]] = []
    current_run: list[dict] = []
    prev_round: int | None = None
    for row in rows:
        round_id = int(row["round"])
        if prev_round is not None and round_id <= prev_round:
            runs.append(current_run)
            current_run = []
        current_run.append(row)
        prev_round = round_id
    if current_run:
        runs.append(current_run)
    return runs[-1]


def parse_experiment(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Experiment must be formatted as label=experiment_dir"
        )
    label, raw_path = value.split("=", maxsplit=1)
    label = label.strip()
    path = Path(raw_path.strip())
    if not label or not str(path):
        raise argparse.ArgumentTypeError(
            "Experiment label and path must both be non-empty"
        )
    return label, path


def build_dataframe(experiments: list[tuple[str, Path]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[Path] = []
    for label, experiment_dir in experiments:
        metrics_path = experiment_dir / "global_eval_metrics.jsonl"
        if not metrics_path.exists():
            missing.append(metrics_path)
            continue
        rows = select_last_run(load_jsonl(metrics_path))
        if not rows:
            continue
        frame = pd.DataFrame(rows).sort_values("round").reset_index(drop=True)
        frame["method"] = label
        frames.append(frame)

    if missing:
        print("Skipped missing metrics files:")
        for path in missing:
            print(f"  {path}")

    if not frames:
        raise FileNotFoundError("No global_eval_metrics.jsonl files were found.")
    return pd.concat(frames, ignore_index=True)


def plot_metric(df: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=df, x="round", y=metric, hue="method", marker="o")
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare experiments using global evaluation metrics only."
    )
    parser.add_argument(
        "--experiment",
        action="append",
        type=parse_experiment,
        required=True,
        help="Method label and experiment directory, formatted as label=path.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/experiments/comparisons",
        help="Directory for comparison plots and summary table.",
    )
    parser.add_argument("--title-prefix", default="Method Progress")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(args.experiment)
    df.to_csv(output_dir / "method_global_eval_summary.csv", index=False)

    if "accuracy" in df.columns:
        plot_metric(
            df,
            "accuracy",
            output_dir / "method_accuracy_progress.png",
            f"{args.title_prefix}: Accuracy",
        )
    if "f1_macro" in df.columns:
        plot_metric(
            df,
            "f1_macro",
            output_dir / "method_f1_macro_progress.png",
            f"{args.title_prefix}: F1-macro",
        )
    if "loss" in df.columns:
        plot_metric(
            df,
            "loss",
            output_dir / "method_loss_progress.png",
            f"{args.title_prefix}: Loss",
        )

    print(f"Saved comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()
