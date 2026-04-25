from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from brain_tumor_fl.data import create_client_partitions, discover_dataset_layout


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
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.extend(_parse_json_stream(line))
    return rows


def select_last_run(rows: list[dict]) -> list[dict]:
    if not rows:
        return rows

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


def build_round_dataframe(round_metrics_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = select_last_run(load_jsonl(round_metrics_path))
    rounds: list[dict] = []
    clients: list[dict] = []
    for row in rows:
        round_id = int(row["round"])
        aggregated = dict(row.get("aggregated_metrics", {}))
        aggregated["round"] = round_id
        rounds.append(aggregated)
        for client in row.get("client_reports", []):
            partition_id = int(client.get("partition_id", -1))
            clients.append(
                {
                    "round": round_id,
                    "client_label": f"C{partition_id}" if partition_id >= 0 else str(client.get("client_id")),
                    **client,
                }
            )
    round_df = pd.DataFrame(rounds).sort_values("round").reset_index(drop=True)
    client_df = pd.DataFrame(clients)
    if not client_df.empty and {"client_label", "round"}.issubset(client_df.columns):
        client_df = client_df.sort_values(["client_label", "round"]).reset_index(drop=True)
    return round_df, client_df


def build_global_eval_dataframe(global_metrics_path: Path) -> pd.DataFrame:
    rows = select_last_run(load_jsonl(global_metrics_path))
    return pd.DataFrame(rows).sort_values("round").reset_index(drop=True)


def plot_accuracy_dynamics(
    round_df: pd.DataFrame,
    global_eval_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=round_df, x="round", y="train_accuracy", marker="o", label="Train Accuracy", ax=ax)
    sns.lineplot(data=global_eval_df, x="round", y="accuracy", marker="o", label="Test Accuracy", ax=ax)
    ax.set_title("Accuracy Dynamics Across Federated Rounds (non-IID)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_dynamics_non_iid.png", dpi=200)
    plt.close(fig)


def plot_loss_dynamics(
    round_df: pd.DataFrame,
    global_eval_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=round_df, x="round", y="train_loss", marker="o", label="Train Loss", ax=ax)
    sns.lineplot(data=global_eval_df, x="round", y="loss", marker="o", label="Test Loss", ax=ax)
    ax.set_title("Loss Dynamics Across Federated Rounds (non-IID)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_dynamics_non_iid.png", dpi=200)
    plt.close(fig)


def plot_f1_dynamics(
    round_df: pd.DataFrame,
    global_eval_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    local_f1_column = "train_f1" if "train_f1" in round_df.columns else "val_f1"
    local_f1_label = "Train F1-macro" if local_f1_column == "train_f1" else "Local Val F1-macro"
    sns.lineplot(data=round_df, x="round", y=local_f1_column, marker="o", label=local_f1_label, ax=ax)
    sns.lineplot(data=global_eval_df, x="round", y="f1_macro", marker="o", label="Test F1-macro", ax=ax)
    ax.set_title("F1-macro Dynamics Across Federated Rounds (non-IID)")
    ax.set_xlabel("Round")
    ax.set_ylabel("F1-macro")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "f1_macro_dynamics_non_iid.png", dpi=200)
    plt.close(fig)


def plot_train_loss_boxplot(client_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=client_df, x="round", y="train_loss", ax=ax)
    ax.set_title("Train Loss Distribution Across Clients")
    ax.set_xlabel("Round")
    ax.set_ylabel("Train Loss")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "train_loss_boxplot_clients.png", dpi=200)
    plt.close(fig)


def plot_trust_score_by_client(client_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(
        data=client_df,
        x="round",
        y="trust_score",
        hue="client_label",
        marker="o",
        ax=ax,
    )
    ax.set_title("Trust Score by Client Across Federated Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Trust Score")
    ax.set_ylim(0.0, max(1.05, float(client_df["trust_score"].max()) + 0.05))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Client", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "trust_score_by_client.png", dpi=200)
    plt.close(fig)


def plot_update_l2_norm_by_client(client_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(
        data=client_df,
        x="round",
        y="update_l2_norm",
        hue="client_label",
        marker="o",
        ax=ax,
    )
    ax.set_title("Update L2 Norm by Client Across Federated Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Update L2 Norm")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Client", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "update_l2_norm_by_client.png", dpi=200)
    plt.close(fig)


def build_partition_tables(
    dataset_root: Path,
    num_clients: int,
    partition_mode: str,
    alpha: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    layout = discover_dataset_layout(str(dataset_root), test_split=0.15, seed=seed)
    partitions = create_client_partitions(
        samples=layout.train_samples,
        num_clients=num_clients,
        partition_mode=partition_mode,
        dirichlet_alpha=alpha,
        seed=seed,
    )

    volume_rows: list[dict] = []
    class_rows: list[dict] = []
    for client_id, indices in enumerate(partitions):
        volume_rows.append({"client": f"C{client_id}", "num_samples": len(indices)})
        counts = {label: 0 for label in layout.classes}
        for idx in indices:
            _, class_id = layout.train_samples[idx]
            counts[layout.classes[class_id]] += 1
        for class_name, count in counts.items():
            class_rows.append(
                {"client": f"C{client_id}", "class_name": class_name, "count": count}
            )

    return pd.DataFrame(volume_rows), pd.DataFrame(class_rows)


def plot_data_volume_distribution(volume_df: pd.DataFrame, output_dir: Path, partition_mode: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=volume_df, x="client", y="num_samples", color="#4C78A8", ax=ax)
    ax.set_title(f"Distribution of Training Data Volume Across Clients ({partition_mode})")
    ax.set_xlabel("Client")
    ax.set_ylabel("Number of Training Samples")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"client_data_volume_{partition_mode}.png", dpi=200)
    plt.close(fig)


def plot_class_distribution(
    class_df: pd.DataFrame,
    output_dir: Path,
    partition_mode: str,
    alpha: float,
) -> None:
    pivot_df = class_df.pivot(index="client", columns="class_name", values="count").fillna(0)
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(pivot_df, annot=True, fmt=".0f", cmap="YlGnBu", cbar=True, ax=ax)
    if partition_mode == "dirichlet":
        title = f"Class Distribution Across Clients ({partition_mode}, alpha={alpha})"
        filename = f"class_distribution_{partition_mode}_alpha_{str(alpha).replace('.', '_')}.png"
    else:
        title = f"Class Distribution Across Clients ({partition_mode})"
        filename = f"class_distribution_{partition_mode}.png"
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Client")
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--round-metrics", default="outputs/round_metrics.jsonl")
    parser.add_argument("--global-metrics", default="outputs/global_eval_metrics.jsonl")
    parser.add_argument("--dataset-root", default="brain_tumor_mri")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--partition-mode", default="dirichlet")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/plots")
    args = parser.parse_args()

    if args.experiment_dir is not None:
        experiment_dir = Path(args.experiment_dir)
        round_metrics_path = experiment_dir / "round_metrics.jsonl"
        global_metrics_path = experiment_dir / "global_eval_metrics.jsonl"
        output_dir = experiment_dir / "plots"
    else:
        round_metrics_path = Path(args.round_metrics)
        global_metrics_path = Path(args.global_metrics)
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    round_df, client_df = build_round_dataframe(round_metrics_path)
    global_eval_df = build_global_eval_dataframe(global_metrics_path)

    plot_accuracy_dynamics(round_df, global_eval_df, output_dir)
    plot_loss_dynamics(round_df, global_eval_df, output_dir)
    plot_f1_dynamics(round_df, global_eval_df, output_dir)
    if not client_df.empty and "train_loss" in client_df.columns:
        plot_train_loss_boxplot(client_df, output_dir)
    if not client_df.empty and "trust_score" in client_df.columns:
        plot_trust_score_by_client(client_df, output_dir)
    if not client_df.empty and "update_l2_norm" in client_df.columns:
        plot_update_l2_norm_by_client(client_df, output_dir)

    if args.partition_mode != "local" and args.num_clients > 1:
        volume_df, class_df = build_partition_tables(
            dataset_root=Path(args.dataset_root),
            num_clients=args.num_clients,
            partition_mode=args.partition_mode,
            alpha=args.alpha,
            seed=args.seed,
        )
        plot_data_volume_distribution(volume_df, output_dir, args.partition_mode)
        plot_class_distribution(class_df, output_dir, args.partition_mode, args.alpha)

    print("Saved plots to:", output_dir)


if __name__ == "__main__":
    main()
