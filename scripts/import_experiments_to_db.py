from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch

from brain_tumor_fl.db import (
    ArtifactRecord,
    ClientMetricRecord,
    ExperimentRecord,
    RoundMetricRecord,
    create_database_session_factory,
)


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


def infer_dataset_name(experiment_dir: Path, checkpoint_payload: dict[str, Any] | None) -> str:
    if checkpoint_payload and checkpoint_payload.get("dataset_root"):
        return str(checkpoint_payload["dataset_root"])
    parts = {part.lower() for part in experiment_dir.parts}
    if "cifar100" in parts:
        return "cifar100"
    if "cifar10" in parts:
        return "cifar10"
    return "brain_tumor_mri"


def infer_model_name(experiment_dir: Path, checkpoint_payload: dict[str, Any] | None) -> str:
    if checkpoint_payload and checkpoint_payload.get("model_name"):
        return str(checkpoint_payload["model_name"])
    if "resnet50" in experiment_dir.name.lower():
        return "resnet50"
    return "efficientnet_b0"


def infer_partition_mode(experiment_dir: Path) -> str:
    name = experiment_dir.name.lower()
    if "dirichlet" in name:
        return "dirichlet"
    if "soft" in name:
        return "shards_quantity_skew_soft"
    return "shards_quantity_skew"


def infer_topology_mode(last_round_rows: list[dict], experiment_dir: Path, checkpoint_payload: dict[str, Any] | None) -> str:
    if last_round_rows and last_round_rows[-1].get("topology_mode"):
        return str(last_round_rows[-1]["topology_mode"])
    if checkpoint_payload and checkpoint_payload.get("topology_mode"):
        return str(checkpoint_payload["topology_mode"])
    name = experiment_dir.name.lower()
    if "full_graph" in name:
        return "full_graph"
    if "augmented_ring" in name:
        return "augmented_ring"
    return "ring"


def infer_mode(last_round_rows: list[dict], checkpoint_payload: dict[str, Any] | None) -> str:
    if last_round_rows and last_round_rows[-1].get("mode"):
        return str(last_round_rows[-1]["mode"])
    if checkpoint_payload and checkpoint_payload.get("mode"):
        return str(checkpoint_payload["mode"])
    return "decentralized"


def infer_strategy(last_round_rows: list[dict], checkpoint_payload: dict[str, Any] | None) -> str | None:
    if last_round_rows and last_round_rows[-1].get("strategy_name"):
        return str(last_round_rows[-1]["strategy_name"])
    if checkpoint_payload and checkpoint_payload.get("strategy_name"):
        return str(checkpoint_payload["strategy_name"])
    return None


def infer_soft_mix_ratio(experiment_dir: Path, partition_mode: str) -> float | None:
    if partition_mode != "shards_quantity_skew_soft":
        return None
    name = experiment_dir.name.lower()
    if "soft20" in name:
        return 0.20
    return 0.15


def infer_soft_min_extra_classes(experiment_dir: Path, dataset_name: str, partition_mode: str) -> int | None:
    if partition_mode != "shards_quantity_skew_soft":
        return None
    name = experiment_dir.name.lower()
    if "soft20" in name:
        return 12
    if dataset_name == "cifar100":
        return 8
    return 5


def infer_async_dropout_rate(experiment_dir: Path, async_mode: bool) -> float | None:
    if not async_mode:
        return 0.0
    name = experiment_dir.name.lower()
    if "_d20" in name or name.endswith("d20"):
        return 0.2
    if "soft10" in name or "_d10" in name or name.endswith("d10"):
        return 0.1
    return 0.0


def infer_batch_size(dataset_name: str, experiment_dir: Path) -> int:
    name = experiment_dir.name.lower()
    if "db_smoke" in name:
        return 8
    if dataset_name.startswith("cifar"):
        return 32
    return 16


def infer_local_epochs(experiment_dir: Path) -> int:
    if "db_smoke" in experiment_dir.name.lower():
        return 1
    return 2


def infer_learning_rate(dataset_name: str) -> float:
    if dataset_name.startswith("cifar"):
        return 0.0005
    return 0.0002


def infer_weight_decay(dataset_name: str) -> float:
    if dataset_name.startswith("cifar"):
        return 0.0001
    return 0.00001


def collect_artifacts(experiment_dir: Path) -> list[tuple[str, Path, str]]:
    artifacts: list[tuple[str, Path, str]] = []
    metrics_path = experiment_dir / "round_metrics.jsonl"
    global_metrics_path = experiment_dir / "global_eval_metrics.jsonl"
    analysis_path = experiment_dir / "experiment_analysis.txt"
    latest_checkpoint = experiment_dir / "checkpoints" / "latest_model.pt"
    best_checkpoint = experiment_dir / "checkpoints" / "best_model.pt"

    if metrics_path.exists():
        artifacts.append(("metrics_jsonl", metrics_path, "Per-round aggregated and client metrics"))
    if global_metrics_path.exists():
        artifacts.append(("global_metrics_jsonl", global_metrics_path, "Global evaluation metrics by round"))
    if latest_checkpoint.exists():
        artifacts.append(("checkpoint", latest_checkpoint, "Latest global checkpoint"))
    if best_checkpoint.exists():
        artifacts.append(("checkpoint", best_checkpoint, "Best checkpoint"))
    if analysis_path.exists():
        artifacts.append(("analysis", analysis_path, "Text analysis report"))

    plots_dir = experiment_dir / "plots"
    if plots_dir.exists():
        for plot_path in sorted(plots_dir.glob("*.png")):
            artifacts.append(("plot", plot_path, plot_path.name))

    return artifacts


def import_experiment(session_factory, experiment_dir: Path, force: bool = False) -> tuple[bool, str]:
    metrics_path = experiment_dir / "round_metrics.jsonl"
    global_metrics_path = experiment_dir / "global_eval_metrics.jsonl"
    if not metrics_path.exists() or not global_metrics_path.exists():
        return False, f"skip missing metrics: {experiment_dir}"

    round_rows = select_last_run(load_jsonl(metrics_path))
    eval_rows = select_last_run(load_jsonl(global_metrics_path))
    if not round_rows or not eval_rows:
        return False, f"skip empty metrics: {experiment_dir}"

    latest_checkpoint = experiment_dir / "checkpoints" / "latest_model.pt"
    best_checkpoint = experiment_dir / "checkpoints" / "best_model.pt"
    checkpoint_payload = None
    checkpoint_source = best_checkpoint if best_checkpoint.exists() else latest_checkpoint
    if checkpoint_source.exists():
        checkpoint_payload = torch.load(checkpoint_source, map_location="cpu")

    dataset_name = infer_dataset_name(experiment_dir, checkpoint_payload)
    model_name = infer_model_name(experiment_dir, checkpoint_payload)
    partition_mode = infer_partition_mode(experiment_dir)
    topology_mode = infer_topology_mode(round_rows, experiment_dir, checkpoint_payload)
    mode = infer_mode(round_rows, checkpoint_payload)
    strategy_name = infer_strategy(round_rows, checkpoint_payload)
    async_mode = bool(round_rows[-1].get("async_mode", "async" in experiment_dir.name.lower()))
    num_clients = max(
        int(client.get("partition_id", -1))
        for row in round_rows
        for client in row.get("client_reports", [])
    ) + 1
    num_server_rounds = int(eval_rows[-1]["round"])

    metadata = {
        "dataset-root": dataset_name,
        "model-name": model_name,
        "partition-mode": partition_mode,
        "dirichlet-alpha": 0.5 if partition_mode == "dirichlet" else None,
        "soft-mix-ratio": infer_soft_mix_ratio(experiment_dir, partition_mode),
        "soft-min-extra-classes": infer_soft_min_extra_classes(experiment_dir, dataset_name, partition_mode),
        "topology-mode": topology_mode,
        "topology-extra-offset": 2 if topology_mode == "augmented_ring" else 0,
        "async-mode": async_mode,
        "async-dropout-rate": infer_async_dropout_rate(experiment_dir, async_mode),
        "max-async-dropouts": 2 if async_mode else 0,
        "heterogeneous-nodes": "heterogeneous" in experiment_dir.name.lower(),
        "num-clients": num_clients,
        "num-server-rounds": num_server_rounds,
        "local-epochs": infer_local_epochs(experiment_dir),
        "batch-size": infer_batch_size(dataset_name, experiment_dir),
        "learning-rate": infer_learning_rate(dataset_name),
        "weight-decay": infer_weight_decay(dataset_name),
        "use-pretrained": bool(checkpoint_payload.get("use_pretrained", False)) if checkpoint_payload else False,
        "save-metrics-path": str(metrics_path.resolve()),
        "imported": True,
    }

    best_accuracy = max(float(row.get("accuracy", 0.0)) for row in eval_rows)
    best_f1_macro = max(float(row.get("f1_macro", 0.0)) for row in eval_rows)
    best_loss = min(float(row.get("loss", float("inf"))) for row in eval_rows)

    with session_factory() as session:
        existing = session.query(ExperimentRecord).filter_by(experiment_dir=str(experiment_dir.resolve())).first()
        if existing is not None:
            if not force:
                return False, f"skip already imported: {experiment_dir.name}"
            session.delete(existing)
            session.commit()

        experiment = ExperimentRecord(
            experiment_name=experiment_dir.name,
            experiment_dir=str(experiment_dir.resolve()),
            mode=mode,
            strategy_name=strategy_name,
            dataset_name=dataset_name,
            model_name=model_name,
            partition_mode=partition_mode,
            dirichlet_alpha=metadata["dirichlet-alpha"],
            soft_mix_ratio=metadata["soft-mix-ratio"],
            soft_min_extra_classes=metadata["soft-min-extra-classes"],
            topology_mode=topology_mode,
            topology_extra_offset=metadata["topology-extra-offset"],
            async_mode=async_mode,
            async_dropout_rate=metadata["async-dropout-rate"],
            max_async_dropouts=metadata["max-async-dropouts"],
            heterogeneous_nodes=bool(metadata["heterogeneous-nodes"]),
            num_clients=num_clients,
            num_server_rounds=num_server_rounds,
            local_epochs=int(metadata["local-epochs"]),
            batch_size=int(metadata["batch-size"]),
            learning_rate=float(metadata["learning-rate"]),
            weight_decay=float(metadata["weight-decay"]),
            use_pretrained=bool(metadata["use-pretrained"]),
            status="completed",
            best_accuracy=best_accuracy,
            best_f1_macro=best_f1_macro,
            best_loss=best_loss,
            raw_config_json=json.dumps(metadata, ensure_ascii=True, sort_keys=True),
        )
        session.add(experiment)
        session.flush()

        eval_by_round = {int(row["round"]): row for row in eval_rows}
        imported_round_numbers: set[int] = set()
        for row in round_rows:
            round_number = int(row["round"])
            imported_round_numbers.add(round_number)
            aggregated = dict(row.get("aggregated_metrics", {}))
            global_row = eval_by_round.get(round_number)
            session.add(
                RoundMetricRecord(
                    experiment_id=int(experiment.id),
                    round_number=round_number,
                    train_loss=_float_or_none(aggregated.get("train_loss")),
                    train_accuracy=_float_or_none(aggregated.get("train_accuracy")),
                    train_f1=_float_or_none(aggregated.get("train_f1")),
                    val_loss=_float_or_none(aggregated.get("val_loss")),
                    val_accuracy=_float_or_none(aggregated.get("val_accuracy")),
                    val_f1=_float_or_none(aggregated.get("val_f1")),
                    participating_clients=_float_or_none(aggregated.get("participating_clients")),
                    skipped_clients=_float_or_none(aggregated.get("skipped_clients")),
                    global_loss=_float_or_none(global_row.get("loss")) if global_row else None,
                    global_accuracy=_float_or_none(global_row.get("accuracy")) if global_row else None,
                    global_f1_macro=_float_or_none(global_row.get("f1_macro")) if global_row else None,
                    global_precision_macro=_float_or_none(global_row.get("precision_macro")) if global_row else None,
                    global_recall_macro=_float_or_none(global_row.get("recall_macro")) if global_row else None,
                )
            )

            for report in row.get("client_reports", []):
                session.add(
                    ClientMetricRecord(
                        experiment_id=int(experiment.id),
                        round_number=round_number,
                        client_name=str(report.get("client_id", "")),
                        partition_id=_int_or_none(report.get("partition_id")),
                        num_examples=_int_or_none(report.get("num_examples")),
                        trust_score=_float_or_none(report.get("trust_score")),
                        resource_profile=_str_or_none(report.get("resource_profile")),
                        resource_batch_size=_int_or_none(report.get("resource_batch_size")),
                        virtual_delay_sec=_float_or_none(report.get("virtual_delay_sec")),
                        train_loss=_float_or_none(report.get("train_loss")),
                        train_accuracy=_float_or_none(report.get("train_accuracy")),
                        train_f1=_float_or_none(report.get("train_f1")),
                        val_loss=_float_or_none(report.get("val_loss")),
                        val_accuracy=_float_or_none(report.get("val_accuracy")),
                        val_f1=_float_or_none(report.get("val_f1")),
                        train_time_sec=_float_or_none(report.get("train_time_sec")),
                        update_l2_norm=_float_or_none(report.get("update_l2_norm")),
                        skipped=bool(report.get("skipped", False)),
                        completed_round=_int_or_none(report.get("completed_round")),
                        server_round=_int_or_none(report.get("server_round")),
                    )
                )

        for round_number, global_row in eval_by_round.items():
            if round_number in imported_round_numbers:
                continue
            session.add(
                RoundMetricRecord(
                    experiment_id=int(experiment.id),
                    round_number=round_number,
                    global_loss=_float_or_none(global_row.get("loss")),
                    global_accuracy=_float_or_none(global_row.get("accuracy")),
                    global_f1_macro=_float_or_none(global_row.get("f1_macro")),
                    global_precision_macro=_float_or_none(global_row.get("precision_macro")),
                    global_recall_macro=_float_or_none(global_row.get("recall_macro")),
                )
            )

        for artifact_type, file_path, description in collect_artifacts(experiment_dir):
            session.add(
                ArtifactRecord(
                    experiment_id=int(experiment.id),
                    artifact_type=artifact_type,
                    file_path=str(file_path.resolve()),
                    description=description,
                )
            )

        session.commit()
        return True, f"imported: {experiment_dir.name}"


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def discover_experiment_dirs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob("round_metrics.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Import existing experiment folders into Neon/PostgreSQL.")
    parser.add_argument("--root", default="outputs/experiments")
    parser.add_argument("--force", action="store_true", help="Reimport even if experiment_dir already exists in DB.")
    args = parser.parse_args()

    root = Path(args.root)
    session_factory = create_database_session_factory()
    experiment_dirs = discover_experiment_dirs(root)

    imported = 0
    skipped = 0
    for experiment_dir in experiment_dirs:
        ok, message = import_experiment(session_factory, experiment_dir, force=args.force)
        print(message)
        if ok:
            imported += 1
        else:
            skipped += 1

    print(f"\nDone. Imported={imported}, skipped={skipped}")


if __name__ == "__main__":
    main()
