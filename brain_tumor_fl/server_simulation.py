from __future__ import annotations

import argparse
import json
from logging import INFO
from pathlib import Path
from typing import Any

import torch
from flwr.common.logger import log

from brain_tumor_fl.agents.aggregation_agent import AggregationAgent
from brain_tumor_fl.agents.compute_agent import ComputeAgent
from brain_tumor_fl.agents.monitoring_agent import MonitoringAgent
from brain_tumor_fl.agents.storage_agent import StorageAgent
from brain_tumor_fl.data import discover_dataset_layout
from brain_tumor_fl.db import ExperimentDatabaseRecorder
from brain_tumor_fl.model import build_model
from brain_tumor_fl.training import evaluate_model, get_device, get_parameters, set_parameters
from brain_tumor_fl.utils import coerce_bool


NDArrays = list[Any]


def _clone_ndarrays(parameters: NDArrays) -> NDArrays:
    return [layer.copy() for layer in parameters]


def _weighted_average_metrics(metrics_with_examples: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    total_examples = sum(num_examples for num_examples, _ in metrics_with_examples)
    if total_examples == 0:
        return {}

    keys = set().union(*(metrics.keys() for _, metrics in metrics_with_examples))
    aggregated: dict[str, float] = {}
    for key in keys:
        weighted_sum = 0.0
        for num_examples, metrics in metrics_with_examples:
            if key in metrics:
                weighted_sum += float(metrics[key]) * num_examples
        aggregated[key] = weighted_sum / total_examples
    return aggregated


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parameter-server federated simulation.")
    parser.add_argument("--dataset-root", default="brain_tumor_mri")
    parser.add_argument("--num-server-rounds", type=int, default=30)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--partition-mode", default="dirichlet")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--soft-mix-ratio", type=float, default=0.15)
    parser.add_argument("--soft-min-extra-classes", type=int, default=5)
    parser.add_argument("--strategy-name", default="fedavg")
    parser.add_argument("--proximal-mu", type=float, default=0.01)
    parser.add_argument("--use-pretrained", default="false")
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--save-metrics-path", default="outputs/round_metrics.jsonl")
    return parser.parse_args()


def _save_checkpoint(
    model,
    checkpoint_path: Path,
    round_number: int,
    loss: float,
    metrics: dict[str, float],
    classes: list[str],
    dataset_root: str,
    use_pretrained: bool,
    strategy_name: str,
) -> None:
    payload = {
        "round": round_number,
        "loss": loss,
        "metrics": metrics,
        "classes": classes,
        "model_name": "efficientnet_b0",
        "use_pretrained": use_pretrained,
        "dataset_root": dataset_root,
        "mode": "server",
        "strategy_name": strategy_name,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, checkpoint_path)


def _log_final_summary(
    train_history: list[dict[str, float]],
    eval_history: list[dict[str, float]],
    strategy_name: str,
) -> None:
    log(INFO, "[SUMMARY | %s]", strategy_name)
    log(INFO, "History (metrics, distributed fit):")
    for item in train_history:
        log(
            INFO,
            (
                "round %s: train_loss=%.4f, train_accuracy=%.4f, "
                "val_loss=%.4f, val_accuracy=%.4f, val_f1=%.4f"
            ),
            int(item["round"]),
            float(item["train_loss"]),
            float(item["train_accuracy"]),
            float(item["val_loss"]),
            float(item["val_accuracy"]),
            float(item["val_f1"]),
        )
    log(INFO, "History (metrics, global evaluate):")
    for item in eval_history:
        log(
            INFO,
            "round %s: loss=%.4f, accuracy=%.4f, f1_macro=%.4f",
            int(item["round"]),
            float(item["loss"]),
            float(item["accuracy"]),
            float(item["f1_macro"]),
        )


def main() -> None:
    db_recorder: ExperimentDatabaseRecorder | None = None
    try:
        args = _parse_args()
        strategy_name = str(args.strategy_name).lower()
        if strategy_name not in {"fedavg", "fedprox"}:
            raise ValueError(f"Unsupported server strategy: {args.strategy_name}")

        run_config = {
            "dataset-root": args.dataset_root,
            "num-server-rounds": args.num_server_rounds,
            "num-clients": args.num_clients,
            "partition-mode": args.partition_mode,
            "dirichlet-alpha": args.dirichlet_alpha,
            "soft-mix-ratio": args.soft_mix_ratio,
            "soft-min-extra-classes": args.soft_min_extra_classes,
            "use-pretrained": coerce_bool(args.use_pretrained),
            "local-epochs": args.local_epochs,
            "batch-size": args.batch_size,
            "learning-rate": args.learning_rate,
            "weight-decay": args.weight_decay,
            "val-split": args.val_split,
            "test-split": args.test_split,
            "num-workers": args.num_workers,
            "random-seed": args.random_seed,
        }

        metrics_path = Path(args.save_metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        global_metrics_path = metrics_path.with_name("global_eval_metrics.jsonl")
        checkpoints_dir = metrics_path.parent / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        best_checkpoint_path = checkpoints_dir / "best_model.pt"
        latest_checkpoint_path = checkpoints_dir / "latest_model.pt"

        storage_agent = StorageAgent(run_config)
        global_test_loader, num_classes = storage_agent.load_global_test_loader()
        layout = discover_dataset_layout(
            dataset_root=args.dataset_root,
            test_split=args.test_split,
            seed=args.random_seed,
        )
        use_pretrained = coerce_bool(args.use_pretrained)
        nodes = [ComputeAgent(partition_id=partition_id, config=run_config) for partition_id in range(args.num_clients)]
        monitoring_agent = MonitoringAgent()
        aggregation_agent = AggregationAgent(decentralized_mode=False)
        eval_model = build_model(num_classes=num_classes, use_pretrained=use_pretrained)
        device = get_device()
        eval_model.to(device)
        global_parameters = get_parameters(eval_model)
        best_f1_macro = float("-inf")
        train_history: list[dict[str, float]] = []
        eval_history: list[dict[str, float]] = []
        db_recorder = ExperimentDatabaseRecorder(
            run_config={**run_config, "save-metrics-path": args.save_metrics_path, "model-name": "efficientnet_b0"},
            mode="server",
            strategy_name=strategy_name,
        )

        log(
            INFO,
            "[ServerSimulation] initialized %s clients with strategy=%s",
            len(nodes),
            strategy_name,
        )

        initial_metrics = evaluate_model(eval_model, global_test_loader, device)
        with global_metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "round": 0,
                        "loss": float(initial_metrics["loss"]),
                        "accuracy": float(initial_metrics["accuracy"]),
                        "precision_macro": float(initial_metrics["precision_macro"]),
                        "recall_macro": float(initial_metrics["recall_macro"]),
                        "f1_macro": float(initial_metrics["f1_macro"]),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
        db_recorder.record_global_eval(
            round_number=0,
            loss=float(initial_metrics["loss"]),
            metrics={
                "accuracy": float(initial_metrics["accuracy"]),
                "precision_macro": float(initial_metrics["precision_macro"]),
                "recall_macro": float(initial_metrics["recall_macro"]),
                "f1_macro": float(initial_metrics["f1_macro"]),
            },
        )
        eval_history.append(
            {
                "round": 0,
                "loss": float(initial_metrics["loss"]),
                "accuracy": float(initial_metrics["accuracy"]),
                "f1_macro": float(initial_metrics["f1_macro"]),
            }
        )

        for round_number in range(1, args.num_server_rounds + 1):
            learning_rate = float(args.learning_rate) * (0.97 ** max(round_number - 1, 0))
            client_reports: list[dict[str, Any]] = []
            metrics_with_examples: list[tuple[int, dict[str, float]]] = []
            weighted_updates: list[tuple[NDArrays, float]] = []

            for node in nodes:
                local_parameters, num_examples, metrics = node.fit(
                    _clone_ndarrays(global_parameters),
                    {
                        "server_round": round_number,
                        "learning_rate": learning_rate,
                        "local_epochs": args.local_epochs,
                        "weight_decay": args.weight_decay,
                        "proximal_mu": args.proximal_mu if strategy_name == "fedprox" else 0.0,
                    },
                )
                trust_score = monitoring_agent.score_client(
                    client_id=f"C{node.partition_id}",
                    metrics=metrics,
                )
                client_reports.append(
                    {
                        "client_id": f"C{node.partition_id}",
                        "partition_id": node.partition_id,
                        "num_examples": num_examples,
                        "trust_score": trust_score,
                        **metrics,
                    }
                )
                metrics_with_examples.append(
                    (
                        num_examples,
                        {
                            "train_loss": float(metrics["train_loss"]),
                            "train_accuracy": float(metrics["train_accuracy"]),
                            "train_f1": float(metrics["train_f1"]),
                            "val_loss": float(metrics["val_loss"]),
                            "val_accuracy": float(metrics["val_accuracy"]),
                            "val_f1": float(metrics["val_f1"]),
                            "train_time_sec": float(metrics["train_time_sec"]),
                            "update_l2_norm": float(metrics["update_l2_norm"]),
                        },
                    )
                )
                weighted_updates.append((_clone_ndarrays(local_parameters), float(max(num_examples, 1))))

            global_parameters = aggregation_agent.aggregate(weighted_updates)
            aggregated_metrics = _weighted_average_metrics(metrics_with_examples)
            aggregated_metrics["participating_clients"] = float(len(nodes))
            train_history.append(
                {
                    "round": round_number,
                    "train_loss": float(aggregated_metrics.get("train_loss", 0.0)),
                    "train_accuracy": float(aggregated_metrics.get("train_accuracy", 0.0)),
                    "val_loss": float(aggregated_metrics.get("val_loss", 0.0)),
                    "val_accuracy": float(aggregated_metrics.get("val_accuracy", 0.0)),
                    "val_f1": float(aggregated_metrics.get("val_f1", 0.0)),
                }
            )

            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "round": round_number,
                            "mode": "server",
                            "strategy_name": strategy_name,
                            "aggregated_metrics": aggregated_metrics,
                            "client_reports": client_reports,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
            db_recorder.record_round(
                round_number=round_number,
                aggregated_metrics=dict(aggregated_metrics),
                client_reports=client_reports,
            )

            set_parameters(eval_model, global_parameters)
            eval_metrics = evaluate_model(eval_model, global_test_loader, device)
            scalar_metrics = {
                "accuracy": float(eval_metrics["accuracy"]),
                "precision_macro": float(eval_metrics["precision_macro"]),
                "recall_macro": float(eval_metrics["recall_macro"]),
                "f1_macro": float(eval_metrics["f1_macro"]),
            }
            with global_metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "round": round_number,
                            "loss": float(eval_metrics["loss"]),
                            **scalar_metrics,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
            db_recorder.record_global_eval(
                round_number=round_number,
                loss=float(eval_metrics["loss"]),
                metrics=scalar_metrics,
            )
            eval_history.append(
                {
                    "round": round_number,
                    "loss": float(eval_metrics["loss"]),
                    "accuracy": float(eval_metrics["accuracy"]),
                    "f1_macro": float(eval_metrics["f1_macro"]),
                }
            )

            _save_checkpoint(
                model=eval_model,
                checkpoint_path=latest_checkpoint_path,
                round_number=round_number,
                loss=float(eval_metrics["loss"]),
                metrics=scalar_metrics,
                classes=layout.classes,
                dataset_root=args.dataset_root,
                use_pretrained=use_pretrained,
                strategy_name=strategy_name,
            )
            if scalar_metrics["f1_macro"] >= best_f1_macro:
                best_f1_macro = scalar_metrics["f1_macro"]
                _save_checkpoint(
                    model=eval_model,
                    checkpoint_path=best_checkpoint_path,
                    round_number=round_number,
                    loss=float(eval_metrics["loss"]),
                    metrics=scalar_metrics,
                    classes=layout.classes,
                    dataset_root=args.dataset_root,
                    use_pretrained=use_pretrained,
                    strategy_name=strategy_name,
                )

            log(
                INFO,
                (
                    "[ServerSimulation | %s | round=%s] accuracy=%.4f, f1=%.4f, loss=%.4f"
                ),
                strategy_name,
                round_number,
                scalar_metrics["accuracy"],
                scalar_metrics["f1_macro"],
                float(eval_metrics["loss"]),
            )

        _log_final_summary(
            train_history=train_history,
            eval_history=eval_history,
            strategy_name=strategy_name,
        )
        db_recorder.register_artifact(
            artifact_type="metrics_jsonl",
            file_path=metrics_path,
            description="Per-round aggregated and client metrics",
        )
        db_recorder.register_artifact(
            artifact_type="global_metrics_jsonl",
            file_path=global_metrics_path,
            description="Global evaluation metrics by round",
        )
        db_recorder.register_artifact(
            artifact_type="checkpoint",
            file_path=latest_checkpoint_path,
            description="Latest global checkpoint",
        )
        db_recorder.register_artifact(
            artifact_type="checkpoint",
            file_path=best_checkpoint_path,
            description="Best checkpoint by f1_macro",
        )
        db_recorder.finalize(status="completed")
    except KeyboardInterrupt:
        if db_recorder is not None:
            db_recorder.finalize(
                status="interrupted",
                error_message="Interrupted by user",
            )
        raise
    except Exception as exc:
        if db_recorder is not None:
            db_recorder.finalize(
                status="failed",
                error_message=str(exc),
            )
        raise


if __name__ == "__main__":
    main()
