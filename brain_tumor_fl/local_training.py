from __future__ import annotations

import argparse
import json
from logging import INFO
from pathlib import Path
from typing import Any

import torch
from flwr.common.logger import log

from brain_tumor_fl.data import prepare_centralized_data
from brain_tumor_fl.model import build_model
from brain_tumor_fl.training import evaluate_model, get_device, train_model
from brain_tumor_fl.utils import coerce_bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Centralized local training baseline.")
    parser.add_argument("--dataset-root", default="brain_tumor_mri")
    parser.add_argument("--num-server-rounds", type=int, default=30)
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


def _save_checkpoint(
    model,
    checkpoint_path: Path,
    round_number: int,
    loss: float,
    metrics: dict[str, float],
    classes: list[str],
    dataset_root: str,
    use_pretrained: bool,
) -> None:
    payload = {
        "round": round_number,
        "loss": loss,
        "metrics": metrics,
        "classes": classes,
        "model_name": "efficientnet_b0",
        "use_pretrained": use_pretrained,
        "dataset_root": dataset_root,
        "mode": "local",
        "state_dict": model.state_dict(),
    }
    torch.save(payload, checkpoint_path)


def _log_final_summary(
    train_history: list[dict[str, float]],
    eval_history: list[dict[str, float]],
) -> None:
    log(INFO, "[SUMMARY]")
    log(INFO, "History (metrics, local train):")
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
    args = _parse_args()
    metrics_path = Path(args.save_metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    global_metrics_path = metrics_path.with_name("global_eval_metrics.jsonl")
    checkpoints_dir = metrics_path.parent / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoints_dir / "best_model.pt"
    latest_checkpoint_path = checkpoints_dir / "latest_model.pt"

    data_bundle = prepare_centralized_data(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
        seed=args.random_seed,
    )
    device = get_device()
    use_pretrained = coerce_bool(args.use_pretrained)
    model = build_model(num_classes=data_bundle.num_classes, use_pretrained=use_pretrained)
    model.to(device)
    train_history: list[dict[str, float]] = []
    eval_history: list[dict[str, float]] = []

    best_f1_macro = float("-inf")
    log(
        INFO,
        (
            "[LocalTraining] initialized centralized baseline: "
            "train=%s, val=%s, test=%s, device=%s"
        ),
        data_bundle.summary["num_train"],
        data_bundle.summary["num_val"],
        data_bundle.summary["num_test"],
        device,
    )

    initial_metrics = evaluate_model(model, data_bundle.test_loader, device)
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
        report = train_model(
            model=model,
            train_loader=data_bundle.train_loader,
            val_loader=data_bundle.val_loader,
            local_epochs=args.local_epochs,
            learning_rate=learning_rate,
            weight_decay=args.weight_decay,
            device=device,
        )
        aggregated_metrics = _weighted_average_metrics(
            [
                (
                    data_bundle.summary["num_train"],
                    {
                        "train_loss": float(report["train"]["loss"]),
                        "train_accuracy": float(report["train"]["accuracy"]),
                        "train_f1": float(report["train"]["f1_macro"]),
                        "val_loss": float(report["val"]["loss"]),
                        "val_accuracy": float(report["val"]["accuracy"]),
                        "val_f1": float(report["val"]["f1_macro"]),
                        "train_time_sec": float(report["train_time_sec"]),
                        "update_l2_norm": float(report["update_l2_norm"]),
                    },
                )
            ]
        )
        aggregated_metrics["participating_clients"] = 1.0
        train_history.append(
            {
                "round": round_number,
                "train_loss": float(report["train"]["loss"]),
                "train_accuracy": float(report["train"]["accuracy"]),
                "val_loss": float(report["val"]["loss"]),
                "val_accuracy": float(report["val"]["accuracy"]),
                "val_f1": float(report["val"]["f1_macro"]),
            }
        )

        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "round": round_number,
                        "mode": "local",
                        "aggregated_metrics": aggregated_metrics,
                        "client_reports": [
                            {
                                "client_id": "local_centralized",
                                "partition_id": 0,
                                "num_examples": data_bundle.summary["num_train"],
                                "train_loss": float(report["train"]["loss"]),
                                "train_accuracy": float(report["train"]["accuracy"]),
                                "train_f1": float(report["train"]["f1_macro"]),
                                "val_loss": float(report["val"]["loss"]),
                                "val_accuracy": float(report["val"]["accuracy"]),
                                "val_f1": float(report["val"]["f1_macro"]),
                                "train_time_sec": float(report["train_time_sec"]),
                                "update_l2_norm": float(report["update_l2_norm"]),
                            }
                        ],
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

        eval_metrics = evaluate_model(model, data_bundle.test_loader, device)
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
        eval_history.append(
            {
                "round": round_number,
                "loss": float(eval_metrics["loss"]),
                "accuracy": float(eval_metrics["accuracy"]),
                "f1_macro": float(eval_metrics["f1_macro"]),
            }
        )

        _save_checkpoint(
            model=model,
            checkpoint_path=latest_checkpoint_path,
            round_number=round_number,
            loss=float(eval_metrics["loss"]),
            metrics=scalar_metrics,
            classes=data_bundle.classes,
            dataset_root=args.dataset_root,
            use_pretrained=use_pretrained,
        )
        if scalar_metrics["f1_macro"] >= best_f1_macro:
            best_f1_macro = scalar_metrics["f1_macro"]
            _save_checkpoint(
                model=model,
                checkpoint_path=best_checkpoint_path,
                round_number=round_number,
                loss=float(eval_metrics["loss"]),
                metrics=scalar_metrics,
                classes=data_bundle.classes,
                dataset_root=args.dataset_root,
                use_pretrained=use_pretrained,
            )

        log(
            INFO,
            (
                "[LocalTraining | round=%s] accuracy=%.4f, f1=%.4f, loss=%.4f, "
                "train_loss=%.4f"
            ),
            round_number,
            scalar_metrics["accuracy"],
            scalar_metrics["f1_macro"],
            float(eval_metrics["loss"]),
            float(report["train"]["loss"]),
        )

    _log_final_summary(train_history=train_history, eval_history=eval_history)


if __name__ == "__main__":
    main()
