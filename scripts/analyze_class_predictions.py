from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from brain_tumor_fl.data import discover_dataset_layout, prepare_global_test_loader
from brain_tumor_fl.model import build_model
from brain_tumor_fl.training import get_device


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint was not found: {path}")
    return torch.load(path, map_location="cpu")


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    experiment_dir = Path(args.experiment_dir)
    preferred = experiment_dir / "checkpoints" / "best_model.pt"
    fallback = experiment_dir / "checkpoints" / "latest_model.pt"
    return preferred if preferred.exists() else fallback


def _resolve_output_dir(args: argparse.Namespace, checkpoint_path: Path) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    checkpoints_dir = checkpoint_path.parent
    experiment_dir = checkpoints_dir.parent
    return experiment_dir / "plots" / "class_predictions"


def _collect_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()

    predictions: list[int] = []
    targets: list[int] = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            logits = model(images)
            batch_predictions = torch.argmax(logits, dim=1).detach().cpu().tolist()
            predictions.extend(int(item) for item in batch_predictions)
            targets.extend(int(item) for item in labels.detach().cpu().tolist())

    return np.asarray(targets, dtype=int), np.asarray(predictions, dtype=int)


def _save_confusion_matrix(
    matrix: np.ndarray,
    classes: list[str],
    output_path: Path,
    *,
    normalized: bool,
) -> None:
    plt.figure(figsize=(13, 10) if len(classes) > 20 else (8, 6))
    annotation = len(classes) <= 20
    fmt = ".2f" if normalized else "d"
    sns.heatmap(
        matrix,
        annot=annotation,
        fmt=fmt,
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
    )
    title = "Normalized confusion matrix" if normalized else "Confusion matrix"
    plt.title(title)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=45 if len(classes) <= 20 else 90, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def _save_per_class_metrics(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    metric_names = ["precision", "recall", "f1_score", "class_accuracy"]
    classes = [row["class_name"] for row in rows]
    x = np.arange(len(classes))
    width = 0.2

    plt.figure(figsize=(14, 6) if len(classes) > 20 else (9, 5))
    for idx, metric in enumerate(metric_names):
        values = [float(row[metric]) for row in rows]
        plt.bar(x + (idx - 1.5) * width, values, width, label=metric)

    plt.ylim(0.0, 1.0)
    plt.ylabel("Metric value")
    plt.title("Per-class prediction quality")
    plt.xticks(x, classes, rotation=45 if len(classes) <= 20 else 90, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_per_class_rows(
    report: dict[str, Any],
    confusion: np.ndarray,
    classes: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(classes):
        item = report.get(class_name, {})
        support = int(confusion[class_id].sum())
        correct = int(confusion[class_id, class_id])
        errors = int(support - correct)
        class_accuracy = float(correct / support) if support else 0.0
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "support": support,
                "correct": correct,
                "errors": errors,
                "class_accuracy": class_accuracy,
                "precision": float(item.get("precision", 0.0)),
                "recall": float(item.get("recall", 0.0)),
                "f1_score": float(item.get("f1-score", 0.0)),
            }
        )
    return rows


def _build_top_confusions(
    confusion: np.ndarray,
    classes: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for true_id, true_name in enumerate(classes):
        true_total = int(confusion[true_id].sum())
        for pred_id, pred_name in enumerate(classes):
            if true_id == pred_id:
                continue
            count = int(confusion[true_id, pred_id])
            if count <= 0:
                continue
            pairs.append(
                {
                    "true_class": true_name,
                    "predicted_class": pred_name,
                    "errors": count,
                    "share_of_true_class": float(count / true_total) if true_total else 0.0,
                }
            )
    return sorted(pairs, key=lambda item: int(item["errors"]), reverse=True)[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-class prediction diagnostics for a saved checkpoint."
    )
    parser.add_argument("--experiment-dir", default="", help="Experiment directory.")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path.")
    parser.add_argument("--dataset-root", default="", help="Dataset root, e.g. brain_tumor_mri or cifar100.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--top-k-confusions", type=int, default=20)
    args = parser.parse_args()

    checkpoint_path = _resolve_checkpoint(args)
    checkpoint = _load_checkpoint(checkpoint_path)
    dataset_root = args.dataset_root or str(checkpoint.get("dataset_root", "brain_tumor_mri"))

    output_dir = _resolve_output_dir(args, checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    layout = discover_dataset_layout(dataset_root, test_split=args.test_split, seed=args.seed)
    classes = list(checkpoint.get("classes") or layout.classes)
    test_loader, num_classes = prepare_global_test_loader(
        dataset_root=dataset_root,
        batch_size=args.batch_size,
        test_split=args.test_split,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_model(
        num_classes=num_classes,
        use_pretrained=bool(checkpoint.get("use_pretrained", False)),
        model_name=str(checkpoint.get("model_name", "efficientnet_b0")),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    device = get_device()
    targets, predictions = _collect_predictions(model, test_loader, device)

    labels = list(range(len(classes)))
    confusion = confusion_matrix(targets, predictions, labels=labels)
    normalized_confusion = confusion.astype(np.float64) / np.maximum(
        confusion.sum(axis=1, keepdims=True), 1
    )

    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    per_class_rows = _build_per_class_rows(report, confusion, classes)
    top_confusions = _build_top_confusions(confusion, classes, args.top_k_confusions)

    _save_confusion_matrix(
        confusion,
        classes,
        output_dir / "confusion_matrix.png",
        normalized=False,
    )
    _save_confusion_matrix(
        normalized_confusion,
        classes,
        output_dir / "confusion_matrix_normalized.png",
        normalized=True,
    )
    _save_per_class_metrics(per_class_rows, output_dir / "per_class_metrics.png")
    _write_csv(output_dir / "per_class_metrics.csv", per_class_rows)
    _write_csv(output_dir / "top_confusions.csv", top_confusions)

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_round": checkpoint.get("round", "unknown"),
        "dataset_root": dataset_root,
        "num_classes": len(classes),
        "num_test_samples": int(len(targets)),
        "accuracy": float(report.get("accuracy", 0.0)),
        "macro_precision": float(report.get("macro avg", {}).get("precision", 0.0)),
        "macro_recall": float(report.get("macro avg", {}).get("recall", 0.0)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "total_errors": int(np.sum(targets != predictions)),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
