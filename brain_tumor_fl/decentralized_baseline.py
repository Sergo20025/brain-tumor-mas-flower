from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from logging import INFO
from pathlib import Path
from typing import Any

import torch
from flwr.common.logger import log

from brain_tumor_fl.data import (
    discover_dataset_layout,
    prepare_client_partition,
    prepare_global_test_loader,
)
from brain_tumor_fl.db import ExperimentDatabaseRecorder
from brain_tumor_fl.model import build_model
from brain_tumor_fl.training import evaluate_model, get_device, get_parameters, set_parameters, train_model
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


def _build_topology(
    num_nodes: int,
    topology_mode: str,
    extra_offset: int,
) -> dict[int, list[int]]:
    topology: dict[int, list[int]] = {}
    for node_id in range(num_nodes):
        neighbors: list[int]
        if topology_mode == "ring":
            neighbors = [
                (node_id - 1) % num_nodes,
                (node_id + 1) % num_nodes,
            ]
        elif topology_mode == "augmented_ring":
            neighbors = [
                (node_id - 1) % num_nodes,
                (node_id + 1) % num_nodes,
                (node_id + extra_offset) % num_nodes,
            ]
        elif topology_mode == "full_graph":
            neighbors = [idx for idx in range(num_nodes) if idx != node_id]
        else:
            raise ValueError(f"Unsupported topology mode: {topology_mode}")

        deduped: list[int] = []
        for neighbor_id in neighbors:
            if neighbor_id != node_id and neighbor_id not in deduped:
                deduped.append(neighbor_id)
        topology[node_id] = deduped
    return topology


def _weighted_average_parameters(weighted_updates: list[tuple[NDArrays, float]]) -> NDArrays:
    if not weighted_updates:
        raise ValueError("No updates provided for aggregation.")

    total_weight = sum(max(weight, 0.0) for _, weight in weighted_updates)
    if total_weight <= 0.0:
        total_weight = float(len(weighted_updates))
        weighted_updates = [(params, 1.0) for params, _ in weighted_updates]

    aggregated: NDArrays = []
    layer_count = len(weighted_updates[0][0])
    for layer_id in range(layer_count):
        layer_sum = None
        for parameters, weight in weighted_updates:
            contribution = parameters[layer_id] * (float(weight) / total_weight)
            layer_sum = contribution if layer_sum is None else layer_sum + contribution
        aggregated.append(layer_sum)
    return aggregated


@dataclass
class BaselineNode:
    node_id: int
    partition_id: int
    model: torch.nn.Module
    train_loader: Any
    val_loader: Any
    current_parameters: NDArrays
    num_examples: int

    def train_local_model(
        self,
        round_number: int,
        learning_rate: float,
        local_epochs: int,
        weight_decay: float,
        device: torch.device,
    ) -> dict[str, Any]:
        set_parameters(self.model, _clone_ndarrays(self.current_parameters))
        report = train_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
        )
        self.current_parameters = _clone_ndarrays(get_parameters(self.model))
        return {
            "client_id": f"N{self.partition_id}",
            "partition_id": self.partition_id,
            "num_examples": self.num_examples,
            "train_loss": float(report["train"]["loss"]),
            "train_accuracy": float(report["train"]["accuracy"]),
            "train_f1": float(report["train"]["f1_macro"]),
            "val_loss": float(report["val"]["loss"]),
            "val_accuracy": float(report["val"]["accuracy"]),
            "val_f1": float(report["val"]["f1_macro"]),
            "train_time_sec": float(report["train_time_sec"]),
            "update_l2_norm": float(report["update_l2_norm"]),
        }


class DecentralizedBaselineCoordinator:
    def __init__(self, run_config: dict[str, Any]) -> None:
        self.run_config = run_config
        self.metrics_path = Path(str(run_config["save-metrics-path"]))
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.global_metrics_path = self.metrics_path.with_name("global_eval_metrics.jsonl")
        self.checkpoints_dir = self.metrics_path.parent / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.checkpoints_dir / "best_model.pt"
        self.latest_checkpoint_path = self.checkpoints_dir / "latest_model.pt"
        self.best_f1_macro = float("-inf")
        self.train_history: list[dict[str, float]] = []
        self.eval_history: list[dict[str, float]] = []
        self.db_recorder = ExperimentDatabaseRecorder(
            run_config={**run_config, "model-name": "efficientnet_b0"},
            mode="decentralized_baseline",
        )

        self.global_test_loader, self.num_classes = prepare_global_test_loader(
            dataset_root=str(run_config["dataset-root"]),
            batch_size=int(run_config["batch-size"]),
            test_split=float(run_config["test-split"]),
            num_workers=int(run_config["num-workers"]),
            seed=int(run_config["random-seed"]),
        )
        layout = discover_dataset_layout(
            dataset_root=str(run_config["dataset-root"]),
            test_split=float(run_config["test-split"]),
            seed=int(run_config["random-seed"]),
        )
        self.classes = layout.classes
        self.topology_mode = str(run_config["topology-mode"])
        self.topology_extra_offset = int(run_config["topology-extra-offset"])
        self.topology = _build_topology(
            num_nodes=int(run_config["num-clients"]),
            topology_mode=self.topology_mode,
            extra_offset=self.topology_extra_offset,
        )
        self.device = get_device()
        self.nodes = self._build_nodes()
        self.eval_model = build_model(
            num_classes=self.num_classes,
            use_pretrained=coerce_bool(self.run_config["use-pretrained"]),
        )
        self.eval_model.to(self.device)

    def _build_nodes(self) -> list[BaselineNode]:
        bootstrap_model = build_model(
            num_classes=self.num_classes,
            use_pretrained=coerce_bool(self.run_config["use-pretrained"]),
        )
        initial_parameters = get_parameters(bootstrap_model)
        nodes: list[BaselineNode] = []

        for partition_id in range(int(self.run_config["num-clients"])):
            partition = prepare_client_partition(
                dataset_root=str(self.run_config["dataset-root"]),
                partition_id=partition_id,
                num_clients=int(self.run_config["num-clients"]),
                partition_mode=str(self.run_config["partition-mode"]),
                dirichlet_alpha=float(self.run_config["dirichlet-alpha"]),
                soft_mix_ratio=float(self.run_config.get("soft-mix-ratio", 0.15)),
                soft_min_extra_classes=int(self.run_config.get("soft-min-extra-classes", 5)),
                batch_size=int(self.run_config["batch-size"]),
                val_split=float(self.run_config["val-split"]),
                test_split=float(self.run_config["test-split"]),
                num_workers=int(self.run_config["num-workers"]),
                seed=int(self.run_config["random-seed"]),
            )
            model = build_model(
                num_classes=self.num_classes,
                use_pretrained=coerce_bool(self.run_config["use-pretrained"]),
            )
            model.to(self.device)
            set_parameters(model, initial_parameters)
            nodes.append(
                BaselineNode(
                    node_id=partition_id,
                    partition_id=partition_id,
                    model=model,
                    train_loader=partition.train_loader,
                    val_loader=partition.val_loader,
                    current_parameters=_clone_ndarrays(initial_parameters),
                    num_examples=int(partition.summary["num_train"]),
                )
            )

        log(
            INFO,
            "[DecentralizedBaseline] initialized %s nodes with topology=%s",
            len(nodes),
            self.topology_mode,
        )
        for node_id, neighbors in self.topology.items():
            neighbor_text = ", ".join(f"N{neighbor_id}" for neighbor_id in neighbors)
            log(INFO, "[BaselineTopology] node=N%s neighbors=%s", node_id, neighbor_text)
        return nodes

    def run(self) -> None:
        self._evaluate_and_record(round_number=0)
        for round_number in range(1, int(self.run_config["num-server-rounds"]) + 1):
            self._run_round(round_number)
        self._log_final_summary()

    def _run_round(self, round_number: int) -> None:
        learning_rate = float(self.run_config["learning-rate"]) * (0.97 ** max(round_number - 1, 0))
        local_epochs = int(self.run_config["local-epochs"])
        weight_decay = float(self.run_config["weight-decay"])

        client_reports: list[dict[str, Any]] = []
        metrics_with_examples: list[tuple[int, dict[str, float]]] = []

        for node in self.nodes:
            report = node.train_local_model(
                round_number=round_number,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
                weight_decay=weight_decay,
                device=self.device,
            )
            client_reports.append(report)
            metrics_with_examples.append(
                (
                    node.num_examples,
                    {
                        "train_loss": float(report["train_loss"]),
                        "train_accuracy": float(report["train_accuracy"]),
                        "train_f1": float(report["train_f1"]),
                        "val_loss": float(report["val_loss"]),
                        "val_accuracy": float(report["val_accuracy"]),
                        "val_f1": float(report["val_f1"]),
                        "train_time_sec": float(report["train_time_sec"]),
                        "update_l2_norm": float(report["update_l2_norm"]),
                    },
                )
            )

        local_updates = {node.node_id: _clone_ndarrays(node.current_parameters) for node in self.nodes}
        next_parameters: dict[int, NDArrays] = {}

        for node in self.nodes:
            neighborhood = [node.node_id, *self.topology[node.node_id]]
            weighted_updates = [
                (local_updates[source_id], float(max(self.nodes[source_id].num_examples, 1)))
                for source_id in neighborhood
            ]
            next_parameters[node.node_id] = _weighted_average_parameters(weighted_updates)

        for node in self.nodes:
            node.current_parameters = _clone_ndarrays(next_parameters[node.node_id])

        aggregated_train_metrics = _weighted_average_metrics(metrics_with_examples)
        aggregated_train_metrics["participating_clients"] = float(len(self.nodes))
        self.train_history.append(
            {
                "round": round_number,
                "train_loss": float(aggregated_train_metrics.get("train_loss", 0.0)),
                "train_accuracy": float(aggregated_train_metrics.get("train_accuracy", 0.0)),
                "val_loss": float(aggregated_train_metrics.get("val_loss", 0.0)),
                "val_accuracy": float(aggregated_train_metrics.get("val_accuracy", 0.0)),
                "val_f1": float(aggregated_train_metrics.get("val_f1", 0.0)),
            }
        )
        self._write_round_report(
            round_number=round_number,
            aggregated_metrics=aggregated_train_metrics,
            client_reports=client_reports,
        )
        self._log_round_summary(round_number)
        self._evaluate_and_record(round_number=round_number)

    def _log_round_summary(self, round_number: int) -> None:
        lines: list[str] = []
        for node_id in sorted(self.topology):
            neighbors = ", ".join(f"N{neighbor_id}" for neighbor_id in self.topology[node_id])
            lines.append(f"N{node_id}->[{neighbors}]")
        log(
            INFO,
            "[DecentralizedBaseline | round=%s] topology aggregation: %s",
            round_number,
            "; ".join(lines),
        )

    def _build_virtual_global_parameters(self) -> NDArrays:
        weighted_updates = [
            (_clone_ndarrays(node.current_parameters), float(max(node.num_examples, 1)))
            for node in self.nodes
        ]
        return _weighted_average_parameters(weighted_updates)

    def _evaluate_and_record(self, round_number: int) -> None:
        global_parameters = self._build_virtual_global_parameters()
        set_parameters(self.eval_model, global_parameters)
        metrics = evaluate_model(self.eval_model, self.global_test_loader, self.device)
        scalar_metrics = {
            "accuracy": float(metrics["accuracy"]),
            "precision_macro": float(metrics["precision_macro"]),
            "recall_macro": float(metrics["recall_macro"]),
            "f1_macro": float(metrics["f1_macro"]),
        }
        payload = {
            "round": round_number,
            "loss": float(metrics["loss"]),
            **scalar_metrics,
        }
        with self.global_metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.db_recorder.record_global_eval(
            round_number=round_number,
            loss=float(metrics["loss"]),
            metrics=payload,
        )
        self.eval_history.append(
            {
                "round": round_number,
                "loss": float(metrics["loss"]),
                "accuracy": float(metrics["accuracy"]),
                "f1_macro": float(metrics["f1_macro"]),
            }
        )

        self._save_checkpoint(
            round_number=round_number,
            global_parameters=global_parameters,
            metrics=scalar_metrics,
            loss=float(metrics["loss"]),
        )
        log(
            INFO,
            (
                "[DecentralizedBaseline | round=%s] global evaluation: "
                "accuracy=%.4f, f1=%.4f, loss=%.4f"
            ),
            round_number,
            scalar_metrics["accuracy"],
            scalar_metrics["f1_macro"],
            float(metrics["loss"]),
        )

    def _save_checkpoint(
        self,
        round_number: int,
        global_parameters: NDArrays,
        metrics: dict[str, float],
        loss: float,
    ) -> None:
        set_parameters(self.eval_model, global_parameters)
        payload = {
            "round": round_number,
            "loss": loss,
            "metrics": metrics,
            "classes": self.classes,
            "model_name": "efficientnet_b0",
            "use_pretrained": coerce_bool(self.run_config["use-pretrained"]),
            "dataset_root": str(self.run_config["dataset-root"]),
            "topology_mode": self.topology_mode,
            "mode": "decentralized_baseline",
            "state_dict": self.eval_model.state_dict(),
        }
        torch.save(payload, self.latest_checkpoint_path)

        current_f1 = float(metrics.get("f1_macro", 0.0))
        if current_f1 >= self.best_f1_macro:
            self.best_f1_macro = current_f1
            torch.save(payload, self.best_checkpoint_path)
            log(
                INFO,
                (
                    "[DecentralizedBaseline | round=%s] saved best checkpoint to %s "
                    "(f1_macro=%.4f, accuracy=%.4f)"
                ),
                round_number,
                self.best_checkpoint_path,
                current_f1,
                float(metrics.get("accuracy", 0.0)),
            )

    def _log_final_summary(self) -> None:
        log(INFO, "[SUMMARY | baseline_%s]", self.topology_mode)
        log(INFO, "History (metrics, decentralized baseline fit):")
        for item in self.train_history:
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
        for item in self.eval_history:
            log(
                INFO,
                "round %s: loss=%.4f, accuracy=%.4f, f1_macro=%.4f",
                int(item["round"]),
                float(item["loss"]),
                float(item["accuracy"]),
                float(item["f1_macro"]),
            )

    def _write_round_report(
        self,
        round_number: int,
        aggregated_metrics: dict[str, float],
        client_reports: list[dict[str, Any]],
    ) -> None:
        payload = {
            "round": round_number,
            "mode": "decentralized_baseline",
            "topology_mode": self.topology_mode,
            "aggregated_metrics": dict(aggregated_metrics),
            "client_reports": client_reports,
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.db_recorder.record_round(
            round_number=round_number,
            aggregated_metrics=dict(aggregated_metrics),
            client_reports=client_reports,
        )

    def finalize_experiment(self, status: str, error_message: str | None = None) -> None:
        self.db_recorder.register_artifact(
            artifact_type="metrics_jsonl",
            file_path=self.metrics_path,
            description="Per-round aggregated and client metrics",
        )
        self.db_recorder.register_artifact(
            artifact_type="global_metrics_jsonl",
            file_path=self.global_metrics_path,
            description="Global evaluation metrics by round",
        )
        self.db_recorder.register_artifact(
            artifact_type="checkpoint",
            file_path=self.latest_checkpoint_path,
            description="Latest global checkpoint",
        )
        self.db_recorder.register_artifact(
            artifact_type="checkpoint",
            file_path=self.best_checkpoint_path,
            description="Best checkpoint by f1_macro",
        )
        self.db_recorder.finalize(status=status, error_message=error_message)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decentralized baseline without agent logic."
    )
    parser.add_argument("--dataset-root", default="brain_tumor_mri")
    parser.add_argument("--num-server-rounds", type=int, default=30)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--partition-mode", default="dirichlet")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--soft-mix-ratio", type=float, default=0.15)
    parser.add_argument("--soft-min-extra-classes", type=int, default=5)
    parser.add_argument("--topology-mode", default="augmented_ring")
    parser.add_argument("--topology-extra-offset", type=int, default=2)
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


def _namespace_to_run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset-root": args.dataset_root,
        "num-server-rounds": args.num_server_rounds,
        "num-clients": args.num_clients,
        "partition-mode": args.partition_mode,
        "dirichlet-alpha": args.dirichlet_alpha,
        "soft-mix-ratio": args.soft_mix_ratio,
        "soft-min-extra-classes": args.soft_min_extra_classes,
        "topology-mode": args.topology_mode,
        "topology-extra-offset": args.topology_extra_offset,
        "use-pretrained": coerce_bool(args.use_pretrained),
        "local-epochs": args.local_epochs,
        "batch-size": args.batch_size,
        "learning-rate": args.learning_rate,
        "weight-decay": args.weight_decay,
        "val-split": args.val_split,
        "test-split": args.test_split,
        "num-workers": args.num_workers,
        "random-seed": args.random_seed,
        "save-metrics-path": args.save_metrics_path,
    }


def main() -> None:
    args = _parse_args()
    run_config = _namespace_to_run_config(args)
    coordinator = DecentralizedBaselineCoordinator(run_config)
    try:
        coordinator.run()
    except KeyboardInterrupt:
        coordinator.finalize_experiment(
            status="interrupted",
            error_message="Interrupted by user",
        )
        raise
    except Exception as exc:
        coordinator.finalize_experiment(
            status="failed",
            error_message=str(exc),
        )
        raise
    else:
        coordinator.finalize_experiment(status="completed")


if __name__ == "__main__":
    main()
