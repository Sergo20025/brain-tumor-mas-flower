from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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


@dataclass
class DecentralizedNode:
    node_id: int
    partition_id: int
    compute_agent: ComputeAgent
    monitoring_agent: MonitoringAgent
    aggregation_agent: AggregationAgent
    current_parameters: NDArrays
    num_examples: int
    local_update_parameters: NDArrays | None = None
    trust_score: float = 1.0

    def train_local_model(
        self,
        round_number: int,
        learning_rate: float,
        local_epochs: int,
        weight_decay: float,
    ) -> dict[str, Any]:
        local_parameters, _, metrics = self.compute_agent.fit(
            _clone_ndarrays(self.current_parameters),
            {
                "server_round": round_number,
                "learning_rate": learning_rate,
                "local_epochs": local_epochs,
                "weight_decay": weight_decay,
            },
        )
        self.local_update_parameters = _clone_ndarrays(local_parameters)
        self.trust_score = self.monitoring_agent.score_client(
            client_id=f"N{self.partition_id}",
            metrics=metrics,
        )
        return {
            "client_id": f"N{self.partition_id}",
            "partition_id": self.partition_id,
            "num_examples": self.num_examples,
            "trust_score": self.trust_score,
            **metrics,
        }

    def begin_aggregation_round(self, neighbors: list[int]) -> None:
        self.aggregation_agent.start_round(neighbors)

    def export_local_update(self) -> NDArrays:
        if self.local_update_parameters is None:
            raise ValueError(f"Node N{self.node_id} has no local update prepared.")
        return _clone_ndarrays(self.local_update_parameters)

    def collect_neighbor_updates(
        self,
        nodes: list["DecentralizedNode"],
        topology: dict[int, list[int]],
        trust_scores: dict[int, float],
    ) -> None:
        mixing_weights = self.aggregation_agent.trust_weighted_mixing(
            node_id=self.node_id,
            topology=topology,
            trust_scores=trust_scores,
        )
        for source_id, weight in mixing_weights.items():
            self.aggregation_agent.receive_update(
                source_id=source_id,
                parameters=nodes[source_id].export_local_update(),
                weight=float(weight),
            )

    def finalize_aggregation(self) -> None:
        self.current_parameters = self.aggregation_agent.aggregate_received_updates()

    def evaluate_local_model(self) -> tuple[float, int, dict[str, float]]:
        return self.compute_agent.evaluate(_clone_ndarrays(self.current_parameters))


class DecentralizedCoordinator:
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

        self.storage_agent = StorageAgent(run_config)
        self.global_test_loader, self.num_classes = self.storage_agent.load_global_test_loader()
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
        self.nodes = self._build_nodes()
        self.eval_model = build_model(
            num_classes=self.num_classes,
            use_pretrained=coerce_bool(self.run_config["use-pretrained"]),
        )
        self.eval_device = get_device()
        self.eval_model.to(self.eval_device)

    def _build_nodes(self) -> list[DecentralizedNode]:
        bootstrap_model = build_model(
            num_classes=self.num_classes,
            use_pretrained=coerce_bool(self.run_config["use-pretrained"]),
        )
        initial_parameters = get_parameters(bootstrap_model)
        nodes: list[DecentralizedNode] = []
        for partition_id in range(int(self.run_config["num-clients"])):
            compute_agent = ComputeAgent(partition_id=partition_id, config=self.run_config)
            set_parameters(compute_agent.model, initial_parameters)
            node = DecentralizedNode(
                node_id=partition_id,
                partition_id=partition_id,
                compute_agent=compute_agent,
                monitoring_agent=MonitoringAgent(),
                aggregation_agent=AggregationAgent(
                    decentralized_mode=True,
                    node_id=partition_id,
                ),
                current_parameters=_clone_ndarrays(initial_parameters),
                num_examples=int(compute_agent.partition_summary["num_train"]),
            )
            nodes.append(node)

        log(
            INFO,
            "[DecentralizedCoordinator] initialized %s parameter nodes with topology=%s",
            len(nodes),
            self.topology_mode,
        )
        for node_id, neighbors in self.topology.items():
            neighbor_text = ", ".join(f"N{neighbor_id}" for neighbor_id in neighbors)
            log(INFO, "[TopologyCoordinator] node=N%s neighbors=%s", node_id, neighbor_text)
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
        trust_scores: dict[int, float] = {}

        for node in self.nodes:
            report = node.train_local_model(
                round_number=round_number,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
                weight_decay=weight_decay,
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
            trust_scores[node.node_id] = node.trust_score

        for node in self.nodes:
            node.begin_aggregation_round(self.topology[node.node_id])

        for node in self.nodes:
            node.collect_neighbor_updates(
                nodes=self.nodes,
                topology=self.topology,
                trust_scores=trust_scores,
            )

        for node in self.nodes:
            node.finalize_aggregation()

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
        self._log_round_summary(round_number=round_number, trust_scores=trust_scores)
        self._evaluate_and_record(round_number=round_number)

    def _log_round_summary(self, round_number: int, trust_scores: dict[int, float]) -> None:
        lines: list[str] = []
        for node_id in sorted(self.topology):
            neighbors = ", ".join(f"N{neighbor_id}" for neighbor_id in self.topology[node_id])
            lines.append(
                f"N{node_id}->[{neighbors}] trust={trust_scores.get(node_id, 1.0):.3f}"
            )
        log(
            INFO,
            "[AggregationAgent | round=%s] decentralized topology aggregation: %s",
            round_number,
            "; ".join(lines),
        )

    def _build_virtual_global_parameters(self) -> NDArrays:
        weighted_updates = [
            (_clone_ndarrays(node.current_parameters), float(max(node.num_examples, 1)))
            for node in self.nodes
        ]
        return AggregationAgent(decentralized_mode=False).aggregate(weighted_updates)

    def _evaluate_and_record(self, round_number: int) -> None:
        global_parameters = self._build_virtual_global_parameters()
        set_parameters(self.eval_model, global_parameters)
        metrics = evaluate_model(self.eval_model, self.global_test_loader, self.eval_device)
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
                "[DecentralizedCoordinator | round=%s] global evaluation: "
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
                    "[DecentralizedCoordinator | round=%s] saved best checkpoint to %s "
                    "(f1_macro=%.4f, accuracy=%.4f)"
                ),
                round_number,
                self.best_checkpoint_path,
                current_f1,
                float(metrics.get("accuracy", 0.0)),
            )

    def _log_final_summary(self) -> None:
        log(INFO, "[SUMMARY | %s]", self.topology_mode)
        log(INFO, "History (metrics, decentralized fit):")
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
            "topology_mode": self.topology_mode,
            "aggregated_metrics": dict(aggregated_metrics),
            "client_reports": client_reports,
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fully decentralized federated learning simulation for Brain Tumor MRI."
    )
    parser.add_argument("--dataset-root", default="brain_tumor_mri")
    parser.add_argument("--num-server-rounds", type=int, default=30)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--partition-mode", default="dirichlet")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
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
    parser.add_argument("--decentralized-mode", default="true")
    return parser.parse_args()


def _namespace_to_run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset-root": args.dataset_root,
        "num-server-rounds": args.num_server_rounds,
        "num-clients": args.num_clients,
        "partition-mode": args.partition_mode,
        "dirichlet-alpha": args.dirichlet_alpha,
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
        "decentralized-mode": coerce_bool(args.decentralized_mode),
    }


def main() -> None:
    args = _parse_args()
    run_config = _namespace_to_run_config(args)
    coordinator = DecentralizedCoordinator(run_config)
    coordinator.run()


if __name__ == "__main__":
    main()
