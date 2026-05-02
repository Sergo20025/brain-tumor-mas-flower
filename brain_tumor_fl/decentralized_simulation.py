from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from logging import INFO
from pathlib import Path
from typing import Any

import numpy as np
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


@dataclass(frozen=True)
class NodeResourceProfile:
    name: str
    batch_size: int
    dropout_weight: float
    virtual_delay_sec: float


def _build_resource_profiles(
    num_nodes: int,
    base_batch_size: int,
    heterogeneous_nodes: bool,
) -> dict[int, NodeResourceProfile]:
    if not heterogeneous_nodes:
        return {
            node_id: NodeResourceProfile(
                name="homogeneous",
                batch_size=base_batch_size,
                dropout_weight=1.0,
                virtual_delay_sec=0.0,
            )
            for node_id in range(num_nodes)
        }

    fast_count = max(1, int(round(num_nodes * 0.3)))
    slow_count = max(1, int(round(num_nodes * 0.3)))
    if fast_count + slow_count >= num_nodes:
        slow_count = max(1, num_nodes - fast_count - 1)
    medium_count = max(num_nodes - fast_count - slow_count, 1)

    medium_batch = max(4, int(round(base_batch_size * 0.75)))
    slow_batch = max(4, int(round(base_batch_size * 0.5)))

    ordered_profiles = (
        [NodeResourceProfile("fast", base_batch_size, 0.6, 0.0)] * fast_count
        + [NodeResourceProfile("medium", medium_batch, 1.0, 0.75)] * medium_count
        + [NodeResourceProfile("slow", slow_batch, 1.6, 1.5)] * slow_count
    )
    ordered_profiles = ordered_profiles[:num_nodes]
    return {node_id: ordered_profiles[node_id] for node_id in range(num_nodes)}


@dataclass
class DecentralizedNode:
    node_id: int
    partition_id: int
    compute_agent: ComputeAgent
    monitoring_agent: MonitoringAgent
    aggregation_agent: AggregationAgent
    current_parameters: NDArrays
    num_examples: int
    resource_profile: NodeResourceProfile
    local_update_parameters: NDArrays | None = None
    trust_score: float = 1.0
    last_completed_round: int = 0
    last_train_time_sec: float = 0.0

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
        self.last_train_time_sec = float(metrics["train_time_sec"])
        self.last_completed_round = round_number
        return {
            "client_id": f"N{self.partition_id}",
            "partition_id": self.partition_id,
            "num_examples": self.num_examples,
            "trust_score": self.trust_score,
            "resource_profile": self.resource_profile.name,
            "resource_batch_size": self.resource_profile.batch_size,
            "virtual_delay_sec": self.resource_profile.virtual_delay_sec,
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
        parameter_bank: dict[int, NDArrays] | None = None,
    ) -> None:
        mixing_weights = self.aggregation_agent.trust_weighted_mixing(
            node_id=self.node_id,
            topology=topology,
            trust_scores=trust_scores,
        )
        for source_id, weight in mixing_weights.items():
            if parameter_bank is not None:
                source_parameters = parameter_bank[source_id]
            else:
                source_parameters = nodes[source_id].export_local_update()
            self.aggregation_agent.receive_update(
                source_id=source_id,
                parameters=source_parameters,
                weight=float(weight),
            )

    def finalize_aggregation(self) -> None:
        self.current_parameters = self.aggregation_agent.aggregate_received_updates()

    def evaluate_local_model(self) -> tuple[float, int, dict[str, float]]:
        return self.compute_agent.evaluate(_clone_ndarrays(self.current_parameters))

    def make_skip_report(self, round_number: int) -> dict[str, Any]:
        return {
            "client_id": f"N{self.partition_id}",
            "partition_id": self.partition_id,
            "num_examples": self.num_examples,
            "trust_score": self.trust_score,
            "resource_profile": self.resource_profile.name,
            "resource_batch_size": self.resource_profile.batch_size,
            "virtual_delay_sec": self.resource_profile.virtual_delay_sec,
            "train_loss": 0.0,
            "train_accuracy": 0.0,
            "train_f1": 0.0,
            "val_loss": 0.0,
            "val_accuracy": 0.0,
            "val_f1": 0.0,
            "train_time_sec": 0.0,
            "update_l2_norm": 0.0,
            "skipped": True,
            "completed_round": self.last_completed_round,
            "server_round": round_number,
        }


class AsyncParticipationScheduler:
    def __init__(
        self,
        num_nodes: int,
        dropout_rate: float,
        max_dropouts_per_round: int,
        seed: int,
        dropout_weights: dict[int, float] | None = None,
    ) -> None:
        self.num_nodes = num_nodes
        self.dropout_rate = min(max(dropout_rate, 0.0), 0.95)
        self.max_dropouts_per_round = max(0, min(max_dropouts_per_round, max(num_nodes - 1, 0)))
        self.seed = seed
        self.skip_counts = {node_id: 0 for node_id in range(num_nodes)}
        self.rng = np.random.default_rng(seed)
        self.dropout_weights = dropout_weights or {node_id: 1.0 for node_id in range(num_nodes)}

    def select_active_nodes(self, round_number: int) -> list[int]:
        if self.dropout_rate <= 0.0 or self.max_dropouts_per_round <= 0:
            return list(range(self.num_nodes))

        planned_dropouts = int(round(self.num_nodes * self.dropout_rate))
        dropout_count = max(1, min(planned_dropouts, self.max_dropouts_per_round))
        ordered = sorted(
            range(self.num_nodes),
            key=lambda node_id: (
                self.skip_counts[node_id] / max(self.dropout_weights.get(node_id, 1.0), 0.05),
                float(self.rng.random()),
                int((node_id + round_number) % self.num_nodes),
            ),
        )
        skipped = ordered[:dropout_count]
        for node_id in skipped:
            self.skip_counts[node_id] += 1
        active_nodes = [node_id for node_id in range(self.num_nodes) if node_id not in skipped]
        return active_nodes


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
        self.resume_checkpoint_path = str(run_config.get("resume-checkpoint", "")).strip()
        self.resume_payload: dict[str, Any] | None = None
        self.start_round = 1
        self.db_recorder = ExperimentDatabaseRecorder(
            run_config=run_config,
            mode="decentralized",
        )

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
        self.async_mode = bool(run_config.get("async-mode", False))
        self.async_dropout_rate = float(run_config.get("async-dropout-rate", 0.0))
        self.max_async_dropouts = int(run_config.get("max-async-dropouts", 0))
        self.heterogeneous_nodes = bool(run_config.get("heterogeneous-nodes", False))
        self.topology = _build_topology(
            num_nodes=int(run_config["num-clients"]),
            topology_mode=self.topology_mode,
            extra_offset=self.topology_extra_offset,
        )
        self.resource_profiles = _build_resource_profiles(
            num_nodes=int(run_config["num-clients"]),
            base_batch_size=int(run_config["batch-size"]),
            heterogeneous_nodes=self.heterogeneous_nodes,
        )
        self.nodes = self._build_nodes()
        self.async_scheduler = AsyncParticipationScheduler(
            num_nodes=int(run_config["num-clients"]),
            dropout_rate=self.async_dropout_rate,
            max_dropouts_per_round=self.max_async_dropouts,
            seed=int(run_config["random-seed"]),
            dropout_weights={
                node_id: profile.dropout_weight
                for node_id, profile in self.resource_profiles.items()
            },
        )
        self.eval_model = build_model(
            num_classes=self.num_classes,
            use_pretrained=coerce_bool(self.run_config["use-pretrained"]),
            model_name=str(self.run_config.get("model-name", "efficientnet_b0")),
        )
        self.eval_device = get_device()
        self.eval_model.to(self.eval_device)
        self._load_resume_checkpoint_if_needed()

    def _load_resume_checkpoint_if_needed(self) -> None:
        if not self.resume_checkpoint_path:
            return
        checkpoint_path = Path(self.resume_checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint '{checkpoint_path}' was not found."
            )
        payload = torch.load(checkpoint_path, map_location="cpu")
        self.resume_payload = payload
        self.eval_model.load_state_dict(payload["state_dict"], strict=True)
        self.start_round = int(payload.get("round", 0)) + 1
        self.best_f1_macro = float(payload.get("metrics", {}).get("f1_macro", float("-inf")))
        log(
            INFO,
            (
                "[DecentralizedCoordinator] resuming from checkpoint=%s "
                "starting_round=%s"
            ),
            checkpoint_path,
            self.start_round,
        )

    def _build_nodes(self) -> list[DecentralizedNode]:
        bootstrap_model = build_model(
            num_classes=self.num_classes,
            use_pretrained=coerce_bool(self.run_config["use-pretrained"]),
            model_name=str(self.run_config.get("model-name", "efficientnet_b0")),
        )
        if self.resume_payload is not None:
            bootstrap_model.load_state_dict(self.resume_payload["state_dict"], strict=True)
        initial_parameters = get_parameters(bootstrap_model)
        nodes: list[DecentralizedNode] = []
        for partition_id in range(int(self.run_config["num-clients"])):
            resource_profile = self.resource_profiles[partition_id]
            node_config = dict(self.run_config)
            node_config["batch-size"] = resource_profile.batch_size
            compute_agent = ComputeAgent(partition_id=partition_id, config=node_config)
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
                resource_profile=resource_profile,
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
            profile = self.resource_profiles[node_id]
            log(
                INFO,
                (
                    "[TopologyCoordinator] node=N%s neighbors=%s "
                    "profile=%s batch=%s dropout_weight=%.2f virtual_delay=%.2fs"
                ),
                node_id,
                neighbor_text,
                profile.name,
                profile.batch_size,
                profile.dropout_weight,
                profile.virtual_delay_sec,
            )
        return nodes

    def run(self) -> None:
        if self.start_round <= 1:
            self._evaluate_and_record(round_number=0)
        for round_number in range(self.start_round, int(self.run_config["num-server-rounds"]) + 1):
            self._run_round(round_number)
        self._log_final_summary()

    def _run_round(self, round_number: int) -> None:
        learning_rate = float(self.run_config["learning-rate"]) * (0.97 ** max(round_number - 1, 0))
        local_epochs = int(self.run_config["local-epochs"])
        weight_decay = float(self.run_config["weight-decay"])

        if self.async_mode:
            self._run_async_round(
                round_number=round_number,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
                weight_decay=weight_decay,
            )
            return

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

    def _run_async_round(
        self,
        round_number: int,
        learning_rate: float,
        local_epochs: int,
        weight_decay: float,
    ) -> None:
        active_node_ids = self.async_scheduler.select_active_nodes(round_number)
        skipped_node_ids = [node_id for node_id in range(len(self.nodes)) if node_id not in active_node_ids]
        active_node_ids = self._order_async_nodes(active_node_ids, round_number)

        client_reports: list[dict[str, Any]] = []
        metrics_with_examples: list[tuple[int, dict[str, float]]] = []
        trust_scores: dict[int, float] = {node.node_id: node.trust_score for node in self.nodes}

        for node_id in active_node_ids:
            node = self.nodes[node_id]
            report = node.train_local_model(
                round_number=round_number,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
                weight_decay=weight_decay,
            )
            report["skipped"] = False
            report["completed_round"] = node.last_completed_round
            report["server_round"] = round_number
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

            node.begin_aggregation_round(self.topology[node.node_id])
            parameter_bank: dict[int, NDArrays] = {}
            for source_id in [node.node_id, *self.topology[node.node_id]]:
                source_node = self.nodes[source_id]
                if source_id == node.node_id:
                    parameter_bank[source_id] = source_node.export_local_update()
                else:
                    parameter_bank[source_id] = _clone_ndarrays(source_node.current_parameters)
            node.collect_neighbor_updates(
                nodes=self.nodes,
                topology=self.topology,
                trust_scores=trust_scores,
                parameter_bank=parameter_bank,
            )
            node.finalize_aggregation()

        for node_id in skipped_node_ids:
            client_reports.append(self.nodes[node_id].make_skip_report(round_number))

        aggregated_train_metrics = _weighted_average_metrics(metrics_with_examples)
        aggregated_train_metrics["participating_clients"] = float(len(active_node_ids))
        aggregated_train_metrics["skipped_clients"] = float(len(skipped_node_ids))
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
        self._log_async_round_summary(
            round_number=round_number,
            trust_scores=trust_scores,
            active_node_ids=active_node_ids,
            skipped_node_ids=skipped_node_ids,
        )
        self._evaluate_and_record(round_number=round_number)

    def _order_async_nodes(self, active_node_ids: list[int], round_number: int) -> list[int]:
        if not self.heterogeneous_nodes:
            return active_node_ids
        return sorted(
            active_node_ids,
            key=lambda node_id: (
                self.nodes[node_id].resource_profile.virtual_delay_sec,
                self.nodes[node_id].last_train_time_sec,
                math.fmod(node_id + round_number, max(len(self.nodes), 1)),
            ),
        )

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

    def _log_async_round_summary(
        self,
        round_number: int,
        trust_scores: dict[int, float],
        active_node_ids: list[int],
        skipped_node_ids: list[int],
    ) -> None:
        lines: list[str] = []
        for node_id in sorted(active_node_ids):
            neighbors = ", ".join(f"N{neighbor_id}" for neighbor_id in self.topology[node_id])
            lines.append(
                f"N{node_id}->[{neighbors}] trust={trust_scores.get(node_id, 1.0):.3f}"
            )
        skipped_text = ", ".join(f"N{node_id}" for node_id in skipped_node_ids) if skipped_node_ids else "none"
        log(
            INFO,
            "[AggregationAgent | round=%s | async] active=%s, skipped=%s, topology aggregation: %s",
            round_number,
            len(active_node_ids),
            skipped_text,
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
            "model_name": str(self.run_config.get("model-name", "efficientnet_b0")),
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
            "async_mode": self.async_mode,
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
        description="Fully decentralized federated learning simulation for Brain Tumor MRI."
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
    parser.add_argument("--model-name", default="efficientnet_b0")
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
    parser.add_argument("--async-mode", default="false")
    parser.add_argument("--async-dropout-rate", type=float, default=0.0)
    parser.add_argument("--max-async-dropouts", type=int, default=0)
    parser.add_argument("--heterogeneous-nodes", default="false")
    parser.add_argument("--resume-checkpoint", default="")
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
        "model-name": args.model_name,
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
        "async-mode": coerce_bool(args.async_mode),
        "async-dropout-rate": args.async_dropout_rate,
        "max-async-dropouts": args.max_async_dropouts,
        "heterogeneous-nodes": coerce_bool(args.heterogeneous_nodes),
        "resume-checkpoint": args.resume_checkpoint,
    }


def main() -> None:
    args = _parse_args()
    run_config = _namespace_to_run_config(args)
    coordinator = DecentralizedCoordinator(run_config)
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
