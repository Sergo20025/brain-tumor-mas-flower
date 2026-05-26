from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


NDArrays = list[Any]


def _clone_ndarrays(parameters: NDArrays) -> NDArrays:
    return [layer.copy() for layer in parameters]


def _weighted_average_metrics(
    metrics_with_examples: list[tuple[int, dict[str, float]]],
) -> dict[str, float]:
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
        if topology_mode == "ring":
            neighbors = [(node_id - 1) % num_nodes, (node_id + 1) % num_nodes]
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
    return {
        node_id: ordered_profiles[node_id]
        for node_id in range(num_nodes)
    }


@dataclass
class AggregationAgent:
    decentralized_mode: bool = True
    node_id: int | None = None
    neighbors: list[int] = field(default_factory=list)
    inbound_updates: list[tuple[int, list[np.ndarray], float]] = field(default_factory=list)

    def aggregate(
        self,
        weighted_updates: list[tuple[list[np.ndarray], float]],
    ) -> list[np.ndarray]:
        if not weighted_updates:
            raise ValueError("No updates were provided for aggregation.")

        total_weight = sum(weight for _, weight in weighted_updates)
        if total_weight <= 0:
            total_weight = float(len(weighted_updates))
            weighted_updates = [(params, 1.0) for params, _ in weighted_updates]

        aggregated = [
            np.zeros_like(layer, dtype=np.float64)
            for layer in weighted_updates[0][0]
        ]
        for parameters, weight in weighted_updates:
            for idx, layer in enumerate(parameters):
                aggregated[idx] += layer.astype(np.float64) * weight

        return [
            (layer / total_weight).astype(weighted_updates[0][0][idx].dtype)
            for idx, layer in enumerate(aggregated)
        ]

    def start_round(self, neighbors: list[int]) -> None:
        self.neighbors = list(neighbors)
        self.inbound_updates.clear()

    def receive_update(
        self,
        source_id: int,
        parameters: list[np.ndarray],
        weight: float,
    ) -> None:
        self.inbound_updates.append(
            (source_id, [layer.copy() for layer in parameters], float(weight))
        )

    def aggregate_received_updates(self) -> list[np.ndarray]:
        if not self.inbound_updates:
            raise ValueError("No inbound updates were received for local aggregation.")
        weighted_updates = [
            (parameters, weight)
            for _, parameters, weight in self.inbound_updates
        ]
        aggregated = self.aggregate(weighted_updates)
        self.inbound_updates.clear()
        return aggregated

    @staticmethod
    def metropolis_weights(
        node_id: int,
        topology: dict[int, list[int]],
    ) -> dict[int, float]:
        degree_i = len(topology[node_id])
        weights: dict[int, float] = {}
        for neighbor_id in topology[node_id]:
            degree_j = len(topology[neighbor_id])
            weights[neighbor_id] = 1.0 / (1.0 + max(degree_i, degree_j))

        self_weight = 1.0 - sum(weights.values())
        weights[node_id] = max(self_weight, 0.0)
        return weights

    @classmethod
    def trust_weighted_mixing(
        cls,
        node_id: int,
        topology: dict[int, list[int]],
        trust_scores: dict[int, float],
    ) -> dict[int, float]:
        base_weights = cls.metropolis_weights(node_id=node_id, topology=topology)
        adjusted = {
            source_id: base_weight * max(float(trust_scores.get(source_id, 1.0)), 0.1)
            for source_id, base_weight in base_weights.items()
        }
        total = sum(adjusted.values())
        if total <= 0.0:
            uniform_weight = 1.0 / len(adjusted)
            return {source_id: uniform_weight for source_id in adjusted}
        return {source_id: weight / total for source_id, weight in adjusted.items()}


@dataclass
class MonitoringAgent:
    client_trust: dict[str, float] = field(default_factory=dict)
    update_norm_history: list[float] = field(default_factory=list)

    def score_client(self, client_id: str, metrics: dict[str, Any]) -> float:
        val_accuracy = float(metrics.get("val_accuracy", metrics.get("train_accuracy", 0.0)))
        val_f1 = float(metrics.get("val_f1", metrics.get("train_f1", 0.0)))
        train_time = float(metrics.get("train_time_sec", 1.0))
        update_norm = float(metrics.get("update_l2_norm", 0.0))

        baseline_norm = float(np.median(self.update_norm_history)) if self.update_norm_history else update_norm
        anomaly_penalty = 1.0
        if baseline_norm > 0 and update_norm > 2.5 * baseline_norm:
            anomaly_penalty = 0.5

        time_penalty = 1.0 / (1.0 + 0.15 * np.log1p(max(train_time, 0.0)))
        quality_score = 0.5 * val_accuracy + 0.5 * val_f1
        trust = float(
            np.clip(0.55 + 0.55 * quality_score * time_penalty * anomaly_penalty, 0.4, 1.25)
        )

        self.client_trust[client_id] = trust
        self.update_norm_history.append(update_norm)
        return trust


@dataclass
class DecentralizedNode:
    node_id: int
    partition_id: int
    compute_agent: Any
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

    def make_skip_report(self, round_number: int) -> dict[str, Any]:
        return {
            "client_id": f"N{self.partition_id}",
            "partition_id": self.partition_id,
            "num_examples": self.num_examples,
            "trust_score": self.trust_score,
            "resource_profile": self.resource_profile.name,
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
        self.skip_counts = {node_id: 0 for node_id in range(num_nodes)}
        self.rng = np.random.default_rng(seed)
        self.dropout_weights = dropout_weights or {
            node_id: 1.0 for node_id in range(num_nodes)
        }

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
        return [node_id for node_id in range(self.num_nodes) if node_id not in skipped]


class DecentralizedCoordinator:
    def __init__(self, run_config: dict[str, Any], nodes: list[DecentralizedNode]) -> None:
        self.run_config = run_config
        self.nodes = nodes
        self.topology = _build_topology(
            num_nodes=int(run_config["num-clients"]),
            topology_mode=str(run_config["topology-mode"]),
            extra_offset=int(run_config["topology-extra-offset"]),
        )
        self.async_mode = bool(run_config.get("async-mode", False))
        self.heterogeneous_nodes = bool(run_config.get("heterogeneous-nodes", False))
        self.resource_profiles = _build_resource_profiles(
            num_nodes=int(run_config["num-clients"]),
            base_batch_size=int(run_config["batch-size"]),
            heterogeneous_nodes=self.heterogeneous_nodes,
        )
        self.async_scheduler = AsyncParticipationScheduler(
            num_nodes=int(run_config["num-clients"]),
            dropout_rate=float(run_config.get("async-dropout-rate", 0.0)),
            max_dropouts_per_round=int(run_config.get("max-async-dropouts", 0)),
            seed=int(run_config["random-seed"]),
            dropout_weights={
                node_id: profile.dropout_weight
                for node_id, profile in self.resource_profiles.items()
            },
        )

    def run_round(self, round_number: int) -> dict[str, Any]:
        learning_rate = float(self.run_config["learning-rate"]) * (0.97 ** max(round_number - 1, 0))
        local_epochs = int(self.run_config["local-epochs"])
        weight_decay = float(self.run_config["weight-decay"])

        if self.async_mode:
            return self._run_async_round(round_number, learning_rate, local_epochs, weight_decay)
        return self._run_sync_round(round_number, learning_rate, local_epochs, weight_decay)

    def _run_sync_round(
        self,
        round_number: int,
        learning_rate: float,
        local_epochs: int,
        weight_decay: float,
    ) -> dict[str, Any]:
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
            metrics_with_examples.append((node.num_examples, self._extract_train_metrics(report)))
            trust_scores[node.node_id] = node.trust_score

        for node in self.nodes:
            node.begin_aggregation_round(self.topology[node.node_id])
        for node in self.nodes:
            node.collect_neighbor_updates(self.nodes, self.topology, trust_scores)
        for node in self.nodes:
            node.finalize_aggregation()

        aggregated_metrics = _weighted_average_metrics(metrics_with_examples)
        aggregated_metrics["participating_clients"] = float(len(self.nodes))
        return {
            "round": round_number,
            "async_mode": False,
            "aggregated_metrics": aggregated_metrics,
            "client_reports": client_reports,
        }

    def _run_async_round(
        self,
        round_number: int,
        learning_rate: float,
        local_epochs: int,
        weight_decay: float,
    ) -> dict[str, Any]:
        active_node_ids = self.async_scheduler.select_active_nodes(round_number)
        skipped_node_ids = [
            node_id for node_id in range(len(self.nodes))
            if node_id not in active_node_ids
        ]
        active_node_ids = self._order_async_nodes(active_node_ids, round_number)

        client_reports: list[dict[str, Any]] = []
        metrics_with_examples: list[tuple[int, dict[str, float]]] = []
        trust_scores = {node.node_id: node.trust_score for node in self.nodes}

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
            metrics_with_examples.append((node.num_examples, self._extract_train_metrics(report)))
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

        aggregated_metrics = _weighted_average_metrics(metrics_with_examples)
        aggregated_metrics["participating_clients"] = float(len(active_node_ids))
        aggregated_metrics["skipped_clients"] = float(len(skipped_node_ids))
        return {
            "round": round_number,
            "async_mode": True,
            "aggregated_metrics": aggregated_metrics,
            "client_reports": client_reports,
        }

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

    @staticmethod
    def _extract_train_metrics(report: dict[str, Any]) -> dict[str, float]:
        return {
            "train_loss": float(report["train_loss"]),
            "train_accuracy": float(report["train_accuracy"]),
            "train_f1": float(report["train_f1"]),
            "val_loss": float(report["val_loss"]),
            "val_accuracy": float(report["val_accuracy"]),
            "val_f1": float(report["val_f1"]),
            "train_time_sec": float(report["train_time_sec"]),
            "update_l2_norm": float(report["update_l2_norm"]),
        }
