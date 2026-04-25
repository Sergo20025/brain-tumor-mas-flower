from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brain_tumor_fl.utils import print_agent_log


@dataclass
class AggregationAgent:
    decentralized_mode: bool = True
    node_id: int | None = None
    neighbors: list[int] = field(default_factory=list)
    inbound_updates: list[tuple[int, list[np.ndarray], float]] = field(default_factory=list)

    def aggregate(
        self, weighted_updates: list[tuple[list[np.ndarray], float]]
    ) -> list[np.ndarray]:
        if not weighted_updates:
            raise ValueError("No updates were provided for aggregation.")

        total_weight = sum(weight for _, weight in weighted_updates)
        if total_weight <= 0:
            total_weight = float(len(weighted_updates))
            weighted_updates = [(params, 1.0) for params, _ in weighted_updates]

        aggregated = [
            np.zeros_like(layer, dtype=np.float64) for layer in weighted_updates[0][0]
        ]
        for parameters, weight in weighted_updates:
            for idx, layer in enumerate(parameters):
                aggregated[idx] += layer.astype(np.float64) * weight

        return [
            (layer / total_weight).astype(weighted_updates[0][0][idx].dtype)
            for idx, layer in enumerate(aggregated)
        ]

    def compute_weight(self, num_examples: int, trust_score: float) -> float:
        if not self.decentralized_mode:
            weight = float(num_examples)
        else:
            blended_trust = 0.6 + 0.4 * max(trust_score, 0.1)
            weight = float(max(num_examples, 1) * blended_trust)
        return weight

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
            (
                source_id,
                [layer.copy() for layer in parameters],
                float(weight),
            )
        )

    def aggregate_received_updates(self) -> list[np.ndarray]:
        if not self.inbound_updates:
            raise ValueError("No inbound updates were received for local aggregation.")

        weighted_updates = [
            (parameters, weight)
            for _, parameters, weight in self.inbound_updates
        ]
        aggregated = self.aggregate(weighted_updates)
        source_text = ", ".join(f"N{source_id}:{weight:.3f}" for source_id, _, weight in self.inbound_updates)
        print_agent_log(
            "AggregationAgent",
            f"local aggregation complete from [{source_text}]",
            client_id=f"N{self.node_id}" if self.node_id is not None else None,
        )
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
        if self_weight < 0.0:
            self_weight = 0.0
        weights[node_id] = self_weight
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

    def log_round_summary(
        self,
        server_round: int,
        num_clients: int,
        total_weight: float,
        top_clients: list[tuple[str, float]],
    ) -> None:
        top_text = ", ".join(f"{client_id}:{weight:.2f}" for client_id, weight in top_clients)
        print_agent_log(
            "AggregationAgent",
            (
                f"aggregated {num_clients} client updates, "
                f"total_weight={total_weight:.2f}, top_weights=[{top_text}]"
            ),
            round_number=server_round,
        )
