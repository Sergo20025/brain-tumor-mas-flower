from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain_tumor_fl.utils import print_agent_log


@dataclass
class AggregationAgent:
    decentralized_mode: bool = True

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
            weight = float(max(num_examples, 1) * max(trust_score, 0.1))
        return weight

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
