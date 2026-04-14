from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any

import numpy as np

from brain_tumor_fl.utils import print_agent_log


@dataclass
class MonitoringAgent:
    save_path: str | None = None
    client_trust: dict[str, float] = field(default_factory=dict)
    update_norm_history: list[float] = field(default_factory=list)

    def score_client(self, client_id: str, metrics: dict[str, Any]) -> float:
        val_accuracy = float(metrics.get("val_accuracy", metrics.get("train_accuracy", 0.0)))
        val_f1 = float(metrics.get("val_f1", metrics.get("train_f1", 0.0)))
        train_time = float(metrics.get("train_time_sec", 1.0))
        update_norm = float(metrics.get("update_l2_norm", 0.0))

        baseline_norm = median(self.update_norm_history) if self.update_norm_history else update_norm
        anomaly_penalty = 1.0
        if baseline_norm > 0 and update_norm > 2.5 * baseline_norm:
            anomaly_penalty = 0.5

        time_penalty = 1.0 / (1.0 + np.log1p(max(train_time, 0.0)))
        quality_score = 0.5 * val_accuracy + 0.5 * val_f1
        trust = float(np.clip(0.2 + 1.4 * quality_score * time_penalty * anomaly_penalty, 0.1, 1.5))

        self.client_trust[client_id] = trust
        self.update_norm_history.append(update_norm)
        print_agent_log(
            "MonitoringAgent",
            (
                f"trust updated: trust={trust:.3f}, val_acc={val_accuracy:.4f}, "
                f"val_f1={val_f1:.4f}, train_time={train_time:.2f}s, "
                f"update_l2={update_norm:.4f}, anomaly_penalty={anomaly_penalty:.2f}"
            ),
            client_id=client_id,
        )
        return trust

    def summarize_round(self, round_number: int, metrics: dict[str, Any]) -> dict[str, Any]:
        print_agent_log(
            "MonitoringAgent",
            (
                f"round summary: accuracy={float(metrics.get('accuracy', 0.0)):.4f}, "
                f"f1={float(metrics.get('f1_macro', 0.0)):.4f}, "
                f"loss={float(metrics.get('loss', 0.0)):.4f}"
            ),
            round_number=round_number,
        )
        return {"round": round_number, **metrics}
