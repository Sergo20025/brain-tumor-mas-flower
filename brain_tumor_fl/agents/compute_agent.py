from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flwr.client import NumPyClient

from brain_tumor_fl.agents.storage_agent import StorageAgent
from brain_tumor_fl.model import build_model
from brain_tumor_fl.training import (
    evaluate_model,
    get_device,
    get_parameters,
    set_parameters,
    train_model,
)
from brain_tumor_fl.utils import coerce_bool, print_agent_log


@dataclass
class ComputeAgent:
    partition_id: int
    config: dict[str, Any]

    def __post_init__(self) -> None:
        self.storage_agent = StorageAgent(self.config)
        bundle = self.storage_agent.load_partition(self.partition_id)
        self.train_loader = bundle["train_loader"]
        self.val_loader = bundle["val_loader"]
        self.test_loader = bundle["test_loader"]
        self.partition_summary = bundle["summary"]
        self.device = get_device()
        self.model = build_model(
            num_classes=int(bundle["num_classes"]),
            use_pretrained=coerce_bool(self.config["use-pretrained"]),
        )
        print_agent_log(
            "ComputeAgent",
            (
                f"client initialized on device={self.device}, "
                f"train={self.partition_summary['num_train']}, "
                f"val={self.partition_summary['num_val']}"
            ),
            partition_id=self.partition_id,
        )

    def fit(self, parameters: list[Any], fit_config: dict[str, Any]) -> tuple[list[Any], int, dict[str, Any]]:
        round_number = int(fit_config.get("server_round", 0))
        print_agent_log(
            "ComputeAgent",
            (
                f"start local training: epochs={int(fit_config['local_epochs'])}, "
                f"lr={float(fit_config['learning_rate']):.6f}, "
                f"weight_decay={float(fit_config['weight_decay']):.6f}"
            ),
            partition_id=self.partition_id,
            round_number=round_number,
        )
        set_parameters(self.model, parameters)
        report = train_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            local_epochs=int(fit_config["local_epochs"]),
            learning_rate=float(fit_config["learning_rate"]),
            weight_decay=float(fit_config["weight_decay"]),
            device=self.device,
            proximal_mu=float(fit_config.get("proximal_mu", 0.0)),
        )

        metrics = {
            "partition_id": self.partition_id,
            "num_examples": self.partition_summary["num_train"],
            "train_loss": report["train"]["loss"],
            "train_accuracy": report["train"]["accuracy"],
            "train_f1": report["train"]["f1_macro"],
            "val_loss": report["val"]["loss"],
            "val_accuracy": report["val"]["accuracy"],
            "val_f1": report["val"]["f1_macro"],
            "train_time_sec": report["train_time_sec"],
            "update_l2_norm": report["update_l2_norm"],
        }

        print_agent_log(
            "ComputeAgent",
            (
                f"finish local training: train_acc={metrics['train_accuracy']:.4f}, "
                f"val_acc={metrics['val_accuracy']:.4f}, val_f1={metrics['val_f1']:.4f}, "
                f"train_loss={metrics['train_loss']:.4f}, val_loss={metrics['val_loss']:.4f}, "
                f"time={metrics['train_time_sec']:.2f}s"
            ),
            partition_id=self.partition_id,
            round_number=round_number,
        )

        return get_parameters(self.model), int(self.partition_summary["num_train"]), metrics

    def evaluate(self, parameters: list[Any]) -> tuple[float, int, dict[str, Any]]:
        set_parameters(self.model, parameters)
        metrics = evaluate_model(self.model, self.test_loader, self.device)
        print_agent_log(
            "ComputeAgent",
            (
                f"local evaluation: accuracy={metrics['accuracy']:.4f}, "
                f"precision={metrics['precision_macro']:.4f}, "
                f"recall={metrics['recall_macro']:.4f}, f1={metrics['f1_macro']:.4f}, "
                f"loss={metrics['loss']:.4f}"
            ),
            partition_id=self.partition_id,
        )
        return float(metrics["loss"]), len(self.test_loader.dataset), {
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
        }


class BrainTumorClient(NumPyClient):
    def __init__(self, compute_agent: ComputeAgent) -> None:
        self.compute_agent = compute_agent

    def get_parameters(self, config: dict[str, Any]) -> list[Any]:
        return get_parameters(self.compute_agent.model)

    def fit(
        self, parameters: list[Any], config: dict[str, Any]
    ) -> tuple[list[Any], int, dict[str, Any]]:
        return self.compute_agent.fit(parameters, config)

    def evaluate(
        self, parameters: list[Any], config: dict[str, Any]
    ) -> tuple[float, int, dict[str, Any]]:
        return self.compute_agent.evaluate(parameters)
