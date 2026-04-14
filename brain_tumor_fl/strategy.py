from __future__ import annotations

import json
from logging import INFO
from pathlib import Path
from typing import Any

import torch
from flwr.common import NDArrays, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server.strategy import FedAvg

from brain_tumor_fl.agents.aggregation_agent import AggregationAgent
from brain_tumor_fl.agents.monitoring_agent import MonitoringAgent
from brain_tumor_fl.agents.storage_agent import StorageAgent
from brain_tumor_fl.data import discover_dataset_layout
from brain_tumor_fl.model import build_model
from brain_tumor_fl.training import evaluate_model, get_device, get_parameters, set_parameters
from brain_tumor_fl.utils import coerce_bool


def _weighted_average_metrics(metrics_with_examples: list[tuple[int, dict[str, Scalar]]]) -> dict[str, Scalar]:
    total_examples = sum(num_examples for num_examples, _ in metrics_with_examples)
    if total_examples == 0:
        return {}

    keys = set().union(*(metrics.keys() for _, metrics in metrics_with_examples))
    aggregated: dict[str, Scalar] = {}
    for key in keys:
        weighted_sum = 0.0
        for num_examples, metrics in metrics_with_examples:
            if key in metrics:
                weighted_sum += float(metrics[key]) * num_examples
        aggregated[key] = weighted_sum / total_examples
    return aggregated


def _coerce_to_ndarrays(parameters: NDArrays | Parameters) -> NDArrays:
    if isinstance(parameters, list):
        return parameters
    return parameters_to_ndarrays(parameters)


class TrustAwareFedAvg(FedAvg):
    def __init__(
        self,
        storage_agent: StorageAgent,
        monitoring_agent: MonitoringAgent,
        aggregation_agent: AggregationAgent,
        run_config: dict[str, Any],
    ) -> None:
        self.storage_agent = storage_agent
        self.monitoring_agent = monitoring_agent
        self.aggregation_agent = aggregation_agent
        self.run_config = run_config
        self.metrics_path = Path(str(run_config["save-metrics-path"]))
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.global_metrics_path = self.metrics_path.with_name("global_eval_metrics.jsonl")
        self.checkpoints_dir = self.metrics_path.parent / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.checkpoints_dir / "best_model.pt"
        self.latest_checkpoint_path = self.checkpoints_dir / "latest_model.pt"
        self.best_f1_macro = float("-inf")
        initial_parameters = get_initial_parameters(run_config)

        super().__init__(
            fraction_fit=float(run_config["fraction-fit"]),
            fraction_evaluate=float(run_config["fraction-evaluate"]),
            min_available_clients=int(run_config["min-available-clients"]),
            min_fit_clients=int(run_config["num-clients"]),
            min_evaluate_clients=int(run_config["num-clients"]),
            evaluate_fn=self._get_evaluate_fn(),
            fit_metrics_aggregation_fn=_weighted_average_metrics,
            evaluate_metrics_aggregation_fn=_weighted_average_metrics,
            initial_parameters=initial_parameters,
        )

    def configure_fit(self, server_round, parameters, client_manager):
        from flwr.common import FitIns

        fit_cfg = []
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        base_lr = float(self.run_config["learning-rate"]) * (0.97 ** max(server_round - 1, 0))
        base_epochs = int(self.run_config["local-epochs"])

        for client in clients:
            trust = float(self.monitoring_agent.client_trust.get(client.cid, 1.0))
            local_epochs = max(1, base_epochs - 1) if trust < 0.45 else base_epochs
            config = {
                "server_round": server_round,
                "learning_rate": base_lr,
                "local_epochs": local_epochs,
                "weight_decay": float(self.run_config["weight-decay"]),
            }
            fit_cfg.append((client, FitIns(parameters, config)))
            log(
                INFO,
                "[ServerCoordinator | round=%s] assign client=%s, trust=%.3f, epochs=%s, lr=%.6f",
                server_round,
                client.cid,
                trust,
                local_epochs,
                base_lr,
            )
        return fit_cfg

    def aggregate_fit(self, server_round, results, failures):
        if not results or (failures and not self.accept_failures):
            return None, {}

        weighted_updates: list[tuple[NDArrays, float]] = []
        round_reports: list[dict[str, Any]] = []
        metrics_with_examples: list[tuple[int, dict[str, Scalar]]] = []
        client_weights: list[tuple[str, float]] = []

        for client_proxy, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            metrics = dict(fit_res.metrics)
            num_examples = int(fit_res.num_examples)
            trust = self.monitoring_agent.score_client(client_proxy.cid, metrics)
            weight = self.aggregation_agent.compute_weight(num_examples=num_examples, trust_score=trust)
            client_weights.append((client_proxy.cid, weight))

            weighted_updates.append((ndarrays, weight))
            metrics_with_examples.append((num_examples, fit_res.metrics))
            round_reports.append(
                {
                    "client_id": client_proxy.cid,
                    "num_examples": num_examples,
                    "trust_score": trust,
                    **metrics,
                }
            )
            log(
                INFO,
                (
                    "[ServerCoordinator | round=%s] received client=%s, "
                    "num_examples=%s, train_acc=%.4f, val_acc=%.4f, trust=%.3f, weight=%.2f"
                ),
                server_round,
                client_proxy.cid,
                num_examples,
                float(metrics.get("train_accuracy", 0.0)),
                float(metrics.get("val_accuracy", 0.0)),
                trust,
                weight,
            )

        aggregated_ndarrays = self.aggregation_agent.aggregate(weighted_updates)
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        aggregated_metrics = _weighted_average_metrics(metrics_with_examples)
        aggregated_metrics["participating_clients"] = len(results)
        total_weight = sum(weight for _, weight in client_weights)
        top_clients = sorted(client_weights, key=lambda item: item[1], reverse=True)[:3]
        self.aggregation_agent.log_round_summary(
            server_round=server_round,
            num_clients=len(results),
            total_weight=total_weight,
            top_clients=top_clients,
        )

        self._write_round_report(
            server_round=server_round,
            aggregated_metrics=aggregated_metrics,
            client_reports=round_reports,
        )
        return aggregated_parameters, aggregated_metrics

    def _write_round_report(
        self,
        server_round: int,
        aggregated_metrics: dict[str, Scalar],
        client_reports: list[dict[str, Any]],
    ) -> None:
        payload = {
            "round": server_round,
            "aggregated_metrics": dict(aggregated_metrics),
            "client_reports": client_reports,
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _write_global_eval_report(
        self,
        server_round: int,
        loss: float,
        metrics: dict[str, float],
    ) -> None:
        payload = {
            "round": server_round,
            "loss": loss,
            **metrics,
        }
        with self.global_metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _get_evaluate_fn(self):
        test_loader, num_classes = self.storage_agent.load_global_test_loader()
        layout = discover_dataset_layout(
            dataset_root=str(self.run_config["dataset-root"]),
            test_split=float(self.run_config["test-split"]),
            seed=int(self.run_config["random-seed"]),
        )
        model = build_model(
            num_classes=num_classes,
            use_pretrained=coerce_bool(self.run_config["use-pretrained"]),
        )
        device = get_device()

        def evaluate_fn(server_round: int, parameters: NDArrays | Parameters, config):
            del config
            ndarrays = _coerce_to_ndarrays(parameters)
            set_parameters(model, ndarrays)
            metrics = evaluate_model(model, test_loader, device)
            scalar_metrics = {
                "accuracy": float(metrics["accuracy"]),
                "precision_macro": float(metrics["precision_macro"]),
                "recall_macro": float(metrics["recall_macro"]),
                "f1_macro": float(metrics["f1_macro"]),
            }
            self.monitoring_agent.summarize_round(
                round_number=server_round,
                metrics=metrics,
            )
            self._write_global_eval_report(
                server_round=server_round,
                loss=float(metrics["loss"]),
                metrics=scalar_metrics,
            )
            self._save_checkpoint(
                model=model,
                round_number=server_round,
                metrics=scalar_metrics,
                loss=float(metrics["loss"]),
                classes=layout.classes,
            )
            return float(metrics["loss"]), scalar_metrics

        return evaluate_fn

    def _save_checkpoint(
        self,
        model,
        round_number: int,
        metrics: dict[str, float],
        loss: float,
        classes: list[str],
    ) -> None:
        payload = {
            "round": round_number,
            "loss": loss,
            "metrics": metrics,
            "classes": classes,
            "model_name": "efficientnet_b0",
            "use_pretrained": coerce_bool(self.run_config["use-pretrained"]),
            "dataset_root": str(self.run_config["dataset-root"]),
            "state_dict": model.state_dict(),
        }
        torch.save(payload, self.latest_checkpoint_path)

        current_f1 = float(metrics.get("f1_macro", 0.0))
        if current_f1 >= self.best_f1_macro:
            self.best_f1_macro = current_f1
            torch.save(payload, self.best_checkpoint_path)
            log(
                INFO,
                (
                    "[ServerCoordinator | round=%s] saved best checkpoint to %s "
                    "(f1_macro=%.4f, accuracy=%.4f)"
                ),
                round_number,
                self.best_checkpoint_path,
                current_f1,
                float(metrics.get("accuracy", 0.0)),
            )


def get_initial_parameters(run_config: dict[str, Any]) -> Parameters:
    storage_agent = StorageAgent(run_config)
    _, num_classes = storage_agent.load_global_test_loader()
    model = build_model(
        num_classes=num_classes,
        use_pretrained=coerce_bool(run_config["use-pretrained"]),
    )
    return ndarrays_to_parameters(get_parameters(model))
