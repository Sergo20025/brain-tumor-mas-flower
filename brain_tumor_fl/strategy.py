from __future__ import annotations

import json
from logging import INFO
from pathlib import Path
from typing import Any

import torch
from flwr.common import (
    NDArrays,
    EvaluateIns,
    FitIns,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
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


def _clone_ndarrays(parameters: NDArrays) -> NDArrays:
    return [layer.copy() for layer in parameters]


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
        self.initial_ndarrays = _coerce_to_ndarrays(initial_parameters)
        self.client_partition_map: dict[str, int] = {}
        self.partition_client_map: dict[int, str] = {}
        self.client_num_examples: dict[str, int] = {}
        self.client_local_parameters: dict[str, NDArrays] = {}
        self.client_dispatch_parameters: dict[str, NDArrays] = {}
        self.ring_neighbors: dict[str, list[str]] = {}
        self.ring_ready = False

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
        fit_cfg = []
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        base_lr = float(self.run_config["learning-rate"]) * (0.97 ** max(server_round - 1, 0))
        base_epochs = int(self.run_config["local-epochs"])

        for client in clients:
            fit_parameters = parameters
            neighbor_info = ""
            if self.aggregation_agent.decentralized_mode and self.ring_ready:
                local_parameters = self.client_dispatch_parameters.get(client.cid)
                if local_parameters is not None:
                    fit_parameters = ndarrays_to_parameters(_clone_ndarrays(local_parameters))
                neighbor_info = self._format_neighbor_info(client.cid)

            config = {
                "server_round": server_round,
                "learning_rate": base_lr,
                "local_epochs": base_epochs,
                "weight_decay": float(self.run_config["weight-decay"]),
            }
            fit_cfg.append((client, FitIns(fit_parameters, config)))
            log(
                INFO,
                "[ServerCoordinator | round=%s] assign client=%s, epochs=%s, lr=%.6f%s",
                server_round,
                client.cid,
                base_epochs,
                base_lr,
                neighbor_info,
            )
        return fit_cfg

    def configure_evaluate(self, server_round, parameters, client_manager):
        if self.fraction_evaluate == 0.0:
            return []

        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        evaluate_cfg = []
        for client in clients:
            eval_parameters = parameters
            if self.aggregation_agent.decentralized_mode and self.ring_ready:
                local_parameters = self.client_dispatch_parameters.get(client.cid)
                if local_parameters is not None:
                    eval_parameters = ndarrays_to_parameters(_clone_ndarrays(local_parameters))
            evaluate_cfg.append((client, EvaluateIns(eval_parameters, {})))
        return evaluate_cfg

    def aggregate_fit(self, server_round, results, failures):
        if not results or (failures and not self.accept_failures):
            return None, {}

        round_reports: list[dict[str, Any]] = []
        metrics_with_examples: list[tuple[int, dict[str, Scalar]]] = []
        global_weighted_updates: list[tuple[NDArrays, float]] = []

        for client_proxy, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            metrics = dict(fit_res.metrics)
            num_examples = int(fit_res.num_examples)
            partition_id = int(metrics.get("partition_id", len(self.client_partition_map)))
            trust = self.monitoring_agent.score_client(client_proxy.cid, metrics)

            self.client_partition_map[client_proxy.cid] = partition_id
            self.partition_client_map[partition_id] = client_proxy.cid
            self.client_num_examples[client_proxy.cid] = num_examples
            self.client_local_parameters[client_proxy.cid] = _clone_ndarrays(ndarrays)

            metrics_with_examples.append((num_examples, fit_res.metrics))
            round_reports.append(
                {
                    "client_id": client_proxy.cid,
                    "partition_id": partition_id,
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
                float(num_examples),
            )

        if self.aggregation_agent.decentralized_mode:
            if not self.ring_ready:
                self._initialize_augmented_ring_topology()
            self.client_dispatch_parameters = self._compute_ring_dispatch_parameters()
            for client_id, local_params in self.client_dispatch_parameters.items():
                global_weighted_updates.append(
                    (local_params, float(max(self.client_num_examples.get(client_id, 1), 1)))
                )
            aggregated_ndarrays = self.aggregation_agent.aggregate(global_weighted_updates)
            self._log_ring_round_summary(server_round)
        else:
            for client_id, local_params in self.client_local_parameters.items():
                global_weighted_updates.append(
                    (local_params, float(max(self.client_num_examples.get(client_id, 1), 1)))
                )
            aggregated_ndarrays = self.aggregation_agent.aggregate(global_weighted_updates)

        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        aggregated_metrics = _weighted_average_metrics(metrics_with_examples)
        aggregated_metrics["participating_clients"] = len(results)
        if not self.aggregation_agent.decentralized_mode:
            client_weights = [
                (client_id, float(max(self.client_num_examples.get(client_id, 1), 1)))
                for client_id in self.client_local_parameters
            ]
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

    def _initialize_augmented_ring_topology(self) -> None:
        ordered_pairs = sorted(self.partition_client_map.items(), key=lambda item: item[0])
        ordered_clients = [client_id for _, client_id in ordered_pairs]
        num_clients = len(ordered_clients)
        if num_clients == 0:
            return

        default_offset = max(2, num_clients // 2)
        extra_offset = int(self.run_config.get("topology-extra-offset", default_offset))
        self.ring_neighbors = {}

        for idx, client_id in enumerate(ordered_clients):
            candidate_indices = [
                (idx - 1) % num_clients,
                (idx + 1) % num_clients,
                (idx + extra_offset) % num_clients,
            ]
            neighbor_ids: list[str] = []
            for neighbor_idx in candidate_indices:
                neighbor_id = ordered_clients[neighbor_idx]
                if neighbor_id != client_id and neighbor_id not in neighbor_ids:
                    neighbor_ids.append(neighbor_id)

            if len(neighbor_ids) < 3:
                for shift in range(2, num_clients):
                    neighbor_id = ordered_clients[(idx + shift) % num_clients]
                    if neighbor_id != client_id and neighbor_id not in neighbor_ids:
                        neighbor_ids.append(neighbor_id)
                    if len(neighbor_ids) == 3:
                        break

            self.ring_neighbors[client_id] = neighbor_ids[:3]

        self.ring_ready = True

        log(
            INFO,
            (
                "[TopologyCoordinator] initialized augmented ring topology with %s nodes "
                "(extra_offset=%s)"
            ),
            num_clients,
            extra_offset,
        )
        for partition_id, client_id in ordered_pairs:
            neighbors = self.ring_neighbors.get(client_id, [])
            neighbor_partitions = [
                str(self.client_partition_map.get(neighbor_id, "?")) for neighbor_id in neighbors
            ]
            log(
                INFO,
                "[TopologyCoordinator] node=%s neighbors=%s",
                partition_id,
                ", ".join(neighbor_partitions),
            )

    def _compute_ring_dispatch_parameters(self) -> dict[str, NDArrays]:
        dispatch_parameters: dict[str, NDArrays] = {}
        for client_id, local_parameters in self.client_local_parameters.items():
            neighborhood_updates: list[tuple[NDArrays, float]] = [(_clone_ndarrays(local_parameters), 1.0)]
            for neighbor_id in self.ring_neighbors.get(client_id, []):
                neighbor_parameters = self.client_local_parameters.get(neighbor_id)
                if neighbor_parameters is not None:
                    neighborhood_updates.append((_clone_ndarrays(neighbor_parameters), 1.0))
            dispatch_parameters[client_id] = self.aggregation_agent.aggregate(neighborhood_updates)
        return dispatch_parameters

    def _format_neighbor_info(self, client_id: str) -> str:
        neighbors = self.ring_neighbors.get(client_id, [])
        if not neighbors:
            return ""
        labels = [str(self.client_partition_map.get(neighbor_id, "?")) for neighbor_id in neighbors]
        return f", neighbors=[{', '.join(labels)}]"

    def _log_ring_round_summary(self, server_round: int) -> None:
        topologies = []
        for client_id, neighbors in sorted(
            self.ring_neighbors.items(),
            key=lambda item: self.client_partition_map.get(item[0], -1),
        ):
            partition_id = self.client_partition_map.get(client_id, -1)
            neighbor_labels = [str(self.client_partition_map.get(neighbor_id, "?")) for neighbor_id in neighbors]
            topologies.append(f"{partition_id}->[{', '.join(neighbor_labels)}]")
        log(
            INFO,
            "[AggregationAgent | round=%s] augmented ring neighbor aggregation: %s",
            server_round,
            "; ".join(topologies),
        )

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
