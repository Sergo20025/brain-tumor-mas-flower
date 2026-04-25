from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: Iterable[np.ndarray]) -> None:
    state_dict = model.state_dict()
    params_dict = zip(state_dict.keys(), parameters, strict=True)
    new_state_dict = {
        key: torch.tensor(value, dtype=state_dict[key].dtype)
        for key, value in params_dict
    }
    model.load_state_dict(new_state_dict, strict=True)


def _collect_epoch_predictions(
    logits: torch.Tensor, labels: torch.Tensor, preds: list[int], targets: list[int]
) -> None:
    preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
    targets.extend(labels.detach().cpu().tolist())


def _compute_metrics(losses: list[float], preds: list[int], targets: list[int]) -> dict[str, float]:
    if not targets:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
        }

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(targets, preds)),
        "precision_macro": float(
            precision_score(targets, preds, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(targets, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(targets, preds, average="macro", zero_division=0)),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    local_epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    proximal_mu: float = 0.0,
) -> dict[str, Any]:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    initial_parameters = [param.detach().cpu().clone() for param in model.parameters()]
    proximal_references = None
    if proximal_mu > 0.0:
        proximal_references = [
            param.detach().clone().to(device) for param in model.parameters()
        ]

    start_time = time.perf_counter()
    model.train()

    train_losses: list[float] = []
    train_preds: list[int] = []
    train_targets: list[int] = []

    for _ in range(local_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if proximal_references is not None:
                proximal_term = torch.zeros(1, device=device)
                for current, reference in zip(
                    model.parameters(), proximal_references, strict=True
                ):
                    proximal_term += torch.sum((current - reference) ** 2)
                loss = loss + 0.5 * float(proximal_mu) * proximal_term
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))
            _collect_epoch_predictions(logits, labels, train_preds, train_targets)

    train_metrics = _compute_metrics(train_losses, train_preds, train_targets)
    val_metrics = evaluate_model(model, val_loader, device)
    elapsed = time.perf_counter() - start_time

    squared_sum = 0.0
    for current, initial in zip(model.parameters(), initial_parameters, strict=True):
        delta = current.detach().cpu() - initial
        squared_sum += float(torch.sum(delta * delta).item())

    return {
        "train": train_metrics,
        "val": val_metrics,
        "train_time_sec": float(elapsed),
        "update_l2_norm": float(np.sqrt(squared_sum)),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    losses: list[float] = []
    preds: list[int] = []
    targets: list[int] = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        losses.append(float(loss.item()))
        _collect_epoch_predictions(logits, labels, preds, targets)

    return _compute_metrics(losses, preds, targets)
