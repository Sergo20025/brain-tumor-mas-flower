from __future__ import annotations

import logging

from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    resnet50,
)

LOGGER = logging.getLogger(__name__)


def build_model(
    num_classes: int,
    use_pretrained: bool = True,
    model_name: str = "efficientnet_b0",
) -> nn.Module:
    """Create a supported backbone and adapt the classifier to the task."""
    normalized_name = model_name.strip().lower()

    if normalized_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if use_pretrained else None
        try:
            model = efficientnet_b0(weights=weights)
        except Exception as exc:  # pragma: no cover - depends on external cache/network
            LOGGER.warning(
                "Falling back to random EfficientNet-B0 initialization: %s", exc
            )
            model = efficientnet_b0(weights=None)

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if normalized_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if use_pretrained else None
        try:
            model = resnet50(weights=weights)
        except Exception as exc:  # pragma: no cover - depends on external cache/network
            LOGGER.warning(
                "Falling back to random ResNet-50 initialization: %s", exc
            )
            model = resnet50(weights=None)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model_name: {model_name}")
