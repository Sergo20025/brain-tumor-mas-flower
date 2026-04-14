from __future__ import annotations

import logging

from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

LOGGER = logging.getLogger(__name__)


def build_model(num_classes: int, use_pretrained: bool = True) -> nn.Module:
    """Create EfficientNet-B0 and adapt the classifier to the task."""
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
