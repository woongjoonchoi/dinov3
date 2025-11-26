"""Linear classification utilities using GAP features for the window baseline2 backbone."""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from dinov3.hub.utils import DINOV3_BASE_URL
from dinov3.hub.backbones import convert_path_or_url_to_url
from dionv3_window_base2 import DinoVisionTransformerWindow


class ClassifierWeights(Enum):
    IMAGENET1K = "IMAGENET1K"


class _GapLinearClassifierWrapper(nn.Module):
    def __init__(self, *, backbone: nn.Module, linear_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.linear_head = linear_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        patch_tokens = features["x_norm_patchtokens"]
        gap = patch_tokens.mean(dim=1)
        return self.linear_head(gap)


def _make_dinov3_window_base2_gap_classification_head(
    *,
    backbone_name: str = "dinov3_window_base2",
    embed_dim: int = 4096,
    num_classes: int = 1_000,
    pretrained: bool = False,
    classifier_weights: ClassifierWeights | str = ClassifierWeights.IMAGENET1K,
    check_hash: bool = False,
    **kwargs: Any,
) -> nn.Module:
    linear_head = nn.Linear(embed_dim, num_classes)
    if pretrained:
        if type(classifier_weights) is ClassifierWeights:
            assert classifier_weights == ClassifierWeights.IMAGENET1K, (
                f"Unsupported weights for linear classifier: {classifier_weights}"
            )
            weights_name = classifier_weights.value.lower()
            hash = kwargs["hash"] if "hash" in kwargs else None
            hash_suffix = f"-{hash}" if hash else ""
            model_filename = f"{backbone_name}_{weights_name}_linear_head{hash_suffix}.pth"
            url = os.path.join(DINOV3_BASE_URL, backbone_name, model_filename)
        else:
            url = convert_path_or_url_to_url(str(classifier_weights))
        state_dict = torch.hub.load_state_dict_from_url(
            url, map_location="cpu", check_hash=check_hash
        )
        linear_head.load_state_dict(state_dict, strict=True)
    return linear_head


def _make_dinov3_window_base2_gap_classifier(
    *,
    backbone_kwargs: Optional[Dict[str, Any]] = None,
    backbone_checkpoint: Optional[str | Path] = None,
    pretrained: bool = False,
    classifier_weights: ClassifierWeights | str = ClassifierWeights.IMAGENET1K,
    check_hash: bool = False,
    num_classes: int = 1_000,
    **kwargs: Any,
) -> nn.Module:
    backbone_kwargs = backbone_kwargs or {}
    backbone = DinoVisionTransformerWindow(**backbone_kwargs)
    if backbone_checkpoint is not None:
        checkpoint_path = Path(backbone_checkpoint)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        backbone.load_state_dict(state_dict, strict=False)
    embed_dim = backbone.embed_dim
    linear_head = _make_dinov3_window_base2_gap_classification_head(
        embed_dim=embed_dim,
        num_classes=num_classes,
        pretrained=pretrained,
        classifier_weights=classifier_weights,
        check_hash=check_hash,
        **kwargs,
    )
    return _GapLinearClassifierWrapper(backbone=backbone, linear_head=linear_head)


def dinov3_window_base2_gap_lc(
    *,
    backbone_kwargs: Optional[Dict[str, Any]] = None,
    backbone_checkpoint: Optional[str | Path] = None,
    pretrained: bool = False,
    weights: ClassifierWeights | str = ClassifierWeights.IMAGENET1K,
    check_hash: bool = False,
    num_classes: int = 1_000,
    **kwargs: Any,
) -> nn.Module:
    """Linear classifier on top of GAP features from a DINOv3 window baseline2 backbone."""
    return _make_dinov3_window_base2_gap_classifier(
        backbone_kwargs=backbone_kwargs,
        backbone_checkpoint=backbone_checkpoint,
        pretrained=pretrained,
        classifier_weights=weights,
        check_hash=check_hash,
        num_classes=num_classes,
        **kwargs,
    )


__all__ = [
    "ClassifierWeights",
    "_GapLinearClassifierWrapper",
    "_make_dinov3_window_base2_gap_classification_head",
    "_make_dinov3_window_base2_gap_classifier",
    "dinov3_window_base2_gap_lc",
]
