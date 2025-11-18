"""Utility to evaluate ImageNet train/validation accuracy with the DINOv3 classifier.

This script loads the ViT-7B/16 linear classifier head released with DINOv3 and
computes the top-1 accuracy on the provided ImageNet splits.  The script only
requires the directory structure that ``torchvision.datasets.ImageFolder``
expects, i.e. ``<split>/<class_name>/<image>.JPEG``.

Example::

    python tools/eval_imagenet_accuracy.py \
        --train-dir /datasets/imagenet/train \
        --val-dir /datasets/imagenet/val \
        --batch-size 128 --device cuda
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from dinov3.hub.classifiers import (
    ClassifierWeights,
    dinov3_vit7b16_lc,
)
from dinov3.hub.backbones import Weights as BackboneWeights


def _resolve_weights(value: Optional[str], enum_type):
    """Convert a CLI weight argument to the correct type for torch hub loaders."""

    if value is None:
        return value
    normalized = value.strip().upper()
    try:
        return enum_type[normalized]
    except KeyError:
        # Treat as path or URL.
        return value


def make_transform(resize_size: int = 256, crop_size: Optional[int] = 224):
    """Return the standard ImageNet transform recommended in the README."""

    to_image = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    center_crop = []
    if crop_size is not None:
        center_crop = [v2.CenterCrop(crop_size)]
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return v2.Compose([to_image, resize, *center_crop, to_float, normalize])


@torch.inference_mode()
def evaluate_dataset(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    total_correct = 0
    total_samples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == targets).sum().item()
        total_samples += targets.numel()
    if total_samples == 0:
        raise RuntimeError("Dataset appears to be empty.")
    return total_correct / total_samples


def _build_loader(root: pathlib.Path, batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
    dataset = datasets.ImageFolder(root=str(root), transform=make_transform())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=pathlib.Path, default=None, help="Path to the ImageNet training split.")
    parser.add_argument("--val-dir", type=pathlib.Path, default=None, help="Path to the ImageNet validation split.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size per device.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    parser.add_argument(
        "--classifier-weights",
        type=str,
        default=ClassifierWeights.IMAGENET1K.name,
        help="Classifier weights enum name, checkpoint path, or URL.",
    )
    parser.add_argument(
        "--backbone-weights",
        type=str,
        default=BackboneWeights.LVD1689M.name,
        help="Backbone weights enum name, checkpoint path, or URL.",
    )
    parser.add_argument("--pin-memory", action="store_true", help="Pin dataloader memory for faster host->device copies.")
    args = parser.parse_args()
    if args.train_dir is None and args.val_dir is None:
        parser.error("At least one of --train-dir or --val-dir must be specified.")
    return args


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    classifier_weights = _resolve_weights(args.classifier_weights, ClassifierWeights)
    backbone_weights = _resolve_weights(args.backbone_weights, BackboneWeights)

    model = dinov3_vit7b16_lc(weights=classifier_weights, backbone_weights=backbone_weights)
    model.to(device)
    model.eval()

    splits = []
    if args.train_dir is not None:
        splits.append(("train", args.train_dir))
    if args.val_dir is not None:
        splits.append(("val", args.val_dir))

    for split_name, split_dir in splits:
        loader = _build_loader(split_dir, args.batch_size, args.num_workers, args.pin_memory)
        accuracy = evaluate_dataset(model, loader, device)
        total_images = len(loader.dataset)
        print(f"{split_name} accuracy: {accuracy * 100:.2f}% ({total_images} images)")


if __name__ == "__main__":
    main()
