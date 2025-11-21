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
import os
import pathlib
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
def evaluate_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    distributed: bool,
) -> float:
    total_correct = torch.tensor(0, device=device, dtype=torch.long)
    total_samples = torch.tensor(0, device=device, dtype=torch.long)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == targets).sum()
        total_samples += targets.numel()
    if distributed:
        dist.all_reduce(total_correct)
        dist.all_reduce(total_samples)
    if total_samples.item() == 0:
        raise RuntimeError("Dataset appears to be empty.")
    return (total_correct.float() / total_samples.float()).item()


def _build_loader(
    root: pathlib.Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    distributed: bool,
    rank: int,
    world_size: int,
) -> DataLoader:
    dataset = datasets.ImageFolder(root=str(root), transform=make_transform())
    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
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
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use torch.distributed with torchrun for multi-GPU evaluation.",
    )
    parser.add_argument("--pin-memory", action="store_true", help="Pin dataloader memory for faster host->device copies.")
    args = parser.parse_args()
    if args.train_dir is None and args.val_dir is None:
        parser.error("At least one of --train-dir or --val-dir must be specified.")
    return args


def main() -> None:
    args = _parse_args()
    if args.distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = torch.device(args.device)
        rank = 0
        world_size = 1
    classifier_weights = _resolve_weights(args.classifier_weights, ClassifierWeights)
    backbone_weights = _resolve_weights(args.backbone_weights, BackboneWeights)

    model = dinov3_vit7b16_lc(weights=classifier_weights, backbone_weights=backbone_weights)
    model.to(device)
    if args.distributed:
        model = DDP(model, device_ids=[device] if device.type == "cuda" else None)
    model.eval()

    splits = []
    if args.train_dir is not None:
        splits.append(("train", args.train_dir))
    if args.val_dir is not None:
        splits.append(("val", args.val_dir))

    for split_name, split_dir in splits:
        loader = _build_loader(
            split_dir,
            args.batch_size,
            args.num_workers,
            args.pin_memory,
            args.distributed,
            rank,
            world_size,
        )
        accuracy = evaluate_dataset(model, loader, device, args.distributed)
        total_images = len(loader.dataset)
        if not args.distributed or rank == 0:
            print(f"{split_name} accuracy: {accuracy * 100:.2f}% ({total_images} images)")

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
