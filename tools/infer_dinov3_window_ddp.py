"""Run ImageNet-style inference with a DinoV3 window backbone using DDP.

This script mirrors :mod:`tools.eval_imagenet_accuracy_ddp` but targets the
``DinoVisionTransformerWindow`` backbone. It constructs the backbone and a
linear classifier head from local checkpoints, wraps them with
``DistributedDataParallel`` when requested, and reports per-split top-1
accuracies.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm

from dinov3.hub.classifiers import _LinearClassifierWrapper
from dionv3_window import DinoVisionTransformerWindow


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

    for images, targets in tqdm(loader):
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
    parser.add_argument("--distributed", action="store_true", help="Use torch.distributed with torchrun for multi-GPU evaluation.")
    parser.add_argument("--pin-memory", action="store_true", help="Pin dataloader memory for faster host->device copies.")
    parser.add_argument("--backbone-checkpoint", type=pathlib.Path, required=True, help="Checkpoint containing the DinoVisionTransformerWindow weights.")
    parser.add_argument("--head-checkpoint", type=pathlib.Path, required=True, help="Linear head checkpoint aligned with the backbone.")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of output classes for the classifier head.")
    parser.add_argument(
        "--model-kwargs",
        type=json.loads,
        default={},
        help="JSON object of keyword arguments forwarded to DinoVisionTransformerWindow (e.g. '{\\"embed_dim\\": 1024}')",
    )
    args = parser.parse_args()
    if args.train_dir is None and args.val_dir is None:
        parser.error("At least one of --train-dir or --val-dir must be specified.")
    return args


def _load_state_dict(path: pathlib.Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unexpected checkpoint structure in {path}")
    if any(k.startswith("module.") for k in checkpoint):
        checkpoint = {k.removeprefix("module."): v for k, v in checkpoint.items()}
    return checkpoint


def _build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    backbone = DinoVisionTransformerWindow(**args.model_kwargs)
    backbone_state = _load_state_dict(args.backbone_checkpoint)
    backbone.load_state_dict(backbone_state, strict=True)

    embed_dim = backbone.embed_dim
    linear_in_dim = 2 * embed_dim
    linear_head = torch.nn.Linear(linear_in_dim, args.num_classes)
    head_state = _load_state_dict(args.head_checkpoint)
    linear_head.load_state_dict(head_state, strict=True)

    model = _LinearClassifierWrapper(backbone=backbone, linear_head=linear_head)
    model.to(device)
    return model


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

    model = _build_model(args, device)
    if args.distributed:
        model = DDP(model, device_ids=[device] if device.type == "cuda" else None)
    model.eval()

    splits: list[tuple[str, pathlib.Path]] = []
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
