"""Extract DinoV3 Baseline1_3 backbone activations and save them per image/class.

This script mirrors ``tools/save_backbone_activations.py`` but targets the
``LocalGlobalHybridVisionTransformer`` backbone. For each image in an
``ImageFolder``-style dataset it stores a tensor containing the concatenated
class token and mean patch token activations so they can be reused for tasks
such as training a linear classifier.
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

from dinov3_window_base1_3 import LocalGlobalHybridVisionTransformer


class ImageFolderWithPaths(datasets.ImageFolder):
    """ImageFolder that also returns the image path."""

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, target, path


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


def _build_loader(
    root: pathlib.Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    distributed: bool,
    rank: int,
    world_size: int,
) -> DataLoader:
    dataset = ImageFolderWithPaths(root=str(root), transform=make_transform())
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
    parser.add_argument("--data-dir", type=pathlib.Path, required=True, help="Path to the dataset root.")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True, help="Directory to write activation files.")
    parser.add_argument("--backbone-checkpoint", type=pathlib.Path, required=True, help="Path to the backbone checkpoint.")
    parser.add_argument(
        "--model-kwargs",
        type=json.loads,
        default={},
        help="JSON object of keyword arguments forwarded to LocalGlobalHybridVisionTransformer.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per device.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    parser.add_argument("--distributed", action="store_true", help="Use torch.distributed with torchrun for multi-GPU extraction.")
    parser.add_argument("--pin-memory", action="store_true", help="Pin dataloader memory for faster host->device copies.")
    return parser.parse_args()


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


@torch.inference_mode()
def _extract_and_save(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_root: pathlib.Path,
    distributed: bool,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    for images, targets, paths in tqdm(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        features = model.module.forward_features(images) if distributed else model.forward_features(images)

        cls_token = features["x_norm_clstoken"]
        patch_tokens = features["x_norm_patchtokens"]
        activations = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1).cpu()

        for activation, target, path in zip(activations, targets, paths):
            class_name = loader.dataset.classes[target]
            class_dir = output_root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            image_stem = pathlib.Path(path).stem
            output_path = class_dir / f"{image_stem}.pt"

            torch.save(
                {
                    "activation": activation.clone(),
                    "class_idx": int(target),
                    "class_name": class_name,
                    "source_path": str(path),
                },
                output_path,
            )


def _build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    backbone = LocalGlobalHybridVisionTransformer(**args.model_kwargs)
    backbone_state = _load_state_dict(args.backbone_checkpoint)
    backbone.load_state_dict(backbone_state, strict=True)
    backbone.to(device)
    return backbone


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

    loader = _build_loader(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        args.pin_memory,
        args.distributed,
        rank,
        world_size,
    )

    _extract_and_save(model, loader, device, args.output_dir, args.distributed)

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
