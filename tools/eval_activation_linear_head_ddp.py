"""Evaluate a linear classifier head on precomputed activations using DDP.

The script expects activations saved with ``torch.save`` in the following
format::

    torch.save(
        {
            "activation": activation,  # Tensor expected as the linear head input
            "class_idx": int(target),
            "class_name": class_name,
            "source_path": str(path),
        },
        output_path,
    )

Files should be arranged similarly to ImageNet folders, i.e.::

    <data_dir>/val/<class_name>/sample0.pt
    <data_dir>/val/<class_name>/sample1.pt
    ...

Usage examples::

    # Single GPU / CPU
    python tools/eval_activation_linear_head_ddp.py \
        --data-dir /path/to/val \
        --linear-head-checkpoint /path/to/linear_head.pth

    # Multi-GPU
    torchrun --nproc_per_node=8 tools/eval_activation_linear_head_ddp.py \
        --data-dir /path/to/val \
        --linear-head-checkpoint /path/to/linear_head.pth \
        --distributed
"""

from __future__ import annotations

import argparse
import os
import pathlib
from typing import List, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ActivationDataset(Dataset[Tuple[torch.Tensor, int]]):
    """Dataset for activations saved under class-named folders."""

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.samples: List[pathlib.Path] = []
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            self.samples.extend(sorted(class_dir.glob("*.pt")))
        if not self.samples:
            raise RuntimeError(f"No activation files found under {root}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path = self.samples[index]
        data = torch.load(path)
        if "activation" not in data:
            raise KeyError(f"Missing 'activation' key in {path}.")
        activation = data["activation"]
        if not isinstance(activation, torch.Tensor):
            activation = torch.tensor(activation)
        activation = activation.float()
        target = int(data.get("class_idx"))
        return activation, target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True, help="Directory containing activation .pt files.")
    parser.add_argument("--linear-head-checkpoint", type=pathlib.Path, required=True, help="Path to the linear head checkpoint.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size per device.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    parser.add_argument("--distributed", action="store_true", help="Use torch.distributed with torchrun for multi-GPU evaluation.")
    parser.add_argument("--pin-memory", action="store_true", help="Pin dataloader memory for faster host->device copies.")
    return parser.parse_args()


def _init_distributed():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device, dist.get_rank(), dist.get_world_size()


def _build_loader(
    root: pathlib.Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    distributed: bool,
    rank: int,
    world_size: int,
) -> DataLoader:
    dataset = ActivationDataset(root)
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


def _load_linear_head(checkpoint: pathlib.Path, device: torch.device) -> torch.nn.Module:
    state_dict = torch.load(checkpoint, map_location="cpu")
    if "weight" not in state_dict:
        raise KeyError("Linear head checkpoint must contain a 'weight' tensor.")
    out_features, in_features = state_dict["weight"].shape
    model = torch.nn.Linear(in_features, out_features)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def evaluate_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    distributed: bool,
) -> float:
    total_correct = torch.tensor(0, device=device, dtype=torch.long)
    total_samples = torch.tensor(0, device=device, dtype=torch.long)

    for activations, targets in tqdm(loader):
        activations = activations.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(activations)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == targets).sum()
        total_samples += targets.numel()
    if distributed:
        dist.all_reduce(total_correct)
        dist.all_reduce(total_samples)
    if total_samples.item() == 0:
        raise RuntimeError("Dataset appears to be empty.")
    return (total_correct.float() / total_samples.float()).item()


def main() -> None:
    args = _parse_args()
    if args.distributed:
        device, rank, world_size = _init_distributed()
    else:
        device = torch.device(args.device)
        rank, world_size = 0, 1

    model = _load_linear_head(args.linear_head_checkpoint, device)
    if args.distributed:
        model = DDP(model, device_ids=[device] if device.type == "cuda" else None)

    loader = _build_loader(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        args.pin_memory,
        args.distributed,
        rank,
        world_size,
    )

    accuracy = evaluate_dataset(model, loader, device, args.distributed)
    total_samples = len(loader.dataset)
    if not args.distributed or rank == 0:
        print(f"accuracy: {accuracy * 100:.2f}% ({total_samples} samples)")

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
