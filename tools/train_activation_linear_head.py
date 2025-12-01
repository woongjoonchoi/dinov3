"""Train a linear classifier using precomputed backbone activations.

The script loads activations saved by ``tools/save_dinov3_window_base1_1_activations.py``
or similar utilities and optimizes only a linear head. Both training and
validation losses are logged to Weights & Biases for easy comparison.
"""
from __future__ import annotations

import argparse
import os
import pathlib
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm


class ActivationDataset(Dataset[Tuple[torch.Tensor, int]]):
    """Dataset for activation files arranged in class-named folders."""

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        class_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
        if not class_dirs:
            raise RuntimeError(f"No class folders found under {root}.")
        self.classes: List[str] = [d.name for d in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        self.samples: List[Tuple[pathlib.Path, int]] = []
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            for path in sorted(class_dir.glob("*.pt")):
                self.samples.append((path, class_idx))
        if not self.samples:
            raise RuntimeError(f"No activation files found under {root}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, target = self.samples[index]
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, torch.Tensor):
            activation = payload
        elif isinstance(payload, dict):
            activation = payload.get("activation") or payload.get("activations")
            if activation is None:
                raise KeyError(f"Missing activation tensor in {path}.")
        else:
            raise TypeError(f"Unsupported payload type in {path}: {type(payload)}")

        if not isinstance(activation, torch.Tensor):
            activation = torch.tensor(activation)
        if activation.ndim != 1:
            activation = activation.view(-1)
        return activation.float(), target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=pathlib.Path, required=True, help="Directory of training activations.")
    parser.add_argument("--val-dir", type=pathlib.Path, required=True, help="Directory of validation activations.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size per device.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate for SGD.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for SGD.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD.")
    parser.add_argument("--checkpoint", type=pathlib.Path, default=pathlib.Path("linear_head.pth"), help="Where to save the trained head.")
    parser.add_argument("--load-linear-head", type=pathlib.Path, default=None, help="Optional path to a pre-trained linear head to initialize from.")
    parser.add_argument("--wandb-project", type=str, default=None, help="Weights & Biases project name (if set, logging is enabled).")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Optional custom W&B run name.")
    parser.add_argument("--backend", type=str, default="nccl", help="Distributed backend for torch.distributed.init_process_group.")
    parser.add_argument("--local-rank", type=int, default=None, help="Local rank provided by torchrun for DDP training.")
    return parser.parse_args()


@dataclass
class _RunMetadata:
    in_features: int
    num_classes: int
    train_samples: int
    val_samples: int


def _build_loader(
    root: pathlib.Path,
    batch_size: int,
    num_workers: int,
    *,
    sampler: DistributedSampler | None = None,
    shuffle: bool = True,
    dataset: ActivationDataset | None = None,
) -> tuple[DataLoader, ActivationDataset]:
    dataset = dataset or ActivationDataset(root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, dataset


def _init_wandb(args: argparse.Namespace, meta: _RunMetadata):
    if args.wandb_project is None:
        return None
    try:
        import wandb
    except ImportError:  # pragma: no cover - optional dependency
        print("wandb not installed; skipping logging.")
        return None

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config={
        "in_features": meta.in_features,
        "num_classes": meta.num_classes,
        "train_samples": meta.train_samples,
        "val_samples": meta.val_samples,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
    })
    return wandb


def _load_linear_head_state(path: pathlib.Path, device: torch.device) -> dict:
    """Load a linear head state dict, stripping common wrappers."""

    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint type at {path}: {type(state_dict)}")

    # Remove a possible DistributedDataParallel "module." prefix.
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("module."):
            new_key = key[len("module."):]
        cleaned_state_dict[new_key] = value
    return cleaned_state_dict


def _train_one_epoch(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer, *, world_size: int
) -> float:
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    total_samples = torch.tensor(0, device=device)
    for activations, targets in tqdm(loader, desc="train"):
        activations = activations.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(activations)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.detach() * batch_size
        total_samples += batch_size

    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    return (total_loss / total_samples).item()


@torch.inference_mode()
def _evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, *, world_size: int) -> float:
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_samples = torch.tensor(0, device=device)
    for activations, targets in tqdm(loader, desc="val"):
        activations = activations.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(activations)
        loss = criterion(logits, targets)
        batch_size = targets.size(0)
        total_loss += loss * batch_size
        total_samples += batch_size

    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    return (total_loss / total_samples).item()


def main() -> None:
    args = _parse_args()
    local_rank = args.local_rank
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        dist.init_process_group(backend=args.backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_sampler: DistributedSampler | None = None
    val_sampler: DistributedSampler | None = None
    train_dataset = ActivationDataset(args.train_dir)
    val_dataset = ActivationDataset(args.val_dir)
    if distributed:
        train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
        val_sampler = DistributedSampler(dataset=val_dataset, shuffle=False)
    train_loader, train_dataset = _build_loader(
        args.train_dir,
        args.batch_size,
        args.num_workers,
        sampler=train_sampler,
        shuffle=True,
        dataset=train_dataset,
    )
    val_loader, val_dataset = _build_loader(
        args.val_dir,
        args.batch_size,
        args.num_workers,
        sampler=val_sampler,
        shuffle=False,
        dataset=val_dataset,
    )

    sample_activation, _ = train_dataset[0]
    in_features = sample_activation.numel()
    num_classes = len(train_dataset.classes)

    model = torch.nn.Linear(in_features, num_classes).to(device)
    if args.load_linear_head is not None:
        state_dict = _load_linear_head_state(args.load_linear_head, device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Loaded linear head has incompatible parameters: "
                f"missing={missing}, unexpected={unexpected}"
            )

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )

    meta = _RunMetadata(
        in_features=in_features,
        num_classes=num_classes,
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
    )
    is_main_process = not distributed or dist.get_rank() == 0
    wandb = _init_wandb(args, meta) if is_main_process else None

    for epoch in range(1, args.epochs + 1):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss = _train_one_epoch(model, train_loader, device, optimizer, world_size=world_size)
        val_loss = _evaluate(model, val_loader, device, world_size=world_size)
        if is_main_process:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            if wandb is not None:
                wandb.log({"train/loss": train_loss, "val/loss": val_loss, "epoch": epoch})

    if is_main_process:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        to_save = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(to_save, args.checkpoint)
        if wandb is not None:
            wandb.save(str(args.checkpoint))

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
