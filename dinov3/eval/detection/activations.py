"""Utilities for running the PlainDETR head on cached backbone activations."""

from __future__ import annotations

import pathlib
from typing import List, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from .models.backbone import build_backbone
from .models.detr import PlainDETR, PlainDETRHeadOnly
from .models.transformer import build_transformer


class ActivationDetDataset(Dataset):
    """Dataset wrapper that loads cached backbone activations from disk."""

    def __init__(self, base_dataset: Dataset, activation_root: Union[str, pathlib.Path]):
        self.base = base_dataset
        self.activation_root = pathlib.Path(activation_root)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        _, target = self.base[idx]
        file_name = target["file_name"]

        act_path = (self.activation_root / file_name).with_suffix(".pt")
        act = torch.load(act_path, map_location="cpu")

        out_tensors = act["out_tensors"]
        out_masks = act["out_masks"]
        pos = act["pos"]

        srcs = [t.unsqueeze(0) for t in out_tensors]
        masks = [m.unsqueeze(0) for m in out_masks]
        pos_list = [p.unsqueeze(0) for p in pos]

        meta = act.get("meta", {})
        return srcs, masks, pos_list, target, meta


def collate_activations(batch):
    """Stack per-level activations along the batch dimension."""

    batch_size = len(batch)
    num_levels = len(batch[0][0])

    all_srcs: List[torch.Tensor] = []
    all_masks: List[torch.Tensor] = []
    all_pos: List[torch.Tensor] = []

    for lvl in range(num_levels):
        src_level = [batch[i][0][lvl] for i in range(batch_size)]
        mask_level = [batch[i][1][lvl] for i in range(batch_size)]
        pos_level = [batch[i][2][lvl] for i in range(batch_size)]

        all_srcs.append(torch.cat(src_level, dim=0))
        all_masks.append(torch.cat(mask_level, dim=0))
        all_pos.append(torch.cat(pos_level, dim=0))

    targets = [b[3] for b in batch]
    metas = [b[4] for b in batch]

    return all_srcs, all_masks, all_pos, targets, metas


def build_head_only_model(
    args,
    backbone_model: torch.nn.Module,
    checkpoint_path: Union[str, pathlib.Path],
    device: torch.device,
) -> PlainDETRHeadOnly:
    """Construct a head-only model on top of a CPU PlainDETR base."""

    backbone = build_backbone(backbone_model, args)
    transformer = build_transformer(args)
    det_full = PlainDETR(
        backbone,
        transformer,
        num_classes=args.num_classes,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        num_queries_one2one=args.num_queries_one2one,
        num_queries_one2many=args.num_queries_one2many,
        mixed_selection=args.mixed_selection,
    )
    det_full.to("cpu")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    det_full.load_state_dict(state, strict=False)

    head = PlainDETRHeadOnly(det_full)
    head.to(device)
    head.eval()
    return head


def build_activation_dataloader(
    base_dataset: Dataset,
    activation_root: Union[str, pathlib.Path],
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool = False,
    pin_memory: bool = True,
    sampler=None,
    dataset: ActivationDetDataset | None = None,
) -> DataLoader:
    """Return a DataLoader that serves cached activation batches."""

    dataset = dataset or ActivationDetDataset(base_dataset, activation_root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_activations,
    )


@torch.inference_mode()
def eval_head_only(
    head: PlainDETRHeadOnly,
    activation_loader: DataLoader,
    device: torch.device,
    criterion=None,
    postprocessors=None,
    *,
    target_size_key: str = "orig_size",
) -> Tuple[list[dict], list]:
    """Run a head-only evaluation loop.

    Returns a tuple ``(losses, processed_outputs)`` where losses are detached
    dictionaries (if a criterion is provided) and processed_outputs contain the
    post-processed predictions (if a postprocessor is provided).
    """

    head.eval()
    losses: list[dict] = []
    processed: list = []

    for srcs, masks, pos, targets, _ in activation_loader:
        srcs = [s.to(device, non_blocking=True) for s in srcs]
        masks = [m.to(device, non_blocking=True) for m in masks]
        pos = [p.to(device, non_blocking=True) for p in pos]

        outputs = head(srcs, masks, pos)

        if criterion is not None:
            loss_dict = criterion(outputs, targets)
            losses.append({k: v.detach().cpu() for k, v in loss_dict.items()})

        if postprocessors is not None and "bbox" in postprocessors:
            target_sizes = torch.stack([torch.as_tensor(t[target_size_key]) for t in targets]).to(device)
            original_sizes = None
            if target_size_key != "orig_size" and "orig_size" in targets[0]:
                original_sizes = torch.stack([torch.as_tensor(t["orig_size"]) for t in targets]).to(device)
            processed.extend(postprocessors["bbox"](outputs, target_sizes, original_sizes))

    return losses, processed

