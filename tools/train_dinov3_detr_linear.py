#!/usr/bin/env python
"""Train DINOv3 DETR head with a frozen ViT backbone using DDP."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from torchvision.transforms.functional import (
    InterpolationMode,
)
from tqdm import tqdm

from scipy.optimize import linear_sum_assignment

from dinov3.hub.backbones import dinov3_vit7b16, dinov3_vitl16plus
from dinov3.eval.detection.config import DetectionHeadConfig
from dinov3.eval.detection.models.detr import PostProcess, build_model
from dinov3.eval.detection.models.position_encoding import PositionEncoding
from detr_utils import box_ops


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def make_wandb_run_name(args: argparse.Namespace) -> str:
    prefix = getattr(args, "experiment_name", None) or "coco-detr"
    parts = [
        prefix,
        "coco",
        args.backbone_name,
        f"{args.split}-{args.val_split}",
    ]
    if args.max_size is not None:
        parts.append(f"res{args.max_size}")
    parts.extend(
        [
            f"ep{args.epochs}",
            f"bs{args.batch_size}",
            f"lr{args.lr:g}",
        ]
    )
    if args.weight_decay > 0:
        parts.append(f"wd{args.weight_decay:g}")
    seed = getattr(args, "seed", None)
    if seed is not None:
        parts.append(f"s{seed}")
    return "_".join(parts)


@dataclass
class _RunMetadata:
    num_classes: int
    train_images: int
    val_images: int
    backbone_name: str
    max_size: Optional[int]


def _init_wandb(args: argparse.Namespace, meta: _RunMetadata):
    if args.wandb_project is None:
        return None

    try:
        import wandb
        from wandb.errors import CommError
    except ImportError:
        print("wandb not installed; skipping logging.")
        return None

    try:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "dataset": "coco",
                "train_split": args.split,
                "val_split": args.val_split,
                "num_classes": meta.num_classes,
                "backbone_name": meta.backbone_name,
                "max_size": meta.max_size,
                "train_images": meta.train_images,
                "val_images": meta.val_images,
                "batch_size": args.batch_size,
                "val_batch_size": args.val_batch_size,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "score_threshold": args.score_threshold,
            },
            settings=wandb.Settings(
                insecure_disable_ssl=True,
            ),
        )
    except CommError as e:
        print(f"[WARN] wandb init failed (network/SSL): {e}")
        print("       → continuing WITHOUT wandb logging.")
        return None

    return wandb


def _load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict", "model_state"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unexpected checkpoint structure in {path}")
    if any(k.startswith("module.") for k in checkpoint):
        checkpoint = {k.removeprefix("module."): v for k, v in checkpoint.items()}
    return checkpoint


def _remap_window_block_keys_to_patch_only(model, state_dict: Dict[str, torch.Tensor]):
    from dinov3_window_base1_1.vit import _PatchOnlyWindowBlock

    window_block_indices = [i for i, blk in enumerate(model.blocks) if isinstance(blk, _PatchOnlyWindowBlock)]
    if not window_block_indices:
        return state_dict

    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for idx in window_block_indices:
            prefix = f"blocks.{idx}."
            nested_prefix = f"blocks.{idx}.patch_block."
            if key.startswith(nested_prefix):
                break
            if key.startswith(prefix):
                new_key = nested_prefix + key[len(prefix) :]
                break
        remapped[new_key] = value
    return remapped


def _build_model_from_checkpoints(args: argparse.Namespace, device: torch.device,num_classes=91) -> nn.Module:
    # print(f"num_classes: {num_classes}")
    # exit()
    detection_kwargs = dict(
        with_box_refine=True,
        two_stage=True,
        mixed_selection=True,
        look_forward_twice=True,
        k_one2many=6,
        lambda_one2many=1.0,
        num_queries_one2one=1500,
        num_queries_one2many=1500,
        reparam=True,
        position_embedding=PositionEncoding.SINE,
        num_feature_levels=1,
        dec_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        norm_type="pre_norm",
        proposal_feature_levels=4,
        proposal_min_size=50,
        decoder_type="global_rpe_decomp",
        decoder_use_checkpoint=False,
        decoder_rpe_hidden_dim=512,
        decoder_rpe_type="linear",
        layers_to_use=None,
        blocks_to_train=None,
        add_transformer_encoder=True,
        num_encoder_layers=6,
        backbone_use_layernorm=False,
        num_classes=91,
        # num_classes=num_classes,
        aux_loss=True,
        topk=1500,
        hidden_dim=768,
        nheads=8,
    )
    config = DetectionHeadConfig(**detection_kwargs)
    n_windows_sqrt_map = {"dinov3_vit7b16": 3, "dinov3_vitl16plus": 2}

    backbone_class_map = {
        "dinov3_vit7b16": dinov3_vit7b16,
        "dinov3_vitl16plus": dinov3_vitl16plus,
        "dinov3_window_base1_1": None,
        "dinov3_window_base1_3": None,
        "b1_1": None,
        "b1_3": None,
    }
    if args.backbone_name not in backbone_class_map:
        raise ValueError(f"Unsupported backbone_name: {args.backbone_name}")

    if args.backbone_name in {"dinov3_window_base1_1", "dinov3_window_base1_3", "b1_1", "b1_3"}:
        # local backbones
        if args.backbone_name in {"dinov3_window_base1_1", "b1_1"}:
            from dinov3_window_base1_1 import DinoVisionTransformerWindowBaseline1_1 as backbone_class
        else:
            from dinov3_window_base1_3 import LocalGlobalHybridVisionTransformer as backbone_class
        backbone = backbone_class(pretrained=False, weights=None, check_hash=False, window_size=16)
        state_dict = _load_checkpoint(args.backbone_checkpoint)
        state_dict = _remap_window_block_keys_to_patch_only(backbone, state_dict)
        backbone.load_state_dict(state_dict, strict=True)
        n_windows_sqrt = 3 if args.backbone_name in {"dinov3_window_base1_1", "b1_1"} else 2
    else:
        backbone = backbone_class_map[args.backbone_name](pretrained=False, weights=None, check_hash=False)
        n_windows_sqrt = n_windows_sqrt_map.get(args.backbone_name, 0)
        backbone_state = _load_checkpoint(args.backbone_checkpoint)
        backbone.load_state_dict(backbone_state, strict=False)

    config.n_windows_sqrt = n_windows_sqrt
    config.proposal_in_stride = backbone.patch_size
    config.proposal_tgt_strides = [int(m * backbone.patch_size) for m in (0.5, 1, 2, 4)]
    if config.layers_to_use is None:
        config.layers_to_use = [m * backbone.n_blocks // 4 - 1 for m in range(1, 5)]

    detector = build_model(backbone, config)
    if args.detector_checkpoint is not None:
        detector_state = _load_checkpoint(args.detector_checkpoint)
        detector.load_state_dict(detector_state, strict=False)
    detector.num_queries = detector.num_queries_one2one
    detector.transformer.two_stage_num_proposals = detector.num_queries

    model = DetectorWithPostProcess(detector=detector, postprocessor=PostProcess(config.topk, config.reparam))
    model.to(device)
    return model


class DetectorWithPostProcess(nn.Module):
    def __init__(self, detector, postprocessor):
        super().__init__()
        self.detector = detector
        self.postprocessor = postprocessor

    def forward(self, samples: list[torch.Tensor]):
        outputs = self.detector(samples)
        return outputs

    def postprocess(self, outputs, samples: list[torch.Tensor], metas: List[dict]):
        sizes_tensor = torch.tensor([sample.shape[1:] for sample in samples], device=samples[0].device)
        orig_sizes = torch.tensor([m["orig_size"] for m in metas], device=samples[0].device)
        return self.postprocessor(outputs, target_sizes=sizes_tensor, original_target_sizes=orig_sizes)


class ResizeAllSides:
    def __init__(self, target_size: int):
        self.target_size = target_size

    def __call__(self, image):
        width, height = image.size
        if width == self.target_size and height == self.target_size:
            return image
        return F.resize(image, (self.target_size, self.target_size), interpolation=InterpolationMode.BICUBIC)


class CocoDetectionWithTargets(CocoDetection):
    def __init__(self, root: str, ann_file: str, transform: transforms.Compose, max_size: int | None = None):
        super().__init__(root=root, annFile=ann_file)
        self.image_transform = transform
        self.max_size = max_size
        self._cat_id_to_contiguous = {cat_id: idx for idx, cat_id in enumerate(sorted(self.coco.getCatIds()))}
        self._contiguous_to_cat_id = {v: k for k, v in self._cat_id_to_contiguous.items()}

    def __getitem__(self, index):
        image, targets = super().__getitem__(index)
        image_id = self.ids[index]
        orig_w, orig_h = image.size
        if self.max_size:
            image = ResizeAllSides(self.max_size)(image)
        resized_w, resized_h = image.size
        image = self.image_transform(image)
        boxes = []
        labels = []
        for ann in targets:
            bbox = ann.get("bbox")
            if bbox is None:
                continue
            x_min, y_min, w, h = bbox
            x_center = x_min + 0.5 * w
            y_center = y_min + 0.5 * h
            boxes.append([x_center / resized_w, y_center / resized_h, w / resized_w, h / resized_h])
            labels.append(self._cat_id_to_contiguous[ann["category_id"]])
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
            "orig_size": torch.as_tensor([orig_h, orig_w]),
            "resized_size": torch.as_tensor([resized_h, resized_w]),
            "label_to_coco": self._contiguous_to_cat_id,
        }
        return image, target


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


def _collate_fn(batch: Iterable[Tuple[torch.Tensor, dict]]):
    images, targets = zip(*batch)
    return list(images), list(targets)


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "All costs can't be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(out_bbox), box_ops.box_cxcywh_to_xyxy(tgt_bbox)
        )
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for c, size in zip(C, sizes):
            if size == 0:
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue
            row_ind, col_ind = linear_sum_assignment(c[:, :size])
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64), torch.as_tensor(col_ind, dtype=torch.int64)))
        return [(i[0].to(out_prob.device), i[1].to(out_prob.device)) for i in indices]


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, device=src_logits.device, dtype=torch.int64)
        target_classes[idx] = target_classes_o
        # print(f"empty_weight shape :{self.empty_weight.shape}")
        # print(f"src_logits shape :{src_logits.shape}") 
        # exit()

        loss_ce = nn.functional.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = nn.functional.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=outputs["pred_logits"].device)
        pred_logits = outputs["pred_logits"]
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)

        # 4. Loss 계산
        # card_pred를 float로 변환하여 정답과 비교
        card_err = nn.functional.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err.sum() / outputs["pred_logits"].shape[0]}
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs["pred_logits"].device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(getattr(self, f"loss_{loss}")(outputs, targets, indices, num_boxes))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = getattr(self, f"loss_{loss}")(aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


def build_criterion(num_classes: int, num_decoder_layers: int) -> Tuple[SetCriterion, Dict[str, float]]:
    matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)
    weight_dict: Dict[str, float] = {"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2}
    if num_decoder_layers > 1:
        for i in range(num_decoder_layers - 1):
            weight_dict.update({f"loss_ce_{i}": 1, f"loss_bbox_{i}": 5, f"loss_giou_{i}": 2})
    losses = ["labels", "boxes", "cardinality"]
    criterion = SetCriterion(num_classes=num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1, losses=losses)
    return criterion, weight_dict


def coco_id_mapping(coco: COCO) -> dict[int, int]:
    cat_ids = sorted(coco.getCatIds())
    return {idx: cat_id for idx, cat_id in enumerate(cat_ids)}


def evaluate_predictions(coco_gt: COCO, predictions: List[dict]) -> Dict[str, float]:
    if len(predictions) == 0:
        raise RuntimeError("No predictions were produced; cannot run COCO evaluation.")

    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.imgIds = coco_gt.getImgIds()
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats
    keys = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    return {f"val/{k}": float(v) for k, v in zip(keys, stats)}


def evaluate_predictions_with_logging(
    coco_gt: COCO,
    predictions: List[dict],
    *,
    iteration: int | None = None,
    require_nonempty: bool = True,
) -> None:
    prefix = "Final" if iteration is None else f"Iteration {iteration}"
    if len(predictions) == 0:
        message = f"{prefix}: no predictions; skipping evaluation"
        if require_nonempty:
            raise RuntimeError(message)
        print(message)
        return

    print(f"{prefix}: evaluating {len(predictions)} predictions")
    evaluate_predictions(coco_gt, predictions)


def evaluate(
    model: DetectorWithPostProcess,
    criterion: SetCriterion,
    dataloader: DataLoader,
    device: torch.device,
    distributed: bool,
    rank: int,
    world_size: int,
) -> Tuple[float, Dict[str, float] | None]:
    model.eval()
    criterion.eval()
    base_model = model.module if isinstance(model, DDP) else model
    predictions_all: List[dict] = []
    losses_total = 0.0
    dataset = dataloader.dataset
    with torch.no_grad():
        for iteration, (images, targets) in enumerate(tqdm(dataloader, desc="Validating", total=len(dataloader)), start=1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            losses_total += loss.item()
            processed = base_model.postprocess(outputs, images, targets)
            batch_predictions: List[dict] = []
            for pred, target in zip(processed, targets):
                boxes = pred["boxes"].cpu()
                scores = pred["scores"].cpu()
                labels = pred["labels"].cpu()
                label_map = target.get("label_to_coco", {})
                image_id = int(target["image_id"].item())
                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box.tolist()
                    coco_box = [x_min, y_min, x_max - x_min, y_max - y_min]
                    # coco_label = int(label_map.get(int(label), int(label)))
                    coco_label = int(label)
                    batch_predictions.append(
                        {"image_id": image_id, "category_id": coco_label, "bbox": coco_box, "score": float(score)}
                    )

            if distributed:
                gathered_batches: List[List[dict]] = [None for _ in range(world_size)]  # type: ignore[list-item]
                dist.all_gather_object(gathered_batches, batch_predictions)
                if rank == 0:
                    merged_batch = [pred for sublist in gathered_batches for pred in sublist]
                    predictions_all.extend(merged_batch)
                    evaluate_predictions_with_logging(
                        dataset.coco, predictions_all, iteration=iteration, require_nonempty=False
                    )
            else:
                predictions_all.extend(batch_predictions)
                evaluate_predictions_with_logging(
                    dataset.coco, predictions_all, iteration=iteration, require_nonempty=False
                )

    total_tensor = torch.tensor([losses_total], device=device)
    if distributed:
        dist.all_reduce(total_tensor)
    mean_loss = total_tensor.item() / (len(dataloader) * (world_size if distributed else 1))

    if distributed:
        gathered_predictions: List[List[dict]] = [None for _ in range(world_size)]  # type: ignore[list-item]
        dist.all_gather_object(gathered_predictions, predictions_all)
        if rank == 0:
            predictions_all = [p for sub in gathered_predictions for p in sub]
        else:
            predictions_all = []

    metrics: Dict[str, float] | None = None
    if (not distributed) or rank == 0:
        metrics = evaluate_predictions(dataloader.dataset.coco, predictions_all)
    return mean_loss, metrics


def train_one_epoch(
    model: DetectorWithPostProcess,
    criterion: SetCriterion,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int,
    global_step: int,
    wandb: Any | None,
) -> int:
    model.train()
    criterion.train()
    for images, targets in tqdm(dataloader, desc=f"Epoch {epoch}", total=len(dataloader)):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        losses = sum(
            loss_dict[k] * criterion.weight_dict[k]
            for k in loss_dict
            if k in criterion.weight_dict
        )
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=0.1
        )
        optimizer.step()

        if global_step % log_interval == 0 and wandb is not None:
            log_data = {"train/loss": losses.item()}
            for k, v in loss_dict.items():
                if k.startswith("loss"):
                    log_data[f"train/{k}"] = v.item()
            for i, g in enumerate(optimizer.param_groups):
                if "lr" in g:
                    log_data[f"train/lr_group_{i}"] = g["lr"]
            wandb.log(log_data, step=global_step)
        global_step += 1
    return global_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINOv3 detector on COCO with frozen backbone.")
    parser.add_argument("--experiment-name", type=str, default=None, help="Optional experiment name prefix for wandb run name.")
    parser.add_argument("--coco-root", required=True, help="Path to COCO dataset root (containing annotations/ and split folders).")
    parser.add_argument("--split", default="train2017", help="Image split to train (e.g., train2017).")
    parser.add_argument("--val-split", default="val2017", help="Image split to validate (e.g., val2017).")
    parser.add_argument(
        "--ann-file",
        default=None,
        help="Optional path to annotations file for training split; defaults to annotations/instances_<split>.json under --coco-root.",
    )
    parser.add_argument(
        "--val-ann-file",
        default=None,
        help="Optional path to annotations file for validation split; defaults to annotations/instances_<val-split>.json under --coco-root.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--val-batch-size", type=int, default=2, help="Batch size for validation.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for training (cuda or cpu).")
    parser.add_argument("--distributed", action="store_true", help="Use torch.distributed with torchrun for multi-GPU training.")
    parser.add_argument("--pin-memory", action="store_true", help="Pin dataloader memory for faster host->device copies.")
    parser.add_argument("--max-size", type=int, default=None, help="Optional maximum size for image sides; keeps aspect ratio square resize.")
    parser.add_argument("--prefetch-factor", type=int, default=1, help="prefetch factor.")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Discard predictions below this confidence (eval only).")
    parser.add_argument("--backbone-name", type=str, default="dinov3_vit7b16", help="backbone-name")
    parser.add_argument("--backbone-checkpoint", type=Path, required=True, help="Path to the pretrained backbone checkpoint to load directly.")
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=None,
        help="Path to the detector head checkpoint to load directly (optional).",
    )
    parser.add_argument("--output-json", default="coco_predictions.json", help="Where to store raw COCO-format predictions.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for head parameters.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory to store checkpoints and logs.")
    parser.add_argument("--wandb-project", type=str, default=None, help="Weights & Biases project name.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Weights & Biases run name.")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval in steps.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed used for naming.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    args.output_dir.mkdir(parents=True, exist_ok=True)

    coco_root = Path(args.coco_root)
    train_ann = Path(args.ann_file) if args.ann_file is not None else coco_root / "annotations" / f"instances_{args.split}.json"
    val_ann = Path(args.val_ann_file) if args.val_ann_file is not None else coco_root / "annotations" / f"instances_{args.val_split}.json"
    train_root = coco_root / args.split
    val_root = coco_root / args.val_split

    if not train_ann.exists():
        raise FileNotFoundError(f"Annotation file not found: {train_ann}")
    if not val_ann.exists():
        raise FileNotFoundError(f"Annotation file not found: {val_ann}")
    if not train_root.exists():
        raise FileNotFoundError(f"Image directory not found: {train_root}")
    if not val_root.exists():
        raise FileNotFoundError(f"Image directory not found: {val_root}")

    train_dataset = CocoDetectionWithTargets(str(train_root), str(train_ann), transform=build_transform(), max_size=args.max_size)
    val_dataset = CocoDetectionWithTargets(str(val_root), str(val_ann), transform=build_transform(), max_size=args.max_size)

    # num_classes = len(train_dataset.coco.getCatIds())
    # print("num_classes from COCO:", num_classes)
    # exit()

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        pin_memory=args.pin_memory,
        collate_fn=_collate_fn,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        sampler=val_sampler,
        shuffle=False,
        pin_memory=args.pin_memory,
        collate_fn=_collate_fn,
        prefetch_factor=args.prefetch_factor,
    )

    if args.wandb_run_name is None:
        args.wandb_run_name = make_wandb_run_name(args)

    model = _build_model_from_checkpoints(args, device,
                                        #   num_classes=num_classes
                                        
                                          )

    # Freeze backbone parameters
    if hasattr(model.detector, "backbone"):
        backbone_module = model.detector.backbone
        for param in backbone_module.parameters():
            param.requires_grad = False
    for name, module in model.detector.named_modules():
        if "backbone" in name:
            for param in module.parameters():
                param.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    num_decoder_layers = len(model.detector.transformer.decoder.layers)
    criterion, weight_dict = build_criterion(
        # num_classes=num_classes, 
        num_classes=90,
        num_decoder_layers=num_decoder_layers)
    criterion.to(device)


    num_classes_meta = getattr(model.detector, "num_classes", 91)
    meta = _RunMetadata(
        num_classes=num_classes_meta,
        train_images=len(train_dataset),
        val_images=len(val_dataset),
        backbone_name=args.backbone_name,
        max_size=args.max_size,
    )

    is_main_process = (not args.distributed) or rank == 0
    wandb = _init_wandb(args, meta) if is_main_process else None

    global_step = 0

    if args.distributed:
        model = DDP(model, device_ids=[device] if device.type == "cuda" else None,
                    find_unused_parameters=True
                    )
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        global_step = train_one_epoch(
            model=model,
            criterion=criterion,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_interval=args.log_interval,
            global_step=global_step,
            wandb=wandb,
        )

        if args.distributed:
            torch.cuda.synchronize(device) if device.type == "cuda" else None

        val_loss, metrics = evaluate(
            model, criterion, val_loader, device, distributed=args.distributed, rank=rank, world_size=world_size
        )
        if (not args.distributed) or rank == 0:
            log_data = {"val/loss": val_loss}
            if metrics:
                log_data.update(metrics)
            if wandb is not None:
                wandb.log(log_data, step=global_step)
            else:
                print(log_data)

            ckpt = {
                "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            }
            ckpt_path = args.output_dir / f"checkpoint_{epoch}.pth"
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
