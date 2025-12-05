"""Validate ADE20K segmentation with DDP using DINOv3 utilities.

This script builds the ADE20K validation dataloader and segmentation model from
``dinov3.eval.segmentation`` helpers, runs distributed evaluation, logs
intermediate metrics every ``log-interval`` iterations, and prints the final
results.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import torch
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

import dinov3.distributed as distributed
from dinov3.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader, make_dataset
from dinov3.eval.segmentation.config import SegmentationConfig
from dinov3.eval.segmentation.inference import make_inference
from dinov3.eval.segmentation.metrics import (
    calculate_intersect_and_union,
    calculate_segmentation_metrics,
)
from dinov3.eval.segmentation.models import build_segmentation_decoder
from dinov3.eval.segmentation.transforms import make_segmentation_eval_transforms
from dinov3.eval.setup import load_model_and_context
from dinov3.hub.segmentors import dinov3_vit7b16_ms
from dinov3.logging import MetricLogger
from dinov3.run.init import job_context

logger = logging.getLogger("dinov3")


@torch.inference_mode()
def _summarize_metrics(metric_values: torch.Tensor) -> dict[str, float]:
    """Compute segmentation metrics as percentages."""

    if distributed.is_enabled():
        metric_values = torch.cat(distributed.gather_all_tensors(metric_values))
    if metric_values.numel() == 0:
        return {}
    aggregated = calculate_segmentation_metrics(metric_values, metrics=["mIoU", "dice", "fscore"])
    return {name: round(value.cpu().item() * 100, 2) for name, value in aggregated.items()}


@torch.inference_mode()
def _split_batch_views(batch_img, batch_size: int) -> list[list[torch.Tensor]]:
    """Convert dataloader batch output into per-sample view lists."""

    if torch.is_tensor(batch_img):
        if batch_img.dim() == 4:
            return [[sample] for sample in batch_img]
        if batch_img.dim() == 3:
            return [[batch_img]]

    if isinstance(batch_img, (list, tuple)):
        if not batch_img:
            return [[] for _ in range(batch_size)]

        first_view = batch_img[0]
        if torch.is_tensor(first_view):
            # Default collate stacks the batch dimension for each view.
            if first_view.dim() >= 4:
                return [[view[sample_idx] for view in batch_img] for sample_idx in range(batch_size)]
            # If views are not stacked, we assume a single-sample batch.
            if batch_size == 1:
                return [list(batch_img)]

        if isinstance(first_view, (list, tuple)) and len(batch_img) == batch_size:
            return [list(views) for views in batch_img]

    raise TypeError(f"Unsupported batch format for validation: {type(batch_img)}")


def _run_validation(
    model: torch.nn.Module | DDP,
    dataloader: Iterable,
    *,
    num_classes: int,
    eval_res: int,
    eval_stride: int,
    decoder_head_type: str,
    autocast_dtype: torch.dtype,
    log_interval: int,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    metric_logger = MetricLogger(delimiter="  ")
    intersections: list[torch.Tensor] = []

    for iteration, (batch_img, (_, gt)) in enumerate(
        metric_logger.log_every(dataloader, 10, header="Validation: "), start=1
    ):
        if torch.is_tensor(gt):
            gt_batch = gt.to(device)
            if gt_batch.dim() == 2:
                gt_batch = gt_batch.unsqueeze(0)
        elif isinstance(gt, (list, tuple)):
            gt_batch = torch.stack([torch.as_tensor(target) for target in gt]).to(device)
        else:
            raise TypeError(f"Unsupported ground truth format: {type(gt)}")

        per_sample_views = _split_batch_views(batch_img, batch_size=gt_batch.shape[0])

        inference_model = model.module if isinstance(model, DDP) else model
        for sample_views, gt_sample in zip(per_sample_views, gt_batch):
            sample_views = [img.to(device).to(dtype=autocast_dtype) for img in sample_views]
            aggregated_preds = torch.zeros(
                1, num_classes, gt_sample.shape[-2], gt_sample.shape[-1], device=device
            )

            for img_idx, img in enumerate(sample_views):
                aggregated_preds += make_inference(
                    img,
                    inference_model,
                    inference_mode="slide",
                    decoder_head_type=decoder_head_type,
                    rescale_to=gt_sample.shape[-2:],
                    n_output_channels=num_classes,
                    crop_size=(eval_res, eval_res),
                    stride=(eval_stride, eval_stride),
                    apply_horizontal_flip=(img_idx and img_idx >= len(sample_views) / 2),
                    output_activation=lambda x: torch.nn.functional.softmax(x, dim=1),
                )

            aggregated_preds = (aggregated_preds / len(sample_views)).argmax(dim=1, keepdim=True)
            intersect_and_union = calculate_intersect_and_union(
                aggregated_preds[0],
                gt_sample,
                num_classes=num_classes,
                reduce_zero_label=True,
            )
            intersections.append(intersect_and_union)

        if iteration % log_interval == 0:
            metrics = _summarize_metrics(torch.stack(intersections))
            if distributed.is_main_process():
                logger.info("Intermediate metrics after %d iterations: %s", iteration, metrics)

        del aggregated_preds, intersect_and_union, gt

    metrics = _summarize_metrics(torch.stack(intersections))
    if distributed.is_main_process():
        logger.info("Final validation metrics: %s", metrics)
    return metrics


def _build_config(args: argparse.Namespace) -> SegmentationConfig:
    base_cfg = OmegaConf.load(args.config)
    structured_cfg = OmegaConf.structured(SegmentationConfig)
    overrides = {
        "datasets": {"root": str(args.dataset_root), "val": "ADE20K:split=VAL"},
        "load_from": args.load_from,
        "output_dir": str(args.output_dir),
        "num_workers": args.num_workers,
    }

    if any((args.model_config, args.backbone_weights, args.backbone_hub)):
        overrides["model"] = {
            "config_file": args.model_config,
            "pretrained_weights": args.backbone_weights,
            "dino_hub": args.backbone_hub,
        }
    if args.eval_size is not None:
        overrides.update(
            {
                "transforms": {"eval": {"img_size": args.eval_size}},
                "eval": {"crop_size": args.eval_size},
            }
        )
    overrides = OmegaConf.create(overrides)
    merged = OmegaConf.merge(structured_cfg, base_cfg, overrides)
    return OmegaConf.to_object(merged)


def _build_model(config: SegmentationConfig, device: torch.device, use_ddp: bool):
    if config.load_from == "dinov3_vit7b16_ms":
        logger.info("Loading dinov3_vit7b16_ms segmentation head via torch hub.")
        segmentation_model = dinov3_vit7b16_ms(
            autocast_dtype=config.model_dtype.autocast_dtype, check_hash=True
        )
    else:
        if config.model is None:
            raise ValueError(
                "Backbone configuration is required when load_from points to a checkpoint. "
                "Please provide --model-config/--backbone-weights or --backbone-hub."
            )
        backbone, _ = load_model_and_context(config.model, output_dir=config.output_dir)
        segmentation_model = build_segmentation_decoder(
            backbone,
            config.decoder_head.backbone_out_layers,
            config.decoder_head.type,
            hidden_dim=config.decoder_head.hidden_dim,
            num_classes=config.decoder_head.num_classes,
            autocast_dtype=config.model_dtype.autocast_dtype,
            dropout=config.decoder_head.dropout,
        )
        state_dict = torch.load(config.load_from, map_location="cpu")["model"]
        segmentation_model.load_state_dict(state_dict, strict=False)
    segmentation_model = segmentation_model.to(device)
    if use_ddp:
        return DDP(segmentation_model, device_ids=[device])
    return segmentation_model


def _build_dataloader(config: SegmentationConfig, *, pin_memory: bool, batch_size: int):
    eval_res = config.eval.crop_size
    transforms = make_segmentation_eval_transforms(
        img_size=eval_res,
        inference_mode="slide",
        use_tta=config.eval.use_tta,
        tta_ratios=config.transforms.eval.tta_ratios,
        mean=config.transforms.mean,
        std=config.transforms.std,
    )
    dataset = DatasetWithEnumeratedTargets(
        make_dataset(dataset_str=f"{config.datasets.val}:root={config.datasets.root}", transforms=transforms)
    )
    sampler_type = SamplerType.DISTRIBUTED if distributed.is_enabled() else None
    return make_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=config.num_workers,
        sampler_type=sampler_type,
        drop_last=False,
        shuffle=False,
        persistent_workers=True,
        pin_memory=pin_memory,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to the ADE20K dataset root.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("dinov3/eval/segmentation/configs/config-ade20k-m2f-inference.yaml"),
        help="Path to the segmentation config file.",
    )
    parser.add_argument(
        "--load-from",
        type=str,
        default="dinov3_vit7b16_ms",
        help="Checkpoint path or torch hub name for the segmentation decoder head.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Backbone config file used when loading a local checkpoint.",
    )
    parser.add_argument(
        "--backbone-weights",
        type=str,
        default=None,
        help="Backbone checkpoint used when loading a local segmentation head.",
    )
    parser.add_argument(
        "--backbone-hub",
        type=str,
        default=None,
        help="Backbone identifier to load from torch.hub instead of a config file.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ade20k_val"), help="Directory for logs.")
    parser.add_argument("--num-workers", type=int, default=6, help="Number of dataloader workers per process.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for validation.")
    parser.add_argument("--eval-size", type=int, default=None, help="Resize validation images to this square resolution.")
    parser.add_argument("--log-interval", type=int, default=10, help="Iterations between intermediate metric logs.")
    parser.add_argument(
        "--distributed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap the model with DistributedDataParallel and enable distributed setup.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable pin_memory for the validation dataloader.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _build_config(args)

    with job_context(output_dir=str(args.output_dir), distributed_enabled=args.distributed):
        device = torch.device(
            f"cuda:{distributed.get_rank()}" if torch.cuda.is_available() else "cpu"
        )
        segmentation_model = _build_model(config, device=device, use_ddp=args.distributed)
        dataloader = _build_dataloader(config, pin_memory=args.pin_memory, batch_size=args.batch_size)
        _ = _run_validation(
            segmentation_model,
            dataloader,
            num_classes=config.decoder_head.num_classes,
            eval_res=config.eval.crop_size,
            eval_stride=config.eval.stride,
            decoder_head_type=config.decoder_head.type,
            autocast_dtype=config.model_dtype.autocast_dtype,
            log_interval=args.log_interval,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
