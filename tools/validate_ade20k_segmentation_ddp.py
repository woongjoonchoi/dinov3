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
        batch_img = [img.to(device).to(dtype=autocast_dtype) for img in batch_img]
        gt = gt.to(device)[0]
        aggregated_preds = torch.zeros(1, num_classes, gt.shape[-2], gt.shape[-1], device=device)

        inference_model = model.module if isinstance(model, DDP) else model
        for img_idx, img in enumerate(batch_img):
            aggregated_preds += make_inference(
                img,
                inference_model,
                inference_mode="slide",
                decoder_head_type=decoder_head_type,
                rescale_to=gt.shape[-2:],
                n_output_channels=num_classes,
                crop_size=(eval_res, eval_res),
                stride=(eval_stride, eval_stride),
                apply_horizontal_flip=(img_idx and img_idx >= len(batch_img) / 2),
                output_activation=lambda x: torch.nn.functional.softmax(x, dim=1),
            )

        aggregated_preds = (aggregated_preds / len(batch_img)).argmax(dim=1, keepdim=True)
        intersect_and_union = calculate_intersect_and_union(
            aggregated_preds[0],
            gt,
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
        assert config.model is not None, "Backbone configuration is required when load_from is a checkpoint path."
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


def _build_dataloader(config: SegmentationConfig, pin_memory: bool):
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
        batch_size=1,
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ade20k_val"), help="Directory for logs.")
    parser.add_argument("--num-workers", type=int, default=6, help="Number of dataloader workers per process.")
    parser.add_argument("--eval-size", type=int, default=None, help="Resize validation images to this square resolution.")
    parser.add_argument("--log-interval", type=int, default=10, help="Iterations between intermediate metric logs.")
    parser.add_argument(
        "--ddp",
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

    with job_context(output_dir=str(args.output_dir), distributed_enabled=args.ddp):
        device = torch.device(
            f"cuda:{distributed.get_rank()}" if torch.cuda.is_available() else "cpu"
        )
        segmentation_model = _build_model(config, device=device, use_ddp=args.ddp)
        dataloader = _build_dataloader(config, pin_memory=args.pin_memory)
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
