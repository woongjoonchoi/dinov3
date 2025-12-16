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
from typing import Dict, Iterable

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
from dinov3.models import init_fp8
from dinov3_window_base1_1 import DinoVisionTransformerWindowBaseline1_1
from dinov3_window_base1_1.vit import _PatchOnlyWindowBlock
from dinov3_window_base1_3 import LocalGlobalHybridVisionTransformer

logger = logging.getLogger("dinov3")


def _load_state_dict(path: Path | str) -> Dict[str, torch.Tensor]:
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


def _remap_window_block_keys_to_patch_only(
    model: DinoVisionTransformerWindowBaseline1_1, state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Adapt checkpoints from pre-patch-only window blocks."""

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


def _build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    """Build a backbone based on ``args.backbone_name`` with optional FP8 conversion."""

    backbone_name = args.backbone_name
    apply_fp8 = bool(getattr(args, "fp8_enabled", False))
    args.fp8_enabled = apply_fp8
    args.fp8_filter = getattr(args, "fp8_filter", None)

    if backbone_name == "b1_1":
        backbone = DinoVisionTransformerWindowBaseline1_1(**args.model_kwargs)
        state_dict = _remap_window_block_keys_to_patch_only(
            backbone, _load_state_dict(args.backbone_checkpoint)
        )
    elif backbone_name == "b1_3":
        backbone = LocalGlobalHybridVisionTransformer(**args.model_kwargs)
        state_dict = _load_state_dict(args.backbone_checkpoint)
    else:
        raise ValueError(f"Unsupported backbone_name: {backbone_name}")

    backbone.load_state_dict(state_dict, strict=True)

    # Keep the model on CPU while reshaping modules for FP8, then move to the target device for DDP.
    if apply_fp8:
        backbone = init_fp8(backbone, args)

    backbone.to(device)
    return backbone


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
    # print(f"eval_stride :{eval_stride}")
    # print(f"eval_res :{eval_res}")
    # exit()
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
        # print(f"batch_img shape :{batch_img[0].shape}")
        # per_sample_views = _split_batch_views(batch_img, batch_size=gt_batch.shape[0])
        # print(f"per_sample_viess shape : {per_sample_views[0][0].shape}")
        # exit()
        inference_model = model.module if isinstance(model, DDP) else model
        # for sample_views, gt_sample in zip(per_sample_views, gt_batch):
        #     sample_views = [img.to(device).to(dtype=autocast_dtype) for img in sample_views]
        #     aggregated_preds = torch.zeros(
        #         1, num_classes, gt_sample.shape[-2], gt_sample.shape[-1], device=device
        #     )

        #     for img_idx, img in enumerate(sample_views):
        #         aggregated_preds += make_inference(
        #             img,
        #             inference_model,
        #             inference_mode="slide",
        #             decoder_head_type=decoder_head_type,
        #             rescale_to=gt_sample.shape[-2:],
        #             n_output_channels=num_classes,
        #             crop_size=(eval_res, eval_res),
        #             # stride=(eval_stride, eval_stride),
        #             stride=(eval_res, eval_res),
        #             apply_horizontal_flip=(img_idx and img_idx >= len(sample_views) / 2),
        #             output_activation=lambda x: torch.nn.functional.softmax(x, dim=1),
        #         )

        #     aggregated_preds = (aggregated_preds / len(sample_views)).argmax(dim=1, keepdim=True)
        #     intersect_and_union = calculate_intersect_and_union(
        #         aggregated_preds[0],
        #         gt_sample,
        #         num_classes=num_classes,
        #         reduce_zero_label=True,
        #     )
        #     intersections.append(intersect_and_union)

        batch_img = [img.to(device).to(dtype=autocast_dtype) for img in batch_img]
        gt = gt.to(device)[0]
        # print(f"batchh img shape :{batch_img[0].shape}")
        # exit()
        aggregated_preds = torch.zeros(1, num_classes, gt.shape[-2], gt.shape[-1])
        # print(f"eval_res :{eval_res}")
        # print(f"eval_res :{eval_res}")
        for img_idx, img in enumerate(batch_img):
            aggregated_preds += make_inference(
                img,
                inference_model,
                inference_mode="slide",
                decoder_head_type=decoder_head_type,
                rescale_to=gt.shape[-2:],
                n_output_channels=num_classes,
                # crop_size=(eval_res, eval_res),
                # stride=(eval_res, eval_res),
                crop_size=eval_res,
                stride=eval_res,
                apply_horizontal_flip=(img_idx and img_idx >= len(batch_img) / 2),
                # output_activation=partial(torch.nn.functional.softmax, dim=1),
                output_activation=lambda x: torch.nn.functional.softmax(x, dim=1),
            )
        aggregated_preds = (aggregated_preds / len(batch_img)).argmax(dim=1, keepdim=True).to(device)
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

    if any((args.model_config, args.backbone_weights, args.backbone_hub)):
        overrides["model"] = {
            "config_file": str(args.model_config) if args.model_config else None,
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


def _build_segmentation_model(
    config: SegmentationConfig,
    device: torch.device,
    use_ddp: bool,
    custom_backbone_args: argparse.Namespace | None = None,
):
    use_custom_backbone = custom_backbone_args is not None and custom_backbone_args.backbone_name

    if config.load_from == "dinov3_vit7b16_ms" and not use_custom_backbone:
        logger.info("Loading dinov3_vit7b16_ms segmentation head via torch hub.")
        segmentation_model = dinov3_vit7b16_ms(
            autocast_dtype=config.model_dtype.autocast_dtype, check_hash=True
        )
    else:
        if use_custom_backbone:
            if not custom_backbone_args.backbone_checkpoint:
                raise ValueError(
                    "--backbone-checkpoint is required when providing --backbone-name for custom loading."
                )
            backbone = _build_model(custom_backbone_args, device)
        else:
            if config.model is None:
                raise ValueError(
                    "Backbone configuration is required when load_from points to a checkpoint. "
                    "Please provide --model-config/--backbone-weights or --backbone-hub."
                )
            if config.model.config_file is None and config.model.dino_hub is None:
                raise ValueError(
                    "When loading a local segmentation checkpoint, please pass --model-config "
                    "to specify the backbone config file or --backbone-hub for a hub model."
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

        state_dict = _load_segmentation_head_state_dict(config.load_from)
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
    parser.add_argument(
        "--backbone-name",
        type=str,
        default=None,
        help="Custom backbone name for local loading (e.g., b1_1, b1_3).",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for the custom backbone used with --backbone-name.",
    )
    parser.add_argument(
        "--fp8-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable FP8 conversion for the custom backbone when supported.",
    )
    parser.add_argument(
        "--model-kwargs",
        type=str,
        default=None,
        help="YAML/JSON string of kwargs for the custom backbone constructor.",
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
    args = parser.parse_args()

    if args.model_kwargs:
        args.model_kwargs = OmegaConf.to_object(OmegaConf.create(args.model_kwargs))
    else:
        args.model_kwargs = {}

    return args


def _load_segmentation_head_state_dict(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unexpected checkpoint structure in {path}")

    candidate_keys = ("model", "state_dict", "model_state_dict", "decoder")
    state_dict = None
    for key in candidate_keys:
        maybe_state = checkpoint.get(key)
        if isinstance(maybe_state, dict):
            state_dict = maybe_state
            break

    if state_dict is None:
        # If it already looks like a state dict, accept it.
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            state_dict = checkpoint
        else:
            available_keys = ", ".join(sorted(checkpoint.keys()))
            raise RuntimeError(
                f"No state dict found in checkpoint {path}; available keys: {available_keys}"
            )

    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    return state_dict


def main() -> int:
    args = parse_args()
    config = _build_config(args)
    config.eval.crop_size=(args.eval_size,args.eval_size)
    # print(config.eval)
    # print(f"config eval crop :{config.eval.crop_size}")
    # exit()
    with job_context(output_dir=str(args.output_dir), distributed_enabled=args.distributed):
        device = torch.device(
            f"cuda:{distributed.get_rank()}" if torch.cuda.is_available() else "cpu"
        )
        segmentation_model = _build_segmentation_model(
            config, device=device, use_ddp=args.distributed, custom_backbone_args=args
        )
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
