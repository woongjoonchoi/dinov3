"""Run PlainDETR head-only inference using cached backbone activations.

This script mirrors the expected data flow for a Deformable DETR-style model
but skips the backbone forward pass by reading precomputed activations from
disk. It supports distributed execution via ``torchrun`` and can optionally
save the raw decoder outputs for each sample.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
from typing import Any, Callable, Dict, List, Union
from pathlib import Path
from torch import nn
import torch
import torch.distributed as dist
from torchvision.datasets import CocoDetection
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dinov3.eval.detection.config import DetectionHeadConfig
from dinov3.eval.detection.models.backbone import build_backbone
from dinov3.eval.detection.models.detr import PlainDETR, PlainDETRHeadOnly
from dinov3.eval.detection.models.position_encoding import PositionEncoding
from dinov3.eval.detection.models.transformer import build_transformer
from dinov3.eval.detection.util.misc import inverse_sigmoid

from torchvision import transforms

from dinov3.hub.backbones import dinov3_vit7b16, dinov3_vitl16plus
from dinov3_window_base1_1 import DinoVisionTransformerWindowBaseline1_1
from dinov3_window_base1_1.vit import _PatchOnlyWindowBlock
from dinov3_window_base1_3 import LocalGlobalHybridVisionTransformer

def _resolve_weights(value: Optional[str]) -> Optional[BackboneWeights | str]:
    """Convert CLI weight string to hub enum when possible."""

    if value is None:
        return None
    normalized = value.strip().upper()
    try:
        return BackboneWeights[normalized]
    except KeyError:
        return value


def _remap_window_block_keys_to_patch_only(
    model: DinoVisionTransformerWindowBaseline1_1, state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Adapt checkpoints from pre-patch-only window blocks (copied from eval script)."""

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


def _load_checkpoint_state(path: pathlib.Path) -> Dict[str, torch.Tensor]:
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


def _build_backbone_model(args: argparse.Namespace) -> tuple[torch.nn.Module, int]:
    """Create the vision backbone following ``eval_coco_detections`` logic.

    Returns the instantiated backbone and the inferred ``n_windows_sqrt`` value.
    """

    backbone_class_map = {
        "dinov3_vit7b16": dinov3_vit7b16,
        "dinov3_vitl16plus": dinov3_vitl16plus,
        "dinov3_window_base1_1": DinoVisionTransformerWindowBaseline1_1,
        "dinov3_window_base1_3": LocalGlobalHybridVisionTransformer,
        "b1_1": DinoVisionTransformerWindowBaseline1_1,
        "b1_3": LocalGlobalHybridVisionTransformer,
    }
    n_windows_sqrt_map = {
        "dinov3_vit7b16": 3,
        "dinov3_vitl16plus": 2,
    }

    if args.backbone_name not in backbone_class_map:
        raise ValueError(f"Unsupported backbone_name: {args.backbone_name}")

    backbone_class = backbone_class_map[args.backbone_name]
    n_windows_sqrt = args.n_windows_sqrt

    if args.backbone_checkpoint is not None:
        if not args.backbone_checkpoint.exists():
            raise FileNotFoundError(f"Backbone checkpoint not found: {args.backbone_checkpoint}")
        # backbone = backbone_class(pretrained=False, weights=None, check_hash=args.check_hash)

        if args.backbone_name == "b1_1" or args.backbone_name == "b1_3":
            backbone = backbone_class(
                pretrained=False,
                weights=None,
                window_size = 16,
                check_hash=args.check_hash,
            )
        else :
            backbone = backbone_class(
                pretrained=False,
                weights=None,
                check_hash=args.check_hash,
            )
        state_dict = _load_checkpoint_state(args.backbone_checkpoint)
        if isinstance(backbone, DinoVisionTransformerWindowBaseline1_1):
            state_dict = _remap_window_block_keys_to_patch_only(backbone, state_dict)
        backbone.load_state_dict(state_dict, strict=False)
    else:
        weights = _resolve_weights(args.backbone_weights)
        backbone = backbone_class(
            pretrained=args.backbone_pretrained,
            weights=weights,
            window_size=16,
            check_hash=args.check_hash,
        )

    if n_windows_sqrt is None:
        n_windows_sqrt = getattr(backbone, "n_windows_sqrt", n_windows_sqrt_map.get(args.backbone_name, 3))

    backbone.eval()
    return backbone, n_windows_sqrt


class CocoDetectionForEval(CocoDetection):
    def __init__(
        self,
        root: str,
        ann_file: str,
        *,
        transform: transforms.Compose,
        max_size: int | None = None,
    ) -> None:
        super().__init__(root=root, annFile=ann_file)
        self.image_transform = transform
        self.max_size = max_size

    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        image_id = self.ids[index]
        orig_w, orig_h = image.size

        if self.max_size:
            # image = ResizeShortSide(self.max_size)(image)
            image = ResizeAllSides(self.max_size)(image)
        resized_w, resized_h = image.size

        image = self.image_transform(image)
        return image, {
            "image_id": int(image_id),
            "orig_size": (orig_h, orig_w),
            "resized_size": (resized_h, resized_w),
        }

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )

def _resolve_callable(spec: str) -> Callable[..., Any]:
    module_name, function_name = spec.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def _parse_list_of_ints(value: str | None) -> list[int] | None:
    if value is None:
        return None
    if value.strip() == "":
        return None
    return [int(v) for v in value.split(",")]


def _parse_args() -> argparse.Namespace:
    defaults = DetectionHeadConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", required=True, help="Path to COCO dataset root (containing annotations/ and split folders).")
    parser.add_argument("--split", default="val2017", help="Image split to evaluate (e.g., val2017).")
    parser.add_argument(
        "--ann-file",
        default=None,
        help="Optional path to annotations file; defaults to annotations/instances_<split>.json under --coco-root.",
    )
    parser.add_argument("--activation-root", type=pathlib.Path, required=True, help="Directory containing .pt activation files.")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="Detection checkpoint with transformer/head weights.")
    # parser.add_argument(
    #     "--dataset-builder",
    #     type=str,
    #     required=True,
    #     help="Callable spec 'module:function' that returns the base detection dataset.",
    # )
    # parser.add_argument(
    #     "--dataset-builder-kwargs",
    #     type=json.loads,
    #     default={},
    #     help="JSON string of kwargs forwarded to the dataset builder.",
    # )
    parser.add_argument(
        "--backbone-builder",
        type=str,
        default="dinov3.hub.backbones:dinov3_vit7b16",
        help="Callable spec 'module:function' that constructs the backbone model.",
    )
    parser.add_argument(
        "--backbone-builder-kwargs",
        type=json.loads,
        default={},
        help="JSON string of kwargs forwarded to the backbone builder.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per device.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device.")
    parser.add_argument("--distributed", action="store_true", help="Use torch.distributed for multi-GPU inference.")
    parser.add_argument("--pin-memory", action="store_true", help="Pin dataloader memory for faster host->device copies.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=None, help="Optional directory to save raw predictions.")

    parser.add_argument("--num-classes", type=int, default=defaults.num_classes)
    parser.add_argument("--num-feature-levels", type=int, default=defaults.num_feature_levels)
    parser.add_argument("--num-queries-one2one", type=int, default=defaults.num_queries_one2one)
    parser.add_argument("--num-queries-one2many", type=int, default=defaults.num_queries_one2many)
    parser.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--nheads", type=int, default=defaults.nheads)
    parser.add_argument("--dec-layers", type=int, default=defaults.dec_layers)
    parser.add_argument("--dim-feedforward", type=int, default=defaults.dim_feedforward)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--norm-type", type=str, default=defaults.norm_type)
    parser.add_argument("--proposal-feature-levels", type=int, default=defaults.proposal_feature_levels)
    parser.add_argument("--proposal-min-size", type=int, default=defaults.proposal_min_size)
    parser.add_argument("--decoder-type", type=str, default=defaults.decoder_type)
    parser.add_argument("--decoder-use-checkpoint", action="store_true", default=defaults.decoder_use_checkpoint)
    parser.add_argument("--decoder-rpe-hidden-dim", type=int, default=defaults.decoder_rpe_hidden_dim)
    parser.add_argument("--decoder-rpe-type", type=str, default=defaults.decoder_rpe_type)
    parser.add_argument("--look-forward-twice", action="store_true", default=defaults.look_forward_twice)
    parser.add_argument("--k-one2many", type=int, default=defaults.k_one2many)
    parser.add_argument("--lambda-one2many", type=float, default=defaults.lambda_one2many)
    parser.add_argument("--n-windows-sqrt", type=int, default=defaults.n_windows_sqrt)
    parser.add_argument("--proposal-in-stride", type=int, default=defaults.proposal_in_stride if defaults.proposal_in_stride else None)
    parser.add_argument(
        "--proposal-tgt-strides",
        type=_parse_list_of_ints,
        default=defaults.proposal_tgt_strides,
    )
    parser.add_argument("--add-transformer-encoder", action="store_true", default=defaults.add_transformer_encoder)
    parser.add_argument("--no-add-transformer-encoder", action="store_false", dest="add_transformer_encoder")
    parser.add_argument("--num-encoder-layers", type=int, default=defaults.num_encoder_layers)
    parser.add_argument("--backbone-use-layernorm", action="store_true", default=defaults.backbone_use_layernorm)
    parser.add_argument(
        "--position-embedding",
        type=str,
        default=defaults.position_embedding.value if isinstance(defaults.position_embedding, PositionEncoding) else defaults.position_embedding,
    )
    parser.add_argument("--no-look-forward-twice", action="store_false", dest="look_forward_twice")
    parser.add_argument("--no-aux-loss", action="store_false", dest="aux_loss", help="Disable auxiliary decoder losses.")
    parser.add_argument("--no-with-box-refine", action="store_false", dest="with_box_refine", help="Disable box refinement.")
    parser.add_argument("--one-stage", action="store_false", dest="two_stage", help="Disable two-stage transformer proposals.")
    parser.add_argument("--no-mixed-selection", action="store_false", dest="mixed_selection", help="Disable mixed selection.")
    
    parser.set_defaults(
        aux_loss=defaults.aux_loss,
        with_box_refine=defaults.with_box_refine,
        two_stage=defaults.two_stage,
        mixed_selection=defaults.mixed_selection,
        add_transformer_encoder=defaults.add_transformer_encoder,
        look_forward_twice=defaults.look_forward_twice,
    )
    return parser.parse_args()


class PlainDETRHeadOnly(nn.Module):
    """Head-only wrapper that reuses a :class:`PlainDETR` without its backbone."""

    def __init__(self, base: PlainDETR):
        super().__init__()
        self.transformer = base.transformer
        self.input_proj = base.input_proj
        self.query_embed = getattr(base, "query_embed", None)
        self.class_embed = base.class_embed
        self.bbox_embed = base.bbox_embed
        self.num_queries = base.num_queries
        self.num_feature_levels = base.num_feature_levels
        self.aux_loss = base.aux_loss
        self.with_box_refine = base.with_box_refine
        self.two_stage = base.two_stage
        self.num_queries_one2one = base.num_queries_one2one
        self.mixed_selection = base.mixed_selection

    def forward(self, srcs: List[torch.Tensor], masks: List[torch.Tensor], pos: List[torch.Tensor]):
        """
        Args:
            srcs:  List of per-level feature maps with shape ``[B, C_l, H_l, W_l]``.
            masks: List of attention masks with shape ``[B, H_l, W_l]``.
            pos:   List of positional encodings with shape ``[B, D_pos, H_l, W_l]``.
        """

        proj_srcs: List[torch.Tensor] = []
        for layer, src in enumerate(srcs):
            proj_srcs.append(self.input_proj[layer](src))

        query_embeds = None
        if (self.query_embed is not None) and (not self.two_stage or self.mixed_selection):
            query_embeds = self.query_embed.weight[0 : self.num_queries, :]

        self_attn_mask = torch.zeros(
            [
                self.num_queries,
                self.num_queries,
            ],
            dtype=bool,
            device=proj_srcs[0].device,
        )
        self_attn_mask[
            self.num_queries_one2one :,
            0 : self.num_queries_one2one,
        ] = True
        self_attn_mask[
            0 : self.num_queries_one2one,
            self.num_queries_one2one :,
        ] = True

        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
            enc_outputs_delta,
            output_proposals,
            max_shape,
        ) = self.transformer(proj_srcs, masks, pos, query_embeds, self_attn_mask)

        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()

            outputs_classes_one2one.append(outputs_class[:, 0 : self.num_queries_one2one])
            outputs_classes_one2many.append(outputs_class[:, self.num_queries_one2one :])

            outputs_coords_one2one.append(outputs_coord[:, 0 : self.num_queries_one2one])
            outputs_coords_one2many.append(outputs_coord[:, self.num_queries_one2one :])

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)

        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)

        out = {
            "pred_logits": outputs_classes_one2one[-1],
            "pred_boxes": outputs_coords_one2one[-1],
            "pred_logits_one2many": outputs_classes_one2many[-1],
            "pred_boxes_one2many": outputs_coords_one2many[-1],
        }
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_classes_one2one, outputs_coords_one2one)
            out["aux_outputs_one2many"] = self._set_aux_loss(outputs_classes_one2many, outputs_coords_one2many)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


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


def _build_config(args: argparse.Namespace) -> DetectionHeadConfig:
    config = DetectionHeadConfig()
    config.num_classes = args.num_classes
    config.num_feature_levels = args.num_feature_levels
    config.num_queries_one2one = args.num_queries_one2one
    config.num_queries_one2many = args.num_queries_one2many
    config.hidden_dim = args.hidden_dim
    config.nheads = args.nheads
    config.dec_layers = args.dec_layers
    config.dim_feedforward = args.dim_feedforward
    config.dropout = args.dropout
    config.norm_type = args.norm_type
    config.proposal_feature_levels = args.proposal_feature_levels
    config.proposal_min_size = args.proposal_min_size
    config.decoder_type = args.decoder_type
    config.decoder_use_checkpoint = args.decoder_use_checkpoint
    config.decoder_rpe_hidden_dim = args.decoder_rpe_hidden_dim
    config.decoder_rpe_type = args.decoder_rpe_type
    config.look_forward_twice = args.look_forward_twice
    config.k_one2many = args.k_one2many
    config.lambda_one2many = args.lambda_one2many
    config.n_windows_sqrt = args.n_windows_sqrt
    config.proposal_in_stride = args.proposal_in_stride
    config.proposal_tgt_strides = args.proposal_tgt_strides
    config.add_transformer_encoder = args.add_transformer_encoder
    config.num_encoder_layers = args.num_encoder_layers
    config.backbone_use_layernorm = args.backbone_use_layernorm
    config.position_embedding = PositionEncoding[args.position_embedding.upper()]
    config.aux_loss = args.aux_loss
    config.with_box_refine = args.with_box_refine
    config.two_stage = args.two_stage
    config.mixed_selection = args.mixed_selection
    return config


def _split_outputs(outputs: Dict[str, Any], batch_index: int) -> Dict[str, Any]:
    per_item = {
        "pred_logits": outputs["pred_logits"][batch_index].cpu(),
        "pred_boxes": outputs["pred_boxes"][batch_index].cpu(),
    }
    if "pred_logits_one2many" in outputs:
        per_item["pred_logits_one2many"] = outputs["pred_logits_one2many"][batch_index].cpu()
    if "pred_boxes_one2many" in outputs:
        per_item["pred_boxes_one2many"] = outputs["pred_boxes_one2many"][batch_index].cpu()
    if "aux_outputs" in outputs:
        per_item["aux_outputs"] = [
            {"pred_logits": aux["pred_logits"][batch_index].cpu(), "pred_boxes": aux["pred_boxes"][batch_index].cpu()}
            for aux in outputs["aux_outputs"]
        ]
    if "aux_outputs_one2many" in outputs:
        per_item["aux_outputs_one2many"] = [
            {
                "pred_logits": aux["pred_logits"][batch_index].cpu(),
                "pred_boxes": aux["pred_boxes"][batch_index].cpu(),
            }
            for aux in outputs["aux_outputs_one2many"]
        ]
    if "enc_outputs" in outputs:
        per_item["enc_outputs"] = {
            "pred_logits": outputs["enc_outputs"]["pred_logits"][batch_index].cpu(),
            "pred_boxes": outputs["enc_outputs"]["pred_boxes"][batch_index].cpu(),
        }
    return per_item


@torch.inference_mode()
def _run_inference(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    output_dir: pathlib.Path | None,
    rank: int,
) -> None:
    if output_dir is not None and rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    for srcs, masks, pos, targets, metas in tqdm(loader):
        srcs = [s.to(device, non_blocking=True) for s in srcs]
        masks = [m.to(device, non_blocking=True) for m in masks]
        pos = [p.to(device, non_blocking=True) for p in pos]

        outputs = model(srcs, masks, pos)

        if output_dir is None or rank != 0:
            continue

        batch_size = len(targets)
        for idx in range(batch_size):
            target = targets[idx]
            meta = metas[idx]
            file_name = meta.get("file_name", target.get("file_name", f"sample_{idx}"))
            file_path = pathlib.Path(file_name)
            save_path = (output_dir / file_path).with_suffix(".pt")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(_split_outputs(outputs, idx), save_path)


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

    config = _build_config(args)

    backbone_builder = _resolve_callable(args.backbone_builder)
    backbone_model = backbone_builder(**args.backbone_builder_kwargs)

    dataset_builder = _resolve_callable(args.dataset_builder)
    base_dataset = dataset_builder(**args.dataset_builder_kwargs)
    
    coco_root = Path(args.coco_root)
    ann_file = (
        Path(args.ann_file)
        if args.ann_file is not None
        else coco_root / "annotations" / f"instances_{args.split}.json"
    )
    image_root = coco_root / args.split

    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    if not args.backbone_checkpoint.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {args.backbone_checkpoint}")
    if not args.detector_checkpoint.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {args.detector_checkpoint}")

    base_dataset = CocoDetectionForEval(
        root=str(image_root), ann_file=str(ann_file), transform=build_transform()
    )
    
    model = build_head_only_model(config, backbone_model, args.checkpoint, device)
    if args.distributed:
        model = DDP(model, device_ids=[device] if device.type == "cuda" else None)

    activation_dataset = ActivationDetDataset(base_dataset, args.activation_root)
    sampler = None
    if args.distributed:
        sampler = DistributedSampler(activation_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    loader = build_activation_dataloader(
        base_dataset,
        args.activation_root,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        pin_memory=args.pin_memory,
        sampler=sampler,
        dataset=activation_dataset,
    )

    _run_inference(model, loader, device, output_dir=args.output_dir, rank=rank)

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

