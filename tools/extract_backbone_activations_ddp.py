"""Extract DINOv3 backbone activations using the COCO eval backbone loader.

This script mirrors the backbone construction logic from
``tools/eval_coco_detections.py`` so that each baseline backbone is loaded in
the same way (including checkpoint remapping for window baselines). It runs the
backbone with position encoding on a COCO split and saves one ``.pt`` file per
image containing the per-level activations and positional encodings.
"""

from __future__ import annotations

import argparse
import os
import pathlib
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode

from dinov3.eval.detection.models.backbone import BackboneWithPositionEncoding, build_backbone
from dinov3.eval.detection.models.position_encoding import PositionEncoding
from dinov3.eval.detection.util.misc import NestedTensor, nested_tensor_from_tensor_list
from dinov3.hub.backbones import Weights as BackboneWeights
from dinov3.hub.backbones import dinov3_vit7b16, dinov3_vitl16plus
from dinov3_window_base1_1 import DinoVisionTransformerWindowBaseline1_1
from dinov3_window_base1_1.vit import _PatchOnlyWindowBlock
from dinov3_window_base1_3 import LocalGlobalHybridVisionTransformer
from dinov3.eval.detection.models.detr import PostProcess, build_model
from dinov3.eval.detection.config import DetectionHeadConfig

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


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ResizeShortSide:
    def __init__(self, target_size: int):
        self.target_size = target_size

    def __call__(self, image):
        width, height = image.size  # (W, H)
        short = min(width, height)
        if short == self.target_size:
            return image
        scale = self.target_size / short
        new_w = int(round(width * scale))
        new_h = int(round(height * scale))
        return F.resize(image, (new_h, new_w), interpolation=InterpolationMode.BICUBIC)


class ResizeAllSides:
    def __init__(self, target_size: int):
        self.target_size = target_size

    def __call__(self, image):
        width, height = image.size  # (W, H)
        if width == self.target_size and height == self.target_size:
            return image
        # F.resize expects (H, W)
        return F.resize(
            image,
            (self.target_size, self.target_size),
            interpolation=InterpolationMode.BICUBIC,
        )

def _make_transform(max_size: Optional[int]) -> transforms.Compose:
    resize: List[object] = []
    if max_size is not None:
        # resize = [ResizeShortSide(max_size)]
        resize = [ResizeAllSides(max_size)]
    return transforms.Compose(
        [
            *resize,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


class CocoDetectionForActivations(CocoDetection):
    def __init__(
        self,
        root: str,
        ann_file: str,
        *,
        transform: transforms.Compose,
    ) -> None:
        super().__init__(root=root, annFile=ann_file)
        self.image_transform = transform

    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        image_id = self.ids[index]
        info = self.coco.loadImgs(image_id)[0]
        file_name = info.get("file_name", str(image_id))
        orig_w, orig_h = image.size

        image = self.image_transform(image)
        return image, {
            "image_id": int(image_id),
            "orig_size": (orig_h, orig_w),
            "file_name": file_name,
        }


def _collate_coco(batch: Iterable[Tuple[torch.Tensor, dict]]):
    images, metas = zip(*batch)
    nested = nested_tensor_from_tensor_list(list(images))
    return nested, list(metas)


def _build_loader(
    coco_root: pathlib.Path,
    split: str,
    ann_file: pathlib.Path,
    max_size: Optional[int],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    distributed: bool,
    rank: int,
    world_size: int,
    prefetch_factor : int = 1
) -> DataLoader:
    dataset = CocoDetectionForActivations(
        root=str(coco_root / split), ann_file=str(ann_file), transform=_make_transform(max_size)
    )
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
        collate_fn=_collate_coco,
        prefetch_factor=prefetch_factor
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=pathlib.Path, required=True, help="Path to COCO dataset root.")
    parser.add_argument("--split", type=str, default="val2017", help="Dataset split to use (e.g., val2017).")
    parser.add_argument(
        "--ann-file",
        type=pathlib.Path,
        default=None,
        help="Optional annotation file (defaults to annotations/instances_<split>.json).",
    )
    parser.add_argument("--max-size", type=int, default=None, help="Resize short side to this size before normalization.")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True, help="Directory to write activation files.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per device.")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader worker count.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    parser.add_argument("--distributed", action="store_true", help="Use torch.distributed for multi-GPU extraction.")
    parser.add_argument("--pin-memory", action="store_true", help="Pin dataloader memory for faster host->device copies.")
    parser.add_argument("--prefetch_factor", type=int, default=1 ,help="Pin dataloader memory for faster host->device copies.")
    parser.add_argument("--backbone-name", type=str, default="dinov3_vit7b16", help="Backbone name used in eval script.")
    parser.add_argument("--backbone-checkpoint", type=pathlib.Path, default=None, help="Local backbone checkpoint path.")
    parser.add_argument("--detector-checkpoint", type=pathlib.Path, default=None, help="Local backbone checkpoint path.")
    parser.add_argument("--backbone-pretrained", action="store_true", help="Load pretrained weights when no checkpoint.")
    parser.add_argument("--backbone-weights", type=str, default=None, help="Backbone weights enum name or path.")
    parser.add_argument("--check-hash", action="store_true", help="Enable hub hash checking when loading weights.")
    parser.add_argument("--n-windows-sqrt", type=int, default=None, help="Override n_windows_sqrt (defaults to backbone logic).")
    parser.add_argument("--save-fp16", action="store_true", help="Save activations in float16 to reduce disk usage.")
    parser.add_argument("--layers-to-use", type=int, default=1, help="Number of backbone blocks to concatenate.")
    parser.add_argument("--backbone-use-layernorm", action="store_true", help="Apply LayerNorm2D to backbone outputs.")
    parser.add_argument(
        "--blocks-to-train",
        type=int,
        nargs="*",
        default=None,
        help="Backbone block indices to train (others frozen).",
    )
    parser.add_argument(
        "--position-embedding",
        type=str,
        choices=[p.value for p in PositionEncoding],
        default=PositionEncoding.SINE.value,
        help="Type of positional encoding to build.",
    )
    parser.add_argument("--num-feature-levels", type=int, default=1, help="Number of feature levels for position encoding.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Transformer hidden dimension for position encoding.")
    parser.add_argument("--start-index", type=int, default=0, help="Starting global index for saved files.")
    return parser.parse_args()


def _build_backbone_args(args: argparse.Namespace, n_windows_sqrt: int) -> SimpleNamespace:
    return SimpleNamespace(
        blocks_to_train=args.blocks_to_train,
        layers_to_use=args.layers_to_use,
        backbone_use_layernorm=args.backbone_use_layernorm,
        n_windows_sqrt=n_windows_sqrt,
        position_embedding=PositionEncoding(args.position_embedding),
        num_feature_levels=args.num_feature_levels,
        hidden_dim=args.hidden_dim,
    )


def _move_nested_to_device(sample: NestedTensor, device: torch.device) -> NestedTensor:
    return sample.to(device, non_blocking=True)


def build_dinov3_detector_custom(
    *,
    # 1) 이미 만들어진 backbone 모듈을 직접 넘기고 싶을 때
    backbone: Optional[nn.Module] = None,
    # 2) 아니면 기존 dinov3 backbone 이름으로 생성
    backbone_name: str = "dinov3_vit7b16",
    backbone_pretrained: bool = True,
    backbone_weights: BackboneWeights | str = BackboneWeights.LVD1689M,
    # 3) 로컬 backbone ckpt 경로 (주어지면 이걸 우선 사용)
    backbone_ckpt_path: Optional[str] = None,
    # 4) detector head ckpt 로컬 경로 (옵션)
    detector_ckpt_path: Optional[str] = None,
    num_classes: int = 91,   # COCO 기준 91
    check_hash: bool = False,
    **kwargs,
) -> nn.Module:
    """
    - dinov3의 _make_dinov3_detector를 기반으로,
      - backbone은 로컬 ckpt 또는 공식 weights 둘 다 지원
      - detector head는 로컬 ckpt에서 로딩
    - .to(device), .eval()은 여기서 호출하지 않고, 호출하는 쪽에서 처리.
    """

    # -------------------------------
    # 1. Detection head config 생성
    # -------------------------------
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
        num_classes=num_classes,
        aux_loss=True,
        topk=1500,
        hidden_dim=768,
        nheads=8,
    )
    config = DetectionHeadConfig(**detection_kwargs)

    # -------------------------------
    # 2. backbone 생성 또는 주입
    # -------------------------------
    n_windows_sqrt_map = {
        "dinov3_vit7b16": 3,
        "dinov3_vitl16plus": 2,
    }
    if backbone is None:
        backbone_class_map = {
            "dinov3_vit7b16": dinov3_vit7b16,
            "dinov3_vitl16plus": dinov3_vitl16plus,
            "dinov3_window_base1_1": DinoVisionTransformerWindowBaseline1_1,
            "dinov3_window_base1_3": LocalGlobalHybridVisionTransformer,
            "b1_1": DinoVisionTransformerWindowBaseline1_1,
            "b1_3": LocalGlobalHybridVisionTransformer,
        }

        if backbone_name not in backbone_class_map:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        backbone_class = backbone_class_map[backbone_name]

        # (A) 로컬 backbone ckpt를 사용하고 싶을 때
        if backbone_ckpt_path is not None:
            if not os.path.isfile(backbone_ckpt_path):
                raise FileNotFoundError(
                    f"backbone_ckpt_path not found: {backbone_ckpt_path}"
                )

            # 구조만 만들고
            if backbone_name == "b1_1" or backbone_name == "b1_3":
                backbone = backbone_class(
                    pretrained=False,
                    weights=None,
                    window_size = 16,
                    check_hash=check_hash,
                )
            else :
                backbone = backbone_class(
                    pretrained=False,
                    weights=None,
                    check_hash=check_hash,
                )

            # 로컬 ckpt에서 weight 로딩
            b_ckpt = torch.load(backbone_ckpt_path, map_location="cpu")
            b_state = b_ckpt["model"] if isinstance(b_ckpt, dict) and "model" in b_ckpt else b_ckpt
            if isinstance(backbone, DinoVisionTransformerWindowBaseline1_1):
                b_state = _remap_window_block_keys_to_patch_only(backbone, b_state)
            backbone.load_state_dict(b_state, strict=False)

        # (B) 공식 dinov3 BackboneWeights / URL 방식을 그대로 쓰고 싶을 때
        else:
            backbone = backbone_class(
                pretrained=backbone_pretrained,
                weights=backbone_weights,
                check_hash=check_hash,
            )
    else:
        # custom backbone이면 n_windows_sqrt는 속성에서 가져오고,
        # 없으면 기본값 3
        n_windows_sqrt = getattr(backbone, "n_windows_sqrt", 3)

    if backbone is not None:
        n_windows_sqrt = getattr(backbone, "n_windows_sqrt", n_windows_sqrt_map.get(backbone_name, 3))

    # 원래 코드도 backbone은 eval()로 고정
    backbone.eval()

    config.n_windows_sqrt = n_windows_sqrt
    config.proposal_in_stride = backbone.patch_size
    config.proposal_tgt_strides = [
        int(m * backbone.patch_size) for m in (0.5, 1, 2, 4)
    ]

    if config.layers_to_use is None:
        # e.g. [2, 5, 8, 11] for a backbone with 12 blocks
        config.layers_to_use = [m * backbone.n_blocks // 4 - 1 for m in range(1, 5)]

    # -------------------------------
    # 3. detector head 생성
    # -------------------------------
    detector = build_model(backbone, config)

    model = detector.backbone

    # 여기서는 .to(device), .eval() 호출 안 함
    return model

@torch.inference_mode()
def _extract_and_save(
    backbone: BackboneWithPositionEncoding | DDP,
    loader: DataLoader,
    device: torch.device,
    output_root: pathlib.Path,
    start_index: int,
    save_fp16: bool,
    rank: int,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    global_index = start_index
    for batch_idx, (samples, metas) in enumerate(loader):
        nested: NestedTensor = samples
        nested = _move_nested_to_device(nested, device)

        if isinstance(backbone, DDP):
            out, pos = backbone.module(nested)
        else:
            out, pos = backbone(nested)

        batch_size = out[0].tensors.shape[0]
        # print(f"out length : {len(out)}")
        # print(f"out tensro shaep :{out[0].tensors.shape}")
        # exit()
        for in_batch_idx in range(batch_size):
            single_out_tensors = [lvl.tensors[in_batch_idx].detach().cpu().clone() for lvl in out]
            single_out_masks = [lvl.mask[in_batch_idx].detach().cpu().clone() for lvl in out]
            single_pos = [p[in_batch_idx].detach().cpu().clone() for p in pos]

            if save_fp16:
                single_out_tensors = [t.half() for t in single_out_tensors]
                single_pos = [p.half() for p in single_pos]

            meta = metas[in_batch_idx]
            image_id = meta.get("image_id")
            file_name = meta.get("file_name")
            filename = f"{global_index:08d}.pt"

            rel_path = pathlib.Path(file_name).with_suffix(".pt")
            # output_path = output_root / filename

            output_path = output_root / rel_path

            activations = {
                "out_tensors": single_out_tensors,
                "out_masks": single_out_masks,
                "pos": single_pos,
                "meta": {
                    "global_index": global_index,
                    "batch_index": batch_idx,
                    "in_batch_index": in_batch_idx,
                    "image_id": image_id,
                    "file_name": file_name,
                    "orig_size": meta.get("orig_size"),
                    "rank": rank,
                },
            }

            torch.save(activations, output_path)
            print(f"[{global_index:08d}] saved activations to {output_path}")

            global_index += 1


def main() -> None:
    args = _parse_args()

    if args.distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = torch.device(args.device)
        rank = 0
        world_size = 1

    print(f"Running on rank {rank} / {world_size} with device {device}")
    model = build_dinov3_detector_custom(
        # backbone_name="dinov3_vit7b16",
        backbone_name=args.backbone_name,
        backbone_pretrained=False,
        backbone_weights=None,
        backbone_ckpt_path=args.backbone_checkpoint,
        detector_ckpt_path=args.detector_checkpoint,
        num_classes=91,
    )
    # backbone_model, n_windows_sqrt = _build_backbone_model(args)
    # backbone_args = _build_backbone_args(args, n_windows_sqrt)
    # backbone_with_pe = build_backbone(backbone_model, backbone_args)

    backbone_with_pe = build_dinov3_detector_custom(
        # backbone_name="dinov3_vit7b16",
        backbone_name=args.backbone_name,
        backbone_pretrained=False,
        backbone_weights=None,
        backbone_ckpt_path=args.backbone_checkpoint,
        detector_ckpt_path=args.detector_checkpoint,
        num_classes=91,
    )
    backbone_with_pe.to(device)

    # if args.distributed:
        # backbone_with_pe = DDP(backbone_with_pe, device_ids=[device] if device.type == "cuda" else None)

    backbone_with_pe.eval()

    coco_root = args.coco_root
    ann_file = args.ann_file or coco_root / "annotations" / f"instances_{args.split}.json"
    image_root = coco_root / args.split

    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    loader = _build_loader(
        coco_root,
        args.split,
        ann_file,
        args.max_size,
        args.batch_size,
        args.num_workers,
        args.pin_memory,
        args.distributed,
        rank,
        world_size,
    )

    _extract_and_save(
        backbone_with_pe,
        loader,
        device,
        args.output_dir,
        args.start_index,
        args.save_fp16,
        rank,
    )

    if args.distributed:
        dist.barrier()


if __name__ == "__main__":
    main()
