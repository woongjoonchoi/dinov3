#!/usr/bin/env python
"""Run COCO detection evaluation with the DINOv3 detection head."""

from __future__ import annotations

import argparse
import os
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from dinov3.hub import detectors
from dinov3.eval.detection.models.detr import  PostProcess

from torch import nn

from dinov3.eval.detection.models.detr import PostProcess, build_model
from dinov3.eval.detection.config import DetectionHeadConfig
from dinov3.eval.detection.models.position_encoding import PositionEncoding

from dinov3.hub.backbones import Weights as BackboneWeights, dinov3_vit7b16, dinov3_vitl16plus
from dinov3_window_base1_3 import LocalGlobalHybridVisionTransformer
from dinov3_window_base1_1 import DinoVisionTransformerWindowBaseline1_1
from dinov3_window_base1_1.vit import _PatchOnlyWindowBlock
# 네가 이미 위에 정의한 클래스라면 import 해서 쓰고,
# 별도 파일이면 그대로 복붙해서 써도 됨.
class DetectorWithProcessor(torch.nn.Module):
    """
    takes as input a list of (3, H, W) normalized image tensors and outputs
    a list of dicts with keys "scores", "labels" and "boxes" (format XYXY)
    """

    def __init__(self, detector, postprocessor):
        super().__init__()
        self.detector = detector
        self.postprocessor = postprocessor

    def forward(self, samples: list[torch.Tensor]):
        outputs = self.detector(samples)
        sizes_tensor = torch.tensor(
            [sample.shape[1:] for sample in samples],
            device=samples[0].device,
        )  # N * [3, H, W]
        return self.postprocessor(
            outputs,
            target_sizes=sizes_tensor,
            original_target_sizes=sizes_tensor,
        )


def _remap_window_block_keys_to_patch_only(
    model: DinoVisionTransformerWindowBaseline1_1, state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Adapt checkpoints from pre-patch-only window blocks.

    Older checkpoints may store window block parameters directly under
    ``blocks.{i}.norm1``/``attn``/``mlp``. The current Baseline1_1 wraps those
    window blocks inside ``_PatchOnlyWindowBlock.patch_block``. This helper
    moves keys into the nested module so they can load cleanly with ``strict``
    semantics.
    """

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
                # Already in the expected location for patch-only blocks.
                break
            if key.startswith(prefix):
                # Move legacy window block parameters under patch_block.
                new_key = nested_prefix + key[len(prefix) :]
                break
        remapped[new_key] = value
    return remapped


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

    # -------------------------------
    # 4. detector head 로컬 weight 로딩 (옵션)
    # -------------------------------
    if detector_ckpt_path is not None:
        if not os.path.isfile(detector_ckpt_path):
            raise FileNotFoundError(
                f"detector_ckpt_path not found: {detector_ckpt_path}"
            )
        ckpt = torch.load(detector_ckpt_path, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        detector.load_state_dict(state_dict, strict=False)

    # inference용 설정 (원래 코드와 동일)
    detector.num_queries = detector.num_queries_one2one
    detector.transformer.two_stage_num_proposals = detector.num_queries

    postprocessor = PostProcess(config.topk, config.reparam)
    model = DetectorWithProcessor(detector=detector, postprocessor=postprocessor)

    # 여기서는 .to(device), .eval() 호출 안 함
    return model

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class DetectorWithProcessor(nn.Module):
    def __init__(self, detector, postprocessor):
        super().__init__()
        self.detector = detector          # PlainDETRReParam
        self.postprocessor = postprocessor  # PostProcess(reparam=True)

    def forward(self, samples: list[Tensor], metas: list[dict]):
        outputs = self.detector(samples)

        resized_sizes = torch.tensor(
            [m["resized_size"] for m in metas],
            device=samples[0].device,
            dtype=outputs["pred_boxes"].dtype,
        )
        orig_sizes = torch.tensor(
            [m["orig_size"] for m in metas],
            device=samples[0].device,
            dtype=outputs["pred_boxes"].dtype,
        )

        return self.postprocessor(
            outputs,
            target_sizes=resized_sizes,
            original_target_sizes=orig_sizes,
        )


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
        # F.resize expects (H, W)
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


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


def _collate_fn(batch: Iterable[Tuple[torch.Tensor, dict]]):
    images, metas = zip(*batch)
    return list(images), list(metas)


def coco_id_mapping(coco: COCO) -> dict[int, int]:
    cat_ids = sorted(coco.getCatIds())
    return {idx: cat_id for idx, cat_id in enumerate(cat_ids)}


def evaluate_predictions(coco_gt: COCO, predictions: List[dict]) -> None:
    if len(predictions) == 0:
        raise RuntimeError("No predictions were produced; cannot run COCO evaluation.")

    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.imgIds = coco_gt.getImgIds()
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DINOv3 detector on COCO.")
    parser.add_argument("--coco-root", required=True, help="Path to COCO dataset root (containing annotations/ and split folders).")
    parser.add_argument("--split", default="val2017", help="Image split to evaluate (e.g., val2017).")
    parser.add_argument(
        "--ann-file",
        default=None,
        help="Optional path to annotations file; defaults to annotations/instances_<split>.json under --coco-root.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for evaluation.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for inference (cuda or cpu).")
    parser.add_argument("--distributed", action="store_true", help="Use torch.distributed with torchrun for multi-GPU evaluation.")
    parser.add_argument("--pin-memory", action="store_true", help="Pin dataloader memory for faster host->device copies.")
    parser.add_argument("--max-size", type=int, default=None, help="Optional maximum size for the shortest image side; keeps aspect ratio.")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Discard predictions below this confidence.")
    parser.add_argument("--backbone-name" , type=str, default="dinov3_vit7b16",help="backbone-name")
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        required=True,
        help="Path to the pretrained backbone checkpoint to load directly.",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        required=True,
        help="Path to the detector head checkpoint to load directly.",
    )
    parser.add_argument("--output-json", default="coco_predictions.json", help="Where to store raw COCO-format predictions.")
    return parser.parse_args()


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


def _build_model_from_checkpoints(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = build_dinov3_detector_custom(
        backbone_name=args.backbone_name,
        backbone_pretrained=False,
        backbone_weights=None,
        backbone_ckpt_path=str(args.backbone_checkpoint),
        detector_ckpt_path=str(args.detector_checkpoint),
        num_classes=91,
    )
    model.to(device)
    return model


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

    coco_root = Path(args.coco_root)
    ann_file = (
        Path(args.ann_file)
        if args.ann_file is not None
        else coco_root / "annotations" / f"instances_{args.split}.json"
    )
    image_root = coco_root / args.split

    # if rank == 0 :
    #     ckpt = torch.load(args.detector_checkpoint, map_location="cpu")
    #     print(ckpt['model'].keys())
    #     state = ckpt.get("model", ckpt)  # some checkpoints store under "model"
    #     print(len(state))
    #     from collections import Counter

    #     prefixes = Counter(k.split('.')[0] for k in state.keys())
    #     print(prefixes)
    
    # exit()
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    if not args.backbone_checkpoint.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {args.backbone_checkpoint}")
    if not args.detector_checkpoint.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {args.detector_checkpoint}")

    dataset = CocoDetectionForEval(
        root=str(image_root), ann_file=str(ann_file), transform=build_transform(), max_size=args.max_size
    )

    id_map = coco_id_mapping(dataset.coco)

    # ckpt = torch.load(args.detector_checkpoint, map_location="cpu")

    # print(ckpt.keys())  # 보통 'model', 'args', 'epoch', ... 이런거 있음

    # adv = ckpt.get("advance", None)
    # print(type(adv))
    # if isinstance(adv, dict):
    #     print("advance keys:", adv.keys())

    # exit()
    sampler = None
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=_collate_fn,
        sampler=sampler,
        shuffle=False,
        pin_memory=args.pin_memory,
    )
    # model = _build_model_from_checkpoints(args, device)
    # model = detectors.dinov3_vit7b16_de(
    #     pretrained=True,  # 이미 weights를 직접 주니까 굳이 True일 필요 없음
    #     weights=str(args.detector_checkpoint),        # <- 여기 경로 문자열
    #     backbone_weights=str(args.backbone_checkpoint),  # <- 여기 경로 문자열
    # )
    
    # hub_model = torch.hub.load(
    #     '/app', 
    #     'dinov3_vit7b16_de',
    #       source="local", 
    #       weights=str(args.detector_checkpoint), 
    #       backbone_weights=str(args.backbone_checkpoint))
    # detector = hub_model.detector  # 또는 hub_model 자체
    # postprocessor = PostProcess(topk=100, reparam=True)

    # model = DetectorWithProcessor(detector, postprocessor)

    
    model = build_dinov3_detector_custom(
        # backbone_name="dinov3_vit7b16",
        backbone_name=args.backbone_name,
        backbone_pretrained=False,
        backbone_weights=None,
        backbone_ckpt_path=args.backbone_checkpoint,
        detector_ckpt_path=args.detector_checkpoint,
        num_classes=91,
    )
    model.to(device)
    
    if args.distributed:
        model = DDP(model, device_ids=[device] if device.type == "cuda" else None)
    model.eval()

    # # 1) 데이터셋에서 한 샘플 뽑기
    # idx = 0  # 아무거나
    # img, meta = dataset[idx]
    # print("meta:", meta)

    # img_id = meta["image_id"]
    # h_orig, w_orig = meta["orig_size"]
    # h_resized, w_resized = meta["resized_size"]
    # print("orig_size   :", h_orig, w_orig)
    # print("resized_size:", h_resized, w_resized)
    # print("tensor shape:", img.shape)  # (3, H_resized, W_resized)

    # # 2) GT 박스 확인
    # ann_ids = dataset.coco.getAnnIds(imgIds=[img_id])
    # anns = dataset.coco.loadAnns(ann_ids)
    # print(f"#GT boxes for image_id={img_id}: {len(anns)}")
    # for a in anns[:5]:
    #     print("GT bbox (x,y,w,h):", a["bbox"], "category_id:", a["category_id"])

    # # 3) 모델 prediction 한 장만
    # model.eval()
    # with torch.no_grad():
    #     # 여기서 model은 DDP 감싸기 전 기준 (DDP면 model.module)
    #     out = model([img.to(device)], [meta])[0]   # DetectorWithProcessor가 (images, metas)를 받도록 한 버전 기준

    # boxes  = out["boxes"].cpu()
    # scores = out["scores"].cpu()
    # labels = out["labels"].cpu()

    # print("pred boxes shape:", boxes.shape)   # K x 4
    # print("top5 scores:", scores[:5])
    # print("top5 boxes:", boxes[:5])
    # print("top5 labels:", labels[:5])
    # print("boxes min/max x:", boxes[:, [0, 2]].min().item(), boxes[:, [0, 2]].max().item())
    # print("boxes min/max y:", boxes[:, [1, 3]].min().item(), boxes[:, [1, 3]].max().item())

    # exit()
    predictions_all: List[dict] = []

    with torch.no_grad():
        for iteration, (images, metas) in enumerate(
            tqdm(dataloader, desc="Evaluating", total=len(dataloader)), start=1
        ):
            inputs = [img.to(device) for img in images]
            outputs = model(inputs,metas)
            # outputs = model(inputs)
            batch_predictions: List[dict] = []
            # print(outputs[0].keys())        # boxes, scores, labels?
            # print(outputs[0]["boxes"][:5])
            # print(outputs[0]["labels"][:5])
            # print(outputs[0]["scores"][:5])
            # exit()
            for output, meta in zip(outputs, metas):
                boxes = output["boxes"].cpu()
                scores = output["scores"].cpu()
                labels = output["labels"].cpu()

                # h_orig, w_orig = meta["orig_size"]
                # h_resized, w_resized = meta["resized_size"]
                # scale_x = w_orig / w_resized
                # scale_y = h_orig / h_resized

                # boxes = boxes.clone()
                # boxes[:, [0, 2]] *= scale_x
                # boxes[:, [1, 3]] *= scale_y

                for box, score, label in zip(boxes, scores, labels):
                    if score < args.score_threshold:
                        continue

                    x_min, y_min, x_max, y_max = box.tolist()
                    coco_box = [x_min, y_min, x_max - x_min, y_max - y_min]
                    batch_predictions.append(
                        {
                            "image_id": meta["image_id"],
                            # "category_id": id_map.get(int(label), int(label)),
                            "category_id": int(label),
                            "bbox": coco_box,
                            "score": float(score),
                        }
                    )

            if args.distributed:
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

    if not args.distributed or rank == 0:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(predictions_all))
        print(f"Saved {len(predictions_all)} predictions to {output_path}")

        evaluate_predictions_with_logging(dataset.coco, predictions_all, iteration=None)

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()