#!/usr/bin/env python
"""Run COCO detection evaluation with the DINOv3 detection head."""

from __future__ import annotations

import argparse
import os
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

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
            image = ResizeShortSide(self.max_size)(image)
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
    model = detectors.dinov3_vit7b16_de(pretrained=False, weights="", backbone_weights="")
    detector_module = model.detector
    backbone_wrapper = detector_module.backbone[0]
    if hasattr(backbone_wrapper, "backbone"):
        backbone_module = backbone_wrapper.backbone
    elif hasattr(backbone_wrapper, "_backbone"):
        backbone_module = backbone_wrapper._backbone
    else:
        raise AttributeError(
            "Backbone wrapper does not expose an inner backbone as 'backbone' or '_backbone'."
        )

    backbone_state = _load_checkpoint(args.backbone_checkpoint)
    backbone_module.load_state_dict(backbone_state, strict=True)

    detector_state = _load_checkpoint(args.detector_checkpoint)
    detector_module.load_state_dict(detector_state, strict=False)

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
    
    hub_model = torch.hub.load(
        '/app', 
        'dinov3_vit7b16_de',
          source="local", 
          weights=str(args.detector_checkpoint), 
          backbone_weights=str(args.backbone_checkpoint))
    detector = hub_model.detector  # 또는 hub_model 자체
    postprocessor = PostProcess(topk=100, reparam=True)

    model = DetectorWithProcessor(detector, postprocessor)
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