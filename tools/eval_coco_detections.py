#!/usr/bin/env python
"""Run COCO detection evaluation with the DINOv3 detection head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from dinov3.hub import backbones, detectors

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ResizeLongestSide:
    def __init__(self, max_size: int):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        longest = max(width, height)
        if longest <= self.max_size:
            return image
        scale = self.max_size / longest
        new_size = (int(round(height * scale)), int(round(width * scale)))
        return F.resize(image, new_size, interpolation=InterpolationMode.BICUBIC)


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
            image = ResizeLongestSide(self.max_size)(image)
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
    parser.add_argument("--max-size", type=int, default=None, help="Optional maximum size for the longest image side; keeps aspect ratio.")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Discard predictions below this confidence.")
    parser.add_argument("--detector-weights", default=detectors.DetectionWeights.COCO2017, help="Detector checkpoint to use.")
    parser.add_argument("--backbone-weights", default=backbones.Weights.LVD1689M, help="Backbone checkpoint to use.")
    parser.add_argument("--output-json", default="coco_predictions.json", help="Where to store raw COCO-format predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    dataset = CocoDetectionForEval(
        root=str(image_root), ann_file=str(ann_file), transform=build_transform(), max_size=args.max_size
    )

    id_map = coco_id_mapping(dataset.coco)

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=_collate_fn
    )

    device = torch.device(args.device)
    model = detectors.dinov3_vit7b16_de(
        pretrained=True, weights=args.detector_weights, backbone_weights=args.backbone_weights
    )
    model.to(device).eval()

    predictions: List[dict] = []

    with torch.no_grad():
        for images, metas in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
            inputs = [img.to(device) for img in images]
            outputs = model(inputs)

            for output, meta in zip(outputs, metas):
                boxes = output["boxes"].cpu()
                scores = output["scores"].cpu()
                labels = output["labels"].cpu()

                h_orig, w_orig = meta["orig_size"]
                h_resized, w_resized = meta["resized_size"]
                scale_x = w_orig / w_resized
                scale_y = h_orig / h_resized

                boxes = boxes.clone()
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y

                for box, score, label in zip(boxes, scores, labels):
                    if score < args.score_threshold:
                        continue

                    x_min, y_min, x_max, y_max = box.tolist()
                    coco_box = [x_min, y_min, x_max - x_min, y_max - y_min]
                    predictions.append(
                        {
                            "image_id": meta["image_id"],
                            "category_id": id_map.get(int(label), int(label)),
                            "bbox": coco_box,
                            "score": float(score),
                        }
                    )

    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(predictions))
    print(f"Saved {len(predictions)} predictions to {output_path}")

    evaluate_predictions(dataset.coco, predictions)


if __name__ == "__main__":
    main()
