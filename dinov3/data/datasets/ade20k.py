# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union

from PIL import Image

from .decoders import Decoder, DenseTargetDecoder, ImageDataDecoder
from .extended import ExtendedVisionDataset


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"

    @property
    def dirname(self) -> str:
        return {
            _Split.TRAIN: "training",
            _Split.VAL: "validation",
        }[self]


def _image_to_segmentation_path(image_relpath: str) -> str:
    """Map an image relative path to the matching annotation relative path."""

    # Adequate for ADE20K folder naming convention: images/ADE/<split>/... -> annotations/ADE/<split>/...
    base, _ = os.path.splitext(image_relpath)
    return base.replace(os.path.join("images", "ADE"), os.path.join("annotations", "ADE"), 1) + ".png"


def _load_file_paths(root: str, split: _Split) -> Tuple[List[str], List[str]]:
    """Return image and annotation relative paths for a split.

    The loader prefers the official ADE text split files when present, but
    gracefully falls back to walking the ADE folder structure (images/ADE/<split>
    and annotations/ADE/<split>) if those files are not available.
    """

    list_file = os.path.join(root, f"ADE20K_object150_{split.value}.txt")

    if os.path.exists(list_file):
        with open(list_file) as f:
            split_file_names = sorted(f.read().strip().split("\n"))

        image_paths = [os.path.join("images", el) for el in split_file_names]
        segmentation_paths = [_image_to_segmentation_path(p) for p in image_paths]
        return image_paths, segmentation_paths

    # Fallback: traverse the ADE folder layout.
    image_root = os.path.join(root, "images", "ADE", split.dirname)
    if not os.path.isdir(image_root):
        raise FileNotFoundError(
            f"ADE20K images for split '{split.value}' not found under {image_root}."
        )

    discovered_images: List[str] = []
    for dirpath, _, filenames in os.walk(image_root):
        for fname in filenames:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                rel_dir = os.path.relpath(dirpath, root)
                discovered_images.append(os.path.join(rel_dir, fname))

    if not discovered_images:
        raise FileNotFoundError(
            f"No ADE20K images discovered under {image_root}; ensure the dataset is mounted correctly."
        )

    discovered_images.sort()
    segmentation_paths = [_image_to_segmentation_path(p) for p in discovered_images]
    return discovered_images, segmentation_paths


class ADE20K(ExtendedVisionDataset):
    Split = Union[_Split]
    Labels = Union[Image.Image]

    def __init__(
        self,
        split: "ADE20K.Split",
        root: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_decoder: Decoder = ImageDataDecoder,
        target_decoder: Decoder = DenseTargetDecoder,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=image_decoder,
            target_decoder=target_decoder,
        )

        self.image_paths, self.target_paths = _load_file_paths(root, split)

    def get_image_data(self, index: int) -> bytes:
        image_relpath = self.image_paths[index]
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Any:
        target_relpath = self.target_paths[index]
        target_full_path = os.path.join(self.root, target_relpath)
        with open(target_full_path, mode="rb") as f:
            target_data = f.read()
        return target_data

    def __len__(self) -> int:
        return len(self.image_paths)
