# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import csv
import logging
import os
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import webdataset as wds

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset

from torchvision.datasets.vision import StandardTransform

logger = logging.getLogger("dinov3")
_Target = int


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 1_281_167,
            _Split.VAL: 50_000,
            _Split.TEST: 100_000,
        }
        return split_lengths[self]

    def get_dirname(self, class_id: Optional[str] = None) -> str:
        return self.value if class_id is None else os.path.join(self.value, class_id)

    def get_image_relpath(self, actual_index: int, class_id: Optional[str] = None) -> str:
        dirname = self.get_dirname(class_id)
        if self == _Split.TRAIN:
            basename = f"{class_id}_{actual_index}"
        else:  # self in (_Split.VAL, _Split.TEST):
            basename = f"ILSVRC2012_{self.value}_{actual_index:08d}"
        return os.path.join(dirname, basename + ".JPEG")

    def parse_image_relpath(self, image_relpath: str) -> Tuple[str, int]:
        assert self != _Split.TEST
        dirname, filename = os.path.split(image_relpath)
        class_id = os.path.split(dirname)[-1]
        basename, _ = os.path.splitext(filename)
        actual_index = int(basename.split("_")[-1])
        return class_id, actual_index


class BLIP3Kale(wds.WebDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "BLIP3Kale.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        # list all .tar files
        trainset_url = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.tar')]

        super().__init__(
            trainset_url,
            resampled=True,
            shardshuffle=True,
            nodesplitter=wds.split_by_node,
        )
        self._split = split

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

        self.target_decoder = TargetDecoder
        self.image_decoder = ImageDataDecoder

    @property
    def split(self) -> "BLIP3Kale.Split":
        return self._split

    def make_sample(self, example):
        image = example['jpg']
        target = example['txt']
        target = self.target_decoder(target).decode()

        try:
            image = self.image_decoder(image).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {example['__key__']}") from e

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
