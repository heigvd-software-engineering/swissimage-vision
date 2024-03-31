import random
from pathlib import Path
from xml.etree import ElementTree as ET

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

from dataset.solar_dataset import SolarDataset


def seed_worker(worker_id):
    """
    Helper function to seed workers with different seeds for
    reproducibility.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_fn(batch):
    return tuple(zip(*batch))


class SolarDataModule(L.LightningDataModule):
    def __init__(
        self,
        ann_dir: Path,
        img_dir: Path,
        image_size: int,
        seed: int,
        split: float,
        batch_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
    ) -> None:
        """Initialize SolarDataModule.

        Args:
            ann_dir (Path): Path to the directory containing the annotations.
            img_dir (Path): Path to the directory containing the images.
            image_size (int): Size of the images.
            seed (int): Seed for reproducibility.
            split (float): Fraction of the data to use for training.
            batch_size (int): Batch size for the DataLoader.
            num_workers (int, optional): Number of workers for the DataLoader. Defaults to 8.
            pin_memory (bool, optional): Whether to pin memory. Defaults to True.
        """
        super().__init__()
        self.ann_dir = ann_dir
        self.img_dir = img_dir
        self.image_size = image_size
        self.seed = seed
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.gen = torch.Generator().manual_seed(self.seed)

        self.train_transform = self._get_transform(is_train=True)
        self.val_transform = self._get_transform(is_train=False)

    def setup(self, stage: str = None) -> None:
        # Load data from disk, split into train, val, test sets
        data = []
        for ann_file in self.ann_dir.glob("*.xml"):
            tree = ET.parse(ann_file)
            objects = tree.findall("object")
            boxes = []
            for obj in objects:
                boxes.append(
                    [
                        int(obj.find("bndbox/xmin").text),
                        int(obj.find("bndbox/ymin").text),
                        int(obj.find("bndbox/xmax").text),
                        int(obj.find("bndbox/ymax").text),
                    ]
                )
            data.append(
                {
                    "image": str(self.img_dir / tree.find("filename").text),
                    "boxes": boxes,
                }
            )
        if len(data) == 0:
            raise ValueError("No data found. Check the paths.")
        indices = torch.randperm(len(data)).tolist()
        train_size = int(self.split * len(data))
        train = torch.utils.data.Subset(data, indices[:train_size])
        val = torch.utils.data.Subset(data, indices[train_size:])

        self.train_dataset = SolarDataset(
            metadata=train, transform=self.train_transform
        )
        self.val_dataset = SolarDataset(metadata=val, transform=self.val_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers // 2,
            worker_init_fn=seed_worker,
            generator=self.gen,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers // 2,
            worker_init_fn=seed_worker,
            generator=self.gen,
            pin_memory=self.pin_memory,
        )

    def _get_transform(self, is_train: bool):
        transforms = []
        if is_train:
            transforms.extend(
                [
                    T.RandomHorizontalFlip(0.5),
                    T.RandomVerticalFlip(0.5),
                    T.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                    ),
                ]
            )

        transforms.extend(
            [
                T.Resize((self.image_size, self.image_size)),
                T.ToDtype(torch.float, scale=True),
                T.ToPureTensor(),
            ]
        )
        return T.Compose(transforms)
