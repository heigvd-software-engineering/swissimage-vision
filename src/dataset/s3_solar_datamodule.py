import json
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

from dataset.s3_solar_dataset import S3SolarDataset
from utils.seed import seed_worker


class S3SolarDataModule(L.LightningDataModule):
    def __init__(
        self,
        ann_path: Path,
        image_size: int,
        seed: int,
        split: float,
        batch_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
    ) -> None:
        """Initialize S3SolarDataModule.

        Args:
            ann_path (Path): Path to annotations (json).
            image_size (int): Size of the images.
            seed (int): Seed for reproducibility.
            split (float): Fraction of the data to use for training.
            batch_size (int): Batch size for the DataLoader.
            num_workers (int, optional): Number of workers for the DataLoader. Defaults to 8.
            pin_memory (bool, optional): Whether to pin memory. Defaults to True.
        """
        super().__init__()
        self.ann_path = ann_path
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
        # Load data from preprocessed DVC stage
        with open(self.ann_path, "r") as f:
            data = json.load(f)
        if len(data) == 0:
            raise ValueError("No data found. Check the paths.")
        indices = torch.randperm(len(data)).tolist()
        train_size = int(self.split * len(data))
        train = torch.utils.data.Subset(data, indices[:train_size])
        val = torch.utils.data.Subset(data, indices[train_size:])

        self.train_dataset = S3SolarDataset(
            metadata=train, transform=self.train_transform
        )
        self.val_dataset = S3SolarDataset(metadata=val, transform=self.val_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
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
                        brightness=0.25, contrast=0.5, saturation=0.5, hue=0.1
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
