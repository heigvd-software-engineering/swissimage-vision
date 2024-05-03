import shutil
import zipfile
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

from dataset.bdappv_dataset import BdappvDataset
from utils.seed import seed_worker


class BdappvDataModule(L.LightningDataModule):
    DATA_PATH = Path("data/raw/bdappv.zip")
    EXTRACT_PATH = Path("data/extracted")
    ROOT_DIRS = [EXTRACT_PATH / "bdappv/google", EXTRACT_PATH / "bdappv/ign"]

    def __init__(
        self,
        image_size: int,
        seed: int,
        split: float,
        batch_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
    ) -> None:
        """Initialize S3SolarDataModule.

        Args:
            image_size (int): Size of the images.
            seed (int): Seed for reproducibility.
            split (float): Fraction of the data to use for training.
            batch_size (int): Batch size for the DataLoader.
            num_workers (int, optional): Number of workers for the DataLoader. Defaults to 8.
            pin_memory (bool, optional): Whether to pin memory. Defaults to True.
        """
        super().__init__()
        self.image_size = image_size
        self.seed = seed
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.gen = torch.Generator().manual_seed(self.seed)

        self.train_transform = self._get_transform(is_train=True)
        self.val_transform = self._get_transform(is_train=False)

    def prepare_data(self) -> None:
        if self.EXTRACT_PATH.exists():
            return

        print("[INFO] Extracting data...")
        self.EXTRACT_PATH.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(self.DATA_PATH, "r") as zip_ref:
            zip_ref.extractall(str(self.EXTRACT_PATH))

    def setup(self, stage: str = None) -> None:
        # root should have img/ and mask/ folders
        data = []
        for root_dir in self.ROOT_DIRS:
            for mask_path in root_dir.glob("mask/*.png"):
                image_path = root_dir / "img" / mask_path.name
                data.append((str(image_path), str(mask_path)))
        indices = torch.randperm(len(data)).tolist()
        train_size = int(self.split * len(data))

        print(f"Found {len(data)} samples:")
        print(f"  - Training: {train_size}")
        print(f"  - Validation: {len(data) - train_size}")

        train = torch.utils.data.Subset(data, indices[:train_size])
        val = torch.utils.data.Subset(data, indices[train_size:])

        self.train_dataset = BdappvDataset(
            metadata=train, transform=self.train_transform
        )
        self.val_dataset = BdappvDataset(metadata=val, transform=self.val_transform)

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
                        brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1
                    ),
                    T.RandomResizedCrop(
                        size=(self.image_size, self.image_size),
                        scale=(0.6, 1.0),
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
