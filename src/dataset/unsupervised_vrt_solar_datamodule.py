from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

from dataset.unsupervised_vrt_solar_dataset import UnsupervisedVRTSolarDataset
from utils.extract_tiles import get_bounds
from utils.seed import seed_worker


def collate_fn(batch) -> tuple:
    masks, meta = zip(*batch)
    return torch.stack(masks), meta


class UnsupervisedVRTSolarDataModule(L.LightningDataModule):
    def __init__(
        self,
        tile_size: int,
        tif_path: str,
        commune_name: str,
        commune_x_ratio: float,
        commune_y_ratio: float,
        image_size: int,
        seed: int,
        split: float,
        batch_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
    ) -> None:
        """Initialize S3SolarDataModule.

        Args:
            tile_size (int): Size of the tiles.
            tif_path (str): Path to the tif file.
            commune_name (str): Name of the commune.
            commune_x_ratio (float): Ratio for the commune x axis.
            commune_y_ratio (float): Ratio for the commune y axis.
            image_size (int): Size of the images.
            seed (int): Seed for reproducibility.
            split (float): Fraction of the data to use for training.
            batch_size (int): Batch size for the DataLoader.
            num_workers (int, optional): Number of workers for the DataLoader. Defaults to 8.
            pin_memory (bool, optional): Whether to pin memory. Defaults to True.
        """
        super().__init__()
        self.tile_size = tile_size
        self.tif_path = tif_path
        self.commune_name = commune_name
        self.commune_x_ratio = commune_x_ratio
        self.commune_y_ratio = commune_y_ratio

        self.image_size = image_size
        self.seed = seed
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.gen = torch.Generator().manual_seed(self.seed)

        # self.train_transform = self._get_transform(is_train=True)
        self.val_transform = self._get_transform(is_train=False)

    def setup(self, stage: str = None) -> None:
        # self.train_dataset = UnsupervisedVRTSolarDataset(transform=self.train_transform)
        self.val_dataset = UnsupervisedVRTSolarDataset(
            transform=self.val_transform,
            tile_size=self.tile_size,
            tif_path=self.tif_path,
            gdf_bounds=get_bounds(self.commune_name),
            commune_x_ratio=self.commune_x_ratio,
            commune_y_ratio=self.commune_y_ratio,
        )

    # Note: Currently not used
    # def train_dataloader(self) -> DataLoader:
    #     return DataLoader(
    #         self.train_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #         drop_last=False,
    #         num_workers=self.num_workers // 2,
    #         worker_init_fn=seed_worker,
    #         generator=self.gen,
    #         pin_memory=self.pin_memory,
    #     )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=self.gen,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
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

    def __del__(self):
        del self.val_dataset
