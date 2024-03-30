import random
from pathlib import Path
from xml.etree import ElementTree as ET

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset.solar_dataset import SolarDataset


def seed_worker(worker_id):
    """
    Helper function to seed workers with different seeds for
    reproducibility.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SolarDataModule(L.LightningDataModule):
    def __init__(
        self,
        ann_dir: Path,
        img_dir: Path,
        seed: int,
        split: float,
        batch_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.ann_dir = ann_dir
        self.img_dir = img_dir
        self.seed = seed
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.gen = torch.Generator().manual_seed(self.seed)

    def setup(self, stage: str = None) -> None:
        # Load data from disk, split into train, val, test sets
        data = {"image": [], "bb": []}
        for ann_file in self.ann_dir.glob("*.xml"):
            tree = ET.parse(ann_file)
            data["image"].append(str(self.img_dir / tree.find("filename").text))
            data["bb"].append(
                [
                    int(tree.find("object/bndbox/xmin").text),
                    int(tree.find("object/bndbox/ymin").text),
                    int(tree.find("object/bndbox/xmax").text),
                    int(tree.find("object/bndbox/ymax").text),
                ]
            )
        data = pd.DataFrame(data)
        shuffled_data = data.sample(frac=1, random_state=self.seed)
        train = shuffled_data[: int(self.split * len(shuffled_data))]
        val = shuffled_data[int(self.split * len(shuffled_data)) :]
        self.train_dataset = SolarDataset(train)
        self.val_dataset = SolarDataset(val)

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
