import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class SolarDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, transform=None, target_transform=None):
        self.metadata = metadata
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]
        image = read_image(sample["image"])
        bb = torch.tensor(sample["bb"])
        target = {"image": image, "bb": bb}
        return target
