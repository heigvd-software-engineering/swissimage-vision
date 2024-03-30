import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image


class SolarDataset(Dataset):
    def __init__(self, metadata: list[dict], transform=None, target_transform=None):
        self.metadata = metadata
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample = self.metadata[idx]
        image = read_image(sample["image"])
        bbs = torch.tensor(sample["bbs"])
        # pad bounding boxes to be 10 in length
        bbs = F.pad(bbs, (0, 0, 0, 10 - bbs.shape[0]))
        return image, bbs
