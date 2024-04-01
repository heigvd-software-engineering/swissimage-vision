import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F


class SolarDataset(Dataset):
    def __init__(
        self,
        metadata: list[dict],
        transform: T.Compose,
    ):
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample = self.metadata[idx]
        image = read_image(sample["image"])

        targets = {}
        if sample["boxes"]:  # If there are annotations
            boxes = torch.tensor(sample["boxes"], dtype=torch.float)
            targets["boxes"] = tv_tensors.BoundingBoxes(
                boxes, format="XYXY", canvas_size=F.get_size(image)
            )
            targets["labels"] = torch.ones((boxes.shape[0],), dtype=torch.int64)
        else:
            targets["boxes"] = torch.empty((0, 4), dtype=torch.float)
            targets["labels"] = torch.empty((0,), dtype=torch.int64)

        image, targets = self.transform(image, targets)
        return image, targets
