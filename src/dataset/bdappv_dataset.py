import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import ImageReadMode
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F

import utils


class BdappvDataset(Dataset):
    def __init__(
        self,
        metadata: list[dict],
        transform: T.Compose,
    ) -> None:
        self.metadata = metadata
        self.transform = transform
        self.s3 = utils.s3.get_s3_resource()

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx) -> tuple[tv_tensors.Image, tv_tensors.Mask]:
        image_path, mask_path = self.metadata[idx]
        image = torchvision.io.read_image(image_path, mode=ImageReadMode.RGB)
        image = F.to_image(image)

        mask = torchvision.io.read_image(mask_path, mode=ImageReadMode.GRAY)
        mask = F.to_dtype(mask, torch.float, scale=True)

        targets = {}
        targets["masks"] = tv_tensors.Mask(mask, dtype=torch.float)

        image, targets = self.transform(image, targets)
        return image, targets["masks"]
