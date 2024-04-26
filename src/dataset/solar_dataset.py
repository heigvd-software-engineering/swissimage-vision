import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F

import utils


class SolarDataset(Dataset):
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

    def __getitem__(self, idx) -> tuple[Image.Image, tv_tensors.Mask]:
        sample = self.metadata[idx]
        bucket, prefix = utils.s3.split_url_to_bucket_and_prefix(sample["image"])
        image = F.to_image(utils.s3.get_image(self.s3, bucket, prefix))

        targets = {}
        mask = Image.new("L", image.shape[1:], 0)
        if sample["polys"]:
            draw = ImageDraw.Draw(mask)
            for points in sample["polys"]:
                points = [tuple(point) for point in points]
                # Create polygon from points
                draw.polygon(points, fill=1)

        targets["masks"] = tv_tensors.Mask(mask, dtype=torch.float)

        image, targets = self.transform(image, targets)
        return image, targets["masks"]
