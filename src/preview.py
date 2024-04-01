from pathlib import Path

import lightning as L
import torch
import torchvision
import yaml
from torchvision.transforms.v2 import functional as F

from dataset.solar_datamodule import SolarDataModule


def preview(
    seed: int,
    data_root: str,
    split: float,
    batch_size: int,
    image_size: int,
    num_workers: int,
    pin_memory: bool,
    max_samples: int,
    output_dir: Path,
) -> None:
    L.seed_everything(seed)

    dm = SolarDataModule(
        ann_dir=Path(data_root) / "annotations",
        img_dir=Path(data_root) / "images",
        image_size=image_size,
        seed=seed,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dm.setup()

    output_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    for images, targets in dm.train_dataloader():
        for image, target in zip(images, targets):
            if sample_count == max_samples:
                return
            image = F.to_dtype(image, torch.uint8, scale=True)
            boxes = target["boxes"]
            sample = torchvision.utils.draw_bounding_boxes(
                image, boxes, width=1, colors="blue"
            )
            print("[INFO] Saved to", str(output_dir / f"sample_{sample_count}.png"))
            torchvision.io.write_png(
                sample, str(output_dir / f"sample_{sample_count}.png")
            )
            sample_count += 1


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    datamodule_params = params["datamodule"]
    datamodule_params["num_workers"] = 0
    datamodule_params["pin_memory"] = False
    preview(**datamodule_params, max_samples=10, output_dir=Path("data/preview"))


if __name__ == "__main__":
    main()
