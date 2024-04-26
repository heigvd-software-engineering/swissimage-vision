from pathlib import Path

import lightning as L
import torch
import torchvision
import yaml
from torchvision.transforms.v2 import functional as F

from dataset.solar_datamodule import SolarDataModule


def save_samples(
    dl: torch.utils.data.DataLoader, max_samples: int, prefix: str, output_dir: Path
) -> None:
    sample_count = 0
    for images, targets in dl:
        for image, target in zip(images, targets):
            if sample_count == max_samples:
                return
            image = F.to_dtype(image, torch.uint8, scale=True)
            sample = torchvision.utils.draw_segmentation_masks(
                image, target["masks"], alpha=0.4, colors="blue"
            )
            torchvision.utils.draw_keypoints
            print("[INFO] Saved to", str(output_dir / f"{prefix}_{sample_count}.png"))
            torchvision.io.write_png(
                sample, str(output_dir / f"{prefix}_{sample_count}.png")
            )
            sample_count += 1


def preview(
    seed: int,
    split: float,
    batch_size: int,
    image_size: int,
    num_workers: int,
    pin_memory: bool,
    ann_path: Path,
    max_samples: int,
    output_dir: Path,
) -> None:
    L.seed_everything(seed)

    dm = SolarDataModule(
        ann_path=ann_path,
        image_size=image_size,
        seed=seed,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dm.setup()

    output_dir.mkdir(parents=True, exist_ok=True)
    save_samples(
        dl=dm.train_dataloader(),
        max_samples=max_samples,
        prefix="train",
        output_dir=output_dir,
    )
    save_samples(
        dl=dm.val_dataloader(),
        max_samples=max_samples,
        prefix="val",
        output_dir=output_dir,
    )


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    datamodule_setup_params = params["train"]["datamodule"]["setup"]
    preview(
        **datamodule_setup_params,
        num_workers=0,
        pin_memory=False,
        ann_path=Path("data/preprocessed/annotations.json"),
        max_samples=10,
        output_dir=Path("data/preview"),
    )


if __name__ == "__main__":
    main()
