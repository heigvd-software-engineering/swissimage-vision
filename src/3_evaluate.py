from pathlib import Path

import lightning as L
import torch
import torchvision
import yaml
from torchvision.transforms.v2 import functional as F

from dataset.solar_datamodule import SolarDataModule
from model.deeplabv3 import DeepLabV3


def evaluate(
    seed: int,
    ann_path: Path,
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
        ann_path=ann_path,
        image_size=image_size,
        seed=seed,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dm.setup()

    model = DeepLabV3.load_from_checkpoint("out/model.ckpt")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        model = torch.nn.DataParallel(model)
        model.to(device)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    for images, targets in dm.train_dataloader():
        with torch.no_grad():
            outputs = model(images)
        for image, target, output in zip(images, targets, outputs):
            if sample_count == max_samples:
                return
            image = F.to_dtype(image, torch.uint8, scale=True)
            target_mask = F.to_dtype(target, bool)
            output_mask = (output > 0.5).detach().cpu()
            sample = torchvision.utils.draw_segmentation_masks(
                image, target_mask, alpha=0.4, colors="blue"
            )
            sample = torchvision.utils.draw_segmentation_masks(
                sample, output_mask, alpha=0.4, colors="red"
            )
            print("[INFO] Saved to", str(output_dir / f"sample_{sample_count}.png"))
            torchvision.io.write_png(
                sample, str(output_dir / f"sample_{sample_count}.png")
            )
            sample_count += 1


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    train_params = params["train"]
    datamodule_setup_params = train_params["datamodule"]["setup"]

    evaluate(
        **datamodule_setup_params,
        ann_path=Path("data/preprocessed/annotations.json"),
        num_workers=0,
        pin_memory=False,
        max_samples=10,
        output_dir=Path("data/evaluate"),
    )


if __name__ == "__main__":
    main()
