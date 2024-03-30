from pathlib import Path

from dataset.solar_datamodule import SolarDataModule


def main() -> None:
    dm = SolarDataModule(
        ann_dir=Path("data/raw/solar/annotations/xmls"),
        img_dir=Path("data/raw/solar/images"),
        seed=42,
        split=0.8,
        batch_size=32,
        num_workers=2,
        pin_memory=False,
    )
    dm.setup()
    for batch in dm.train_dataloader():
        print(batch)
        print(batch["image"].shape, batch["bb"].shape)
        exit()


if __name__ == "__main__":
    main()
