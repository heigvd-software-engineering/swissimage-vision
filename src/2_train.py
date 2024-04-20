import shutil
import time
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.strategies import DDPStrategy

from dataset.solar_datamodule import SolarDataModule
from model.fasterrcnn import FasterRCNN


def train(
    dm_seed: int,
    ann_path: Path,
    split: float,
    batch_size: int,
    image_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    num_classes: int,
    trainable_backbone_layers: int,
    lr: float,
    lr_momentum: float,
    lr_decay_rate: float,
    lr_sched_step_size: int,
    lr_sched_gamma: float,
    save_ckpt: bool,
    es_patience: bool,
    epochs: int,
    precision: str | None,
) -> None:
    L.seed_everything(seed)

    dm = SolarDataModule(
        ann_path=ann_path,
        image_size=image_size,
        seed=dm_seed,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = FasterRCNN(
        num_classes=num_classes,
        trainable_backbone_layers=trainable_backbone_layers,
        lr=lr,
        lr_momentum=lr_momentum,
        lr_decay_rate=lr_decay_rate,
        lr_sched_step_size=lr_sched_step_size,
        lr_sched_gamma=lr_sched_gamma,
    )

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    callbacks = [
        LearningRateMonitor(logging_interval="epoch", log_momentum=True),
    ]

    if save_ckpt:
        callbacks.append(
            ModelCheckpoint(
                filename="{epoch:02d}-{step}-{val_iou:.3f}",
                monitor="val_iou",
                save_top_k=1,
                mode="min",
            )
        )
    if es_patience is not None:
        callbacks.append(
            EarlyStopping(monitor="val_iou", patience=es_patience, mode="min"),
        )

    trainer = L.Trainer(
        log_every_n_steps=5,
        max_epochs=epochs,
        precision=precision if precision else "32-true",
        strategy=(
            DDPStrategy(
                static_graph=True,
            )
            if torch.cuda.is_available()
            else "auto"
        ),
        benchmark=True if torch.cuda.is_available() else False,
        callbacks=callbacks,
        limit_val_batches=0 if split == 1 else None,
    )

    trainer.fit(
        model,
        datamodule=dm,
    )

    # Copy model to root folder
    ckpt_folder = (
        sorted(
            Path("lightning_logs").glob("version_*"),
            key=lambda x: int(x.name.split("_")[-1]),
        )[-1]
        / "checkpoints"
    )

    if save_ckpt:
        model_path = list(ckpt_folder.glob("*.ckpt"))[0]
    else:
        model_path = ckpt_folder / "last.ckpt"
        trainer.save_checkpoint(model_path)

    # Wait for model to be saved
    max_retries = 20
    retries = 0
    out_path = Path("out/model.ckpt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        if retries >= max_retries:
            raise RuntimeError("Model not found")
        if model_path.exists():
            shutil.copy(model_path, out_path)
            break
        time.sleep(1)
        retries += 1


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    train_params = params["train"]
    datamodule_params = train_params["datamodule"]
    datamodule_setup_params = datamodule_params["setup"]
    datamodule_setup_params["dm_seed"] = datamodule_setup_params.pop("seed")

    train(
        **datamodule_setup_params,
        ann_path=Path("data/preprocessed/annotations.json"),
        num_workers=datamodule_params["num_workers"],
        pin_memory=datamodule_params["pin_memory"],
        **train_params["model"],
        **train_params["training"],
    )


if __name__ == "__main__":
    main()
