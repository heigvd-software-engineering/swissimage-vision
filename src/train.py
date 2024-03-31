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
from torchsummary import summary

from dataset.solar_datamodule import SolarDataModule
from model.fasterrcnn import FasterRCNNModel


def train(
    seed: int,
    data_root: str,
    split: float,
    batch_size: int,
    image_size: int,
    num_workers: int,
    pin_memory: bool,
    num_classes: int,
    lr: float,
    lr_momentum: float,
    lr_decay_rate: float,
    lr_sched_step_size: int,
    lr_sched_gamma: float,
    save_ckpt: bool,
    es_patience: bool,
    epochs: int,
    precision: str | None,
    accumulate_grad_batches: int | None,
) -> None:
    L.seed_everything(seed)

    dm = SolarDataModule(
        ann_dir=Path(data_root) / "annotations/xmls",
        img_dir=Path(data_root) / "images",
        image_size=image_size,
        seed=seed,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = FasterRCNNModel(
        num_classes=num_classes,
        lr=lr,
        lr_momentum=lr_momentum,
        lr_decay_rate=lr_decay_rate,
        lr_sched_step_size=lr_sched_step_size,
        lr_sched_gamma=lr_sched_gamma,
    )

    if not torch.cuda.is_available():
        summary(model, batch_size=batch_size)
    else:
        torch.set_float32_matmul_precision("high")

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if save_ckpt:
        callbacks.append(
            ModelCheckpoint(
                filename="{epoch:02d}-{step}-{val_loss:.3f}",
                monitor="val_loss",
                save_top_k=1,
                mode="min",
            )
        )
    if es_patience is not None:
        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=es_patience, mode="min"),
        )

    trainer = L.Trainer(
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
        accumulate_grad_batches=accumulate_grad_batches or 1,
        callbacks=callbacks,
        limit_val_batches=0 if split == 1 else None,
    )

    trainer.fit(
        model,
        datamodule=dm,
    )


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    prepare_params = params["prepare"]
    prepare_params.pop("seed")
    prepare_params.pop("num_workers")
    train_params = params["train"]

    train(**prepare_params, **train_params)


if __name__ == "__main__":
    main()
