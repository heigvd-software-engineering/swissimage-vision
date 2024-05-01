from typing import Optional

import lightning as L
import torch
import torchmetrics.classification
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class DeepLabV3(L.LightningModule):
    def __init__(
        self,
        # Hyperparameters
        num_classes: int,
        lr: float,
        lr_decay_rate: float,
        lr_sched_step_size: Optional[int],
        lr_sched_gamma: Optional[float],
    ):
        """Initialize Faster R-CNN model.

        Args:
            num_classes (int): Number of classes in the dataset.
            lr (float): Learning rate for the optimizer.
            lr_decay_rate (float): Weight decay for the optimizer.
            lr_sched_step_size (int): Step size for the learning rate scheduler.
            lr_sched_gamma (float): Gamma for the learning rate scheduler.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = self.hparams.lr
        self.lr_decay_rate = self.hparams.lr_decay_rate
        self.lr_sched_step_size = self.hparams.lr_sched_step_size
        self.lr_sched_gamma = self.hparams.lr_sched_gamma

        self.model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT,
            aux_loss=None,
        )
        for param in self.model.parameters():
            param.requires_grad = False

        in_channels = self.model.classifier[0].convs[0][0].in_channels
        self.model.classifier = DeepLabHead(
            in_channels=in_channels, num_classes=self.hparams.num_classes
        )
        self.criterion = torch.nn.MSELoss(reduction="mean")
        # Metrics
        threshold = 0.5
        self.val_prec = torchmetrics.classification.BinaryPrecision(threshold=threshold)
        self.val_rec = torchmetrics.classification.BinaryRecall(threshold=threshold)
        self.val_acc = torchmetrics.classification.BinaryAccuracy(threshold=threshold)
        self.val_f1 = torchmetrics.classification.BinaryF1Score(threshold=threshold)
        self.val_auc_roc = torchmetrics.classification.BinaryAUROC()

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            params,
            lr=self.lr,
            weight_decay=self.lr_decay_rate,
        )
        if self.lr_sched_step_size is None or self.lr_sched_gamma is None:
            return optimizer
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_sched_step_size, gamma=self.lr_sched_gamma
        )
        return [optimizer], [lr_scheduler]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)["out"]

    def training_step(
        self, batch: list[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, targets = batch

        output = self(images)
        loss = self.criterion(output, targets)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(
        self, batch: list[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, targets = batch
        output = self(images)

        self.val_prec(output, targets),
        self.log("val_prec", self.val_prec, prog_bar=False)
        self.val_rec(output, targets),
        self.log("val_rec", self.val_rec, prog_bar=False)
        self.val_acc(output, targets),
        self.log("val_acc", self.val_acc, prog_bar=False)
        self.val_f1(output, targets),
        self.log("val_f1", self.val_f1, prog_bar=False)
        self.val_auc_roc(output, targets)
        self.log("val_auc_roc", self.val_auc_roc, prog_bar=False)
        loss = self.criterion(output, targets)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def get_metrics(self) -> dict[str, float]:
        callback_metrics = self.trainer.callback_metrics
        return {
            "train_loss": callback_metrics["train_loss"].detach().cpu().item(),
            "val_loss": callback_metrics["val_loss"].detach().cpu().item(),
            "val_prec": callback_metrics["val_prec"].detach().cpu().item(),
            "val_rec": callback_metrics["val_rec"].detach().cpu().item(),
            "val_acc": callback_metrics["val_acc"].detach().cpu().item(),
            "val_f1": callback_metrics["val_f1"].detach().cpu().item(),
            "val_auc_roc": callback_metrics["val_auc_roc"].detach().cpu().item(),
        }
