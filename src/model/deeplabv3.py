import lightning as L
import torch
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class DeepLabV3(L.LightningModule):
    def __init__(
        self,
        # Hyperparameters
        num_classes: int,
        lr: float,
        lr_decay_rate: float,
        lr_sched_step_size: int,
        lr_sched_gamma: float,
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
        in_channels = self.model.classifier[0].convs[0][0].in_channels
        self.model.classifier = DeepLabHead(
            in_channels=in_channels, num_classes=self.hparams.num_classes
        )
        self.criterion = torch.nn.MSELoss(reduction="mean")

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            params,
            lr=self.lr,
            weight_decay=self.lr_decay_rate,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_sched_step_size, gamma=self.lr_sched_gamma
        )
        return [optimizer], [lr_scheduler]

    def forward(
        self, images: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        images = [img for img in images]
        return self.model(images)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, targets = batch
        output = self.model(images)
        loss = self.criterion(output["out"], targets)

        self.log(
            "train_loss",
            loss,
            batch_size=len(images),
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, targets = batch
        output = self.model(images)
        loss = self.criterion(output["out"], targets)

        self.log(
            "val_loss",
            loss,
            batch_size=len(images),
            prog_bar=True,
            sync_dist=True,
        )
        return loss
