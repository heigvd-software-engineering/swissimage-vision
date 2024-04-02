import lightning as L
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNN(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        # Hyperparameters
        lr: float,
        lr_momentum: float,
        lr_decay_rate: float,
        lr_sched_step_size: int,
        lr_sched_gamma: float,
    ):
        """Initialize Faster R-CNN model.

        Args:
            num_classes (int): Number of classes in the dataset.
            lr (float): Learning rate for the optimizer.
            lr_momentum (float): Momentum for the optimizer.
            lr_decay_rate (float): Weight decay for the optimizer.
            lr_sched_step_size (int): Step size for the learning rate scheduler.
            lr_sched_gamma (float): Gamma for the learning rate scheduler.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["num_classes"])
        self.lr = self.hparams.lr
        self.lr_momentum = self.hparams.lr_momentum
        self.lr_decay_rate = self.hparams.lr_decay_rate
        self.lr_sched_step_size = self.hparams.lr_sched_step_size
        self.lr_sched_gamma = self.hparams.lr_sched_gamma

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights="DEFAULT", trainable_backbone_layers=5
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=self.lr,
            momentum=self.lr_momentum,
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
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss_val = loss.item()

        self.log(
            "train_loss",
            loss_val,
            batch_size=len(images),
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, targets = batch
        # In validation mode, the loss is not calculated.
        # See https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn
        outputs = self.model(images)

        iou = self._get_ciou(targets, outputs)
        self.log("val_ciou", iou, batch_size=len(images), prog_bar=True, sync_dist=True)

    def _get_ciou(self, targets: torch.Tensor, outputs: torch.Tensor) -> float:
        iou = torch.stack(
            [
                torchvision.ops.complete_box_iou(
                    outputs[i]["boxes"], targets[i]["boxes"]
                ).mean()
                for i in range(len(targets))
            ]
        ).mean()
        if torch.isnan(iou):
            return 0.0
        return iou.item()
