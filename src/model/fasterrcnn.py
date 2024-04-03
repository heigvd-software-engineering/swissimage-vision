import lightning as L
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNN(L.LightningModule):
    def __init__(
        self,
        # Hyperparameters
        num_classes: int,
        trainable_backbone_layers: int,
        lr: float,
        lr_momentum: float,
        lr_decay_rate: float,
        lr_sched_step_size: int,
        lr_sched_gamma: float,
    ):
        """Initialize Faster R-CNN model.

        Args:
            num_classes (int): Number of classes in the dataset.
            trainable_backbone_layers (int): Number of trainable layers in the backbone.
                                             Valid values are between 0 and 5.
            lr (float): Learning rate for the optimizer.
            lr_momentum (float): Momentum for the optimizer.
            lr_decay_rate (float): Weight decay for the optimizer.
            lr_sched_step_size (int): Step size for the learning rate scheduler.
            lr_sched_gamma (float): Gamma for the learning rate scheduler.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = self.hparams.lr
        self.lr_momentum = self.hparams.lr_momentum
        self.lr_decay_rate = self.hparams.lr_decay_rate
        self.lr_sched_step_size = self.hparams.lr_sched_step_size
        self.lr_sched_gamma = self.hparams.lr_sched_gamma

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights="DEFAULT",
            trainable_backbone_layers=self.hparams_initial.trainable_backbone_layers,
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.hparams_initial.num_classes
        )

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

        iou = self._get_iou(targets, outputs)
        self.log("val_iou", iou, batch_size=len(images), prog_bar=True, sync_dist=True)

    def _get_iou(
        self, targets: list[torch.Tensor], outputs: list[torch.Tensor]
    ) -> torch.Tensor:
        ious = []
        for i in range(len(targets)):
            iou_diag = torchvision.ops.box_iou(
                outputs[i]["boxes"], targets[i]["boxes"]
            ).diag()
            ious.append(
                torch.nn.functional.pad(
                    iou_diag, (0, len(targets[i]) - iou_diag.size(0)), value=0
                ).mean()
            )
        return torch.stack(ious).mean()
