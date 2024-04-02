import bentoml
import torch
import yaml

from model.fasterrcnn import FasterRCNN


def export(batch_size: int, image_size: int, num_classes: int, model_name: str) -> None:
    model = FasterRCNN.load_from_checkpoint("out/model.ckpt", num_classes=num_classes)
    script_module = model.to_torchscript(
        example_inputs=torch.rand(
            batch_size, 3, image_size, image_size, dtype=torch.float32
        )
    )
    bentoml.torchscript.save_model(model_name, script_module)
    bentoml.models.export_model(f"{model_name}:latest", f"out/{model_name}.bentomodel")


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    datamodule_params = params["datamodule"]
    train_params = params["train"]
    export_params = params["export"]

    export(
        batch_size=datamodule_params["batch_size"],
        image_size=datamodule_params["image_size"],
        num_classes=train_params["num_classes"],
        model_name=export_params["model_name"],
    )


if __name__ == "__main__":
    main()
