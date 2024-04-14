import bentoml
import torch
import yaml
from PIL import Image
from torchvision.transforms.v2 import functional as F

from model.fasterrcnn import FasterRCNN


def export(batch_size: int, image_size: int, model_name: str) -> None:
    model = FasterRCNN.load_from_checkpoint("out/model.ckpt")
    model.eval()
    model.to("cuda")
    script_module = model.to_torchscript(
        method="script",
        example_inputs=torch.rand(
            batch_size,
            3,
            image_size,
            image_size,
            dtype=torch.float32,
            device="cuda",
        ),
    )

    def preprocess(images: Image.Image | list[Image.Image]) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = [images]
        images = [image.convert("RGB") for image in images]
        images_tensor = torch.stack([F.to_image(image) for image in images])
        images_input = F.to_dtype(images_tensor, torch.float32, scale=True)
        images_input = F.resize(images_input, [image_size, image_size])
        return images_input

    def postprocess(
        outputs: tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]],
        orig_image_sizes: list[tuple[int, int]],
    ) -> list[dict[str, list[float]]]:
        _, detections = outputs

        predictions = []
        for detection, orig_image_size in zip(detections, orig_image_sizes):
            scale_x = orig_image_size[0] / image_size
            scale_y = orig_image_size[1] / image_size
            boxes = detection["boxes"].cpu()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            scores = detection["scores"].cpu()
            predictions.append(
                {
                    "boxes": boxes.tolist(),
                    "scores": scores.tolist(),
                }
            )
        return predictions

    bentoml.torchscript.save_model(
        model_name,
        script_module,
        signatures={"__call__": {"batch_dim": 0, "batchable": True}},
        custom_objects={"preprocess": preprocess, "postprocess": postprocess},
    )
    bentoml.models.export_model(f"{model_name}:latest", f"out/{model_name}.bentomodel")


def main() -> None:
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available, please run on a machine with a GPU.")

    params = yaml.safe_load(open("params.yaml"))
    datamodule_setup_params = params["train"]["datamodule"]["setup"]
    export_params = params["export"]

    export(
        batch_size=datamodule_setup_params["batch_size"],
        image_size=datamodule_setup_params["image_size"],
        model_name=export_params["model_name"],
    )


if __name__ == "__main__":
    main()
