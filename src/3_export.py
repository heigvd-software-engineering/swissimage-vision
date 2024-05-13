from pathlib import Path

import bentoml
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms.v2 import functional as F

from model.deeplabv3 import DeepLabV3


def export(batch_size: int, image_size: int, model_name: str) -> None:
    model = DeepLabV3.load_from_checkpoint("out/train/model.ckpt")
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
        outputs: torch.Tensor,
        orig_image_sizes: list[tuple[int, int]],
    ) -> list[np.ndarray]:
        outputs = outputs > 0.5
        outputs = outputs.to(torch.uint8) * 255

        masks = []
        for i, orig_image_size in enumerate(orig_image_sizes):
            masks.append(
                F.resize(outputs[i], orig_image_size).detach().cpu().squeeze().numpy()
            )
        return masks

    bentoml.torchscript.save_model(
        model_name,
        script_module,
        signatures={"__call__": {"batch_dim": 0, "batchable": True}},
        custom_objects={"preprocess": preprocess, "postprocess": postprocess},
    )
    out_dir = Path("out/export")
    out_dir.mkdir(parents=True, exist_ok=True)
    bentoml.models.export_model(
        f"{model_name}:latest", str(out_dir / f"{model_name}.bentomodel")
    )


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
