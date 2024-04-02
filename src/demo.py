from functools import partial

import gradio as gr
import torch
import torchvision
import yaml
from PIL import Image
from torchvision.transforms.v2 import functional as F

from model.fasterrcnn import FasterRCNN


def dectect(model: FasterRCNN, device: str, image_size: int, image: Image) -> Image:
    if image is None:
        return None
    image_tensor = F.to_image(image)
    image_input = F.to_dtype(image_tensor, torch.float32, scale=True)
    image_input = F.resize(image_input, [image_size, image_size])
    image_input = image_input.to(device).unsqueeze(0)

    scale_x = image.size[0] / image_size
    scale_y = image.size[1] / image_size

    threshold = 0.75

    with torch.no_grad():
        outputs = model(image_input)
        boxes = outputs[0]["boxes"]
        boxes = boxes[outputs[0]["scores"] > threshold]
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        scores = list(filter(lambda x: x > threshold, outputs[0]["scores"].tolist()))
        scores_str = list(map(lambda x: str(round(x, 3)), scores))
        image_output = torchvision.utils.draw_bounding_boxes(
            image_tensor, boxes, labels=scores_str, width=3, colors="red"
        )
        res = F.to_pil_image(image_output)
        return res


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    train_params = params["train"]
    datamodule_params = params["datamodule"]

    model = FasterRCNN.load_from_checkpoint(
        "out/model.ckpt", num_classes=train_params["num_classes"]
    )
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        model = torch.nn.DataParallel(model)
        model.to(device)
    model.eval()

    demo = gr.Interface(
        fn=partial(dectect, model, device, datamodule_params["image_size"]),
        inputs=gr.Image(type="pil", label="Input image"),
        outputs=gr.Image(type="pil", label="Output image"),
        examples=[
            "data/prepared/images/solar_2.JPG",
            "data/prepared/images/solar_3.JPG",
            "data/prepared/images/solar_4.JPG",
        ],
        allow_flagging="never",
        analytics_enabled=False,
    )
    demo.launch()


if __name__ == "__main__":
    main()
