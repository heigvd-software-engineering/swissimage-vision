# from pathlib import Path

# import cv2
# import matplotlib.pyplot as plt

# from dataset.solar_datamodule import SolarDataModule


# def main() -> None:
#     dm = SolarDataModule(
#         ann_dir=Path("data/raw/solar/annotations/xmls"),
#         img_dir=Path("data/raw/solar/images"),
#         seed=42,
#         split=0.8,
#         batch_size=32,
#         num_workers=0,
#         pin_memory=False,
#     )
#     dm.setup()
#     for batch in dm.train_dataloader():
#         images, targets = batch
#         for i in range(len(images)):
#             img = images[i].permute(1, 2, 0).numpy()
#             bbs = targets[i].numpy()
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#             img = img.copy()
#             for bb in bbs:
#                 cv2.rectangle(
#                     img,
#                     (bb[0], bb[1]),
#                     (bb[2], bb[3]),
#                     (0, 255, 0),
#                     2,
#                 )
#             cv2.imwrite("sample.jpg", img)
#             input()


# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
import torch
from torchvision.io import read_image
from torchvision.ops import nms


def get_bboxes(output, threshold=0.3, iou_threshold=0.1):
    # Get bounding box coordinates
    bboxes = output[..., :4].squeeze(0).detach().cpu()
    # Get objectness score
    objectness = output[..., 4].squeeze(0).detach().cpu()
    # Get class probabilities
    class_probs = output[..., 5:].squeeze(0).detach().cpu()
    # Get class labels
    class_labels = torch.argmax(class_probs, dim=-1)
    # Get class scores
    class_scores = torch.max(class_probs, dim=-1)[0]
    # Filter out low scoring boxes
    mask = objectness > threshold
    bboxes, class_labels, class_scores = (
        bboxes[mask],
        class_labels[mask],
        class_scores[mask],
    )
    # Convert bounding boxes to format (x1, y1, x2, y2)
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # x1 = center_x - width / 2
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # y1 = center_y - height / 2
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # x2 = x1 + width
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # y2 = y1 + height
    # Apply non-maximum suppression
    keep = nms(bboxes, class_scores, iou_threshold)
    bboxes, class_labels, class_scores = (
        bboxes[keep].numpy(),
        class_labels[keep].numpy(),
        class_scores[keep].numpy(),
    )
    # Return list of bounding boxes
    return [
        {"bbox": bbox, "class_label": class_label, "class_score": class_score}
        for bbox, class_label, class_score in zip(bboxes, class_labels, class_scores)
    ]


# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

img = read_image("data/raw/solar/images/solar_4.JPG")
# img = read_image("stop.jpg")
# resize image to 512x512
img = img.unsqueeze(0)
img = torch.nn.functional.interpolate(img, size=512)
# convert to float and normalize
img = img.float() / 255.0
out = model(img)

# Convert the image tensor to numpy and transpose it to HWC format
img_np = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
# img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
img_np = img_np.copy()

# Convert the image from [0, 1] range to [0, 255] range and convert to uint8
img_np = (img_np * 255).astype(np.uint8)

# Draw the bounding boxes
for bbox in get_bboxes(out):
    print(bbox)
    # Get the bounding box coordinates
    x1, y1, x2, y2 = bbox["bbox"].astype(int)
    # Draw the rectangle
    cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Draw the class label and score
    label = f"{bbox['class_label']}: {bbox['class_score']:.2f}"
    cv2.putText(
        img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
    )

cv2.imwrite("output.jpg", img_np)
