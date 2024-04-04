import json
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms.v2 import functional as F


def create_xml_annotation(
    image_path: Path, boxes: list[torch.Tensor], output_dir: Path
) -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = image_path.name
    if boxes:
        for box in boxes:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = "solar"
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(box[0])
            ET.SubElement(bndbox, "ymin").text = str(box[1])
            ET.SubElement(bndbox, "xmax").text = str(box[2])
            ET.SubElement(bndbox, "ymax").text = str(box[3])
    tree = ET.ElementTree(root)
    xml_out_path = output_dir / (image_path.stem + ".xml")
    tree.write(xml_out_path)


def get_bboxes_from_labels(labels: list[dict]) -> list[list[int]]:
    # label = {
    #     "x": 91.82068355586543,
    #     "y": 85.74240453746339,
    #     "width": 4.025513972711175,
    #     "height": 5.344855420815499,
    #     "rotation": 0,
    #     "original_width": 1024,
    #     "original_height": 1024
    #   },
    # where x, y, width, height are in percentage
    bboxes = []
    for label in labels:
        x = label["x"] / 100 * label["original_width"]
        y = label["y"] / 100 * label["original_height"]
        width = label["width"] / 100 * label["original_width"]
        height = label["height"] / 100 * label["original_height"]
        bb = [x, y, x + width, y + height]
        bboxes.append(list(map(round, bb)))
    return bboxes


def get_bboxes_from_mask(mask: np.ndarray, size_thresh=0.01) -> list[list[int]]:
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w / mask.shape[1] > size_thresh and h / mask.shape[0] > size_thresh:
            bboxes.append([x, y, x + w, y + h])
    return bboxes


def prepare(root_raw_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract the zip files
    with zipfile.ZipFile(root_raw_dir / "solar.zip", "r") as zip_ref:
        zip_ref.extractall(output_dir / "solar")
    with zipfile.ZipFile(root_raw_dir / "PV01.zip", "r") as zip_ref:
        zip_ref.extractall(output_dir / "PV01")

    image_dir = output_dir / "images"
    ann_dir = output_dir / "annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Solar Dataset
    shutil.copytree(output_dir / "solar/images/", image_dir, dirs_exist_ok=True)
    shutil.copytree(output_dir / "solar/annotations/xmls/", ann_dir, dirs_exist_ok=True)

    # PV01 Dataset
    # Iterate over the images and create xml annotations bounding boxes
    for image_path in (output_dir / "PV01").rglob("*.bmp"):
        if not image_path.name.endswith("_label.bmp"):
            image = Image.open(image_path)
            image_out_path = image_dir / image_path.with_suffix(".png").name
            image.save(image_out_path)

            mask_path = image_path.parent / (image_path.stem + "_label.bmp")
            mask = Image.open(mask_path).convert("L")
            boxes = get_bboxes_from_mask(np.array(mask))
            create_xml_annotation(image_out_path, boxes, ann_dir)

    # NyonTiles Dataset
    with open(root_raw_dir / "nyon_tiles/annotations.json", "r") as f:
        annotations = json.load(f)
    for ann in annotations:
        image_name = ann["image"].split("/")[-1]
        image_path = root_raw_dir / "nyon_tiles/images" / image_name
        boxes = []
        if ann.get("label"):
            boxes = get_bboxes_from_labels(ann["label"])
        image_out_path = image_dir / image_name
        create_xml_annotation(image_out_path, boxes, ann_dir)
        shutil.copy(image_path, image_dir)
    # Delete the extracted files
    shutil.rmtree(output_dir / "solar")
    shutil.rmtree(output_dir / "PV01")


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    output_dir = params["datamodule"]["data_root"]
    prepare(root_raw_dir=Path("data/raw"), output_dir=Path(output_dir))


if __name__ == "__main__":
    main()
