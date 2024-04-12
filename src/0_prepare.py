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


def prepare_solar_dataset(
    root_raw_dir: Path, out_dir: Path, img_dir: Path, ann_dir: Path
) -> None:
    with zipfile.ZipFile(root_raw_dir / "solar.zip", "r") as zip_ref:
        zip_ref.extractall(out_dir / "solar")

    shutil.copytree(out_dir / "solar/images/", img_dir, dirs_exist_ok=True)
    shutil.copytree(out_dir / "solar/annotations/xmls/", ann_dir, dirs_exist_ok=True)
    shutil.rmtree(out_dir / "solar")


def prepare_pv01_dataset(
    root_raw_dir: Path, out_dir: Path, img_dir: Path, ann_dir: Path
) -> None:
    with zipfile.ZipFile(root_raw_dir / "PV01.zip", "r") as zip_ref:
        zip_ref.extractall(out_dir / "PV01")

    # Iterate over the images and create xml annotations bounding boxes
    for image_path in (out_dir / "PV01").rglob("*.bmp"):
        if not image_path.name.endswith("_label.bmp"):
            image = Image.open(image_path)
            image_out_path = img_dir / image_path.with_suffix(".png").name
            image.save(image_out_path)

            mask_path = image_path.parent / (image_path.stem + "_label.bmp")
            mask = Image.open(mask_path).convert("L")
            boxes = get_bboxes_from_mask(np.array(mask))
            create_xml_annotation(image_out_path, boxes, ann_dir)
    shutil.rmtree(out_dir / "PV01")


def prepare_tiles_dataset(
    tiles_dir: Path, raw_ann_path: Path, img_dir: Path, ann_dir: Path
) -> None:
    if not raw_ann_path.exists():
        return

    with open(raw_ann_path, "r") as f:
        annotations = json.load(f)

    for ann in annotations:
        image_name = ann["image"].split("/")[-1]
        image_path = tiles_dir / image_name
        boxes = []
        if ann.get("label"):
            boxes = get_bboxes_from_labels(ann["label"])
        image_out_path = img_dir / image_name
        create_xml_annotation(image_out_path, boxes, ann_dir)
        shutil.copy(image_path, img_dir)


def prepare(root_raw_dir: Path, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_dir = out_dir / "images"
    ann_dir = out_dir / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    prepare_solar_dataset(root_raw_dir, out_dir, img_dir, ann_dir)
    prepare_pv01_dataset(root_raw_dir, out_dir, img_dir, ann_dir)
    prepare_tiles_dataset(
        tiles_dir=Path("data/extracted/tiles"),
        raw_ann_path=Path("data/labels/annotations.json"),
        img_dir=img_dir,
        ann_dir=ann_dir,
    )


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    out_dir = params["datamodule"]["data_root"]
    prepare(root_raw_dir=Path("data/raw"), out_dir=Path(out_dir))


if __name__ == "__main__":
    main()
