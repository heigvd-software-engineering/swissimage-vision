import json
import multiprocessing
from functools import partial
from pathlib import Path

import yaml
from dotenv import load_dotenv

import utils


def parse_annotation(ann_key: str, bucket: str) -> dict:
    s3 = utils.s3.get_s3_resource()
    file = utils.s3.get_file(s3, bucket, ann_key)
    data = json.load(file)
    ann = {}
    if data["task"]["is_labeled"]:
        ann["image"] = data["task"]["data"]["image_url"]
        boxes = []
        if data["result"]:
            for result in data["result"]:
                img_width = result["original_width"]
                img_height = result["original_height"]
                # We convert the relative coordinates to absolute pixel values
                x = result["value"]["x"] / 100 * img_width
                y = result["value"]["y"] / 100 * img_height
                width = result["value"]["width"] / 100 * img_width
                height = result["value"]["height"] / 100 * img_height
                # We store the bounding box coordinates as a list of lists
                # [xmin, ymin, xmax, ymax]
                boxes.append([x, y, x + width, y + height])
        ann["boxes"] = boxes
    return ann


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    prepare_params = params["prepare"]
    load_dotenv(override=True)
    s3 = utils.s3.get_s3_resource()
    annotations = utils.s3.list_files(
        s3,
        params["bucket"],
        Path(prepare_params["s3_dest_prepared_path"]) / "annotations",
    )
    parsed_annotations = []
    with multiprocessing.Pool(processes=10) as pool:
        parsed_annotations = pool.map(
            partial(parse_annotation, bucket=params["bucket"]), annotations
        )
    # We sort to ensure that race conditions don't affect the order of the annotations and
    # therefore don't invalidate the DVC cache
    parsed_annotations = sorted(parsed_annotations, key=lambda x: x["image"])
    with open("data/preprocessed/annotations.json", "w") as f:
        json.dump(parsed_annotations, f)


if __name__ == "__main__":
    main()
