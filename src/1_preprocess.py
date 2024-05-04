import json
import multiprocessing
import random
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
        polys = []
        if data["result"]:
            for result in data["result"]:
                w = result["original_width"]
                h = result["original_height"]
                points = []
                for point in result["value"]["points"]:
                    points.append([point[0] / 100 * w, point[1] / 100 * h])
                polys.append(points)
        ann["polys"] = polys
    return ann


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    prepare_params = params["prepare"]
    seed = params["train"]["datamodule"]["setup"]["seed"]
    random.seed(seed)

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

    # Split limit to 50% annotations that do not have masks (polys = [])
    split_limit = int(len(parsed_annotations) * 0.5)
    no_polys = [ann for ann in parsed_annotations if not ann["polys"]]
    random.shuffle(no_polys)
    with_polys = [ann for ann in parsed_annotations if ann["polys"]]
    parsed_annotations = no_polys[:split_limit] + with_polys

    # We sort to ensure that race conditions don't affect the order of the annotations and
    # therefore don't invalidate the DVC cache
    parsed_annotations = sorted(parsed_annotations, key=lambda x: x["image"])

    print(f"[INFO] Total annotations: {len(parsed_annotations)}")

    out_dir = Path("out/preprocess")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save the parsed annotations to a JSON file
    with open(out_dir / "annotations.json", "w") as f:
        json.dump(parsed_annotations, f)


if __name__ == "__main__":
    main()
