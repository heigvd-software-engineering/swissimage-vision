import base64
import io

import bentoml
import cv2
import geopandas as gpd
import numpy as np
import shapely
import yaml
from PIL import Image

params = yaml.safe_load(open("params.yaml"))
batch_size = params["train"]["datamodule"]["setup"]["batch_size"]
model_name = params["export"]["model_name"]


# TODO: Add adaptive batching
@bentoml.service(
    resources={"cpu": "10", "gpu": 1},
    traffic={"timeout": 10},
    http={
        "cors": {
            "enabled": True,
            "access_control_allow_origins": "*",
            "access_control_allow_credentials": True,
            "access_control_allow_methods": "*",
            "access_control_allow_headers": "*",
            "access_control_allow_origin_regex": None,
            "access_control_max_age": None,
            "access_control_expose_headers": "*",
        }
    },
)
class SolarSegmentationService:
    def __init__(self) -> None:
        self.bento_model = bentoml.models.get(model_name)
        self.preprocess = self.bento_model.custom_objects["preprocess"]
        self.postprocess = self.bento_model.custom_objects["postprocess"]
        self.model = self.bento_model.load_model().to("cuda")

    @bentoml.api
    def inference(self, images: Image.Image | list[Image.Image]) -> list[np.ndarray]:
        if not isinstance(images, list):
            images = [images]
        batch = self.preprocess(images).to("cuda")
        results = self.model(batch)
        return self.postprocess(results, [img.size for img in images])

    @bentoml.api
    def predict_poly(self, image: Image.Image | str) -> dict:
        if isinstance(image, str):
            # load image from dataurl
            bytes_str = base64.b64decode(image.split(",")[1])
            image = Image.open(io.BytesIO(bytes_str))
        image.save("tmp.png")
        mask = (self.inference(image))[0]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polys = []
        for contour in contours:
            if contour.shape[0] >= 4 and cv2.contourArea(contour) > 0:
                pts = contour.reshape(-1, 2)
                polygon = shapely.geometry.Polygon(pts)
                simplified_polygon = polygon.simplify(
                    tolerance=5, preserve_topology=False
                )
                polys.append(simplified_polygon)
        return gpd.GeoSeries(polys).__geo_interface__
