from __future__ import annotations

import json
from typing import Annotated

import bentoml
import yaml
from bentoml.validators import ContentType
from PIL.Image import Image
from pydantic import Field

model_name = yaml.safe_load(open("params.yaml"))["bentoml"]["model_name"]


class SolarDetectionRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    bento_model = bentoml.models.get(model_name)

    def __init__(self) -> None:
        self.model = self.bento_model.load_model()

    @bentoml.api()
    def predict(
        self,
        image: Annotated[Image, ContentType("image/jpeg")] = Field(
            description="Stallelite image to detect solar panels"
        ),
    ) -> Annotated[str, ContentType("application/json")]:
        predictions = self.model.predict(image)

        return json.dumps(self.postprocess(predictions))
