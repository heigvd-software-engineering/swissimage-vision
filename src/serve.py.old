import bentoml
import yaml
from bentoml.io import JSON
from bentoml.io import Image as BentoImage
from PIL import Image
from pydantic import BaseModel

params = yaml.safe_load(open("params.yaml"))
batch_size = params["datamodule"]["batch_size"]
model_name = params["export"]["model_name"]


class SolarDetectionRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    bento_model = bentoml.models.get(model_name)

    def __init__(self) -> None:
        self.preprocess = self.bento_model.custom_objects["preprocess"]
        self.postprocess = self.bento_model.custom_objects["postprocess"]
        self.model = self.bento_model.load_model().to("cuda")

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def inference(
        self, input_imgs: Image.Image | list[Image.Image]
    ) -> list[dict[str, list[float]]]:
        batch = self.preprocess(input_imgs).to("cuda")
        results = self.model(batch)
        return self.postprocess(results, [img.size for img in input_imgs])


solar_detection_runner = bentoml.Runner(
    SolarDetectionRunnable,
    name="solar_detection_runner",
    max_batch_size=batch_size,
    max_latency_ms=250,
)

svc = bentoml.Service("solar_detection_service", runners=[solar_detection_runner])


class ModelOputput(BaseModel):
    boxes: list[list[float]]
    scores: list[float]


class ModelOputputList(BaseModel):
    __root__: list[ModelOputput]


@svc.api(input=BentoImage(), output=JSON(pydantic_model=ModelOputput))
async def predict(input_img):
    batch_ret = await solar_detection_runner.inference.async_run([input_img])
    return batch_ret[0]
