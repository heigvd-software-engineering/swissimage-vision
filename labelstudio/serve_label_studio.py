import io
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

HOST = "localhost"
PORT = 8081
DATA_ROOT = Path("data/raw/nyon_tiles")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.post("/setup")
async def setup(request: Request):
    data = await request.json()
    print(data)
    return JSONResponse({"status": "ok"})


@app.post("/webhook")
async def webhook(request: Request):
    # data = await request.json()
    return Response(status_code=200)


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    image_urls = [t["data"]["image"] for t in data["tasks"]]
    images_paths = [
        image_url.split(f"http://{HOST}:{PORT}/")[1] for image_url in image_urls
    ]
    images = [Image.open(image_path) for image_path in images_paths]
    predictions = []
    for image in images:
        byte_arr = io.BytesIO()
        image.save(byte_arr, format="PNG")
        byte_value = byte_arr.getvalue()
        res = requests.post(
            "http://localhost:3000/predict",
            headers={
                "Content-Type": "image/png",
                "accept": "application/json",
            },
            data=byte_value,
        )
        data = res.json()
        boxes = data["boxes"]
        results = []
        for box in boxes:
            x = box[0] * 100 / image.size[0]
            y = box[1] * 100 / image.size[1]
            width = (box[2] - box[0]) * 100 / image.size[0]
            height = (box[3] - box[1]) * 100 / image.size[1]
            results.append(
                {
                    "from_name": "label",
                    "source": "$image",
                    "to_name": "image",
                    "type": "rectangle",
                    "value": {
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "rotation": 0,
                    },
                }
            )
        predictions.append(
            {
                "model_version": "latest",
                "result": results,
            }
        )
    return JSONResponse(content={"results": predictions})


@app.get("/files.txt")
async def files():
    images = (DATA_ROOT / "images").glob("*.png")
    urls = [f"http://{HOST}:{PORT}/{image}" for image in images]
    urls_str = "\n".join(urls)
    return Response(urls_str, media_type="text/plain")


@app.get(f"/{DATA_ROOT}/images/" + "{image_name}")
async def images(image_name: str):
    image_path = DATA_ROOT / "images" / image_name
    return Response(image_path.read_bytes(), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(
        __name__ + ":app",
        host=HOST,
        port=PORT,
        reload=False,
        workers=6,
    )
