from pathlib import Path

import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/files.txt")
async def files():
    images = (DATA_ROOT / "images").glob("*.png")
    urls = [f"http://{HOST}:{PORT}/{image}" for image in images][:10]
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
