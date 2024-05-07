from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import geopandas as gpd
import lightning as L
import numpy as np
import rasterio
import torch
import yaml
from dotenv import load_dotenv
from rasterio.plot import reshape_as_image
from torch import nn
from torchvision.transforms.v2 import functional as F
from tqdm import tqdm

from model.deeplabv3 import DeepLabV3
from utils.extract_tiles import get_bounds, get_window_from_geometry


def predict_batch(
    model: nn.Module,
    device: str,
    image_size: int,
    batch: np.ndarray,
    threshold: float = 0.5,
) -> list[np.ndarray]:
    """Predict bounding boxes for a batch of tiles.

    Args:
        model: PyTorch model.
        device: Device to run the model on.
        tile_size: Size of the tiles.
        image_size: Size of the images.
        batch: Batch of tiles.
        threshold: Confidence threshold.

    Returns:
        Bounding boxes.
    """
    batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
    batch_tensor = F.resize(batch_tensor, [image_size, image_size])

    with torch.no_grad():
        outputs: torch.Tensor = model(batch_tensor)
        outputs = outputs > threshold
        outputs = outputs.to(torch.uint8) * 255
        return outputs.detach().cpu().squeeze(dim=1).numpy()


def producer(
    queue: Queue,
    tif_path: str,
    gdf_bounds: gpd.GeoDataFrame,
    commune_x_ratio: float,
    commune_y_ratio: float,
    total_tiles: int,
    batch_size: int,
    tile_size: int,
) -> None:
    """Produce batches of tiles to be processed by the consumer.

    Args:
        queue: Queue to put the batches.
        tif_path: Path to the GeoTIFF image.
        gdg_bounds: GeoDataFrame with the bounds.
        commune_x_ratio: Ratio for the commune x axis.
        commune_y_ratio: Ratio for the commune y axis.
        total_tiles: Total number of tiles.
        batch_size: Batch size.
        tile_size: Size of the tiles.
    """
    batch = []
    batch_coordinates = []

    with rasterio.open(tif_path) as image:
        window = get_window_from_geometry(
            image, gdf_bounds, commune_x_ratio, commune_y_ratio
        )
        col_off = window.col_off
        row_off = window.row_off
        width = window.width
        height = window.height
        col_end = col_off + width
        row_end = row_off + height

        with tqdm(total=total_tiles, desc="Producing tiles", position=1) as pbar:
            for y in range(row_off, row_end, tile_size):
                for x in range(col_off, col_end, tile_size):
                    window = rasterio.windows.Window(x, y, tile_size, tile_size)
                    tile = reshape_as_image(image.read(window=window))
                    batch.append(tile.transpose(2, 0, 1))
                    batch_coordinates.append((x, y))

                    if len(batch) == batch_size:
                        queue.put((np.array(batch), batch_coordinates))
                        batch = []
                        batch_coordinates = []

                    pbar.update()

    # Put remaining batch if it's not empty
    if batch:
        queue.put((np.array(batch), batch_coordinates))

    # Signal that the producer is done
    queue.put(None)


def consumer(
    queue: Queue,
    tif_col_off: int,
    tif_row_off: int,
    tif_width: int,
    tif_height: int,
    model: nn.Module,
    device: str,
    tile_size: int,
    image_size: int,
    # crs: str,
    # transform: rasterio.Affine,
    out_path: Path,
) -> None:
    """Consume batches of tiles and predict bounding boxes.

    Args:
        queue: Queue to get the batches.
        model: PyTorch model.
        device: Device to run the model on.
        tile_size: Size of the tiles.
        image_size: Size of the images.
        crs: Coordinate reference system.
        transform: Affine transformation matrix.
        out_path: Path to save the results.
    """
    pbar = tqdm(
        desc="Processing batches",
        position=0,
        leave=False,
    )
    mask_image = np.empty((tif_height, tif_width), dtype=np.uint8)
    while True:
        item = queue.get()
        if item is None:
            break

        batch, batch_coordinates = item
        batch_preds = predict_batch(
            model=model,
            device=device,
            image_size=image_size,
            batch=batch,
        )
        for mask, (x, y) in zip(batch_preds, batch_coordinates):
            x_start = x - tif_col_off
            y_start = y - tif_row_off
            mask = cv2.resize(mask, (tile_size, tile_size))
            print(np.max(mask))
            mask_image[y_start : y_start + tile_size, x_start : x_start + tile_size] = (
                mask
            )
        pbar.update()

    # gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)

    # out_path.parent.mkdir(parents=True, exist_ok=True)

    # gdf.to_file(str(out_path), driver="GPKG")
    pbar.close()


def detect_tif(
    model: nn.Module,
    device: str,
    tif_path: Path,
    gdf_bounds: gpd.GeoDataFrame,
    commune_x_ratio: float,
    commune_y_ratio: float,
    tile_size: int,
    batch_size: int,
    image_size: int,
    out_path: Path,
) -> None:
    """Detect solar panels in a GeoTIFF image and save the results to a GeoPackage file.

    Args:
        model: PyTorch model.
        device: Device to run the model on.
        batch_size: Batch size.
        image_size: Size of the images.
        tif_path: Path to the GeoTIFF image.
    """
    with rasterio.open(tif_path) as image:
        crs = str(image.crs)
        window = get_window_from_geometry(
            image, gdf_bounds, commune_x_ratio, commune_y_ratio
        )
        col_off = window.col_off
        row_off = window.row_off
        width = window.width
        height = window.height

    total_tiles = (width // tile_size) * (height // tile_size)

    queue = Queue(maxsize=10)
    prod = Process(
        target=producer,
        kwargs=dict(
            queue=queue,
            tif_path=tif_path,
            gdf_bounds=gdf_bounds,
            commune_x_ratio=commune_x_ratio,
            commune_y_ratio=commune_y_ratio,
            total_tiles=total_tiles,
            batch_size=batch_size,
            tile_size=tile_size,
        ),
        daemon=True,
    )

    prod.start()

    consumer(
        queue=queue,
        tif_col_off=col_off,
        tif_row_off=row_off,
        tif_width=width,
        tif_height=height,
        model=model,
        device=device,
        tile_size=tile_size,
        image_size=image_size,
        # crs=crs,
        # transform=transform,
        out_path=out_path,
    )

    prod.join()


def main() -> None:
    load_dotenv(override=True)

    params = yaml.safe_load(open("params.yaml"))
    train_params = params["train"]
    datamodule_params = train_params["datamodule"]["setup"]
    prepare_params = params["prepare"]
    L.seed_everything(train_params["model"]["seed"])

    src_bucket = prepare_params["src_bucket"]
    s3_src_vrt_path = prepare_params["s3_src_vrt_path"]
    tif_path = Path("/vsis3_streaming") / src_bucket / s3_src_vrt_path

    commune_name = prepare_params["commune_name"]
    commune_x_ratio = prepare_params["commune_x_ratio"]
    commune_y_ratio = prepare_params["commune_y_ratio"]
    gdf_bounds = get_bounds(commune_name=commune_name)

    model = DeepLabV3.load_from_checkpoint("out/train/model.ckpt")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        model = torch.nn.DataParallel(model)
        model.to(device)
    model.eval()

    detect_tif(
        model=model,
        device=device,
        tif_path=tif_path,
        gdf_bounds=gdf_bounds,
        commune_x_ratio=commune_x_ratio,
        commune_y_ratio=commune_y_ratio,
        tile_size=prepare_params["tile_size"],
        batch_size=datamodule_params["batch_size"],
        image_size=datamodule_params["image_size"],
        out_path=Path("out/detect/nyon_solar.gpkg"),
    )


if __name__ == "__main__":
    main()
