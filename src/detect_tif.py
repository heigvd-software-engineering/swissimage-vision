from multiprocessing import Process, Queue
from pathlib import Path

import geopandas as gpd
import lightning as L
import numpy as np
import rasterio
import shapely
import torch
import yaml
from numpy.lib.stride_tricks import sliding_window_view
from torchvision.transforms.v2 import functional as F
from tqdm import tqdm

from model.fasterrcnn import FasterRCNN


def predict_batch(
    model: FasterRCNN,
    device: str,
    tile_size: int,
    image_size: int,
    batch: np.ndarray,
    threshold: float = 0.9,
) -> list[np.ndarray]:
    """Predict bounding boxes for a batch of tiles.

    Args:
        model: FasterRCNN model.
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
    scale = tile_size / image_size

    with torch.no_grad():
        outputs = model(batch_tensor)
        batch_boxes = []
        for output in outputs:
            boxes: torch.FloatTensor = output["boxes"]
            boxes = boxes[output["scores"] > threshold]
            boxes *= scale
            batch_boxes.append(boxes.cpu().numpy())
        return batch_boxes


def producer(
    queue: Queue,
    total_tiles: int,
    img_arr: np.ndarray,
    batch_size: int,
    tile_size: int,
    overlap_size: int,
) -> None:
    """Produce batches of tiles to be processed by the consumer.

    Args:
        queue: Queue to put the batches.
        total_tiles: Total number of tiles.
        img_arr: Image array.
        batch_size: Batch size.
        tile_size: Size of the tiles.
        overlap_size: Size of the overlap.
    """
    pbar = tqdm(
        total=total_tiles,
        desc="Producing batches",
        position=1,
        leave=False,
    )
    tiles = sliding_window_view(img_arr, (tile_size, tile_size, 3))[
        :: tile_size - overlap_size,
        :: tile_size - overlap_size,
        :,
    ]
    tiles = np.squeeze(tiles, axis=2)

    batch = []
    batch_coordinates = []
    for row_idx, row in enumerate(tiles):
        for col_idx, tile in enumerate(row):
            batch.append(tile.transpose(2, 0, 1))
            batch_coordinates.append((row_idx, col_idx))

            if len(batch) == batch_size:
                queue.put((np.array(batch), batch_coordinates))
                batch = []
                batch_coordinates = []
                pbar.update(batch_size)

    # Put remaining batch if it's not empty
    if batch:
        queue.put((np.array(batch), batch_coordinates))

    # Signal that the producer is done
    queue.put(None)
    pbar.close()


def consumer(
    queue: Queue,
    total_tiles: int,
    model: FasterRCNN,
    device: str,
    batch_size: int,
    tile_size: int,
    image_size: int,
    overlap_size: int,
    crs: str,
    transform: rasterio.Affine,
    out_path: Path,
) -> None:
    """Consume batches of tiles and predict bounding boxes.

    Args:
        queue: Queue to get the batches.
        total_tiles: Total number of tiles.
        model: FasterRCNN model.
        device: Device to run the model on.
        batch_size: Batch size.
        tile_size: Size of the tiles.
        image_size: Size of the images.
        overlap_size: Size of the overlap.
        crs: Coordinate reference system.
        transform: Affine transformation matrix.
        out_path: Path to save the results.
    """
    pbar = tqdm(
        total=total_tiles // batch_size,
        desc="Processing batches",
        position=0,
        leave=False,
    )
    geometries = []
    while True:
        item = queue.get()
        if item is None:
            break

        batch, batch_coordinates = item
        batch_boxes = predict_batch(
            model=model,
            device=device,
            tile_size=tile_size,
            image_size=image_size,
            batch=batch,
        )
        for boxes, (row_idx, col_idx) in zip(batch_boxes, batch_coordinates):
            boxes[:, 0] += col_idx * (tile_size - overlap_size)
            boxes[:, 1] += row_idx * (tile_size - overlap_size)
            boxes[:, 2] += col_idx * (tile_size - overlap_size)
            boxes[:, 3] += row_idx * (tile_size - overlap_size)
            # Transform the box coordinates from pixel to spatial coordinates
            for i in range(boxes.shape[0]):
                xmin, ymin = transform * (boxes[i, 0], boxes[i, 1])
                xmax, ymax = transform * (boxes[i, 2], boxes[i, 3])
                geometries.append(shapely.box(xmin, ymin, xmax, ymax, ccw=False))
        pbar.update()

    gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)
    gdf.to_file(str(out_path), driver="GPKG")
    pbar.close()


def detect_tif(
    model: FasterRCNN,
    device: str,
    tile_size: int,
    batch_size: int,
    image_size: int,
    tif_path: Path,
    out_path: Path,
) -> None:
    """Detect solar panels in a GeoTIFF image and save the results to a GeoPackage file.

    Args:
        model: FasterRCNN model.
        device: Device to run the model on.
        batch_size: Batch size.
        image_size: Size of the images.
        tif_path: Path to the GeoTIFF image.
    """
    with rasterio.open(tif_path) as image:
        crs = str(image.crs)
        transform = image.transform
        img_arr = image.read().transpose(1, 2, 0)

    img_arr = np.float32(img_arr) / 255.0

    overlap_cnt = 0
    overlap_size = int(tile_size * 1 / (overlap_cnt + 1)) if overlap_cnt > 0 else 0
    total_tiles = ((img_arr.shape[0] - tile_size) // (tile_size - overlap_size) + 1) * (
        (img_arr.shape[1] - tile_size) // (tile_size - overlap_size) + 1
    )

    queue = Queue(maxsize=10)
    prod = Process(
        target=producer,
        kwargs=dict(
            queue=queue,
            total_tiles=total_tiles,
            img_arr=img_arr,
            batch_size=batch_size,
            tile_size=tile_size,
            overlap_size=overlap_size,
        ),
        daemon=True,
    )

    prod.start()

    consumer(
        queue=queue,
        total_tiles=total_tiles,
        model=model,
        device=device,
        batch_size=batch_size,
        tile_size=tile_size,
        image_size=image_size,
        overlap_size=overlap_size,
        crs=crs,
        transform=transform,
        out_path=out_path,
    )

    prod.join()


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    prepare_params = params["prepare"]
    train_params = params["train"]
    datamodule_params = params["datamodule"]["setup"]

    L.seed_everything(train_params["model"]["seed"])
    model = FasterRCNN.load_from_checkpoint("out/model.ckpt")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        model = torch.nn.DataParallel(model)
        model.to(device)
    model.eval()

    detect_tif(
        model=model,
        device=device,
        src_bucket=prepare_params["src_bucket"],
        s3_src_vrt_path=prepare_params["s3_src_vrt_path"],
        tile_size=prepare_params["tile_size"],
        batch_size=datamodule_params["batch_size"],
        image_size=datamodule_params["image_size"],
        out_path=Path("out/nyon_solar.gpkg"),
    )


if __name__ == "__main__":
    main()
