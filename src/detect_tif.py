from pathlib import Path

import cv2
import geopandas as gpd
import lightning as L
import numpy as np
import shapely
import torch
import yaml
from dotenv import load_dotenv
from torch import nn
from tqdm import tqdm

from dataset.unsupervised_vrt_solar_datamodule import UnsupervisedVRTSolarDataModule
from model.deeplabv3 import DeepLabV3


def predict_batch(
    model: nn.Module,
    batch: torch.Tensor,
    threshold: float = 0.5,
) -> list[np.ndarray]:
    """Predict masks for a batch of tiles.

    Args:
        model: PyTorch model.
        batch: Batch of tiles.
        threshold: Confidence threshold.

    Returns:
        Predicted masks.
    """
    with torch.no_grad():
        outputs: torch.Tensor = model(batch)
        outputs = outputs > threshold
        outputs = outputs.to(torch.uint8) * 255
        return outputs.detach().cpu().squeeze(dim=1).numpy()


def detect_tif(
    tile_size: int,
    tif_path: Path,
    commune_name: str,
    commune_x_ratio: float,
    commune_y_ratio: float,
    dm_params: dict,
    model: nn.Module,
    out_path: Path,
) -> None:
    """Detect solar panels in a GeoTIFF image and save the results to a GeoPackage file.

    Args:
        tile_size (int): Size of the tiles.
        tif_path (Path): Path to the GeoTIFF image.
        commune_name (str): Name of the commune.
        commune_x_ratio (float): Ratio for the commune x axis.
        commune_y_ratio (float): Ratio for the commune y axis.
        dm_params (dict): Parameters for the DataModule.
        model (nn.Module): PyTorch model.
        out_path (Path): Path to save the results.
    """

    dm = UnsupervisedVRTSolarDataModule(
        tile_size=tile_size,
        tif_path=tif_path,
        commune_name=commune_name,
        commune_x_ratio=commune_x_ratio,
        commune_y_ratio=commune_y_ratio,
        **dm_params,
    )
    dm.setup()

    polys = []
    for batch, metas in tqdm(dm.val_dataloader(), desc="Predicting"):
        masks = predict_batch(model, batch)
        for mask, meta in zip(masks, metas):
            mask = cv2.resize(mask, (tile_size, tile_size))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if contour.shape[0] >= 4 and cv2.contourArea(contour) > 0:
                    pts = contour.reshape(-1, 2)
                    pts = np.apply_along_axis(lambda x: meta["transform"] * x, 1, pts)
                    polys.append(shapely.geometry.Polygon(pts))

    lake_pred_polygons = gpd.GeoDataFrame(geometry=polys, crs=dm.val_dataset.get_crs())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lake_pred_polygons.to_file(out_path, driver="GPKG")

    del dm  # Close the dataset


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

    model = DeepLabV3.load_from_checkpoint("out/train/model.ckpt")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        model = torch.nn.DataParallel(model)
        model.to(device)
    model.eval()

    detect_tif(
        tile_size=prepare_params["tile_size"],
        tif_path=tif_path,
        commune_x_ratio=prepare_params["commune_x_ratio"],
        commune_y_ratio=prepare_params["commune_y_ratio"],
        commune_name=prepare_params["commune_name"],
        dm_params=dict(**datamodule_params, num_workers=6, pin_memory=True),
        model=model,
        out_path=Path("out/detect/nyon_solar.gpkg"),
    )


if __name__ == "__main__":
    main()
