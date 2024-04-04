import multiprocessing
import os
import shutil
from pathlib import Path

import boto3
import geopandas as gpd
import numpy as np
import rasterio
import yaml
from dotenv import load_dotenv
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
from tqdm import tqdm

import utils


def get_bounds(commune_name: str) -> gpd.GeoDataFrame:
    commune_boundries = gpd.read_file(
        "data/raw/swissBOUNDARIES3D_1_5_LV95_LN02.gpkg", layer="tlm_hoheitsgebiet"
    )
    for _, row in commune_boundries.iterrows():
        if row["name"] == commune_name:
            return gpd.GeoDataFrame(
                geometry=[row["geometry"]], crs=commune_boundries.crs
            )
    raise ValueError(f"Commune {commune_name} not found in the dataset")


def save_tif_from_bounds(
    src_path: Path,
    out_path: Path,
    gdf_bounds: gpd.GeoDataFrame,
    x_ratio: float,
    y_ratio: float,
) -> None:
    with rasterio.open(src_path) as src:
        gdf = gdf_bounds.to_crs(str(src.crs))
        geometry = gdf.geometry.unary_union
        # create window for the geometry bounds
        window = src.window(*geometry.bounds)
        # new window with 2/3 of width
        window = rasterio.windows.Window(
            window.col_off,
            window.row_off,
            window.width * x_ratio,
            window.height * y_ratio,
        )
        # Read the data in the window
        cropped_data = src.read(window=window)

        # Define the transform for the cropped data
        transform = src.window_transform(window)

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            width=cropped_data.shape[2],
            height=cropped_data.shape[1],
            transform=transform,
        )
        # Create a new TIFF file and write the cropped data into it
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(cropped_data)


def save_tile(
    tile: np.ndarray,
    row_idx: int,
    col_idx: int,
    out_dir: Path,
) -> None:
    image = Image.fromarray(tile)
    image.save(out_dir / f"tile_{row_idx}_{col_idx}.png", optimize=True, quality=90)


def tile_tif(
    tif_path: Path,
    tile_size: int,
    overlap_size: int,
    processes: int,
    out_dir: Path,
) -> None:
    with rasterio.open(tif_path) as image:
        img_arr = image.read().transpose(1, 2, 0)

    total_tiles = ((img_arr.shape[0] - tile_size) // (tile_size - overlap_size) + 1) * (
        (img_arr.shape[1] - tile_size) // (tile_size - overlap_size) + 1
    )

    pbar = tqdm(
        total=total_tiles,
        desc="Producing tiles",
        leave=False,
    )

    tiles = sliding_window_view(img_arr, (tile_size, tile_size, 3))[
        :: tile_size - overlap_size,
        :: tile_size - overlap_size,
        :,
    ]
    tiles = np.squeeze(tiles, axis=2)

    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    with multiprocessing.Pool(processes=processes) as pool:
        for row_idx, row in enumerate(tiles):
            for col_idx, tile in enumerate(row):
                pool.apply_async(
                    save_tile,
                    kwds=dict(
                        tile=tile,
                        row_idx=row_idx,
                        col_idx=col_idx,
                        out_dir=out_dir,
                    ),
                    callback=lambda _: pbar.update(),
                )

        pool.close()
        pool.join()

    pbar.close()


def main() -> None:
    load_dotenv(override=True)
    params = yaml.safe_load(open("params.yaml"))
    extract_tiles_params = params["extract_tiles"]
    out_dir = Path("data/extracted")
    bucket = "swissimage-vision"

    # 1. Extract the commune bounds
    print("[INFO] Extracting the commune bounds")
    gdf_bounds = get_bounds(commune_name=extract_tiles_params["commune_name"])

    # 2. Save a cropped tif file from the bounds
    print("[INFO] Saving cropped tif file")
    tif_out_path = out_dir / f"{extract_tiles_params["commune_name"].lower()}.tif"
    save_tif_from_bounds(
        src_path=Path("/vsis3_streaming/swissimage-0.1m/files.vrt"),
        out_path=tif_out_path,
        gdf_bounds=gdf_bounds,
        x_ratio=extract_tiles_params["x_ratio"],
        y_ratio=extract_tiles_params["y_ratio"],
    )
    # 3. Tile the tif file
    print("[INFO] Extracting tiles")
    tiles_out_dir = out_dir / "tiles"
    tile_tif(
        tif_path=tif_out_path,
        tile_size=extract_tiles_params["tile_size"],
        overlap_size=extract_tiles_params["overlap_size"],
        processes=10,
        out_dir=tiles_out_dir,
    )
    # 4. Upload the tiles to S3 for later labelling
    print("[INFO] Uploading tiles to S3")
    s3 = boto3.resource(
        "s3",
        endpoint_url="https://" + os.environ["AWS_S3_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    utils.s3.delete_files(s3, bucket=bucket, folder=Path("data/tiles"))
    utils.s3.upload_files(
        s3,
        path=tiles_out_dir,
        bucket=bucket,
        dest_folder=Path("data/tiles"),
    )
    utils.s3.upload_file(
        s3,
        file_path=tif_out_path,
        bucket=bucket,
        dest_key=f"data/{tif_out_path.name}",
    )
    # Delete the tif file as we do not use it and it would slow down dvc
    tif_out_path.unlink()


if __name__ == "__main__":
    main()
