import multiprocessing
import os
import tempfile
from pathlib import Path

import boto3
import geopandas as gpd
import rasterio
import rasterio.windows
from PIL import Image
from rasterio.plot import reshape_as_image
from tqdm import tqdm

import utils


def get_bounds(commune_name: str) -> gpd.GeoDataFrame:
    """
    Extracts the bounds of a commune from the swissBOUNDARIES3D dataset from a
    given commune name.

    Args:
        commune_name: Name of the commune.
    """
    commune_boundries = gpd.read_file(
        "data/raw/swissBOUNDARIES3D_1_5_LV95_LN02.gpkg", layer="tlm_hoheitsgebiet"
    )
    for _, row in commune_boundries.iterrows():
        if row["name"] == commune_name:
            return gpd.GeoDataFrame(
                geometry=[row["geometry"]], crs=commune_boundries.crs
            )
    raise ValueError(f"Commune {commune_name} not found in the dataset")


def crop_and_save_tile_as_png(
    src: rasterio.DatasetReader,
    window: rasterio.windows.Window,
    s3: boto3.resource,
    bucket: str,
    tiles_dest_path: Path,
) -> None:
    """
    Crops the data from the rasterio dataset reader, stores it as a temporary
    PNG file and uploads it to S3.

    Args:
        src: rasterio dataset reader.
        window: window to crop.
        s3: boto3 resource.
        bucket: S3 bucket.
        tiles_dest_path: destination path for the tile.
    """
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        cropped_data = src.read(window=window)
        Image.fromarray(reshape_as_image(cropped_data)).save(
            tmp.name, format="PNG", optimize=True, quality=90
        )
        utils.s3.upload_file(
            s3=s3,
            file_path=tmp.name,
            bucket=bucket,
            dest_key=str(tiles_dest_path),
        )


def get_window_from_geometry(
    src: rasterio.DatasetReader,
    gdf_bounds: gpd.GeoDataFrame,
    x_ratio: float = 1.0,
    y_ratio: float = 1.0,
) -> rasterio.windows.Window:
    """
    Get the window from the bounds of a GeoDataFrame.

    Args:
        src: rasterio dataset reader.
        gdf_bounds: GeoDataFrame with the bounds.
        x_ratio: Ratio for the x axis.
        y_ratio: Ratio for the y axis.
    """
    gdf = gdf_bounds.to_crs(src.crs)
    geometry = gdf.geometry.unary_union
    # Create window for the geometry bounds
    window = src.window(*geometry.bounds)
    col_off = round(window.col_off)
    row_off = round(window.row_off)
    width = round(window.width * x_ratio)
    height = round(window.height * y_ratio)
    return rasterio.windows.Window(col_off, row_off, width, height)


def tile_tif(
    src_path: Path,
    tiles_dest_dir: Path,
    prefix: str,
    s3: boto3.resource,
    bucket: str,
    gdf_bounds: gpd.GeoDataFrame,
    x_ratio: float,
    y_ratio: float,
    tile_size: int,
) -> None:
    """
    Tiles a tif file based on the bounds of a GeoDataFrame and uploads the tiles
    to S3.

    Args:
        src_path: Path to the tif file.
        tiles_dest_dir: Destination directory for the tiles.
        s3: boto3 resource.
        bucket: S3 bucket.
        gdf_bounds: GeoDataFrame with the bounds.
        x_ratio: Ratio for the x axis.
        y_ratio: Ratio for the y axis.
        tile_size: Size of the tiles.
    """
    # TODO: Tiling the tif file is currently not multiprocessing-friendly because
    #       of src and s3 not being pickleable. There is an opportunity to
    #       parallelize this process.
    with rasterio.open(src_path) as src:
        window = get_window_from_geometry(src, gdf_bounds, x_ratio, y_ratio)
        col_off = window.col_off
        row_off = window.row_off
        width = window.width
        height = window.height
        col_end = col_off + width
        row_end = row_off + height

        total_tiles = (width // tile_size) * (height // tile_size)
        with tqdm(total=total_tiles, desc="Producing tiles") as pbar:
            for y in range(row_off, row_end, tile_size):
                for x in range(col_off, col_end, tile_size):
                    window = rasterio.windows.Window(x, y, tile_size, tile_size)
                    crop_and_save_tile_as_png(
                        src=src,
                        window=window,
                        s3=s3,
                        bucket=bucket,
                        tiles_dest_path=tiles_dest_dir / f"{prefix}_{x}_{y}.png",
                    )
                    pbar.update()


def extract_tiles(
    src_bucket: str,
    s3_src_vrt_path: Path,
    s3_dest_prepared_path: Path,
    dest_bucket: str,
    commune_name: str,
    commune_x_ratio: int,
    commune_y_ratio: int,
    tile_size: int,
) -> None:
    # 1. Extract the commune bounds
    print("[INFO] Extracting the commune bounds")
    gdf_bounds = get_bounds(commune_name=commune_name)

    # 2. Tile the tif file
    print("[INFO] Tiling the tif file")
    s3 = boto3.resource(
        "s3",
        endpoint_url="https://" + os.environ["AWS_S3_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    tile_tif(
        src_path=Path("/vsis3_streaming") / src_bucket / s3_src_vrt_path,
        tiles_dest_dir=s3_dest_prepared_path / "images",
        prefix=f'{src_bucket.replace(".", "")}_{commune_name.lower()}',
        s3=s3,
        bucket=dest_bucket,
        gdf_bounds=gdf_bounds,
        x_ratio=commune_x_ratio,
        y_ratio=commune_y_ratio,
        tile_size=tile_size,
    )
