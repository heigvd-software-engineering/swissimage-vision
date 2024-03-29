import os

import matplotlib.pyplot as plt
import rasterio
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error
from osgeo import gdal, osr


def main():
    load_dotenv(override=True)
    # Create a MinIO client
    client = Minio(
        os.getenv("MINIO_URL"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=True,
    )

    try:
        # Get the VRT file from the bucket
        data = client.get_object("swissimage-0.1m", "files.vrt")
        data_bytes = data.read()

        # Create a virtual file in memory from the VRT data
        vrt_path = "/vsimem/files.vrt"
        gdal.FileFromMemBuffer(vrt_path, data_bytes)

        # Open the VRT dataset with GDAL
        ds = gdal.Open(vrt_path)

        # Load the commune boundary
        nyon_boundry = gdal.OpenEx("data/commune.gpkg", gdal.OF_VECTOR)
        # Get the commune boundary layer
        nyon_boundry_layer = nyon_boundry.GetLayer(0)
        # Save tif file for the intersection of the VRT and the commune boundary
        gdal.Warp(
            "data/nyon.tif",
            ds,
            cutlineDSName="data/commune.gpkg",
            cutlineLayer=nyon_boundry_layer.GetName(),
            cropToCutline=True,
        )

    except S3Error as err:
        print("S3 error: ", err)


if __name__ == "__main__":
    main()
