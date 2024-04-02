import geopandas as gpd
import rasterio
from dotenv import load_dotenv
from rasterio.windows import Window

COMMUNE = "nyon"


def main():
    load_dotenv(override=True)

    # Open the GeoPackage file and get the geometry
    with rasterio.open("/vsis3_streaming/swissimage-0.1m/files.vrt") as src:
        gdf = gpd.read_file(f"data/raw/{COMMUNE}.gpkg")
        gdf = gdf.to_crs(str(src.crs))
        geometry = gdf.geometry.unary_union
        # create window for the geometry bounds
        window = src.window(*geometry.bounds)
        # new window with 2/3 of width
        window = Window(
            window.col_off, window.row_off, window.width * 2 / 3, window.height * 3 / 4
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
        with rasterio.open(f"data/raw/{COMMUNE}.tif", "w", **profile) as dst:
            dst.write(cropped_data)


if __name__ == "__main__":
    main()
