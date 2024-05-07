import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.plot import reshape_as_image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F

from utils.extract_tiles import get_window_from_geometry


class UnsupervisedVRTSolarDataset(Dataset):
    def __init__(
        self,
        transform: T.Compose,
        tile_size: int,
        tif_path: str,
        gdf_bounds: gpd.GeoDataFrame,
        commune_x_ratio: float,
        commune_y_ratio: float,
    ) -> None:
        self.transform = transform
        self.tile_size = tile_size

        self.dataset = rasterio.open(tif_path)
        self.window = get_window_from_geometry(
            src=self.dataset,
            gdf_bounds=gdf_bounds,
            x_ratio=commune_x_ratio,
            y_ratio=commune_y_ratio,
        )

    def __len__(self) -> int:
        return (self.window.width // self.tile_size) * (
            self.window.height // self.tile_size
        )

    def get_crs(self) -> dict:
        return self.dataset.crs

    def _index_to_window(
        self, idx: int
    ) -> tuple[rasterio.windows.Window, rasterio.windows.transform]:
        col = idx % (self.window.width // self.tile_size)
        row = idx // (self.window.width // self.tile_size)
        window = rasterio.windows.Window(
            col_off=self.window.col_off + col * self.tile_size,
            row_off=self.window.row_off + row * self.tile_size,
            width=self.tile_size,
            height=self.tile_size,
        )
        transform = rasterio.windows.transform(window, self.dataset.transform)
        return window, transform

    def __getitem__(self, idx) -> tuple[Image.Image]:
        window, transform = self._index_to_window(idx)
        image = self.dataset.read(window=window)
        image = reshape_as_image(image)
        image = F.to_image(image)
        image = self.transform(image)
        return image, dict(transform=transform)

    def __del__(self):
        self.dataset.close()
