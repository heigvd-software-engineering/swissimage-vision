import multiprocessing
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import rasterio
import yaml
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
from tqdm import tqdm


def save_tile(
    tile: np.ndarray,
    row_idx: int,
    col_idx: int,
    image_size: int | None,
    images_dir: Path,
) -> None:
    image = Image.fromarray(tile)
    if image_size is not None:
        image = image.resize((image_size, image_size))
    image.save(images_dir / f"tile_{row_idx}_{col_idx}.png", optimize=True, quality=90)


def tile_tif(
    tif_path: Path,
    tile_size: int,
    overlap_size: int,
    processes: int,
    image_size: int | None,
    out_root: Path,
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

    if out_root.exists():
        shutil.rmtree(out_root)

    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    with multiprocessing.Pool(processes=processes) as pool:
        for row_idx, row in enumerate(tiles):
            for col_idx, tile in enumerate(row):
                pool.apply_async(
                    save_tile,
                    kwds=dict(
                        tile=tile,
                        row_idx=row_idx,
                        col_idx=col_idx,
                        image_size=image_size,
                        images_dir=images_dir,
                    ),
                    callback=lambda _: pbar.update(),
                )

        pool.close()
        pool.join()

    pbar.close()


def main() -> None:
    tile_size = 1024
    overlap_cnt = 0
    overlap_size = int(tile_size * 1 / (overlap_cnt + 1)) if overlap_cnt > 0 else 0

    tile_tif(
        tif_path=Path("data/raw/nyon.tif"),
        tile_size=tile_size,
        overlap_size=overlap_size,
        processes=10,
        image_size=None,
        out_root=Path("data/raw/nyon_tiles"),
    )


if __name__ == "__main__":
    main()
