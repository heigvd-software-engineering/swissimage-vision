from pathlib import Path

import yaml
from dotenv import load_dotenv

from utils.extract_tiles import extract_tiles


def main() -> None:
    load_dotenv(override=True)
    params = yaml.safe_load(open("params.yaml"))
    prepare_params = params["prepare"]

    extract_tiles(
        bucket=params["bucket"],
        s3_src_vrt_path=Path(prepare_params["s3_src_vrt_path"]),
        s3_prepared_path=Path(prepare_params["s3_prepared_path"]),
        commune_name=prepare_params["commune_name"],
        commune_x_ratio=prepare_params["commune_x_ratio"],
        commune_y_ratio=prepare_params["commune_y_ratio"],
        tile_size=prepare_params["tile_size"],
    )


if __name__ == "__main__":
    main()
