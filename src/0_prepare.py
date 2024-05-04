from pathlib import Path

import yaml
from dotenv import load_dotenv

from utils.extract_tiles import extract_tiles


def main() -> None:
    load_dotenv(override=True)
    params = yaml.safe_load(open("params.yaml"))
    prepare_params = params["prepare"]

    extract_tiles(
        src_bucket=prepare_params["src_bucket"],
        s3_src_vrt_path=Path(prepare_params["s3_src_vrt_path"]),
        s3_dest_prepared_path=Path(prepare_params["s3_dest_prepared_path"]),
        dest_bucket=params["bucket"],
        commune_name=prepare_params["commune_name"],
        commune_x_ratio=prepare_params["commune_x_ratio"],
        commune_y_ratio=prepare_params["commune_y_ratio"],
        tile_size=prepare_params["tile_size"],
    )

    # Create dummy file to make other DVC stages depend on this one
    dummy_file = Path("out/prepare/depends.txt")
    content = "NOTE: This a dummy file is used to make other DVC stages depend on this one (prepare)."
    dummy_file.parent.mkdir(parents=True, exist_ok=True)
    dummy_file.write_text(content)


if __name__ == "__main__":
    main()
