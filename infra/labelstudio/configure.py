import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from label_studio_sdk import Client
from label_studio_sdk.project import ProjectSampling


def configure(ls: Client, bucket: str):
    print("[INFO] Configuring Label Studio...")
    # 1. Create a new project
    # https://labelstud.io/sdk/project.html#label_studio_sdk.project.Project.start_project
    project = ls.start_project(
        title="Swissimage Vision",
        description="Label satellite images for Swissimage Vision",
        label_config=(Path(__file__).resolve().parent / "label_config.xml").read_text(),
        show_skip_button=True,
        enable_empty_annotation=True,
        show_annotation_history=True,
        sampling=ProjectSampling.RANDOM.value,
        evaluate_predictions_automatically=False,
    )
    # 2. Create import storage
    import_storage = project.connect_s3_import_storage(
        bucket=bucket,
        prefix="data/prepared/images",
        regex_filter=r".*\.png",
        use_blob_urls=True,
        presign=True,
        title="Swissimage Vision Images",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        s3_endpoint="https://" + os.environ["AWS_S3_ENDPOINT"],
    )
    project.sync_storage(
        storage_type=import_storage["type"], storage_id=import_storage["id"]
    )
    # 3. Create export storage
    export_storage = project.connect_s3_export_storage(
        bucket=bucket,
        prefix="data/prepared/annotations",
        title="Swissimage Vision Annotations",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        s3_endpoint="https://" + os.environ["AWS_S3_ENDPOINT"],
        can_delete_objects=True,
    )
    project.sync_storage(
        storage_type=export_storage["type"], storage_id=export_storage["id"]
    )
    print("[INFO] Configuration completed")


def main():
    load_dotenv()

    ls = Client(
        url=os.environ["LABEL_STUDIO_HOST"],
        api_key=os.environ["LABEL_STUDIO_TOKEN"],
    )
    projects = ls.get_projects()
    if projects:
        print("[INFO] Project already exists, skipping configuration")
        return

    params = yaml.safe_load(open("params.yaml"))
    configure(ls=ls, bucket=params["bucket"])


if __name__ == "__main__":
    main()
