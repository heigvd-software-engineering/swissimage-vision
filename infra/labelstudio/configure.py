import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from label_studio_sdk import Client
from label_studio_sdk.project import ProjectSampling


def configure(
    ls: Client,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    s3_endpoint: str,
    project_config: dict,
    import_storage_config: dict,
    export_storage_config: dict,
) -> None:
    """Configure Label Studio project with import and export storage

    Args:
        ls (Client): Label Studio client
        aws_access_key_id (str): AWS access key ID
        aws_secret_access_key (str): AWS secret access key
        s3_endpoint (str): AWS S3 endpoint
        project_config (dict): Project configuration
        import_storage_config (dict): Import storage configuration
        export_storage_config (dict): Export storage configuration

    See also:
        https://labelstud.io/sdk/project.html#label_studio_sdk.project.Project.start_project
        https://labelstud.io/sdk/project.html#label_studio_sdk.project.Project.connect_s3_import_storage
        https://labelstud.io/sdk/project.html#label_studio_sdk.project.Project.connect_s3_export_storage
    """
    print("[INFO] Configuring Label Studio...")
    # 1. Create a new project
    label_config_path = Path(__file__).resolve().parent / "label_config.xml"
    project = ls.start_project(
        **project_config, label_config=label_config_path.read_text()
    )
    # 2. Create import storage
    import_storage = project.connect_s3_import_storage(
        **import_storage_config,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        s3_endpoint=s3_endpoint,
    )
    project.sync_storage(
        storage_type=import_storage["type"], storage_id=import_storage["id"]
    )
    # 3. Create export storage
    export_storage = project.connect_s3_export_storage(
        **export_storage_config,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        s3_endpoint=s3_endpoint,
    )
    project.sync_storage(
        storage_type=export_storage["type"], storage_id=export_storage["id"]
    )
    print("[INFO] Configuration completed")


def main() -> None:
    load_dotenv(override=True)

    ls = Client(
        url=os.environ["LABEL_STUDIO_HOST"], api_key=os.environ["LABEL_STUDIO_TOKEN"]
    )
    # Check if a project already exists
    # (may be modifed to check if a project with a specific name exists)
    projects = ls.get_projects()
    if projects:
        print("[INFO] Project already exists, skipping configuration")
        return

    params = yaml.safe_load(open("params.yaml"))
    bucket = params["bucket"]
    prepared_path = Path(params["prepare"]["s3_dest_prepared_path"])
    # ------------------------------------------------------------------------
    # Configuration parameters
    # ------------------------------------------------------------------------
    configure(
        ls=ls,
        # AWS credentials
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        s3_endpoint="https://" + os.environ["AWS_S3_ENDPOINT"],
        # Label Studio project configuration
        project_config=dict(
            title="SwissImage Vision",
            description="Label satellite images for SwissImage Vision",
            show_skip_button=True,
            enable_empty_annotation=True,
            show_annotation_history=True,
            sampling=ProjectSampling.RANDOM.value,
            evaluate_predictions_automatically=False,
        ),
        # Label Studio import storage configuration
        import_storage_config=dict(
            bucket=bucket,
            prefix=str(prepared_path / "images"),
            regex_filter=r".*\.png",
            use_blob_urls=True,
            presign=True,
            title="SwissImage Vision Images",
        ),
        # Label Studio export storage configuration
        export_storage_config=dict(
            bucket=bucket,
            prefix=str(prepared_path / "annotations"),
            title="SwissImage Vision Annotations",
            can_delete_objects=True,
        ),
    )


if __name__ == "__main__":
    main()
