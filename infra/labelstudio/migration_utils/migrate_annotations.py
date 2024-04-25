"""
Used when a new instance of Label Studio is deployed, annotations in
S3 store different project, task and annotation IDs. This script migrates
annotations from S3 to Label Studio by iterating over all annotations and
creating them if the task exists and is not labeled yet.

Not performed in this script:
    - Decide what to do with old annotations in S3. (delete or keep)
    - Sync annotations from Label Studio to S3 after migration.
"""

import json
import os
from pathlib import Path

import boto3
import yaml
from dotenv import load_dotenv
from label_studio_sdk import Client, Project
from tqdm import tqdm


def migrate_annotations(
    bucket: str, prepared_path: Path, project: Project, task_mapping: dict
) -> None:
    """
    Migrate annotations from S3 to Label Studio.

    Args:
        bucket (str): S3 bucket name
        prepared_path (Path): Prepared path
        project (Project): Label Studio project
        task_mapping (dict): Task mapping
    """
    s3 = boto3.client(
        "s3",
        endpoint_url="https://" + os.environ["AWS_S3_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    # Iterate over all annotations
    annotations = s3.list_objects_v2(Bucket=bucket, Prefix=str(prepared_path))
    for ann in tqdm(annotations["Contents"], desc="Updating annotations"):
        ann_key = ann["Key"]
        ann_body = s3.get_object(Bucket=bucket, Key=ann_key)["Body"].read()
        ann_dict = json.loads(ann_body)

        key = ann_dict["task"]["data"]["image_url"].split("/")[-1]
        # Skip if task is not found or already labeled
        if not task_mapping.get(key) or task_mapping[key]["is_labeled"]:
            continue

        project.create_annotation(
            task_mapping[key]["id"],
            completed_by=1,
            result=ann_dict["result"],
            was_cancelled=ann_dict["was_cancelled"],
            ground_truth=ann_dict["ground_truth"],
            draft_created_at=ann_dict["draft_created_at"],
            lead_time=ann_dict["lead_time"],
            last_action=ann_dict["last_action"],
            updated_by=1,
        )


def get_task_mapping(project: Project) -> dict:
    """
    Get task mapping.

    Args:
        ls (Client): Label Studio client
        project (Project): Label Studio project

    Returns:
        dict: Task mapping
    """
    print("[INFO] Fetching tasks...")
    tasks = project.get_tasks()
    task_mapping = {}
    for task in tasks:
        key = task["storage_filename"].split("/")[-1]
        task_mapping[key] = {"id": task["id"], "is_labeled": task["is_labeled"]}
    return task_mapping


def main() -> None:
    load_dotenv(override=True)

    ls = Client(
        url=os.environ["LABEL_STUDIO_HOST"], api_key=os.environ["LABEL_STUDIO_TOKEN"]
    )

    projects = ls.get_projects()
    if not projects:
        print("[ERROR] Project not found.")
        exit(1)

    # Assume that the first project is the one we need
    project: Project = projects[0]

    params = yaml.safe_load(open("params.yaml"))
    bucket = params["bucket"]
    prepared_path = Path(params["prepare"]["s3_dest_prepared_path"])

    migrate_annotations(
        bucket=bucket,
        prepared_path=prepared_path,
        project=project,
        task_mapping=get_task_mapping(),
    )
    print("[INFO] Annotations updated!")
    print("[INFO] Next step: Sync annotation manully in the Label Studio UI.")


if __name__ == "__main__":
    main()
