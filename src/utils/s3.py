import concurrent.futures
import os
from pathlib import Path

import boto3


def upload_file(
    s3: boto3.resource, file_path: Path, bucket: str, dest_key: str
) -> None:
    s3.meta.client.upload_file(
        Filename=str(file_path),
        Bucket=bucket,
        Key=dest_key,
    )
    print(f"[INFO] Uploaded {file_path} to s3://{bucket}/{dest_key}")


def upload_files(
    s3: boto3.resource,
    path: Path,
    bucket: str,
    dest_folder: Path,
    delete_if_exists: bool = False,
    max_workers: int = 10,
) -> None:
    if delete_if_exists:
        # Check if dest_folder exists in the bucket and delete it if it does
        print(f"[INFO] Deleting existing files in s3://{bucket}/{dest_folder}")
        for obj in s3.Bucket(bucket).objects.filter(Prefix=str(dest_folder)):
            obj.delete()

    # Recursively upload all files in path to the bucket
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for root, _, files in os.walk(path):
            for file in files:
                file_path = Path(root) / file
                dest_key = str(dest_folder / file_path.relative_to(path))
                executor.submit(
                    upload_file,
                    kwargs=dict(
                        s3=s3,
                        file_path=file_path,
                        bucket=bucket,
                        dest_key=dest_key,
                    ),
                )
