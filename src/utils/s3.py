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


def upload_files(
    s3: boto3.resource,
    path: Path,
    bucket: str,
    dest_folder: Path,
    max_workers: int = 10,
) -> None:
    # Recursively upload all files in path to the bucket
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for root, _, files in os.walk(path):
            for file in files:
                file_path = Path(root) / file
                dest_key = str(dest_folder / file_path.relative_to(path))
                executor.submit(
                    upload_file,
                    s3,
                    file_path,
                    bucket,
                    dest_key,
                )


def delete_file(s3: boto3.resource, bucket: str, key: str) -> None:
    s3.Object(bucket, key).delete()


def delete_files(
    s3: boto3.resource, bucket: str, folder: Path, max_workers: int = 10
) -> None:
    # Delete all files in the folder
    print(f"[INFO] Deleting files in {folder} from s3://{bucket}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for obj in s3.Bucket(bucket).objects.filter(Prefix=str(folder)):
            executor.submit(delete_file, s3, bucket, obj.key)
