import concurrent.futures
import os
from io import BytesIO
from pathlib import Path

import boto3
from PIL import Image


def get_s3_resource() -> boto3.resource:
    return boto3.resource(
        "s3",
        endpoint_url="https://" + os.environ["AWS_S3_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


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


def list_files(s3: boto3.resource, bucket: str, prefix: str | Path) -> list[str]:
    if isinstance(prefix, Path):
        prefix = str(prefix)
    # List all files in the bucket with the given prefix
    return [obj.key for obj in s3.Bucket(bucket).objects.filter(Prefix=prefix)]


def get_file(s3: boto3.resource, bucket: str, key: str) -> BytesIO:
    # Get a file from the bucket
    obj = s3.Object(bucket, key)
    return BytesIO(obj.get()["Body"].read())


def get_image(s3: boto3.resource, bucket: str, key: str) -> Image:
    data = get_file(s3, bucket, key)
    return Image.open(data)


def split_url_to_bucket_and_prefix(url: str) -> tuple[str, str]:
    """
    Split an S3 URL into bucket and prefix.

    Example:
        s3://bucket-name/prefix/to/file -> ('bucket-name', 'prefix/to/file')

    Args:
        url (str): S3 URL

    Returns:
        tuple[str, str]: Bucket and prefix
    """
    url = url.replace("s3://", "")
    parts = url.split("/")
    bucket = parts[0]
    prefix = "/".join(parts[1:])
    return bucket, prefix
