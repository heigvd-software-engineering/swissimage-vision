import os
from typing import Optional

from dotenv import load_dotenv
from requests.sessions import Session


class LabelStudioSession:
    def __init__(self, host: str, token: str):
        self.session = Session()
        self.session.headers.update({"Authorization": "Token " + token})
        self.session.base_url = host

    def request(self, method: str, url: str, data: Optional[dict] = None) -> dict:
        response = self.session.request(method, self.session.base_url + url, json=data)
        response.raise_for_status()
        return response.json()


def project_exists(session: LabelStudioSession, title: str) -> bool:
    projects = session.request("get", "/api/projects/")
    num_same = len(
        list(
            filter(
                lambda p: p["title"] == title,
                projects["results"],
            )
        )
    )
    return projects["count"] > 0 and num_same > 0


def create_project(session: LabelStudioSession, config: dict) -> None:
    if not project_exists(session, config["project"]["title"]):
        print("[INFO] Creating project...")
        with open("labelstudio/label_config.xml", "r") as f:
            label_config = f.read()
        project = session.request(
            "post",
            "/api/projects/",
            data=dict(
                label_config=label_config,
                **config["project"],
            ),
        )
        print("[INFO] Configuring S3 storage...")
        storage = session.request(
            "post",
            "/api/storages/s3/",
            data=dict(
                project=project["id"],
                **config["s3"],
            ),
        )
        print("[INFO] Syncing S3 storage...")
        session.request(
            "post",
            f"/api/storages/s3/{storage['id']}/sync",
        )
    print("[INFO] Done!")


def configure(session: LabelStudioSession) -> None:
    create_project(
        session,
        config=dict(
            project=dict(
                title="SwissImage",
                is_published=True,
                evaluate_predictions_automatically=True,
                sampling="Uniform sampling",
            ),
            s3=dict(
                synchronizable=True,
                presign=True,
                title="swissimage-tiles",
                bucket="swissimage-vision",
                prefix="data/tiles",
                regex_filter=".*\.png",
                s3_endpoint="https://" + os.environ["AWS_S3_ENDPOINT"],
                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                recursive_scan=False,
                use_blob_urls=True,
            ),
        ),
    )


def main() -> None:
    load_dotenv(".env", override=True)
    session = LabelStudioSession(
        host=os.environ["LABELSTUDIO_HOST"],
        token=os.environ["LABELSTUDIO_TOKEN"],
    )
    configure(session)


if __name__ == "__main__":
    main()
