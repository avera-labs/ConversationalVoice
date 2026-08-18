from pathlib import PurePosixPath
from urllib.parse import urlsplit

import boto3


class ObjectStorage:
    def __init__(self, client, bucket):
        self.client = client
        self.bucket = bucket

    @classmethod
    def create(cls, env):
        args = {"region_name": env.s3_region}
        if env.s3_endpoint_url:
            args["endpoint_url"] = env.s3_endpoint_url
        return cls(boto3.client("s3", **args), env.s3_bucket)

    def _key(self, uri):
        if not isinstance(uri, str) or not uri or any(c.isspace() for c in uri):
            raise ValueError("invalid S3 URI")
        parsed = urlsplit(uri)
        if (
            parsed.scheme != "s3"
            or parsed.netloc != self.bucket
            or not parsed.path.startswith("/")
            or parsed.path.startswith("//")
            or parsed.query
            or parsed.fragment
            or "@" in parsed.netloc
            or ":" in parsed.netloc
            or "%" in parsed.path
            or "\\" in parsed.path
            or any(x in {"", ".", ".."} for x in parsed.path[1:].split("/"))
        ):
            raise ValueError("invalid S3 URI")
        return parsed.path[1:]

    def download(self, uri, path, *, maximum_bytes=None):
        try:
            self.client.download_file(self.bucket, self._key(uri), str(path))
            size = path.stat().st_size
            if size <= 0 or (maximum_bytes is not None and size > maximum_bytes):
                raise OSError("object size is invalid")
            return size
        except Exception:
            path.unlink(missing_ok=True)
            raise

    def output_uris(self, audio_uri):
        key = PurePosixPath(self._key(audio_uri))
        base = key.parent / "results" / "separated"
        return tuple(f"s3://{self.bucket}/{base}/speaker-{i}.wav" for i in range(2))

    def upload(self, uri, path):
        self.client.upload_file(
            str(path),
            self.bucket,
            self._key(uri),
            ExtraArgs={"ContentType": "audio/wav"},
        )

    def close(self):
        close = getattr(self.client, "close", None)
        if close:
            close()
