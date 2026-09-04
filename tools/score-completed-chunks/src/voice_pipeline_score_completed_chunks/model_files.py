from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import requests

from .errors import ScoringError


@dataclass(frozen=True, slots=True)
class ModelFile:
    name: str
    url: str
    sha256: str
    size_bytes: int


NISQA_MODEL = ModelFile(
    "nisqa.tar",
    "https://raw.githubusercontent.com/gabrielmittag/NISQA/fe84f0f252abec382b24367d5b22498a7ce34dbb/weights/nisqa.tar",
    "7ec4cf937514dd3f8860b21e66fabd8ca87a168572675ef8d979c4c4ad2e805c",
    1_051_663,
)

DNSMOS_PRIMARY = ModelFile(
    "sig_bak_ovr.onnx",
    "https://raw.githubusercontent.com/microsoft/DNS-Challenge/591184a9fcb2cbdec02520fed81a32bbbf9d73ff/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
    "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd",
    1_157_965,
)

DNSMOS_P808 = ModelFile(
    "model_v8.onnx",
    "https://raw.githubusercontent.com/microsoft/DNS-Challenge/591184a9fcb2cbdec02520fed81a32bbbf9d73ff/DNSMOS/DNSMOS/model_v8.onnx",
    "9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91",
    224_860,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_model_file(specification: ModelFile, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if (
            destination.stat().st_size == specification.size_bytes
            and file_sha256(destination) == specification.sha256
        ):
            return destination
        raise ScoringError("model_identity_mismatch", specification.name)
    temporary = destination.with_suffix(destination.suffix + ".download")
    temporary.unlink(missing_ok=True)
    try:
        with requests.get(specification.url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for block in response.iter_content(1024 * 1024):
                    if block:
                        output.write(block)
                        output.flush()
            if (
                temporary.stat().st_size != specification.size_bytes
                or file_sha256(temporary) != specification.sha256
            ):
                raise ScoringError("model_identity_mismatch", specification.name)
            os.replace(temporary, destination)
    except ScoringError:
        temporary.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise ScoringError("model_download_failed", specification.name) from exc
    return destination
