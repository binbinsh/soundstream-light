"""Model management helpers for the SoundStream CLI."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

DEFAULT_BASE_URL = (
    "https://raw.githubusercontent.com/google/lyra/main/lyra/model_coeffs"
)

MODEL_FILES: Dict[str, str] = {
    "soundstream_encoder.tflite": "237da0b16226fb0a68d49551c81bebf614e42142af72764448359f68581e88f1",
    "lyragan.tflite": "50e7839cda4f30599a3a34cfaddcd9c98449b049eee884e1a74295780448ad06",
    "quantizer.tflite": "841ee05989687d32d1f616a8ad1419fc7eda37599d3c05c7cf5a0d60ccf8ac5c",
}


@dataclass(frozen=True)
class DownloadReport:
    fetched: tuple[Path, ...]
    skipped: tuple[Path, ...]
    failed: tuple[Path, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_models(model_dir: Path) -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    for filename, checksum in MODEL_FILES.items():
        file_path = model_dir / filename
        if not file_path.exists():
            results[filename] = False
            continue
        results[filename] = _sha256(file_path) == checksum
    return results


def _download_via_curl(url: str, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["curl", "-L", url, "-o", str(destination)],
        check=False,
    )
    return result.returncode == 0


def download_models(
    model_dir: Path,
    base_url: str = DEFAULT_BASE_URL,
    overwrite: bool = False,
    local_source: Path | None = None,
) -> DownloadReport:
    fetched: list[Path] = []
    skipped: list[Path] = []
    failed: list[Path] = []

    for filename in MODEL_FILES:
        target = model_dir / filename
        if target.exists() and not overwrite:
            skipped.append(target)
            continue

        if local_source is not None:
            source_file = local_source / filename
            if not source_file.exists():
                failed.append(target)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target)
        else:
            url = f"{base_url.rstrip('/')}/{filename}"
            if not _download_via_curl(url, target):
                failed.append(target)
                continue

        if _sha256(target) == MODEL_FILES[filename]:
            fetched.append(target)
        else:
            failed.append(target)
            target.unlink(missing_ok=True)

    return DownloadReport(
        fetched=tuple(fetched), skipped=tuple(skipped), failed=tuple(failed)
    )

