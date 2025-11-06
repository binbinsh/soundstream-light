"""Shared helpers for locating resources and invoking the native binary."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from . import models

ENV_BINARY = "SOUNDSTREAM_CLI_BINARY"
ENV_MODELS = "SOUNDSTREAM_MODELS_DIR"


def _coerce_path(value: Path | str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser()


def workspace_root() -> Path | None:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "include" / "soundstream").exists():
            return candidate
    return None


def resolve_models_dir(value: Path | str | os.PathLike[str] | None) -> Path:
    direct = _coerce_path(value)
    if direct is not None:
        return direct
    env_value = os.environ.get(ENV_MODELS)
    if env_value:
        return Path(env_value).expanduser()
    root = workspace_root()
    if root is not None:
        return root / "models"
    return Path.cwd() / "models"


def detect_binary() -> Path | None:
    env_value = os.environ.get(ENV_BINARY)
    if env_value:
        candidate = Path(env_value).expanduser()
        if candidate.exists():
            return candidate
    root = workspace_root()
    if root is not None:
        repo_candidate = root / "build" / "soundstream_cli"
        if repo_candidate.exists():
            return repo_candidate
    return None


def locate_binary(value: Path | str | os.PathLike[str] | None) -> tuple[Path | None, str | None]:
    direct = _coerce_path(value)
    candidate = direct if direct is not None else detect_binary()
    if candidate is None:
        return None, "Native binary not found. Supply --binary or set SOUNDSTREAM_CLI_BINARY."
    if candidate.exists() and os.access(candidate, os.X_OK):
        return candidate, None
    message = f"Binary not found or not executable: {candidate}. Build with CMake first."
    return None, message


def resolve_binary(value: Path | str | os.PathLike[str] | None) -> Path:
    candidate, error = locate_binary(value)
    if candidate is None:
        raise RuntimeError(error or "Native binary not found")
    return candidate


def ensure_models(
    value: Path | str | os.PathLike[str] | None,
    *,
    auto_download: bool,
    base_url: str = models.DEFAULT_BASE_URL,
    overwrite: bool = False,
    local_source: Path | str | os.PathLike[str] | None = None,
) -> Path:
    destination = resolve_models_dir(value)
    destination.mkdir(parents=True, exist_ok=True)
    verification = models.verify_models(destination)
    if all(verification.values()):
        return destination
    if not auto_download:
        missing = ", ".join(name for name, ok in verification.items() if not ok)
        raise RuntimeError(f"Missing or invalid model files: {missing}")
    report = models.download_models(
        destination,
        base_url=base_url,
        overwrite=overwrite,
        local_source=_coerce_path(local_source),
    )
    if report.failed:
        joined = ", ".join(str(path) for path in report.failed)
        raise RuntimeError(f"Failed to fetch model files: {joined}")
    verification = models.verify_models(destination)
    if all(verification.values()):
        return destination
    remaining = ", ".join(name for name, ok in verification.items() if not ok)
    raise RuntimeError(f"Model verification failed: {remaining}")


def run_cli(binary: Path, args: list[str]) -> int:
    command = [str(binary)] + args
    result = subprocess.run(command, check=False)
    return result.returncode

