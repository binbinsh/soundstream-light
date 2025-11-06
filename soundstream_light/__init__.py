"""SoundStream Light Python interface."""

from contextlib import suppress
from importlib import metadata

from .api import EmbeddingMetadata, decode, encode, models_directory, resolve_resources
from .cli import app

__all__ = [
    "app",
    "__version__",
    "encode",
    "decode",
    "resolve_resources",
    "models_directory",
    "EmbeddingMetadata",
]

_version = "0.0.0"
with suppress(metadata.PackageNotFoundError):
    _version = metadata.version("soundstream-light")

__version__ = _version

