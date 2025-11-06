"""High-level Python helpers for encoding and decoding audio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np  # type: ignore[import-not-found]

from . import models
from .runtime import ensure_models, resolve_models_dir

try:  # Optional torch support
    import torch  # type: ignore[import-not-found]

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is optional
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

try:
    from . import _native as _NATIVE
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "soundstream_light._native extension is missing."
        " Build the C++ project with CMake before using the"
        " Python encode/decode helpers."
    ) from exc


@dataclass(frozen=True)
class EmbeddingMetadata:
    sample_rate_hz: int
    num_channels: int
    original_num_samples: int
    embedding_dim: int


DEFAULT_SAMPLE_RATE_HZ = 16000


def _is_torch_tensor(value: Any) -> bool:
    return _TORCH_AVAILABLE and isinstance(value, torch.Tensor)


def _to_numpy_audio(wav: Any) -> Tuple[np.ndarray, bool]:
    """Return (np.ndarray, input_was_torch)."""
    if isinstance(wav, np.ndarray):
        array = np.asarray(wav)
        return array, False
    if _is_torch_tensor(wav):
        tensor = torch.as_tensor(wav)
        array = tensor.detach().cpu().numpy()
        return array, True
    if isinstance(wav, (list, tuple)):
        array = np.asarray(wav)
        return array, False
    raise TypeError("wav must be a NumPy array, torch.Tensor, list, or tuple")


def _ensure_mono(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        if 1 in array.shape:
            return np.reshape(array, (-1,))
    raise ValueError("SoundStream encoder only supports mono PCM waveforms")


def _prepare_pcm(array: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(array)
    if array.dtype == np.int16:
        return array
    if array.dtype in (np.float32, np.float64):
        clipped = np.clip(array.astype(np.float32), -1.0, 1.0)
        return (clipped * 32768.0).round().astype(np.int16)
    raise TypeError("wav must have dtype int16, float32, or float64")


def _to_numpy_embeddings(embeddings: Any) -> Tuple[np.ndarray, bool]:
    if isinstance(embeddings, np.ndarray):
        array = np.asarray(embeddings)
        return array, False
    if _is_torch_tensor(embeddings):
        tensor = torch.as_tensor(embeddings)
        array = tensor.detach().cpu().numpy()
        return array, True
    if isinstance(embeddings, (list, tuple)):
        array = np.asarray(embeddings)
        return array, False
    raise TypeError("embeddings must be a NumPy array, torch.Tensor, list, or tuple")


def encode(
    wav: Any,
    *,
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    models_dir: Path | str | None = None,
    threads: int = 1,
    use_xnnpack: bool = True,
    auto_download: bool = True,
    base_url: str = models.DEFAULT_BASE_URL,
    overwrite_models: bool = False,
    local_model_source: Path | str | None = None,
    return_torch: bool | None = None,
) -> tuple[Any, EmbeddingMetadata]:
    """Encode a mono waveform into SoundStream embeddings.

    Parameters follow the CLI behaviour. The return value is a tuple containing:

    - embeddings: 2D array (frames, embedding_dim) as NumPy by default or torch if requested.
    - metadata: EmbeddingMetadata describing the produced embeddings.
    """

    waveform, was_torch = _to_numpy_audio(wav)
    mono = _ensure_mono(waveform)
    pcm = _prepare_pcm(mono)

    resolved_models = ensure_models(
        models_dir,
        auto_download=auto_download,
        base_url=base_url,
        overwrite=overwrite_models,
        local_source=local_model_source,
    )

    embeddings_np, metadata_dict = _NATIVE.encode(
        pcm,
        sample_rate_hz,
        1,
        str(resolved_models),
        use_xnnpack,
        threads,
    )

    metadata = EmbeddingMetadata(
        sample_rate_hz=int(metadata_dict["sample_rate_hz"]),
        num_channels=int(metadata_dict["num_channels"]),
        original_num_samples=int(metadata_dict["original_num_samples"]),
        embedding_dim=int(metadata_dict["embedding_dim"]),
    )

    embeddings_np = np.ascontiguousarray(embeddings_np.astype(np.float32))

    produce_torch = return_torch if return_torch is not None else was_torch
    if produce_torch:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed; cannot return torch tensors")
        embeddings_out = torch.from_numpy(embeddings_np)
    else:
        embeddings_out = embeddings_np

    return embeddings_out, metadata


def decode(
    embeddings: Any,
    *,
    metadata: EmbeddingMetadata | None = None,
    sample_rate_hz: int | None = None,
    num_channels: int = 1,
    original_num_samples: int | None = None,
    models_dir: Path | str | None = None,
    threads: int = 1,
    use_xnnpack: bool = True,
    auto_download: bool = True,
    base_url: str = models.DEFAULT_BASE_URL,
    overwrite_models: bool = False,
    local_model_source: Path | str | None = None,
    return_torch: bool | None = None,
) -> Any:
    """Decode embeddings back into a mono waveform.

    If *metadata* from :func:`encode` is provided, it is used to restore the
    original sample count and validation metadata. Otherwise supply
    ``sample_rate_hz`` and optionally ``original_num_samples``.
    """

    embedding_array, was_torch = _to_numpy_embeddings(embeddings)
    if embedding_array.ndim != 2:
        raise ValueError("embeddings must be shaped as (frames, embedding_dim)")
    embedding_array = np.ascontiguousarray(embedding_array.astype(np.float32))

    if metadata is not None:
        rate = metadata.sample_rate_hz
        channels = metadata.num_channels
        original_samples = metadata.original_num_samples
    else:
        rate = sample_rate_hz if sample_rate_hz is not None else DEFAULT_SAMPLE_RATE_HZ
        channels = num_channels
        original_samples = original_num_samples if original_num_samples is not None else -1

    resolved_models = ensure_models(
        models_dir,
        auto_download=auto_download,
        base_url=base_url,
        overwrite=overwrite_models,
        local_source=local_model_source,
    )

    waveform_np = _NATIVE.decode(
        embedding_array,
        int(rate),
        int(channels),
        int(original_samples),
        str(resolved_models),
        use_xnnpack,
        threads,
    )

    waveform_np = np.ascontiguousarray(waveform_np.astype(np.float32))

    produce_torch = return_torch if return_torch is not None else was_torch
    if produce_torch:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed; cannot return torch tensors")
        return torch.from_numpy(waveform_np)
    return waveform_np


def resolve_resources(
    *,
    models_dir: Path | str | None = None,
    auto_download: bool = True,
    base_url: str = models.DEFAULT_BASE_URL,
    overwrite_models: bool = False,
    local_model_source: Path | str | None = None,
) -> Path:
    """Ensure model assets exist and return their directory."""

    return ensure_models(
        models_dir,
        auto_download=auto_download,
        base_url=base_url,
        overwrite=overwrite_models,
        local_source=local_model_source,
    )


def models_directory(models_dir: Path | str | None = None) -> Path:
    return resolve_models_dir(models_dir)

