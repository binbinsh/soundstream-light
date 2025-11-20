"""Quality metrics and round-trip evaluation helpers."""

from __future__ import annotations

import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np  # type: ignore[import-not-found]

from . import api, models


def _align_lengths(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim signals to the shortest length."""
    target_len = min(reference.shape[0], estimate.shape[0])
    return reference[:target_len], estimate[:target_len]


def _as_mono(waveform: np.ndarray) -> np.ndarray:
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim == 2:
        return np.mean(waveform, axis=1)
    raise ValueError("Audio must be mono or 2D with channels on axis 1")


def load_wav_mono(path: Path | str) -> tuple[int, np.ndarray]:
    """Load a PCM WAV file into float32 mono samples in [-1, 1]."""
    resolved = Path(path)
    with wave.open(str(resolved), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        if sample_width != 2:
            raise ValueError(f"Only 16-bit PCM WAV is supported (got {sample_width * 8}-bit).")
        frames = handle.readframes(handle.getnframes())
    pcm = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels)
        pcm = _as_mono(pcm)
    waveform = pcm.astype(np.float32) / 32768.0
    return sample_rate, waveform


def write_wav_mono(path: Path | str, waveform: np.ndarray, sample_rate: int) -> None:
    """Persist a mono float waveform ([-1, 1]) as 16-bit PCM WAV."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(_as_mono(waveform).astype(np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).round().astype(np.int16)
    with wave.open(str(resolved), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def si_snr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant SNR in dB."""
    ref, est = _align_lengths(_as_mono(reference), _as_mono(estimate))
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    target = np.dot(est, ref) / (np.dot(ref, ref) + eps) * ref
    noise = est - target
    return float(10.0 * np.log10(np.sum(target * target) / (np.sum(noise * noise) + eps)))


def stoi_score(reference: np.ndarray, estimate: np.ndarray, sample_rate_hz: int) -> float:
    """Short-Time Objective Intelligibility (0-1)."""
    try:
        from pystoi import stoi  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pystoi is required to compute STOI. Install soundstream-light[metrics].") from exc
    ref, est = _align_lengths(_as_mono(reference), _as_mono(estimate))
    return float(stoi(ref, est, sample_rate_hz, extended=False))


def pesq_score(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate_hz: int,
    mode: str | None = None,
) -> float:
    """PESQ MOS-LQO (1–4.5) using the 8k/16k reference implementation."""
    try:
        from pesq import pesq  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pesq is required to compute PESQ. Install soundstream-light[metrics].") from exc
    if sample_rate_hz not in (8000, 16000):
        raise ValueError("PESQ only supports 8 kHz (nb) or 16 kHz (wb) sampling rates.")
    ref, est = _align_lengths(_as_mono(reference), _as_mono(estimate))
    band_mode = mode if mode is not None else ("wb" if sample_rate_hz > 8000 else "nb")
    return float(pesq(sample_rate_hz, np.clip(ref, -1.0, 1.0), np.clip(est, -1.0, 1.0), band_mode))


def compute_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate_hz: int,
    *,
    compute_stoi: bool = True,
    compute_pesq: bool = False,
) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    scores["si_snr_db"] = si_snr(reference, estimate)
    if compute_stoi:
        scores["stoi"] = stoi_score(reference, estimate, sample_rate_hz)
    else:
        scores["stoi"] = None
    if compute_pesq:
        scores["pesq"] = pesq_score(reference, estimate, sample_rate_hz)
    else:
        scores["pesq"] = None
    return scores


@dataclass(frozen=True)
class RoundtripReport:
    sample_rate_hz: int
    metrics: dict[str, float | None]
    encode_seconds: float
    decode_seconds: float
    rtf: float | None
    reconstruction: np.ndarray


def evaluate_roundtrip(
    wav_path: Path | str,
    *,
    models_dir: Path | str | None = None,
    threads: int = 1,
    use_xnnpack: bool = True,
    compute_stoi: bool = True,
    compute_pesq: bool = False,
    auto_download_models: bool = True,
    base_url: str = models.DEFAULT_BASE_URL,
    overwrite_models: bool = False,
    local_model_source: Path | str | None = None,
    encode_fn: Callable[..., Any] = api.encode,
    decode_fn: Callable[..., Any] = api.decode,
) -> RoundtripReport:
    """Run encode+decode on a WAV file and measure SI-SNR/STOI/PESQ plus RTF."""
    sample_rate, clean = load_wav_mono(Path(wav_path))

    encode_start = time.perf_counter()
    embeddings, metadata = encode_fn(
        clean,
        sample_rate_hz=sample_rate,
        models_dir=models_dir,
        threads=threads,
        use_xnnpack=use_xnnpack,
        auto_download=auto_download_models,
        base_url=base_url,
        overwrite_models=overwrite_models,
        local_model_source=local_model_source,
    )
    encode_seconds = time.perf_counter() - encode_start

    decode_start = time.perf_counter()
    reconstruction = decode_fn(
        embeddings,
        metadata=metadata,
        models_dir=models_dir,
        threads=threads,
        use_xnnpack=use_xnnpack,
        auto_download=auto_download_models,
        base_url=base_url,
        overwrite_models=overwrite_models,
        local_model_source=local_model_source,
    )
    decode_seconds = time.perf_counter() - decode_start

    reference_aligned, recon_aligned = _align_lengths(clean, reconstruction)
    recon_aligned = np.clip(recon_aligned.astype(np.float32), -1.0, 1.0)

    metrics = compute_metrics(
        reference_aligned,
        recon_aligned,
        sample_rate,
        compute_stoi=compute_stoi,
        compute_pesq=compute_pesq,
    )

    duration_seconds = reference_aligned.shape[0] / float(sample_rate) if sample_rate else None
    rtf = (
        (encode_seconds + decode_seconds) / duration_seconds
        if duration_seconds is not None and duration_seconds > 0
        else None
    )

    return RoundtripReport(
        sample_rate_hz=sample_rate,
        metrics=metrics,
        encode_seconds=encode_seconds,
        decode_seconds=decode_seconds,
        rtf=rtf,
        reconstruction=recon_aligned,
    )
