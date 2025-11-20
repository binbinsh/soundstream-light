from __future__ import annotations

import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from types import ModuleType

import numpy as np  # type: ignore[import-not-found]

_stub_native = ModuleType("soundstream_light._native")
_stub_native.encode = lambda *args, **kwargs: None  # type: ignore[assignment]
_stub_native.decode = lambda *args, **kwargs: None  # type: ignore[assignment]
sys.modules.setdefault("soundstream_light._native", _stub_native)

from soundstream_light import api, metrics


class MetricsTests(TestCase):
    def test_si_snr_prefers_cleaner_signal(self) -> None:
        reference = np.array([1.0, 0.0, -1.0, 0.5], dtype=np.float32)
        slightly_noisy = reference + 0.05
        inverted = -reference

        cleaner_score = metrics.si_snr(reference, slightly_noisy)
        noisier_score = metrics.si_snr(reference, inverted)
        self.assertGreater(cleaner_score, noisier_score)

    def test_roundtrip_evaluation_with_stubs(self) -> None:
        sample_rate = 16000
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
        waveform = 0.1 * np.sin(2.0 * math.pi * 440.0 * t)

        with TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "tone.wav"
            metrics.write_wav_mono(wav_path, waveform, sample_rate)

            def fake_encode(wav, **kwargs):
                metadata = api.EmbeddingMetadata(
                    sample_rate_hz=sample_rate,
                    num_channels=1,
                    original_num_samples=wav.shape[0],
                    embedding_dim=4,
                )
                return np.zeros((2, 4), dtype=np.float32), metadata

            def fake_decode(embeddings, metadata=None, **kwargs):
                # Simulate a slightly attenuated reconstruction.
                return waveform * 0.9

            report = metrics.evaluate_roundtrip(
                wav_path,
                compute_stoi=False,
                compute_pesq=False,
                encode_fn=fake_encode,
                decode_fn=fake_decode,
            )

            expected_snr = metrics.si_snr(waveform, waveform * 0.9)
            self.assertAlmostEqual(report.metrics["si_snr_db"], expected_snr, places=4)
            self.assertIsNotNone(report.rtf)
            self.assertGreaterEqual(report.rtf, 0.0)
