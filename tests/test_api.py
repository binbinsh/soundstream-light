from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np  # type: ignore[import-not-found]

_stub_native = ModuleType("soundstream_light._native")
_stub_native.encode = lambda *args, **kwargs: None  # type: ignore[assignment]
_stub_native.decode = lambda *args, **kwargs: None  # type: ignore[assignment]
sys.modules.setdefault("soundstream_light._native", _stub_native)

from soundstream_light import api


class ApiTests(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.native = SimpleNamespace(
            encode=MagicMock(
                return_value=(
                    np.ones((2, 4), dtype=np.float32),
                    {
                        "sample_rate_hz": 16000,
                        "num_channels": 1,
                        "original_num_samples": 320,
                        "embedding_dim": 4,
                    },
                )
            ),
            decode=MagicMock(return_value=np.linspace(-1.0, 1.0, 6, dtype=np.float32)),
        )
        self.ensure_models_patch = patch(
            "soundstream_light.api.ensure_models", return_value=Path("/tmp/models")
        )
        self.native_patch = patch("soundstream_light.api._NATIVE", self.native)
        self.module_patch = patch.dict(
            sys.modules, {"soundstream_light._native": self.native}, clear=False
        )
        self.ensure_models_mock = self.ensure_models_patch.start()
        self.native_patch.start()
        self.module_patch.start()
        self.addCleanup(self.ensure_models_patch.stop)
        self.addCleanup(self.native_patch.stop)
        self.addCleanup(self.module_patch.stop)

    def test_encode_returns_numpy_and_metadata(self) -> None:
        wav = np.zeros(320, dtype=np.int16)
        embeddings, metadata = api.encode(wav, auto_download=False)

        self.ensure_models_mock.assert_called_once()
        self.native.encode.assert_called_once()
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, (2, 4))
        self.assertEqual(metadata.sample_rate_hz, 16000)
        self.assertEqual(metadata.original_num_samples, 320)

    def test_decode_uses_metadata(self) -> None:
        embeddings = np.ones((2, 4), dtype=np.float32)
        metadata = api.EmbeddingMetadata(16000, 1, 320, 4)
        waveform = api.decode(embeddings, metadata=metadata, auto_download=False)

        self.native.decode.assert_called_once()
        self.assertIsInstance(waveform, np.ndarray)
        self.assertEqual(waveform.dtype, np.float32)

    def test_decode_requires_torch_when_requested(self) -> None:
        with patch("soundstream_light.api._TORCH_AVAILABLE", False):
            with self.assertRaises(RuntimeError):
                api.decode(
                    np.ones((1, 4), dtype=np.float32),
                    metadata=api.EmbeddingMetadata(16000, 1, 160, 4),
                    auto_download=False,
                    return_torch=True,
                )

