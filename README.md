# soundstream-light

Minimal SoundStream encoder/decoder tooling with a native C++ CLI and a Python API.

## Python Usage (PyPI)

Install package with `uv`,

```bash
uv pip install soundstream-light
```

Use `soundstream_light` in Python,

```python
from scipy.io import wavfile
import numpy as np
from soundstream_light import encode, decode

rate, pcm = wavfile.read("test.wav")
waveform = pcm.astype(np.float32) / 32768.0

embeddings, meta = encode(waveform)
recon = decode(embeddings, metadata=meta)
wavfile.write("test_recon.wav", rate, (np.clip(recon, -1.0, 1.0) * 32768.0).astype(np.int16))
```

The module auto-downloads models on first use if they are missing.

## Fetch Models

Use the bundled CLI to download the required `.tflite` files manually (defaults to `./models` or `SOUNDSTREAM_MODELS_DIR`). If you are working from the repository, prefix the command with `uv run` so the local package is used.

```bash
uv run soundstream-cli models fetch
```

With models present, you can test the encoder/decoder round trip using the bundled `test.wav` example:

```bash
uv run soundstream-cli encode test.wav test.embeddings
uv run soundstream-cli decode test.embeddings test_recon.wav
```

## Build from source

You can also build from source code by cloning this repo. Make sure that `git`, `cmake`, a C++17 compiler, and typical build tools (e.g. make, ninja, perl) are available.


### Prepare TensorFlow Lite headers and libraries:

```bash
bash scripts/setup_tflite_from_source.sh --clean --branch v2.15.0 --generator Ninja --build-type Release
```

### Configure and build:

```bash
cmake -S . -B build \
  -DTFLITE_SOURCE_DIR="$(pwd)/third_party/tflite-src" \
  -DTFLITE_BUILD_DIR="$(pwd)/third_party/tflite-src/cmake_build"
cmake --build build --target soundstream_cli
```

The binary appears at `build/soundstream_cli`. Then build PyPI artifacts from source,

```bash
export TFLITE_SOURCE_DIR="$(pwd)/third_party/tflite-src"
export TFLITE_BUILD_DIR="$(pwd)/third_party/tflite-src/cmake_build"
uv build
```

## Verify the Build

Run the C++ round-trip test (requires models):

```bash
# Download models once if you haven't already
uv pip install dist/soundstream_light-*.whl
uv run soundstream-cli models fetch

cmake --build build --target soundstream_roundtrip_test
ctest --test-dir build --output-on-failure
```

Python unit tests can be executed without compiling the extension:

```bash
uv run python -m unittest tests.test_api
```

## License

Apache License 2.0 — see `LICENSE`.

## Acknowledgements

This project builds on the source code of [Lyra](https://github.com/google/lyra).
