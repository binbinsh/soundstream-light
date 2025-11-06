#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "soundstream/constants.h"
#include "soundstream/pipeline.h"

namespace py = pybind11;

namespace {

constexpr float kInt16Scale = 32768.0f;

py::array EnsureContiguous(const py::handle& source) {
  py::array array = py::array::ensure(source, py::array::c_style);
  if (!array) {
    throw std::invalid_argument("Expected a contiguous array");
  }
  return array;
}

std::vector<int16_t> ExtractPcmSamples(const py::array& array,
                                       int expected_channels) {
  py::buffer_info info = array.request();
  if (info.ndim < 1 || info.ndim > 2) {
    throw std::invalid_argument("Audio buffer must be 1D or 2D");
  }

  size_t channel_dim = 0;
  size_t samples_dim = 0;
  if (info.ndim == 1) {
    channel_dim = 1;
    samples_dim = static_cast<size_t>(info.shape[0]);
  } else {  // ndim == 2
    const size_t dim0 = static_cast<size_t>(info.shape[0]);
    const size_t dim1 = static_cast<size_t>(info.shape[1]);
    if (dim0 == static_cast<size_t>(expected_channels) && dim1 != 1) {
      channel_dim = dim0;
      samples_dim = dim1;
    } else if (dim1 == static_cast<size_t>(expected_channels) && dim0 != 1) {
      channel_dim = dim1;
      samples_dim = dim0;
    } else if (dim0 == 1) {
      channel_dim = dim0;
      samples_dim = dim1;
    } else if (dim1 == 1) {
      channel_dim = dim1;
      samples_dim = dim0;
    } else {
      throw std::invalid_argument(
          "Audio buffer must have a single channel dimension");
    }
  }

  if (channel_dim != static_cast<size_t>(expected_channels)) {
    throw std::invalid_argument("SoundStream encoder only supports mono audio");
  }

  const size_t total_samples = samples_dim;
  std::vector<int16_t> pcm(total_samples);

  const char* format = info.format.c_str();
  if (format == py::format_descriptor<int16_t>::format()) {
    std::memcpy(pcm.data(), info.ptr,
                total_samples * sizeof(int16_t));
    return pcm;
  }

  if (format == py::format_descriptor<float>::format()) {
    const float* source = static_cast<const float*>(info.ptr);
    for (size_t i = 0; i < total_samples; ++i) {
      float value = std::clamp(source[i], -1.0f, 1.0f);
      pcm[i] = static_cast<int16_t>(std::round(value * kInt16Scale));
    }
    return pcm;
  }

  if (format == py::format_descriptor<double>::format()) {
    const double* source = static_cast<const double*>(info.ptr);
    for (size_t i = 0; i < total_samples; ++i) {
      double value = std::clamp(source[i], -1.0, 1.0);
      pcm[i] = static_cast<int16_t>(
          std::round(static_cast<float>(value) * kInt16Scale));
    }
    return pcm;
  }

  throw std::invalid_argument(
      "Audio buffer must be int16, float32, or float64");
}

py::array_t<float> EmbeddingsToArray(const soundstream::EmbeddingFile& data) {
  const size_t frames = data.frame_count();
  if (frames == 0 || data.embedding_dim <= 0) {
    return py::array_t<float>(
        py::array::ShapeContainer{py::ssize_t{0}, py::ssize_t{0}});
  }

  py::array_t<float> array(py::array::ShapeContainer{
      static_cast<py::ssize_t>(frames),
      static_cast<py::ssize_t>(data.embedding_dim)});
  auto mutable_data = array.mutable_data();
  std::memcpy(mutable_data, data.embeddings.data(),
              data.embeddings.size() * sizeof(float));
  return array;
}

py::array_t<float> SamplesToFloatArray(const std::vector<int16_t>& samples) {
  py::array_t<float> array(static_cast<py::ssize_t>(samples.size()));
  float* dest = array.mutable_data();
  for (size_t i = 0; i < samples.size(); ++i) {
    dest[i] = static_cast<float>(samples[i]) / kInt16Scale;
  }
  return array;
}

}  // namespace

PYBIND11_MODULE(_native, m) {
  m.doc() = "SoundStream Light native bindings";

  m.def(
      "encode",
      [](py::handle wav,
         int sample_rate_hz,
         int num_channels,
         const std::string& model_dir,
         bool use_xnnpack,
         int num_threads) {
        if (sample_rate_hz != soundstream::kInternalSampleRateHz) {
          throw std::invalid_argument(
              "SoundStream encoder expects 16 kHz mono audio");
        }
        if (num_channels != 1) {
          throw std::invalid_argument(
              "SoundStream encoder only supports mono audio");
        }

        py::array wav_array = EnsureContiguous(wav);
        std::vector<int16_t> pcm =
            ExtractPcmSamples(wav_array, num_channels);

        soundstream::WavData wav_data;
        wav_data.sample_rate_hz = sample_rate_hz;
        wav_data.num_channels = num_channels;
        wav_data.samples = std::move(pcm);

        soundstream::EncodeBufferOptions options;
        options.model_dir = model_dir;
        options.use_xnnpack = use_xnnpack;
        options.num_threads = num_threads;

        auto embedding = soundstream::EncodeBuffer(options, wav_data);
        if (!embedding.has_value()) {
          throw std::runtime_error("Failed to encode audio");
        }

        py::array embeddings = EmbeddingsToArray(*embedding);
        py::dict metadata;
        metadata["sample_rate_hz"] = embedding->sample_rate_hz;
        metadata["num_channels"] = embedding->num_channels;
        metadata["original_num_samples"] = embedding->original_num_samples;
        metadata["embedding_dim"] = embedding->embedding_dim;

        return py::make_tuple(std::move(embeddings), std::move(metadata));
      },
      py::arg("wav"),
      py::arg("sample_rate_hz"),
      py::arg("num_channels") = 1,
      py::arg("model_dir"),
      py::arg("use_xnnpack") = true,
      py::arg("num_threads") = 1);

  m.def(
      "decode",
      [](py::handle embeddings,
         int sample_rate_hz,
         int num_channels,
         int original_num_samples,
         const std::string& model_dir,
         bool use_xnnpack,
         int num_threads) {
        if (sample_rate_hz != soundstream::kInternalSampleRateHz) {
          throw std::invalid_argument(
              "SoundStream decoder expects embeddings at 16 kHz");
        }
        if (num_channels != 1) {
          throw std::invalid_argument(
              "SoundStream decoder only supports mono audio");
        }

        py::array embeddings_array = EnsureContiguous(embeddings);
        py::buffer_info info = embeddings_array.request();
        if (info.ndim != 2) {
          throw std::invalid_argument(
              "Embeddings must be a 2D array of shape (frames, dim)");
        }

        const size_t frames = static_cast<size_t>(info.shape[0]);
        const size_t dim = static_cast<size_t>(info.shape[1]);
        std::vector<float> buffer(frames * dim);

        const char* format = info.format.c_str();
        if (format == py::format_descriptor<float>::format()) {
          std::memcpy(buffer.data(), info.ptr, buffer.size() * sizeof(float));
        } else if (format == py::format_descriptor<double>::format()) {
          const double* source = static_cast<const double*>(info.ptr);
          for (size_t i = 0; i < buffer.size(); ++i) {
            buffer[i] = static_cast<float>(source[i]);
          }
        } else {
          throw std::invalid_argument(
              "Embeddings must be float32 or float64");
        }

        soundstream::EmbeddingFile data;
        data.sample_rate_hz = sample_rate_hz;
        data.num_channels = num_channels;
        data.embedding_dim = static_cast<int>(dim);
        data.original_num_samples = original_num_samples;
        data.embeddings = std::move(buffer);

        soundstream::DecodeBufferOptions options;
        options.model_dir = model_dir;
        options.use_xnnpack = use_xnnpack;
        options.num_threads = num_threads;

        auto wav = soundstream::DecodeBuffer(options, data);
        if (!wav.has_value()) {
          throw std::runtime_error("Failed to decode embeddings");
        }

        return SamplesToFloatArray(wav->samples);
      },
      py::arg("embeddings"),
      py::arg("sample_rate_hz"),
      py::arg("num_channels") = 1,
      py::arg("original_num_samples"),
      py::arg("model_dir"),
      py::arg("use_xnnpack") = true,
      py::arg("num_threads") = 1);
}

