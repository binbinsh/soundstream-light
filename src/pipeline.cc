// Copyright 2024 SoundStream Light Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "soundstream/pipeline.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <optional>
#include <vector>

#include "soundstream/constants.h"
#include "soundstream/decoder.h"
#include "soundstream/embedding_io.h"
#include "soundstream/encoder.h"
#include "soundstream/wav_io.h"

namespace soundstream {

namespace {

bool EnsureModelDirectoryExists(const std::filesystem::path& dir) {
  if (!std::filesystem::exists(dir)) {
    std::cerr << "Model directory does not exist: " << dir << std::endl;
    return false;
  }
  return true;
}

std::optional<EmbeddingFile> EncodeInternal(const WavData& wav,
                                            const EncodeBufferOptions& options) {
  if (!EnsureModelDirectoryExists(options.model_dir)) {
    return std::nullopt;
  }

  if (wav.num_channels != 1) {
    std::cerr << "SoundStream encoder only supports mono audio." << std::endl;
    return std::nullopt;
  }
  if (wav.sample_rate_hz != kInternalSampleRateHz) {
    std::cerr << "SoundStream encoder expects sample rate "
              << kInternalSampleRateHz << " Hz." << std::endl;
    return std::nullopt;
  }

  auto encoder = Encoder::Create(options.model_dir, options.use_xnnpack,
                                 options.num_threads);
  if (!encoder.has_value()) {
    return std::nullopt;
  }

  const int embedding_dim = encoder->embedding_dim();
  const size_t total_samples = wav.samples.size();
  if (total_samples == 0) {
    std::cerr << "Input WAV is empty." << std::endl;
    return std::nullopt;
  }

  const size_t frame_capacity =
      (total_samples + kNumSamplesPerHop - 1) / kNumSamplesPerHop;

  std::vector<float> embeddings;
  embeddings.reserve(frame_capacity * static_cast<size_t>(embedding_dim));
  std::vector<int16_t> frame(kNumSamplesPerHop, 0);

  for (size_t offset = 0; offset < total_samples;
       offset += kNumSamplesPerHop) {
    const size_t remaining =
        std::min<size_t>(kNumSamplesPerHop, total_samples - offset);
    std::fill(frame.begin(), frame.end(), 0);
    if (remaining > 0) {
      std::copy_n(
          wav.samples.begin() + static_cast<std::ptrdiff_t>(offset),
          remaining, frame.begin());
    }

    auto embedding = encoder->EncodeFrame(frame);
    if (!embedding.has_value()) {
      return std::nullopt;
    }
    embeddings.insert(embeddings.end(), embedding->begin(), embedding->end());
  }

  EmbeddingFile file_data;
  file_data.sample_rate_hz = wav.sample_rate_hz;
  file_data.num_channels = wav.num_channels;
  file_data.embedding_dim = embedding_dim;
  file_data.original_num_samples =
      static_cast<int>(wav.samples.size());
  file_data.embeddings = std::move(embeddings);

  return file_data;
}

std::optional<WavData> DecodeInternal(const EmbeddingFile& data,
                                      const DecodeBufferOptions& options) {
  if (!EnsureModelDirectoryExists(options.model_dir)) {
    return std::nullopt;
  }

  if (data.num_channels != 1) {
    std::cerr << "Only mono embeddings are supported." << std::endl;
    return std::nullopt;
  }
  if (data.sample_rate_hz != kInternalSampleRateHz) {
    std::cerr << "Embeddings were produced at " << data.sample_rate_hz
              << " Hz but decoder expects " << kInternalSampleRateHz
              << " Hz." << std::endl;
    return std::nullopt;
  }

  auto decoder = Decoder::Create(options.model_dir, options.use_xnnpack,
                                 options.num_threads);
  if (!decoder.has_value()) {
    return std::nullopt;
  }
  if (decoder->embedding_dim() != data.embedding_dim) {
    std::cerr << "Embedding dimension mismatch between stored data and"
              << " decoder." << std::endl;
    return std::nullopt;
  }

  const size_t frame_count = data.frame_count();
  if (frame_count == 0) {
    std::cerr << "Embedding file has no frames." << std::endl;
    return std::nullopt;
  }
  std::vector<int16_t> samples;
  samples.reserve(frame_count * kNumSamplesPerHop);

  for (size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
    const float* frame_ptr =
        data.embeddings.data() +
        frame_index * static_cast<size_t>(data.embedding_dim);
    std::span<const float> frame(frame_ptr, data.embedding_dim);
    auto decoded = decoder->DecodeFrame(frame);
    if (!decoded.has_value()) {
      return std::nullopt;
    }
    samples.insert(samples.end(), decoded->begin(), decoded->end());
  }

  if (data.original_num_samples >= 0 &&
      static_cast<size_t>(data.original_num_samples) < samples.size()) {
    samples.resize(static_cast<size_t>(data.original_num_samples));
  }

  WavData wav;
  wav.sample_rate_hz = data.sample_rate_hz;
  wav.num_channels = data.num_channels;
  wav.samples = std::move(samples);
  return wav;
}

}  // namespace

bool EncodeAudio(const EncodeOptions& options) {
  auto wav = ReadWav16(options.input_wav);
  if (!wav.has_value()) {
    return false;
  }
  EncodeBufferOptions buffer_options;
  buffer_options.model_dir = options.model_dir;
  buffer_options.use_xnnpack = options.use_xnnpack;
  buffer_options.num_threads = options.num_threads;

  auto embedding = EncodeBuffer(buffer_options, *wav);
  if (!embedding.has_value()) {
    return false;
  }
  return WriteEmbeddings(options.output_embeddings, *embedding);
}

bool DecodeAudio(const DecodeOptions& options) {
  auto file_data = ReadEmbeddings(options.input_embeddings);
  if (!file_data.has_value()) {
    return false;
  }
  DecodeBufferOptions buffer_options;
  buffer_options.model_dir = options.model_dir;
  buffer_options.use_xnnpack = options.use_xnnpack;
  buffer_options.num_threads = options.num_threads;

  auto wav = DecodeBuffer(buffer_options, *file_data);
  if (!wav.has_value()) {
    return false;
  }
  return WriteWav16(options.output_wav, *wav);
}

std::optional<EmbeddingFile> EncodeBuffer(const EncodeBufferOptions& options,
                                          const WavData& wav) {
  return EncodeInternal(wav, options);
}

std::optional<WavData> DecodeBuffer(const DecodeBufferOptions& options,
                                    const EmbeddingFile& data) {
  return DecodeInternal(data, options);
}

}  // namespace soundstream

