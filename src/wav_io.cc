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

#include "soundstream/wav_io.h"

#include <array>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

namespace {

template <typename T>
bool ReadLittleEndian(std::istream& stream, T* value) {
  static_assert(std::is_integral_v<T>, "Integral type required");
  std::array<std::byte, sizeof(T)> buffer{};
  if (!stream.read(reinterpret_cast<char*>(buffer.data()), buffer.size())) {
    return false;
  }
  T result = 0;
  for (size_t i = 0; i < buffer.size(); ++i) {
    const auto byte = static_cast<unsigned char>(buffer[i]);
    result |= static_cast<T>(static_cast<T>(byte) << (8 * i));
  }
  *value = result;
  return true;
}

void WriteLittleEndian(std::ostream& stream, uint32_t value) {
  for (int i = 0; i < 4; ++i) {
    const char byte = static_cast<char>((value >> (8 * i)) & 0xFF);
    stream.put(byte);
  }
}

void WriteLittleEndian(std::ostream& stream, uint16_t value) {
  for (int i = 0; i < 2; ++i) {
    const char byte = static_cast<char>((value >> (8 * i)) & 0xFF);
    stream.put(byte);
  }
}

}  // namespace

namespace soundstream {

std::optional<WavData> ReadWav16(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open WAV file: " << path << std::endl;
    return std::nullopt;
  }

  auto require_chunk = [&](const char* expected) {
    char chunk[4];
    if (!file.read(chunk, 4)) {
      return false;
    }
    return std::memcmp(chunk, expected, 4) == 0;
  };

  if (!require_chunk("RIFF")) {
    std::cerr << "Missing RIFF header in WAV file: " << path << std::endl;
    return std::nullopt;
  }

  uint32_t riff_size = 0;
  if (!ReadLittleEndian(file, &riff_size)) {
    std::cerr << "Failed to read RIFF chunk size in: " << path << std::endl;
    return std::nullopt;
  }
  (void)riff_size;

  if (!require_chunk("WAVE")) {
    std::cerr << "Missing WAVE header in WAV file: " << path << std::endl;
    return std::nullopt;
  }

  bool fmt_parsed = false;
  uint16_t audio_format = 0;
  uint16_t num_channels = 0;
  uint32_t sample_rate = 0;
  uint16_t bits_per_sample = 0;
  std::vector<int16_t> samples;

  while (file && !file.eof()) {
    char chunk_id[4];
    if (!file.read(chunk_id, 4)) {
      break;
    }
    uint32_t chunk_size = 0;
    if (!ReadLittleEndian(file, &chunk_size)) {
      std::cerr << "Failed to read chunk size in WAV file: " << path
                << std::endl;
      return std::nullopt;
    }

    const std::streampos next_chunk =
        file.tellg() + static_cast<std::streamoff>(chunk_size);

    if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
      if (chunk_size < 16) {
        std::cerr << "fmt chunk too small in WAV file: " << path
                  << std::endl;
        return std::nullopt;
      }
      if (!ReadLittleEndian(file, &audio_format) ||
          !ReadLittleEndian(file, &num_channels) ||
          !ReadLittleEndian(file, &sample_rate)) {
        std::cerr << "Failed to read fmt chunk fields in: " << path
                  << std::endl;
        return std::nullopt;
      }

      uint32_t byte_rate = 0;
      uint16_t block_align = 0;
      if (!ReadLittleEndian(file, &byte_rate) ||
          !ReadLittleEndian(file, &block_align) ||
          !ReadLittleEndian(file, &bits_per_sample)) {
        std::cerr << "Failed to read fmt chunk remainder in: " << path
                  << std::endl;
        return std::nullopt;
      }

      if (audio_format != 1) {
        std::cerr
            << "Unsupported WAV audio format (expected PCM 16-bit) in: "
            << path << std::endl;
        return std::nullopt;
      }
      if (bits_per_sample != 16) {
        std::cerr << "Unsupported WAV bit depth (expected 16) in: " << path
                  << std::endl;
        return std::nullopt;
      }
      if (block_align == 0 || num_channels == 0) {
        std::cerr << "Invalid WAV fmt values in: " << path << std::endl;
        return std::nullopt;
      }

      fmt_parsed = true;
    } else if (std::memcmp(chunk_id, "data", 4) == 0) {
      if (!fmt_parsed) {
        std::cerr
            << "Encountered data chunk before fmt chunk in WAV file: "
            << path << std::endl;
        return std::nullopt;
      }
      const size_t num_samples = chunk_size / sizeof(int16_t);
      samples.resize(num_samples);
      if (!file.read(reinterpret_cast<char*>(samples.data()), chunk_size)) {
        std::cerr << "Failed to read sample data from: " << path << std::endl;
        return std::nullopt;
      }
    } else {
      file.seekg(next_chunk);
    }

    file.seekg(next_chunk);
  }

  if (!fmt_parsed || samples.empty()) {
    std::cerr << "Failed to parse WAV file or no audio data: " << path
              << std::endl;
    return std::nullopt;
  }

  WavData data;
  data.sample_rate_hz = static_cast<int>(sample_rate);
  data.num_channels = static_cast<int>(num_channels);
  data.samples = std::move(samples);
  return data;
}

bool WriteWav16(const std::filesystem::path& path, const WavData& data) {
  if (data.num_channels <= 0) {
    std::cerr << "Number of channels must be positive when writing WAV."
              << '\n';
    return false;
  }
  if (data.sample_rate_hz <= 0) {
    std::cerr << "Sample rate must be positive when writing WAV." << '\n';
    return false;
  }

  std::ofstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open WAV file for writing: " << path << std::endl;
    return false;
  }

  const uint32_t data_chunk_size =
      static_cast<uint32_t>(data.samples.size() * sizeof(int16_t));
  const uint32_t fmt_chunk_size = 16;
  const uint32_t riff_chunk_size =
      4 + (8 + fmt_chunk_size) + (8 + data_chunk_size);

  file.write("RIFF", 4);
  WriteLittleEndian(file, riff_chunk_size);
  file.write("WAVE", 4);

  file.write("fmt ", 4);
  WriteLittleEndian(file, fmt_chunk_size);
  WriteLittleEndian(file, static_cast<uint16_t>(1));
  WriteLittleEndian(file, static_cast<uint16_t>(data.num_channels));
  WriteLittleEndian(file, static_cast<uint32_t>(data.sample_rate_hz));

  const uint32_t byte_rate = static_cast<uint32_t>(data.sample_rate_hz) *
                             static_cast<uint32_t>(data.num_channels) * 2;
  const uint16_t block_align = static_cast<uint16_t>(data.num_channels * 2);
  const uint16_t bits_per_sample = 16;
  WriteLittleEndian(file, byte_rate);
  WriteLittleEndian(file, block_align);
  WriteLittleEndian(file, bits_per_sample);

  file.write("data", 4);
  WriteLittleEndian(file, data_chunk_size);
  file.write(reinterpret_cast<const char*>(data.samples.data()),
             data_chunk_size);

  return file.good();
}

}  // namespace soundstream

