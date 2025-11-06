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

#include "soundstream/embedding_io.h"

#include <array>
#include <fstream>
#include <iostream>
#include <optional>
#include <vector>

namespace {

constexpr std::array<char, 4> kMagic = {'S', 'S', 'E', '0'};
constexpr uint32_t kFormatVersion = 1;

template <typename T>
void WriteLittleEndian(std::ostream& stream, T value) {
  for (size_t i = 0; i < sizeof(T); ++i) {
    const char byte = static_cast<char>((static_cast<uint64_t>(value) >>
                                         (8 * i)) & 0xFF);
    stream.put(byte);
  }
}

template <typename T>
bool ReadLittleEndian(std::istream& stream, T* value) {
  std::array<char, sizeof(T)> buffer{};
  if (!stream.read(buffer.data(), buffer.size())) {
    return false;
  }
  T result = 0;
  for (size_t i = 0; i < buffer.size(); ++i) {
    result |= static_cast<T>(static_cast<unsigned char>(buffer[i]) << (8 * i));
  }
  *value = result;
  return true;
}

}  // namespace

namespace soundstream {

bool WriteEmbeddings(const std::filesystem::path& path,
                     const EmbeddingFile& data) {
  if (data.embedding_dim <= 0) {
    std::cerr << "Embedding dimension must be positive." << std::endl;
    return false;
  }
  if (data.sample_rate_hz <= 0 || data.num_channels <= 0) {
    std::cerr << "Sample rate and channel count must be positive."
              << std::endl;
    return false;
  }
  if (data.original_num_samples < 0) {
    std::cerr << "Original sample count cannot be negative." << std::endl;
    return false;
  }
  if (data.embeddings.size() % static_cast<size_t>(data.embedding_dim) != 0) {
    std::cerr << "Embedding buffer size is not divisible by embedding dim."
              << std::endl;
    return false;
  }

  std::ofstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open embedding file for writing: " << path
              << std::endl;
    return false;
  }

  file.write(kMagic.data(), kMagic.size());
  WriteLittleEndian<uint32_t>(file, kFormatVersion);
  WriteLittleEndian<uint32_t>(file, static_cast<uint32_t>(data.sample_rate_hz));
  WriteLittleEndian<uint32_t>(file, static_cast<uint32_t>(data.num_channels));
  WriteLittleEndian<uint32_t>(file, static_cast<uint32_t>(data.embedding_dim));
  WriteLittleEndian<uint32_t>(file,
                              static_cast<uint32_t>(data.frame_count()));
  WriteLittleEndian<uint32_t>(file,
                              static_cast<uint32_t>(data.original_num_samples));

  file.write(reinterpret_cast<const char*>(data.embeddings.data()),
             data.embeddings.size() * sizeof(float));
  return file.good();
}

std::optional<EmbeddingFile> ReadEmbeddings(
    const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open embedding file: " << path << std::endl;
    return std::nullopt;
  }

  std::array<char, 4> magic{};
  if (!file.read(magic.data(), magic.size()) || magic != kMagic) {
    std::cerr << "Invalid embedding file magic in: " << path << std::endl;
    return std::nullopt;
  }

  uint32_t version = 0;
  if (!ReadLittleEndian(file, &version) || version != kFormatVersion) {
    std::cerr << "Unsupported embedding file version in: " << path
              << std::endl;
    return std::nullopt;
  }

  uint32_t sample_rate = 0;
  uint32_t num_channels = 0;
  uint32_t embedding_dim = 0;
  uint32_t frame_count = 0;
  uint32_t original_num_samples = 0;
  if (!ReadLittleEndian(file, &sample_rate) ||
      !ReadLittleEndian(file, &num_channels) ||
      !ReadLittleEndian(file, &embedding_dim) ||
      !ReadLittleEndian(file, &frame_count) ||
      !ReadLittleEndian(file, &original_num_samples)) {
    std::cerr << "Failed to read embedding file header from: " << path
              << std::endl;
    return std::nullopt;
  }

  const size_t expected_values =
      static_cast<size_t>(embedding_dim) * static_cast<size_t>(frame_count);
  std::vector<float> embeddings(expected_values);
  if (!file.read(reinterpret_cast<char*>(embeddings.data()),
                 embeddings.size() * sizeof(float))) {
    std::cerr << "Failed to read embedding payload from: " << path
              << std::endl;
    return std::nullopt;
  }

  EmbeddingFile result;
  result.sample_rate_hz = static_cast<int>(sample_rate);
  result.num_channels = static_cast<int>(num_channels);
  result.embedding_dim = static_cast<int>(embedding_dim);
  result.original_num_samples = static_cast<int>(original_num_samples);
  result.embeddings = std::move(embeddings);
  return result;
}

}  // namespace soundstream

