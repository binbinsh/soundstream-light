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

#ifndef SOUNDSTREAM_EMBEDDING_IO_H_
#define SOUNDSTREAM_EMBEDDING_IO_H_

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace soundstream {

struct EmbeddingFile {
  int sample_rate_hz;
  int num_channels;
  int embedding_dim;
  int original_num_samples;
  std::vector<float> embeddings;  // Frames laid out sequentially.

  size_t frame_count() const {
    if (embedding_dim == 0) {
      return 0;
    }
    return embeddings.size() / static_cast<size_t>(embedding_dim);
  }
};

bool WriteEmbeddings(const std::filesystem::path& path,
                     const EmbeddingFile& data);

std::optional<EmbeddingFile> ReadEmbeddings(
    const std::filesystem::path& path);

}  // namespace soundstream

#endif  // SOUNDSTREAM_EMBEDDING_IO_H_

