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

#ifndef SOUNDSTREAM_DECODER_H_
#define SOUNDSTREAM_DECODER_H_

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "soundstream/constants.h"
#include "soundstream/model_wrapper.h"

namespace soundstream {

class Decoder {
 public:
  static std::optional<Decoder> Create(const std::filesystem::path& model_dir,
                                       bool use_xnnpack = true,
                                       int num_threads = 1);

  Decoder(Decoder&&) noexcept = default;
  Decoder& operator=(Decoder&&) noexcept = default;

  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;

  int embedding_dim() const { return embedding_dim_; }

  std::optional<std::vector<int16_t>> DecodeFrame(
      std::span<const float> embedding);

 private:
  Decoder(ModelWrapper&& model, int embedding_dim);

  ModelWrapper model_;
  int embedding_dim_;
};

}  // namespace soundstream

#endif  // SOUNDSTREAM_DECODER_H_

