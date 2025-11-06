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

#include "soundstream/decoder.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace {

inline float ClipToUnit(float value) {
  return std::clamp(value, -1.0f, 1.0f);
}

inline int16_t UnitToInt16(float value) {
  const float clipped = ClipToUnit(value);
  const float scaled =
      clipped * -static_cast<float>(std::numeric_limits<int16_t>::min());
  const float rounded = std::nearbyint(scaled);
  if (rounded > static_cast<float>(std::numeric_limits<int16_t>::max())) {
    return std::numeric_limits<int16_t>::max();
  }
  if (rounded < static_cast<float>(std::numeric_limits<int16_t>::min())) {
    return std::numeric_limits<int16_t>::min();
  }
  return static_cast<int16_t>(rounded);
}

}  // namespace

namespace soundstream {

std::optional<Decoder> Decoder::Create(const std::filesystem::path& model_dir,
                                       bool use_xnnpack, int num_threads) {
  const std::filesystem::path model_path = model_dir / kDecoderModelName;
  auto wrapper = ModelWrapper::Create(model_path.string(), use_xnnpack,
                                      num_threads);
  if (!wrapper.has_value()) {
    std::cerr << "Failed to initialise SoundStream decoder." << std::endl;
    return std::nullopt;
  }

  const auto input = wrapper->InputTensor<float>(0);
  const int embedding_dim = static_cast<int>(input.size());
  return Decoder(std::move(*wrapper), embedding_dim);
}

Decoder::Decoder(ModelWrapper&& model, int embedding_dim)
    : model_(std::move(model)), embedding_dim_(embedding_dim) {}

std::optional<std::vector<int16_t>> Decoder::DecodeFrame(
    std::span<const float> embedding) {
  if (embedding.size() != static_cast<size_t>(embedding_dim_)) {
    std::cerr << "Expected embedding dimension " << embedding_dim_
              << " but received " << embedding.size() << std::endl;
    return std::nullopt;
  }

  auto input = model_.InputTensor<float>(0);
  std::copy(embedding.begin(), embedding.end(), input.begin());

  if (!model_.Invoke()) {
    std::cerr << "Failed to run SoundStream decoder inference." << std::endl;
    return std::nullopt;
  }

  const auto output = model_.OutputTensor<float>(0);
  std::vector<int16_t> samples;
  samples.reserve(output.size());
  std::transform(output.begin(), output.end(), std::back_inserter(samples),
                 UnitToInt16);
  return samples;
}

}  // namespace soundstream

