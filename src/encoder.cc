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

#include "soundstream/encoder.h"

#include <algorithm>
#include <iostream>
#include <limits>

namespace {

inline float Int16ToUnit(int16_t value) {
  return -static_cast<float>(value) /
         static_cast<float>(std::numeric_limits<int16_t>::min());
}

}  // namespace

namespace soundstream {

std::optional<Encoder> Encoder::Create(const std::filesystem::path& model_dir,
                                       bool use_xnnpack, int num_threads) {
  const std::filesystem::path model_path = model_dir / kEncoderModelName;
  auto wrapper = ModelWrapper::Create(model_path.string(), use_xnnpack,
                                      num_threads);
  if (!wrapper.has_value()) {
    std::cerr << "Failed to initialise SoundStream encoder." << std::endl;
    return std::nullopt;
  }

  const auto output = wrapper->OutputTensor<float>(0);
  const int embedding_dim = static_cast<int>(output.size());
  return Encoder(std::move(*wrapper), embedding_dim);
}

Encoder::Encoder(ModelWrapper&& model, int embedding_dim)
    : model_(std::move(model)), embedding_dim_(embedding_dim) {}

std::optional<std::vector<float>> Encoder::EncodeFrame(
    std::span<const int16_t> frame_samples) {
  if (frame_samples.size() != static_cast<size_t>(kNumSamplesPerHop)) {
    std::cerr << "Expected " << kNumSamplesPerHop
              << " samples per frame but received " << frame_samples.size()
              << std::endl;
    return std::nullopt;
  }

  auto input = model_.InputTensor<float>(0);
  std::transform(frame_samples.begin(), frame_samples.end(), input.begin(),
                 Int16ToUnit);

  if (!model_.Invoke()) {
    std::cerr << "Failed to run SoundStream encoder inference." << std::endl;
    return std::nullopt;
  }

  const auto output = model_.OutputTensor<float>(0);
  return std::vector<float>(output.begin(), output.end());
}

}  // namespace soundstream

