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

#include <filesystem>
#include <iostream>
#include <random>

#include "soundstream/pipeline.h"
#include "soundstream/wav_io.h"
#include "soundstream/constants.h"

namespace {

std::filesystem::path TempFile(const std::string& prefix,
                               const std::string& suffix) {
  auto rng = std::mt19937{std::random_device{}()};
  auto dist = std::uniform_int_distribution<int>{0, 1'000'000};
  const auto temp_dir = std::filesystem::temp_directory_path();
  std::filesystem::path candidate;
  do {
    candidate = temp_dir / (prefix + std::to_string(dist(rng)) + suffix);
  } while (std::filesystem::exists(candidate));
  return candidate;
}

}  // namespace

int main() {
  const std::filesystem::path models_path = SOUNDSTREAM_MODELS_PATH;
  const std::filesystem::path sample_path = SOUNDSTREAM_SAMPLE_WAV;

  if (!std::filesystem::exists(models_path)) {
    std::cerr << "Model directory missing: " << models_path << std::endl;
    return 1;
  }

  if (!std::filesystem::exists(sample_path)) {
    std::cerr << "Sample WAV missing: " << sample_path << std::endl;
    return 1;
  }

  auto source_wav = soundstream::ReadWav16(sample_path);
  if (!source_wav.has_value()) {
    std::cerr << "Failed to read sample WAV: " << sample_path << std::endl;
    return 1;
  }

  const std::filesystem::path embeddings_path = TempFile("soundstream", ".sse");
  const std::filesystem::path output_wav = TempFile("soundstream", ".wav");

  soundstream::EncodeOptions encode_options;
  encode_options.input_wav = sample_path;
  encode_options.output_embeddings = embeddings_path;
  encode_options.model_dir = models_path;

  if (!soundstream::EncodeAudio(encode_options)) {
    std::cerr << "Encoding failed" << std::endl;
    return 1;
  }

  soundstream::DecodeOptions decode_options;
  decode_options.input_embeddings = embeddings_path;
  decode_options.output_wav = output_wav;
  decode_options.model_dir = models_path;

  if (!soundstream::DecodeAudio(decode_options)) {
    std::cerr << "Decoding failed" << std::endl;
    return 1;
  }

  auto original = soundstream::ReadWav16(sample_path);
  auto decoded = soundstream::ReadWav16(output_wav);
  if (!original.has_value() || !decoded.has_value()) {
    std::cerr << "Unable to read back WAV files" << std::endl;
    return 1;
  }

  if (decoded->samples.empty()) {
    std::cerr << "Decoded output is empty" << std::endl;
    return 1;
  }

  const size_t original_samples = original->samples.size();
  const size_t decoded_samples = decoded->samples.size();
  const double ratio = static_cast<double>(decoded_samples) /
                       static_cast<double>(original_samples);
  if (ratio < 0.8 || ratio > 1.25) {
    std::cerr << "Decoded sample count deviates too much" << std::endl;
    return 1;
  }

  std::filesystem::remove(embeddings_path);
  std::filesystem::remove(output_wav);
  return 0;
}

