// Copyright 2024 SoundStream Light Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under this License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SOUNDSTREAM_PIPELINE_H_
#define SOUNDSTREAM_PIPELINE_H_

#include <filesystem>
#include <optional>

#include "soundstream/embedding_io.h"
#include "soundstream/wav_io.h"

namespace soundstream {

struct EncodeOptions {
  std::filesystem::path input_wav;
  std::filesystem::path output_embeddings;
  std::filesystem::path model_dir;
  bool use_xnnpack = true;
  int num_threads = 1;
};

struct DecodeOptions {
  std::filesystem::path input_embeddings;
  std::filesystem::path output_wav;
  std::filesystem::path model_dir;
  bool use_xnnpack = true;
  int num_threads = 1;
};

bool EncodeAudio(const EncodeOptions& options);

bool DecodeAudio(const DecodeOptions& options);

struct EncodeBufferOptions {
  std::filesystem::path model_dir;
  bool use_xnnpack = true;
  int num_threads = 1;
};

struct DecodeBufferOptions {
  std::filesystem::path model_dir;
  bool use_xnnpack = true;
  int num_threads = 1;
};

std::optional<EmbeddingFile> EncodeBuffer(const EncodeBufferOptions& options,
                                          const WavData& wav);

std::optional<WavData> DecodeBuffer(const DecodeBufferOptions& options,
                                    const EmbeddingFile& data);

}  // namespace soundstream

#endif  // SOUNDSTREAM_PIPELINE_H_

