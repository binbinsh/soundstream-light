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

#ifndef SOUNDSTREAM_WAV_IO_H_
#define SOUNDSTREAM_WAV_IO_H_

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace soundstream {

struct WavData {
  int sample_rate_hz;
  int num_channels;
  std::vector<int16_t> samples;
};

std::optional<WavData> ReadWav16(const std::filesystem::path& path);

bool WriteWav16(const std::filesystem::path& path, const WavData& data);

}  // namespace soundstream

#endif  // SOUNDSTREAM_WAV_IO_H_

