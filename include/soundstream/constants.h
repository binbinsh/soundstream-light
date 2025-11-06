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

#ifndef SOUNDSTREAM_CONSTANTS_H_
#define SOUNDSTREAM_CONSTANTS_H_

namespace soundstream {

inline constexpr int kFrameRate = 50;               // Frames per second.
inline constexpr int kInternalSampleRateHz = 16000; // Model sample rate.
inline constexpr int kNumFeatures = 64;             // Embedding dimension.
inline constexpr int kNumSamplesPerHop =
    kInternalSampleRateHz / kFrameRate;             // Samples per embedding.

inline constexpr const char kEncoderModelName[] = "soundstream_encoder.tflite";
inline constexpr const char kDecoderModelName[] = "lyragan.tflite";

}  // namespace soundstream

#endif  // SOUNDSTREAM_CONSTANTS_H_

