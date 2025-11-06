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

#include <charconv>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "soundstream/pipeline.h"

using soundstream::DecodeAudio;
using soundstream::DecodeOptions;
using soundstream::EncodeAudio;
using soundstream::EncodeOptions;

namespace {

void PrintUsage() {
  std::cout << "Usage:\n"
            << "  soundstream_cli encode --wav <input.wav> --embeddings <out.sse>"
            << " --models <dir> [--threads N] [--no-xnn]\n"
            << "  soundstream_cli decode --embeddings <input.sse> --wav "
            << "<out.wav> --models <dir> [--threads N] [--no-xnn]\n";
}

std::optional<int> ParseInt(std::string_view value) {
  int result = 0;
  const auto* begin = value.data();
  const auto* end = begin + value.size();
  const auto status = std::from_chars(begin, end, result);
  if (status.ec != std::errc() || status.ptr != end) {
    return std::nullopt;
  }
  return result;
}

struct CommonFlags {
  std::filesystem::path models;
  int threads = 1;
  bool use_xnnpack = true;
};

bool ParseCommonFlags(int argc, char** argv, int start_index, CommonFlags* flags,
                      std::vector<int>* consumed_indices) {
  for (int i = start_index; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg == "--models" && i + 1 < argc) {
      flags->models = argv[i + 1];
      consumed_indices->push_back(i);
      consumed_indices->push_back(i + 1);
      ++i;
    } else if (arg == "--threads" && i + 1 < argc) {
      auto value = ParseInt(argv[i + 1]);
      if (!value.has_value() || value.value() <= 0) {
        std::cerr << "Invalid value for --threads." << std::endl;
        return false;
      }
      flags->threads = value.value();
      consumed_indices->push_back(i);
      consumed_indices->push_back(i + 1);
      ++i;
    } else if (arg == "--no-xnn") {
      flags->use_xnnpack = false;
      consumed_indices->push_back(i);
    }
  }
  if (flags->models.empty()) {
    std::cerr << "--models is required." << std::endl;
    return false;
  }
  return true;
}

int HandleEncode(int argc, char** argv) {
  EncodeOptions options;
  CommonFlags common;
  std::vector<int> consumed;
  if (!ParseCommonFlags(argc, argv, 2, &common, &consumed)) {
    return 1;
  }
  options.model_dir = common.models;
  options.num_threads = common.threads;
  options.use_xnnpack = common.use_xnnpack;

  for (int i = 2; i < argc; ++i) {
    if (std::find(consumed.begin(), consumed.end(), i) != consumed.end()) {
      continue;
    }
    std::string_view arg(argv[i]);
    if (arg == "--wav" && i + 1 < argc) {
      options.input_wav = argv[i + 1];
      consumed.push_back(i);
      consumed.push_back(i + 1);
      ++i;
    } else if (arg == "--embeddings" && i + 1 < argc) {
      options.output_embeddings = argv[i + 1];
      consumed.push_back(i);
      consumed.push_back(i + 1);
      ++i;
    }
  }

  if (options.input_wav.empty() || options.output_embeddings.empty()) {
    std::cerr << "--wav and --embeddings are required for encode." << std::endl;
    return 1;
  }

  if (!EncodeAudio(options)) {
    return 1;
  }
  return 0;
}

int HandleDecode(int argc, char** argv) {
  DecodeOptions options;
  CommonFlags common;
  std::vector<int> consumed;
  if (!ParseCommonFlags(argc, argv, 2, &common, &consumed)) {
    return 1;
  }
  options.model_dir = common.models;
  options.num_threads = common.threads;
  options.use_xnnpack = common.use_xnnpack;

  for (int i = 2; i < argc; ++i) {
    if (std::find(consumed.begin(), consumed.end(), i) != consumed.end()) {
      continue;
    }
    std::string_view arg(argv[i]);
    if (arg == "--embeddings" && i + 1 < argc) {
      options.input_embeddings = argv[i + 1];
      consumed.push_back(i);
      consumed.push_back(i + 1);
      ++i;
    } else if (arg == "--wav" && i + 1 < argc) {
      options.output_wav = argv[i + 1];
      consumed.push_back(i);
      consumed.push_back(i + 1);
      ++i;
    }
  }

  if (options.input_embeddings.empty() || options.output_wav.empty()) {
    std::cerr << "--embeddings and --wav are required for decode." << std::endl;
    return 1;
  }

  if (!DecodeAudio(options)) {
    return 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    PrintUsage();
    return 1;
  }

  std::string_view command(argv[1]);
  if (command == "encode") {
    return HandleEncode(argc, argv);
  }
  if (command == "decode") {
    return HandleDecode(argc, argv);
  }
  if (command == "--help" || command == "-h") {
    PrintUsage();
    return 0;
  }

  std::cerr << "Unknown subcommand: " << command << std::endl;
  PrintUsage();
  return 1;
}

