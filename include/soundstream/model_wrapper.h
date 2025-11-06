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

#ifndef SOUNDSTREAM_MODEL_WRAPPER_H_
#define SOUNDSTREAM_MODEL_WRAPPER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/signature_runner.h"

namespace soundstream {

class ModelWrapper {
 public:
  static std::optional<ModelWrapper> Create(const std::string& model_path,
                                            bool use_xnnpack,
                                            int num_threads = 1);

  ModelWrapper(ModelWrapper&&) noexcept = default;
  ModelWrapper& operator=(ModelWrapper&&) noexcept = default;

  ModelWrapper(const ModelWrapper&) = delete;
  ModelWrapper& operator=(const ModelWrapper&) = delete;

  tflite::Interpreter* interpreter() { return interpreter_.get(); }
  const tflite::Interpreter* interpreter() const { return interpreter_.get(); }

  bool Invoke();

  template <class T>
  std::span<T> InputTensor(int index) {
    auto* tensor = interpreter_->typed_input_tensor<T>(index);
    auto* metadata = interpreter_->input_tensor(index);
    return std::span<T>(tensor, metadata->bytes / sizeof(T));
  }

  template <class T>
  std::span<const T> OutputTensor(int index) {
    auto* tensor = interpreter_->typed_output_tensor<T>(index);
    auto* metadata = interpreter_->output_tensor(index);
    return std::span<const T>(tensor, metadata->bytes / sizeof(T));
  }

 private:
  ModelWrapper(std::unique_ptr<tflite::FlatBufferModel> model,
               std::unique_ptr<tflite::Interpreter> interpreter);

  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
};

}  // namespace soundstream

#endif  // SOUNDSTREAM_MODEL_WRAPPER_H_

