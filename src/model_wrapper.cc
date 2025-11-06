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

#include "soundstream/model_wrapper.h"

#include <iostream>

#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

namespace soundstream {

std::optional<ModelWrapper> ModelWrapper::Create(const std::string& model_path,
                                                 bool use_xnnpack,
                                                 int num_threads) {
  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  if (model == nullptr) {
    std::cerr << "Failed to load TFLite model: " << model_path << "\n";
    return std::nullopt;
  }

  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  if (num_threads > 0 && builder.SetNumThreads(num_threads) != kTfLiteOk) {
    std::cerr << "Failed to configure TFLite threads for model: "
              << model_path << "\n";
    return std::nullopt;
  }

  std::unique_ptr<tflite::Interpreter> interpreter;
  if (builder(&interpreter) != kTfLiteOk) {
    std::cerr << "Failed to build TFLite interpreter for model: "
              << model_path << "\n";
    return std::nullopt;
  }

  if (use_xnnpack) {
    auto options = TfLiteXNNPackDelegateOptionsDefault();
    options.num_threads = num_threads > 0 ? num_threads : 1;
    options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_QU8;

    std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate(
        TfLiteXNNPackDelegateCreate(&options), &TfLiteXNNPackDelegateDelete);
    if (delegate == nullptr) {
      std::cerr << "Failed to create XNNPACK delegate for model: "
                << model_path << "\n";
      return std::nullopt;
    }
    delegate->flags |= kTfLiteDelegateFlagsAllowDynamicTensors;
    const TfLiteStatus status =
        interpreter->ModifyGraphWithDelegate(std::move(delegate));
    if (status == kTfLiteDelegateError) {
      std::cerr << "XNNPACK delegate setup failed, continuing without it for "
                << model_path << "\n";
    } else if (status != kTfLiteOk) {
      std::cerr << "Unable to apply XNNPACK delegate for model: "
                << model_path << "\n";
      return std::nullopt;
    }
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors for model: " << model_path
              << "\n";
    return std::nullopt;
  }

  return ModelWrapper(std::move(model), std::move(interpreter));
}

ModelWrapper::ModelWrapper(std::unique_ptr<tflite::FlatBufferModel> model,
                           std::unique_ptr<tflite::Interpreter> interpreter)
    : model_(std::move(model)), interpreter_(std::move(interpreter)) {}

bool ModelWrapper::Invoke() { return interpreter_->Invoke() == kTfLiteOk; }

}  // namespace soundstream

