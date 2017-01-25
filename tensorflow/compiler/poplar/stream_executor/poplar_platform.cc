/* Copyright 2017 Graphcore Ltd
 */

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/poplar/stream_executor/poplar_platform.h"

#include "tensorflow/compiler/poplar/stream_executor/poplar_executor.h"

#include "tensorflow/stream_executor/poplar/poplar_platform_id.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/ptr_util.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/status_macros.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"

namespace gpu = ::perftools::gputools;

namespace perftools {
namespace gputools {
namespace poplar {

PoplarPlatform::PoplarPlatform() : name_("Poplar") {}

PoplarPlatform::~PoplarPlatform() {}

Platform::Id PoplarPlatform::id() const { return kPoplarPlatformId; }

int PoplarPlatform::VisibleDeviceCount() const {
  return 1;
}

const string& PoplarPlatform::Name() const { return name_; }

port::StatusOr<StreamExecutor*> PoplarPlatform::ExecutorForDevice(int ordinal) {
  StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = PluginConfig();
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*> PoplarPlatform::ExecutorForDeviceWithPluginConfig(
    int device_ordinal, const PluginConfig& plugin_config) {
  StreamExecutorConfig config;
  config.ordinal = device_ordinal;
  config.plugin_config = plugin_config;
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*> PoplarPlatform::GetExecutor(
    const StreamExecutorConfig& config) {
  mutex_lock lock(executors_mutex_);

  port::StatusOr<StreamExecutor*> status = executor_cache_.Get(config);
  if (status.ok()) {
    return status.ValueOrDie();
  }

  port::StatusOr<std::unique_ptr<StreamExecutor>> executor =
      GetUncachedExecutor(config);
  if (!executor.ok()) {
    return executor.status();
  }

  StreamExecutor* naked_executor = executor.ValueOrDie().get();
  SE_RETURN_IF_ERROR(
      executor_cache_.Insert(config, executor.ConsumeValueOrDie()));
  return naked_executor;
}

port::StatusOr<std::unique_ptr<StreamExecutor>>
PoplarPlatform::GetUncachedExecutor(const StreamExecutorConfig& config) {
  auto executor = port::MakeUnique<StreamExecutor>(
      this, port::MakeUnique<PoplarExecutor>(config.plugin_config));
  auto init_status = executor->Init(config.ordinal, config.device_options);
  if (!init_status.ok()) {
    return port::Status{
        port::error::INTERNAL,
        port::Printf(
            "failed initializing StreamExecutor for device ordinal %d: %s",
            config.ordinal, init_status.ToString().c_str())};
  }

  return std::move(executor);
}

void PoplarPlatform::RegisterTraceListener(
    std::unique_ptr<TraceListener> listener) {
  LOG(FATAL) << "not yet implemented: register poplar trace listener";
}

void PoplarPlatform::UnregisterTraceListener(TraceListener* listener) {
  LOG(FATAL) << "not yet implemented: unregister poplar trace listener";
}

static void InitializePoplarPlatform() {
  std::unique_ptr<gpu::Platform> platform(new gpu::poplar::PoplarPlatform);
  SE_CHECK_OK(gpu::MultiPlatformManager::RegisterPlatform(std::move(platform)));
}

}  // namespace poplar
}  // namespace gputools
}  // namespace perftools

REGISTER_MODULE_INITIALIZER(
    poplar_platform, perftools::gputools::poplar::InitializePoplarPlatform());

DECLARE_MODULE_INITIALIZER(multi_platform_manager);
// Note that module initialization sequencing is not supported in the
// open-source project, so this will be a no-op there.
REGISTER_MODULE_INITIALIZER_SEQUENCE(poplar_platform, multi_platform_manager);
