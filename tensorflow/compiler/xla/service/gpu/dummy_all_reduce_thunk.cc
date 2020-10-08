/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace gpu {

struct NcclAllReduceConfig::AuxData {};

NcclAllReduceConfig::NcclAllReduceConfig(NcclAllReduceConfig &&) = default;
NcclAllReduceConfig::~NcclAllReduceConfig() = default;

NcclAllReduceConfig GetNcclAllReduceConfig(const HloInstruction *instr) {
  NcclAllReduceConfig config = {};
  return config;
}

/* static */ bool NcclAllReduceThunk::NcclIsEnabled() {
  return false;  // Skylark selects this source file if NCCL is disabled.
}

/* static */ bool NcclAllReduceThunk::CanImplement(const HloInstruction* crs) {
  return false;
}

Status NcclAllReduceThunk::ExecuteOnStream(const ExecuteParams& params) {
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
}

/*static*/ absl::flat_hash_set<GlobalDeviceId>
NcclAllReduceThunk::DevicesWithOpenNcclChannels() {
  return {};
}

NcclAllReduceThunk::NcclAllReduceThunk(
    ThunkInfo thunk_info, NcclAllReduceConfig &&config,
    std::vector<NcclAllReduceThunk::Buffer> buffers)
    : Thunk(Thunk::kNcclAllReduce, thunk_info),
      config_(std::move(config)),
      buffers_(std::move(buffers)) {}

}  // namespace gpu
}  // namespace xla
