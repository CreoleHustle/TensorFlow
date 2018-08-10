/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/xla_computation.h"

#include <utility>

#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

StatusOr<ProgramShape> XlaComputation::GetProgramShape() const {
  TF_RET_CHECK(proto_.has_program_shape());
  return proto_.program_shape();
}

StatusOr<std::unique_ptr<HloSnapshot>> XlaComputation::Snapshot() const {
  if (IsNull()) {
    return InvalidArgument("Computation is invalid.");
  }
  auto session = MakeUnique<HloSnapshot>();
  *session->mutable_hlo()->mutable_hlo_module() = proto_;
  return std::move(session);
}

}  // namespace xla
