/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/collective_send_recv_combiner.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using CollectiveSendRecvCombinerTest = HloTestBase;

TEST_F(CollectiveSendRecvCombinerTest, Transformed) {
  const char* kHloStr = R"(
  ENTRY main {
    data = f32[] constant(5)
    recv-start = token[] after-all()
    recv = (f32[], u32[], token[]) recv(recv-start), channel_id=1, frontend_attributes={_xla_send_recv_source_target_pairs="{{3,0}}"}
    send = (f32[], u32[], token[]) send(data, recv-start), channel_id=1, frontend_attributes={_xla_send_recv_source_target_pairs="{{0,3}}"}
    recv-done = (f32[], token[]) recv-done(recv), channel_id=1, frontend_attributes={_xla_send_recv_source_target_pairs="{{3,0}}"}
    send-done = token[] send-done(send), channel_id=1, frontend_attributes={_xla_send_recv_source_target_pairs="{{0,3}}"}
    ROOT out = f32[] get-tuple-element(recv-done), index=0
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule((kHloStr)));
  CollectiveSendRecvCombiner combiner;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combiner.Run(module.get()));
  EXPECT_TRUE(changed);
}
}  // namespace
}  // namespace xla
