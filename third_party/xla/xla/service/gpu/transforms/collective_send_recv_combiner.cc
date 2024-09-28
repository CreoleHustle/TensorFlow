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

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"
namespace xla {

absl::StatusOr<bool> CollectiveSendRecvCombiner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  int wrapped_computation_index = 0;
  for (HloComputation* computation : module->MakeComputationPostOrder()) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kSend ||
          instruction->opcode() == HloOpcode::kRecv) {
        auto builder = HloComputation::Builder(absl::StrCat(
            "wrapped_", instruction->name(), wrapped_computation_index++));
        int operand_counter = 0;
        std::vector<HloInstruction*> operands;
        std::vector<HloInstruction*> async_start_inputs;
        std::vector<Shape> async_start_input_shapes;
        for (auto operand : instruction->operands()) {
          operands.push_back(
              builder.AddInstruction(HloInstruction::CreateParameter(
                  operand_counter, operand->shape(),
                  absl::StrCat("param", operand_counter))));
          async_start_inputs.push_back(operand);
          async_start_input_shapes.push_back(operand->shape());
          operand_counter++;
        }
        auto root = builder.AddInstruction(
            instruction->CloneWithNewOperands(instruction->shape(), operands));

        // Create async-start and async-done instructions.
        Shape async_start_shape = ShapeUtil::MakeTupleShape(
            {ShapeUtil::MakeTupleShape(async_start_input_shapes), root->shape(),
             ShapeUtil::MakeScalarShape(S32)});
        auto async_start =
            computation->AddInstruction(HloInstruction::CreateAsyncStart(
                async_start_shape, async_start_inputs,
                module->AddEmbeddedComputation(builder.Build(root))));
        auto async_done = computation->AddInstruction(
            HloInstruction::CreateAsyncDone(root->shape(), async_start));
        for (auto instruction_user : instruction->users()) {
          if (instruction_user->opcode() == HloOpcode::kSendDone ||
              instruction_user->opcode() == HloOpcode::kRecvDone) {
            TF_RETURN_IF_ERROR(
                instruction_user->ReplaceAllUsesWithDifferentShape(async_done));
            TF_RETURN_IF_ERROR(
                instruction->ReplaceAllUsesWithDifferentShape(async_start));
            TF_RETURN_IF_ERROR(instruction_user->parent()->RemoveInstruction(
                instruction_user));
            TF_RETURN_IF_ERROR(
                instruction->parent()->RemoveInstruction(instruction));
          }
        }
        changed = true;
      }
    }
  }
  return changed;
}
}  // namespace xla
