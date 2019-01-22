/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/hlo_matcher.h"

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/meta_graph.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"

#include <queue>
#include <set>
#include <stack>

using ::absl::StrCat;

namespace xla {
namespace poplarplugin {
namespace {

absl::optional<int64> GetOperandIndexForNodeId(
    const HloMatcherNode& pattern_node, const NodeId& operand_id) {
  auto it = absl::c_find(pattern_node.operands, operand_id);
  return it != pattern_node.operands.end()
             ? absl::optional<int64>(
                   std::distance(pattern_node.operands.begin(), it))
             : absl::nullopt;
}

bool IsValidCandidate(
    NodeId candidate_node_id, HloInstruction* candidate_inst,
    absl::flat_hash_map<NodeId, HloInstruction*>& parital_matching,
    absl::flat_hash_map<NodeId, absl::flat_hash_set<HloInstruction*>>&
        invalid_pairings,
    const HloMatcherPattern& pattern) {
  // Make sure we haven't already tried this pairing.
  if (invalid_pairings[candidate_node_id].count(candidate_inst) != 0) {
    return false;
  }

  // This is not a valid candidate if we already matched this candidate NodeId.
  if (parital_matching.count(candidate_node_id) > 0) {
    return false;
  }

  // This is not a valid candidate if we already matched this candidate
  // instruction.
  auto it = absl::c_find_if(parital_matching,
                            [&](std::pair<NodeId, HloInstruction*> iso_pair) {
                              return iso_pair.second == candidate_inst;
                            });
  if (it != parital_matching.end()) {
    return false;
  }

  HloMatcherNode matched_node = pattern.GetPatternNodes()[candidate_node_id];

  // Check the opcode - note that a parameter opcode means ANY instruction.
  if (matched_node.opcode != HloOpcode::kParameter) {
    if (matched_node.opcode != candidate_inst->opcode()) {
      return false;
    }
  }

  // Check the node condition.
  if (matched_node.node_condition &&
      !(*matched_node.node_condition)(candidate_inst)) {
    return false;
  }

  // Check that the operands match up
  auto operand_ids = pattern.GetNodesToOperandsMetaGraph()[candidate_node_id];
  for (NodeId operand_id : operand_ids) {
    auto operand_idx = *GetOperandIndexForNodeId(matched_node, operand_id);

    auto it = parital_matching.find(operand_id);
    if (it == parital_matching.end()) {
      continue;
    }
    HloInstruction* operand_inst = it->second;
    if (candidate_inst->mutable_operand(operand_idx) != operand_inst) {
      return false;
    }
  }

  return true;
}

absl::flat_hash_map<NodeId, absl::flat_hash_set<HloInstruction*>>
FindValidCandidates(
    absl::flat_hash_map<NodeId, HloInstruction*>& parital_matching,
    absl::flat_hash_map<NodeId, absl::flat_hash_set<HloInstruction*>>&
        invalid_pairings,
    const HloMatcherPattern& pattern) {
  absl::flat_hash_map<NodeId, absl::flat_hash_set<HloInstruction*>> targets;
  // Find possible isomorphisms.
  // We first look for target nodes by traversing from an instruction to it's
  // operand.
  for (auto iso_pair : parital_matching) {
    NodeId node_id = iso_pair.first;
    HloInstruction* matched_inst = iso_pair.second;
    HloMatcherNode matcher_node = pattern.GetPatternNodes()[node_id];

    // Go through the operands and check whether they are valid isomorphisms.
    auto operand_ids = pattern.GetNodesToOperandsMetaGraph()[node_id];
    for (NodeId operand_id : operand_ids) {
      auto operand_idx = *GetOperandIndexForNodeId(matcher_node, operand_id);
      HloInstruction* operand_inst = matched_inst->mutable_operand(operand_idx);
      if (IsValidCandidate(operand_id, operand_inst, parital_matching,
                           invalid_pairings, pattern)) {
        targets[operand_id].insert(operand_inst);
      }
    }
  }

  if (targets.empty()) {
    // If there are no targets, then traverse from an instruction to it's users.
    for (auto iso_pair : parital_matching) {
      NodeId operand_id = iso_pair.first;
      HloInstruction* matched_inst = iso_pair.second;

      auto node_ids = pattern.GetOperandsToNodesMetaGraph()[operand_id];
      // We label every possible user with a node_id if the user uses the
      // matched_inst at correct index.
      for (NodeId node_id : node_ids) {
        HloMatcherNode matcher_node = pattern.GetPatternNodes()[node_id];
        for (HloInstruction* user_inst : matched_inst->users()) {
          auto operand_idx =
              *GetOperandIndexForNodeId(matcher_node, operand_id);

          if (operand_idx < user_inst->operand_count() &&
              user_inst->mutable_operand(operand_idx) == matched_inst) {
            if (IsValidCandidate(node_id, user_inst, parital_matching,
                                 invalid_pairings, pattern)) {
              targets[node_id].insert(user_inst);
            }
          }
        }
      }
    }
  }
  return targets;
}

// We use the VF2 algorithm - published paper "An Improved Algorithm for
// Matching Large Graphs" by L. P. Cordella, P. Foggia, C. Sansone, M. Vento -
// with one difference being we are matching DAGs - we only ever visit nodes
// connected to the partial matching.
bool MatchDAGIsomorphism(
    absl::flat_hash_map<NodeId, HloInstruction*>& parital_matching,
    absl::flat_hash_map<NodeId, absl::flat_hash_set<HloInstruction*>>&
        invalid_pairings,
    const HloMatcherPattern& pattern) {
  // Base condition - we matched the pattern if we matched every node in it and
  // the graph is valid.
  if (parital_matching.size() == pattern.GetPatternNodes().size()) {
    // TODO A proof to show that this is not required.
    // A check that goes through the isomorphism and checks that the mapping is
    // correct.
    // For every iso mapping pair, go through the operands and check they match
    // up.
    for (auto iso_pair : parital_matching) {
      NodeId node_id = iso_pair.first;
      HloInstruction* inst = iso_pair.second;
      HloMatcherNode matched_node = pattern.GetPatternNodes()[node_id];

      for (unsigned operand_idx = 0;
           operand_idx < pattern.GetPatternNodes()[node_id].operands.size();
           operand_idx++) {
        NodeId operand_id =
            pattern.GetPatternNodes()[node_id].operands[operand_idx];
        HloInstruction* operand_inst = parital_matching[operand_id];
        if (inst->mutable_operand(operand_idx) != operand_inst) {
          return false;
        }
      }
    }
    return true;
  } else {
    // Given the current state, find nodes that when matched also give a valid
    // state and traverse those.
    auto candidates =
        FindValidCandidates(parital_matching, invalid_pairings, pattern);
    for (auto pair : candidates) {
      NodeId candidate_node_id = pair.first;
      for (HloInstruction* candidate_inst : pair.second) {
        // Match the DAG with the new pairing candidate_node_id <->
        // candidate_inst added.
        parital_matching[candidate_node_id] = candidate_inst;

        if (MatchDAGIsomorphism(parital_matching, invalid_pairings, pattern)) {
          return true;
        }

        // If this was not a successful match then we need to remove it from
        // partial matching.
        parital_matching.erase(candidate_node_id);
        invalid_pairings[candidate_node_id].insert(candidate_inst);
      }
    }
    return false;
  }
}

// Start of the DAG match - we start the search from output[0].
bool MatchDAGIsomorphism(
    absl::flat_hash_map<NodeId, HloInstruction*>& parital_matching,
    HloInstruction* first_output, const HloMatcherPattern& pattern) {
  NodeId first_output_node_id = pattern.GetOutputs()[0];
  absl::flat_hash_map<NodeId, absl::flat_hash_set<HloInstruction*>>
      invalid_pairings;
  if (IsValidCandidate(first_output_node_id, first_output, parital_matching,
                       invalid_pairings, pattern)) {
    // Add the initial matching for the first output.
    parital_matching[first_output_node_id] = first_output;
    return MatchDAGIsomorphism(parital_matching, invalid_pairings, pattern);
  }
  return false;
}

}  // namespace

HloMatcherPattern::HloMatcherPattern(PatternType type,
                                     PatternMetaTarget meta_target,
                                     PatternInputs inputs,
                                     PatternOutputs outputs,
                                     Pattern pattern_nodes)
    : type(type),
      meta_target(meta_target),
      inputs(inputs),
      outputs(outputs),
      pattern_nodes(pattern_nodes),
      pattern_graphs(VerifyAndGetGraphs()) {}

const PatternType& HloMatcherPattern::GetType() const { return type; }

const PatternMetaTarget& HloMatcherPattern::GetMetaTarget() const {
  return meta_target;
}

const PatternInputs& HloMatcherPattern::GetInputs() const { return inputs; }

const PatternOutputs& HloMatcherPattern::GetOutputs() const { return outputs; }

const Pattern& HloMatcherPattern::GetPatternNodes() const {
  return pattern_nodes;
};

const MetaGraph<NodeId>& HloMatcherPattern::GetNodesToOperandsMetaGraph()
    const {
  return pattern_graphs.first;
};

const MetaGraph<NodeId>& HloMatcherPattern::GetOperandsToNodesMetaGraph()
    const {
  return pattern_graphs.second;
};

std::pair<MetaGraph<NodeId>, MetaGraph<NodeId>>
HloMatcherPattern::VerifyAndGetGraphs() {
  const std::string prefix = "[Pattern " + type + "] ";

  // A pattern needs to have an output.
  if (outputs.size() == 0) {
    throw std::invalid_argument(
        prefix + "Pattern has no outputs, at least one required.");
  }

  // Make sure inputs are unique and that they point to a label in the pattern.
  absl::flat_hash_set<NodeId> inputs_set;
  for (auto input : inputs) {
    if (input < 0 || input >= pattern_nodes.size()) {
      throw std::invalid_argument(prefix + "Input with label " +
                                  std::to_string(input) +
                                  " does not exist in the pattern.");
    }

    if (inputs_set.count(input)) {
      throw std::invalid_argument(
          prefix + "Input with label " + std::to_string(input) +
          " already defined. Pattern inputs need to be unique.");
    }
    inputs_set.insert(input);
  }

  // Make sure outputs are unique and that they point to a label in the pattern.
  absl::flat_hash_set<NodeId> outputs_set;
  for (auto output : outputs) {
    if (output < 0 || output >= pattern_nodes.size()) {
      throw std::invalid_argument(prefix + "Output with label " +
                                  std::to_string(output) +
                                  " does not exist in the pattern.");
    }

    if (outputs_set.count(output)) {
      throw std::invalid_argument(
          prefix + "Output with label " + std::to_string(output) +
          " already defined. Pattern outputs need to be unique.");
    }
    outputs_set.insert(output);
  }

  // Check that an output is not an input or vice versa.
  absl::flat_hash_set<NodeId> input_output_overlap;
  for (auto input : inputs_set) {
    if (outputs_set.contains(input)) {
      throw std::invalid_argument(
          prefix + "An input is not allowed to be an output (labels " +
          absl::StrJoin(input_output_overlap, ", ") + ").");
    }
  }

  const auto get_operands = [this, &prefix](NodeId label) {
    // Verify that the node with label is defined in the pattern.
    if (label < 0 || label >= pattern_nodes.size()) {
      throw std::invalid_argument(prefix + "Unknown node " +
                                  std::to_string(label) +
                                  " which was not defined in the pattern.");
    }
    return pattern_nodes[label].operands;
  };

  // Create a graph.
  MetaGraph<NodeId> operands_to_nodes(outputs, get_operands);
  MetaGraph<NodeId> nodes_to_operands = operands_to_nodes.Transpose();

  // Check that an input doesn't have operands.
  for (auto input : inputs) {
    if (!nodes_to_operands[input].empty()) {
      throw std::invalid_argument(
          prefix + "Input with label " + std::to_string(input) +
          " has an input - this is currently not supported.");
    }
  }

  // Verify that the graph is connected - i.e. any two pairs of nodes in the
  // pattern can be reached.
  // The strategy is to perform a traversal where the next node is either one of
  // the child edges or parent edges which have not yet been visited.
  absl::flat_hash_set<NodeId> visited;
  std::stack<NodeId> to_visit;
  to_visit.push(outputs[0]);

  while (!to_visit.empty()) {
    NodeId current_node = to_visit.top();
    to_visit.pop();
    visited.insert(current_node);

    absl::flat_hash_set<NodeId> candidates = operands_to_nodes[current_node];
    candidates.insert(nodes_to_operands[current_node].begin(),
                      nodes_to_operands[current_node].end());
    for (auto candidate : candidates) {
      // Only traverse unvisited nodes.
      bool traverse = visited.count(candidate) == 0;
      if (traverse) {
        to_visit.push(candidate);
      }
    }
  }

  for (int64 label = 0; label < pattern_nodes.size(); label++) {
    if (visited.find(label) == visited.end()) {
      throw std::invalid_argument(prefix + "Node with label " +
                                  std::to_string(label) +
                                  " is disconnected from the graph. The "
                                  "graph needs to be connected.");
    }
  }

  return {nodes_to_operands, operands_to_nodes};
}

HloMatcher::HloMatcher(const std::vector<HloMatcherPattern>& patterns,
                       struct CompilerAnnotations& annotations,
                       bool root_computation_only,
                       unsigned look_through_max_depth)
    : patterns_(std::move(patterns)),
      annotations_(annotations),
      root_computation_only_(root_computation_only),
      look_through_max_depth_(look_through_max_depth) {}

// A set of sets of ops which are all associative together
static std::set<std::set<HloOpcode>> associative_ops_sets = {
    {HloOpcode::kMultiply}, {HloOpcode::kAdd},
};

absl::optional<Trace> HloMatcher::FindNextMatchingOp(
    HloInstruction* user, HloInstruction* inst, const HloOpcode desiredOpcode) {
  for (const auto ops_set : associative_ops_sets) {
    // user needs to be an associative op
    if (!ops_set.count(user->opcode())) {
      continue;
    }

    // Non recursive depth first DAG traversal to try and find an inst with
    // right opcode using associativity
    std::stack<Trace> to_visit;
    // The list of instructions visited while searching for each pattern
    std::set<HloInstruction*> visited = {user};

    // Traverse from inst
    Trace start_trace = {{user, user->operand_index(inst)}};
    to_visit.push(start_trace);
    while (!to_visit.empty()) {
      // Get value of the stack
      auto current = to_visit.top();
      to_visit.pop();

      HloInstruction* current_inst =
          current.back().inst->mutable_operand(current.back().op_idx);
      visited.insert(current_inst);
      // Check if the current instruction matches
      if (current_inst->opcode() == desiredOpcode) {
        current.push_back({current_inst, -1});
        return current;
      }

      // Check the current instruction is associative and matches the shape,
      // if not then we can't look through it
      if (!(ops_set.count(current_inst->opcode()) &&
            ShapeUtil::Equal(current_inst->shape(), inst->shape()))) {
        continue;
      }

      // Add operands to visit without going past the maximum search depth
      if (current.size() - 1 < look_through_max_depth_) {
        for (int64 i = 0; i < current_inst->operand_count(); i++) {
          auto* operand = current_inst->mutable_operand(i);
          // Only add the operand if:
          // * we have never seen it before
          // * it has one user
          // * it has the same shape
          if (visited.count(operand) == 0 && operand->user_count() == 1 &&
              ShapeUtil::Equal(operand->shape(), inst->shape())) {
            // We need to know which operand will be replaced at the root
            // instruction - we only need to know this on depth 0, otherwise
            // keep it the same
            auto next_trace = current;
            next_trace.push_back({current_inst, i});
            to_visit.push(next_trace);
          }
        }
      }
    }
  }
  return absl::nullopt;
}

bool HloMatcher::MatchPatternSingleOutput(HloInstruction* root,
                                          const HloMatcherPattern& pattern,
                                          HloMatcherMatched& match) {
  match.instruction_mapping[pattern.GetOutputs()[0]] = root;

  // Construct a mapping from a pattern node to all other pattern nodes which
  // use it
  std::vector<std::set<std::pair<unsigned int, unsigned int>>> node_mapping(
      pattern.GetPatternNodes().size());

  // Create lookup for input indexes to parameter number
  std::map<NodeId, int64> input_id_to_param_num;
  for (int64 i = 0; i < pattern.GetInputs().size(); i++) {
    input_id_to_param_num[pattern.GetInputs()[i]] = i;
  }

  const auto is_input = [&input_id_to_param_num](const NodeId pid) {
    return input_id_to_param_num.count(pid);
  };

  for (unsigned int node_num = 0; node_num < pattern.GetPatternNodes().size();
       node_num++) {
    for (unsigned int op_idx = 0;
         op_idx < pattern.GetPatternNodes()[node_num].operands.size();
         op_idx++) {
      node_mapping[pattern.GetPatternNodes()[node_num].operands[op_idx]].insert(
          {node_num, op_idx});
    }

    if (node_num) {
      match.instruction_mapping[node_num] = nullptr;
    }
  }

  for (unsigned int node_num = 0; node_num < pattern.GetPatternNodes().size();
       node_num++) {
    HloInstruction* inst = match.instruction_mapping[node_num];
    if (inst == nullptr) {
      return false;
    }

    const HloMatcherNode& node(pattern.GetPatternNodes()[node_num]);

    if (node.opcode != HloOpcode::kParameter) {
      if (node.opcode != inst->opcode()) {
        // Try to find an op using associativity, unless this is the first node
        // or search depth is 0 or this inst is used more than once
        if (node_num == 0 || look_through_max_depth_ == 0 ||
            inst->user_count() != 1) {
          return false;
        }
        unsigned int user_node_num = node_mapping[node_num].begin()->first;
        auto* user = match.instruction_mapping[user_node_num];
        auto optional_trace = FindNextMatchingOp(user, inst, node.opcode);
        // Check whether we managed to find a match
        if (!optional_trace) {
          return false;
        }
        Trace found = *optional_trace;
        match.instruction_mapping[node_num] = found.back().inst;
        inst = found.back().inst;
        match.replacement_traces.push_back(found);
      }
    }

    if (node.node_condition && !(*node.node_condition)(inst)) {
      return false;
    }

    if (!is_input(node_num)) {
      if ((node.operands.size() > 0) &&
          (inst->operand_count() != node.operands.size())) {
        return false;
      }

      for (unsigned int i = 0; i < node.operands.size(); i++) {
        HloInstruction* operand = inst->mutable_operand(i);
        int n = node.operands[i];

        if (n >= match.instruction_mapping.size()) {
          LOG(FATAL) << "Invalid matcher reference " << n;
        }

        if (match.instruction_mapping[n] != nullptr) {
          // Instructions can only match once
          if (match.instruction_mapping[n] != operand) {
            return false;
          }
        } else {
          // Each instruction can match only one entry in the pattern
          auto it =
              absl::c_find_if(match.instruction_mapping,
                              [&](std::pair<NodeId, HloInstruction*> iso_pair) {
                                return iso_pair.second == operand;
                              });
          if (it != match.instruction_mapping.end()) {
            return false;
          }

          match.instruction_mapping[n] = operand;
        }
      }
    }
  }
  return true;
}

bool HloMatcher::MatchPattern(HloInstruction* root,
                              const unsigned pattern_idx) {
  HloMatcherMatched match(root->parent(), pattern_idx);
  auto& pattern = patterns_[pattern_idx];

  bool matched = false;
  if (pattern.GetOutputs().size() == 1) {
    // TODO - T5965
    // We still use the old algorithm for the matching of patterns with a single
    // output because that algorithm supports associative look through matching.
    // Remove this algorithm once the new algorithm supports it.
    matched = MatchPatternSingleOutput(root, pattern, match);
  } else {
    matched = MatchDAGIsomorphism(match.instruction_mapping, root, pattern);
  }

  return matched ? HandleMatch(match) : false;
}

bool HloMatcher::MatchPatternStart(HloComputation* computation) {
  bool matched = false;

  // Non recursive depth first DAG traversal to match the patterns - note that
  // we restart the search after every match.
  bool start_from_root = true;
  while (start_from_root) {
    start_from_root = false;

    for (unsigned i = 0; i < patterns_.size(); i++) {
      auto pattern = patterns_[i];
      std::stack<HloInstruction*> to_visit;
      // The list of instructions visited while searching for each pattern
      std::set<HloInstruction*> visited;

      // Traverse from root
      to_visit.push(computation->root_instruction());
      while (!to_visit.empty()) {
        HloInstruction* instruction = to_visit.top();
        to_visit.pop();
        visited.insert(instruction);
        // A pattern can have multiple outputs. We start the pattern match when
        // we find an instruction which matches the first output of the pattern.
        auto first_output = pattern.GetOutputs()[0];
        if (instruction->opcode() ==
            pattern.GetPatternNodes()[first_output].opcode) {
          // Try matching the whole pattern
          if (MatchPattern(instruction, i)) {
            VLOG(1) << "Matched pattern type " << pattern.GetType() << ".";
            matched = true;
            // Restart the matcher
            start_from_root = true;
            break;
          }
        }
        for (HloInstruction* operand : instruction->operands()) {
          if (visited.count(operand) == 0) {
            to_visit.push(operand);
          }
        }
      }

      // Restart the matcher
      if (start_from_root) {
        break;
      }
    }
  }
  return matched;
}

StatusOr<bool> HloMatcher::Run(HloModule* module) {
  bool matched = false;
  if (root_computation_only_) {
    HloComputation* comp = module->entry_computation();
    matched = MatchPatternStart(comp);
  } else {
    // Copy list of computations as we will be introducing new ones
    std::vector<HloComputation*> comps(module->computations().begin(),
                                       module->computations().end());

    for (auto* comp : comps) {
      if (!comp->IsFusionComputation() && !IsPopOpsFusion(comp)) {
        matched |= MatchPatternStart(comp);
      }
    }
  }

  return matched;
}

std::set<HloInstruction*> HloMatcher::ReorderGraph(
    const HloMatcherMatched& matched) {
  std::set<HloInstruction*> modified_instructions;
  for (auto trace : matched.replacement_traces) {
    auto root = trace[0];
    auto target_user = trace.rbegin()[1];  // second to last element
    auto target = trace.back();

    root.inst->ReplaceAllUsesWith(target_user.inst);
    target_user.inst->ReplaceOperandWith(target_user.op_idx, root.inst);
    root.inst->ReplaceOperandWith(root.op_idx, target.inst);
    absl::c_transform(trace, std::inserter(modified_instructions,
                                           modified_instructions.begin()),
                      [](InstructionIndex const& x) { return x.inst; });
  }
  return modified_instructions;
}

HloInstruction* HloMatcher::OutlineExpressionFromComputation(
    const HloMatcherMatched& matched,
    const std::string& outlined_computation_name,
    std::vector<HloInstruction*> forced_parameters) {
  HloComputation* computation = matched.computation;
  auto pattern = patterns_[matched.pattern_idx];
  HloModule* module = computation->parent();

  // First we need to update the graph with any instructions that will be
  // reordered
  auto modified_instructions = ReorderGraph(matched);
  // A map from original instructions to their new counterparts
  absl::flat_hash_map<NodeId, HloInstruction*> outlined;
  // A set of nodes which we have already outlined.
  absl::flat_hash_set<NodeId> outlined_node_ids;
  // A set of nodes which we can outline because all the operands have been
  // outlined.
  absl::flat_hash_set<NodeId> to_outline;
  // Arguments to the new computation.
  std::vector<HloInstruction*> arguments;
  // A node can be outlined if all the operands have been outlined and it has
  // not been outlined yet.
  const auto can_outline = [&](NodeId node_id) {
    for (auto operand_id : pattern.GetNodesToOperandsMetaGraph()[node_id]) {
      if (outlined_node_ids.count(operand_id) == 0) {
        return false;
      }
    }
    return outlined_node_ids.count(node_id) == 0;
  };

  // First outline all the parameters.
  auto builder = HloComputation::Builder(outlined_computation_name);
  for (unsigned parameter_num = 0; parameter_num < pattern.GetInputs().size();
       parameter_num++) {
    NodeId node_id = pattern.GetInputs()[parameter_num];
    HloInstruction* param_input = matched.instruction_mapping.at(node_id);
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(parameter_num, param_input->shape(),
                                        StrCat("arg_", parameter_num)));
    outlined[node_id] = param;
    outlined_node_ids.insert(node_id);
    arguments.push_back(param_input);
    // Check what we can outline.
    absl::c_copy_if(pattern.GetOperandsToNodesMetaGraph()[node_id],
                    std::inserter(to_outline, std::begin(to_outline)),
                    can_outline);
  }
  // Add all the instructions which have no dependencies to be outlined as well
  // (for example constants).
  for (auto pair : pattern.GetNodesToOperandsMetaGraph()) {
    NodeId node_id = pair.first;
    auto edges = pair.second;
    if (edges.empty() && outlined_node_ids.count(node_id) == 0) {
      to_outline.insert(node_id);
    }
  }

  // Now outline all the remaining nodes
  while (!to_outline.empty()) {
    // Get an instruction which is ready to be outlined.
    NodeId node_id = *to_outline.begin();
    to_outline.erase(node_id);

    HloInstruction* old_inst = matched.instruction_mapping.at(node_id);
    HloInstruction* new_inst = builder.AddInstruction(old_inst->Clone());
    outlined[node_id] = new_inst;
    outlined_node_ids.insert(node_id);
    // Replace all the operands
    for (int64 operand = 0; operand < new_inst->operand_count(); ++operand) {
      NodeId operand_id = pattern.GetPatternNodes()[node_id].operands[operand];
      TF_CHECK_OK(new_inst->ReplaceOperandWith(operand, outlined[operand_id]));
    }
    // Check if we can outline more instructions.
    absl::c_copy_if(pattern.GetOperandsToNodesMetaGraph()[node_id],
                    std::inserter(to_outline, std::begin(to_outline)),
                    can_outline);
  }

  // Sanity check - make sure we have outlined everything.
  if (outlined.size() != pattern.GetPatternNodes().size()) {
    LOG(FATAL) << "Failed to outline a pattern correctly - not all "
                  "instructions have been outlined. "
               << outlined.size() << " " << pattern.GetPatternNodes().size();
  }
  // If we have multiple outputs then create a root tuple - otherwise output[0]
  // is the root.
  HloInstruction* root;
  if (pattern.GetOutputs().size() > 1) {
    std::vector<HloInstruction*> outputs;
    absl::c_transform(pattern.GetOutputs(), std::back_inserter(outputs),
                      [&](NodeId node_id) { return outlined[node_id]; });
    root = builder.AddInstruction(HloInstruction::CreateTuple(outputs));
  } else {
    root = outlined[pattern.GetOutputs()[0]];
  }

  // Add forced parameters as arguments - DCE does not remove unused parameters.
  // This allows us to link and easily maintain outputs of a fwd pass to the bwd
  // pass.
  for (unsigned i = 0; i < forced_parameters.size(); i++) {
    HloInstruction* inst = forced_parameters[i];
    const unsigned parameter_num = arguments.size() + i;
    builder.AddInstruction(HloInstruction::CreateParameter(
        parameter_num, inst->shape(), StrCat("arg_", parameter_num)));
    arguments.push_back(inst);
  }

  // Creates a fusion call to the nested computation.
  HloComputation* fusion_computation =
      module->AddEmbeddedComputation(builder.Build(root));

  // Ensure that all parameters are a dependency of the root
  for (auto* param : fusion_computation->parameter_instructions()) {
    if (param->user_count() == 0) {
      param->AddControlDependencyTo(root);
    }
  }

  HloInstruction* fusion =
      matched.computation->AddInstruction(HloInstruction::CreateFusion(
          root->shape(), HloInstruction::FusionKind::kCustom, arguments,
          fusion_computation));

  fusion_computation->SetFusionInstruction(fusion);

  auto* old = matched.instruction_mapping.at(pattern.GetMetaTarget());

  PoplarBackendConfig backend_config;
  if (old->opcode() == HloOpcode::kConvolution) {
    auto* cfg = backend_config.mutable_fusion_config();
    *(cfg->mutable_window()) = old->window();
    *(cfg->mutable_dimension_numbers()) = old->convolution_dimension_numbers();
    cfg->set_feature_group_count(old->feature_group_count());
    cfg->set_batch_group_count(old->batch_group_count());
  }
  fusion->set_backend_config(backend_config);

  fusion->set_metadata(old->metadata());
  if (old->has_sharding()) {
    fusion->set_sharding(old->sharding());
  }

  // Replace the uses with the new outputs.
  if (pattern.GetOutputs().size() > 1) {
    // For multiple outputs use GTEs.
    for (unsigned tuple_id = 0; tuple_id < pattern.GetOutputs().size();
         tuple_id++) {
      NodeId node_id = pattern.GetOutputs()[tuple_id];
      HloInstruction* old_inst = matched.instruction_mapping.at(node_id);
      HloInstruction* gte =
          computation->AddInstruction(HloInstruction::CreateGetTupleElement(
              old_inst->shape(), fusion, tuple_id));
      TF_CHECK_OK(old_inst->ReplaceAllUsesWith(gte));
    }
  } else {
    HloInstruction* old_inst =
        matched.instruction_mapping.at(pattern.GetOutputs()[0]);
    TF_CHECK_OK(old_inst->ReplaceAllUsesWith(fusion));
  }

  // Remove all the dead instructions in the graph after outlining.
  // DF Traversal from every output node - note that we can't call
  // RemoveInstructionAndUnusedOperands as it doesn't allow us to remove state
  // full ops.
  for (NodeId output_node_id : pattern.GetOutputs()) {
    std::queue<NodeId> to_visit;
    to_visit.push(output_node_id);
    absl::flat_hash_set<NodeId> visited;

    while (!to_visit.empty()) {
      NodeId node_id = to_visit.front();
      to_visit.pop();
      HloInstruction* inst = matched.instruction_mapping.at(node_id);

      // Don't remove nodes already visited or the instructions with users.
      if (visited.count(node_id) != 0 || inst->user_count() != 0) {
        continue;
      }

      for (auto operand_id : pattern.GetNodesToOperandsMetaGraph()[node_id]) {
        to_visit.push(operand_id);
      }

      visited.insert(node_id);
      TF_CHECK_OK(computation->RemoveInstruction(inst));
    }
  }

  return fusion;
}

}  // namespace poplarplugin
}  // namespace xla
