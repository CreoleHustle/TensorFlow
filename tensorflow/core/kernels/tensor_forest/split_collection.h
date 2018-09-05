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
#ifndef TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_SPLIT_COLLECTION_OPERATORS_H_
#define TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_SPLIT_COLLECTION_OPERATORS_H_

#include <vector>
#include "tensorflow/contrib/decision_trees/proto/generic_tree_model.pb.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/grow_stats.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/input_data.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/input_target.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/leaf_model_operators.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/params.h"
#include "tensorflow/contrib/tensor_forest/proto/fertile_stats.pb.h"
#include "tensorflow/contrib/tensor_forest/proto/tensor_forest_params.pb.h"

namespace tensorflow {

// Class that can initialize and update split collections, and
// report if one is finished and ready to split.  Designed to be inherited
// from to implement techniques such as pruning and early/delayed finishing.
class SplitCollection {
 public:
  explicit SplitCollection(const bool& is_regression)
      : is_regression_(is_regression) {}
  virtual ~SplitCollection() {}

  // Return a new GrowStats object according to stats_type_;
  virtual std::unique_ptr<GrowStats> CreateGrowStats(int32 node_id,
                                                     int32 depth) const;

  // Initialize from a previously serialized proto.
  virtual void ExtractFromProto(const FertileStats& stats);

  // Serialize contents to the given proto.
  virtual void PackToProto(FertileStats* stats) const;

  // Updates the slot's candidates with the new example.
  // Assumes slot has been initialized.
  virtual void AddExample(const std::unique_ptr<TensorDataSet>& input_data,
                          const InputTarget* target,
                          const std::vector<int>& examples,
                          int32 node_id) const;

  // Create a new candidate and initialize it with the given example.
  virtual void CreateAndInitializeCandidateWithExample(
      const std::unique_ptr<TensorDataSet>& input_data,
      const InputTarget* target, int example, int32 node_id) const;

  // Create a new GrowStats for the given node id and initialize it.
  virtual void InitializeSlot(int32 node_id, int32 depth);

  // Called when the resource is deserialized, possibly needing an
  // initialization.
  virtual void MaybeInitialize() {
    if (stats_.empty()) {
      InitializeSlot(0, 0);
    }
  }

  // Perform any necessary cleanup for any tracked state for the slot.
  virtual void ClearSlot(int32 node_id) { stats_.erase(node_id); }

  // Return true if slot is fully initialized.
  virtual bool IsInitialized(int32 node_id) const;

  // Return true if slot is finished.
  virtual bool IsFinished(int32 node_id) const {
    return stats_.at(node_id)->IsFinished();
  }

  // Fill in best with the best split that node_id has, return true if this
  // was successful, false if no good split was found.
  virtual bool BestSplit(int32 node_id, SplitCandidate* best,
                         int32* depth) const;

 protected:
  const bool& is_regression_;
  std::unordered_map<int32, std::unique_ptr<GrowStats>> stats_;
};
class ClassificationSplitCollection() : public SplitCollection {
 private
  // For every slot track total class counts seen at this leaf
  std::unordered_map<int32, std::vector<float>> total_counts_;
  // Also track the number of classes seen for not splitting pure leaves.
  std::unordered_map<int32, int32> num_outputs_seen_;
  // Left-branch taken class counts at this leaf for each split.
  // This is a flat vector for memory-performance reasons.
  // left_counts_[i * num_outputs_ + j] has the j-th class count for split i.
  std::unordered_map<int32, std::vector<float>> num_outputs_seen_;
}
class SplitCollectionFactory {
 public:
  static std::unique_ptr<SplitCollection> CreateSplitCollection(
      const bool& is_regression);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_SPLIT_COLLECTION_OPERATORS_H_
