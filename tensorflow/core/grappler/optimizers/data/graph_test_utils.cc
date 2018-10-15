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

#include "tensorflow/core/grappler/optimizers/data/graph_test_utils.h"

#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace grappler {
namespace graph_tests_utils {

NodeDef MakeMapNode(absl::string_view name, absl::string_view input_node_name,
                    absl::string_view function_name) {
  return test::function::NDef(
      name, "MapDataset", {string(input_node_name)},
      {{"f", FunctionDefHelper::FunctionRef(string(function_name))},
       {"Targuments", {}},
       {"output_shapes", gtl::ArraySlice<TensorShape>{}},
       {"output_types", gtl::ArraySlice<DataType>{}}});
}

NodeDef MakeFilterNode(absl::string_view name,
                       absl::string_view input_node_name,
                       absl::string_view function_name) {
  return test::function::NDef(
      name, "FilterDataset", {string(input_node_name)},
      {{"predicate", FunctionDefHelper::FunctionRef(string(function_name))},
       {"Targuments", {}},
       {"output_shapes", gtl::ArraySlice<TensorShape>{}},
       {"output_types", gtl::ArraySlice<TensorShape>{}}});
}

NodeDef MakeMapAndBatchNode(absl::string_view name,
                            absl::string_view input_node_name,
                            absl::string_view batch_size_node_name,
                            absl::string_view num_parallel_calls_node_name,
                            absl::string_view drop_remainder_node_name,
                            absl::string_view function_name) {
  return test::function::NDef(
      name, "MapAndBatchDatasetV2",
      {string(input_node_name), "", string(batch_size_node_name),
       string(num_parallel_calls_node_name), string(drop_remainder_node_name)},
      {{"predicate", FunctionDefHelper::FunctionRef(string(function_name))},
       {"Targuments", {}},
       {"output_shapes", gtl::ArraySlice<TensorShape>{}},
       {"output_types", gtl::ArraySlice<TensorShape>{}}});
}

}  // end namespace graph_tests_utils
}  // end namespace grappler
}  // end namespace tensorflow
