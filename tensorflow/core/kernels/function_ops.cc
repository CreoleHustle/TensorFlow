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

#include <deque>
#include <vector>

#include "tensorflow/core/kernels/function_ops.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/gradients.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

ArgOp::ArgOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("index", &index_));
}

void ArgOp::Compute(OpKernelContext* ctx) {
  auto frame = ctx->call_frame();
  OP_REQUIRES(ctx, frame != nullptr, errors::Internal("no call frame"));
  Tensor val;
  OP_REQUIRES_OK(ctx, frame->GetArg(index_, &val));
  OP_REQUIRES(ctx, val.dtype() == dtype_,
              errors::InvalidArgument("Type mismatch: actual ",
                                      DataTypeString(val.dtype()),
                                      " vs. expect ", DataTypeString(dtype_)));
  ctx->set_output(0, val);
}

RetvalOp::RetvalOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("index", &index_));
}

void RetvalOp::Compute(OpKernelContext* ctx) {
  const Tensor& val = ctx->input(0);
  OP_REQUIRES(ctx, val.dtype() == dtype_,
              errors::InvalidArgument("Type mismatch: actual ",
                                      DataTypeString(val.dtype()),
                                      " vs. expect ", DataTypeString(dtype_)));
  auto frame = ctx->call_frame();
  OP_REQUIRES(ctx, frame != nullptr, errors::Internal("no call frame"));
  OP_REQUIRES_OK(ctx, frame->SetRetval(index_, val));
}

REGISTER_SYSTEM_KERNEL_BUILDER(Name(kArgOp).Device(DEVICE_CPU), ArgOp);
REGISTER_SYSTEM_KERNEL_BUILDER(Name(kRetOp).Device(DEVICE_CPU), RetvalOp);

#if TENSORFLOW_USE_SYCL
#define REGISTER(type)     \
  REGISTER_KERNEL_BUILDER( \
      Name(kArgOp).Device(DEVICE_SYCL).TypeConstraint<type>("T"), ArgOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER)
TF_CALL_bool(REGISTER) REGISTER_KERNEL_BUILDER(Name(kArgOp)
                                                   .Device(DEVICE_SYCL)
                                                   .HostMemory("output")
                                                   .TypeConstraint<int32>("T"),
                                               ArgOp);
#undef REGISTER
#define REGISTER(type)     \
  REGISTER_KERNEL_BUILDER( \
      Name(kRetOp).Device(DEVICE_SYCL).TypeConstraint<type>("T"), RetvalOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER)
TF_CALL_bool(REGISTER) REGISTER_KERNEL_BUILDER(Name(kRetOp)
                                                   .Device(DEVICE_SYCL)
                                                   .HostMemory("input")
                                                   .TypeConstraint<int32>("T"),
                                               RetvalOp);
#undef REGISTER
#endif

#define REGISTER(type)     \
  REGISTER_KERNEL_BUILDER( \
      Name(kArgOp).Device(DEVICE_GPU).TypeConstraint<type>("T"), ArgOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER)
TF_CALL_bool(REGISTER) REGISTER_KERNEL_BUILDER(Name(kArgOp)
                                                   .Device(DEVICE_GPU)
                                                   .HostMemory("output")
                                                   .TypeConstraint<int32>("T"),
                                               ArgOp);
#undef REGISTER

REGISTER_KERNEL_BUILDER(Name(kArgOp)
                            .Device(DEVICE_GPU)
                            .HostMemory("output")
                            .TypeConstraint<ResourceHandle>("T"),
                        ArgOp);

REGISTER_KERNEL_BUILDER(Name(kArgOp)
                            .Device(DEVICE_GPU)
                            .HostMemory("output")
                            .TypeConstraint<string>("T"),
                        ArgOp);

#define REGISTER(type)     \
  REGISTER_KERNEL_BUILDER( \
      Name(kRetOp).Device(DEVICE_GPU).TypeConstraint<type>("T"), RetvalOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER)
TF_CALL_bool(REGISTER) REGISTER_KERNEL_BUILDER(Name(kRetOp)
                                                   .Device(DEVICE_GPU)
                                                   .HostMemory("input")
                                                   .TypeConstraint<int32>("T"),
                                               RetvalOp);
REGISTER_KERNEL_BUILDER(Name(kRetOp)
                            .Device(DEVICE_GPU)
                            .TypeConstraint<ResourceHandle>("T")
                            .HostMemory("input"),
                        RetvalOp);

REGISTER_KERNEL_BUILDER(Name(kRetOp)
                            .Device(DEVICE_GPU)
                            .TypeConstraint<string>("T")
                            .HostMemory("input"),
                        RetvalOp);
#undef REGISTER

class PassOn : public OpKernel {
 public:
  explicit PassOn(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES(ctx, ctx->num_inputs() == ctx->num_outputs(),
                errors::Internal("#inputs != #outputs : ", ctx->num_inputs(),
                                 " vs. ", ctx->num_outputs()));
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      OP_REQUIRES(
          ctx, input_type(i) == output_type(i),
          errors::Internal("Input and output types for position ", i,
                           " do not match: ", DataTypeString(input_type(i)),
                           " vs. ", DataTypeString(output_type(i))));
    }
  }

  void Compute(OpKernelContext* ctx) override {
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      ctx->set_output(i, ctx->input(i));
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("_ListToArray").Device(DEVICE_CPU), PassOn);
REGISTER_KERNEL_BUILDER(Name("_ArrayToList").Device(DEVICE_CPU), PassOn);

#define REGISTER_GPU_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_ListToArray").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      PassOn);                                                           \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_ArrayToList").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      PassOn);

REGISTER_GPU_KERNELS(Eigen::half);
REGISTER_GPU_KERNELS(float);
REGISTER_GPU_KERNELS(double);

#undef REGISTER_GPU_KERNELS

REGISTER_KERNEL_BUILDER(Name("_ListToArray")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        PassOn);
REGISTER_KERNEL_BUILDER(Name("_ArrayToList")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        PassOn);

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("_ListToArray").Device(DEVICE_SYCL).TypeConstraint<type>("T"), \
      PassOn);                                                            \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("_ArrayToList").Device(DEVICE_SYCL).TypeConstraint<type>("T"), \
      PassOn);

REGISTER_SYCL_KERNELS(float);
REGISTER_SYCL_KERNELS(double);

#undef REGISTER_SYCL_KERNELS

REGISTER_KERNEL_BUILDER(Name("_ListToArray")
                            .Device(DEVICE_SYCL)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        PassOn);
REGISTER_KERNEL_BUILDER(Name("_ArrayToList")
                            .Device(DEVICE_SYCL)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        PassOn);
#endif  // TENSORFLOW_USE_SYCL
}  // namespace tensorflow
