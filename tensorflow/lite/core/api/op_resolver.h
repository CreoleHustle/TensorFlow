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
#ifndef TENSORFLOW_LITE_CORE_API_OP_RESOLVER_H_
#define TENSORFLOW_LITE_CORE_API_OP_RESOLVER_H_

#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Opaque type similar to TfLiteDelegate / TfLiteOpaqueDelegate.
// This is used for cases (e.g. when using "TF Lite with Google Play Services")
// where the TF Lite runtime might be built using a newer (or older)
// version of the TF Lite sources than the app, and hence might have a
// different definition of the TfLiteDelegate type. TF Lite APIs use
// TfLiteOpaqueDelegate rather than TfLiteDelegate when they want to
// refer to a delegate defined with that potentially different version
// of the TfLiteDelegate type.
struct TfLiteOpaqueDelegateStruct;

namespace tflite {

/// Abstract interface that returns TfLiteRegistrations given op codes or custom
/// op names. This is the mechanism that ops being referenced in the flatbuffer
/// model are mapped to executable function pointers (TfLiteRegistrations).
class OpResolver {
 public:
  /// Finds the op registration for a builtin operator by enum code.
  virtual const TfLiteRegistration* FindOp(tflite::BuiltinOperator op,
                                           int version) const = 0;
  /// Finds the op registration of a custom operator by op name.
  virtual const TfLiteRegistration* FindOp(const char* op,
                                           int version) const = 0;

  using TfLiteDelegatePtrVector =
      std::vector<std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>>;
  // Returns optional delegates for resolving and handling ops in the flatbuffer
  // model. This may be used in addition to the standard TfLiteRegistration
  // lookup for graph resolution.
  // WARNING: This API is deprecated, GetDelegateCreators is preferred.
  virtual TfLiteDelegatePtrVector GetDelegates(int num_threads) const {
    return {};
  }

  // Represents a function that creates a TfLite delegate instance.
  using TfLiteDelegateCreator =
      std::function<std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
          int /*num_threads*/)>;
  using TfLiteDelegateCreators = std::vector<TfLiteDelegateCreator>;
  // Returns a vector of delegate creators to create optional delegates for
  // resolving and handling ops in the flatbuffer model. This may be used in
  // addition to the standard TfLiteRegistration lookup for graph resolution.
  virtual TfLiteDelegateCreators GetDelegateCreators() const { return {}; }

  // Represent a function that creates an opaque delegate instance.
  using OpaqueDelegatePtr =
      std::unique_ptr<struct TfLiteOpaqueDelegateStruct,
                      void (*)(struct TfLiteOpaqueDelegateStruct*)>;
  using OpaqueDelegateCreator =
      std::function<OpaqueDelegatePtr(int /*num_threads*/)>;
  using OpaqueDelegateCreators = std::vector<OpaqueDelegateCreator>;
  // Returns a vector of delegate creators to create optional delegates for
  // resolving and handling ops in the flatbuffer model. This may be used in
  // addition to the standard TfLiteRegistration lookup for graph resolution.
  virtual OpaqueDelegateCreators GetOpaqueDelegateCreators() const {
    return {};
  }

  virtual ~OpResolver() {}

 private:
  /// Returns true if this OpResolver may contain any "user defined" ops.
  /// By "user defined" ops, we mean any op definitions other than those
  /// contained in tflite::ops::builtin::BuiltinOpResolver.
  ///
  /// If this method returns true, it doesn't necessarily mean that the
  /// OpResolver contains a user-defined op, just that the absence of
  /// user-defined ops can't be guaranteed.
  ///
  /// Note that "user-defined" ops are not the same as "custom" ops;
  /// BuiltinOpResolver may support certain "custom" ops, in addition to
  /// "builtin" ops, and may not support all of the "builtin" op enum values.
  virtual bool MayContainUserDefinedOps() const { return true; }

  friend class OpResolverInternal;
};

// Handles the logic for converting between an OperatorCode structure extracted
// from a flatbuffer and information about a registered operator
// implementation.
TfLiteStatus GetRegistrationFromOpCode(const OperatorCode* opcode,
                                       const OpResolver& op_resolver,
                                       ErrorReporter* error_reporter,
                                       const TfLiteRegistration** registration);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_API_OP_RESOLVER_H_
