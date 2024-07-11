/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/pjrt/mlir_to_hlo.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"  // from @llvm-project
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "stablehlo/dialect/Serialization.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/transforms/Passes.h"  // from @stablehlo
#include "xla/debug_options_flags.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/utils.h"
#include "xla/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {

absl::Status MlirToXlaComputation(mlir::ModuleOp module,
                                  XlaComputation& xla_computation,
                                  bool use_tuple_args, bool return_tuple) {
  mlir::MLIRContext* context = module->getContext();
  mlir::BaseScopedDiagnosticHandler diagnostic_handler(context);
  {
    mlir::PassManager pm(context);
    pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::mhlo::createChloLegalizeToHloPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    // In order to export to XLA, we must sink constants to control flow
    // regions, since XLA uses functional control flow.
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::mhlo::createSinkConstantsToControlFlowPass());
    if (failed(pm.run(module))) {
      VLOG(1) << "MHLO->HLO lowering passes failed.";
      module->dump();
      return diagnostic_handler.ConsumeStatus();
    }

    VLOG(5) << "MHLO module after lowering, before HLO import ";
    if (VLOG_IS_ON(5)) {
      module->dump();
    }
  }

  // TODO(b/345414638): Delete when we move Shardonnay as the first pass in the
  // XLA pipeline.
  if (use_tuple_args && GetDebugOptionsFromFlags().xla_use_shardonnay()) {
    // Shardonnay can't handle tuple args when round-tripping. So delay using
    // tuples until after Shardonnay is run.
    sdy::addFrontendAttribute(module, sdy::kUseTupleArgs,
                              mlir::StringAttr::get(context, "t"));
    use_tuple_args = false;
  }

  // create config options use use_tuple_args, return_tuple set:
  mlir::MlirToHloConversionOptions options;
  options.use_tuple_args = use_tuple_args;
  options.return_tuple = return_tuple;

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      mlir::ConvertMlirHloToHloModule(module, options));

  xla_computation = XlaComputation(hlo_module->ToProto());
  return absl::OkStatus();
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ParseMlirModuleString(
    absl::string_view mlir_module_str, mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::ml_program::MLProgramDialect>();
  registry.insert<mlir::shape::ShapeDialect>();
  mlir::func::registerAllExtensions(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  context.appendDialectRegistry(registry);

  mlir::BaseScopedDiagnosticHandler diagnostic_handler(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(
          llvm::StringRef(mlir_module_str.data(), mlir_module_str.size()),
          // IR may be invalid because some fields may be using DenseElements
          // instead of DenseArray. We rectify that below and verify after.
          mlir::ParserConfig{&context, /*verifyAfterParse=*/false});
  if (!module) {
    return diagnostic_handler.ConsumeStatus();
  }

  // In
  // https://github.com/google/jax/commit/184e3a88004680dbf34328b05c5fc0d869cc4a93,
  // fields on some ops were changed to use Dense{Bool,I64}ArrayAttr instead of
  // I64DenseElementsAttr (DenseIntElementsAttr). Some clients still expect
  // dense elements, not dense arrays, so when serializing we always convert the
  // arrays to elements. The elements need to be converted back to arrays when
  // deserializing.
  // TODO: b/320507168 - Remove the conversion code, and verifyAfterParse.
  TF_RETURN_IF_ERROR(UpgradeVersionedStablehlo(*module));
  if (failed(module->verifyInvariants())) {
    VLOG(1) << "MLIR verification failed.";
    module->dump();
    return diagnostic_handler.ConsumeStatus();
  }
  return std::move(module);
}

absl::Status ParseMlirModuleStringAndConvertToXlaComputation(
    absl::string_view mlir_module_str, XlaComputation& xla_computation,
    bool use_tuple_args, bool return_tuple) {
  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      xla::ParseMlirModuleString(mlir_module_str, context));
  return xla::MlirToXlaComputation(*module, xla_computation, use_tuple_args,
                                   return_tuple);
}

absl::StatusOr<std::string> SerializeUsingNativeBytecode(
    mlir::ModuleOp module) {
  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  mlir::BytecodeWriterConfig config;
  // Pin bytecode version to 1 until transition to stable.
  // TODO: b/285913864 - Remove post enabling frameworks to set it.
  config.setDesiredBytecodeVersion(1);
  // In
  // https://github.com/google/jax/commit/184e3a88004680dbf34328b05c5fc0d869cc4a93,
  // fields on some ops were changed to use Dense{Bool,I64}ArrayAttr instead of
  // I64DenseElementsAttr (DenseIntElementsAttr). Some clients still expect
  // dense elements, not dense arrays, so convert the arrays to elements before
  // serializing. The elements need to be converted back to arrays when
  // deserializing.
  // TODO: b/320507168 - Remove this conversion code.
  mlir::OwningOpRef<mlir::ModuleOp> cloned = module.clone();
  if (mlir::failed(mlir::writeBytecodeToFile(*cloned, os, config))) {
    return absl::InvalidArgumentError("mlir::writeBytecodeToFile failed");
  }
  return bytecode;
}

absl::StatusOr<std::string> SerializeUsingVersionedStablehlo(
    mlir::ModuleOp mlir_module, absl::string_view target, bool inplace) {
  mlir::MLIRContext* context = mlir_module->getContext();
  mlir::BaseScopedDiagnosticHandler diagnostic_handler(context);

  // Legalize CHLO -> [StableHLO+Shape] -> StableHLO
  // Preserve higher-level ops with XLA support. To be replaced by composites.
  mlir::PassManager pm(context);
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createChloLegalizeToHighLevelMhloPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createChloLegalizeToStablehloPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createShapeLegalizeToStablehloPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  if (!mlir::succeeded(pm.run(mlir_module))) {
    const absl::Status status = diagnostic_handler.ConsumeStatus();
    return absl::InvalidArgumentError(
        absl::StrCat("CHLO => [MHLO+Shape] => StableHLO failed;\n\nDetailed "
                     "error from MLIR: ",
                     status.message()));
  }

  // Avoid mutating the original module if it will be reused elsewhere
  mlir::OwningOpRef<mlir::ModuleOp> cloned;
  if (!inplace) {
    cloned = mlir_module.clone();
    mlir_module = *cloned;
  }

  // Serialize portable artifact
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  if (failed(mlir::stablehlo::serializePortableArtifact(mlir_module, target,
                                                        os))) {
    const absl::Status status = diagnostic_handler.ConsumeStatus();
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to serialize StableHLO;\n\nDetailed error from MLIR: ",
        status.message()));
  }
  return buffer;
}

absl::Status UpgradeVersionedStablehlo(mlir::ModuleOp mlir_module) {
  // Upgrade if VHLO
  mlir::PassManager pm(mlir_module->getContext());
  mlir::stablehlo::createStablehloDeserializePipeline(pm);
  if (!mlir::succeeded(pm.run(mlir_module)))
    return xla::InvalidArgument("Failed to upgrade versioned StableHLO.");
  return absl::OkStatus();
}

std::string GetDefaultStablehloVersion() {
  // This version must be >=12w old.
  // See https://github.com/openxla/stablehlo/tags
  //   0.19.0 - Mar 13, 2024
  return "0.19.0";
}

absl::StatusOr<std::string> Serialize(mlir::ModuleOp module,
                                      std::optional<int64_t> /*plugin_version*/,
                                      absl::string_view target, bool inplace) {
  // Current PJRT users expect 12 weeks forward compat, VHLO provides this
  // compat.
  // TODO (b/344930098): Allow VHLO interop and remove the all_stablehlo check
  bool all_stablehlo = true;
  module->walk([&](mlir::Operation* op) {
    if (!llvm::isa<mlir::ModuleOp>(op) &&
        !llvm::isa<mlir::stablehlo::StablehloDialect, mlir::func::FuncDialect,
                   mlir::chlo::ChloDialect>(op->getDialect())) {
      std::cout << op->getDialect()->getNamespace().str() << "\n";
      all_stablehlo = false;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!all_stablehlo) {
    return SerializeUsingNativeBytecode(module);
  }
  return SerializeUsingVersionedStablehlo(module, target, inplace);
}

}  // namespace xla
