# Platform-specific build configurations.

load("@protobuf//:protobuf.bzl", "proto_gen")
load("@protobuf//:protobuf.bzl", "py_proto_library")
load("//tensorflow:tensorflow.bzl", "if_not_mobile")
load("//tensorflow:tensorflow.bzl", "if_not_windows")

# Appends a suffix to a list of deps.
def tf_deps(deps, suffix):
  tf_deps = []

  # If the package name is in shorthand form (ie: does not contain a ':'),
  # expand it to the full name.
  for dep in deps:
    tf_dep = dep

    if not ":" in dep:
      dep_pieces = dep.split("/")
      tf_dep += ":" + dep_pieces[len(dep_pieces) - 1]

    tf_deps += [tf_dep + suffix]

  return tf_deps


def _CcHdrs(srcs, use_grpc_plugin=False):
  ret = [s[:-len(".proto")] + ".pb.h" for s in srcs]
  if use_grpc_plugin:
    ret += [s[:-len(".proto")] + ".grpc.pb.h" for s in srcs]
  return ret

def _CcSrcs(srcs, use_grpc_plugin=False):
  ret = [s[:-len(".proto")] + ".pb.cc" for s in srcs]
  if use_grpc_plugin:
    ret += [s[:-len(".proto")] + ".grpc.pb.cc" for s in srcs]
  return ret

def cc_proto_library(
        name,
        srcs=[],
        deps=[],
        cc_libs=[],
        include=None,
        protoc="//:protoc",
        internal_bootstrap_hack=False,
        use_grpc_plugin=False,
        default_runtime="//:protobuf",
        default_header=False,
        **kargs):
  """Bazel rule to create a C++ protobuf library from proto source files
  NOTE: the rule is only an internal workaround to generate protos. The
  interface may change and the rule may be removed when bazel has introduced
  the native rule.
  Args:
    name: the name of the cc_proto_library.
    srcs: the .proto files of the cc_proto_library.
    deps: a list of dependency labels; must be cc_proto_library.
    cc_libs: a list of other cc_library targets depended by the generated
        cc_library.
    include: a string indicating the include path of the .proto files.
    protoc: the label of the protocol compiler to generate the sources.
    internal_bootstrap_hack: a flag indicate the cc_proto_library is used only
        for bootstraping. When it is set to True, no files will be generated.
        The rule will simply be a provider for .proto files, so that other
        cc_proto_library can depend on it.
    use_grpc_plugin: a flag to indicate whether to call the grpc C++ plugin
        when processing the proto files.
    default_runtime: the implicitly default runtime which will be depended on by
        the generated cc_library target.
    **kargs: other keyword arguments that are passed to cc_library.
  """

  includes = []
  if include != None:
    includes = [include]

  if internal_bootstrap_hack:
    # For pre-checked-in generated files, we add the internal_bootstrap_hack
    # which will skip the codegen action.
    proto_gen(
        name=name + "_genproto",
        srcs=srcs,
        deps=[s + "_genproto" for s in deps],
        includes=includes,
        protoc=protoc,
        visibility=["//visibility:public"],
    )
    # An empty cc_library to make rule dependency consistent.
    native.cc_library(
        name=name,
        **kargs)
    return

  grpc_cpp_plugin = None
  if use_grpc_plugin:
    grpc_cpp_plugin = "//external:grpc_cpp_plugin"

  gen_srcs = _CcSrcs(srcs, use_grpc_plugin)
  gen_hdrs = _CcHdrs(srcs, use_grpc_plugin)
  outs = gen_srcs + gen_hdrs

  proto_gen(
      name=name + "_genproto",
      srcs=srcs,
      deps=[s + "_genproto" for s in deps],
      includes=includes,
      protoc=protoc,
      plugin=grpc_cpp_plugin,
      plugin_language="grpc",
      gen_cc=1,
      outs=outs,
      visibility=["//visibility:public"],
  )

  if default_runtime and not default_runtime in cc_libs:
    cc_libs += [default_runtime]
  if use_grpc_plugin:
    cc_libs += ["//external:grpc_lib"]

  if default_header:
    header_only_name = name
    impl_name = name + "_impl"
  else:
    header_only_name = name + "_headers_only"
    impl_name = name
    
  native.cc_library(
      name=impl_name,
      srcs=gen_srcs,
      hdrs=gen_hdrs,
      deps=cc_libs + deps,
      includes=includes,
      **kargs)
  native.cc_library(
      name=header_only_name,
      deps=["@protobuf//:protobuf_headers"],
      hdrs=gen_hdrs,
      **kargs)

def tf_proto_library_cc(name, srcs = [], has_services = None,
                        protodeps = [],
                        visibility = [], testonly = 0,
                        cc_libs = [],
                        cc_stubby_versions = None,
                        cc_grpc_version = None,
                        j2objc_api_version = 1,
                        cc_api_version = 2, go_api_version = 2,
                        java_api_version = 2, py_api_version = 2,
                        js_api_version = 2, js_codegen = "jspb",
                        default_header = False):
  native.filegroup(
      name = name + "_proto_srcs",
      srcs = srcs + tf_deps(protodeps, "_proto_srcs"),
      testonly = testonly,
      visibility = visibility,
  )

  use_grpc_plugin = None
  if cc_grpc_version:
    use_grpc_plugin = True
  cc_proto_library(
      name = name + "_cc",
      srcs = srcs,
      deps = tf_deps(protodeps, "_cc") + ["@protobuf//:cc_wkt_protos"],
      cc_libs = cc_libs + ["@protobuf//:protobuf_headers"],
      copts = if_not_windows([
          "-Wno-unknown-warning-option",
          "-Wno-unused-but-set-variable",
          "-Wno-sign-compare",
      ]),
      protoc = "@protobuf//:protoc",
      default_runtime = "@protobuf//:protobuf_headers",
      use_grpc_plugin = use_grpc_plugin,
      testonly = testonly,
      visibility = visibility,
      default_header = default_header,
  )

def tf_proto_library_py(name, srcs=[], protodeps=[], deps=[], visibility=[],
                        testonly=0,
                        srcs_version="PY2AND3"):
  py_proto_library(
      name = name + "_py",
      srcs = srcs,
      srcs_version = srcs_version,
      deps = deps + tf_deps(protodeps, "_py") + ["@protobuf//:protobuf_python"],
      protoc = "@protobuf//:protoc",
      default_runtime = "@protobuf//:protobuf_python",
      visibility = visibility,
      testonly = testonly,
  )

def tf_proto_library(name, srcs = [], has_services = None,
                     protodeps = [],
                     visibility = [], testonly = 0,
                     cc_libs = [],
                     cc_api_version = 2, cc_grpc_version = None,
                     go_api_version = 2,
                     j2objc_api_version = 1,
                     java_api_version = 2, py_api_version = 2,
                     js_api_version = 2, js_codegen = "jspb",
                     default_header = False):
  """Make a proto library, possibly depending on other proto libraries."""
  tf_proto_library_cc(
      name = name,
      srcs = srcs,
      protodeps = protodeps,
      cc_grpc_version = cc_grpc_version,
      cc_libs = cc_libs,
      testonly = testonly,
      visibility = visibility,
      default_header = default_header,
  )

  tf_proto_library_py(
      name = name,
      srcs = srcs,
      protodeps = protodeps,
      srcs_version = "PY2AND3",
      testonly = testonly,
      visibility = visibility,
  )

def tf_additional_lib_hdrs(exclude = []):
  windows_hdrs = native.glob([
      "platform/default/*.h",
      "platform/windows/*.h",
      "platform/posix/error.h",
  ], exclude = exclude)
  return select({
    "//tensorflow:windows" : windows_hdrs,
    "//tensorflow:windows_msvc" : windows_hdrs,
    "//conditions:default" : native.glob([
        "platform/default/*.h",
        "platform/posix/*.h",
      ], exclude = exclude),
  })

def tf_additional_lib_srcs(exclude = []):
  windows_srcs = native.glob([
      "platform/default/*.cc",
      "platform/windows/*.cc",
      "platform/posix/error.cc",
  ], exclude = exclude)
  return select({
    "//tensorflow:windows" : windows_srcs,
    "//tensorflow:windows_msvc" : windows_srcs,
    "//conditions:default" : native.glob([
        "platform/default/*.cc",
        "platform/posix/*.cc",
      ], exclude = exclude),
  })

def tf_additional_minimal_lib_srcs():
  return [
      "platform/default/integral_types.h",
      "platform/default/mutex.h",
  ]

def tf_additional_proto_hdrs():
  return [
      "platform/default/integral_types.h",
      "platform/default/logging.h",
      "platform/default/protobuf.h"
  ]

def tf_additional_proto_srcs():
  return [
      "platform/default/logging.cc",
      "platform/default/protobuf.cc",
  ]

def tf_env_time_hdrs():
  return [
      "platform/env_time.h",
  ]

def tf_env_time_srcs():
  win_env_time = native.glob([
    "platform/windows/env_time.cc",
    "platform/env_time.cc",
  ], exclude = [])
  return select({
    "//tensorflow:windows" : win_env_time,
    "//tensorflow:windows_msvc" : win_env_time,
    "//conditions:default" : native.glob([
        "platform/posix/env_time.cc",
        "platform/env_time.cc",
      ], exclude = []),
  })

def tf_additional_stream_executor_srcs():
  return ["platform/default/stream_executor.h"]

def tf_additional_cupti_wrapper_deps():
  return ["//tensorflow/core/platform/default/gpu:cupti_wrapper"]

def tf_additional_gpu_tracer_srcs():
  return ["platform/default/gpu_tracer.cc"]

def tf_additional_gpu_tracer_cuda_deps():
  return []

def tf_additional_gpu_tracer_deps():
  return []

def tf_additional_libdevice_data():
  return []

def tf_additional_libdevice_deps():
  return ["@local_config_cuda//cuda:cuda_headers"]

def tf_additional_libdevice_srcs():
  return ["platform/default/cuda_libdevice_path.cc"]

def tf_additional_test_deps():
  return []

def tf_additional_test_srcs():
  return [
      "platform/default/test_benchmark.cc",
  ] + select({
      "//tensorflow:windows" : [
          "platform/windows/test.cc"
        ],
      "//conditions:default" : [
          "platform/posix/test.cc",
        ],
    })

def tf_kernel_tests_linkstatic():
  return 0

def tf_additional_lib_defines():
  return select({
      "//tensorflow:with_jemalloc_linux_x86_64": ["TENSORFLOW_USE_JEMALLOC"],
      "//tensorflow:with_jemalloc_linux_ppc64le":["TENSORFLOW_USE_JEMALLOC"],
      "//conditions:default": [],
  })

def tf_additional_lib_deps():
  return select({
      "//tensorflow:with_jemalloc_linux_x86_64": ["@jemalloc"],
      "//tensorflow:with_jemalloc_linux_ppc64le": ["@jemalloc"],
      "//conditions:default": [],
  })

def tf_additional_core_deps():
  return select({
      "//tensorflow:with_gcp_support": [
          "//tensorflow/core/platform/cloud:gcs_file_system",
      ],
      "//conditions:default": [],
  }) + select({
      "//tensorflow:with_hdfs_support": [
          "//tensorflow/core/platform/hadoop:hadoop_file_system",
      ],
      "//conditions:default": [],
  })

# TODO(jart, jhseu): Delete when GCP is default on.
def tf_additional_cloud_op_deps():
  return select({
      "//tensorflow:windows": [],
      "//tensorflow:android": [],
      "//tensorflow:ios": [],
      "//tensorflow:with_gcp_support": [
        "//tensorflow/contrib/cloud:bigquery_reader_ops_op_lib",
      ],
      "//conditions:default": [],
  })

# TODO(jart, jhseu): Delete when GCP is default on.
def tf_additional_cloud_kernel_deps():
  return select({
      "//tensorflow:windows": [],
      "//tensorflow:android": [],
      "//tensorflow:ios": [],
      "//tensorflow:with_gcp_support": [
        "//tensorflow/contrib/cloud/kernels:bigquery_reader_ops",
      ],
      "//conditions:default": [],
  })

def tf_lib_proto_parsing_deps():
  return [
      ":protos_all_cc",
      "//tensorflow/core/platform/default/build_config:proto_parsing",
  ]

def tf_additional_verbs_lib_defines():
  return select({
      "//tensorflow:with_verbs_support": ["TENSORFLOW_USE_VERBS"],
      "//conditions:default": [],
  })

def tf_additional_mpi_lib_defines():
  return select({
      "//tensorflow:with_mpi_support": ["TENSORFLOW_USE_MPI"],
      "//conditions:default": [],
  })

def tf_pyclif_proto_library(name, proto_lib, proto_srcfile="", visibility=None,
                            **kwargs):
  pass
