"""Default (OSS) build versions of TensorFlow general-purpose build extensions."""

load(
    "//tensorflow:tensorflow.bzl",
    _ADDITIONAL_API_INDEXABLE_SETTINGS = "ADDITIONAL_API_INDEXABLE_SETTINGS",
    _cc_header_only_library = "cc_header_only_library",
    _custom_op_cc_header_only_library = "custom_op_cc_header_only_library",
    _clean_dep = "clean_dep",
    _cuda_py_test = "cuda_py_test",
    _filegroup = "filegroup",
    _genrule = "genrule",
    _get_compatible_with_portable = "get_compatible_with_portable",
    _if_indexing_source_code = "if_indexing_source_code",
    _if_not_mobile_or_arm_or_macos_or_lgpl_restricted = "if_not_mobile_or_arm_or_macos_or_lgpl_restricted",
    _if_portable = "if_portable",
    _internal_tfrt_deps = "internal_tfrt_deps",
    _pybind_extension = "pybind_extension",
    _pybind_library = "pybind_library",
    _pytype_library = "pytype_library",
    _pywrap_tensorflow_macro = "pywrap_tensorflow_macro",
    _replace_with_portable_tf_lib_when_required = "replace_with_portable_tf_lib_when_required",
    _tensorflow_opensource_extra_deps = "tensorflow_opensource_extra_deps",
    _tf_cc_shared_library = "tf_cc_shared_library",
    _tf_cuda_cc_test = "tf_cuda_cc_test",
    _tf_cuda_cc_tests = "tf_cuda_cc_tests",
    _tf_custom_op_py_library = "tf_custom_op_py_library",
    _tf_disable_ptxas_warning_flags = "tf_disable_ptxas_warning_flags",
    _tf_external_workspace_visible = "tf_external_workspace_visible",
    _tf_gen_op_libs = "tf_gen_op_libs",
    _tf_gen_op_wrapper_cc = "tf_gen_op_wrapper_cc",
    _tf_gen_op_wrappers_cc = "tf_gen_op_wrappers_cc",
    _tf_generate_proto_text_sources = "tf_generate_proto_text_sources",
    _tf_grpc_cc_dependencies = "tf_grpc_cc_dependencies",
    _tf_grpc_dependencies = "tf_grpc_dependencies",
    _tf_kernel_library = "tf_kernel_library",
    _tf_monitoring_framework_deps = "tf_monitoring_framework_deps",
    _tf_monitoring_python_deps = "tf_monitoring_python_deps",
    _tf_py_build_info_genrule = "tf_py_build_info_genrule",
    _tf_py_test = "tf_py_test",
    _tf_pybind_cc_library_wrapper = "tf_pybind_cc_library_wrapper",
    _tf_python_framework_friends = "tf_python_framework_friends",
    _tf_python_pybind_extension = "tf_python_pybind_extension",
    _tf_selective_registration_deps = "tf_selective_registration_deps",
    _tf_version_info_genrule = "tf_version_info_genrule",
    _tfcompile_dfsan_abilists = "tfcompile_dfsan_abilists",
    _tfcompile_dfsan_enabled = "tfcompile_dfsan_enabled",
    _tfcompile_target_cpu = "tfcompile_target_cpu",
    _pywrap_aware_tf_cc_shared_object = "pywrap_aware_tf_cc_shared_object",
)
load("@local_tsl//third_party/py/rules_pywrap:pywrap.default.bzl",
    _pywrap_aware_filegroup = "pywrap_aware_filegroup",
    _pywrap_aware_genrule = "pywrap_aware_genrule",
    _pywrap_aware_cc_import = "pywrap_aware_cc_import",
    _pywrap_aware_py_strict_library = "pywrap_aware_py_strict_library",
    _pywrap_library = "pywrap_library",
    _pywrap_common_library = "pywrap_common_library",
    _stripped_cc_info = "stripped_cc_info",
)

clean_dep = _clean_dep
if_not_mobile_or_arm_or_macos_or_lgpl_restricted = _if_not_mobile_or_arm_or_macos_or_lgpl_restricted
if_portable = _if_portable
ADDITIONAL_API_INDEXABLE_SETTINGS = _ADDITIONAL_API_INDEXABLE_SETTINGS
if_indexing_source_code = _if_indexing_source_code
pywrap_tensorflow_macro = _pywrap_tensorflow_macro
tf_cc_shared_library = _tf_cc_shared_library
pytype_library = _pytype_library
tf_py_test = _tf_py_test
tf_py_strict_test = _tf_py_test
cuda_py_test = _cuda_py_test
cuda_py_strict_test = _cuda_py_test
tf_cuda_cc_test = _tf_cuda_cc_test
tf_cuda_cc_tests = _tf_cuda_cc_tests
tf_py_build_info_genrule = _tf_py_build_info_genrule
tf_version_info_genrule = _tf_version_info_genrule
tf_custom_op_py_library = _tf_custom_op_py_library
tf_custom_op_py_strict_library = _tf_custom_op_py_library
tensorflow_opensource_extra_deps = _tensorflow_opensource_extra_deps
pybind_library = _pybind_library
pybind_extension = _pybind_extension
tf_python_pybind_extension = _tf_python_pybind_extension
tf_pybind_cc_library_wrapper = _tf_pybind_cc_library_wrapper
tf_monitoring_framework_deps = _tf_monitoring_framework_deps
tf_monitoring_python_deps = _tf_monitoring_python_deps
tf_selective_registration_deps = _tf_selective_registration_deps
tfcompile_target_cpu = _tfcompile_target_cpu
tfcompile_dfsan_enabled = _tfcompile_dfsan_enabled
tfcompile_dfsan_abilists = _tfcompile_dfsan_abilists
tf_external_workspace_visible = _tf_external_workspace_visible
tf_grpc_dependencies = _tf_grpc_dependencies
tf_grpc_cc_dependencies = _tf_grpc_cc_dependencies
get_compatible_with_portable = _get_compatible_with_portable
cc_header_only_library = _cc_header_only_library
custom_op_cc_header_only_library = _custom_op_cc_header_only_library
tf_gen_op_libs = _tf_gen_op_libs
tf_gen_op_wrapper_cc = _tf_gen_op_wrapper_cc
tf_gen_op_wrappers_cc = _tf_gen_op_wrappers_cc
tf_generate_proto_text_sources = _tf_generate_proto_text_sources
tf_kernel_library = _tf_kernel_library
filegroup = _filegroup
genrule = _genrule
internal_tfrt_deps = _internal_tfrt_deps
tf_disable_ptxas_warning_flags = _tf_disable_ptxas_warning_flags
replace_with_portable_tf_lib_when_required = _replace_with_portable_tf_lib_when_required
tf_python_framework_friends = _tf_python_framework_friends
pywrap_aware_tf_cc_shared_object = _pywrap_aware_tf_cc_shared_object
pywrap_aware_filegroup = _pywrap_aware_filegroup
pywrap_aware_genrule = _pywrap_aware_genrule
pywrap_aware_cc_import = _pywrap_aware_cc_import
pywrap_aware_py_strict_library = _pywrap_aware_py_strict_library
pywrap_library = _pywrap_library
pywrap_common_library = _pywrap_common_library
stripped_cc_info = _stripped_cc_info