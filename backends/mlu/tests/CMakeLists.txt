# MLU regression tools. They only need the provider sources and the framework,
# so they are available even when the model examples are switched off.

# Operator comparison against the CPU backend, plus the comparator self-checks
# and numerical regression gates.
add_executable(mlu_op_test op_test.cc)
link_infini_train_exe(mlu_op_test
  EXTRA_ARCHIVES infini_train_mlu infini_train_mlu_kernels)
set_target_properties(mlu_op_test PROPERTIES
  OUTPUT_NAME op_test
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
