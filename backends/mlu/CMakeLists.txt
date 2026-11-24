# Cambricon MLU provider.
#
# Registers a PrivateUse1 backend named "mlu": the runtime, the kernel host
# integration and the BANG C device code all live under this directory, so no
# InfiniTrain source file is modified. BANG C sources are compiled by cncc with
# a custom command; host sources are ordinary C++ compiled by the selected host
# compiler.

find_program(CNCC_EXECUTABLE cncc HINTS "${NEUWARE_HOME}/bin" REQUIRED)
find_library(CNRT_LIBRARY cnrt HINTS "${NEUWARE_HOME}/lib64" REQUIRED)

# CNNL ships separately from the base Neuware image. When it is missing the
# provider retains the host GEMM fallback and does not compile the CNNL projection.
option(INFINITRAIN_MLU_WITH_CNNL "Enable the CNNL handle and BF16 vocabulary projection" ON)
if(INFINITRAIN_MLU_WITH_CNNL)
  find_library(CNNL_LIBRARY cnnl HINTS "${NEUWARE_HOME}/lib64")
  if(NOT CNNL_LIBRARY OR NOT EXISTS "${NEUWARE_HOME}/include/cnnl.h")
    message(WARNING
      "CNNL was not found under ${NEUWARE_HOME}; configuring without BLAS "
      "handle support. Install CNNL and reconfigure to enable it.")
    set(INFINITRAIN_MLU_WITH_CNNL OFF)
  endif()
endif()

# BANG C target architecture; empty lets cncc use its default.
set(MLU_ARCH "" CACHE STRING "BANG C target arch, for example mtp_592")

option(INFINITRAIN_MLU_BUILD_TEST_TOOLS
  "Build the MLU operator regression tools" ON)

# ------------------------------------------------------------------------------
# Runtime, registration entry and kernel host integration (host compiler)
# ------------------------------------------------------------------------------
set(MLU_BACKEND_SOURCES
  src/backend.cc
  src/runtime/mlu_runtime_common.cc
  src/runtime/mlu_guard_impl.cc
  src/kernels/no_op.cc
  src/kernels/fill.cc
  src/kernels/cast.cc
  src/kernels/accumulate_grad.cc
  src/kernels/elementwise.cc
  src/kernels/embedding.cc
  src/kernels/layernorm.cc
  src/kernels/softmax.cc
  src/kernels/cross_entropy.cc
  src/kernels/transform.cc
  src/kernels/linear.cc
)

add_library(infini_train_mlu STATIC ${MLU_BACKEND_SOURCES})
add_library(InfiniTrain::Backend::MLU ALIAS infini_train_mlu)

# Only include/infini_train_mlu is public: examples and tools consume the
# registration entry point, never the provider's internal layout. Internal
# headers (common/, runtime/, kernels/) resolve against src/, which stays
# PRIVATE and is added explicitly by the targets that need it.
target_include_directories(infini_train_mlu
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${INFINITRAIN_SOURCE_DIR}"
)
target_include_directories(infini_train_mlu SYSTEM PRIVATE
    "${NEUWARE_HOME}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/bang_host"
)

target_link_libraries(infini_train_mlu PUBLIC
  InfiniTrain::infini_train
  ${CNRT_LIBRARY}
)

if(INFINITRAIN_MLU_WITH_CNNL)
  target_compile_definitions(infini_train_mlu PUBLIC USE_CNNL=1)
  target_link_libraries(infini_train_mlu PUBLIC ${CNNL_LIBRARY})
endif()

# ------------------------------------------------------------------------------
# BANG C kernels (*.mlu): CMake has no BANG C language, so cncc is invoked
# directly and the resulting objects form their own archive.
# ------------------------------------------------------------------------------
file(GLOB MLU_KERNEL_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.mlu")

set(MLU_KERNEL_OBJECTS)
foreach(src ${MLU_KERNEL_SOURCES})
  get_filename_component(name ${src} NAME_WE)
  set(obj "${CMAKE_CURRENT_BINARY_DIR}/${name}.o")
  set(arch_flags)
  if(MLU_ARCH)
    list(APPEND arch_flags --bang-mlu-arch=${MLU_ARCH})
  endif()
  add_custom_command(
    OUTPUT ${obj}
    COMMAND ${CMAKE_COMMAND} -E env NEUWARE_HOME=${NEUWARE_HOME}
      ${CNCC_EXECUTABLE} -std=c++17 -fPIC -c ${src} -o ${obj} ${arch_flags}
      "-I${CMAKE_CURRENT_SOURCE_DIR}/src"
      "-I${NEUWARE_HOME}/include"
    DEPENDS ${src}
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mlu_kernel_api.h"
    COMMENT "cncc ${name}.mlu"
    VERBATIM
  )
  list(APPEND MLU_KERNEL_OBJECTS ${obj})
endforeach()

add_library(infini_train_mlu_kernels STATIC ${MLU_KERNEL_OBJECTS})
set_target_properties(infini_train_mlu_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(infini_train_mlu_kernels PUBLIC infini_train_mlu)

# Static registrations (framework, CPU kernels, MLU runtime and MLU kernels) are
# only discovered through global initialization, so final executables must link
# them whole-archive. Reuse the framework helper instead of spelling out the
# link-group rules here.
add_library(infini_train_backend_mlu_executable INTERFACE)
add_library(InfiniTrain::Backend::MLUExecutable ALIAS
  infini_train_backend_mlu_executable)
link_infini_train_exe(infini_train_backend_mlu_executable
  EXTRA_ARCHIVES infini_train_mlu infini_train_mlu_kernels)

if(BUILD_TEST)
  message(FATAL_ERROR
    "The MLU provider does not wire the shared InfiniTrain test suites yet. "
    "Configure with -DBUILD_TEST=OFF and use the MLU regression tools "
    "(-DINFINITRAIN_MLU_BUILD_TEST_TOOLS=ON) instead of reporting CPU test "
    "results for the MLU device.")
endif()

if(INFINITRAIN_MLU_BUILD_TEST_TOOLS)
  add_subdirectory(tests)
endif()

if(INFINITRAIN_BACKENDS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
