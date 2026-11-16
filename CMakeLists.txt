# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.25)
cmake_policy(SET CMP0141 NEW)

project(cxxgpr
    VERSION 0.1.0
    DESCRIPTION "Exact Gaussian Process Regression library in C++23"
    LANGUAGES CXX)

set(CXXGPR_VERSION "${PROJECT_VERSION}")
set(CXXGPR_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
file(MAKE_DIRECTORY "${CXXGPR_GENERATED_INCLUDE_DIR}/cxxgpr")

include(CTest)
option(CXXGPR_BUILD_BENCHMARKS "Build Google Benchmark targets" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerPolicy)
include(ProjectOptions)
include(Dependencies)
include(AllocationAudit)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_config.hpp.in"
    "${CXXGPR_GENERATED_INCLUDE_DIR}/cxxgpr/build_config.hpp"
    @ONLY
)

add_custom_target(cxxgpr_git_revision
    BYPRODUCTS "${CXXGPR_GENERATED_INCLUDE_DIR}/cxxgpr/git_revision.hpp"
    COMMAND "${CMAKE_COMMAND}"
        "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}"
        "-DOUTPUT=${CXXGPR_GENERATED_INCLUDE_DIR}/cxxgpr/git_revision.hpp"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/GitRevision.cmake"
    COMMENT "Updating cxxgpr git revision metadata"
    VERBATIM
)

# Public headers under include/cxxgpr/ stay free of Eigen, nlohmann/json, and OpenMP.
add_library(cxxgpr STATIC
    src/core/build_info.cpp
    src/core/checked.cpp
    src/core/error.cpp
    src/core/interval.cpp
    src/core/likelihood.cpp
    src/core/rng.cpp
    src/kernel/compiled.cpp
    src/kernel/custom.cpp
    src/kernel/distance.cpp
    src/kernel/exp.cpp
    src/kernel/leaf_eval.cpp
    src/kernel/leaves.cpp
    src/kernel/spec.cpp
    src/linalg/cholesky.cpp
    src/linalg/dense.cpp
    src/linalg/jitter.cpp
    src/linalg/parallel.cpp
    src/precision/refine.cpp
    src/transform/columnwise.cpp
    src/transform/input.cpp
    src/transform/pipeline.cpp
    src/transform/target.cpp
)
add_library(cxxgpr::cxxgpr ALIAS cxxgpr)
add_dependencies(cxxgpr cxxgpr_git_revision)
cxxgpr_configure_target(cxxgpr)

target_include_directories(cxxgpr
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CXXGPR_GENERATED_INCLUDE_DIR}
)

target_link_libraries(cxxgpr
    PRIVATE
        cxxgpr::eigen
        nlohmann_json::nlohmann_json
)
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(cxxgpr PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(cxxgpr_numeric_canary qa/numeric_canary/main.cpp)
cxxgpr_configure_target(cxxgpr_numeric_canary)
target_link_libraries(cxxgpr_numeric_canary PRIVATE cxxgpr cxxgpr::eigen)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

if(CXXGPR_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
