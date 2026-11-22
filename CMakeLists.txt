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
include(GNUInstallDirs)
option(CXXGPR_BUILD_TESTS "Build cxxgpr unit and verification tests" ${PROJECT_IS_TOP_LEVEL})
option(CXXGPR_BUILD_CANARY "Build numeric canary executable" ${PROJECT_IS_TOP_LEVEL})
option(CXXGPR_BUILD_BENCHMARKS "Build Google Benchmark targets" OFF)
option(CXXGPR_BUILD_EXAMPLES "Build the documentation example programs under docs/examples" OFF)
option(CXXGPR_INSTALL "Generate CMake install targets for cxxgpr" ${PROJECT_IS_TOP_LEVEL})

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
    src/gpr/distance_engine.cpp
    src/gpr/gpr_engine.cpp
    src/kernel/compiled.cpp
    src/kernel/custom.cpp
    src/kernel/distance.cpp
    src/kernel/distance_cache.cpp
    src/kernel/exp.cpp
    src/kernel/leaf_eval.cpp
    src/kernel/leaves.cpp
    src/kernel/spec.cpp
    src/linalg/cholesky.cpp
    src/linalg/dense.cpp
    src/linalg/jitter.cpp
    src/linalg/packed_blas3.cpp
    src/linalg/parallel.cpp
    src/optimizer/bounds.cpp
    src/optimizer/fixed.cpp
    src/optimizer/fsa.cpp
    src/optimizer/lbfgs.cpp
    src/optimizer/nelder_mead.cpp
    src/optimizer/trust_region.cpp
    src/persist/config_json.cpp
    src/persist/persist.cpp
    src/persist/persist_distance.cpp
    src/persist/safetensors.cpp
    src/persist/save_id.cpp
    src/precision/refine.cpp
    src/transform/columnwise.cpp
    src/transform/input.cpp
    src/transform/pipeline.cpp
    src/transform/target.cpp
    src/workspace/buffers.cpp
    src/workspace/inference.cpp
    src/workspace/leaf_cache.cpp
    src/workspace/mll.cpp
)
add_library(cxxgpr::cxxgpr ALIAS cxxgpr)
add_dependencies(cxxgpr cxxgpr_git_revision)
cxxgpr_configure_target(cxxgpr)

target_include_directories(cxxgpr
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CXXGPR_GENERATED_INCLUDE_DIR}
)

target_link_libraries(cxxgpr
    PRIVATE
        $<BUILD_INTERFACE:cxxgpr::eigen>
        $<INSTALL_INTERFACE:Eigen3::Eigen>
        nlohmann_json::nlohmann_json
)
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(cxxgpr PRIVATE OpenMP::OpenMP_CXX)
endif()

if(CXXGPR_BUILD_CANARY)
    add_executable(cxxgpr_numeric_canary qa/numeric_canary/main.cpp)
    cxxgpr_configure_target(cxxgpr_numeric_canary)
    target_link_libraries(cxxgpr_numeric_canary PRIVATE cxxgpr cxxgpr::eigen)
endif()

if(CXXGPR_BUILD_TESTS AND BUILD_TESTING)
    add_subdirectory(tests)
endif()

if(CXXGPR_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(CXXGPR_BUILD_EXAMPLES)
    add_subdirectory(docs/examples)
endif()

if(CXXGPR_INSTALL)
    include(CMakePackageConfigHelpers)

    install(TARGETS cxxgpr
        EXPORT cxxgprTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )

    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/cxxgpr"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING PATTERN "*.hpp"
    )

    install(EXPORT cxxgprTargets
        FILE cxxgprTargets.cmake
        NAMESPACE cxxgpr::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cxxgpr"
    )

    set(CXXGPR_HAS_OPENMP OFF)
    if(TARGET OpenMP::OpenMP_CXX)
        set(CXXGPR_HAS_OPENMP ON)
    endif()

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cxxgprConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cxxgprConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cxxgpr"
    )

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/cxxgprConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMinorVersion
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/cxxgprConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/cxxgprConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cxxgpr"
    )

    install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
        "${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY_NOTICES.md"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/cxxgpr"
    )
endif()
