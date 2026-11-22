# SPDX-License-Identifier: Apache-2.0
find_package(GTest CONFIG REQUIRED)
find_package(Threads REQUIRED)

add_library(cxxgpr_test_support STATIC
    support/allocation_audit.cpp
    support/small_stack.cpp
)
cxxgpr_configure_target(cxxgpr_test_support)
cxxgpr_configure_allocation_audit_target(cxxgpr_test_support)
target_include_directories(cxxgpr_test_support
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/support"
)
target_link_libraries(cxxgpr_test_support
    PUBLIC
        cxxgpr::eigen
        GTest::gtest
        Threads::Threads
)

function(cxxgpr_add_test test_name)
    cmake_parse_arguments(ARG "" "" "SOURCES;LABELS;LIBRARIES;DEFINITIONS" ${ARGN})
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "cxxgpr_add_test(${test_name}): SOURCES is required")
    endif()
    if(NOT ARG_LABELS)
        message(FATAL_ERROR "cxxgpr_add_test(${test_name}): LABELS is required")
    endif()

    string(REPLACE "." "_" _target_suffix "${test_name}")
    set(_target "cxxgpr_test_${_target_suffix}")

    add_executable("${_target}" ${ARG_SOURCES})
    cxxgpr_configure_target("${_target}")
    cxxgpr_configure_allocation_audit_target("${_target}")
    if(ARG_DEFINITIONS)
        target_compile_definitions("${_target}" PRIVATE ${ARG_DEFINITIONS})
    endif()
    target_link_libraries("${_target}" PRIVATE
        GTest::gtest_main
        cxxgpr::cxxgpr
        cxxgpr_test_support
        ${ARG_LIBRARIES}
    )
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries("${_target}" PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_test(NAME "${test_name}" COMMAND $<TARGET_FILE:${_target}>)
    set_tests_properties("${test_name}" PROPERTIES LABELS "${ARG_LABELS}")
endfunction()

cxxgpr_add_test(core.build_info
    SOURCES core/build_info_test.cpp
    LABELS unit property negative
    DEFINITIONS CXXGPR_MANIFEST_PATH="${CMAKE_SOURCE_DIR}/vcpkg.json"
    LIBRARIES nlohmann_json::nlohmann_json
)

cxxgpr_add_test(core.error
    SOURCES core/error_test.cpp
    LABELS unit
)

cxxgpr_add_test(core.interval
    SOURCES core/interval_test.cpp
    LABELS unit negative
)

cxxgpr_add_test(core.likelihood
    SOURCES core/likelihood_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(core.checked
    SOURCES core/checked_test.cpp
    LABELS unit negative
)

cxxgpr_add_test(core.views
    SOURCES core/views_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(core.rng
    SOURCES core/rng_test.cpp
    LABELS unit property
)

cxxgpr_add_test(precision.policy
    SOURCES precision/policy_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(precision.refine
    SOURCES precision/refine_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(linalg.jitter
    SOURCES linalg/jitter_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(linalg.dense
    SOURCES linalg/dense_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(linalg.cholesky
    SOURCES linalg/cholesky_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(linalg.parallel
    SOURCES linalg/parallel_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(transform.input
    SOURCES transform/input_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(transform.target
    SOURCES transform/target_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(transform.pipeline
    SOURCES transform/pipeline_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(transform.columnwise
    SOURCES transform/columnwise_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(kernel.leaves
    SOURCES kernel/leaves_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(kernel.spec
    SOURCES kernel/spec_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(kernel.compiled
    SOURCES kernel/compiled_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(kernel.distance
    SOURCES kernel/distance_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(kernel.exp
    SOURCES kernel/exp_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(kernel.custom
    SOURCES kernel/custom_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(kernel.distance_cache
    SOURCES kernel/distance_cache_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(workspace.leaf_cache
    SOURCES workspace/leaf_cache_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(workspace.mll
    SOURCES workspace/mll_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(workspace.inference
    SOURCES workspace/inference_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(optimizer.optimizers
    SOURCES optimizer/optimizer_test.cpp
    LABELS unit property negative
)

cxxgpr_add_test(gpr.models
    SOURCES gpr/gpr_test.cpp
    LABELS unit property negative integration
)

cxxgpr_add_test(gpr.distance
    SOURCES gpr/distance_gpr_test.cpp
    LABELS unit property negative integration
)

cxxgpr_add_test(persist.models
    SOURCES persist/persist_test.cpp
    LABELS unit property negative integration
)

cxxgpr_add_test(persist.distance
    SOURCES persist/distance_persist_test.cpp
    LABELS unit property negative integration
)

cxxgpr_add_test(integration.path_matrix
    SOURCES integration/path_matrix_test.cpp
    LABELS integration property
)

cxxgpr_add_test(verification.golden
    SOURCES verification/golden_test.cpp
    LABELS verification integration
)

cxxgpr_add_test(verification.hotpath_ratchet
    SOURCES verification/hotpath_ratchet_test.cpp
    LABELS verification property
)

if(DEFINED CXXGPR_THIRD_PARTY_NOTICES)
    add_test(NAME toolchain.third_party_notices
        COMMAND "${CMAKE_COMMAND}"
            "-DNOTICES_DIR=${CXXGPR_THIRD_PARTY_NOTICES}"
            "-DSHARE_DIR=${CXXGPR_THIRD_PARTY_SHARE}"
            "-DROOT_NOTICE=${CMAKE_SOURCE_DIR}/THIRD_PARTY_NOTICES.md"
            "-DTOP_BUILD_TESTING=${BUILD_TESTING}"
            "-DTOP_BUILD_BENCHMARKS=${CXXGPR_BUILD_BENCHMARKS}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/toolchain/check_notices.cmake"
    )
    set_tests_properties(toolchain.third_party_notices PROPERTIES
        LABELS "unit;verification"
    )
endif()

add_test(NAME qa.numeric_canary
    COMMAND $<TARGET_FILE:cxxgpr_numeric_canary>
)
set_tests_properties(qa.numeric_canary PROPERTIES
    LABELS "verification"
)

if(CXXGPR_INSTALL)
    add_test(NAME integration.package_install
        COMMAND "${CMAKE_COMMAND}"
            "-DCXXGPR_BUILD_DIR=${CMAKE_BINARY_DIR}"
            "-DCXXGPR_CONFIG=$<CONFIG>"
            "-DCXXGPR_GENERATOR=${CMAKE_GENERATOR}"
            "-DCXXGPR_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DCXXGPR_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
            "-DCXXGPR_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
            "-DCXXGPR_VCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}"
            "-DCXXGPR_VCPKG_INSTALLED_DIR=${VCPKG_INSTALLED_DIR}"
            -P "${CMAKE_SOURCE_DIR}/cmake/VerifyPackage.cmake"
    )
    set_tests_properties(integration.package_install PROPERTIES
        LABELS "integration"
    )
endif()
