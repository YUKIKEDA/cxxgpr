# SPDX-License-Identifier: Apache-2.0
find_package(GTest CONFIG REQUIRED)

add_library(cxxgpr_test_support STATIC
    support/allocation_audit.cpp
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
    LABELS unit property
)

cxxgpr_add_test(core.rng
    SOURCES core/rng_test.cpp
    LABELS unit property
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
