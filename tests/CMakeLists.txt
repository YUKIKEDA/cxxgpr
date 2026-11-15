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

add_executable(cxxgpr_test_core_build_info
    core/build_info_test.cpp
)
cxxgpr_configure_target(cxxgpr_test_core_build_info)
cxxgpr_configure_allocation_audit_target(cxxgpr_test_core_build_info)
target_compile_definitions(cxxgpr_test_core_build_info PRIVATE
    CXXGPR_MANIFEST_PATH="${CMAKE_SOURCE_DIR}/vcpkg.json"
)
target_link_libraries(cxxgpr_test_core_build_info PRIVATE
    GTest::gtest_main
    cxxgpr::cxxgpr
    cxxgpr_test_support
    nlohmann_json::nlohmann_json
)
add_test(NAME core.build_info
    COMMAND $<TARGET_FILE:cxxgpr_test_core_build_info>
)

if(DEFINED CXXGPR_THIRD_PARTY_NOTICES)
    add_test(NAME toolchain.third_party_notices
        COMMAND "${CMAKE_COMMAND}"
            "-DNOTICES_DIR=${CXXGPR_THIRD_PARTY_NOTICES}"
            "-DROOT_NOTICE=${CMAKE_SOURCE_DIR}/THIRD_PARTY_NOTICES.md"
            "-DTOP_BUILD_TESTING=${BUILD_TESTING}"
            "-DTOP_BUILD_BENCHMARKS=${CXXGPR_BUILD_BENCHMARKS}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/toolchain/check_notices.cmake"
    )
endif()

add_test(NAME qa.numeric_canary
    COMMAND $<TARGET_FILE:cxxgpr_numeric_canary>
)
