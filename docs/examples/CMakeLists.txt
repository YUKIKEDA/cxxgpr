# SPDX-License-Identifier: Apache-2.0
# Example programs included verbatim by the user documentation (docs/en, docs/ja).
#
# Two ways to build:
#   1. In-tree:     cmake ... -DCXXGPR_BUILD_EXAMPLES=ON   (top-level project, registers CTest tests)
#   2. Stand-alone: cmake -S docs/examples -B <dir> -DCMAKE_PREFIX_PATH=<install prefix>
#      (consumes the installed package exactly like a user project would)
if(NOT PROJECT_NAME STREQUAL "cxxgpr")
    cmake_minimum_required(VERSION 3.25)
    project(cxxgpr_examples LANGUAGES CXX)
    include(CTest)
    find_package(cxxgpr CONFIG REQUIRED)
    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# Every docs/examples/*.cpp becomes one executable and one CTest test that must exit with status 0.
file(GLOB CXXGPR_EXAMPLE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
foreach(_src IN LISTS CXXGPR_EXAMPLE_SOURCES)
    get_filename_component(_name "${_src}" NAME_WE)
    add_executable(cxxgpr_example_${_name} "${_src}")
    set_target_properties(cxxgpr_example_${_name} PROPERTIES OUTPUT_NAME "${_name}")
    target_compile_features(cxxgpr_example_${_name} PRIVATE cxx_std_23)
    target_link_libraries(cxxgpr_example_${_name} PRIVATE cxxgpr::cxxgpr)
    if(MSVC)
        target_compile_options(cxxgpr_example_${_name} PRIVATE /W4 /WX /permissive- /utf-8 /EHsc)
    else()
        target_compile_options(cxxgpr_example_${_name} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME example.${_name} COMMAND $<TARGET_FILE:cxxgpr_example_${_name}>)
    set_tests_properties(example.${_name} PROPERTIES LABELS "integration" TIMEOUT 120
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endforeach()
