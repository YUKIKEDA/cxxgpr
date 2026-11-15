# SPDX-License-Identifier: Apache-2.0
find_package(Eigen3 5.0 CONFIG REQUIRED)
find_package(nlohmann_json 3.12 CONFIG REQUIRED)
find_package(OpenMP COMPONENTS CXX)

add_library(cxxgpr_eigen INTERFACE)
add_library(cxxgpr::eigen ALIAS cxxgpr_eigen)
target_link_libraries(cxxgpr_eigen INTERFACE Eigen3::Eigen)
target_compile_definitions(cxxgpr_eigen INTERFACE
    EIGEN_MPL2_ONLY
    EIGEN_DONT_PARALLELIZE
)

if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
    set(CXXGPR_THIRD_PARTY_SHARE "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share")
    set(CXXGPR_THIRD_PARTY_NOTICES "${CMAKE_BINARY_DIR}/third-party-notices")
    file(REMOVE_RECURSE "${CXXGPR_THIRD_PARTY_NOTICES}")
    file(MAKE_DIRECTORY "${CXXGPR_THIRD_PARTY_NOTICES}")

    set(_cxxgpr_required_notices eigen3 nlohmann-json)
    if(BUILD_TESTING)
        list(APPEND _cxxgpr_required_notices gtest)
    endif()
    if(CXXGPR_BUILD_BENCHMARKS)
        list(APPEND _cxxgpr_required_notices benchmark)
    endif()

    file(GLOB _cxxgpr_share_notices "${CXXGPR_THIRD_PARTY_SHARE}/*/copyright")
    foreach(_notice IN LISTS _cxxgpr_share_notices)
        get_filename_component(_pkg_dir "${_notice}" DIRECTORY)
        get_filename_component(_pkg "${_pkg_dir}" NAME)
        if(_pkg STREQUAL "gtest" AND NOT BUILD_TESTING)
            continue()
        endif()
        if(_pkg STREQUAL "benchmark" AND NOT CXXGPR_BUILD_BENCHMARKS)
            continue()
        endif()
        file(COPY_FILE "${_notice}" "${CXXGPR_THIRD_PARTY_NOTICES}/${_pkg}.txt")
    endforeach()

    foreach(_pkg IN LISTS _cxxgpr_required_notices)
        if(NOT EXISTS "${CXXGPR_THIRD_PARTY_NOTICES}/${_pkg}.txt")
            message(FATAL_ERROR
                "Missing third-party copyright notice for required package '${_pkg}' under ${CXXGPR_THIRD_PARTY_SHARE}")
        endif()
    endforeach()
endif()
