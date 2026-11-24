# SPDX-License-Identifier: Apache-2.0
foreach(_var IN ITEMS CXXGPR_BUILD_DIR CXXGPR_CONFIG CXXGPR_GENERATOR)
    if(NOT DEFINED "${_var}" OR "${${_var}}" STREQUAL "")
        message(FATAL_ERROR "VerifyPackage.cmake requires ${_var}")
    endif()
endforeach()

set(_pkg_root "${CXXGPR_BUILD_DIR}/_pkg_verify/${CXXGPR_CONFIG}")
set(_install_dir "${_pkg_root}/install")
set(_consumer_src "${_pkg_root}/consumer_src")
set(_consumer_bld "${_pkg_root}/consumer_bld")

file(REMOVE_RECURSE "${_pkg_root}")
file(MAKE_DIRECTORY "${_consumer_src}")

execute_process(
    COMMAND "${CMAKE_COMMAND}"
        --install "${CXXGPR_BUILD_DIR}"
        --config "${CXXGPR_CONFIG}"
        --prefix "${_install_dir}"
    RESULT_VARIABLE _install_rc
)
if(NOT _install_rc EQUAL 0)
    message(FATAL_ERROR "cmake --install failed with ${_install_rc}")
endif()

if(NOT EXISTS "${_install_dir}/include/cxxgpr/gpr/gpr.hpp")
    message(FATAL_ERROR "Missing installed header cxxgpr/gpr/gpr.hpp")
endif()
if(NOT EXISTS "${_install_dir}/share/cxxgpr/LICENSE")
    message(FATAL_ERROR "Missing installed LICENSE")
endif()
if(NOT EXISTS "${_install_dir}/share/cxxgpr/THIRD_PARTY_NOTICES.md")
    message(FATAL_ERROR "Missing installed THIRD_PARTY_NOTICES.md")
endif()
file(GLOB _config_files
    "${_install_dir}/lib/cmake/cxxgpr/cxxgprConfig.cmake"
    "${_install_dir}/lib64/cmake/cxxgpr/cxxgprConfig.cmake"
)
if(NOT _config_files)
    message(FATAL_ERROR "Missing installed cxxgprConfig.cmake under ${_install_dir}")
endif()

file(WRITE "${_consumer_src}/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.25)
project(cxxgpr_consumer_verify LANGUAGES CXX)
if(MSVC AND NOT DEFINED CMAKE_CXX23_STANDARD_COMPILE_OPTION)
    set(CMAKE_CXX23_STANDARD_COMPILE_OPTION "/std:c++23preview")
    set(CMAKE_CXX23_EXTENSION_COMPILE_OPTION "/std:c++23preview")
endif()
find_package(cxxgpr 0.1.0 CONFIG REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE cxxgpr::cxxgpr)
]=])

file(WRITE "${_consumer_src}/main.cpp" [=[
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
#include <array>
#include <cmath>
#include <iostream>

int main() {
    const std::array<double, 4> x_train{0.0, 1.0, 2.0, 3.0};
    const std::array<double, 4> y_train{0.0, 1.0, 0.5, 1.2};
    const std::array<double, 2> x_test{0.5, 1.5};

    const auto x_ref = cxxgpr::MatrixRef<double>(x_train.data(), 4U, 1U, 4U);
    const auto xq_ref = cxxgpr::MatrixRef<double>(x_test.data(), 2U, 1U, 2U);
    const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());

    auto trainer_res = cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs>(spec).with_noise_variance(1e-4);
    if (!trainer_res) {
        std::cerr << "trainer config failed: " << trainer_res.error().message() << '\n';
        return 1;
    }

    auto fitted = trainer_res->fit(x_ref, y_train);
    if (!fitted) {
        std::cerr << "fit failed: " << fitted.error().message() << '\n';
        return 2;
    }

    auto pred = fitted->predict(xq_ref);
    if (!pred) {
        std::cerr << "predict failed: " << pred.error().message() << '\n';
        return 3;
    }
    if (pred->mean.size() != 2 || !std::isfinite(pred->mean[0]) || !std::isfinite(pred->mean[1])) {
        std::cerr << "invalid prediction output\n";
        return 4;
    }
    return 0;
}
]=])

set(_cfg_args
    -S "${_consumer_src}"
    -B "${_consumer_bld}"
    -G "${CXXGPR_GENERATOR}"
    "-DCMAKE_PREFIX_PATH=${_install_dir}"
    "-DCMAKE_BUILD_TYPE=${CXXGPR_CONFIG}"
    "-DCMAKE_CONFIGURATION_TYPES=${CXXGPR_CONFIG}"
)
if(DEFINED CXXGPR_CXX_COMPILER AND NOT "${CXXGPR_CXX_COMPILER}" STREQUAL "")
    list(APPEND _cfg_args "-DCMAKE_CXX_COMPILER=${CXXGPR_CXX_COMPILER}")
endif()
if(DEFINED CXXGPR_CXX_FLAGS AND NOT "${CXXGPR_CXX_FLAGS}" STREQUAL "")
    list(APPEND _cfg_args "-DCMAKE_CXX_FLAGS=${CXXGPR_CXX_FLAGS}")
endif()
if(DEFINED CXXGPR_TOOLCHAIN_FILE AND NOT "${CXXGPR_TOOLCHAIN_FILE}" STREQUAL "")
    list(APPEND _cfg_args
        "-DCMAKE_TOOLCHAIN_FILE=${CXXGPR_TOOLCHAIN_FILE}"
        "-DVCPKG_MANIFEST_MODE=OFF"
    )
    if(DEFINED CXXGPR_VCPKG_INSTALLED_DIR AND NOT "${CXXGPR_VCPKG_INSTALLED_DIR}" STREQUAL "")
        list(APPEND _cfg_args "-DVCPKG_INSTALLED_DIR=${CXXGPR_VCPKG_INSTALLED_DIR}")
    endif()
    if(DEFINED CXXGPR_VCPKG_TARGET_TRIPLET AND NOT "${CXXGPR_VCPKG_TARGET_TRIPLET}" STREQUAL "")
        list(APPEND _cfg_args "-DVCPKG_TARGET_TRIPLET=${CXXGPR_VCPKG_TARGET_TRIPLET}")
    endif()
elseif(DEFINED CXXGPR_VCPKG_INSTALLED_DIR AND NOT "${CXXGPR_VCPKG_INSTALLED_DIR}" STREQUAL ""
       AND DEFINED CXXGPR_VCPKG_TARGET_TRIPLET AND NOT "${CXXGPR_VCPKG_TARGET_TRIPLET}" STREQUAL "")
    list(APPEND _cfg_args "-DCMAKE_FIND_ROOT_PATH=${CXXGPR_VCPKG_INSTALLED_DIR}/${CXXGPR_VCPKG_TARGET_TRIPLET}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" ${_cfg_args}
    RESULT_VARIABLE _cfg_rc
)
if(NOT _cfg_rc EQUAL 0)
    message(FATAL_ERROR "Consumer configure failed with ${_cfg_rc}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${_consumer_bld}" --config "${CXXGPR_CONFIG}"
    RESULT_VARIABLE _bld_rc
)
if(NOT _bld_rc EQUAL 0)
    message(FATAL_ERROR "Consumer build failed with ${_bld_rc}")
endif()

set(_exe "${_consumer_bld}/consumer")
if(WIN32)
    if(EXISTS "${_consumer_bld}/${CXXGPR_CONFIG}/consumer.exe")
        set(_exe "${_consumer_bld}/${CXXGPR_CONFIG}/consumer.exe")
    else()
        set(_exe "${_consumer_bld}/consumer.exe")
    endif()
endif()

if(WIN32 AND DEFINED CXXGPR_MKL_DLL_DIR AND NOT "${CXXGPR_MKL_DLL_DIR}" STREQUAL "")
    set(ENV{PATH} "${CXXGPR_MKL_DLL_DIR};$ENV{PATH}")
endif()

execute_process(
    COMMAND "${_exe}"
    RESULT_VARIABLE _run_rc
)
if(NOT _run_rc EQUAL 0)
    message(FATAL_ERROR "Consumer run failed with ${_run_rc}")
endif()
