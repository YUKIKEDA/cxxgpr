# SPDX-License-Identifier: Apache-2.0

include(FetchContent)

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

set(CXXGPR_BLAS_BACKEND "AUTO" CACHE STRING "Dense BLAS backend (AUTO, OpenBLAS, MKL, BuiltIn)")
set_property(CACHE CXXGPR_BLAS_BACKEND PROPERTY STRINGS "AUTO" "OpenBLAS" "MKL" "BuiltIn")
if(NOT CXXGPR_BLAS_BACKEND MATCHES "^(AUTO|OpenBLAS|MKL|BuiltIn)$")
    message(FATAL_ERROR "Unsupported CXXGPR_BLAS_BACKEND='${CXXGPR_BLAS_BACKEND}'. Choose AUTO, OpenBLAS, MKL, or BuiltIn.")
endif()

set(CXXGPR_ACTIVE_BLAS_BACKEND "BuiltIn")
set(CXXGPR_BLAS_TARGET "")
set(CXXGPR_MKL_ROOT_HINT "")
set(CXXGPR_MKL_DLL_DIR "")

function(_cxxgpr_try_find_mkl required_mode)
    if(NOT DEFINED MKL_ROOT AND DEFINED ENV{MKLROOT} AND NOT "$ENV{MKLROOT}" STREQUAL "")
        set(MKL_ROOT "$ENV{MKLROOT}" CACHE PATH "Intel oneMKL root directory")
    elseif(NOT DEFINED MKL_ROOT AND WIN32 AND EXISTS "C:/vcpkg/packages/intel-mkl_x64-windows")
        set(MKL_ROOT "C:/vcpkg/packages/intel-mkl_x64-windows" CACHE PATH "Intel oneMKL root directory")
    elseif(NOT DEFINED MKL_ROOT AND EXISTS "/opt/intel/oneapi/mkl/latest")
        set(MKL_ROOT "/opt/intel/oneapi/mkl/latest" CACHE PATH "Intel oneMKL root directory")
    endif()
    if(UNIX AND NOT APPLE AND NOT DEFINED MKL_THREADING)
        set(MKL_THREADING "gnu_thread" CACHE STRING "MKL threading layer")
    endif()
    if(required_mode)
        find_package(MKL CONFIG REQUIRED)
    else()
        find_package(MKL CONFIG QUIET)
    endif()
    if(MKL_FOUND AND TARGET MKL::MKL)
        set(CXXGPR_ACTIVE_BLAS_BACKEND "MKL" PARENT_SCOPE)
        set(CXXGPR_BLAS_TARGET "MKL::MKL" PARENT_SCOPE)
        if(DEFINED MKL_ROOT AND NOT "${MKL_ROOT}" STREQUAL "")
            set(CXXGPR_MKL_ROOT_HINT "${MKL_ROOT}" PARENT_SCOPE)
            if(WIN32 AND EXISTS "${MKL_ROOT}/bin")
                set(CXXGPR_MKL_DLL_DIR "${MKL_ROOT}/bin" PARENT_SCOPE)
            endif()
        endif()
    endif()
endfunction()

if(CXXGPR_BLAS_BACKEND STREQUAL "OpenBLAS")
    if(MSVC)
        message(WARNING
            "OpenBLAS on Windows MSVC uses generic C kernels and may cause severe thread contention; prefer BuiltIn or MKL.")
    endif()
    find_package(OpenBLAS CONFIG REQUIRED)
    if(TARGET OpenBLAS::OpenBLAS)
        set(CXXGPR_BLAS_TARGET "OpenBLAS::OpenBLAS")
    elseif(TARGET openblas)
        set(CXXGPR_BLAS_TARGET "openblas")
    else()
        message(FATAL_ERROR "OpenBLAS package found without an OpenBLAS target")
    endif()
    set(CXXGPR_ACTIVE_BLAS_BACKEND "OpenBLAS")
elseif(CXXGPR_BLAS_BACKEND STREQUAL "MKL")
    _cxxgpr_try_find_mkl(TRUE)
elseif(CXXGPR_BLAS_BACKEND STREQUAL "BuiltIn")
    set(CXXGPR_ACTIVE_BLAS_BACKEND "BuiltIn")
else()
    if(MSVC)
        set(CXXGPR_ACTIVE_BLAS_BACKEND "BuiltIn")
    else()
        find_package(OpenBLAS CONFIG QUIET)
        if(OpenBLAS_FOUND AND TARGET OpenBLAS::OpenBLAS)
            set(CXXGPR_ACTIVE_BLAS_BACKEND "OpenBLAS")
            set(CXXGPR_BLAS_TARGET "OpenBLAS::OpenBLAS")
        elseif(OpenBLAS_FOUND AND TARGET openblas)
            set(CXXGPR_ACTIVE_BLAS_BACKEND "OpenBLAS")
            set(CXXGPR_BLAS_TARGET "openblas")
        else()
            _cxxgpr_try_find_mkl(FALSE)
        endif()
    endif()
endif()

message(STATUS "cxxgpr BLAS backend: ${CXXGPR_ACTIVE_BLAS_BACKEND}")

set(CXXGPR_FMATH_GIT_COMMIT "a81f8e53e9f007ed5df355bb4a99ff799fd75066")
FetchContent_Declare(
    fmath
    GIT_REPOSITORY https://github.com/herumi/fmath.git
    GIT_TAG        ${CXXGPR_FMATH_GIT_COMMIT}
    GIT_SHALLOW    FALSE
)
FetchContent_MakeAvailable(fmath)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64)$")
    if(MSVC)
        enable_language(ASM_MASM)
        set(_cxxgpr_fmath_asm "${fmath_SOURCE_DIR}/src/fmath.asm")
    else()
        enable_language(ASM)
        set(_cxxgpr_fmath_asm "${fmath_SOURCE_DIR}/src/fmath.S")
    endif()
    add_library(cxxgpr_fmath OBJECT
        "${fmath_SOURCE_DIR}/src/cpu.cpp"
        "${_cxxgpr_fmath_asm}"
    )
    set(CXXGPR_HAS_FMATH_ASM 1)
else()
    add_library(cxxgpr_fmath OBJECT
        "${fmath_SOURCE_DIR}/src/cpu.cpp"
    )
    set(CXXGPR_HAS_FMATH_ASM 0)
endif()

target_include_directories(cxxgpr_fmath SYSTEM PRIVATE
    "${fmath_SOURCE_DIR}/include"
    "${fmath_SOURCE_DIR}/src"
)
set_target_properties(cxxgpr_fmath PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)
target_compile_definitions(cxxgpr_fmath PRIVATE
    $<$<BOOL:${MSVC}>:_CRT_SECURE_NO_WARNINGS>
)
target_compile_options(cxxgpr_fmath PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${MSVC}>>:/W0>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<BOOL:${MSVC}>>>:-w>
)

if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
    set(CXXGPR_THIRD_PARTY_SHARE "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share")
    set(CXXGPR_THIRD_PARTY_NOTICES "${CMAKE_BINARY_DIR}/third-party-notices")
    file(MAKE_DIRECTORY "${CXXGPR_THIRD_PARTY_NOTICES}")

    set(_cxxgpr_required_notices eigen3 nlohmann-json)
    if(CXXGPR_ACTIVE_BLAS_BACKEND STREQUAL "OpenBLAS"
       OR EXISTS "${CXXGPR_THIRD_PARTY_SHARE}/openblas/copyright")
        list(APPEND _cxxgpr_required_notices openblas)
    endif()
    if(BUILD_TESTING AND CXXGPR_BUILD_TESTS)
        list(APPEND _cxxgpr_required_notices gtest)
    endif()
    if(CXXGPR_BUILD_BENCHMARKS)
        list(APPEND _cxxgpr_required_notices benchmark)
    endif()

    file(GLOB _cxxgpr_share_notices "${CXXGPR_THIRD_PARTY_SHARE}/*/copyright")
    foreach(_notice IN LISTS _cxxgpr_share_notices)
        get_filename_component(_pkg_dir "${_notice}" DIRECTORY)
        get_filename_component(_pkg "${_pkg_dir}" NAME)
        if(_pkg STREQUAL "gtest" AND NOT (BUILD_TESTING AND CXXGPR_BUILD_TESTS))
            continue()
        endif()
        if(_pkg STREQUAL "benchmark" AND NOT CXXGPR_BUILD_BENCHMARKS)
            continue()
        endif()
        configure_file("${_notice}" "${CXXGPR_THIRD_PARTY_NOTICES}/${_pkg}.txt" COPYONLY)
    endforeach()

    foreach(_pkg IN LISTS _cxxgpr_required_notices)
        if(NOT EXISTS "${CXXGPR_THIRD_PARTY_NOTICES}/${_pkg}.txt")
            message(FATAL_ERROR
                "Missing third-party copyright notice for required package '${_pkg}' under ${CXXGPR_THIRD_PARTY_SHARE}")
        endif()
    endforeach()

    set(CXXGPR_FMATH_NOTICE_ORIG "${fmath_SOURCE_DIR}/readme.md")
    configure_file("${CXXGPR_FMATH_NOTICE_ORIG}" "${CXXGPR_THIRD_PARTY_NOTICES}/fmath.txt" COPYONLY)

    set(CXXGPR_MKL_NOTICE_ORIG "")
    if(CXXGPR_ACTIVE_BLAS_BACKEND STREQUAL "MKL" AND NOT "${CXXGPR_MKL_ROOT_HINT}" STREQUAL "")
        if(EXISTS "${CXXGPR_MKL_ROOT_HINT}/share/intel-mkl/copyright")
            set(CXXGPR_MKL_NOTICE_ORIG "${CXXGPR_MKL_ROOT_HINT}/share/intel-mkl/copyright")
            configure_file("${CXXGPR_MKL_NOTICE_ORIG}" "${CXXGPR_THIRD_PARTY_NOTICES}/intel-mkl.txt" COPYONLY)
        endif()
    endif()
endif()
