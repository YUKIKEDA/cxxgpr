# SPDX-License-Identifier: Apache-2.0
include(CheckCXXSourceCompiles)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    if(MSVC_VERSION LESS 1944)
        message(FATAL_ERROR
            "MSVC 19.44+ (Visual Studio 17.14+) is required for C++23 std::expected support; found ${MSVC_VERSION}")
    endif()
    set(CMAKE_CXX23_STANDARD_COMPILE_OPTION "/std:c++23preview")
    set(CMAKE_CXX23_EXTENSION_COMPILE_OPTION "/std:c++23preview")
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")
    set(CXXGPR_FLOATING_POINT "strict (/fp:strict)")
else()
    set(CXXGPR_FLOATING_POINT "strict (-ffp-contract=off -fno-fast-math)")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(CXXGPR_OS_KEY "windows")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CXXGPR_OS_KEY "linux")
else()
    string(TOLOWER "${CMAKE_SYSTEM_NAME}" CXXGPR_OS_KEY)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|x86_64|amd64)$")
    set(CXXGPR_ARCH_KEY "x64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|aarch64|arm64)$")
    set(CXXGPR_ARCH_KEY "arm64")
else()
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" CXXGPR_ARCH_KEY)
endif()

string(REPLACE "." ";" _cxxgpr_ver_parts "${CMAKE_CXX_COMPILER_VERSION}")
list(GET _cxxgpr_ver_parts 0 _cxxgpr_ver_major)
list(GET _cxxgpr_ver_parts 1 _cxxgpr_ver_minor)
if(MSVC)
    set(CXXGPR_COMPILER_KEY "msvc${_cxxgpr_ver_major}.${_cxxgpr_ver_minor}")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CXXGPR_COMPILER_KEY "gnu${_cxxgpr_ver_major}.${_cxxgpr_ver_minor}")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CXXGPR_COMPILER_KEY "clang${_cxxgpr_ver_major}.${_cxxgpr_ver_minor}")
else()
    string(TOLOWER "${CMAKE_CXX_COMPILER_ID}${_cxxgpr_ver_major}.${_cxxgpr_ver_minor}" CXXGPR_COMPILER_KEY)
endif()

set(_cxxgpr_saved_flags "${CMAKE_REQUIRED_FLAGS}")
if(MSVC)
    set(CMAKE_REQUIRED_FLAGS "/std:c++23preview /EHsc")
else()
    set(CMAKE_REQUIRED_FLAGS "-std=c++23")
endif()
check_cxx_source_compiles("
#include <expected>
static_assert(__cpp_lib_expected >= 202211L);
int main() {
    std::expected<int, int> val(21);
    auto doubled = val.transform([](int x) { return x * 2; });
    return doubled.value_or(0) == 42 ? 0 : 1;
}
" CXXGPR_HAS_STD_EXPECTED)
set(CMAKE_REQUIRED_FLAGS "${_cxxgpr_saved_flags}")

if(NOT CXXGPR_HAS_STD_EXPECTED)
    message(FATAL_ERROR "C++23 std::expected with monadic transform is required")
endif()
