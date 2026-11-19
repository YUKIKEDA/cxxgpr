# SPDX-License-Identifier: Apache-2.0
function(cxxgpr_configure_target target)
    target_compile_features(${target} PUBLIC cxx_std_23)
    target_compile_definitions(${target} PRIVATE
        CXXGPR_BUILD_CONFIGURATION="$<CONFIG>"
    )
    if(MSVC)
        target_compile_options(${target} PRIVATE
            /W4
            /WX
            /permissive-
            /utf-8
            /EHsc
            /fp:precise
            /bigobj
        )
        if(CXXGPR_ARCH_KEY STREQUAL "x64")
            if(CXXGPR_X86_ISA STREQUAL "AVX512")
                target_compile_options(${target} PRIVATE /arch:AVX512)
            elseif(CXXGPR_X86_ISA STREQUAL "AVX2")
                target_compile_options(${target} PRIVATE /arch:AVX2)
            endif()
        endif()
    else()
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Werror
            -Wconversion
            -Wsign-conversion
            -fno-fast-math
        )
        if(CXXGPR_ARCH_KEY STREQUAL "x64")
            if(CXXGPR_X86_ISA STREQUAL "AVX512")
                target_compile_options(${target} PRIVATE -mavx512f -mavx512dq -mavx512vl -mfma)
            elseif(CXXGPR_X86_ISA STREQUAL "AVX2")
                target_compile_options(${target} PRIVATE -mavx2 -mfma)
            endif()
        endif()
    endif()
endfunction()
