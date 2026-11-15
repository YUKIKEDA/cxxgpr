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
            /fp:strict
        )
    else()
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Werror
            -Wconversion
            -Wsign-conversion
            -ffp-contract=off
            -fno-fast-math
        )
    endif()
endfunction()
