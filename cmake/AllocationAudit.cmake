# SPDX-License-Identifier: Apache-2.0
function(cxxgpr_configure_allocation_audit_target target)
    target_compile_definitions(${target} PRIVATE EIGEN_RUNTIME_NO_MALLOC)
    if(MSVC)
        target_compile_options(${target} PRIVATE /UEIGEN_NO_DEBUG /GL-)
    else()
        target_compile_options(${target} PRIVATE -UEIGEN_NO_DEBUG -fno-lto)
    endif()
endfunction()
