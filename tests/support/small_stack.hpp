// SPDX-License-Identifier: Apache-2.0
/// @file small_stack.hpp
/// @brief Runs a test body on a thread whose stack is deliberately small.
/// @details Lets a test bound the stack that a deep recursion needs on every platform, independently of the main
/// thread stack size set by the operating system or the shell.

#pragma once

#include <cstddef>
#include <functional>

/// @brief Test support utilities for allocation auditing and compiler dead-store barriers.
namespace cxxgpr::test_support
{

    /// @brief Stack budget, in bytes, of the deep-model work without AddressSanitizer.
    /// @return 1 MiB with MSVC, the stack that Windows reserves for a thread by default, and 128 KiB elsewhere.
    [[nodiscard]] constexpr std::size_t uninstrumented_deep_model_stack_bytes() noexcept
    {
#if defined(_MSC_VER)
        return std::size_t{1024} * 1024U;
#else
        return std::size_t{128} * 1024U;
#endif
    }

    /// @brief Reports whether the build instruments the stack with AddressSanitizer redzones.
    /// @return True when AddressSanitizer is enabled.
    [[nodiscard]] constexpr bool address_sanitizer_enabled() noexcept
    {
#if defined(__SANITIZE_ADDRESS__)
        return true;
#elif defined(__has_feature)
#    if __has_feature(address_sanitizer)
        return true;
#    else
        return false;
#    endif
#else
        return false;
#endif
    }

    /// @brief Stack budget, in bytes, for saving and loading the deepest models that config.json accepts.
    /// @details Each value is about twice what that work needs once every recursion level keeps only its children
    /// on the stack, rounded up to a power of two, while keeping temporaries on every level needs several times
    /// more. Measured with GCC 14: 69 KiB unoptimized and 63 KiB optimized against 128 KiB, and 290 KiB with
    /// AddressSanitizer, whose redzones enlarge every frame, against 1 MiB. MSVC could not be measured here, and a
    /// 256 KiB budget derived from Clang without optimization (91 KiB) proved too small for MSVC Debug in CI, so
    /// MSVC is held to the requirement itself: the deepest accepted models fit in the 1 MiB default stack of a
    /// Windows thread. The code before the per-level stack reduction exceeded that stack in MSVC Debug CI.
    /// @return 1 MiB under AddressSanitizer, otherwise uninstrumented_deep_model_stack_bytes().
    [[nodiscard]] constexpr std::size_t deep_model_stack_bytes() noexcept
    {
        return address_sanitizer_enabled() ? std::size_t{1024} * 1024U : uninstrumented_deep_model_stack_bytes();
    }

    /// @brief Runs body to completion on a new thread whose whole stack reservation is stack_bytes.
    /// @details Uses pthread_attr_setstacksize on POSIX and CreateThread with STACK_SIZE_PARAM_IS_A_RESERVATION on
    /// Windows, so that the limit is the same kind of bound on both. A body that needs more stack crashes the test
    /// process, which CTest reports as a failure.
    /// @param[in] stack_bytes Stack size of the thread in bytes; the platform may round it up to its granularity.
    /// @param[in] body Work to run on the thread. It must not throw.
    /// @return True when the thread ran body to completion, false when the platform refused to create the thread.
    [[nodiscard]] bool run_with_stack_size(std::size_t stack_bytes, const std::function<void()>& body);

}  // namespace cxxgpr::test_support
