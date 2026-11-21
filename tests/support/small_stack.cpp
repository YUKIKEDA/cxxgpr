// SPDX-License-Identifier: Apache-2.0
/// @file small_stack.cpp
/// @brief Thread launcher with an explicit stack size for stack-depth tests.
/// @details Implements run_with_stack_size with the native thread API of each platform, since std::thread offers no
/// way to choose the stack size.

#include "small_stack.hpp"

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <pthread.h>
#endif

/// @brief Test support utilities for allocation auditing and compiler dead-store barriers.
namespace cxxgpr::test_support
{
    namespace
    {

#ifdef _WIN32
        /// @brief Thread entry point that runs the std::function passed as its argument.
        /// @param[in] arg Pointer to the std::function<void()> to run.
        /// @return Always 0.
        DWORD WINAPI run_body(LPVOID arg)
        {
            (*static_cast<std::function<void()>*>(arg))();
            return 0;
        }
#else
        /// @brief Thread entry point that runs the std::function passed as its argument.
        /// @param[in] arg Pointer to the std::function<void()> to run.
        /// @return Always a null pointer.
        void* run_body(void* arg)
        {
            (*static_cast<std::function<void()>*>(arg))();
            return nullptr;
        }
#endif

    }  // namespace

    /// @brief Runs body to completion on a new thread whose whole stack reservation is stack_bytes.
    /// @param[in] stack_bytes Stack size of the thread in bytes; the platform may round it up to its granularity.
    /// @param[in] body Work to run on the thread. It must not throw.
    /// @return True when the thread ran body to completion, false when the platform refused to create the thread.
    bool run_with_stack_size(std::size_t stack_bytes, const std::function<void()>& body)
    {
        // The thread receives a pointer to this copy, which outlives it because the thread is joined below.
        std::function<void()> task = body;
#ifdef _WIN32
        // STACK_SIZE_PARAM_IS_A_RESERVATION makes stack_bytes the reserved size; without it the size only sets
        // the initial commit and the reservation stays at the executable's default of 1 MiB.
        HANDLE thread = CreateThread(nullptr, stack_bytes, run_body, &task, STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
        if (thread == nullptr)
        {
            return false;
        }
        const bool joined = WaitForSingleObject(thread, INFINITE) == WAIT_OBJECT_0;
        CloseHandle(thread);
        return joined;
#else
        pthread_attr_t attr;
        if (pthread_attr_init(&attr) != 0)
        {
            return false;
        }
        pthread_t thread{};
        const bool created =
            pthread_attr_setstacksize(&attr, stack_bytes) == 0 && pthread_create(&thread, &attr, run_body, &task) == 0;
        pthread_attr_destroy(&attr);
        return created && pthread_join(thread, nullptr) == 0;
#endif
    }

}  // namespace cxxgpr::test_support
