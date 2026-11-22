// SPDX-License-Identifier: Apache-2.0
/// @file allocation_audit.hpp
/// @brief Thread-local heap allocation auditing utilities for zero-allocation hot-path tests.
/// @details Provides a thread-local allocation counter backed by global operator new overrides and an
/// RAII scope guard that tracks C++ heap allocations and optionally forbids Eigen runtime malloc.

#pragma once

#include <cstddef>
#include <source_location>

/// @brief Test support utilities for allocation auditing and compiler dead-store barriers.
namespace cxxgpr::test_support
{

    /// @brief Selects which heap allocation mechanisms are audited within NoAllocationScope.
    /// @details Controls whether Eigen's runtime malloc flag is disabled alongside C++ operator new tracking.
    enum class AuditMode
    {
        /// @brief Tracks global C++ operator new allocations and forbids Eigen runtime malloc.
        CppAndEigen,
        /// @brief Tracks global C++ operator new allocations without modifying Eigen runtime malloc state.
        CppOnly,
    };

    /// @brief Controls whether NoAllocationScope asserts zero heap allocations on scope exit.
    enum class AuditExpectation
    {
        /// @brief Reports a GoogleTest non-fatal failure on destruction if delta_allocations() != 0.
        AssertZeroOnExit,
        /// @brief Leaves allocation count verification to explicit caller assertions (for negative tests).
        Manual,
    };

    /// @brief Returns the thread-local cumulative count of tracked C++ heap allocations.
    /// @details Counts every invocation of the overridden global operator new functions on the calling thread.
    /// @return Cumulative number of tracked C++ heap allocations on the current thread.
    [[nodiscard]] std::size_t thread_allocation_count() noexcept;

    /// @brief Observes a pointer via a volatile sink to prevent dead-store elimination in tests.
    /// @details Forces the compiler to materialize computed buffers before measuring allocation deltas.
    /// @param[in] ptr Pointer to test data whose computation must not be eliminated as dead code.
    void touch_for_test(const void* ptr) noexcept;

    /// @brief RAII scope tracking C++ heap allocations and optionally forbidding Eigen runtime malloc.
    /// @details Snapshots the thread-local allocation counter on entry and optionally disables Eigen's
    /// internal runtime malloc flag until destruction restores the previous state.
    class NoAllocationScope final
    {
    public:
        /// @brief Enters an allocation audit scope and snapshots the current thread allocation count.
        /// @details When mode is AuditMode::CppAndEigen, disables Eigen runtime malloc until destruction.
        /// @param[in] mode Allocation auditing mode selecting C++ and Eigen checks or C++ tracking only.
        /// @param[in] expectation Whether destruction asserts zero tracked allocations.
        /// @param[in] loc Source location reported if AssertZeroOnExit detects allocations.
        explicit NoAllocationScope(
            AuditMode mode = AuditMode::CppAndEigen,
            AuditExpectation expectation = AuditExpectation::AssertZeroOnExit,
            std::source_location loc = std::source_location::current()
        ) noexcept;

        /// @brief Exits the allocation audit scope, checks expectations, and restores Eigen malloc state.
        /// @details Restores Eigen::internal::is_malloc_allowed() if AuditMode::CppAndEigen was active.
        ~NoAllocationScope() noexcept;

        /// @brief Deleted copy constructor because an RAII audit guard owns a scoped state transition.
        /// @details Copying a scope guard would corrupt Eigen runtime malloc restoration on destruction.
        NoAllocationScope(const NoAllocationScope&) = delete;

        /// @brief Deleted copy assignment operator to preserve single-scope lifetime semantics.
        /// @details Scope guards cannot be copy-assigned after entering an audit region.
        /// @return Reference to *this (deleted).
        NoAllocationScope& operator=(const NoAllocationScope&) = delete;

        /// @brief Deleted move constructor because an RAII audit guard cannot transfer scoped ownership.
        /// @details Prevents out-of-order destruction across nested audit scopes.
        NoAllocationScope(NoAllocationScope&&) = delete;

        /// @brief Deleted move assignment operator to preserve single-scope lifetime semantics.
        /// @details Scope guards cannot be move-assigned after entering an audit region.
        /// @return Reference to *this (deleted).
        NoAllocationScope& operator=(NoAllocationScope&&) = delete;

        /// @brief Returns the number of tracked C++ heap allocations since entering this scope.
        /// @details Computes the difference between the current thread allocation count and start_count_.
        /// @return Tracked C++ heap allocations executed on the calling thread since construction.
        [[nodiscard]] std::size_t delta_allocations() const noexcept;

    private:
        /// @brief Thread-local allocation counter snapshot captured at scope entry.
        std::size_t start_count_;
        /// @brief Auditing mode active for the lifetime of this scope guard.
        AuditMode mode_;
        /// @brief Scope-exit assertion policy.
        AuditExpectation expectation_;
        /// @brief Saved Eigen::internal::is_malloc_allowed() flag restored on scope exit.
        bool prev_eigen_allowed_;
        /// @brief Caller source location used for scope-exit failure diagnostics.
        std::source_location loc_;
    };

}  // namespace cxxgpr::test_support
