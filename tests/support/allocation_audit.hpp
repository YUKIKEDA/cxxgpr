// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

namespace cxxgpr::test_support
{

    enum class AuditMode
    {
        CppAndEigen,
        CppOnly,
    };

    /// Returns the thread-local cumulative count of tracked C++ heap allocations.
    [[nodiscard]] std::size_t thread_allocation_count() noexcept;

    /// Observes a pointer via a volatile sink to prevent dead-store elimination in tests.
    void touch_for_test(const void* ptr) noexcept;

    /// RAII scope tracking C++ heap allocations and optionally forbidding Eigen runtime malloc.
    class NoAllocationScope final
    {
    public:
        explicit NoAllocationScope(AuditMode mode = AuditMode::CppAndEigen) noexcept;
        ~NoAllocationScope() noexcept;

        NoAllocationScope(const NoAllocationScope&) = delete;
        NoAllocationScope& operator=(const NoAllocationScope&) = delete;
        NoAllocationScope(NoAllocationScope&&) = delete;
        NoAllocationScope& operator=(NoAllocationScope&&) = delete;

        /// Returns the number of tracked C++ heap allocations since entering this scope.
        [[nodiscard]] std::size_t delta_allocations() const noexcept;

    private:
        std::size_t start_count_;
        AuditMode mode_;
        bool prev_eigen_allowed_;
    };

}  // namespace cxxgpr::test_support
