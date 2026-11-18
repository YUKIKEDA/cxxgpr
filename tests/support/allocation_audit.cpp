// SPDX-License-Identifier: Apache-2.0
/// @file allocation_audit.cpp
/// @brief Implementation of global operator new allocation counters and NoAllocationScope guards.
/// @details Overrides all standard aligned and unaligned global operator new/delete variants to increment
/// a thread-local counter without triggering recursive allocations, and toggles Eigen's runtime malloc guard.

#include "allocation_audit.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <cstdlib>
#include <new>

namespace
{

    /// @brief Thread-local counter incremented on each tracked global operator new call.
    thread_local std::size_t g_thread_alloc_count = 0;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

    /// @brief Volatile pointer sink written by touch_for_test() to defeat dead-store elimination.
    volatile const void* g_escape_sink = nullptr;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

    /// @brief Allocates unaligned storage via std::malloc and increments the thread allocation counter.
    /// @details Requests at least 1 byte when size is zero to satisfy C++ unique-pointer allocation rules.
    /// @param size Requested allocation size in bytes.
    /// @return Pointer to allocated storage, or nullptr on allocation failure.
    void* alloc_unaligned(std::size_t size) noexcept
    {
        // Increment before malloc so even failed allocations are visible to audit counters.
        ++g_thread_alloc_count;
        // Clamp zero-byte allocations to 1 byte for portable non-null returns across libc implementations.
        return std::malloc(size == 0 ? 1 : size);
    }

    /// @brief Allocates over-aligned storage via platform aligned malloc and increments the thread counter.
    /// @details Enforces a minimum alignment of sizeof(void*) as required by posix_memalign and _aligned_malloc.
    /// @param size Requested allocation size in bytes.
    /// @param alignment Requested alignment in bytes.
    /// @return Pointer to aligned storage, or nullptr on allocation failure.
    void* alloc_aligned(std::size_t size, std::size_t alignment) noexcept
    {
        ++g_thread_alloc_count;
        // Ensure alignment satisfies platform minimums (>= sizeof(void*)).
        const std::size_t effective_align = alignment < sizeof(void*) ? sizeof(void*) : alignment;
        const std::size_t effective_size = size == 0 ? effective_align : size;
#ifdef _WIN32
        return _aligned_malloc(effective_size, effective_align);
#else
        void* ptr = nullptr;
        if (posix_memalign(&ptr, effective_align, effective_size) != 0)
        {
            return nullptr;
        }
        return ptr;
#endif
    }

    /// @brief Releases storage allocated by alloc_aligned() using the matching platform deallocator.
    /// @details Dispatches to _aligned_free on Windows and std::free on POSIX platforms.
    /// @param ptr Pointer previously returned by alloc_aligned(), or nullptr.
    void free_aligned(void* ptr) noexcept
    {
        // Match _aligned_malloc on Windows because std::free is invalid for _aligned_malloc pointers.
#ifdef _WIN32
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }

}  // namespace

// NOLINTBEGIN(readability-inconsistent-declaration-parameter-name)
/// @brief Replaces global throwing single-object operator new with audited unaligned allocation.
/// @details Increments the thread-local allocation count and throws std::bad_alloc on failure.
/// @param size Number of bytes to allocate.
/// @return Non-null pointer to allocated storage.
void* operator new(std::size_t size)
{
    if (void* ptr = alloc_unaligned(size))
    {
        return ptr;
    }
    throw std::bad_alloc();
}

/// @brief Replaces global throwing array operator new[] by delegating to single-object operator new.
/// @details Shares the same audited allocation counter and bad_alloc behavior.
/// @param size Number of bytes to allocate.
/// @return Non-null pointer to allocated array storage.
void* operator new[](std::size_t size)
{
    return ::operator new(size);
}

/// @brief Replaces global nothrow single-object operator new with audited unaligned allocation.
/// @details Increments the thread-local allocation count and returns nullptr on failure.
/// @param size Number of bytes to allocate.
/// @return Pointer to allocated storage, or nullptr if allocation fails.
void* operator new(std::size_t size, const std::nothrow_t&) noexcept
{
    return alloc_unaligned(size);
}

/// @brief Replaces global nothrow array operator new[] with audited unaligned allocation.
/// @details Increments the thread-local allocation count and returns nullptr on failure.
/// @param size Number of bytes to allocate.
/// @return Pointer to allocated array storage, or nullptr if allocation fails.
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept
{
    return alloc_unaligned(size);
}

/// @brief Replaces global unsized single-object operator delete using std::free.
/// @details Pairs with unaligned operator new overrides backed by std::malloc.
/// @param ptr Pointer to storage to free, or nullptr.
void operator delete(void* ptr) noexcept
{
    std::free(ptr);
}

/// @brief Replaces global unsized array operator delete[] using std::free.
/// @details Pairs with unaligned operator new[] overrides backed by std::malloc.
/// @param ptr Pointer to array storage to free, or nullptr.
void operator delete[](void* ptr) noexcept
{
    std::free(ptr);
}

/// @brief Replaces global sized single-object operator delete using std::free.
/// @details Ignores size because std::malloc tracks block extents internally.
/// @param ptr Pointer to storage to free, or nullptr.
void operator delete(void* ptr, std::size_t) noexcept
{
    std::free(ptr);
}

/// @brief Replaces global sized array operator delete[] using std::free.
/// @details Ignores size because std::malloc tracks block extents internally.
/// @param ptr Pointer to array storage to free, or nullptr.
void operator delete[](void* ptr, std::size_t) noexcept
{
    std::free(ptr);
}

/// @brief Replaces global nothrow single-object operator delete using std::free.
/// @details Pairs with nothrow unaligned operator new on constructor unwinds.
/// @param ptr Pointer to storage to free, or nullptr.
void operator delete(void* ptr, const std::nothrow_t&) noexcept
{
    std::free(ptr);
}

/// @brief Replaces global nothrow array operator delete[] using std::free.
/// @details Pairs with nothrow unaligned operator new[] on constructor unwinds.
/// @param ptr Pointer to array storage to free, or nullptr.
void operator delete[](void* ptr, const std::nothrow_t&) noexcept
{
    std::free(ptr);
}

/// @brief Replaces global throwing aligned single-object operator new with audited aligned allocation.
/// @details Increments the thread-local allocation count and throws std::bad_alloc on failure.
/// @param size Number of bytes to allocate.
/// @param alignment Required alignment in bytes.
/// @return Non-null pointer to aligned storage.
void* operator new(std::size_t size, std::align_val_t alignment)
{
    if (void* ptr = alloc_aligned(size, static_cast<std::size_t>(alignment)))
    {
        return ptr;
    }
    throw std::bad_alloc();
}

/// @brief Replaces global throwing aligned array operator new[] by delegating to aligned operator new.
/// @details Shares the same audited aligned allocation counter and bad_alloc behavior.
/// @param size Number of bytes to allocate.
/// @param alignment Required alignment in bytes.
/// @return Non-null pointer to aligned array storage.
void* operator new[](std::size_t size, std::align_val_t alignment)
{
    return ::operator new(size, alignment);
}

/// @brief Replaces global nothrow aligned single-object operator new with audited aligned allocation.
/// @details Increments the thread-local allocation count and returns nullptr on failure.
/// @param size Number of bytes to allocate.
/// @param alignment Required alignment in bytes.
/// @return Pointer to aligned storage, or nullptr if allocation fails.
void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    return alloc_aligned(size, static_cast<std::size_t>(alignment));
}

/// @brief Replaces global nothrow aligned array operator new[] with audited aligned allocation.
/// @details Increments the thread-local allocation count and returns nullptr on failure.
/// @param size Number of bytes to allocate.
/// @param alignment Required alignment in bytes.
/// @return Pointer to aligned array storage, or nullptr if allocation fails.
void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    return alloc_aligned(size, static_cast<std::size_t>(alignment));
}

/// @brief Replaces global unsized aligned single-object operator delete using free_aligned().
/// @details Pairs with aligned operator new overrides using platform aligned allocation.
/// @param ptr Pointer to aligned storage to free, or nullptr.
void operator delete(void* ptr, std::align_val_t) noexcept
{
    free_aligned(ptr);
}

/// @brief Replaces global unsized aligned array operator delete[] using free_aligned().
/// @details Pairs with aligned operator new[] overrides using platform aligned allocation.
/// @param ptr Pointer to aligned array storage to free, or nullptr.
void operator delete[](void* ptr, std::align_val_t) noexcept
{
    free_aligned(ptr);
}

/// @brief Replaces global sized aligned single-object operator delete using free_aligned().
/// @details Ignores size and alignment because the platform aligned allocator tracks block extents.
/// @param ptr Pointer to aligned storage to free, or nullptr.
void operator delete(void* ptr, std::size_t, std::align_val_t) noexcept
{
    free_aligned(ptr);
}

/// @brief Replaces global sized aligned array operator delete[] using free_aligned().
/// @details Ignores size and alignment because the platform aligned allocator tracks block extents.
/// @param ptr Pointer to aligned array storage to free, or nullptr.
void operator delete[](void* ptr, std::size_t, std::align_val_t) noexcept
{
    free_aligned(ptr);
}

/// @brief Replaces global nothrow aligned single-object operator delete using free_aligned().
/// @details Pairs with nothrow aligned operator new on constructor unwinds.
/// @param ptr Pointer to aligned storage to free, or nullptr.
void operator delete(void* ptr, std::align_val_t, const std::nothrow_t&) noexcept
{
    free_aligned(ptr);
}

/// @brief Replaces global nothrow aligned array operator delete[] using free_aligned().
/// @details Pairs with nothrow aligned operator new[] on constructor unwinds.
/// @param ptr Pointer to aligned array storage to free, or nullptr.
void operator delete[](void* ptr, std::align_val_t, const std::nothrow_t&) noexcept
{
    free_aligned(ptr);
}

// NOLINTEND(readability-inconsistent-declaration-parameter-name)

namespace cxxgpr::test_support
{

    /// @brief Returns the thread-local cumulative count of tracked C++ heap allocations.
    /// @details Reads the thread-local counter updated by all overridden global operator new functions.
    /// @return Cumulative number of tracked allocations on the calling thread.
    std::size_t thread_allocation_count() noexcept
    {
        return g_thread_alloc_count;
    }

    /// @brief Observes a pointer via a volatile sink to inhibit dead-store elimination.
    /// @details Writes ptr into a global volatile sink without reading or dereferencing the pointee.
    /// @param ptr Pointer to test data that must remain live.
    void touch_for_test(const void* ptr) noexcept
    {
        g_escape_sink = ptr;
    }

    /// @brief Snapshots thread_allocation_count() and optionally disables Eigen runtime malloc.
    /// @details Saves Eigen::internal::is_malloc_allowed() prior to disabling it so nested scopes restore properly.
    /// @param mode Allocation auditing mode selecting C++ and Eigen checks or C++ tracking only.
    NoAllocationScope::NoAllocationScope(AuditMode mode) noexcept :
        start_count_(g_thread_alloc_count),
        mode_(mode),
        prev_eigen_allowed_(Eigen::internal::is_malloc_allowed())
    {
        // Only suppress Eigen runtime malloc when both C++ and Eigen auditing are requested.
        if (mode_ == AuditMode::CppAndEigen)
        {
            Eigen::internal::set_is_malloc_allowed(false);
        }
    }

    /// @brief Restores Eigen runtime malloc state when leaving a CppAndEigen audit scope.
    /// @details Preserves nesting invariants by restoring prev_eigen_allowed_ rather than hardcoding true.
    NoAllocationScope::~NoAllocationScope() noexcept
    {
        // Restore the exact pre-scope Eigen malloc state so outer audit scopes remain intact.
        if (mode_ == AuditMode::CppAndEigen)
        {
            Eigen::internal::set_is_malloc_allowed(prev_eigen_allowed_);
        }
    }

    /// @brief Computes tracked C++ heap allocations performed on this thread since scope construction.
    /// @details Subtracts start_count_ captured at scope entry from the current thread-local counter.
    /// @return Number of tracked allocations executed within this scope on the calling thread.
    std::size_t NoAllocationScope::delta_allocations() const noexcept
    {
        return g_thread_alloc_count - start_count_;
    }

}  // namespace cxxgpr::test_support
