// SPDX-License-Identifier: Apache-2.0
#include "allocation_audit.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <cstdlib>
#include <new>

namespace
{

    thread_local std::size_t g_thread_alloc_count = 0;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
    volatile const void* g_escape_sink = nullptr;       // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

    void* alloc_unaligned(std::size_t size) noexcept
    {
        ++g_thread_alloc_count;
        return std::malloc(size == 0 ? 1 : size);
    }

    void* alloc_aligned(std::size_t size, std::size_t alignment) noexcept
    {
        ++g_thread_alloc_count;
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

    void free_aligned(void* ptr) noexcept
    {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }

}  // namespace

// NOLINTBEGIN(readability-inconsistent-declaration-parameter-name)
void* operator new(std::size_t size)
{
    if (void* ptr = alloc_unaligned(size))
    {
        return ptr;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size)
{
    return ::operator new(size);
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept
{
    return alloc_unaligned(size);
}

void* operator new[](std::size_t size, const std::nothrow_t&) noexcept
{
    return alloc_unaligned(size);
}

void operator delete(void* ptr) noexcept
{
    std::free(ptr);
}

void operator delete[](void* ptr) noexcept
{
    std::free(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept
{
    std::free(ptr);
}

void operator delete[](void* ptr, std::size_t) noexcept
{
    std::free(ptr);
}

void operator delete(void* ptr, const std::nothrow_t&) noexcept
{
    std::free(ptr);
}

void operator delete[](void* ptr, const std::nothrow_t&) noexcept
{
    std::free(ptr);
}

void* operator new(std::size_t size, std::align_val_t alignment)
{
    if (void* ptr = alloc_aligned(size, static_cast<std::size_t>(alignment)))
    {
        return ptr;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size, std::align_val_t alignment)
{
    return ::operator new(size, alignment);
}

void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    return alloc_aligned(size, static_cast<std::size_t>(alignment));
}

void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    return alloc_aligned(size, static_cast<std::size_t>(alignment));
}

void operator delete(void* ptr, std::align_val_t) noexcept
{
    free_aligned(ptr);
}

void operator delete[](void* ptr, std::align_val_t) noexcept
{
    free_aligned(ptr);
}

void operator delete(void* ptr, std::size_t, std::align_val_t) noexcept
{
    free_aligned(ptr);
}

void operator delete[](void* ptr, std::size_t, std::align_val_t) noexcept
{
    free_aligned(ptr);
}

void operator delete(void* ptr, std::align_val_t, const std::nothrow_t&) noexcept
{
    free_aligned(ptr);
}

void operator delete[](void* ptr, std::align_val_t, const std::nothrow_t&) noexcept
{
    free_aligned(ptr);
}

// NOLINTEND(readability-inconsistent-declaration-parameter-name)

namespace cxxgpr::test_support
{

    std::size_t thread_allocation_count() noexcept
    {
        return g_thread_alloc_count;
    }

    void touch_for_test(const void* ptr) noexcept
    {
        g_escape_sink = ptr;
    }

    NoAllocationScope::NoAllocationScope(AuditMode mode) noexcept :
        start_count_(g_thread_alloc_count),
        mode_(mode),
        prev_eigen_allowed_(Eigen::internal::is_malloc_allowed())
    {
        if (mode_ == AuditMode::CppAndEigen)
        {
            Eigen::internal::set_is_malloc_allowed(false);
        }
    }

    NoAllocationScope::~NoAllocationScope() noexcept
    {
        if (mode_ == AuditMode::CppAndEigen)
        {
            Eigen::internal::set_is_malloc_allowed(prev_eigen_allowed_);
        }
    }

    std::size_t NoAllocationScope::delta_allocations() const noexcept
    {
        return g_thread_alloc_count - start_count_;
    }

}  // namespace cxxgpr::test_support
