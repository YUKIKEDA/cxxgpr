// SPDX-License-Identifier: Apache-2.0
/// @file eigen_map.hpp
/// @brief Internal Eigen strided map adapters and matrix/span validation helpers.
/// @details Translates non-owning column-major MatrixRef/MatrixMut views into Eigen::Map objects and enforces shape,
/// stride, and finiteness preconditions inside the linalg implementation layer.

#pragma once

#include "cxxgpr/core/checked.hpp"
#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/precision/policy.hpp"

#include <Eigen/Core>

#include <cmath>
#include <concepts>
#include <cstddef>
#include <expected>
#include <span>

#if defined(_M_X64) || defined(__x86_64__)
#    include <immintrin.h>
#endif

/// @brief Internal Eigen adapters and validation utilities for linear algebra implementations.
namespace cxxgpr::linalg::detail
{

    /// @brief Dynamic-sized column-major Eigen matrix type parameterized by scalar T.
    /// @details Used as the target storage layout for unaligned strided Eigen::Map views.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    template <KernelScalarConcept T>
    using ColMajorDynamic = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    /// @brief Read-only Eigen::Map over a column-major buffer with runtime outer stride.
    /// @details Wraps caller-owned buffers without alignment assumptions or copies.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    template <KernelScalarConcept T>
    using ConstEigenMap = Eigen::Map<const ColMajorDynamic<T>, Eigen::Unaligned, Eigen::OuterStride<>>;

    /// @brief Mutable Eigen::Map over a column-major buffer with runtime outer stride.
    /// @details Wraps caller-owned buffers without alignment assumptions or copies.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    template <KernelScalarConcept T>
    using MutEigenMap = Eigen::Map<ColMajorDynamic<T>, Eigen::Unaligned, Eigen::OuterStride<>>;

    /// @brief Wraps a read-only MatrixRef view as an Eigen ConstEigenMap.
    /// @details Preserves column-major dimensions and outer column stride with zero copies.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] mat Validated read-only column-major matrix view.
    /// @return Strided read-only Eigen map aliasing mat.data().
    template <KernelScalarConcept T> [[nodiscard]] inline ConstEigenMap<T> to_eigen_map(MatrixRef<T> mat) noexcept
    {
        return ConstEigenMap<T>(
            mat.data(), static_cast<Eigen::Index>(mat.rows()), static_cast<Eigen::Index>(mat.cols()),
            Eigen::OuterStride<>(static_cast<Eigen::Index>(mat.col_stride()))
        );
    }

    /// @brief Wraps a mutable MatrixMut view as an Eigen MutEigenMap.
    /// @details Preserves column-major dimensions and outer column stride with zero copies.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] mat Validated mutable column-major matrix view.
    /// @return Strided mutable Eigen map aliasing mat.data().
    template <KernelScalarConcept T> [[nodiscard]] inline MutEigenMap<T> to_eigen_map(MatrixMut<T> mat) noexcept
    {
        return MutEigenMap<T>(
            mat.data(), static_cast<Eigen::Index>(mat.rows()), static_cast<Eigen::Index>(mat.cols()),
            Eigen::OuterStride<>(static_cast<Eigen::Index>(mat.col_stride()))
        );
    }

    /// @brief Validates non-emptiness, stride overflow, column stride, and optional squareness of a matrix view.
    /// @details Enforces the validation order EmptyInput -> SizeOverflow -> LengthMismatch before any indexing occurs.
    /// @tparam MatView Matrix view type (MatrixRef<T> or MatrixMut<T>).
    /// @param[in] mat Matrix view to validate.
    /// @param[in] require_square When true, rejects non-square matrices with LengthMismatch.
    /// @return Empty expected on success, or GprError on invalid view geometry.
    /// @retval GprErrorCode::EmptyInput When the view is empty or its base pointer is null.
    /// @retval GprErrorCode::SizeOverflow When col_stride * cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When col_stride < rows, or when require_square is true and rows != cols.
    template <typename MatView>
    [[nodiscard]] inline std::expected<void, GprError> validate_matrix_view(MatView mat, bool require_square)
    {
        // Enforce validation order: EmptyInput -> SizeOverflow -> LengthMismatch.
        if (mat.is_empty() || mat.data() == nullptr)
        {
            return std::unexpected(GprError::empty_input());
        }
        const auto len_check = checked_mul(mat.col_stride(), mat.cols());
        if (!len_check.has_value())
        {
            return std::unexpected(len_check.error());
        }
        if (mat.col_stride() < mat.rows())
        {
            return std::unexpected(
                GprError::length_mismatch("matrix column stride must be at least the number of rows")
            );
        }
        if (require_square && !mat.is_square())
        {
            return std::unexpected(GprError::length_mismatch("matrix must be square"));
        }
        return {};
    }

    /// @brief Checks whether every entry in a contiguous scalar slice `ptr[0..len)` is finite.
    /// @details Uses branchless 4-way unrolled AVX-512, AVX2, or SSE2 exponent-mask checks on x86_64 with scalar
    /// `std::isfinite` fallback so normal, zero, and subnormal values pass while NaN and +/-Inf return false.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] ptr Pointer to the start of the contiguous slice.
    /// @param[in] len Number of contiguous elements to inspect.
    /// @return True iff every element in `ptr[0..len)` is finite.
    template <KernelScalarConcept T>
    [[nodiscard]] inline bool all_finite_contiguous(const T* ptr, std::size_t len) noexcept
    {
        if (len == 0U)
        {
            return true;
        }
        std::size_t i = 0U;
        if constexpr (std::same_as<T, double>)
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            const __m512i exp_mask = _mm512_set1_epi64(0x7FF0000000000000LL);
            unsigned int bad0 = 0U;
            unsigned int bad1 = 0U;
            for (; i + 32U <= len; i += 32U)
            {
                const __m512i v0 = _mm512_castpd_si512(_mm512_loadu_pd(ptr + i));
                const __m512i v1 = _mm512_castpd_si512(_mm512_loadu_pd(ptr + i + 8U));
                const __m512i v2 = _mm512_castpd_si512(_mm512_loadu_pd(ptr + i + 16U));
                const __m512i v3 = _mm512_castpd_si512(_mm512_loadu_pd(ptr + i + 24U));
                bad0 |= static_cast<unsigned int>(_mm512_cmpeq_epi64_mask(_mm512_and_si512(v0, exp_mask), exp_mask));
                bad1 |= static_cast<unsigned int>(_mm512_cmpeq_epi64_mask(_mm512_and_si512(v1, exp_mask), exp_mask));
                bad0 |= static_cast<unsigned int>(_mm512_cmpeq_epi64_mask(_mm512_and_si512(v2, exp_mask), exp_mask));
                bad1 |= static_cast<unsigned int>(_mm512_cmpeq_epi64_mask(_mm512_and_si512(v3, exp_mask), exp_mask));
            }
            unsigned int bad = bad0 | bad1;
            for (; i + 8U <= len; i += 8U)
            {
                const __m512i v = _mm512_castpd_si512(_mm512_loadu_pd(ptr + i));
                bad |= static_cast<unsigned int>(_mm512_cmpeq_epi64_mask(_mm512_and_si512(v, exp_mask), exp_mask));
            }
            if (bad != 0U)
            {
                return false;
            }
#elifdef __AVX2__
            const __m256i exp_mask = _mm256_set1_epi64x(0x7FF0000000000000LL);
            __m256i bad0 = _mm256_setzero_si256();
            __m256i bad1 = _mm256_setzero_si256();
            for (; i + 16U <= len; i += 16U)
            {
                const __m256i v0 = _mm256_castpd_si256(_mm256_loadu_pd(ptr + i));
                const __m256i v1 = _mm256_castpd_si256(_mm256_loadu_pd(ptr + i + 4U));
                const __m256i v2 = _mm256_castpd_si256(_mm256_loadu_pd(ptr + i + 8U));
                const __m256i v3 = _mm256_castpd_si256(_mm256_loadu_pd(ptr + i + 12U));
                bad0 = _mm256_or_si256(bad0, _mm256_cmpeq_epi64(_mm256_and_si256(v0, exp_mask), exp_mask));
                bad1 = _mm256_or_si256(bad1, _mm256_cmpeq_epi64(_mm256_and_si256(v1, exp_mask), exp_mask));
                bad0 = _mm256_or_si256(bad0, _mm256_cmpeq_epi64(_mm256_and_si256(v2, exp_mask), exp_mask));
                bad1 = _mm256_or_si256(bad1, _mm256_cmpeq_epi64(_mm256_and_si256(v3, exp_mask), exp_mask));
            }
            __m256i bad = _mm256_or_si256(bad0, bad1);
            for (; i + 4U <= len; i += 4U)
            {
                const __m256i v = _mm256_castpd_si256(_mm256_loadu_pd(ptr + i));
                bad = _mm256_or_si256(bad, _mm256_cmpeq_epi64(_mm256_and_si256(v, exp_mask), exp_mask));
            }
            if (_mm256_movemask_epi8(bad) != 0)
            {
                return false;
            }
#elif defined(_M_X64) || defined(__x86_64__)
            const __m128i exp_mask = _mm_set_epi32(0x7FF00000, 0, 0x7FF00000, 0);
            const __m128i exp_target = _mm_set_epi32(0x7FF00000, -1, 0x7FF00000, -1);
            __m128i bad0 = _mm_setzero_si128();
            __m128i bad1 = _mm_setzero_si128();
            for (; i + 8U <= len; i += 8U)
            {
                const __m128i v0 = _mm_castpd_si128(_mm_loadu_pd(ptr + i));
                const __m128i v1 = _mm_castpd_si128(_mm_loadu_pd(ptr + i + 2U));
                const __m128i v2 = _mm_castpd_si128(_mm_loadu_pd(ptr + i + 4U));
                const __m128i v3 = _mm_castpd_si128(_mm_loadu_pd(ptr + i + 6U));
                bad0 = _mm_or_si128(bad0, _mm_cmpeq_epi32(_mm_and_si128(v0, exp_mask), exp_target));
                bad1 = _mm_or_si128(bad1, _mm_cmpeq_epi32(_mm_and_si128(v1, exp_mask), exp_target));
                bad0 = _mm_or_si128(bad0, _mm_cmpeq_epi32(_mm_and_si128(v2, exp_mask), exp_target));
                bad1 = _mm_or_si128(bad1, _mm_cmpeq_epi32(_mm_and_si128(v3, exp_mask), exp_target));
            }
            __m128i bad = _mm_or_si128(bad0, bad1);
            for (; i + 2U <= len; i += 2U)
            {
                const __m128i v = _mm_castpd_si128(_mm_loadu_pd(ptr + i));
                bad = _mm_or_si128(bad, _mm_cmpeq_epi32(_mm_and_si128(v, exp_mask), exp_target));
            }
            if (_mm_movemask_epi8(bad) != 0)
            {
                return false;
            }
#endif
        }
        else
        {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            const __m512i exp_mask = _mm512_set1_epi32(0x7F800000);
            unsigned int bad0 = 0U;
            unsigned int bad1 = 0U;
            for (; i + 64U <= len; i += 64U)
            {
                const __m512i v0 = _mm512_castps_si512(_mm512_loadu_ps(ptr + i));
                const __m512i v1 = _mm512_castps_si512(_mm512_loadu_ps(ptr + i + 16U));
                const __m512i v2 = _mm512_castps_si512(_mm512_loadu_ps(ptr + i + 32U));
                const __m512i v3 = _mm512_castps_si512(_mm512_loadu_ps(ptr + i + 48U));
                bad0 |= static_cast<unsigned int>(_mm512_cmpeq_epi32_mask(_mm512_and_si512(v0, exp_mask), exp_mask));
                bad1 |= static_cast<unsigned int>(_mm512_cmpeq_epi32_mask(_mm512_and_si512(v1, exp_mask), exp_mask));
                bad0 |= static_cast<unsigned int>(_mm512_cmpeq_epi32_mask(_mm512_and_si512(v2, exp_mask), exp_mask));
                bad1 |= static_cast<unsigned int>(_mm512_cmpeq_epi32_mask(_mm512_and_si512(v3, exp_mask), exp_mask));
            }
            unsigned int bad = bad0 | bad1;
            for (; i + 16U <= len; i += 16U)
            {
                const __m512i v = _mm512_castps_si512(_mm512_loadu_ps(ptr + i));
                bad |= static_cast<unsigned int>(_mm512_cmpeq_epi32_mask(_mm512_and_si512(v, exp_mask), exp_mask));
            }
            if (bad != 0U)
            {
                return false;
            }
#elifdef __AVX2__
            const __m256i exp_mask = _mm256_set1_epi32(0x7F800000);
            __m256i bad0 = _mm256_setzero_si256();
            __m256i bad1 = _mm256_setzero_si256();
            for (; i + 32U <= len; i += 32U)
            {
                const __m256i v0 = _mm256_castps_si256(_mm256_loadu_ps(ptr + i));
                const __m256i v1 = _mm256_castps_si256(_mm256_loadu_ps(ptr + i + 8U));
                const __m256i v2 = _mm256_castps_si256(_mm256_loadu_ps(ptr + i + 16U));
                const __m256i v3 = _mm256_castps_si256(_mm256_loadu_ps(ptr + i + 24U));
                bad0 = _mm256_or_si256(bad0, _mm256_cmpeq_epi32(_mm256_and_si256(v0, exp_mask), exp_mask));
                bad1 = _mm256_or_si256(bad1, _mm256_cmpeq_epi32(_mm256_and_si256(v1, exp_mask), exp_mask));
                bad0 = _mm256_or_si256(bad0, _mm256_cmpeq_epi32(_mm256_and_si256(v2, exp_mask), exp_mask));
                bad1 = _mm256_or_si256(bad1, _mm256_cmpeq_epi32(_mm256_and_si256(v3, exp_mask), exp_mask));
            }
            __m256i bad = _mm256_or_si256(bad0, bad1);
            for (; i + 8U <= len; i += 8U)
            {
                const __m256i v = _mm256_castps_si256(_mm256_loadu_ps(ptr + i));
                bad = _mm256_or_si256(bad, _mm256_cmpeq_epi32(_mm256_and_si256(v, exp_mask), exp_mask));
            }
            if (_mm256_movemask_epi8(bad) != 0)
            {
                return false;
            }
#elif defined(_M_X64) || defined(__x86_64__)
            const __m128i exp_mask = _mm_set1_epi32(0x7F800000);
            __m128i bad0 = _mm_setzero_si128();
            __m128i bad1 = _mm_setzero_si128();
            for (; i + 16U <= len; i += 16U)
            {
                const __m128i v0 = _mm_castps_si128(_mm_loadu_ps(ptr + i));
                const __m128i v1 = _mm_castps_si128(_mm_loadu_ps(ptr + i + 4U));
                const __m128i v2 = _mm_castps_si128(_mm_loadu_ps(ptr + i + 8U));
                const __m128i v3 = _mm_castps_si128(_mm_loadu_ps(ptr + i + 12U));
                bad0 = _mm_or_si128(bad0, _mm_cmpeq_epi32(_mm_and_si128(v0, exp_mask), exp_mask));
                bad1 = _mm_or_si128(bad1, _mm_cmpeq_epi32(_mm_and_si128(v1, exp_mask), exp_mask));
                bad0 = _mm_or_si128(bad0, _mm_cmpeq_epi32(_mm_and_si128(v2, exp_mask), exp_mask));
                bad1 = _mm_or_si128(bad1, _mm_cmpeq_epi32(_mm_and_si128(v3, exp_mask), exp_mask));
            }
            __m128i bad = _mm_or_si128(bad0, bad1);
            for (; i + 4U <= len; i += 4U)
            {
                const __m128i v = _mm_castps_si128(_mm_loadu_ps(ptr + i));
                bad = _mm_or_si128(bad, _mm_cmpeq_epi32(_mm_and_si128(v, exp_mask), exp_mask));
            }
            if (_mm_movemask_epi8(bad) != 0)
            {
                return false;
            }
#endif
        }
        for (; i < len; ++i)
        {
            if (!std::isfinite(ptr[i]))
            {
                return false;
            }
        }
        return true;
    }

    /// @brief Verifies that all entries in the lower triangle (including diagonal) of mat are finite.
    /// @details Sweeps column by column with row >= col using SIMD slice checks so uninitialized upper entries are
    /// ignored.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] mat Validated square read-only matrix view.
    /// @return Empty expected when all lower-triangular elements are finite, or GprError otherwise.
    /// @retval GprErrorCode::NonFiniteInput When any lower-triangular entry is NaN or infinite.
    template <KernelScalarConcept T>
    [[nodiscard]] inline std::expected<void, GprError> require_lower_finite(MatrixRef<T> mat)
    {
        // Check strictly lower + diagonal entries only so upper triangle may remain uninitialized.
        const std::size_t n = mat.rows();
        for (std::size_t col = 0; col < n; ++col)
        {
            if (!all_finite_contiguous(&mat(col, col), n - col))
            {
                return std::unexpected(GprError::non_finite_input());
            }
        }
        return {};
    }

    /// @brief Verifies that every entry in mat is finite.
    /// @details Scans all contiguous data in a single SSE2 pass when packed, or column by column otherwise.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] mat Validated read-only matrix view.
    /// @return Empty expected when all elements are finite, or GprError otherwise.
    /// @retval GprErrorCode::NonFiniteInput When any matrix element is NaN or infinite.
    template <KernelScalarConcept T>
    [[nodiscard]] inline std::expected<void, GprError> require_matrix_finite(MatrixRef<T> mat)
    {
        const std::size_t rows = mat.rows();
        const std::size_t cols = mat.cols();
        if (mat.is_contiguous())
        {
            if (!all_finite_contiguous(mat.data(), rows * cols))
            {
                return std::unexpected(GprError::non_finite_input());
            }
            return {};
        }
        for (std::size_t col = 0; col < cols; ++col)
        {
            if (!all_finite_contiguous(mat.col(col).data(), rows))
            {
                return std::unexpected(GprError::non_finite_input());
            }
        }
        return {};
    }

    /// @brief Verifies that every element in a contiguous scalar span is finite.
    /// @details Rejects NaN and infinity while permitting subnormal finite numbers.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] values Contiguous span of scalar values to inspect.
    /// @return Empty expected when all elements are finite, or GprError otherwise.
    /// @retval GprErrorCode::NonFiniteInput When any element in values is NaN or infinite.
    template <KernelScalarConcept T>
    [[nodiscard]] inline std::expected<void, GprError> require_span_finite(std::span<const T> values)
    {
        // Reject NaN/Inf entries while preserving subnormal finite values.
        if (!all_finite_contiguous(values.data(), values.size()))
        {
            return std::unexpected(GprError::non_finite_input());
        }
        return {};
    }

    /// @brief Converts a mutable MatrixMut view into an aliasing read-only MatrixRef view.
    /// @details Copies pointer, dimensions, and column stride with zero allocations.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] mat Mutable matrix view to alias as read-only.
    /// @return Read-only MatrixRef aliasing the same underlying buffer.
    template <KernelScalarConcept T> [[nodiscard]] inline MatrixRef<T> as_ref(MatrixMut<T> mat) noexcept
    {
        return MatrixRef<T>(mat.data(), mat.rows(), mat.cols(), mat.col_stride());
    }

}  // namespace cxxgpr::linalg::detail
