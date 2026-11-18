// SPDX-License-Identifier: Apache-2.0
/// @file dense.hpp
/// @brief Dense linear algebra primitives over column-major matrix and vector views.
/// @details Declares diagonal updates, lower-triangle copying and symmetrization, symmetric Frobenius inner products,
/// deterministic vector dot products, and column-major matrix-vector multiplication.

#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/precision/policy.hpp"

#include <expected>
#include <span>

/// @brief Dense and factorized linear algebra primitives for cxxgpr.
namespace cxxgpr::linalg
{

    /// @brief Adds a finite scalar offset to every diagonal entry of a square matrix in place.
    /// @details Mutates only the main diagonal entries a(i, i) without reading or modifying off-diagonal elements.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param a Square mutable matrix view to modify in place.
    /// @param value Finite scalar offset added to each diagonal entry.
    /// @return Empty expected on success, or GprError on invalid input.
    /// @retval GprErrorCode::EmptyInput When the matrix view is empty or null.
    /// @retval GprErrorCode::SizeOverflow When col_stride * cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When col_stride < rows or the matrix is not square.
    /// @retval GprErrorCode::NonFiniteInput When value is NaN or infinite.
    template <KernelScalarConcept T> [[nodiscard]] std::expected<void, GprError> add_to_diag(MatrixMut<T> a, T value);

    /// @brief Copies the lower triangle (including diagonal) of a square matrix src into dst.
    /// @details Strictly upper-triangular entries in both src and dst are neither read nor modified.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param src Square read-only source matrix view.
    /// @param dst Square mutable destination matrix view of matching order.
    /// @return Empty expected on success, or GprError on invalid dimensions or views.
    /// @retval GprErrorCode::EmptyInput When src or dst is empty or null.
    /// @retval GprErrorCode::SizeOverflow When stride * cols overflows std::size_t on either view.
    /// @retval GprErrorCode::LengthMismatch When either view has invalid stride, is non-square, or dimensions differ.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> copy_lower_triangle(MatrixRef<T> src, MatrixMut<T> dst);

    /// @brief Mirrors the strictly lower triangle of a square matrix into its strictly upper triangle.
    /// @details Produces a symmetric matrix in place when only the lower triangle was populated.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param a Square mutable matrix view to symmetrize in place.
    /// @return Empty expected on success, or GprError on invalid matrix view.
    /// @retval GprErrorCode::EmptyInput When a is empty or null.
    /// @retval GprErrorCode::SizeOverflow When col_stride * cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When col_stride < rows or a is not square.
    template <KernelScalarConcept T> [[nodiscard]] std::expected<void, GprError> symmetrize_lower(MatrixMut<T> a);

    /// @brief Computes the symmetric Frobenius inner product <A, B>_F reading only the lower triangles.
    /// @details Accumulates diagonal products once and strictly lower-triangular products twice in double precision in
    /// deterministic column-major order.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param a First square read-only symmetric matrix view (lower triangle read).
    /// @param b Second square read-only symmetric matrix view (lower triangle read).
    /// @return Symmetric Frobenius inner product in double precision, or GprError on failure.
    /// @retval GprErrorCode::EmptyInput When a or b is empty or null.
    /// @retval GprErrorCode::SizeOverflow When stride * cols overflows std::size_t on either view.
    /// @retval GprErrorCode::LengthMismatch When either view has invalid stride, is non-square, or dimensions differ.
    /// @retval GprErrorCode::NonFiniteInput When any lower-triangular entry of a or b is non-finite.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<double, GprError> frobenius_lower(MatrixRef<T> a, MatrixRef<T> b);

    /// @brief Computes the Euclidean dot product of two equal-length vectors in deterministic ascending order.
    /// @details Accumulates products in double precision from index 0 to n - 1 without SIMD reordering.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param a First read-only vector span.
    /// @param b Second read-only vector span of equal length.
    /// @return Euclidean inner product in double precision, or GprError on failure.
    /// @retval GprErrorCode::EmptyInput When either span is empty.
    /// @retval GprErrorCode::LengthMismatch When a.size() != b.size().
    /// @retval GprErrorCode::NonFiniteInput When any element of a or b is non-finite.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<double, GprError> dot(std::span<const T> a, std::span<const T> b);

    /// @brief Computes out = A * x in column-major order for a matrix A and vector x.
    /// @details Accumulates contributions column by column for cache-local column-major access without heap
    /// allocations.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param a Read-only column-major matrix view of shape m x n.
    /// @param x Read-only input vector of length n.
    /// @param out Preallocated mutable output vector of length m.
    /// @return Empty expected on success, or GprError on invalid dimensions or non-finite inputs.
    /// @retval GprErrorCode::EmptyInput When a, x, or out is empty or null.
    /// @retval GprErrorCode::SizeOverflow When col_stride * cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When stride < rows, x.size() != a.cols(), or out.size() != a.rows().
    /// @retval GprErrorCode::NonFiniteInput When any entry of a or x is non-finite.
    template <KernelScalarConcept T>
    [[nodiscard]] std::expected<void, GprError> matvec_columns(MatrixRef<T> a, std::span<const T> x, std::span<T> out);

}  // namespace cxxgpr::linalg
