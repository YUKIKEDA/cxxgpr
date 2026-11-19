// SPDX-License-Identifier: Apache-2.0
/// @file dense.cpp
/// @brief Implementations of dense column-major matrix and vector primitives.
/// @details Provides explicit float and double instantiations of diagonal updates, lower-triangle copying and
/// symmetrization, symmetric Frobenius inner products, deterministic vector dot products, and matrix-vector products.

#include "cxxgpr/linalg/dense.hpp"

#include "cxxgpr/core/checked.hpp"

#include <cmath>
#include <cstddef>

#include "linalg/eigen_map.hpp"

namespace cxxgpr::linalg
{

    /// @brief Adds a finite scalar offset to every diagonal entry of a square matrix in place.
    /// @details Validates squareness and finiteness of value before mutating a(i, i) across all indices.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in,out] a Square mutable matrix view to modify in place.
    /// @param[in] value Finite scalar offset added to each diagonal entry.
    /// @return Empty expected on success, or GprError on invalid input.
    /// @retval GprErrorCode::EmptyInput When a is empty or null.
    /// @retval GprErrorCode::SizeOverflow When col_stride * cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When col_stride < rows or a is not square.
    /// @retval GprErrorCode::NonFiniteInput When value is NaN or infinite.
    template <KernelScalarConcept T> std::expected<void, GprError> add_to_diag(MatrixMut<T> a, T value)
    {
        // Validate view geometry first, then verify the diagonal increment is finite.
        if (const auto check = detail::validate_matrix_view(a, true); !check.has_value())
        {
            return check;
        }
        if (!std::isfinite(value))
        {
            return std::unexpected(GprError::non_finite_input());
        }
        const std::size_t n = a.rows();
        for (std::size_t i = 0; i < n; ++i)
        {
            a(i, i) += value;
        }
        return {};
    }

    /// @brief Copies the lower triangle (including diagonal) of a square matrix src into dst.
    /// @details Validates both square views and matching order, then copies row >= col in column-major order.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] src Square read-only source matrix view.
    /// @param[out] dst Square mutable destination matrix view of matching order.
    /// @return Empty expected on success, or GprError on invalid dimensions or views.
    /// @retval GprErrorCode::EmptyInput When src or dst is empty or null.
    /// @retval GprErrorCode::SizeOverflow When stride * cols overflows std::size_t on either view.
    /// @retval GprErrorCode::LengthMismatch When either view has invalid stride, is non-square, or dimensions differ.
    template <KernelScalarConcept T>
    std::expected<void, GprError> copy_lower_triangle(MatrixRef<T> src, MatrixMut<T> dst)
    {
        // Validate view shapes and dimension parity before accessing elements.
        if (const auto check = detail::validate_matrix_view(src, true); !check.has_value())
        {
            return check;
        }
        if (const auto check = detail::validate_matrix_view(dst, true); !check.has_value())
        {
            return check;
        }
        if (src.rows() != dst.rows())
        {
            return std::unexpected(GprError::length_mismatch("source and destination matrix sizes"));
        }
        // Sweep column by column from the diagonal downward so upper triangles remain untouched.
        const std::size_t n = src.rows();
        for (std::size_t col = 0; col < n; ++col)
        {
            for (std::size_t row = col; row < n; ++row)
            {
                dst(row, col) = src(row, col);
            }
        }
        return {};
    }

    /// @brief Mirrors the strictly lower triangle of a square matrix into its strictly upper triangle.
    /// @details Leaves diagonal entries untouched and copies a(i, j) into a(j, i) for i > j.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in,out] a Square mutable matrix view to symmetrize in place.
    /// @return Empty expected on success, or GprError on invalid matrix view.
    /// @retval GprErrorCode::EmptyInput When a is empty or null.
    /// @retval GprErrorCode::SizeOverflow When col_stride * cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When col_stride < rows or a is not square.
    template <KernelScalarConcept T> std::expected<void, GprError> symmetrize_lower(MatrixMut<T> a)
    {
        if (const auto check = detail::validate_matrix_view(a, true); !check.has_value())
        {
            return check;
        }
        // Read contiguous column slices below the diagonal and write their symmetric upper counterparts.
        const std::size_t n = a.rows();
        for (std::size_t j = 0; j < n; ++j)
        {
            for (std::size_t i = j + 1; i < n; ++i)
            {
                a(j, i) = a(i, j);
            }
        }
        return {};
    }

    /// @brief Computes the symmetric Frobenius inner product <A, B>_F reading only the lower triangles.
    /// @details Validates views, dimensions, and lower-triangle finiteness before accumulating diagonal terms once and
    /// strictly lower terms twice in double precision.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] a First square read-only symmetric matrix view (lower triangle read).
    /// @param[in] b Second square read-only symmetric matrix view (lower triangle read).
    /// @return Symmetric Frobenius inner product in double precision, or GprError on failure.
    /// @retval GprErrorCode::EmptyInput When a or b is empty or null.
    /// @retval GprErrorCode::SizeOverflow When stride * cols overflows std::size_t on either view.
    /// @retval GprErrorCode::LengthMismatch When either view has invalid stride, is non-square, or dimensions differ.
    /// @retval GprErrorCode::NonFiniteInput When any lower-triangular entry of a or b is non-finite.
    template <KernelScalarConcept T> std::expected<double, GprError> frobenius_lower(MatrixRef<T> a, MatrixRef<T> b)
    {
        // Enforce validation order: view geometry -> dimension match -> lower-triangle finiteness.
        if (const auto check = detail::validate_matrix_view(a, true); !check.has_value())
        {
            return std::unexpected(check.error());
        }
        if (const auto check = detail::validate_matrix_view(b, true); !check.has_value())
        {
            return std::unexpected(check.error());
        }
        if (a.rows() != b.rows())
        {
            return std::unexpected(GprError::length_mismatch("matrices in frobenius_lower must share dimension"));
        }
        if (const auto fin_a = detail::require_lower_finite(a); !fin_a.has_value())
        {
            return std::unexpected(fin_a.error());
        }
        if (const auto fin_b = detail::require_lower_finite(b); !fin_b.has_value())
        {
            return std::unexpected(fin_b.error());
        }
        // Accumulate <A, B>_F = sum_i A_ii B_ii + 2 * sum_{i > j} A_ij B_ij in double precision.
        const std::size_t n = a.rows();
        double sum = 0.0;
        for (std::size_t col = 0; col < n; ++col)
        {
            sum += static_cast<double>(a(col, col)) * static_cast<double>(b(col, col));
            for (std::size_t row = col + 1; row < n; ++row)
            {
                sum += 2.0 * static_cast<double>(a(row, col)) * static_cast<double>(b(row, col));
            }
        }
        return sum;
    }

    /// @brief Computes the Euclidean dot product of two equal-length vectors in deterministic ascending order.
    /// @details Validates non-emptiness, matching length, and element finiteness before accumulating in double
    /// precision.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] a First read-only vector span.
    /// @param[in] b Second read-only vector span of equal length.
    /// @return Euclidean inner product in double precision, or GprError on failure.
    /// @retval GprErrorCode::EmptyInput When either span is empty.
    /// @retval GprErrorCode::LengthMismatch When a.size() != b.size().
    /// @retval GprErrorCode::NonFiniteInput When any element of a or b is non-finite.
    template <KernelScalarConcept T> std::expected<double, GprError> dot(std::span<const T> a, std::span<const T> b)
    {
        // Enforce validation order: empty -> length mismatch -> finiteness.
        if (a.empty() || b.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        if (const auto cnt = require_count(a.size(), b.size(), "dot vector length"); !cnt.has_value())
        {
            return std::unexpected(cnt.error());
        }
        if (const auto fin_a = detail::require_span_finite(a); !fin_a.has_value())
        {
            return std::unexpected(fin_a.error());
        }
        if (const auto fin_b = detail::require_span_finite(b); !fin_b.has_value())
        {
            return std::unexpected(fin_b.error());
        }
        // Sum in strict ascending index order for reproducible floating-point rounding.
        double sum = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        }
        return sum;
    }

    /// @brief Computes out = A * x in column-major order for a matrix A and vector x.
    /// @details Validates matrix geometry, span lengths, and finiteness of A and x, then zeroes out and accumulates
    /// column-by-column axpy updates.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param[in] a Read-only column-major matrix view of shape m x n.
    /// @param[in] x Read-only input vector of length n.
    /// @param[out] out Preallocated mutable output vector of length m.
    /// @return Empty expected on success, or GprError on invalid dimensions or non-finite inputs.
    /// @retval GprErrorCode::EmptyInput When a, x, or out is empty or null.
    /// @retval GprErrorCode::SizeOverflow When col_stride * cols overflows std::size_t.
    /// @retval GprErrorCode::LengthMismatch When stride < rows, x.size() != a.cols(), or out.size() != a.rows().
    /// @retval GprErrorCode::NonFiniteInput When any entry of a or x is non-finite.
    template <KernelScalarConcept T>
    std::expected<void, GprError> matvec_columns(MatrixRef<T> a, std::span<const T> x, std::span<T> out)
    {
        // Enforce validation order: view geometry -> empty vectors -> dimension matches -> input finiteness.
        if (const auto check = detail::validate_matrix_view(a, false); !check.has_value())
        {
            return check;
        }
        if (x.empty() || out.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        if (const auto cnt_x = require_count(x.size(), a.cols(), "matvec input vector length"); !cnt_x.has_value())
        {
            return cnt_x;
        }
        if (const auto cnt_out = require_count(out.size(), a.rows(), "matvec output vector length");
            !cnt_out.has_value())
        {
            return cnt_out;
        }
        if (const auto fin_a = detail::require_matrix_finite(a); !fin_a.has_value())
        {
            return fin_a;
        }
        if (const auto fin_x = detail::require_span_finite(x); !fin_x.has_value())
        {
            return fin_x;
        }

        // Accumulate out += A(:, col) * x[col] so reads from column-major A remain sequential.
        const std::size_t m = a.rows();
        const std::size_t n = a.cols();
        for (std::size_t i = 0; i < m; ++i)
        {
            out[i] = T{0};
        }
        for (std::size_t col = 0; col < n; ++col)
        {
            const T x_col = x[col];
            for (std::size_t row = 0; row < m; ++row)
            {
                out[row] += a(row, col) * x_col;
            }
        }
        return {};
    }

    template std::expected<void, GprError> add_to_diag<float>(MatrixMut<float>, float);
    template std::expected<void, GprError> add_to_diag<double>(MatrixMut<double>, double);

    template std::expected<void, GprError> copy_lower_triangle<float>(MatrixRef<float>, MatrixMut<float>);
    template std::expected<void, GprError> copy_lower_triangle<double>(MatrixRef<double>, MatrixMut<double>);

    template std::expected<void, GprError> symmetrize_lower<float>(MatrixMut<float>);
    template std::expected<void, GprError> symmetrize_lower<double>(MatrixMut<double>);

    template std::expected<double, GprError> frobenius_lower<float>(MatrixRef<float>, MatrixRef<float>);
    template std::expected<double, GprError> frobenius_lower<double>(MatrixRef<double>, MatrixRef<double>);

    template std::expected<double, GprError> dot<float>(std::span<const float>, std::span<const float>);
    template std::expected<double, GprError> dot<double>(std::span<const double>, std::span<const double>);

    template std::expected<void, GprError> matvec_columns<float>(
        MatrixRef<float>, std::span<const float>, std::span<float>
    );
    template std::expected<void, GprError> matvec_columns<double>(
        MatrixRef<double>, std::span<const double>, std::span<double>
    );

}  // namespace cxxgpr::linalg
