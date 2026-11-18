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
#include <cstddef>
#include <expected>
#include <span>

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
    /// @param mat Validated read-only column-major matrix view.
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
    /// @param mat Validated mutable column-major matrix view.
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
    /// @param mat Matrix view to validate.
    /// @param require_square When true, rejects non-square matrices with LengthMismatch.
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

    /// @brief Verifies that all entries in the lower triangle (including diagonal) of mat are finite.
    /// @details Sweeps column by column with row >= col so uninitialized upper-triangular entries are ignored.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param mat Validated square read-only matrix view.
    /// @return Empty expected when all lower-triangular elements are finite, or GprError otherwise.
    /// @retval GprErrorCode::NonFiniteInput When any lower-triangular entry is NaN or infinite.
    template <KernelScalarConcept T>
    [[nodiscard]] inline std::expected<void, GprError> require_lower_finite(MatrixRef<T> mat)
    {
        // Check strictly lower + diagonal entries only so upper triangle may remain uninitialized.
        const std::size_t n = mat.rows();
        for (std::size_t col = 0; col < n; ++col)
        {
            for (std::size_t row = col; row < n; ++row)
            {
                if (!std::isfinite(mat(row, col)))
                {
                    return std::unexpected(GprError::non_finite_input());
                }
            }
        }
        return {};
    }

    /// @brief Verifies that every entry in mat is finite.
    /// @details Scans all rows across every column in column-major order.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param mat Validated read-only matrix view.
    /// @return Empty expected when all elements are finite, or GprError otherwise.
    /// @retval GprErrorCode::NonFiniteInput When any matrix element is NaN or infinite.
    template <KernelScalarConcept T>
    [[nodiscard]] inline std::expected<void, GprError> require_matrix_finite(MatrixRef<T> mat)
    {
        // Sweep column-major so adjacent rows stay contiguous in memory.
        const std::size_t rows = mat.rows();
        const std::size_t cols = mat.cols();
        for (std::size_t col = 0; col < cols; ++col)
        {
            for (std::size_t row = 0; row < rows; ++row)
            {
                if (!std::isfinite(mat(row, col)))
                {
                    return std::unexpected(GprError::non_finite_input());
                }
            }
        }
        return {};
    }

    /// @brief Verifies that every element in a contiguous scalar span is finite.
    /// @details Rejects NaN and infinity while permitting subnormal finite numbers.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param values Contiguous span of scalar values to inspect.
    /// @return Empty expected when all elements are finite, or GprError otherwise.
    /// @retval GprErrorCode::NonFiniteInput When any element in values is NaN or infinite.
    template <KernelScalarConcept T>
    [[nodiscard]] inline std::expected<void, GprError> require_span_finite(std::span<const T> values)
    {
        // Reject NaN/Inf entries while preserving subnormal finite values.
        for (const T v : values)
        {
            if (!std::isfinite(v))
            {
                return std::unexpected(GprError::non_finite_input());
            }
        }
        return {};
    }

    /// @brief Converts a mutable MatrixMut view into an aliasing read-only MatrixRef view.
    /// @details Copies pointer, dimensions, and column stride with zero allocations.
    /// @tparam T Floating-point kernel scalar type satisfying KernelScalarConcept.
    /// @param mat Mutable matrix view to alias as read-only.
    /// @return Read-only MatrixRef aliasing the same underlying buffer.
    template <KernelScalarConcept T> [[nodiscard]] inline MatrixRef<T> as_ref(MatrixMut<T> mat) noexcept
    {
        return MatrixRef<T>(mat.data(), mat.rows(), mat.cols(), mat.col_stride());
    }

}  // namespace cxxgpr::linalg::detail
