// SPDX-License-Identifier: Apache-2.0
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

namespace cxxgpr::linalg::detail
{

    template <KernelScalarConcept T>
    using ColMajorDynamic = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    template <KernelScalarConcept T>
    using ConstEigenMap = Eigen::Map<const ColMajorDynamic<T>, Eigen::Unaligned, Eigen::OuterStride<>>;

    template <KernelScalarConcept T>
    using MutEigenMap = Eigen::Map<ColMajorDynamic<T>, Eigen::Unaligned, Eigen::OuterStride<>>;

    template <KernelScalarConcept T> [[nodiscard]] inline ConstEigenMap<T> to_eigen_map(MatrixRef<T> mat) noexcept
    {
        return ConstEigenMap<T>(
            mat.data(), static_cast<Eigen::Index>(mat.rows()), static_cast<Eigen::Index>(mat.cols()),
            Eigen::OuterStride<>(static_cast<Eigen::Index>(mat.col_stride()))
        );
    }

    template <KernelScalarConcept T> [[nodiscard]] inline MutEigenMap<T> to_eigen_map(MatrixMut<T> mat) noexcept
    {
        return MutEigenMap<T>(
            mat.data(), static_cast<Eigen::Index>(mat.rows()), static_cast<Eigen::Index>(mat.cols()),
            Eigen::OuterStride<>(static_cast<Eigen::Index>(mat.col_stride()))
        );
    }

    template <typename MatView>
    [[nodiscard]] inline std::expected<void, GprError> validate_matrix_view(MatView mat, bool require_square)
    {
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

    template <KernelScalarConcept T>
    [[nodiscard]] inline std::expected<void, GprError> require_lower_finite(MatrixRef<T> mat)
    {
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

    template <KernelScalarConcept T>
    [[nodiscard]] inline std::expected<void, GprError> require_matrix_finite(MatrixRef<T> mat)
    {
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

    template <KernelScalarConcept T>
    [[nodiscard]] inline std::expected<void, GprError> require_span_finite(std::span<const T> values)
    {
        for (const T v : values)
        {
            if (!std::isfinite(v))
            {
                return std::unexpected(GprError::non_finite_input());
            }
        }
        return {};
    }

    template <KernelScalarConcept T> [[nodiscard]] inline MatrixRef<T> as_ref(MatrixMut<T> mat) noexcept
    {
        return MatrixRef<T>(mat.data(), mat.rows(), mat.cols(), mat.col_stride());
    }

}  // namespace cxxgpr::linalg::detail
