// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/linalg/dense.hpp"

#include "cxxgpr/core/checked.hpp"

#include <cmath>
#include <cstddef>

#include "linalg/eigen_map.hpp"

namespace cxxgpr::linalg
{

    template <KernelScalarConcept T> std::expected<void, GprError> add_to_diag(MatrixMut<T> a, T value)
    {
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

    template <KernelScalarConcept T>
    std::expected<void, GprError> copy_lower_triangle(MatrixRef<T> src, MatrixMut<T> dst)
    {
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

    template <KernelScalarConcept T> std::expected<void, GprError> symmetrize_lower(MatrixMut<T> a)
    {
        if (const auto check = detail::validate_matrix_view(a, true); !check.has_value())
        {
            return check;
        }
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

    template <KernelScalarConcept T> std::expected<double, GprError> frobenius_lower(MatrixRef<T> a, MatrixRef<T> b)
    {
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

    template <KernelScalarConcept T> std::expected<double, GprError> dot(std::span<const T> a, std::span<const T> b)
    {
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
        double sum = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        }
        return sum;
    }

    template <KernelScalarConcept T>
    std::expected<void, GprError> matvec_columns(MatrixRef<T> a, std::span<const T> x, std::span<T> out)
    {
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
