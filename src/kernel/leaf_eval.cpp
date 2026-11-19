// SPDX-License-Identifier: Apache-2.0
/// @file leaf_eval.cpp
/// @brief Leaf-level kernel evaluation, first derivatives, and analytical Hessians.
/// @details Implements Gram evaluation, diagonal evaluation, log-hyperparameter gradients, and second-order Hessians
/// for all built-in and custom kernel leaves over squared Euclidean distances, point coordinates, and packed ARD
/// squared differences.

#include "kernel/leaf_eval.hpp"

#include "cxxgpr/core/checked.hpp"
#include "cxxgpr/linalg/parallel.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <utility>

#include "kernel/exp_simd.hpp"

#ifdef _OPENMP
#    include <omp.h>
#endif

/// @brief Internal kernel leaf evaluation and differentiation utilities.
namespace cxxgpr::detail
{

    namespace
    {

        /// @brief Visits elements of a square matrix triangle in column-parallel order.
        /// @details Uses area-balanced `linalg::parallel_for_lower_blocks` for compute-bound kernel loops
        /// (`n >= 128U`) on `Lower` and `Full` triangles, and OpenMP column splitting on `Upper`.
        /// @tparam T Storage floating-point scalar type.
        /// @tparam Fn Callable invoked as `fn(row, col)` for each selected coordinate.
        /// @param[in] n Matrix side length.
        /// @param[in] uplo Active matrix region (`Lower`, `Upper`, or `Full`).
        /// @param[in] fn Element visitor callback.
        template <KernelScalarConcept T, typename Fn> void visit_triangle(std::size_t n, Triangle uplo, Fn&& fn)
        {
            if (uplo == Triangle::Lower)
            {
                auto run_range = [&](std::size_t col_begin, std::size_t col_end)
                {
                    for (std::size_t col = col_begin; col < col_end; ++col)
                    {
                        for (std::size_t row = col; row < n; ++row)
                        {
                            fn(row, col);
                        }
                    }
                };
                if (n >= 128U)
                {
                    linalg::parallel_for_lower_blocks(n, 0U, run_range);
                }
                else
                {
                    run_range(0U, n);
                }
                return;
            }
            const auto n_cols = static_cast<std::ptrdiff_t>(n);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 128U)
#endif
            for (std::ptrdiff_t col_s = 0; col_s < n_cols; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                const std::size_t row_end = (uplo == Triangle::Upper) ? (col + 1U) : n;
                for (std::size_t row = 0U; row < row_end; ++row)
                {
                    fn(row, col);
                }
            }
        }

        /// @brief Executes a column-slice callback over an active square matrix triangle in parallel when beneficial.
        /// @details Processes `(row_begin, row_end)` per column so callers operate on unit-stride column-major slices,
        /// using area-balanced OpenMP blocks for `Lower` and `Full` triangles (`n >= 128U`).
        /// @tparam ColFn Callable invoked as `col_fn(col, row_begin, row_end)`.
        /// @param[in] n Square matrix side length.
        /// @param[in] uplo Active matrix region (`Lower`, `Upper`, or `Full`).
        /// @param[in] col_fn Column-slice callback.
        template <typename ColFn> void for_each_triangle_col(std::size_t n, Triangle uplo, ColFn&& col_fn)
        {
            if (uplo == Triangle::Lower || uplo == Triangle::Full)
            {
                auto run_range = [&](std::size_t col_begin, std::size_t col_end)
                {
                    for (std::size_t col = col_begin; col < col_end; ++col)
                    {
                        col_fn(col, col, n);
                    }
                };
                if (n >= 128U)
                {
                    linalg::parallel_for_lower_blocks(n, 0U, run_range);
                }
                else
                {
                    run_range(0U, n);
                }
                return;
            }
            // Triangle::Upper iterates contiguous upper column slices [0, col + 1).
            const auto n_cols = static_cast<std::ptrdiff_t>(n);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 128U)
#endif
            for (std::ptrdiff_t col_s = 0; col_s < n_cols; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                col_fn(col, 0U, col + 1U);
            }
        }

        /// @brief Mirrors strictly lower entries of a square column-major matrix into its upper triangle when `Full`.
        /// @details No-op for `Triangle::Lower` and `Triangle::Upper`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in,out] out Target square matrix view.
        /// @param[in] uplo Requested triangle mode (`Full` triggers mirroring).
        template <KernelScalarConcept T> void mirror_lower_if_full(MatrixMut<T> out, Triangle uplo) noexcept
        {
            if (uplo != Triangle::Full)
            {
                return;
            }
            const std::size_t n = out.rows();
            for (std::size_t col = 0U; col < n; ++col)
            {
                for (std::size_t row = col + 1U; row < n; ++row)
                {
                    const std::size_t r1 = row;
                    const std::size_t r2 = col;
                    out(r2, r1) = out(r1, r2);
                }
            }
        }

        /// @brief Invokes `fn(std::span<const T> weights)` with precomputed `extra_scale / (ls[k]^2)` weights.
        /// @details Uses a stack array of capacity 64 to keep all common ARD dimensions strictly allocation-free.
        /// @tparam T Storage floating-point scalar type.
        /// @tparam Fn Callable accepting `std::span<const T>` and returning `std::expected<void, GprError>`.
        /// @param[in] ls Per-dimension lengthscale span.
        /// @param[in] extra_scale Additional scalar factor folded into each inverse-squared weight.
        /// @param[in] fn Callback consuming the precomputed inverse-squared lengthscale weights.
        /// @return Result returned by `fn`.
        template <KernelScalarConcept T, typename Fn>
        [[nodiscard]] std::expected<void, GprError> with_ard_inv_ell2(
            std::span<const double> ls, T extra_scale, Fn&& fn
        )
        {
            constexpr std::size_t k_stack_cap = 64U;
            if (ls.size() <= k_stack_cap)
            {
                std::array<T, k_stack_cap> buf{};
                for (std::size_t d = 0U; d < ls.size(); ++d)
                {
                    buf[d] = extra_scale / static_cast<T>(ls[d] * ls[d]);
                }
                return fn(std::span<const T>(buf.data(), ls.size()));
            }
            std::vector<T> heap_buf(ls.size());
            for (std::size_t d = 0U; d < ls.size(); ++d)
            {
                heap_buf[d] = extra_scale / static_cast<T>(ls[d] * ls[d]);
            }
            return fn(std::span<const T>(heap_buf));
        }

        /// @brief Fills a contiguous column slice of weighted squared coordinate differences and transforms via
        /// `slice_fn`.
        /// @details Accumulates feature-by-feature using SSE2 slice helpers (`set_scaled_sq_diff_slice` /
        /// `add_scaled_sq_diff_slice`) over contiguous column memory, supporting isotropic weights or per-dim ARD.
        /// @tparam T Storage floating-point scalar type.
        /// @tparam SliceFn Callable accepting `(T* col_ptr, std::size_t len, std::size_t col, std::size_t row_begin)`.
        /// @param[in] x Input point coordinate matrix `(n, d)`.
        /// @param[out] out Target square matrix `(n, n)`.
        /// @param[in] uplo Active matrix region (`Lower`, `Upper`, or `Full`).
        /// @param[in] weights Per-dimension non-negative inverse-squared scale span, or empty for unit weights.
        /// @param[in] slice_fn In-place transformation applied to each populated contiguous column slice.
        template <KernelScalarConcept T, typename SliceFn>
        void eval_square_points_sq_dist_cols(
            MatrixRef<T> x, MatrixMut<T> out, Triangle uplo, std::span<const T> weights, SliceFn&& slice_fn
        )
        {
            const std::size_t n = x.rows();
            const std::size_t d = x.cols();
            const bool has_weights = !weights.empty();
            for_each_triangle_col(
                n, uplo, [&](std::size_t col, std::size_t row_begin, std::size_t row_end)
            {
                const std::size_t len = row_end - row_begin;
                if (len == 0U)
                {
                    return;
                }
                T* col_ptr = &out(row_begin, col);
                const T w0 = has_weights ? weights[0U] : T{1};
                detail::set_scaled_sq_diff_slice(col_ptr, x.col(0U).data() + row_begin, x(col, 0U), w0, len);
                for (std::size_t dim = 1U; dim < d; ++dim)
                {
                    const T wd = has_weights ? weights[dim] : T{1};
                    const auto col_vec = x.col(dim);
                    detail::add_scaled_sq_diff_slice(col_ptr, col_vec.data() + row_begin, col_vec[col], wd, len);
                }
                out(col, col) = T{0};
                slice_fn(col_ptr, len);
            }
            );
            mirror_lower_if_full(out, uplo);
        }

        /// @brief Fills rectangular cross squared coordinate differences column-by-column via SSE2 and applies
        /// `slice_fn`.
        /// @tparam T Storage floating-point scalar type.
        /// @tparam SliceFn Callable accepting `(T* col_ptr, std::size_t len)`.
        /// @param[in] x1 First point matrix `(n1, d)`.
        /// @param[in] x2 Second point matrix `(n2, d)`.
        /// @param[out] out Target cross matrix `(n1, n2)`.
        /// @param[in] weights Per-dimension inverse-squared weights, or empty for unweighted squared distances.
        /// @param[in] slice_fn In-place kernel transform applied to each column slice.
        template <KernelScalarConcept T, typename SliceFn>
        void eval_cross_points_sq_dist_cols(
            MatrixRef<T> x1, MatrixRef<T> x2, MatrixMut<T> out, std::span<const T> weights, SliceFn&& slice_fn
        )
        {
            const std::size_t n1 = x1.rows();
            const std::size_t n2 = x2.rows();
            const std::size_t d = x1.cols();
            const bool has_weights = !weights.empty();
            const auto cols_s = static_cast<std::ptrdiff_t>(n2);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if ((n1 * n2) >= 65536U)
#endif
            for (std::ptrdiff_t col_s = 0; col_s < cols_s; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                T* col_ptr = &out(0U, col);
                const T w0 = has_weights ? weights[0U] : T{1};
                detail::set_scaled_sq_diff_slice(col_ptr, x1.col(0U).data(), x2(col, 0U), w0, n1);
                for (std::size_t dim = 1U; dim < d; ++dim)
                {
                    const T wd = has_weights ? weights[dim] : T{1};
                    detail::add_scaled_sq_diff_slice(col_ptr, x1.col(dim).data(), x2.col(dim)[col], wd, n1);
                }
                slice_fn(col_ptr, n1);
            }
        }

        /// @brief Validates dimensions and finiteness of a square squared-distance matrix and output view.
        /// @details Enforces empty-input, squareness, shape-match, and active-triangle finiteness checks in
        /// deterministic order before leaf evaluation.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] dist Input squared Euclidean distance matrix.
        /// @param[in] out Target output matrix view.
        /// @param[in] uplo Triangle of `dist` and `out` that will be read and written.
        /// @return Empty expected on success, or an input/shape validation error.
        /// @retval GprErrorCode::EmptyInput If `dist` has zero rows or columns.
        /// @retval GprErrorCode::LengthMismatch If `dist` is non-square or `out` has a different shape.
        /// @retval GprErrorCode::NonFiniteInput If any entry in the active triangle of `dist` is non-finite.
        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> validate_square_dist(
            MatrixRef<T> dist, MatrixMut<T> out, Triangle uplo
        )
        {
            // Enforce deterministic validation order: empty -> squareness -> output shape -> finiteness.
            if (dist.rows() == 0 || dist.cols() == 0)
            {
                return std::unexpected(GprError::empty_input());
            }
            if (dist.rows() != dist.cols())
            {
                return std::unexpected(
                    GprError::length_mismatch("distance matrix for square evaluation must be square")
                );
            }
            if (out.rows() != dist.rows() || out.cols() != dist.cols())
            {
                return std::unexpected(GprError::length_mismatch("output matrix shape mismatch"));
            }
            // Scan only the active triangle so callers do not need to populate the opposite half.
            const std::size_t n = dist.rows();
            for (std::size_t col = 0; col < n; ++col)
            {
                const std::size_t r_begin = (uplo == Triangle::Lower) ? col : 0U;
                const std::size_t r_end = (uplo == Triangle::Upper) ? (col + 1U) : n;
                if (!detail::all_finite_slice(&dist(r_begin, col), r_end - r_begin))
                {
                    return std::unexpected(GprError::non_finite_input());
                }
            }
            return {};
        }

        /// @brief Validates dimensions and finiteness of a rectangular cross squared-distance matrix and output view.
        /// @details Checks non-emptiness, exact shape agreement with `out`, and finiteness of all entries in `dist`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] dist Input rectangular squared-distance matrix.
        /// @param[in] out Target cross-covariance output matrix view.
        /// @return Empty expected on success, or an input/shape validation error.
        /// @retval GprErrorCode::EmptyInput If `dist` has zero rows or columns.
        /// @retval GprErrorCode::LengthMismatch If `out` shape differs from `dist`.
        /// @retval GprErrorCode::NonFiniteInput If any entry of `dist` is non-finite.
        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> validate_cross_dist(MatrixRef<T> dist, MatrixMut<T> out)
        {
            // Check empty input and shape match before scanning every matrix entry for finiteness.
            if (dist.rows() == 0 || dist.cols() == 0)
            {
                return std::unexpected(GprError::empty_input());
            }
            if (out.rows() != dist.rows() || out.cols() != dist.cols())
            {
                return std::unexpected(GprError::length_mismatch("output matrix shape mismatch"));
            }
            if (dist.is_contiguous())
            {
                if (!detail::all_finite_slice(dist.data(), dist.rows() * dist.cols()))
                {
                    return std::unexpected(GprError::non_finite_input());
                }
                return {};
            }
            for (std::size_t col = 0; col < dist.cols(); ++col)
            {
                if (!detail::all_finite_slice(dist.col(col).data(), dist.rows()))
                {
                    return std::unexpected(GprError::non_finite_input());
                }
            }
            return {};
        }

        /// @brief Validates that a point-coordinate matrix is non-empty and finite.
        /// @details Iterates in column-major order for contiguous memory access.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] x Input point matrix of shape `(n_points, n_dims)`.
        /// @return Empty expected on success, or an input validation error.
        /// @retval GprErrorCode::EmptyInput If `x` has zero rows or columns.
        /// @retval GprErrorCode::NonFiniteInput If any coordinate in `x` is NaN or infinite.
        template <KernelScalarConcept T> [[nodiscard]] std::expected<void, GprError> validate_points(MatrixRef<T> x)
        {
            if (x.rows() == 0 || x.cols() == 0)
            {
                return std::unexpected(GprError::empty_input());
            }
            if (x.is_contiguous())
            {
                if (!detail::all_finite_slice(x.data(), x.rows() * x.cols()))
                {
                    return std::unexpected(GprError::non_finite_input());
                }
                return {};
            }
            // Scan in column-major order matching MatrixRef storage layout.
            for (std::size_t col = 0; col < x.cols(); ++col)
            {
                if (!detail::all_finite_slice(x.col(col).data(), x.rows()))
                {
                    return std::unexpected(GprError::non_finite_input());
                }
            }
            return {};
        }

        /// @brief Validates a point matrix and square output buffer for self-covariance or derivative evaluation.
        /// @details Checks non-emptiness, verifies `out` is `x.rows() x x.rows()`, and delegates finiteness checks.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] x Input point matrix of shape `(n, d)`.
        /// @param[in] out Target square output matrix view of shape `(n, n)`.
        /// @return Empty expected on success, or an input/shape validation error.
        /// @retval GprErrorCode::EmptyInput If `x` has zero rows or columns.
        /// @retval GprErrorCode::LengthMismatch If `out` is not `(x.rows(), x.rows())`.
        /// @retval GprErrorCode::NonFiniteInput If any entry of `x` is non-finite.
        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> validate_square_points(MatrixRef<T> x, MatrixMut<T> out)
        {
            if (x.rows() == 0 || x.cols() == 0)
            {
                return std::unexpected(GprError::empty_input());
            }
            if (out.rows() != x.rows() || out.cols() != x.rows())
            {
                return std::unexpected(GprError::length_mismatch("output matrix shape mismatch"));
            }
            return validate_points(x);
        }

        /// @brief Validates two point matrices and a cross-covariance output matrix.
        /// @details Enforces empty checks, `(x1.rows(), x2.rows())` output shape check, feature-dimension match, and
        /// finiteness of both coordinate matrices.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] x1 First point matrix of shape `(n1, d)`.
        /// @param[in] x2 Second point matrix of shape `(n2, d)`.
        /// @param[in] out Target cross-covariance matrix view of shape `(n1, n2)`.
        /// @return Empty expected on success, or an input/dimension/shape validation error.
        /// @retval GprErrorCode::EmptyInput If `x1` or `x2` has zero rows or columns.
        /// @retval GprErrorCode::LengthMismatch If `out` shape differs from `(x1.rows(), x2.rows())`.
        /// @retval GprErrorCode::DimensionMismatch If `x1.cols() != x2.cols()`.
        /// @retval GprErrorCode::NonFiniteInput If any coordinate in `x1` or `x2` is non-finite.
        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> validate_cross_points(
            MatrixRef<T> x1, MatrixRef<T> x2, MatrixMut<T> out
        )
        {
            // Validate empty inputs, output shape, and feature count before scanning matrix entries.
            if (x1.rows() == 0 || x1.cols() == 0 || x2.rows() == 0 || x2.cols() == 0)
            {
                return std::unexpected(GprError::empty_input());
            }
            if (out.rows() != x1.rows() || out.cols() != x2.rows())
            {
                return std::unexpected(GprError::length_mismatch("output matrix shape mismatch"));
            }
            if (x1.cols() != x2.cols())
            {
                return std::unexpected(GprError::dimension_mismatch(x2.cols(), x1.cols()));
            }
            if (auto v = validate_points(x1); !v.has_value())
            {
                return v;
            }
            return validate_points(x2);
        }

        /// @brief Computes the squared Euclidean distance between two point rows.
        /// @details Accumulates $\sum_d (x_1[\text{row1}, d] - x_2[\text{row2}, d])^2$ in scalar `T`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] x1 First point coordinate matrix.
        /// @param[in] row1 Row index in `x1`.
        /// @param[in] x2 Second point coordinate matrix.
        /// @param[in] row2 Row index in `x2`.
        /// @return Squared Euclidean distance between the two points.
        template <KernelScalarConcept T>
        [[nodiscard]] T sq_dist_points(MatrixRef<T> x1, std::size_t row1, MatrixRef<T> x2, std::size_t row2) noexcept
        {
            // Sum squared per-coordinate differences across all feature dimensions.
            T acc = T{0};
            for (std::size_t d = 0; d < x1.cols(); ++d)
            {
                const T diff = x1(row1, d) - x2(row2, d);
                acc += diff * diff;
            }
            return acc;
        }

        /// @brief Materializes a full column-major square squared-distance buffer from points.
        /// @details Used as a fallback when evaluating a `CustomKernel` leaf directly from raw coordinates.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] x Input point matrix of shape `(n, d)`.
        /// @return Dense column-major `n x n` buffer of pairwise squared Euclidean distances.
        template <KernelScalarConcept T> [[nodiscard]] std::vector<T> make_square_dist_buf(MatrixRef<T> x)
        {
            // Build a dense n x n column-major squared-distance matrix for CustomKernel fallback evaluation.
            const std::size_t n = x.rows();
            std::vector<T> buf(n * n, T{0});
            for (std::size_t col = 0; col < n; ++col)
            {
                for (std::size_t row = 0; row < n; ++row)
                {
                    buf[(col * n) + row] = sq_dist_points(x, row, x, col);
                }
            }
            return buf;
        }

        /// @brief Materializes a dense column-major cross squared-distance buffer from two point matrices.
        /// @details Used when evaluating a `CustomKernel` cross-covariance directly from raw coordinates.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] x1 First point matrix of shape `(n1, d)`.
        /// @param[in] x2 Second point matrix of shape `(n2, d)`.
        /// @return Dense column-major `n1 x n2` buffer of pairwise squared Euclidean distances.
        template <KernelScalarConcept T>
        [[nodiscard]] std::vector<T> make_cross_dist_buf(MatrixRef<T> x1, MatrixRef<T> x2)
        {
            // Build a dense n1 x n2 column-major cross squared-distance matrix for CustomKernel evaluation.
            const std::size_t n1 = x1.rows();
            const std::size_t n2 = x2.rows();
            std::vector<T> buf(n1 * n2, T{0});
            for (std::size_t col = 0; col < n2; ++col)
            {
                for (std::size_t row = 0; row < n1; ++row)
                {
                    buf[(col * n1) + row] = sq_dist_points(x1, row, x2, col);
                }
            }
            return buf;
        }

        /// @brief Computes the Euclidean dot product between two point rows.
        /// @details Evaluates $\sum_d x_1[\text{row1}, d] \cdot x_2[\text{row2}, d]$ for linear kernels.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] x1 First point coordinate matrix.
        /// @param[in] row1 Row index in `x1`.
        /// @param[in] x2 Second point coordinate matrix.
        /// @param[in] row2 Row index in `x2`.
        /// @return Dot product of the two feature vectors.
        template <KernelScalarConcept T>
        [[nodiscard]] T dot_points(MatrixRef<T> x1, std::size_t row1, MatrixRef<T> x2, std::size_t row2) noexcept
        {
            // Accumulate inner product across all input dimensions.
            T acc = T{0};
            for (std::size_t d = 0; d < x1.cols(); ++d)
            {
                acc += x1(row1, d) * x2(row2, d);
            }
            return acc;
        }

        /// @brief Computes the ARD-scaled squared distance between two point rows.
        /// @details Evaluates $s^2 = \sum_d ((x_1[\text{row1}, d] - x_2[\text{row2}, d]) / \ell_d)^2$.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] x1 First point coordinate matrix.
        /// @param[in] row1 Row index in `x1`.
        /// @param[in] x2 Second point coordinate matrix.
        /// @param[in] row2 Row index in `x2`.
        /// @param[in] ls Per-dimension lengthscale vector in user units.
        /// @return ARD-scaled squared distance $s^2$.
        template <KernelScalarConcept T>
        [[nodiscard]] T ard_scaled_sq(
            MatrixRef<T> x1, std::size_t row1, MatrixRef<T> x2, std::size_t row2, std::span<const double> ls
        ) noexcept
        {
            // Normalize each coordinate difference by its dimension-specific lengthscale before squaring.
            T acc = T{0};
            for (std::size_t d = 0; d < ls.size(); ++d)
            {
                const T ell = static_cast<T>(ls[d]);
                const T diff = (x1(row1, d) - x2(row2, d)) / ell;
                acc += diff * diff;
            }
            return acc;
        }

        /// @brief Computes ARD-scaled squared distance $s^2$ using precomputed inverse-squared lengthscales.
        /// @details Multiplies each squared coordinate difference by `inv_ell2[d] = 1 / ell_d^2` to avoid inner-loop
        /// floating-point divisions across `(n, n)` evaluations.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] x1 First point coordinate matrix.
        /// @param[in] row1 Row index in `x1`.
        /// @param[in] x2 Second point coordinate matrix.
        /// @param[in] row2 Row index in `x2`.
        /// @param[in] inv_ell2 Precomputed per-dimension inverse-squared lengthscales.
        /// @return ARD-scaled squared distance $s^2$.
        template <KernelScalarConcept T>
        [[nodiscard]] T ard_scaled_sq_weights(
            MatrixRef<T> x1, std::size_t row1, MatrixRef<T> x2, std::size_t row2, std::span<const T> inv_ell2
        ) noexcept
        {
            T acc = T{0};
            for (std::size_t d = 0; d < inv_ell2.size(); ++d)
            {
                const T diff = x1(row1, d) - x2(row2, d);
                acc += (diff * diff) * inv_ell2[d];
            }
            return acc;
        }

        /// @brief Invokes a callback with per-dimension lengthscales without heap allocation when cached.
        /// @details Uses `cached_ls` directly when non-empty (hot path) and falls back to `k.lengthscales()` only for
        /// standalone uncached calls.
        /// @tparam KernelT ARD kernel leaf type exposing `.lengthscales()`.
        /// @tparam Fn Callable accepting `std::span<const double>` and returning `std::expected<void, GprError>`.
        /// @param[in] k ARD kernel leaf instance.
        /// @param[in] cached_ls Pre-cached lengthscale span (empty if uncached).
        /// @param[in] fn Callback consuming the lengthscale span.
        /// @return Result returned by `fn`.
        template <typename KernelT, typename Fn>
        [[nodiscard]] std::expected<void, GprError> with_ard_ls(
            const KernelT& k, std::span<const double> cached_ls, Fn&& fn
        )
        {
            // Avoid heap allocation on compiled hot paths by reading pre-cached lengthscales when provided.
            if (!cached_ls.empty())
            {
                return fn(cached_ls);
            }
            const auto fallback = k.lengthscales();
            return fn(std::span<const double>(fallback));
        }

        /// @brief Returns the Matern radial scaling factor $\sqrt{2\nu}$.
        /// @details Maps `OneHalf -> 1`, `ThreeHalves -> sqrt(3)`, and `FiveHalves -> sqrt(5)`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] nu Matern smoothness parameter.
        /// @return Radial multiplier $\sqrt{2\nu}$.
        template <KernelScalarConcept T> [[nodiscard]] constexpr T matern_scale(MaternNu nu) noexcept
        {
            switch (nu)
            {
                case MaternNu::OneHalf:
                    return T{1};
                case MaternNu::ThreeHalves:
                    return std::numbers::sqrt3_v<T>;
                case MaternNu::FiveHalves:
                    return static_cast<T>(2.23606797749978969641);
            }
            return T{1};
        }

        /// @brief Returns the squared Matern radial scaling factor $2\nu$.
        /// @details Maps `OneHalf -> 1`, `ThreeHalves -> 3`, and `FiveHalves -> 5`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] nu Matern smoothness parameter.
        /// @return Squared radial factor $2\nu$.
        template <KernelScalarConcept T> [[nodiscard]] constexpr T matern_scale_sq(MaternNu nu) noexcept
        {
            switch (nu)
            {
                case MaternNu::OneHalf:
                    return T{1};
                case MaternNu::ThreeHalves:
                    return T{3};
                case MaternNu::FiveHalves:
                    return T{5};
            }
            return T{1};
        }

        /// @brief Evaluates an isotropic RBF kernel from squared distance $d^2$ and $1/(2\ell^2)$.
        /// @details Clamps negative `d2` (from numerical rounding or slight repair artifacts) to zero before applying
        /// `eval_kernel_exp`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] d2 Squared Euclidean distance.
        /// @param[in] inv_two_ell2 Precomputed factor $1 / (2 \ell^2)$.
        /// @param[in] exp_mode Exponential evaluation backend.
        /// @return Kernel value $\exp(-\max(d^2, 0) / (2 \ell^2))$.
        template <KernelScalarConcept T> [[nodiscard]] T eval_rbf_d2(T d2, T inv_two_ell2, KernelExp exp_mode) noexcept
        {
            // Clamp negative rounding noise so the exponent stays non-positive and kernel values stay <= 1.
            const T clamped = d2 < T{0} ? T{0} : d2;
            return eval_kernel_exp(-clamped * inv_two_ell2, exp_mode);
        }

        /// @brief Radial Matern value and log-lengthscale derivative coefficients at scaled radius $u$.
        /// @details Packages $k(u)$ alongside radial first- and second-order log-hyperparameter factors so both
        /// isotropic and ARD Matern formulas reuse consistent exact and `FastApprox` derivatives.
        /// @tparam T Storage floating-point scalar type.
        template <KernelScalarConcept T> struct MaternJetVals
        {
            /// @brief Kernel value $k(u)$.
            T val = T{0};
            /// @brief First-order log-lengthscale radial coefficient $-u k'(u)$.
            T g1 = T{0};
            /// @brief Isotropic second derivative wrt $\log \ell$, equal to $u k'(u) + u^2 k''(u)$.
            T iso_hess = T{0};
            /// @brief ARD second-order radial cross factor $u^2 k''(u) - u k'(u)$.
            T ard_cross = T{0};
        };

        /// @brief Evaluates Matern kernel value and log-lengthscale derivative factors at scaled radius $u$.
        /// @details For $u \le 0$, returns the continuous $r \to 0$ limit `{1, 0, 0, 0}` to avoid $0/0$ singularities.
        /// Under `FastApprox`, propagates the polynomial-times-exponential derivatives using `ExpJet<T>` of $\exp(-u)$.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] u Scaled radius $u = \sqrt{2\nu}\, r / \ell$ (isotropic) or $\sqrt{2\nu\, s^2}$ (ARD).
        /// @param[in] nu Matern smoothness parameter (`OneHalf`, `ThreeHalves`, or `FiveHalves`).
        /// @param[in] exp_mode Exponential evaluation backend.
        /// @return `MaternJetVals<T>` containing value, first derivative, isotropic Hessian, and ARD cross factor.
        template <KernelScalarConcept T>
        [[nodiscard]] MaternJetVals<T> eval_matern_jet_vals(T u, MaternNu nu, KernelExp exp_mode) noexcept
        {
            // At zero distance (diagonal or duplicate points), radial log-theta derivatives vanish smoothly.
            if (u <= T{0})
            {
                return {.val = T{1}, .g1 = T{0}, .iso_hess = T{0}, .ard_cross = T{0}};
            }
            if (exp_mode == KernelExp::Accurate)
            {
                // Closed-form radial derivatives under theta = log(ell) where du/dtheta = -u:
                // g1 = -u*k'(u), iso_hess = u*k'(u) + u^2*k''(u), ard_cross = u^2*k''(u) - u*k'(u).
                const T e = std::exp(-u);
                switch (nu)
                {
                    case MaternNu::OneHalf:
                        // k(u) = exp(-u) => g1 = u*e, iso_hess = u*(u-1)*e, ard_cross = u*(u+1)*e.
                        return {.val = e, .g1 = u * e, .iso_hess = u * (u - T{1}) * e, .ard_cross = u * (u + T{1}) * e};
                    case MaternNu::ThreeHalves:
                        // k(u) = (1+u)*exp(-u) => g1 = u^2*e, iso_hess = u^2*(u-2)*e, ard_cross = u^3*e.
                        return {
                            .val = (T{1} + u) * e,
                            .g1 = (u * u) * e,
                            .iso_hess = (u * u) * (u - T{2}) * e,
                            .ard_cross = (u * u * u) * e
                        };
                    case MaternNu::FiveHalves:
                        // k(u) = (1 + u + u^2/3)*exp(-u) => g1 = u^2*(1+u)/3 * e.
                        return {
                            .val = (T{1} + u + ((u * u) / T{3})) * e,
                            .g1 = (((u * u) * (T{1} + u)) / T{3}) * e,
                            .iso_hess = (((u * u) * ((u * u) - (T{2} * u) - T{2})) / T{3}) * e,
                            .ard_cross = ((u * u * u * u) / T{3}) * e
                        };
                }
            }
            // Under FastApprox, differentiate p(u) * exp_approx(-u) via product rule using exp(-u) jet.
            const ExpJet<T> jet = eval_kernel_exp_jet(-u, exp_mode);
            T val = jet.v;
            T fp = -jet.d1;
            T fpp = jet.d2;
            if (nu == MaternNu::ThreeHalves)
            {
                val = (T{1} + u) * jet.v;
                fp = jet.v - ((T{1} + u) * jet.d1);
                fpp = ((T{1} + u) * jet.d2) - (T{2} * jet.d1);
            }
            else if (nu == MaternNu::FiveHalves)
            {
                const T poly = T{1} + u + ((u * u) / T{3});
                const T dpoly = T{1} + ((T{2} * u) / T{3});
                val = poly * jet.v;
                fp = (dpoly * jet.v) - (poly * jet.d1);
                fpp = ((T{2} / T{3}) * jet.v) - (T{2} * dpoly * jet.d1) + (poly * jet.d2);
            }
            const T g1 = -u * fp;
            return {.val = val, .g1 = g1, .iso_hess = (u * fp) + (u * u * fpp), .ard_cross = (u * u * fpp) - (u * fp)};
        }

        /// @brief Evaluates the scalar Matern kernel value at scaled radius $u$.
        /// @details Returns 1 for $u \le 0$ and evaluates the order-specific polynomial times $\exp(-u)$ otherwise.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] u Scaled radius $u \ge 0$.
        /// @param[in] nu Matern smoothness parameter.
        /// @param[in] exp_mode Exponential evaluation backend.
        /// @return Matern kernel value.
        template <KernelScalarConcept T> [[nodiscard]] T eval_matern_u(T u, MaternNu nu, KernelExp exp_mode) noexcept
        {
            if (u <= T{0})
            {
                return T{1};
            }
            const T e = eval_kernel_exp(-u, exp_mode);
            switch (nu)
            {
                case MaternNu::OneHalf:
                    return e;
                case MaternNu::ThreeHalves:
                    return (T{1} + u) * e;
                case MaternNu::FiveHalves:
                    return (T{1} + u + ((u * u) / T{3})) * e;
            }
            return e;
        }

        /// @brief Evaluates the first-order Matern radial log-lengthscale factor $-u k'(u)$.
        /// @details Delegates to `eval_matern_jet_vals` so `Accurate` and `FastApprox` derivatives stay consistent.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] u Scaled radius $u \ge 0$.
        /// @param[in] nu Matern smoothness parameter.
        /// @param[in] exp_mode Exponential evaluation backend.
        /// @return First radial log-lengthscale derivative factor.
        template <KernelScalarConcept T>
        [[nodiscard]] T eval_matern_grad_g1(T u, MaternNu nu, KernelExp exp_mode) noexcept
        {
            return eval_matern_jet_vals(u, nu, exp_mode).g1;
        }

        /// @brief Evaluates the isotropic Matern second derivative $\partial^2 k / \partial (\log \ell)^2$.
        /// @details Delegates to `eval_matern_jet_vals` and returns $u k'(u) + u^2 k''(u)$.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] u Scaled radius $u \ge 0$.
        /// @param[in] nu Matern smoothness parameter.
        /// @param[in] exp_mode Exponential evaluation backend.
        /// @return Isotropic Matern second log-lengthscale derivative.
        template <KernelScalarConcept T>
        [[nodiscard]] T eval_matern_iso_hess(T u, MaternNu nu, KernelExp exp_mode) noexcept
        {
            return eval_matern_jet_vals(u, nu, exp_mode).iso_hess;
        }

        /// @brief Evaluates the ARD Matern second-derivative radial cross coefficient $u^2 k''(u) - u k'(u)$.
        /// @details Multiplied by dimension weights $w_i w_j$ in `leaf_hess_points` and `leaf_hess_ard_packed`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] u Scaled ARD radius $u = \sqrt{2\nu\, s^2}$.
        /// @param[in] nu Matern smoothness parameter.
        /// @param[in] exp_mode Exponential evaluation backend.
        /// @return Radial cross coefficient for ARD Matern Hessians.
        template <KernelScalarConcept T>
        [[nodiscard]] T eval_matern_ard_cross_coeff(T u, MaternNu nu, KernelExp exp_mode) noexcept
        {
            return eval_matern_jet_vals(u, nu, exp_mode).ard_cross;
        }

        /// @brief Evaluates the isotropic periodic kernel from squared distance $d^2$.
        /// @details Computes $k = \exp(-2 \sin^2(\pi r / p) / \ell^2)$ with $r = \sqrt{\max(d^2, 0)}$.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] d2 Squared Euclidean distance.
        /// @param[in] ell Lengthscale parameter $\ell > 0$.
        /// @param[in] period Period parameter $p > 0$.
        /// @param[in] exp_mode Exponential evaluation backend.
        /// @return Periodic kernel value.
        template <KernelScalarConcept T>
        [[nodiscard]] T eval_periodic_d2(T d2, T ell, T period, KernelExp exp_mode) noexcept
        {
            // Guard sqrt against negative rounding noise before computing phase angle phi = pi * r / period.
            const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
            const T phi = (std::numbers::pi_v<T> * r) / period;
            const T s = std::sin(phi);
            const T z = (T{2} * s * s) / (ell * ell);
            return eval_kernel_exp(-z, exp_mode);
        }

        /// @brief Evaluates the isotropic rational-quadratic kernel from squared distance $d^2$.
        /// @details Computes $(1 + w)^{-\alpha} = \exp(-\alpha \operatorname{log1p}(w))$ where
        /// $w = \max(d^2, 0) / (2 \alpha \ell^2)$.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] d2 Squared Euclidean distance.
        /// @param[in] ell Lengthscale parameter $\ell > 0$.
        /// @param[in] alpha Scale-mixture shape parameter $\alpha > 0$.
        /// @param[in] exp_mode Exponential evaluation backend.
        /// @return Rational-quadratic kernel value.
        template <KernelScalarConcept T> [[nodiscard]] T eval_rq_d2(T d2, T ell, T alpha, KernelExp exp_mode) noexcept
        {
            // Use log1p(w) for numerical stability when w = d^2 / (2 * alpha * ell^2) is small.
            const T clamped = d2 < T{0} ? T{0} : d2;
            const T w = clamped / (T{2} * alpha * ell * ell);
            return eval_kernel_exp(-alpha * std::log1p(w), exp_mode);
        }

        /// @brief Populates a square output matrix triangle from a distance matrix using `elem_fn`.
        /// @details Delegates column-parallel iteration to `visit_triangle`.
        /// @tparam T Storage floating-point scalar type.
        /// @tparam ElemFn Callable invoked as `elem_fn(dist(row, col), row, col)`.
        /// @param[in] dist Input squared-distance matrix.
        /// @param[out] out Target square output matrix.
        /// @param[in] uplo Active matrix triangle to fill.
        /// @param[in] elem_fn Element evaluation functor.
        template <KernelScalarConcept T, typename ElemFn>
        void write_square_dist(MatrixRef<T> dist, MatrixMut<T> out, Triangle uplo, ElemFn&& elem_fn)
        {
            visit_triangle<T>(dist.rows(), uplo, [&](std::size_t row, std::size_t col) {
                out(row, col) = elem_fn(dist(row, col), row, col);
            });
        }

        /// @brief Populates a rectangular cross output matrix from a distance matrix using `elem_fn`.
        /// @details Parallelizes across columns with OpenMP when column count or total element count crosses threshold.
        /// @tparam T Storage floating-point scalar type.
        /// @tparam ElemFn Callable invoked as `elem_fn(dist(row, col), row, col)`.
        /// @param[in] dist Input rectangular squared-distance matrix.
        /// @param[out] out Target rectangular output matrix.
        /// @param[in] elem_fn Element evaluation functor.
        template <KernelScalarConcept T, typename ElemFn>
        void write_cross_dist(MatrixRef<T> dist, MatrixMut<T> out, ElemFn&& elem_fn)
        {
            const std::size_t n_rows = dist.rows();
            const std::size_t n_cols = dist.cols();
            const auto cols_s = static_cast<std::ptrdiff_t>(n_cols);
            // Parallelize across disjoint columns only when matrix area outweighs OpenMP fork/join overhead.
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if ((n_rows * n_cols) >= 65536U)
#endif
            for (std::ptrdiff_t col_s = 0; col_s < cols_s; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                for (std::size_t row = 0; row < n_rows; ++row)
                {
                    out(row, col) = elem_fn(dist(row, col), row, col);
                }
            }
        }

        /// @brief Populates a square output matrix triangle from point coordinates via `elem_fn`.
        /// @details Delegates column-parallel iteration over `uplo` to `visit_triangle`.
        /// @tparam T Storage floating-point scalar type.
        /// @tparam ElemFn Callable invoked as `elem_fn(row, col)`.
        /// @param[in] x Input point matrix.
        /// @param[out] out Target square output matrix.
        /// @param[in] uplo Active matrix triangle to fill.
        /// @param[in] elem_fn Element evaluation functor.
        template <KernelScalarConcept T, typename ElemFn>
        void write_square_points(MatrixRef<T> x, MatrixMut<T> out, Triangle uplo, ElemFn&& elem_fn)
        {
            visit_triangle<T>(x.rows(), uplo, [&](std::size_t row, std::size_t col) {
                out(row, col) = elem_fn(row, col);
            });
        }

        /// @brief Populates a rectangular cross output matrix from two point matrices via `elem_fn`.
        /// @details Parallelizes across columns with OpenMP when matrix area crosses threshold.
        /// @tparam T Storage floating-point scalar type.
        /// @tparam ElemFn Callable invoked as `elem_fn(row, col)`.
        /// @param[in] x1 First point matrix of shape `(n1, d)`.
        /// @param[in] x2 Second point matrix of shape `(n2, d)`.
        /// @param[out] out Target rectangular output matrix of shape `(n1, n2)`.
        /// @param[in] elem_fn Element evaluation functor.
        template <KernelScalarConcept T, typename ElemFn>
        void write_cross_points(MatrixRef<T> x1, MatrixRef<T> x2, MatrixMut<T> out, ElemFn&& elem_fn)
        {
            const std::size_t n_rows = x1.rows();
            const std::size_t n_cols = x2.rows();
            const auto cols_s = static_cast<std::ptrdiff_t>(n_cols);
            // Parallelize over columns so each thread writes contiguous column-major memory.
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if ((n_rows * n_cols) >= 65536U)
#endif
            for (std::ptrdiff_t col_s = 0; col_s < cols_s; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                for (std::size_t row = 0; row < n_rows; ++row)
                {
                    out(row, col) = elem_fn(row, col);
                }
            }
        }

        /// @brief Applies `periodic` radial transformation in place over contiguous column slice `col_ptr[0..len)`.
        /// @details Converts squared distance `col_ptr[k] = d^2` into `-2 * sin^2(pi * sqrt(d^2) / period) / ell^2`
        /// and evaluates vectorized exponentials via `exp_scaled_nonpos_in_place`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in,out] col_ptr Contiguous column slice holding non-negative squared distances.
        /// @param[in] len Number of elements in the column slice.
        /// @param[in] inv_two_ell2_coeff Precomputed negative multiplier `-2 / ell^2`.
        /// @param[in] pi_over_period Precomputed constant `pi / period`.
        /// @param[in] exp_mode Exponential evaluation backend.
        template <KernelScalarConcept T>
        void transform_periodic_d2_slice_in_place(
            T* col_ptr, std::size_t len, T inv_two_ell2_coeff, T pi_over_period, KernelExp exp_mode
        ) noexcept
        {
            for (std::size_t k = 0U; k < len; ++k)
            {
                const T d2 = col_ptr[k];
                const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
                const T s = std::sin(pi_over_period * r);
                col_ptr[k] = s * s;
            }
            detail::exp_scaled_nonpos_in_place(col_ptr, len, inv_two_ell2_coeff, exp_mode);
        }

        /// @brief Applies `rational_quadratic` radial transformation in place over contiguous slice `col_ptr[0..len)`.
        /// @details Assumes `col_ptr[k]` initially holds scaled variable `w = s^2 / (2 * alpha)` (for ARD) or
        /// `d^2 * w_scale` (for isotropic) and overwrites with `(1 + w)^(-alpha) = exp(-alpha * log1p(w))`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in,out] col_ptr Contiguous column slice holding `w >= 0` (or unscaled `d^2` multiplied by `w_scale`).
        /// @param[in] len Number of elements in the column slice.
        /// @param[in] w_scale Scale factor converting raw `d^2` to `w` (pass `T{1}` when already scaled).
        /// @param[in] alpha Rational quadratic shape parameter `alpha > 0`.
        /// @param[in] exp_mode Exponential evaluation backend.
        template <KernelScalarConcept T>
        void transform_rq_slice_in_place(T* col_ptr, std::size_t len, T w_scale, T alpha, KernelExp exp_mode) noexcept
        {
            for (std::size_t k = 0U; k < len; ++k)
            {
                const T d2 = col_ptr[k] < T{0} ? T{0} : col_ptr[k];
                col_ptr[k] = std::log1p(d2 * w_scale);
            }
            detail::exp_scaled_nonpos_in_place(col_ptr, len, -alpha, exp_mode);
        }

        /// @brief Bounds-checks a local kernel hyperparameter index against `count`.
        /// @details Returns `GprErrorCode::IndexOutOfRange` when `idx >= count`.
        /// @param[in] idx Requested local hyperparameter index.
        /// @param[in] count Total parameter count for the leaf.
        /// @return Empty expected if valid, or index-out-of-range error.
        /// @retval GprErrorCode::IndexOutOfRange If `idx >= count`.
        [[nodiscard]] std::expected<void, GprError> require_param_index(std::size_t idx, std::size_t count)
        {
            if (idx < count)
            {
                return {};
            }
            return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
        }

        /// @brief Verifies that point matrix feature dimensionality matches an ARD kernel leaf.
        /// @details Returns `GprErrorCode::DimensionMismatch` when `x_dim != expected`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] x_dim Column count of the input point matrix.
        /// @param[in] expected Lengthscale count expected by the ARD leaf.
        /// @return Empty expected on match, or dimension-mismatch error.
        /// @retval GprErrorCode::DimensionMismatch If `x_dim != expected`.
        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> check_ard_dim(std::size_t x_dim, std::size_t expected)
        {
            if (x_dim != expected)
            {
                return std::unexpected(GprError::dimension_mismatch(x_dim, expected));
            }
            return {};
        }

        // --- Dispatch helpers for each operation ---

        /// @brief Evaluates a single kernel leaf over a square squared-distance matrix.
        /// @details Dispatches statically across isotropic distance-capable leaves and dynamically for `CustomKernel`;
        /// rejects coordinate-only leaves (`RbfArdKernel`, `MaternArdKernel`, `RationalQuadraticArdKernel`,
        /// `LinearKernel`).
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] leaf Kernel leaf variant to evaluate.
        /// @param[in] dist Square squared-distance matrix.
        /// @param[out] out Target square Gram matrix.
        /// @param[in] uplo Active matrix triangle to populate.
        /// @param[in] exp_mode Exponential evaluation backend.
        /// @return Empty expected on success, or validation/operation error.
        /// @retval GprErrorCode::UnsupportedKernelOperation If `leaf` requires point coordinates.
        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> eval_leaf_dist(
            const KernelLeaf& leaf, MatrixRef<T> dist, MatrixMut<T> out, Triangle uplo, KernelExp exp_mode
        )
        {
            // Validate distance matrix shape and active-triangle finiteness before dispatching.
            if (auto v = validate_square_dist(dist, out, uplo); !v.has_value())
            {
                return v;
            }
            return std::visit(
                [&](const auto& k) -> std::expected<void, GprError>
            {
                using LeafT = std::decay_t<decltype(k)>;
                if constexpr (std::is_same_v<LeafT, RbfKernel>)
                {
                    const T ell = static_cast<T>(k.lengthscale());
                    const T neg_inv_2l2 = -T{1} / (T{2} * ell * ell);
                    for_each_triangle_col(
                        dist.rows(), uplo, [&](std::size_t col, std::size_t row_begin, std::size_t row_end)
                    {
                        const std::size_t len = row_end - row_begin;
                        std::copy_n(&dist(row_begin, col), len, &out(row_begin, col));
                        detail::exp_scaled_nonpos_in_place(&out(row_begin, col), len, neg_inv_2l2, exp_mode);
                    }
                    );
                    mirror_lower_if_full(out, uplo);
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, MaternKernel>)
                {
                    const T scale_sq = matern_scale_sq<T>(k.nu()) / static_cast<T>(k.lengthscale() * k.lengthscale());
                    const MaternNu nu = k.nu();
                    for_each_triangle_col(
                        dist.rows(), uplo, [&](std::size_t col, std::size_t row_begin, std::size_t row_end)
                    {
                        const std::size_t len = row_end - row_begin;
                        std::copy_n(&dist(row_begin, col), len, &out(row_begin, col));
                        detail::matern_s2_in_place(&out(row_begin, col), len, scale_sq, nu, exp_mode);
                    }
                    );
                    mirror_lower_if_full(out, uplo);
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, PeriodicKernel>)
                {
                    const T ell = static_cast<T>(k.lengthscale());
                    const T period = static_cast<T>(k.period());
                    const T neg_coeff = -T{2} / (ell * ell);
                    const T pi_p = std::numbers::pi_v<T> / period;
                    for_each_triangle_col(
                        dist.rows(), uplo, [&](std::size_t col, std::size_t row_begin, std::size_t row_end)
                    {
                        const std::size_t len = row_end - row_begin;
                        std::copy_n(&dist(row_begin, col), len, &out(row_begin, col));
                        transform_periodic_d2_slice_in_place(&out(row_begin, col), len, neg_coeff, pi_p, exp_mode);
                    }
                    );
                    mirror_lower_if_full(out, uplo);
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, RationalQuadraticKernel>)
                {
                    const T ell = static_cast<T>(k.lengthscale());
                    const T alpha = static_cast<T>(k.alpha());
                    const T w_scale = T{1} / (T{2} * alpha * ell * ell);
                    for_each_triangle_col(
                        dist.rows(), uplo, [&](std::size_t col, std::size_t row_begin, std::size_t row_end)
                    {
                        const std::size_t len = row_end - row_begin;
                        std::copy_n(&dist(row_begin, col), len, &out(row_begin, col));
                        transform_rq_slice_in_place(&out(row_begin, col), len, w_scale, alpha, exp_mode);
                    }
                    );
                    mirror_lower_if_full(out, uplo);
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, ConstantKernel>)
                {
                    const T c = static_cast<T>(k.constant());
                    write_square_dist(dist, out, uplo, [c](T, std::size_t, std::size_t) { return c; });
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, WhiteKernel>)
                {
                    const T v = static_cast<T>(k.variance());
                    write_square_dist(dist, out, uplo, [v](T, std::size_t row, std::size_t col) {
                        return row == col ? v : T{0};
                    });
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, CustomKernel>)
                {
                    return k.template term<T>().apply(dist, out, uplo);
                }
                else
                {
                    // Reject ARD and Linear leaves because they depend on per-coordinate structure.
                    return std::unexpected(
                        GprError::unsupported_kernel_operation(
                            "coordinate-dependent leaf cannot be evaluated from squared distances"
                        )
                    );
                }
            }, leaf
            );
        }

        /// @brief Evaluates a single kernel leaf directly from point coordinates into a square Gram triangle.
        /// @details Handles all 10 built-in leaf types (including ARD and `LinearKernel`) and falls back to
        /// materializing a temporary distance buffer for `CustomKernel`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] leaf Kernel leaf variant to evaluate.
        /// @param[in] x Input point matrix of shape `(n, d)`.
        /// @param[out] out Target square Gram matrix of shape `(n, n)`.
        /// @param[in] uplo Active matrix triangle to populate.
        /// @param[in] exp_mode Exponential evaluation backend.
        /// @param[in] cached_ls Pre-cached ARD lengthscales, or empty to read from `leaf`.
        /// @return Empty expected on success, or validation/dimension/evaluation error.
        /// @retval GprErrorCode::EmptyInput If `x` has zero rows or columns.
        /// @retval GprErrorCode::LengthMismatch If `out` is not `(x.rows(), x.rows())`.
        /// @retval GprErrorCode::DimensionMismatch If `x.cols()` differs from ARD lengthscale count.
        /// @retval GprErrorCode::NonFiniteInput If any coordinate in `x` is non-finite.
        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> eval_leaf_points(
            const KernelLeaf& leaf,
            MatrixRef<T> x,
            MatrixMut<T> out,
            Triangle uplo,
            KernelExp exp_mode,
            std::span<const double> cached_ls
        )
        {
            // Validate coordinates and square output matrix shape before dispatching by leaf variant.
            if (auto v = validate_square_points(x, out); !v.has_value())
            {
                return v;
            }
            return std::visit(
                [&](const auto& k) -> std::expected<void, GprError>
            {
                using LeafT = std::decay_t<decltype(k)>;
                if constexpr (std::is_same_v<LeafT, RbfKernel>)
                {
                    const T ell = static_cast<T>(k.lengthscale());
                    const T neg_inv_2l2 = -T{1} / (T{2} * ell * ell);
                    eval_square_points_sq_dist_cols(
                        x, out, uplo, std::span<const T>{}, [neg_inv_2l2, exp_mode](T* col_ptr, std::size_t len)
                    { detail::exp_scaled_nonpos_in_place(col_ptr, len, neg_inv_2l2, exp_mode); }
                    );
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, RbfArdKernel>)
                {
                    return with_ard_ls(
                        k, cached_ls, [&](std::span<const double> ls)
                    {
                        if (auto d = check_ard_dim<T>(x.cols(), ls.size()); !d.has_value())
                        {
                            return d;
                        }
                        return with_ard_inv_ell2<T>(
                            ls, T{1}, [&](std::span<const T> weights) -> std::expected<void, GprError>
                        {
                            eval_square_points_sq_dist_cols(
                                x, out, uplo, weights, [exp_mode](T* col_ptr, std::size_t len)
                            { detail::exp_scaled_nonpos_in_place(col_ptr, len, T{-0.5}, exp_mode); }
                            );
                            return {};
                        }
                        );
                    }
                    );
                }
                else if constexpr (std::is_same_v<LeafT, MaternKernel>)
                {
                    const T scale_sq = matern_scale_sq<T>(k.nu()) / static_cast<T>(k.lengthscale() * k.lengthscale());
                    const MaternNu nu = k.nu();
                    eval_square_points_sq_dist_cols(
                        x, out, uplo, std::span<const T>{}, [scale_sq, nu, exp_mode](T* col_ptr, std::size_t len)
                    { detail::matern_s2_in_place(col_ptr, len, scale_sq, nu, exp_mode); }
                    );
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, MaternArdKernel>)
                {
                    return with_ard_ls(
                        k, cached_ls, [&](std::span<const double> ls)
                    {
                        if (auto d = check_ard_dim<T>(x.cols(), ls.size()); !d.has_value())
                        {
                            return d;
                        }
                        const T c = matern_scale_sq<T>(k.nu());
                        const MaternNu nu = k.nu();
                        return with_ard_inv_ell2<T>(
                            ls, T{1}, [&](std::span<const T> weights) -> std::expected<void, GprError>
                        {
                            eval_square_points_sq_dist_cols(
                                x, out, uplo, weights, [c, nu, exp_mode](T* col_ptr, std::size_t len)
                            { detail::matern_s2_in_place(col_ptr, len, c, nu, exp_mode); }
                            );
                            return {};
                        }
                        );
                    }
                    );
                }
                else if constexpr (std::is_same_v<LeafT, PeriodicKernel>)
                {
                    const T ell = static_cast<T>(k.lengthscale());
                    const T period = static_cast<T>(k.period());
                    const T neg_coeff = -T{2} / (ell * ell);
                    const T pi_p = std::numbers::pi_v<T> / period;
                    eval_square_points_sq_dist_cols(
                        x, out, uplo, std::span<const T>{}, [neg_coeff, pi_p, exp_mode](T* col_ptr, std::size_t len)
                    { transform_periodic_d2_slice_in_place(col_ptr, len, neg_coeff, pi_p, exp_mode); }
                    );
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, RationalQuadraticKernel>)
                {
                    const T ell = static_cast<T>(k.lengthscale());
                    const T alpha = static_cast<T>(k.alpha());
                    const T w_scale = T{1} / (T{2} * alpha * ell * ell);
                    eval_square_points_sq_dist_cols(
                        x, out, uplo, std::span<const T>{}, [w_scale, alpha, exp_mode](T* col_ptr, std::size_t len)
                    { transform_rq_slice_in_place(col_ptr, len, w_scale, alpha, exp_mode); }
                    );
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, RationalQuadraticArdKernel>)
                {
                    return with_ard_ls(
                        k, cached_ls, [&](std::span<const double> ls)
                    {
                        if (auto d = check_ard_dim<T>(x.cols(), ls.size()); !d.has_value())
                        {
                            return d;
                        }
                        const T alpha = static_cast<T>(k.alpha());
                        const T inv_2alpha = T{1} / (T{2} * alpha);
                        return with_ard_inv_ell2<T>(
                            ls, inv_2alpha, [&](std::span<const T> weights) -> std::expected<void, GprError>
                        {
                            eval_square_points_sq_dist_cols(
                                x, out, uplo, weights, [alpha, exp_mode](T* col_ptr, std::size_t len)
                            { transform_rq_slice_in_place(col_ptr, len, T{1}, alpha, exp_mode); }
                            );
                            return {};
                        }
                        );
                    }
                    );
                }
                else if constexpr (std::is_same_v<LeafT, ConstantKernel>)
                {
                    const T c = static_cast<T>(k.constant());
                    write_square_points(x, out, uplo, [c](std::size_t, std::size_t) { return c; });
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, LinearKernel>)
                {
                    const T var = static_cast<T>(k.variance());
                    const std::size_t n = x.rows();
                    const std::size_t d = x.cols();
                    for_each_triangle_col(
                        n, uplo, [&](std::size_t col, std::size_t row_begin, std::size_t row_end)
                    {
                        const std::size_t len = row_end - row_begin;
                        if (len == 0U)
                        {
                            return;
                        }
                        T* col_ptr = &out(row_begin, col);
                        detail::set_dot_slice(col_ptr, x.col(0U).data() + row_begin, x(col, 0U), len);
                        for (std::size_t dim = 1U; dim < d; ++dim)
                        {
                            const auto col_vec = x.col(dim);
                            detail::add_dot_slice(col_ptr, col_vec.data() + row_begin, col_vec[col], len);
                        }
                        detail::scale_set_slice(col_ptr, col_ptr, var, len);
                    }
                    );
                    mirror_lower_if_full(out, uplo);
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, CustomKernel>)
                {
                    auto dist_buf = make_square_dist_buf(x);
                    const MatrixRef<T> dist_ref(dist_buf.data(), x.rows(), x.rows(), x.rows());
                    return k.template term<T>().apply(dist_ref, out, uplo);
                }
                else
                {
                    // WhiteKernel is diagonal-only even when evaluated from coordinates.
                    const T var = static_cast<T>(k.variance());
                    write_square_points(x, out, uplo, [var](std::size_t row, std::size_t col) {
                        return row == col ? var : T{0};
                    });
                    return {};
                }
            }, leaf
            );
        }

        /// @brief Evaluates a single kernel leaf over a rectangular cross squared-distance matrix.
        /// @details Writes zero everywhere for `WhiteKernel` cross-covariances and rejects coordinate-only leaves.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] leaf Kernel leaf variant to evaluate.
        /// @param[in] dist Rectangular squared-distance matrix of shape `(n1, n2)`.
        /// @param[out] out Target cross-covariance matrix of shape `(n1, n2)`.
        /// @param[in] exp_mode Exponential evaluation backend.
        /// @return Empty expected on success, or validation/operation error.
        /// @retval GprErrorCode::UnsupportedKernelOperation If `leaf` requires point coordinates.
        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> eval_leaf_cross_dist(
            const KernelLeaf& leaf, MatrixRef<T> dist, MatrixMut<T> out, KernelExp exp_mode
        )
        {
            // Validate rectangular cross distance matrix and output shape.
            if (auto v = validate_cross_dist(dist, out); !v.has_value())
            {
                return v;
            }
            return std::visit(
                [&](const auto& k) -> std::expected<void, GprError>
            {
                using LeafT = std::decay_t<decltype(k)>;
                if constexpr (std::is_same_v<LeafT, RbfKernel>)
                {
                    const T ell = static_cast<T>(k.lengthscale());
                    const T neg_inv_2l2 = -T{1} / (T{2} * ell * ell);
                    const std::size_t n_rows = dist.rows();
                    const auto cols_s = static_cast<std::ptrdiff_t>(dist.cols());
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if ((n_rows * dist.cols()) >= 65536U)
#endif
                    for (std::ptrdiff_t col_s = 0; col_s < cols_s; ++col_s)
                    {
                        const auto col = static_cast<std::size_t>(col_s);
                        std::copy_n(&dist(0U, col), n_rows, &out(0U, col));
                        detail::exp_scaled_nonpos_in_place(&out(0U, col), n_rows, neg_inv_2l2, exp_mode);
                    }
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, MaternKernel>)
                {
                    const T scale_sq = matern_scale_sq<T>(k.nu()) / static_cast<T>(k.lengthscale() * k.lengthscale());
                    const MaternNu nu = k.nu();
                    const std::size_t n_rows = dist.rows();
                    const auto cols_s = static_cast<std::ptrdiff_t>(dist.cols());
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if ((n_rows * dist.cols()) >= 65536U)
#endif
                    for (std::ptrdiff_t col_s = 0; col_s < cols_s; ++col_s)
                    {
                        const auto col = static_cast<std::size_t>(col_s);
                        std::copy_n(&dist(0U, col), n_rows, &out(0U, col));
                        detail::matern_s2_in_place(&out(0U, col), n_rows, scale_sq, nu, exp_mode);
                    }
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, PeriodicKernel>)
                {
                    const T ell = static_cast<T>(k.lengthscale());
                    const T period = static_cast<T>(k.period());
                    const T neg_coeff = -T{2} / (ell * ell);
                    const T pi_p = std::numbers::pi_v<T> / period;
                    const std::size_t n_rows = dist.rows();
                    const auto cols_s = static_cast<std::ptrdiff_t>(dist.cols());
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if ((n_rows * dist.cols()) >= 65536U)
#endif
                    for (std::ptrdiff_t col_s = 0; col_s < cols_s; ++col_s)
                    {
                        const auto col = static_cast<std::size_t>(col_s);
                        std::copy_n(&dist(0U, col), n_rows, &out(0U, col));
                        transform_periodic_d2_slice_in_place(&out(0U, col), n_rows, neg_coeff, pi_p, exp_mode);
                    }
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, RationalQuadraticKernel>)
                {
                    const T ell = static_cast<T>(k.lengthscale());
                    const T alpha = static_cast<T>(k.alpha());
                    const T w_scale = T{1} / (T{2} * alpha * ell * ell);
                    const std::size_t n_rows = dist.rows();
                    const auto cols_s = static_cast<std::ptrdiff_t>(dist.cols());
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if ((n_rows * dist.cols()) >= 65536U)
#endif
                    for (std::ptrdiff_t col_s = 0; col_s < cols_s; ++col_s)
                    {
                        const auto col = static_cast<std::size_t>(col_s);
                        std::copy_n(&dist(0U, col), n_rows, &out(0U, col));
                        transform_rq_slice_in_place(&out(0U, col), n_rows, w_scale, alpha, exp_mode);
                    }
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, ConstantKernel>)
                {
                    const T c = static_cast<T>(k.constant());
                    write_cross_dist(dist, out, [c](T, std::size_t, std::size_t) { return c; });
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, WhiteKernel>)
                {
                    // Cross-covariance between distinct training and query observations has zero white noise.
                    write_cross_dist(dist, out, [](T, std::size_t, std::size_t) { return T{0}; });
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, CustomKernel>)
                {
                    return k.template term<T>().apply_cross(dist, out);
                }
                else
                {
                    return std::unexpected(
                        GprError::unsupported_kernel_operation(
                            "coordinate-dependent leaf cannot be evaluated from squared distances"
                        )
                    );
                }
            }, leaf
            );
        }

        /// @brief Evaluates a single kernel leaf cross-covariance directly from two point matrices.
        /// @details Evaluates `k(x1, x2)` across all built-in and custom leaves; `WhiteKernel` evaluates to zero.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] leaf Kernel leaf variant to evaluate.
        /// @param[in] x1 First point matrix of shape `(n1, d)`.
        /// @param[in] x2 Second point matrix of shape `(n2, d)`.
        /// @param[out] out Target cross-covariance matrix of shape `(n1, n2)`.
        /// @param[in] exp_mode Exponential evaluation backend.
        /// @param[in] cached_ls Pre-cached ARD lengthscales, or empty to read from `leaf`.
        /// @return Empty expected on success, or validation/dimension/evaluation error.
        /// @retval GprErrorCode::EmptyInput If either point matrix is empty.
        /// @retval GprErrorCode::LengthMismatch If `out` shape differs from `(x1.rows(), x2.rows())`.
        /// @retval GprErrorCode::DimensionMismatch If feature counts do not match across `x1`, `x2`, and ARD leaves.
        /// @retval GprErrorCode::NonFiniteInput If any coordinate in `x1` or `x2` is non-finite.
        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> eval_leaf_cross_points(
            const KernelLeaf& leaf,
            MatrixRef<T> x1,
            MatrixRef<T> x2,
            MatrixMut<T> out,
            KernelExp exp_mode,
            std::span<const double> cached_ls
        )
        {
            // Validate point shapes, matching feature count, and finite entries before dispatching.
            if (auto v = validate_cross_points(x1, x2, out); !v.has_value())
            {
                return v;
            }
            return std::visit(
                [&](const auto& k) -> std::expected<void, GprError>
            {
                using LeafT = std::decay_t<decltype(k)>;
                if constexpr (std::is_same_v<LeafT, RbfKernel>)
                {
                    const T ell = static_cast<T>(k.lengthscale());
                    const T neg_inv_2l2 = -T{1} / (T{2} * ell * ell);
                    eval_cross_points_sq_dist_cols(
                        x1, x2, out, std::span<const T>{}, [neg_inv_2l2, exp_mode](T* col_ptr, std::size_t len)
                    { detail::exp_scaled_nonpos_in_place(col_ptr, len, neg_inv_2l2, exp_mode); }
                    );
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, RbfArdKernel>)
                {
                    return with_ard_ls(
                        k, cached_ls, [&](std::span<const double> ls)
                    {
                        if (auto d = check_ard_dim<T>(x1.cols(), ls.size()); !d.has_value())
                        {
                            return d;
                        }
                        return with_ard_inv_ell2<T>(
                            ls, T{1}, [&](std::span<const T> weights) -> std::expected<void, GprError>
                        {
                            eval_cross_points_sq_dist_cols(
                                x1, x2, out, weights, [exp_mode](T* col_ptr, std::size_t len)
                            { detail::exp_scaled_nonpos_in_place(col_ptr, len, T{-0.5}, exp_mode); }
                            );
                            return {};
                        }
                        );
                    }
                    );
                }
                else if constexpr (std::is_same_v<LeafT, MaternKernel>)
                {
                    const T scale_sq = matern_scale_sq<T>(k.nu()) / static_cast<T>(k.lengthscale() * k.lengthscale());
                    const MaternNu nu = k.nu();
                    eval_cross_points_sq_dist_cols(
                        x1, x2, out, std::span<const T>{}, [scale_sq, nu, exp_mode](T* col_ptr, std::size_t len)
                    { detail::matern_s2_in_place(col_ptr, len, scale_sq, nu, exp_mode); }
                    );
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, MaternArdKernel>)
                {
                    return with_ard_ls(
                        k, cached_ls, [&](std::span<const double> ls)
                    {
                        if (auto d = check_ard_dim<T>(x1.cols(), ls.size()); !d.has_value())
                        {
                            return d;
                        }
                        const T c = matern_scale_sq<T>(k.nu());
                        const MaternNu nu = k.nu();
                        return with_ard_inv_ell2<T>(
                            ls, T{1}, [&](std::span<const T> weights) -> std::expected<void, GprError>
                        {
                            eval_cross_points_sq_dist_cols(
                                x1, x2, out, weights, [c, nu, exp_mode](T* col_ptr, std::size_t len)
                            { detail::matern_s2_in_place(col_ptr, len, c, nu, exp_mode); }
                            );
                            return {};
                        }
                        );
                    }
                    );
                }
                else if constexpr (std::is_same_v<LeafT, PeriodicKernel>)
                {
                    const T ell = static_cast<T>(k.lengthscale());
                    const T period = static_cast<T>(k.period());
                    const T neg_coeff = -T{2} / (ell * ell);
                    const T pi_p = std::numbers::pi_v<T> / period;
                    eval_cross_points_sq_dist_cols(
                        x1, x2, out, std::span<const T>{}, [neg_coeff, pi_p, exp_mode](T* col_ptr, std::size_t len)
                    { transform_periodic_d2_slice_in_place(col_ptr, len, neg_coeff, pi_p, exp_mode); }
                    );
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, RationalQuadraticKernel>)
                {
                    const T ell = static_cast<T>(k.lengthscale());
                    const T alpha = static_cast<T>(k.alpha());
                    const T w_scale = T{1} / (T{2} * alpha * ell * ell);
                    eval_cross_points_sq_dist_cols(
                        x1, x2, out, std::span<const T>{}, [w_scale, alpha, exp_mode](T* col_ptr, std::size_t len)
                    { transform_rq_slice_in_place(col_ptr, len, w_scale, alpha, exp_mode); }
                    );
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, RationalQuadraticArdKernel>)
                {
                    return with_ard_ls(
                        k, cached_ls, [&](std::span<const double> ls)
                    {
                        if (auto d = check_ard_dim<T>(x1.cols(), ls.size()); !d.has_value())
                        {
                            return d;
                        }
                        const T alpha = static_cast<T>(k.alpha());
                        const T inv_2alpha = T{1} / (T{2} * alpha);
                        return with_ard_inv_ell2<T>(
                            ls, inv_2alpha, [&](std::span<const T> weights) -> std::expected<void, GprError>
                        {
                            eval_cross_points_sq_dist_cols(
                                x1, x2, out, weights, [alpha, exp_mode](T* col_ptr, std::size_t len)
                            { transform_rq_slice_in_place(col_ptr, len, T{1}, alpha, exp_mode); }
                            );
                            return {};
                        }
                        );
                    }
                    );
                }
                else if constexpr (std::is_same_v<LeafT, ConstantKernel>)
                {
                    const T c = static_cast<T>(k.constant());
                    write_cross_points(x1, x2, out, [c](std::size_t, std::size_t) { return c; });
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, LinearKernel>)
                {
                    const T var = static_cast<T>(k.variance());
                    const std::size_t n1 = x1.rows();
                    const std::size_t n2 = x2.rows();
                    const std::size_t d = x1.cols();
                    const auto cols_s = static_cast<std::ptrdiff_t>(n2);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if ((n1 * n2) >= 65536U)
#endif
                    for (std::ptrdiff_t col_s = 0; col_s < cols_s; ++col_s)
                    {
                        const auto col = static_cast<std::size_t>(col_s);
                        T* col_ptr = &out(0U, col);
                        detail::set_dot_slice(col_ptr, x1.col(0U).data(), x2(col, 0U), n1);
                        for (std::size_t dim = 1U; dim < d; ++dim)
                        {
                            detail::add_dot_slice(col_ptr, x1.col(dim).data(), x2.col(dim)[col], n1);
                        }
                        detail::scale_set_slice(col_ptr, col_ptr, var, n1);
                    }
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, CustomKernel>)
                {
                    auto dist_buf = make_cross_dist_buf(x1, x2);
                    const MatrixRef<T> dist_ref(dist_buf.data(), x1.rows(), x2.rows(), x1.rows());
                    return k.template term<T>().apply_cross(dist_ref, out);
                }
                else
                {
                    // Cross-covariance entries of WhiteKernel are identically zero.
                    write_cross_points(x1, x2, out, [](std::size_t, std::size_t) { return T{0}; });
                    return {};
                }
            }, leaf
            );
        }

        /// @brief Computes $\partial k / \partial (\log \ell)$ for an isotropic RBF leaf given $d^2$.
        /// @details With $z = \max(d^2, 0) / (2 \ell^2)$ and $\theta = \log \ell$, $\partial(-z)/\partial\theta = 2z$,
        /// yielding $2z \cdot \operatorname{exp}'(-z)$.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] d2 Squared Euclidean distance.
        /// @param[in] ell Lengthscale parameter $\ell > 0$.
        /// @param[in] exp_mode Exponential evaluation backend.
        /// @return First derivative with respect to $\log \ell$.
        template <KernelScalarConcept T> [[nodiscard]] T rbf_grad_d2(T d2, T ell, KernelExp exp_mode) noexcept
        {
            // Chain rule wrt theta = log(ell): d(-z)/dtheta = 2 * z where z = d^2 / (2 * ell^2).
            const T clamped = d2 < T{0} ? T{0} : d2;
            const T l2 = ell * ell;
            const T z = clamped / (T{2} * l2);
            const ExpJet<T> jet = eval_kernel_exp_jet(-z, exp_mode);
            return T{2} * z * jet.d1;
        }

        /// @brief Computes $\partial^2 k / \partial (\log \ell)^2$ for an isotropic RBF leaf given $d^2$.
        /// @details Differentiating $2z \cdot \operatorname{exp}'(-z)$ wrt $\theta = \log \ell$ (where
        /// $\partial z/\partial\theta = -2z$) gives $4z(z \cdot \operatorname{exp}''(-z) - \operatorname{exp}'(-z))$.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] d2 Squared Euclidean distance.
        /// @param[in] ell Lengthscale parameter $\ell > 0$.
        /// @param[in] exp_mode Exponential evaluation backend.
        /// @return Second derivative with respect to $\log \ell$.
        template <KernelScalarConcept T> [[nodiscard]] T rbf_hess_d2(T d2, T ell, KernelExp exp_mode) noexcept
        {
            // Differentiate 2*z*exp'(-z) with dz/dtheta = -2*z: 4*z * (z*exp''(-z) - exp'(-z)).
            const T clamped = d2 < T{0} ? T{0} : d2;
            const T l2 = ell * ell;
            const T z = clamped / (T{2} * l2);
            const ExpJet<T> jet = eval_kernel_exp_jet(-z, exp_mode);
            return T{4} * z * ((z * jet.d2) - jet.d1);
        }

        /// @brief Computes first log-hyperparameter derivative of an isotropic Periodic leaf given $d^2$.
        /// @details Let $\phi = \pi r / p$, $s = \sin\phi$, $c = \cos\phi$, and $u = -2 s^2 / \ell^2$. Returns
        /// $(4 s^2 / \ell^2)\operatorname{exp}'(u)$ for `idx == 0` ($\log \ell$) and
        /// $(4 \phi s c / \ell^2)\operatorname{exp}'(u)$ for `idx == 1` ($\log p$).
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] d2 Squared Euclidean distance.
        /// @param[in] ell Lengthscale parameter $\ell > 0$.
        /// @param[in] period Period parameter $p > 0$.
        /// @param[in] idx Local hyperparameter index (`0` for $\log \ell$, `1` for $\log p$).
        /// @param[in] exp_mode Exponential evaluation backend.
        /// @return First derivative with respect to parameter `idx`.
        template <KernelScalarConcept T>
        [[nodiscard]] T periodic_grad_d2(T d2, T ell, T period, std::size_t idx, KernelExp exp_mode) noexcept
        {
            // Let phi = pi * r / period and u = -2*sin^2(phi)/ell^2; du/dlog(ell) = 4*s^2/ell^2 and
            // du/dlog(period) = 4*phi*sin(phi)*cos(phi)/ell^2.
            const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
            const T phi = (std::numbers::pi_v<T> * r) / period;
            const T s = std::sin(phi);
            const T c = std::cos(phi);
            const T l2 = ell * ell;
            const ExpJet<T> jet = eval_kernel_exp_jet(-(T{2} * s * s) / l2, exp_mode);
            if (idx == 0)
            {
                return ((T{4} * s * s) / l2) * jet.d1;
            }
            return ((T{4} * phi * s * c) / l2) * jet.d1;
        }

        /// @brief Computes second log-hyperparameter derivative of an isotropic Periodic leaf given $d^2$.
        /// @details Applies the product and chain rules to $a = \partial u/\partial \log\ell$ and
        /// $b = \partial u/\partial \log p$, handling diagonal $(0,0)$, mixed $(0,1)$, and $(1,1)$ blocks.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] d2 Squared Euclidean distance.
        /// @param[in] ell Lengthscale parameter $\ell > 0$.
        /// @param[in] period Period parameter $p > 0$.
        /// @param[in] i First local hyperparameter index (`0` or `1`).
        /// @param[in] j Second local hyperparameter index (`0` or `1`).
        /// @param[in] exp_mode Exponential evaluation backend.
        /// @return Second derivative $\partial^2 k / \partial \theta_i \partial \theta_j$.
        template <KernelScalarConcept T>
        [[nodiscard]] T periodic_hess_d2(
            T d2, T ell, T period, std::size_t i, std::size_t j, KernelExp exp_mode
        ) noexcept
        {
            // With a = du/dtheta_0 and b = du/dtheta_1, use da/dtheta_0 = -2a, da/dtheta_1 = -2b, and
            // db/dtheta_1 = -(b + 4*phi^2*cos(2*phi)/ell^2).
            const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
            const T phi = (std::numbers::pi_v<T> * r) / period;
            const T s = std::sin(phi);
            const T c = std::cos(phi);
            const T l2 = ell * ell;
            const ExpJet<T> jet = eval_kernel_exp_jet(-(T{2} * s * s) / l2, exp_mode);
            const T a = (T{4} * s * s) / l2;
            const T b = (T{4} * phi * s * c) / l2;
            if (i == 0 && j == 0)
            {
                return a * ((a * jet.d2) - (T{2} * jet.d1));
            }
            if (i != j)
            {
                return b * ((a * jet.d2) - (T{2} * jet.d1));
            }
            const T cos2phi = (c * c) - (s * s);
            return (b * b * jet.d2) - ((b + ((T{4} * phi * phi * cos2phi) / l2)) * jet.d1);
        }

        /// @brief Computes first log-hyperparameter derivative of an isotropic RationalQuadratic leaf from $w$.
        /// @details With $w = d^2 / (2\alpha \ell^2)$ and $b = 1 + w$, returns
        /// $(2\alpha w / b)\operatorname{exp}'(-\alpha\log(1+w))$ for $\log\ell$ (`idx < alpha_idx`) and
        /// $\alpha(w/b - \log(1+w))\operatorname{exp}'(-\alpha\log(1+w))$ for $\log\alpha$ (`idx == alpha_idx`).
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] w Normalized squared distance $d^2 / (2\alpha \ell^2)$.
        /// @param[in] alpha Scale-mixture shape parameter $\alpha > 0$.
        /// @param[in] idx Local hyperparameter index.
        /// @param[in] alpha_idx Index of $\log \alpha$ (`1` for isotropic RQ).
        /// @param[in] exp_mode Exponential evaluation backend.
        /// @return First derivative with respect to parameter `idx`.
        template <KernelScalarConcept T>
        [[nodiscard]] T rq_grad_w(T w, T alpha, std::size_t idx, std::size_t alpha_idx, KernelExp exp_mode) noexcept
        {
            // For u = -alpha * log1p(w) with b = 1 + w: du/dlog(ell) = 2*alpha*w/b and
            // du/dlog(alpha) = alpha * (w/b - log1p(w)).
            const T b = T{1} + w;
            const ExpJet<T> jet = eval_kernel_exp_jet(-alpha * std::log1p(w), exp_mode);
            if (idx < alpha_idx)
            {
                return ((T{2} * alpha * w) / b) * jet.d1;
            }
            return alpha * ((w / b) - std::log1p(w)) * jet.d1;
        }

        /// @brief Computes second log-hyperparameter derivative of an isotropic RationalQuadratic leaf from $w$.
        /// @details Propagates second derivatives of $u = -\alpha \operatorname{log1p}(w)$ across
        /// $(\log\ell, \log\ell)$, mixed $(\log\ell, \log\alpha)$, and $(\log\alpha, \log\alpha)$ pairs.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] w Normalized squared distance $d^2 / (2\alpha \ell^2)$.
        /// @param[in] alpha Scale-mixture shape parameter $\alpha > 0$.
        /// @param[in] i First local hyperparameter index (`0` for $\log\ell$, `1` for $\log\alpha$).
        /// @param[in] j Second local hyperparameter index (`0` for $\log\ell$, `1` for $\log\alpha$).
        /// @param[in] exp_mode Exponential evaluation backend.
        /// @return Second derivative $\partial^2 k / \partial \theta_i \partial \theta_j$.
        template <KernelScalarConcept T>
        [[nodiscard]] T rq_hess_iso(T w, T alpha, std::size_t i, std::size_t j, KernelExp exp_mode) noexcept
        {
            // With g0 = du/dlog(ell) and g1 = du/dlog(alpha): dg0/dlog(ell) = -4*alpha*w/b^2,
            // dg0/dlog(alpha) = 2*alpha*w^2/b^2, and dg1/dlog(alpha) = g1 + alpha*w^2/b^2.
            const T b = T{1} + w;
            const ExpJet<T> jet = eval_kernel_exp_jet(-alpha * std::log1p(w), exp_mode);
            const T g0 = (T{2} * alpha * w) / b;
            const T g1 = alpha * ((w / b) - std::log1p(w));
            if (i == 0 && j == 0)
            {
                return ((g0 * g0) * jet.d2) - (((T{4} * alpha * w) / (b * b)) * jet.d1);
            }
            if (i != j)
            {
                return ((g0 * g1) * jet.d2) + (((T{2} * alpha * w * w) / (b * b)) * jet.d1);
            }
            return ((g1 * g1) * jet.d2) + ((g1 + ((alpha * w * w) / (b * b))) * jet.d1);
        }

    }  // namespace

    /// @brief Classifies the input coordinate mode required by a single kernel leaf.
    /// @details Returns `Either` for `ConstantKernel`/`WhiteKernel`, `Points` for ARD and `LinearKernel` leaves, and
    /// `Dist` for isotropic distance-dependent and `CustomKernel` leaves.
    /// @param[in] leaf Kernel leaf variant.
    /// @return `CoordMode` classifying the leaf's input dependency.
    CoordMode leaf_coord_mode(const KernelLeaf& leaf) noexcept
    {
        if (std::holds_alternative<ConstantKernel>(leaf) || std::holds_alternative<WhiteKernel>(leaf))
        {
            return CoordMode::Either;
        }
        if (std::holds_alternative<RbfArdKernel>(leaf) || std::holds_alternative<MaternArdKernel>(leaf)
            || std::holds_alternative<RationalQuadraticArdKernel>(leaf) || std::holds_alternative<LinearKernel>(leaf))
        {
            return CoordMode::Points;
        }
        return CoordMode::Dist;
    }

    /// @brief Returns the number of log-space hyperparameters owned by a kernel leaf.
    /// @details Returns `dims` for RBF/Matern ARD, `dims + 1` for RQ ARD, `2` for Periodic and isotropic RQ,
    /// `k.num_params()` for `CustomKernel`, and `1` for remaining built-in scalar leaves.
    /// @param[in] leaf Kernel leaf variant.
    /// @return Hyperparameter count for `leaf`.
    std::size_t leaf_num_params(const KernelLeaf& leaf) noexcept
    {
        if (const auto* k = std::get_if<RbfArdKernel>(&leaf))
        {
            return k->num_params();
        }
        if (const auto* k = std::get_if<MaternArdKernel>(&leaf))
        {
            return k->num_params();
        }
        if (std::holds_alternative<PeriodicKernel>(leaf))
        {
            return PeriodicKernel::num_params();
        }
        if (std::holds_alternative<RationalQuadraticKernel>(leaf))
        {
            return RationalQuadraticKernel::num_params();
        }
        if (const auto* k = std::get_if<RationalQuadraticArdKernel>(&leaf))
        {
            return k->num_params();
        }
        if (const auto* k = std::get_if<CustomKernel>(&leaf))
        {
            return k->num_params();
        }
        return 1U;
    }

    /// @brief Copies a leaf's log-space hyperparameters into `out`.
    /// @details Dispatches to `get_params` of the underlying leaf variant.
    /// @param[in] leaf Kernel leaf variant.
    /// @param[out] out Destination span of length `leaf_num_params(leaf)`.
    /// @return Empty expected on success, or length mismatch error.
    /// @retval GprErrorCode::LengthMismatch If `out.size()` differs from `leaf_num_params(leaf)`.
    std::expected<void, GprError> leaf_get_params(const KernelLeaf& leaf, std::span<double> out)
    {
        return std::visit([out](const auto& k) { return k.get_params(out); }, leaf);
    }

    /// @brief Updates a leaf's hyperparameters from log-space `params`.
    /// @details Dispatches to `set_params` of the underlying leaf variant; leaves `leaf` unchanged on error.
    /// @param[in,out] leaf Kernel leaf variant to update.
    /// @param[in] params Log-space parameter span of length `leaf_num_params(leaf)`.
    /// @return Empty expected on success, or length/finitedness/bounds error.
    /// @retval GprErrorCode::LengthMismatch If `params.size()` differs from `leaf_num_params(leaf)`.
    /// @retval GprErrorCode::InvalidHyperparameter If any value lies outside its open parameter interval.
    std::expected<void, GprError> leaf_set_params(KernelLeaf& leaf, std::span<const double> params)
    {
        return std::visit([params](auto& k) { return k.set_params(params); }, leaf);
    }

    /// @brief Copies a leaf's open hyperparameter intervals into `out`.
    /// @details Dispatches to `bounds_into` of the underlying leaf variant.
    /// @param[in] leaf Kernel leaf variant.
    /// @param[out] out Destination span of intervals of length `leaf_num_params(leaf)`.
    /// @return Empty expected on success, or length mismatch error.
    /// @retval GprErrorCode::LengthMismatch If `out.size()` differs from `leaf_num_params(leaf)`.
    std::expected<void, GprError> leaf_bounds_into(const KernelLeaf& leaf, std::span<Interval> out)
    {
        return std::visit([out](const auto& k) { return k.bounds_into(out); }, leaf);
    }

    /// @brief Evaluates a square Gram matrix triangle for `leaf` from squared distances.
    /// @details Public wrapper around `eval_leaf_dist`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] leaf Kernel leaf variant.
    /// @param[in] dist Input square squared-distance matrix.
    /// @param[out] out Output Gram matrix view.
    /// @param[in] uplo Active matrix triangle to fill.
    /// @param[in] exp_mode Exponential evaluation backend.
    /// @return Empty expected on success, or validation/operation error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> leaf_apply_dist(
        const KernelLeaf& leaf, MatrixRef<T> dist, MatrixMut<T> out, Triangle uplo, KernelExp exp_mode
    )
    {
        return eval_leaf_dist(leaf, dist, out, uplo, exp_mode);
    }

    /// @brief Evaluates a square Gram matrix triangle for `leaf` from point coordinates.
    /// @details Public wrapper around `eval_leaf_points`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] leaf Kernel leaf variant.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[out] out Output square Gram matrix of shape `(n, n)`.
    /// @param[in] uplo Active matrix triangle to fill.
    /// @param[in] exp_mode Exponential evaluation backend.
    /// @param[in] cached_ls Pre-cached ARD lengthscales, or empty to query `leaf`.
    /// @return Empty expected on success, or validation/dimension/operation error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> leaf_apply_points(
        const KernelLeaf& leaf,
        MatrixRef<T> x,
        MatrixMut<T> out,
        Triangle uplo,
        KernelExp exp_mode,
        std::span<const double> cached_ls
    )
    {
        return eval_leaf_points(leaf, x, out, uplo, exp_mode, cached_ls);
    }

    /// @brief Evaluates a rectangular cross-covariance matrix for `leaf` from squared distances.
    /// @details Public wrapper around `eval_leaf_cross_dist`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] leaf Kernel leaf variant.
    /// @param[in] dist Input rectangular squared-distance matrix.
    /// @param[out] out Output cross-covariance matrix.
    /// @param[in] exp_mode Exponential evaluation backend.
    /// @return Empty expected on success, or validation/operation error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> leaf_apply_cross_dist(
        const KernelLeaf& leaf, MatrixRef<T> dist, MatrixMut<T> out, KernelExp exp_mode
    )
    {
        return eval_leaf_cross_dist(leaf, dist, out, exp_mode);
    }

    /// @brief Evaluates a rectangular cross-covariance matrix for `leaf` from point coordinates.
    /// @details Public wrapper around `eval_leaf_cross_points`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] leaf Kernel leaf variant.
    /// @param[in] x1 First point matrix of shape `(n1, d)`.
    /// @param[in] x2 Second point matrix of shape `(n2, d)`.
    /// @param[out] out Output cross-covariance matrix of shape `(n1, n2)`.
    /// @param[in] exp_mode Exponential evaluation backend.
    /// @param[in] cached_ls Pre-cached ARD lengthscales, or empty to query `leaf`.
    /// @return Empty expected on success, or validation/dimension/operation error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> leaf_apply_cross_points(
        const KernelLeaf& leaf,
        MatrixRef<T> x1,
        MatrixRef<T> x2,
        MatrixMut<T> out,
        KernelExp exp_mode,
        std::span<const double> cached_ls
    )
    {
        return eval_leaf_cross_points(leaf, x1, x2, out, exp_mode, cached_ls);
    }

    /// @brief Fills the diagonal $k(x_i, x_i)$ of a kernel leaf over point matrix `x`.
    /// @details Stationary isotropic/ARD leaves evaluate to `1`, `ConstantKernel` fills `constant()`, `WhiteKernel`
    /// fills `variance()`, `LinearKernel` evaluates $\sigma^2 \|x_i\|_2^2$, and `CustomKernel` delegates to
    /// `fill_diag`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] leaf Kernel leaf variant.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[out] out Output diagonal span of length `n`.
    /// @param[in] cached_ls Pre-cached ARD lengthscales, or empty to inspect `leaf`.
    /// @return Empty expected on success, or validation/dimension/evaluation error.
    /// @retval GprErrorCode::EmptyInput If `x` is empty.
    /// @retval GprErrorCode::LengthMismatch If `out.size() != x.rows()`.
    /// @retval GprErrorCode::DimensionMismatch If `x.cols()` differs from an ARD leaf's lengthscale count.
    /// @retval GprErrorCode::NonFiniteInput If `x` contains non-finite coordinates.
    template <KernelScalarConcept T>
    std::expected<void, GprError> leaf_fill_diag(
        const KernelLeaf& leaf, MatrixRef<T> x, std::span<T> out, std::span<const double> cached_ls
    )
    {
        // Validate input coordinates and diagonal vector length before dispatching.
        if (auto v = validate_points(x); !v.has_value())
        {
            return v;
        }
        if (auto cnt = require_count(out.size(), x.rows(), "out"); !cnt.has_value())
        {
            return std::unexpected(std::move(cnt).error());
        }
        return std::visit(
            [&](const auto& k) -> std::expected<void, GprError>
        {
            using LeafT = std::decay_t<decltype(k)>;
            if constexpr (std::is_same_v<LeafT, RbfArdKernel> || std::is_same_v<LeafT, MaternArdKernel>
                || std::is_same_v<LeafT, RationalQuadraticArdKernel>)
            {
                const std::size_t ls_dim = !cached_ls.empty() ? cached_ls.size() : k.lengthscales().size();
                if (auto d = check_ard_dim<T>(x.cols(), ls_dim); !d.has_value())
                {
                    return d;
                }
                std::fill(out.begin(), out.end(), T{1});
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, ConstantKernel>)
            {
                std::fill(out.begin(), out.end(), static_cast<T>(k.constant()));
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, WhiteKernel>)
            {
                std::fill(out.begin(), out.end(), static_cast<T>(k.variance()));
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, LinearKernel>)
            {
                // Diagonal of LinearKernel equals variance * ||x_row||_2^2.
                const T var = static_cast<T>(k.variance());
                for (std::size_t row = 0; row < x.rows(); ++row)
                {
                    out[row] = var * dot_points(x, row, x, row);
                }
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, CustomKernel>)
            {
                return k.template term<T>().fill_diag(out);
            }
            else
            {
                // Stationary unit-amplitude isotropic leaves always equal 1 at zero distance.
                std::fill(out.begin(), out.end(), T{1});
                return {};
            }
        }, leaf
        );
    }

    /// @brief Evaluates the first log-hyperparameter derivative $\partial K / \partial \theta_{\text{param\_idx}}$
    /// from squared distances.
    /// @details Computes exact analytical log-theta gradients for isotropic RBF, Matern, Periodic, RationalQuadratic,
    /// Constant ($\partial c / \partial \log c = c$), White ($\sigma_w^2 \delta_{row,col}$), and `CustomKernel`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] leaf Kernel leaf variant.
    /// @param[in] dist Square squared-distance matrix.
    /// @param[out] d_k Output derivative matrix view.
    /// @param[in] param_idx Local hyperparameter index within `leaf`.
    /// @param[in] uplo Active matrix triangle to write.
    /// @param[in] exp_mode Exponential evaluation backend.
    /// @return Empty expected on success, or validation/index/operation error.
    /// @retval GprErrorCode::IndexOutOfRange If `param_idx >= leaf_num_params(leaf)`.
    /// @retval GprErrorCode::UnsupportedKernelOperation If `leaf` requires point coordinates.
    template <KernelScalarConcept T>
    std::expected<void, GprError> leaf_grad_dist(
        const KernelLeaf& leaf,
        MatrixRef<T> dist,
        MatrixMut<T> d_k,
        std::size_t param_idx,
        Triangle uplo,
        KernelExp exp_mode
    )
    {
        // Validate squared-distance matrix, output dimensions, and local hyperparameter index.
        if (auto v = validate_square_dist(dist, d_k, uplo); !v.has_value())
        {
            return v;
        }
        if (auto idx_ok = require_param_index(param_idx, leaf_num_params(leaf)); !idx_ok.has_value())
        {
            return idx_ok;
        }
        return std::visit(
            [&](const auto& k) -> std::expected<void, GprError>
        {
            using LeafT = std::decay_t<decltype(k)>;
            if constexpr (std::is_same_v<LeafT, RbfKernel>)
            {
                const T ell = static_cast<T>(k.lengthscale());
                write_square_dist(dist, d_k, uplo, [ell, exp_mode](T d2, std::size_t, std::size_t) {
                    return rbf_grad_d2(d2, ell, exp_mode);
                });
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, MaternKernel>)
            {
                const T scale = matern_scale<T>(k.nu()) / static_cast<T>(k.lengthscale());
                const MaternNu nu = k.nu();
                write_square_dist(
                    dist, d_k, uplo, [scale, nu, exp_mode](T d2, std::size_t, std::size_t)
                {
                    const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
                    return eval_matern_grad_g1(scale * r, nu, exp_mode);
                }
                );
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, PeriodicKernel>)
            {
                const T ell = static_cast<T>(k.lengthscale());
                const T period = static_cast<T>(k.period());
                write_square_dist(dist, d_k, uplo, [ell, period, param_idx, exp_mode](T d2, std::size_t, std::size_t) {
                    return periodic_grad_d2(d2, ell, period, param_idx, exp_mode);
                });
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, RationalQuadraticKernel>)
            {
                const T ell = static_cast<T>(k.lengthscale());
                const T alpha = static_cast<T>(k.alpha());
                write_square_dist(
                    dist, d_k, uplo, [ell, alpha, param_idx, exp_mode](T d2, std::size_t, std::size_t)
                {
                    const T clamped = d2 < T{0} ? T{0} : d2;
                    const T w = clamped / (T{2} * alpha * ell * ell);
                    return rq_grad_w(w, alpha, param_idx, 1U, exp_mode);
                }
                );
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, ConstantKernel>)
            {
                // Since theta = log(c), d(c)/d(log c) = c everywhere.
                const T c = static_cast<T>(k.constant());
                write_square_dist(dist, d_k, uplo, [c](T, std::size_t, std::size_t) { return c; });
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, WhiteKernel>)
            {
                // Since theta = log(v), d(v * delta)/d(log v) = v on the diagonal and 0 off-diagonal.
                const T v = static_cast<T>(k.variance());
                write_square_dist(dist, d_k, uplo, [v](T, std::size_t row, std::size_t col) {
                    return row == col ? v : T{0};
                });
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, CustomKernel>)
            {
                return k.template term<T>().grad(dist, d_k, param_idx, uplo);
            }
            else
            {
                return std::unexpected(
                    GprError::unsupported_kernel_operation("coordinate-dependent leaf gradient requires points")
                );
            }
        }, leaf
        );
    }

    /// @brief Evaluates the first log-hyperparameter derivative $\partial K / \partial \theta_{\text{param\_idx}}$
    /// from point coordinates.
    /// @details Supports all 10 built-in leaves (including per-dimension ARD lengthscale chain rules and LinearKernel)
    /// as well as `CustomKernel`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] leaf Kernel leaf variant.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[out] d_k Output derivative matrix of shape `(n, n)`.
    /// @param[in] param_idx Local hyperparameter index within `leaf`.
    /// @param[in] uplo Active matrix triangle to write.
    /// @param[in] exp_mode Exponential evaluation backend.
    /// @param[in] cached_ls Pre-cached ARD lengthscales, or empty to query `leaf`.
    /// @return Empty expected on success, or validation/dimension/index/evaluation error.
    /// @retval GprErrorCode::IndexOutOfRange If `param_idx >= leaf_num_params(leaf)`.
    /// @retval GprErrorCode::DimensionMismatch If `x.cols()` mismatches ARD lengthscale count.
    template <KernelScalarConcept T>
    std::expected<void, GprError> leaf_grad_points(
        const KernelLeaf& leaf,
        MatrixRef<T> x,
        MatrixMut<T> d_k,
        std::size_t param_idx,
        Triangle uplo,
        KernelExp exp_mode,
        std::span<const double> cached_ls
    )
    {
        // Validate point coordinates, output matrix shape, and local parameter index.
        if (auto v = validate_square_points(x, d_k); !v.has_value())
        {
            return v;
        }
        if (auto idx_ok = require_param_index(param_idx, leaf_num_params(leaf)); !idx_ok.has_value())
        {
            return idx_ok;
        }
        return std::visit(
            [&](const auto& k) -> std::expected<void, GprError>
        {
            using LeafT = std::decay_t<decltype(k)>;
            if constexpr (std::is_same_v<LeafT, RbfKernel>)
            {
                const T ell = static_cast<T>(k.lengthscale());
                write_square_points(x, d_k, uplo, [x, ell, exp_mode](std::size_t row, std::size_t col) {
                    return rbf_grad_d2(sq_dist_points(x, row, x, col), ell, exp_mode);
                });
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, RbfArdKernel>)
            {
                return with_ard_ls(
                    k, cached_ls, [&](std::span<const double> ls)
                {
                    if (auto d = check_ard_dim<T>(x.cols(), ls.size()); !d.has_value())
                    {
                        return d;
                    }
                    // For ARD RBF, d(-0.5 * s^2)/d(log ell_p) = ((x_r1,p - x_r2,p)^2 / ell_p^2).
                    return with_ard_inv_ell2<T>(
                        ls, T{1}, [&](std::span<const T> inv_ell2) -> std::expected<void, GprError>
                    {
                        const T inv_lp2 = inv_ell2[param_idx];
                        write_square_points(
                            x, d_k, uplo, [x, inv_ell2, param_idx, inv_lp2, exp_mode](std::size_t row, std::size_t col)
                        {
                            const T s2 = ard_scaled_sq_weights(x, row, x, col, inv_ell2);
                            const T diff = x(row, param_idx) - x.col(param_idx)[col];
                            return ((diff * diff) * inv_lp2) * eval_kernel_exp_jet(-T{0.5} * s2, exp_mode).d1;
                        }
                        );
                        return {};
                    }
                    );
                }
                );
            }
            else if constexpr (std::is_same_v<LeafT, MaternKernel>)
            {
                const T scale = matern_scale<T>(k.nu()) / static_cast<T>(k.lengthscale());
                const MaternNu nu = k.nu();
                write_square_points(
                    x, d_k, uplo, [x, scale, nu, exp_mode](std::size_t row, std::size_t col)
                {
                    const T d2 = sq_dist_points(x, row, x, col);
                    const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
                    return eval_matern_grad_g1(scale * r, nu, exp_mode);
                }
                );
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, MaternArdKernel>)
            {
                return with_ard_ls(
                    k, cached_ls, [&](std::span<const double> ls)
                {
                    if (auto d = check_ard_dim<T>(x.cols(), ls.size()); !d.has_value())
                    {
                        return d;
                    }
                    // For ARD Matern, dk/d(log ell_p) = (-u * k'(u)) * w_p where w_p = 2*nu * diff_p^2 / u^2.
                    const T c = matern_scale_sq<T>(k.nu());
                    const MaternNu nu = k.nu();
                    return with_ard_inv_ell2<T>(
                        ls, T{1}, [&](std::span<const T> inv_ell2) -> std::expected<void, GprError>
                    {
                        const T inv_lp2 = inv_ell2[param_idx];
                        write_square_points(
                            x, d_k, uplo,
                            [x, inv_ell2, c, nu, param_idx, inv_lp2, exp_mode](std::size_t row, std::size_t col)
                        {
                            const T u2 = c * ard_scaled_sq_weights(x, row, x, col, inv_ell2);
                            if (u2 <= T{0})
                            {
                                return T{0};
                            }
                            const T u = std::sqrt(u2);
                            const T diff = x(row, param_idx) - x.col(param_idx)[col];
                            const T w_p = (c * (diff * diff) * inv_lp2) / u2;
                            return eval_matern_grad_g1(u, nu, exp_mode) * w_p;
                        }
                        );
                        return {};
                    }
                    );
                }
                );
            }
            else if constexpr (std::is_same_v<LeafT, PeriodicKernel>)
            {
                const T ell = static_cast<T>(k.lengthscale());
                const T period = static_cast<T>(k.period());
                write_square_points(
                    x, d_k, uplo, [x, ell, period, param_idx, exp_mode](std::size_t row, std::size_t col)
                { return periodic_grad_d2(sq_dist_points(x, row, x, col), ell, period, param_idx, exp_mode); }
                );
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, RationalQuadraticKernel>)
            {
                const T ell = static_cast<T>(k.lengthscale());
                const T alpha = static_cast<T>(k.alpha());
                write_square_points(
                    x, d_k, uplo, [x, ell, alpha, param_idx, exp_mode](std::size_t row, std::size_t col)
                {
                    const T w = sq_dist_points(x, row, x, col) / (T{2} * alpha * ell * ell);
                    return rq_grad_w(w, alpha, param_idx, 1U, exp_mode);
                }
                );
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, RationalQuadraticArdKernel>)
            {
                return with_ard_ls(
                    k, cached_ls, [&](std::span<const double> ls)
                {
                    if (auto d = check_ard_dim<T>(x.cols(), ls.size()); !d.has_value())
                    {
                        return d;
                    }
                    // Parameter layout: [log(ell_0), ..., log(ell_{d-1}), log(alpha)].
                    const T alpha = static_cast<T>(k.alpha());
                    const std::size_t alpha_idx = ls.size();
                    return with_ard_inv_ell2<T>(
                        ls, T{1}, [&](std::span<const T> inv_ell2) -> std::expected<void, GprError>
                    {
                        const T inv_lp2 = (param_idx < alpha_idx) ? inv_ell2[param_idx] : T{0};
                        write_square_points(
                            x, d_k, uplo,
                            [x, inv_ell2, alpha, param_idx, alpha_idx, inv_lp2,
                                exp_mode](std::size_t row, std::size_t col)
                        {
                            const T s2 = ard_scaled_sq_weights(x, row, x, col, inv_ell2);
                            const T w = s2 / (T{2} * alpha);
                            const T b = T{1} + w;
                            const T d1_exp = eval_kernel_exp_jet(-alpha * std::log1p(w), exp_mode).d1;
                            if (param_idx < alpha_idx)
                            {
                                const T diff = x(row, param_idx) - x.col(param_idx)[col];
                                return (((diff * diff) * inv_lp2) / b) * d1_exp;
                            }
                            return alpha * ((w / b) - std::log1p(w)) * d1_exp;
                        }
                        );
                        return {};
                    }
                    );
                }
                );
            }
            else if constexpr (std::is_same_v<LeafT, ConstantKernel>)
            {
                const T c = static_cast<T>(k.constant());
                write_square_points(x, d_k, uplo, [c](std::size_t, std::size_t) { return c; });
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, LinearKernel>)
            {
                // Since theta = log(var), d(var * x_i . x_j)/d(log var) = var * (x_i . x_j).
                const T var = static_cast<T>(k.variance());
                write_square_points(x, d_k, uplo, [x, var](std::size_t row, std::size_t col) {
                    return var * dot_points(x, row, x, col);
                });
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, CustomKernel>)
            {
                auto dist_buf = make_square_dist_buf(x);
                const MatrixRef<T> dist_ref(dist_buf.data(), x.rows(), x.rows(), x.rows());
                return k.template term<T>().grad(dist_ref, d_k, param_idx, uplo);
            }
            else
            {
                const T var = static_cast<T>(k.variance());
                write_square_points(x, d_k, uplo, [var](std::size_t row, std::size_t col) {
                    return row == col ? var : T{0};
                });
                return {};
            }
        }, leaf
        );
    }

    /// @brief Evaluates the second log-hyperparameter derivative $\partial^2 K / (\partial \theta_i \partial \theta_j)$
    /// from squared distances.
    /// @details Supports isotropic RBF, Matern, Periodic, RationalQuadratic, Constant, White, and `CustomKernel`
    /// leaves.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] leaf Kernel leaf variant.
    /// @param[in] dist Square squared-distance matrix.
    /// @param[out] d2_k Output second-derivative matrix view.
    /// @param[in] i First local hyperparameter index within `leaf`.
    /// @param[in] j Second local hyperparameter index within `leaf`.
    /// @param[in] uplo Active matrix triangle to write.
    /// @param[in] exp_mode Exponential evaluation backend.
    /// @return Empty expected on success, or validation/index/operation error.
    /// @retval GprErrorCode::IndexOutOfRange If `i` or `j` is >= `leaf_num_params(leaf)`.
    /// @retval GprErrorCode::UnsupportedKernelOperation If `leaf` requires point coordinates.
    template <KernelScalarConcept T>
    std::expected<void, GprError> leaf_hess_dist(
        const KernelLeaf& leaf,
        MatrixRef<T> dist,
        MatrixMut<T> d2_k,
        std::size_t i,
        std::size_t j,
        Triangle uplo,
        KernelExp exp_mode
    )
    {
        // Validate distance matrix, output matrix, and both local parameter indices.
        if (auto v = validate_square_dist(dist, d2_k, uplo); !v.has_value())
        {
            return v;
        }
        const std::size_t np = leaf_num_params(leaf);
        if (auto idx_i = require_param_index(i, np); !idx_i.has_value())
        {
            return idx_i;
        }
        if (auto idx_j = require_param_index(j, np); !idx_j.has_value())
        {
            return idx_j;
        }
        return std::visit(
            [&](const auto& k) -> std::expected<void, GprError>
        {
            using LeafT = std::decay_t<decltype(k)>;
            if constexpr (std::is_same_v<LeafT, RbfKernel>)
            {
                const T ell = static_cast<T>(k.lengthscale());
                write_square_dist(dist, d2_k, uplo, [ell, exp_mode](T d2, std::size_t, std::size_t) {
                    return rbf_hess_d2(d2, ell, exp_mode);
                });
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, MaternKernel>)
            {
                const T scale = matern_scale<T>(k.nu()) / static_cast<T>(k.lengthscale());
                const MaternNu nu = k.nu();
                write_square_dist(
                    dist, d2_k, uplo, [scale, nu, exp_mode](T d2, std::size_t, std::size_t)
                {
                    const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
                    return eval_matern_iso_hess(scale * r, nu, exp_mode);
                }
                );
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, PeriodicKernel>)
            {
                const T ell = static_cast<T>(k.lengthscale());
                const T period = static_cast<T>(k.period());
                write_square_dist(dist, d2_k, uplo, [ell, period, i, j, exp_mode](T d2, std::size_t, std::size_t) {
                    return periodic_hess_d2(d2, ell, period, i, j, exp_mode);
                });
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, RationalQuadraticKernel>)
            {
                const T ell = static_cast<T>(k.lengthscale());
                const T alpha = static_cast<T>(k.alpha());
                write_square_dist(
                    dist, d2_k, uplo, [ell, alpha, i, j, exp_mode](T d2, std::size_t, std::size_t)
                {
                    const T clamped = d2 < T{0} ? T{0} : d2;
                    const T w = clamped / (T{2} * alpha * ell * ell);
                    return rq_hess_iso(w, alpha, i, j, exp_mode);
                }
                );
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, ConstantKernel>)
            {
                // For c = exp(theta), d^2(c)/d(theta^2) = c.
                const T c = static_cast<T>(k.constant());
                write_square_dist(dist, d2_k, uplo, [c](T, std::size_t, std::size_t) { return c; });
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, WhiteKernel>)
            {
                // For v = exp(theta), d^2(v * delta)/d(theta^2) = v * delta.
                const T v = static_cast<T>(k.variance());
                write_square_dist(dist, d2_k, uplo, [v](T, std::size_t row, std::size_t col) {
                    return row == col ? v : T{0};
                });
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, CustomKernel>)
            {
                return k.template term<T>().hess(dist, d2_k, i, j, uplo);
            }
            else
            {
                return std::unexpected(
                    GprError::unsupported_kernel_operation("coordinate-dependent leaf Hessian requires points")
                );
            }
        }, leaf
        );
    }

    /// @brief Evaluates the second log-hyperparameter derivative $\partial^2 K / (\partial \theta_i \partial \theta_j)$
    /// from point coordinates.
    /// @details Implements analytical second derivatives for all 10 built-in leaves (including ARD RBF, ARD Matern,
    /// ARD RationalQuadratic, and LinearKernel) plus `CustomKernel`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] leaf Kernel leaf variant.
    /// @param[in] x Input point matrix of shape `(n, d)`.
    /// @param[out] d2_k Output second-derivative matrix of shape `(n, n)`.
    /// @param[in] i First local hyperparameter index within `leaf`.
    /// @param[in] j Second local hyperparameter index within `leaf`.
    /// @param[in] uplo Active matrix triangle to write.
    /// @param[in] exp_mode Exponential evaluation backend.
    /// @param[in] cached_ls Pre-cached ARD lengthscales, or empty to query `leaf`.
    /// @return Empty expected on success, or validation/dimension/index/evaluation error.
    /// @retval GprErrorCode::IndexOutOfRange If `i` or `j` is >= `leaf_num_params(leaf)`.
    /// @retval GprErrorCode::DimensionMismatch If `x.cols()` mismatches ARD lengthscale count.
    template <KernelScalarConcept T>
    std::expected<void, GprError> leaf_hess_points(
        const KernelLeaf& leaf,
        MatrixRef<T> x,
        MatrixMut<T> d2_k,
        std::size_t i,
        std::size_t j,
        Triangle uplo,
        KernelExp exp_mode,
        std::span<const double> cached_ls
    )
    {
        // Validate point coordinates, output matrix shape, and local hyperparameter indices.
        if (auto v = validate_square_points(x, d2_k); !v.has_value())
        {
            return v;
        }
        const std::size_t np = leaf_num_params(leaf);
        if (auto idx_i = require_param_index(i, np); !idx_i.has_value())
        {
            return idx_i;
        }
        if (auto idx_j = require_param_index(j, np); !idx_j.has_value())
        {
            return idx_j;
        }
        return std::visit(
            [&](const auto& k) -> std::expected<void, GprError>
        {
            using LeafT = std::decay_t<decltype(k)>;
            if constexpr (std::is_same_v<LeafT, RbfKernel>)
            {
                const T ell = static_cast<T>(k.lengthscale());
                write_square_points(x, d2_k, uplo, [x, ell, exp_mode](std::size_t row, std::size_t col) {
                    return rbf_hess_d2(sq_dist_points(x, row, x, col), ell, exp_mode);
                });
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, RbfArdKernel>)
            {
                return with_ard_ls(
                    k, cached_ls, [&](std::span<const double> ls)
                {
                    if (auto d = check_ard_dim<T>(x.cols(), ls.size()); !d.has_value())
                    {
                        return d;
                    }
                    // For ARD RBF: d^2 k / (d theta_i d theta_j) = s_i^2 * s_j^2 * d2 - 2 * delta_ij * s_i^2 * d1.
                    const T ell_i = static_cast<T>(ls[i]);
                    const T ell_j = static_cast<T>(ls[j]);
                    write_square_points(
                        x, d2_k, uplo, [x, ls, i, j, ell_i, ell_j, exp_mode](std::size_t row, std::size_t col)
                    {
                        const T s2 = ard_scaled_sq(x, row, x, col, ls);
                        const ExpJet<T> jet = eval_kernel_exp_jet(-T{0.5} * s2, exp_mode);
                        const T di = (x(row, i) - x(col, i)) / ell_i;
                        const T si2 = di * di;
                        if (i == j)
                        {
                            return (si2 * si2 * jet.d2) - (T{2} * si2 * jet.d1);
                        }
                        const T dj = (x(row, j) - x(col, j)) / ell_j;
                        return si2 * (dj * dj) * jet.d2;
                    }
                    );
                    return std::expected<void, GprError>{};
                }
                );
            }
            else if constexpr (std::is_same_v<LeafT, MaternKernel>)
            {
                const T scale = matern_scale<T>(k.nu()) / static_cast<T>(k.lengthscale());
                const MaternNu nu = k.nu();
                write_square_points(
                    x, d2_k, uplo, [x, scale, nu, exp_mode](std::size_t row, std::size_t col)
                {
                    const T d2 = sq_dist_points(x, row, x, col);
                    const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
                    return eval_matern_iso_hess(scale * r, nu, exp_mode);
                }
                );
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, MaternArdKernel>)
            {
                return with_ard_ls(
                    k, cached_ls, [&](std::span<const double> ls)
                {
                    if (auto d = check_ard_dim<T>(x.cols(), ls.size()); !d.has_value())
                    {
                        return d;
                    }
                    // For ARD Matern: H_ij = (u^2 k''(u) - u k'(u)) * w_i * w_j - 2 * delta_ij * g1(u) * w_i.
                    const T c = matern_scale_sq<T>(k.nu());
                    const MaternNu nu = k.nu();
                    const T ell_i = static_cast<T>(ls[i]);
                    const T ell_j = static_cast<T>(ls[j]);
                    write_square_points(
                        x, d2_k, uplo, [x, ls, c, nu, i, j, ell_i, ell_j, exp_mode](std::size_t row, std::size_t col)
                    {
                        const T u2 = c * ard_scaled_sq(x, row, x, col, ls);
                        if (u2 <= T{0})
                        {
                            return T{0};
                        }
                        const T u = std::sqrt(u2);
                        const T di = (x(row, i) - x(col, i)) / ell_i;
                        const T wi = (c * di * di) / u2;
                        const T dj = (x(row, j) - x(col, j)) / ell_j;
                        const T wj = (c * dj * dj) / u2;
                        T val = eval_matern_ard_cross_coeff(u, nu, exp_mode) * wi * wj;
                        if (i == j)
                        {
                            val -= T{2} * eval_matern_grad_g1(u, nu, exp_mode) * wi;
                        }
                        return val;
                    }
                    );
                    return std::expected<void, GprError>{};
                }
                );
            }
            else if constexpr (std::is_same_v<LeafT, PeriodicKernel>)
            {
                const T ell = static_cast<T>(k.lengthscale());
                const T period = static_cast<T>(k.period());
                write_square_points(x, d2_k, uplo, [x, ell, period, i, j, exp_mode](std::size_t row, std::size_t col) {
                    return periodic_hess_d2(sq_dist_points(x, row, x, col), ell, period, i, j, exp_mode);
                });
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, RationalQuadraticKernel>)
            {
                const T ell = static_cast<T>(k.lengthscale());
                const T alpha = static_cast<T>(k.alpha());
                write_square_points(
                    x, d2_k, uplo, [x, ell, alpha, i, j, exp_mode](std::size_t row, std::size_t col)
                {
                    const T w = sq_dist_points(x, row, x, col) / (T{2} * alpha * ell * ell);
                    return rq_hess_iso(w, alpha, i, j, exp_mode);
                }
                );
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, RationalQuadraticArdKernel>)
            {
                return with_ard_ls(
                    k, cached_ls, [&](std::span<const double> ls)
                {
                    if (auto d = check_ard_dim<T>(x.cols(), ls.size()); !d.has_value())
                    {
                        return d;
                    }
                    // Handle (alpha, alpha), mixed (alpha, ell_k), and lengthscale (ell_i, ell_j) blocks.
                    const T alpha = static_cast<T>(k.alpha());
                    const std::size_t alpha_idx = ls.size();
                    write_square_points(
                        x, d2_k, uplo, [x, ls, alpha, i, j, alpha_idx, exp_mode](std::size_t row, std::size_t col)
                    {
                        const T s2 = ard_scaled_sq(x, row, x, col, ls);
                        const T w = s2 / (T{2} * alpha);
                        const T b = T{1} + w;
                        const ExpJet<T> jet = eval_kernel_exp_jet(-alpha * std::log1p(w), exp_mode);
                        const T g_alpha = alpha * ((w / b) - std::log1p(w));
                        if (i == alpha_idx && j == alpha_idx)
                        {
                            return ((g_alpha * g_alpha) * jet.d2) + ((g_alpha + ((alpha * w * w) / (b * b))) * jet.d1);
                        }
                        const std::size_t r1 = row;
                        const std::size_t r2 = col;
                        if (i == alpha_idx || j == alpha_idx)
                        {
                            const std::size_t ls_idx = (i == alpha_idx) ? j : i;
                            const T diff = (x(r1, ls_idx) - x(r2, ls_idx)) / static_cast<T>(ls[ls_idx]);
                            return ((diff * diff) / b) * ((g_alpha * jet.d2) + ((w / b) * jet.d1));
                        }
                        const T di = (x(r1, i) - x(r2, i)) / static_cast<T>(ls[i]);
                        const T dj = (x(r1, j) - x(r2, j)) / static_cast<T>(ls[j]);
                        const T si2 = di * di;
                        const T sj2 = dj * dj;
                        const T sq_term = (si2 * sj2) / (b * b);
                        T val = (sq_term * jet.d2) + ((sq_term / alpha) * jet.d1);
                        if (i == j)
                        {
                            val -= ((T{2} * si2) / b) * jet.d1;
                        }
                        return val;
                    }
                    );
                    return std::expected<void, GprError>{};
                }
                );
            }
            else if constexpr (std::is_same_v<LeafT, ConstantKernel>)
            {
                const T c = static_cast<T>(k.constant());
                write_square_points(x, d2_k, uplo, [c](std::size_t, std::size_t) { return c; });
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, LinearKernel>)
            {
                // Since var = exp(theta), d^2(var * x_i . x_j)/d(theta^2) = var * (x_i . x_j).
                const T var = static_cast<T>(k.variance());
                write_square_points(x, d2_k, uplo, [x, var](std::size_t row, std::size_t col) {
                    return var * dot_points(x, row, x, col);
                });
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, CustomKernel>)
            {
                auto dist_buf = make_square_dist_buf(x);
                const MatrixRef<T> dist_ref(dist_buf.data(), x.rows(), x.rows(), x.rows());
                return k.template term<T>().hess(dist_ref, d2_k, i, j, uplo);
            }
            else
            {
                const T var = static_cast<T>(k.variance());
                write_square_points(x, d2_k, uplo, [var](std::size_t row, std::size_t col) {
                    return row == col ? var : T{0};
                });
                return {};
            }
        }, leaf
        );
    }

    template std::expected<void, GprError> leaf_apply_dist<double>(
        const KernelLeaf&, MatrixRef<double>, MatrixMut<double>, Triangle, KernelExp
    );
    template std::expected<void, GprError> leaf_apply_dist<float>(
        const KernelLeaf&, MatrixRef<float>, MatrixMut<float>, Triangle, KernelExp
    );
    template std::expected<void, GprError> leaf_apply_points<double>(
        const KernelLeaf&, MatrixRef<double>, MatrixMut<double>, Triangle, KernelExp, std::span<const double>
    );
    template std::expected<void, GprError> leaf_apply_points<float>(
        const KernelLeaf&, MatrixRef<float>, MatrixMut<float>, Triangle, KernelExp, std::span<const double>
    );
    template std::expected<void, GprError> leaf_apply_cross_dist<double>(
        const KernelLeaf&, MatrixRef<double>, MatrixMut<double>, KernelExp
    );
    template std::expected<void, GprError> leaf_apply_cross_dist<float>(
        const KernelLeaf&, MatrixRef<float>, MatrixMut<float>, KernelExp
    );
    template std::expected<void, GprError> leaf_apply_cross_points<double>(
        const KernelLeaf&, MatrixRef<double>, MatrixRef<double>, MatrixMut<double>, KernelExp, std::span<const double>
    );
    template std::expected<void, GprError> leaf_apply_cross_points<float>(
        const KernelLeaf&, MatrixRef<float>, MatrixRef<float>, MatrixMut<float>, KernelExp, std::span<const double>
    );
    template std::expected<void, GprError> leaf_fill_diag<double>(
        const KernelLeaf&, MatrixRef<double>, std::span<double>, std::span<const double>
    );
    template std::expected<void, GprError> leaf_fill_diag<float>(
        const KernelLeaf&, MatrixRef<float>, std::span<float>, std::span<const double>
    );
    template std::expected<void, GprError> leaf_grad_dist<double>(
        const KernelLeaf&, MatrixRef<double>, MatrixMut<double>, std::size_t, Triangle, KernelExp
    );
    template std::expected<void, GprError> leaf_grad_dist<float>(
        const KernelLeaf&, MatrixRef<float>, MatrixMut<float>, std::size_t, Triangle, KernelExp
    );
    template std::expected<void, GprError> leaf_grad_points<double>(
        const KernelLeaf&,
        MatrixRef<double>,
        MatrixMut<double>,
        std::size_t,
        Triangle,
        KernelExp,
        std::span<const double>
    );
    template std::expected<void, GprError> leaf_grad_points<float>(
        const KernelLeaf&, MatrixRef<float>, MatrixMut<float>, std::size_t, Triangle, KernelExp, std::span<const double>
    );
    template std::expected<void, GprError> leaf_hess_dist<double>(
        const KernelLeaf&, MatrixRef<double>, MatrixMut<double>, std::size_t, std::size_t, Triangle, KernelExp
    );
    template std::expected<void, GprError> leaf_hess_dist<float>(
        const KernelLeaf&, MatrixRef<float>, MatrixMut<float>, std::size_t, std::size_t, Triangle, KernelExp
    );
    template std::expected<void, GprError> leaf_hess_points<double>(
        const KernelLeaf&,
        MatrixRef<double>,
        MatrixMut<double>,
        std::size_t,
        std::size_t,
        Triangle,
        KernelExp,
        std::span<const double>
    );
    template std::expected<void, GprError> leaf_hess_points<float>(
        const KernelLeaf&,
        MatrixRef<float>,
        MatrixMut<float>,
        std::size_t,
        std::size_t,
        Triangle,
        KernelExp,
        std::span<const double>
    );

    namespace
    {

        /// @brief Computes the column-major packed lower-triangular index for symmetric entry `(row, col)`.
        /// @details Maps `(row, col)` with $r = \max(\text{row}, \text{col})$ and $c = \min(\text{row}, \text{col})$
        /// to $c(2n - c - 1)/2 + r$.
        /// @param[in] n Square matrix side length.
        /// @param[in] row Row coordinate.
        /// @param[in] col Column coordinate.
        /// @return Linear offset into a packed lower-triangular block of length $n(n + 1)/2$.
        [[nodiscard]] inline std::size_t packed_lower_index(std::size_t n, std::size_t row, std::size_t col) noexcept
        {
            const std::size_t r = row >= col ? row : col;
            const std::size_t c = row >= col ? col : row;
            return ((c * ((2U * n) - c - 1U)) / 2U) + r;
        }

        /// @brief Computes ARD-scaled squared distance $s^2$ from packed differences using precomputed inverse-squared
        /// weights.
        /// @details Multiplies each dimension's packed entry by `weights[dim] = 1 / ell_dim^2` instead of dividing in
        /// the inner loop.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] packed Packed lower-triangular squared differences laid out as `dims x per_dim`.
        /// @param[in] n Number of points.
        /// @param[in] per_dim Stride between consecutive feature dimension blocks, equal to $n(n + 1)/2$.
        /// @param[in] weights Precomputed inverse-squared lengthscales.
        /// @param[in] row Row point index.
        /// @param[in] col Column point index.
        /// @return ARD-scaled squared distance $s^2$.
        template <KernelScalarConcept T>
        [[nodiscard]] T ard_scaled_sq_packed_weights(
            std::span<const T> packed,
            std::size_t n,
            std::size_t per_dim,
            std::span<const T> weights,
            std::size_t row,
            std::size_t col
        ) noexcept
        {
            if (row == col)
            {
                return T{0};
            }
            const std::size_t p = packed_lower_index(n, row, col);
            T acc = T{0};
            for (std::size_t dim = 0; dim < weights.size(); ++dim)
            {
                acc += packed[(dim * per_dim) + p] * weights[dim];
            }
            return acc;
        }

        /// @brief Accumulates packed ARD lower-triangular column slices into `out` and transforms via `slice_fn`.
        /// @details Requires `uplo` to be `Lower` or `Full` so packed storage is unit-stride contiguous within each
        /// column. Mirrors the lower triangle into the upper triangle when `uplo == Triangle::Full`.
        /// @tparam T Storage floating-point scalar type.
        /// @tparam SliceFn Callable invoked as `slice_fn(col_ptr, len)`.
        /// @param[in] ard_packed Packed lower-triangular per-dimension squared differences.
        /// @param[in] n Square matrix side length.
        /// @param[in] per_dim Elements per dimension block `n * (n + 1) / 2`.
        /// @param[in] weights Precomputed `1 / ell_d^2` weights.
        /// @param[out] out Target square matrix `(n, n)`.
        /// @param[in] uplo Active matrix region (`Lower` or `Full`).
        /// @param[in] slice_fn In-place transformation applied to each populated lower column slice.
        template <KernelScalarConcept T, typename SliceFn>
        void eval_packed_ard_lower_cols(
            std::span<const T> ard_packed,
            std::size_t n,
            std::size_t per_dim,
            std::span<const T> weights,
            MatrixMut<T> out,
            Triangle uplo,
            SliceFn&& slice_fn
        )
        {
            for_each_triangle_col(
                n, uplo, [&](std::size_t col, std::size_t row_begin, std::size_t row_end)
            {
                const std::size_t len = row_end - row_begin;
                if (len == 0U)
                {
                    return;
                }
                const std::size_t col_packed_base = ((col * ((2U * n) - col - 1U)) / 2U) + row_begin;
                T* col_ptr = &out(row_begin, col);
                detail::scale_set_slice(col_ptr, &ard_packed[col_packed_base], weights[0U], len);
                for (std::size_t dim = 1U; dim < weights.size(); ++dim)
                {
                    detail::axpy_slice(col_ptr, &ard_packed[(dim * per_dim) + col_packed_base], weights[dim], len);
                }
                out(col, col) = T{0};
                slice_fn(col_ptr, len);
            }
            );
            mirror_lower_if_full(out, uplo);
        }

        /// @brief Evaluates and writes a square symmetric matrix triangle or full matrix in column-parallel order.
        /// @details Uses area-balanced `linalg::parallel_for_lower_blocks` (`n >= 128U`) for `Lower` and `Full`
        /// triangles before mirroring strictly lower entries when `Full`.
        /// @tparam T Storage floating-point scalar type.
        /// @tparam Fn Callable invoked as `fn(row, col)` with `row >= col`.
        /// @param[in] n Square matrix side length.
        /// @param[out] out Target square matrix view.
        /// @param[in] uplo Active matrix region (`Lower`, `Upper`, or `Full`).
        /// @param[in] fn Element evaluation functor.
        template <KernelScalarConcept T, typename Fn>
        void write_square_packed(std::size_t n, MatrixMut<T> out, Triangle uplo, Fn&& fn) noexcept
        {
            if (uplo == Triangle::Upper)
            {
                // Parallelize across columns when filling only the upper triangle.
                const auto n_cols = static_cast<std::ptrdiff_t>(n);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 128U)
#endif
                for (std::ptrdiff_t col_s = 0; col_s < n_cols; ++col_s)
                {
                    const auto col = static_cast<std::size_t>(col_s);
                    for (std::size_t row = 0; row <= col; ++row)
                    {
                        out(row, col) = fn(row, col);
                    }
                }
                return;
            }
            auto run_range = [&](std::size_t col_begin, std::size_t col_end)
            {
                for (std::size_t col = col_begin; col < col_end; ++col)
                {
                    for (std::size_t row = col; row < n; ++row)
                    {
                        out(row, col) = fn(row, col);
                    }
                }
            };
            if (n >= 128U)
            {
                linalg::parallel_for_lower_blocks(n, 0U, run_range);
            }
            else
            {
                run_range(0U, n);
            }
            mirror_lower_if_full(out, uplo);
        }

        /// @brief Evaluates an ARD kernel leaf into `out` directly from precomputed lower-packed $s^2$ distances.
        /// @details For `Lower` and `Full` triangles, copies and transforms contiguous column slices via SSE2 helpers
        /// (`exp_scaled_nonpos_in_place` / `matern_s2_in_place` / `transform_rq_slice_in_place`) and mirrors if `Full`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] leaf ARD kernel leaf variant.
        /// @param[in] s2_packed Precomputed lower-triangular $s^2$ array of length `n * (n + 1) / 2`.
        /// @param[in] n Square matrix side length.
        /// @param[out] out Destination square Gram matrix `(n, n)`.
        /// @param[in] uplo Active matrix region (`Lower`, `Upper`, or `Full`).
        /// @param[in] exp_mode Exponential evaluation backend.
        template <KernelScalarConcept T>
        void eval_ard_from_s2_packed(
            const KernelLeaf& leaf,
            std::span<const T> s2_packed,
            std::size_t n,
            MatrixMut<T> out,
            Triangle uplo,
            KernelExp exp_mode
        )
        {
            auto eval_lower_slices = [&](auto&& slice_fn)
            {
                for_each_triangle_col(
                    n, uplo, [&](std::size_t col, std::size_t row_begin, std::size_t row_end)
                {
                    const std::size_t len = row_end - row_begin;
                    if (len == 0U)
                    {
                        return;
                    }
                    const std::size_t col_packed_base = ((col * ((2U * n) - col - 1U)) / 2U) + row_begin;
                    T* col_ptr = &out(row_begin, col);
                    slice_fn(col_ptr, &s2_packed[col_packed_base], len, col);
                }
                );
                mirror_lower_if_full(out, uplo);
            };

            if (std::holds_alternative<RbfArdKernel>(leaf))
            {
                if (uplo != Triangle::Upper)
                {
                    eval_lower_slices([exp_mode, &out](T* col_ptr, const T* s2_col, std::size_t len, std::size_t col)
                    {
                        detail::scale_set_slice(col_ptr, s2_col, T{1}, len);
                        out(col, col) = T{0};
                        detail::exp_scaled_nonpos_in_place(col_ptr, len, T{-0.5}, exp_mode);
                    });
                    return;
                }
                write_square_packed(
                    n, out, uplo, [s2_packed, n, exp_mode](std::size_t row, std::size_t col)
                {
                    const T s2 = (row == col) ? T{0} : s2_packed[packed_lower_index(n, row, col)];
                    return eval_kernel_exp(-T{0.5} * (s2 < T{0} ? T{0} : s2), exp_mode);
                }
                );
                return;
            }
            if (const auto* mat = std::get_if<MaternArdKernel>(&leaf))
            {
                const T c = matern_scale_sq<T>(mat->nu());
                const MaternNu nu = mat->nu();
                if (uplo != Triangle::Upper)
                {
                    eval_lower_slices([c, nu, exp_mode,
                                          &out](T* col_ptr, const T* s2_col, std::size_t len, std::size_t col)
                    {
                        std::copy_n(s2_col, len, col_ptr);
                        out(col, col) = T{0};
                        detail::matern_s2_in_place(col_ptr, len, c, nu, exp_mode);
                    });
                    return;
                }
                write_square_packed(
                    n, out, uplo, [s2_packed, n, c, nu, exp_mode](std::size_t row, std::size_t col)
                {
                    const T s2 = (row == col) ? T{0} : s2_packed[packed_lower_index(n, row, col)];
                    const T u2 = c * (s2 < T{0} ? T{0} : s2);
                    const T u = u2 > T{0} ? std::sqrt(u2) : T{0};
                    return eval_matern_u(u, nu, exp_mode);
                }
                );
                return;
            }
            if (const auto* rq = std::get_if<RationalQuadraticArdKernel>(&leaf))
            {
                const T alpha = static_cast<T>(rq->alpha());
                const T rq_scale = T{1} / (T{2} * alpha);
                if (uplo != Triangle::Upper)
                {
                    eval_lower_slices([rq_scale, alpha, exp_mode,
                                          &out](T* col_ptr, const T* s2_col, std::size_t len, std::size_t col)
                    {
                        detail::scale_set_slice(col_ptr, s2_col, rq_scale, len);
                        out(col, col) = T{0};
                        transform_rq_slice_in_place(col_ptr, len, T{1}, alpha, exp_mode);
                    });
                    return;
                }
                write_square_packed(
                    n, out, uplo, [s2_packed, n, rq_scale, alpha, exp_mode](std::size_t row, std::size_t col)
                {
                    const T s2 = (row == col) ? T{0} : s2_packed[packed_lower_index(n, row, col)];
                    const T w = (s2 < T{0} ? T{0} : s2) * rq_scale;
                    return eval_kernel_exp(-alpha * std::log1p(w), exp_mode);
                }
                );
            }
        }

    }  // namespace

    /// @brief Evaluates an ARD kernel leaf from precomputed packed per-dimension squared differences.
    /// @details Supports `RbfArdKernel`, `MaternArdKernel`, and `RationalQuadraticArdKernel`, rejecting any other
    /// leaf variant with `GprErrorCode::UnsupportedKernelOperation`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] leaf Kernel leaf variant (must be an ARD leaf).
    /// @param[in] ls Per-dimension lengthscale values in user units.
    /// @param[in] ard_packed Packed lower-triangular squared-difference array of length `ls.size() * n * (n + 1) / 2`.
    /// @param[in] n Number of training points.
    /// @param[out] out Target square Gram matrix of shape `(n, n)`.
    /// @param[in] uplo Active matrix region to write.
    /// @param[in] exp_mode Exponential evaluation backend.
    /// @return Empty expected on success, or validation/shape/operation error.
    /// @retval GprErrorCode::EmptyInput If `n == 0` or `ls.empty()`.
    /// @retval GprErrorCode::LengthMismatch If `out` is not `(n, n)` or `ard_packed.size()` mismatches `ls.size() *
    /// n * (n + 1) / 2`.
    /// @retval GprErrorCode::UnsupportedKernelOperation If `leaf` is not an ARD leaf.
    template <KernelScalarConcept T>
    std::expected<void, GprError> leaf_apply_ard_packed(
        const KernelLeaf& leaf,
        std::span<const double> ls,
        std::span<const T> ard_packed,
        std::size_t n,
        MatrixMut<T> out,
        Triangle uplo,
        KernelExp exp_mode
    )
    {
        // Validate non-empty inputs, square output dimensions, and packed ARD buffer size.
        if (n == 0U || ls.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        if (out.rows() != n || out.cols() != n)
        {
            return std::unexpected(GprError::length_mismatch("output matrix shape mismatch"));
        }
        const std::size_t per_dim = (n * (n + 1U)) / 2U;
        if (ard_packed.size() != ls.size() * per_dim)
        {
            return std::unexpected(GprError::length_mismatch("ARD cache size mismatch"));
        }
        if (std::holds_alternative<RbfArdKernel>(leaf))
        {
            return with_ard_inv_ell2<T>(
                ls, T{1}, [&](std::span<const T> weights) -> std::expected<void, GprError>
            {
                if (uplo != Triangle::Upper)
                {
                    eval_packed_ard_lower_cols(
                        ard_packed, n, per_dim, weights, out, uplo, [exp_mode](T* col_ptr, std::size_t len) noexcept
                    { detail::exp_scaled_nonpos_in_place(col_ptr, len, T{-0.5}, exp_mode); }
                    );
                }
                else
                {
                    write_square_packed(
                        n, out, uplo, [ard_packed, n, per_dim, weights, exp_mode](std::size_t row, std::size_t col)
                    {
                        const T s2 = ard_scaled_sq_packed_weights(ard_packed, n, per_dim, weights, row, col);
                        return eval_kernel_exp(-T{0.5} * s2, exp_mode);
                    }
                    );
                }
                return {};
            }
            );
        }
        if (const auto* mat = std::get_if<MaternArdKernel>(&leaf))
        {
            const T c = matern_scale_sq<T>(mat->nu());
            const MaternNu nu = mat->nu();
            return with_ard_inv_ell2<T>(
                ls, T{1}, [&](std::span<const T> weights) -> std::expected<void, GprError>
            {
                if (uplo != Triangle::Upper)
                {
                    eval_packed_ard_lower_cols(
                        ard_packed, n, per_dim, weights, out, uplo,
                        [c, nu, exp_mode](T* col_ptr, std::size_t len) noexcept
                    { detail::matern_s2_in_place(col_ptr, len, c, nu, exp_mode); }
                    );
                }
                else
                {
                    write_square_packed(
                        n, out, uplo,
                        [ard_packed, n, per_dim, weights, c, nu, exp_mode](std::size_t row, std::size_t col)
                    {
                        const T u2 = c * ard_scaled_sq_packed_weights(ard_packed, n, per_dim, weights, row, col);
                        const T u = u2 > T{0} ? std::sqrt(u2) : T{0};
                        return eval_matern_u(u, nu, exp_mode);
                    }
                    );
                }
                return {};
            }
            );
        }
        if (const auto* rq = std::get_if<RationalQuadraticArdKernel>(&leaf))
        {
            const T alpha = static_cast<T>(rq->alpha());
            const T rq_scale = T{1} / (T{2} * alpha);
            return with_ard_inv_ell2<T>(
                ls, rq_scale, [&](std::span<const T> weights) -> std::expected<void, GprError>
            {
                if (uplo != Triangle::Upper)
                {
                    eval_packed_ard_lower_cols(
                        ard_packed, n, per_dim, weights, out, uplo,
                        [alpha, exp_mode](T* col_ptr, std::size_t len) noexcept
                    { transform_rq_slice_in_place(col_ptr, len, T{1}, alpha, exp_mode); }
                    );
                }
                else
                {
                    write_square_packed(
                        n, out, uplo,
                        [ard_packed, n, per_dim, weights, alpha, exp_mode](std::size_t row, std::size_t col)
                    {
                        const T w = ard_scaled_sq_packed_weights(ard_packed, n, per_dim, weights, row, col);
                        return eval_kernel_exp(-alpha * std::log1p(w), exp_mode);
                    }
                    );
                }
                return {};
            }
            );
        }
        return std::unexpected(GprError::unsupported_kernel_operation("leaf does not consume packed ARD distances"));
    }

    /// @brief Evaluates an ARD kernel leaf using an incrementally updated lower-triangular $s^2$ cache.
    /// @details Maintains `ard_s2_cache` across single-coordinate optimization proposals using $O(n^2)$ delta-weight
    /// AXPY updates on changed lengthscale dimensions when numerically safe, and rebuilds cleanly in $O(d \cdot n^2)$
    /// otherwise. Once `ard_s2_cache` matches `ls`, evaluates `out` in $O(n^2)$ via `eval_ard_from_s2_packed`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] leaf ARD kernel leaf variant.
    /// @param[in] ls Current per-dimension lengthscale values in user units.
    /// @param[in] ard_packed Packed lower-triangular `(dx_d)^2` buffer.
    /// @param[in] n Number of training points.
    /// @param[out] out Target square Gram matrix `(n, n)`.
    /// @param[in] uplo Active matrix region (`Lower`, `Upper`, or `Full`).
    /// @param[in] exp_mode Exponential evaluation backend.
    /// @param[in,out] ard_s2_cache Pre-allocated per-leaf packed $s^2$ buffer of length `n * (n + 1) / 2`.
    /// @param[in,out] prev_ls Pre-allocated lengthscale snapshot buffer of length at least `ls.size()`.
    /// @param[in,out] s2_valid True while `ard_s2_cache` corresponds to `prev_ls[0..d-1]`.
    /// @param[in,out] inc_steps Consecutive incremental AXPY updates applied since last rebuild.
    /// @return Empty expected on success, or validation/shape/operation error.
    template <KernelScalarConcept T>
    std::expected<void, GprError> leaf_apply_ard_packed_incremental(
        const KernelLeaf& leaf,
        std::span<const double> ls,
        std::span<const T> ard_packed,
        std::size_t n,
        MatrixMut<T> out,
        Triangle uplo,
        KernelExp exp_mode,
        std::span<T> ard_s2_cache,
        std::span<double> prev_ls,
        bool& s2_valid,
        std::size_t& inc_steps
    )
    {
        if (n == 0U || ls.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        if (out.rows() != n || out.cols() != n)
        {
            return std::unexpected(GprError::length_mismatch("output matrix shape mismatch"));
        }
        const std::size_t per_dim = (n * (n + 1U)) / 2U;
        if (ard_packed.size() != ls.size() * per_dim)
        {
            return std::unexpected(GprError::length_mismatch("ARD cache size mismatch"));
        }
        if (!std::holds_alternative<RbfArdKernel>(leaf) && !std::holds_alternative<MaternArdKernel>(leaf)
            && !std::holds_alternative<RationalQuadraticArdKernel>(leaf))
        {
            return std::unexpected(
                GprError::unsupported_kernel_operation("leaf does not consume packed ARD distances")
            );
        }
        if (ard_s2_cache.size() != per_dim || prev_ls.size() < ls.size())
        {
            s2_valid = false;
            inc_steps = 0U;
            return leaf_apply_ard_packed(leaf, ls, ard_packed, n, out, uplo, exp_mode);
        }

        const std::size_t d = ls.size();
        constexpr std::size_t k_max_inc_steps = std::same_as<T, float> ? 16U : 64U;
        constexpr double k_min_w_ratio = std::same_as<T, float> ? 0.25 : 1e-2;
        constexpr double k_max_w_ratio = std::same_as<T, float> ? 4.0 : 1e2;

        bool can_increment = s2_valid && (inc_steps < k_max_inc_steps);
        std::size_t changed_dims = 0U;
        if (can_increment)
        {
            for (std::size_t dim = 0U; dim < d; ++dim)
            {
                if (std::bit_cast<std::uint64_t>(prev_ls[dim]) != std::bit_cast<std::uint64_t>(ls[dim]))
                {
                    ++changed_dims;
                    const double old_ell = prev_ls[dim];
                    const double new_ell = ls[dim];
                    if (!(old_ell > 0.0) || !(new_ell > 0.0))
                    {
                        can_increment = false;
                        break;
                    }
                    const double old_w = 1.0 / (old_ell * old_ell);
                    const double new_w = 1.0 / (new_ell * new_ell);
                    const double ratio = new_w / old_w;
                    // Guard against underflowed weights or catastrophic cancellation on large jumps.
                    if (!std::isfinite(old_w) || !std::isfinite(new_w)
                        || !(ratio >= k_min_w_ratio && ratio <= k_max_w_ratio))
                    {
                        can_increment = false;
                        break;
                    }
                }
            }
            if ((changed_dims * 2U) > d)
            {
                can_increment = false;
            }
        }

        if (can_increment)
        {
            if (changed_dims > 0U)
            {
                for (std::size_t dim = 0U; dim < d; ++dim)
                {
                    if (std::bit_cast<std::uint64_t>(prev_ls[dim]) == std::bit_cast<std::uint64_t>(ls[dim]))
                    {
                        continue;
                    }
                    const double old_w = 1.0 / (prev_ls[dim] * prev_ls[dim]);
                    const double new_w = 1.0 / (ls[dim] * ls[dim]);
                    const T delta_w = static_cast<T>(new_w - old_w);
                    if (delta_w != T{0})
                    {
                        const T* dim_ptr = &ard_packed[dim * per_dim];
                        constexpr std::size_t k_chunk = 1024U;
                        if (per_dim >= 8192U)
                        {
                            const auto n_chunks = static_cast<std::ptrdiff_t>((per_dim + k_chunk - 1U) / k_chunk);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                            for (std::ptrdiff_t b = 0; b < n_chunks; ++b)
                            {
                                const std::size_t start = static_cast<std::size_t>(b) * k_chunk;
                                const std::size_t count = std::min(k_chunk, per_dim - start);
                                detail::axpy_slice(ard_s2_cache.data() + start, dim_ptr + start, delta_w, count);
                            }
                        }
                        else
                        {
                            detail::axpy_slice(ard_s2_cache.data(), dim_ptr, delta_w, per_dim);
                        }
                    }
                }
                ++inc_steps;
            }
            std::ranges::copy(ls, prev_ls.begin());
            eval_ard_from_s2_packed<T>(leaf, ard_s2_cache, n, out, uplo, exp_mode);
            return {};
        }

        // Rebuild packed lower s^2 cleanly from scratch across all dimensions, fusing output transform while
        // the column slice is hot in L1 cache when filling Lower or Full triangles.
        const bool fuse_lower_out = (uplo != Triangle::Upper);
        const bool is_rbf = std::holds_alternative<RbfArdKernel>(leaf);
        const auto* mat = std::get_if<MaternArdKernel>(&leaf);
        const auto* rq = std::get_if<RationalQuadraticArdKernel>(&leaf);
        const T m_c = (mat != nullptr) ? matern_scale_sq<T>(mat->nu()) : T{1};
        const MaternNu m_nu = (mat != nullptr) ? mat->nu() : MaternNu::OneHalf;
        const T rq_alpha = (rq != nullptr) ? static_cast<T>(rq->alpha()) : T{1};
        const T rq_scale = T{1} / (T{2} * rq_alpha);

        if (auto res = with_ard_inv_ell2<T>(
                ls, T{1},
                [&](std::span<const T> weights) -> std::expected<void, GprError>
        {
            auto build_cols = [&](std::size_t col_begin, std::size_t col_end)
            {
                for (std::size_t col = col_begin; col < col_end; ++col)
                {
                    const std::size_t len = n - col;
                    const std::size_t col_packed_base = ((col * ((2U * n) - col - 1U)) / 2U) + col;
                    T* s2_ptr = &ard_s2_cache[col_packed_base];
                    detail::scale_set_slice(s2_ptr, &ard_packed[col_packed_base], weights[0U], len);
                    for (std::size_t dim = 1U; dim < weights.size(); ++dim)
                    {
                        detail::axpy_slice(s2_ptr, &ard_packed[(dim * per_dim) + col_packed_base], weights[dim], len);
                    }
                    *s2_ptr = T{0};

                    if (fuse_lower_out)
                    {
                        T* out_col = &out(col, col);
                        if (is_rbf)
                        {
                            detail::scale_set_slice(out_col, s2_ptr, T{1}, len);
                            *out_col = T{0};
                            detail::exp_scaled_nonpos_in_place(out_col, len, T{-0.5}, exp_mode);
                        }
                        else if (mat != nullptr)
                        {
                            std::copy_n(s2_ptr, len, out_col);
                            *out_col = T{0};
                            detail::matern_s2_in_place(out_col, len, m_c, m_nu, exp_mode);
                        }
                        else
                        {
                            detail::scale_set_slice(out_col, s2_ptr, rq_scale, len);
                            *out_col = T{0};
                            transform_rq_slice_in_place(out_col, len, T{1}, rq_alpha, exp_mode);
                        }
                    }
                }
            };
            if (n >= 128U)
            {
                linalg::parallel_for_lower_blocks(n, 0U, build_cols);
            }
            else
            {
                build_cols(0U, n);
            }
            return {};
        }
            );
            !res.has_value())
        {
            return res;
        }
        s2_valid = true;
        inc_steps = 0U;
        std::ranges::copy(ls, prev_ls.begin());
        if (fuse_lower_out)
        {
            mirror_lower_if_full(out, uplo);
        }
        else
        {
            eval_ard_from_s2_packed<T>(leaf, ard_s2_cache, n, out, uplo, exp_mode);
        }
        return {};
    }

    /// @brief Evaluates the first log-hyperparameter derivative of an ARD leaf from packed squared differences.
    /// @details Uses precomputed per-dimension packed blocks `ard_packed` to compute $\partial K / \partial \theta_p$
    /// for `RbfArdKernel`, `MaternArdKernel`, and `RationalQuadraticArdKernel` without reading `X`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] leaf Kernel leaf variant (must be an ARD leaf).
    /// @param[in] ls Per-dimension lengthscale values in user units.
    /// @param[in] ard_packed Packed lower-triangular squared-difference array.
    /// @param[in] n Number of training points.
    /// @param[out] d_k Target square derivative matrix of shape `(n, n)`.
    /// @param[in] param_idx Local hyperparameter index within `leaf`.
    /// @param[in] uplo Active matrix region to write.
    /// @param[in] exp_mode Exponential evaluation backend.
    /// @return Empty expected on success, or validation/shape/index/operation error.
    /// @retval GprErrorCode::EmptyInput If `n == 0` or `ls.empty()`.
    /// @retval GprErrorCode::LengthMismatch If `d_k` or `ard_packed` dimensions mismatch `n` and `ls.size()`.
    /// @retval GprErrorCode::IndexOutOfRange If `param_idx >= leaf_num_params(leaf)`.
    /// @retval GprErrorCode::UnsupportedKernelOperation If `leaf` is not an ARD leaf.
    template <KernelScalarConcept T>
    std::expected<void, GprError> leaf_grad_ard_packed(
        const KernelLeaf& leaf,
        std::span<const double> ls,
        std::span<const T> ard_packed,
        std::size_t n,
        MatrixMut<T> d_k,
        std::size_t param_idx,
        Triangle uplo,
        KernelExp exp_mode
    )
    {
        // Validate non-empty inputs, square derivative matrix shape, packed buffer size, and parameter index.
        if (n == 0U || ls.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        if (d_k.rows() != n || d_k.cols() != n)
        {
            return std::unexpected(GprError::length_mismatch("output matrix shape mismatch"));
        }
        const std::size_t per_dim = (n * (n + 1U)) / 2U;
        if (ard_packed.size() != ls.size() * per_dim)
        {
            return std::unexpected(GprError::length_mismatch("ARD cache size mismatch"));
        }
        if (param_idx >= leaf_num_params(leaf))
        {
            return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
        }
        if (std::holds_alternative<RbfArdKernel>(leaf))
        {
            // For RBF-ARD: dk/d(log ell_p) = (dx_p^2 / ell_p^2) * exp'(-0.5 * s^2).
            return with_ard_inv_ell2<T>(
                ls, T{1}, [&](std::span<const T> weights) -> std::expected<void, GprError>
            {
                const T inv_l2 = weights[param_idx];
                write_square_packed(
                    n, d_k, uplo,
                    [ard_packed, n, per_dim, weights, param_idx, inv_l2, exp_mode](std::size_t row, std::size_t col)
                {
                    const std::size_t p = packed_lower_index(n, row, col);
                    const T s2 = ard_scaled_sq_packed_weights(ard_packed, n, per_dim, weights, row, col);
                    const T sp2 = ard_packed[(param_idx * per_dim) + p] * inv_l2;
                    return sp2 * eval_kernel_exp_jet(-T{0.5} * s2, exp_mode).d1;
                }
                );
                return {};
            }
            );
        }
        if (const auto* mat = std::get_if<MaternArdKernel>(&leaf))
        {
            // For Matern-ARD: dk/d(log ell_p) = g1(u) * (2*nu * dx_p^2 / ell_p^2) / u^2, with limit 0 at u <= 0.
            const T c = matern_scale_sq<T>(mat->nu());
            const MaternNu nu = mat->nu();
            return with_ard_inv_ell2<T>(
                ls, T{1}, [&](std::span<const T> weights) -> std::expected<void, GprError>
            {
                const T inv_l2 = weights[param_idx];
                write_square_packed(
                    n, d_k, uplo,
                    [ard_packed, n, per_dim, weights, c, nu, param_idx, inv_l2,
                        exp_mode](std::size_t row, std::size_t col)
                {
                    const std::size_t p = packed_lower_index(n, row, col);
                    const T u2 = c * ard_scaled_sq_packed_weights(ard_packed, n, per_dim, weights, row, col);
                    if (u2 <= T{0})
                    {
                        return T{0};
                    }
                    const T u = std::sqrt(u2);
                    const T wp = (c * ard_packed[(param_idx * per_dim) + p] * inv_l2) / u2;
                    return eval_matern_grad_g1(u, nu, exp_mode) * wp;
                }
                );
                return {};
            }
            );
        }
        if (const auto* rq = std::get_if<RationalQuadraticArdKernel>(&leaf))
        {
            // For RQ-ARD: indices < ls.size() differentiate wrt log(ell_p); index == ls.size() differentiates alpha.
            const T alpha = static_cast<T>(rq->alpha());
            const std::size_t alpha_idx = ls.size();
            return with_ard_inv_ell2<T>(
                ls, T{1}, [&](std::span<const T> weights) -> std::expected<void, GprError>
            {
                const T inv_l2 = (param_idx < alpha_idx) ? weights[param_idx] : T{0};
                write_square_packed(
                    n, d_k, uplo,
                    [ard_packed, n, per_dim, weights, alpha, param_idx, alpha_idx, inv_l2,
                        exp_mode](std::size_t row, std::size_t col)
                {
                    const std::size_t p = packed_lower_index(n, row, col);
                    const T s2 = ard_scaled_sq_packed_weights(ard_packed, n, per_dim, weights, row, col);
                    const T w = s2 / (T{2} * alpha);
                    const T b = T{1} + w;
                    const T d1_exp = eval_kernel_exp_jet(-alpha * std::log1p(w), exp_mode).d1;
                    if (param_idx < alpha_idx)
                    {
                        const T sp2 = ard_packed[(param_idx * per_dim) + p] * inv_l2;
                        return (sp2 / b) * d1_exp;
                    }
                    return alpha * ((w / b) - std::log1p(w)) * d1_exp;
                }
                );
                return {};
            }
            );
        }
        return std::unexpected(GprError::unsupported_kernel_operation("leaf does not consume packed ARD distances"));
    }

    /// @brief Evaluates the second log-hyperparameter derivative of an ARD leaf from packed squared differences.
    /// @details Computes $\partial^2 K / (\partial \theta_i \partial \theta_j)$ for `RbfArdKernel`, `MaternArdKernel`,
    /// and `RationalQuadraticArdKernel` (including lengthscale-alpha cross terms) using packed ARD distances.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] leaf Kernel leaf variant (must be an ARD leaf).
    /// @param[in] ls Per-dimension lengthscale values in user units.
    /// @param[in] ard_packed Packed lower-triangular squared-difference array.
    /// @param[in] n Number of training points.
    /// @param[out] d2_k Target square second-derivative matrix of shape `(n, n)`.
    /// @param[in] i First local hyperparameter index within `leaf`.
    /// @param[in] j Second local hyperparameter index within `leaf`.
    /// @param[in] uplo Active matrix region to write.
    /// @param[in] exp_mode Exponential evaluation backend.
    /// @return Empty expected on success, or validation/shape/index/operation error.
    /// @retval GprErrorCode::EmptyInput If `n == 0` or `ls.empty()`.
    /// @retval GprErrorCode::LengthMismatch If `d2_k` or `ard_packed` dimensions mismatch `n` and `ls.size()`.
    /// @retval GprErrorCode::IndexOutOfRange If `i` or `j` is >= `leaf_num_params(leaf)`.
    /// @retval GprErrorCode::UnsupportedKernelOperation If `leaf` is not an ARD leaf.
    template <KernelScalarConcept T>
    std::expected<void, GprError> leaf_hess_ard_packed(
        const KernelLeaf& leaf,
        std::span<const double> ls,
        std::span<const T> ard_packed,
        std::size_t n,
        MatrixMut<T> d2_k,
        std::size_t i,
        std::size_t j,
        Triangle uplo,
        KernelExp exp_mode
    )
    {
        // Validate non-empty inputs, square output matrix shape, packed buffer size, and parameter indices.
        if (n == 0U || ls.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        if (d2_k.rows() != n || d2_k.cols() != n)
        {
            return std::unexpected(GprError::length_mismatch("output matrix shape mismatch"));
        }
        const std::size_t per_dim = (n * (n + 1U)) / 2U;
        if (ard_packed.size() != ls.size() * per_dim)
        {
            return std::unexpected(GprError::length_mismatch("ARD cache size mismatch"));
        }
        const std::size_t np = leaf_num_params(leaf);
        if (i >= np || j >= np)
        {
            return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
        }
        if (std::holds_alternative<RbfArdKernel>(leaf))
        {
            // For RBF-ARD: s_i^2 * s_j^2 * d2 - 2 * delta_ij * s_i^2 * d1.
            return with_ard_inv_ell2<T>(
                ls, T{1}, [&](std::span<const T> weights) -> std::expected<void, GprError>
            {
                const T inv_li2 = weights[i];
                const T inv_lj2 = weights[j];
                write_square_packed(
                    n, d2_k, uplo,
                    [ard_packed, n, per_dim, weights, i, j, inv_li2, inv_lj2,
                        exp_mode](std::size_t row, std::size_t col)
                {
                    const std::size_t p = packed_lower_index(n, row, col);
                    const T s2 = ard_scaled_sq_packed_weights(ard_packed, n, per_dim, weights, row, col);
                    const ExpJet<T> jet = eval_kernel_exp_jet(-T{0.5} * s2, exp_mode);
                    const T si2 = ard_packed[(i * per_dim) + p] * inv_li2;
                    if (i == j)
                    {
                        return (si2 * si2 * jet.d2) - (T{2} * si2 * jet.d1);
                    }
                    const T sj2 = ard_packed[(j * per_dim) + p] * inv_lj2;
                    return si2 * sj2 * jet.d2;
                }
                );
                return {};
            }
            );
        }
        if (const auto* mat = std::get_if<MaternArdKernel>(&leaf))
        {
            // For Matern-ARD: ard_cross * w_i * w_j - 2 * delta_ij * g1 * w_i, returning 0 at u <= 0.
            const T c = matern_scale_sq<T>(mat->nu());
            const MaternNu nu = mat->nu();
            return with_ard_inv_ell2<T>(
                ls, T{1}, [&](std::span<const T> weights) -> std::expected<void, GprError>
            {
                const T inv_li2 = weights[i];
                const T inv_lj2 = weights[j];
                write_square_packed(
                    n, d2_k, uplo,
                    [ard_packed, n, per_dim, weights, c, nu, i, j, inv_li2, inv_lj2,
                        exp_mode](std::size_t row, std::size_t col)
                {
                    const std::size_t p = packed_lower_index(n, row, col);
                    const T u2 = c * ard_scaled_sq_packed_weights(ard_packed, n, per_dim, weights, row, col);
                    if (u2 <= T{0})
                    {
                        return T{0};
                    }
                    const T u = std::sqrt(u2);
                    const T wi = (c * ard_packed[(i * per_dim) + p] * inv_li2) / u2;
                    const T wj = (c * ard_packed[(j * per_dim) + p] * inv_lj2) / u2;
                    T val = eval_matern_ard_cross_coeff(u, nu, exp_mode) * wi * wj;
                    if (i == j)
                    {
                        val -= T{2} * eval_matern_grad_g1(u, nu, exp_mode) * wi;
                    }
                    return val;
                }
                );
                return {};
            }
            );
        }
        if (const auto* rq = std::get_if<RationalQuadraticArdKernel>(&leaf))
        {
            // For RQ-ARD: evaluate (alpha, alpha), mixed (alpha, ell_k), and (ell_i, ell_j) blocks.
            const T alpha = static_cast<T>(rq->alpha());
            const std::size_t alpha_idx = ls.size();
            return with_ard_inv_ell2<T>(
                ls, T{1}, [&](std::span<const T> weights) -> std::expected<void, GprError>
            {
                write_square_packed(
                    n, d2_k, uplo,
                    [ard_packed, n, per_dim, weights, alpha, i, j, alpha_idx,
                        exp_mode](std::size_t row, std::size_t col)
                {
                    const std::size_t p = packed_lower_index(n, row, col);
                    const T s2 = ard_scaled_sq_packed_weights(ard_packed, n, per_dim, weights, row, col);
                    const T w = s2 / (T{2} * alpha);
                    const T b = T{1} + w;
                    const ExpJet<T> jet = eval_kernel_exp_jet(-alpha * std::log1p(w), exp_mode);
                    const T g_alpha = alpha * ((w / b) - std::log1p(w));
                    if (i == alpha_idx && j == alpha_idx)
                    {
                        return ((g_alpha * g_alpha) * jet.d2) + ((g_alpha + ((alpha * w * w) / (b * b))) * jet.d1);
                    }
                    if (i == alpha_idx || j == alpha_idx)
                    {
                        const std::size_t ls_idx = (i == alpha_idx) ? j : i;
                        const T s_ls2 = ard_packed[(ls_idx * per_dim) + p] * weights[ls_idx];
                        return (s_ls2 / b) * ((g_alpha * jet.d2) + ((w / b) * jet.d1));
                    }
                    const T si2 = ard_packed[(i * per_dim) + p] * weights[i];
                    const T sj2 = ard_packed[(j * per_dim) + p] * weights[j];
                    const T sq_term = (si2 * sj2) / (b * b);
                    T val = (sq_term * jet.d2) + ((sq_term / alpha) * jet.d1);
                    if (i == j)
                    {
                        val -= ((T{2} * si2) / b) * jet.d1;
                    }
                    return val;
                }
                );
                return {};
            }
            );
        }
        return std::unexpected(GprError::unsupported_kernel_operation("leaf does not consume packed ARD distances"));
    }

    template std::expected<void, GprError> leaf_apply_ard_packed<double>(
        const KernelLeaf&,
        std::span<const double>,
        std::span<const double>,
        std::size_t,
        MatrixMut<double>,
        Triangle,
        KernelExp
    );
    template std::expected<void, GprError> leaf_apply_ard_packed<float>(
        const KernelLeaf&,
        std::span<const double>,
        std::span<const float>,
        std::size_t,
        MatrixMut<float>,
        Triangle,
        KernelExp
    );
    template std::expected<void, GprError> leaf_apply_ard_packed_incremental<double>(
        const KernelLeaf&,
        std::span<const double>,
        std::span<const double>,
        std::size_t,
        MatrixMut<double>,
        Triangle,
        KernelExp,
        std::span<double>,
        std::span<double>,
        bool&,
        std::size_t&
    );
    template std::expected<void, GprError> leaf_apply_ard_packed_incremental<float>(
        const KernelLeaf&,
        std::span<const double>,
        std::span<const float>,
        std::size_t,
        MatrixMut<float>,
        Triangle,
        KernelExp,
        std::span<float>,
        std::span<double>,
        bool&,
        std::size_t&
    );
    template std::expected<void, GprError> leaf_grad_ard_packed<double>(
        const KernelLeaf&,
        std::span<const double>,
        std::span<const double>,
        std::size_t,
        MatrixMut<double>,
        std::size_t,
        Triangle,
        KernelExp
    );
    template std::expected<void, GprError> leaf_grad_ard_packed<float>(
        const KernelLeaf&,
        std::span<const double>,
        std::span<const float>,
        std::size_t,
        MatrixMut<float>,
        std::size_t,
        Triangle,
        KernelExp
    );
    template std::expected<void, GprError> leaf_hess_ard_packed<double>(
        const KernelLeaf&,
        std::span<const double>,
        std::span<const double>,
        std::size_t,
        MatrixMut<double>,
        std::size_t,
        std::size_t,
        Triangle,
        KernelExp
    );
    template std::expected<void, GprError> leaf_hess_ard_packed<float>(
        const KernelLeaf&,
        std::span<const double>,
        std::span<const float>,
        std::size_t,
        MatrixMut<float>,
        std::size_t,
        std::size_t,
        Triangle,
        KernelExp
    );

    /// @brief Contracts all hyperparameters of an ARD leaf against symmetric weight matrix `weight` in one radial pass.
    /// @details Builds scaled squared distances $s^2$ once per lower-triangular column, converts them in place to
    /// weighted radial derivative factors (also accumulating value or RQ-alpha contractions into `col_partials`),
    /// and contracts every dimension deterministically in ascending column order with zero heap allocation for $d \le
    /// 64$.
    /// @tparam T Storage floating-point scalar type (`double` or `float`).
    /// @param[in] leaf ARD kernel leaf variant (`RbfArdKernel`, `MaternArdKernel`, or `RationalQuadraticArdKernel`).
    /// @param[in] ls Per-dimension lengthscale vector in user units.
    /// @param[in] x Column-major `n x d` point matrix (read when `ard_packed` is empty).
    /// @param[in] ard_packed Optional packed lower-triangular `(dx_d)^2` cache of length `d * n * (n + 1) / 2`.
    /// @param[in] weight Symmetric `n x n` adjoint weight matrix (lower triangle read).
    /// @param[in,out] scratch Reusable `n x n` scratch matrix for single-pass radial derivative weights.
    /// @param[in,out] col_partials Per-column partial reduction buffer of length at least `n`.
    /// @param[out] out_grads Output derivative span sized to `leaf_num_params(leaf)`.
    /// @param[in] need_value True when $\langle \text{weight}, K_{\text{leaf}} \rangle_F$ is requested back.
    /// @param[in] exp_mode Exponential evaluation backend (`Accurate` or `FastApprox`).
    /// @return $\langle \text{weight}, K_{\text{leaf}} \rangle_F$ if `need_value` is true (else `0.0`), or error.
    /// @retval GprErrorCode::UnsupportedKernelOperation When `leaf` is not an ARD leaf.
    /// @retval GprErrorCode::EmptyInput When `ls`, `weight`, or uncached `x` is empty.
    /// @retval GprErrorCode::LengthMismatch When shapes or `out_grads` length mismatch.
    /// @retval GprErrorCode::DimensionMismatch When `x.cols() != ls.size()`.
    /// @retval GprErrorCode::WorkspaceTooSmall When `scratch` or `col_partials` is undersized.
    template <KernelScalarConcept T>
    std::expected<double, GprError> leaf_ard_weighted_grads(
        const KernelLeaf& leaf,
        std::span<const double> ls,
        MatrixRef<T> x,
        std::span<const T> ard_packed,
        MatrixRef<T> weight,
        MatrixMut<T> scratch,
        std::span<double> col_partials,
        std::span<double> out_grads,
        bool need_value,
        KernelExp exp_mode
    )
    {
        const bool is_rbf = std::holds_alternative<RbfArdKernel>(leaf);
        const auto* mat = std::get_if<MaternArdKernel>(&leaf);
        const auto* rq = std::get_if<RationalQuadraticArdKernel>(&leaf);
        if (!is_rbf && mat == nullptr && rq == nullptr)
        {
            return std::unexpected(GprError::unsupported_kernel_operation("leaf is not an ARD kernel"));
        }
        if (ls.empty() || weight.is_empty() || weight.data() == nullptr)
        {
            return std::unexpected(GprError::empty_input());
        }
        const std::size_t n = weight.rows();
        if (weight.cols() != n)
        {
            return std::unexpected(GprError::length_mismatch("weight matrix must be square"));
        }
        if (scratch.rows() != n || scratch.cols() != n || col_partials.size() < n)
        {
            return std::unexpected(GprError::workspace_too_small());
        }
        const std::size_t np = leaf_num_params(leaf);
        if (out_grads.size() != np)
        {
            return std::unexpected(GprError::length_mismatch("gradient output span length mismatch"));
        }
        const bool use_packed = !ard_packed.empty();
        const std::size_t per_dim = (n * (n + 1U)) / 2U;
        if (use_packed)
        {
            if (ard_packed.size() != ls.size() * per_dim)
            {
                return std::unexpected(GprError::length_mismatch("ARD cache size mismatch"));
            }
        }
        else
        {
            if (x.is_empty() || x.data() == nullptr)
            {
                return std::unexpected(GprError::empty_input());
            }
            if (x.rows() != n)
            {
                return std::unexpected(GprError::length_mismatch("point matrix row count mismatch"));
            }
            if (auto d_ok = check_ard_dim<T>(x.cols(), ls.size()); !d_ok.has_value())
            {
                return std::unexpected(d_ok.error());
            }
        }

        double leaf_val = 0.0;
        auto weight_res = with_ard_inv_ell2<T>(
            ls, T{1}, [&](std::span<const T> weights) -> std::expected<void, GprError>
        {
            const T m_c = (mat != nullptr) ? matern_scale_sq<T>(mat->nu()) : T{1};
            const MaternNu m_nu = (mat != nullptr) ? mat->nu() : MaternNu::OneHalf;
            const T rq_alpha = (rq != nullptr) ? static_cast<T>(rq->alpha()) : T{1};
            const bool track_col_partials = (rq != nullptr) || need_value;

            // Compute s^2 and weighted radial derivative factors once per lower-triangular column.
            auto radial_cols = [&](std::size_t col_begin, std::size_t col_end)
            {
                for (std::size_t col = col_begin; col < col_end; ++col)
                {
                    const std::size_t len = n - col;
                    T* col_ptr = &scratch(col, col);
                    if (use_packed)
                    {
                        const std::size_t col_packed_base = ((col * ((2U * n) - col - 1U)) / 2U) + col;
                        detail::scale_set_slice(col_ptr, &ard_packed[col_packed_base], weights[0U], len);
                        for (std::size_t dim = 1U; dim < weights.size(); ++dim)
                        {
                            detail::axpy_slice(
                                col_ptr, &ard_packed[(dim * per_dim) + col_packed_base], weights[dim], len
                            );
                        }
                    }
                    else
                    {
                        detail::set_scaled_sq_diff_slice(col_ptr, x.col(0U).data() + col, x(col, 0U), weights[0U], len);
                        for (std::size_t dim = 1U; dim < weights.size(); ++dim)
                        {
                            const auto col_vec = x.col(dim);
                            detail::add_scaled_sq_diff_slice(
                                col_ptr, col_vec.data() + col, col_vec[col], weights[dim], len
                            );
                        }
                    }
                    scratch(col, col) = T{0};

                    double partial = (need_value && rq == nullptr) ? static_cast<double>(weight(col, col)) : 0.0;
                    if (is_rbf)
                    {
                        for (std::size_t row = col + 1U; row < n; ++row)
                        {
                            const T s2 = scratch(row, col);
                            const ExpJet<T> jet = eval_kernel_exp_jet(-T{0.5} * s2, exp_mode);
                            const T w_sym = T{2} * weight(row, col);
                            if (need_value)
                            {
                                partial += static_cast<double>(w_sym) * static_cast<double>(jet.v);
                            }
                            scratch(row, col) = w_sym * jet.d1;
                        }
                    }
                    else if (mat != nullptr)
                    {
                        for (std::size_t row = col + 1U; row < n; ++row)
                        {
                            const T u2 = m_c * scratch(row, col);
                            const T w_sym = T{2} * weight(row, col);
                            if (u2 <= T{0})
                            {
                                if (need_value)
                                {
                                    partial += static_cast<double>(w_sym);
                                }
                                scratch(row, col) = T{0};
                            }
                            else
                            {
                                const T u = std::sqrt(u2);
                                const MaternJetVals<T> mj = eval_matern_jet_vals(u, m_nu, exp_mode);
                                if (need_value)
                                {
                                    partial += static_cast<double>(w_sym) * static_cast<double>(mj.val);
                                }
                                scratch(row, col) = w_sym * ((m_c * mj.g1) / u2);
                            }
                        }
                    }
                    else
                    {
                        for (std::size_t row = col + 1U; row < n; ++row)
                        {
                            const T clamped = scratch(row, col) < T{0} ? T{0} : scratch(row, col);
                            const T w = clamped / (T{2} * rq_alpha);
                            const T b = T{1} + w;
                            const T l1p = std::log1p(w);
                            const ExpJet<T> jet = eval_kernel_exp_jet(-rq_alpha * l1p, exp_mode);
                            const T w_sym = T{2} * weight(row, col);
                            partial +=
                                static_cast<double>(w_sym) * static_cast<double>(rq_alpha * ((w / b) - l1p) * jet.d1);
                            scratch(row, col) = w_sym * (jet.d1 / b);
                        }
                    }

                    if (track_col_partials)
                    {
                        col_partials[col] = partial;
                    }
                }
            };
            if (n >= 128U)
            {
                linalg::parallel_for_lower_blocks(n, 0U, radial_cols);
            }
            else
            {
                radial_cols(0U, n);
            }

            if (rq != nullptr)
            {
                double alpha_sum = 0.0;
                for (std::size_t col = 0U; col < n; ++col)
                {
                    alpha_sum += col_partials[col];
                }
                out_grads[ls.size()] = alpha_sum;
            }
            else if (need_value)
            {
                double v_sum = 0.0;
                for (std::size_t col = 0U; col < n; ++col)
                {
                    v_sum += col_partials[col];
                }
                leaf_val = v_sum;
            }

            // Contract every ARD lengthscale dimension against the weighted radial matrix in scratch.
            for (std::size_t dim = 0U; dim < weights.size(); ++dim)
            {
                const T inv_l2 = weights[dim];
                auto dim_contract = [&](std::size_t col_begin, std::size_t col_end)
                {
                    for (std::size_t col = col_begin; col < col_end; ++col)
                    {
                        if (col + 1U >= n)
                        {
                            col_partials[col] = 0.0;
                            continue;
                        }
                        const T* w_rad_col = &scratch(0U, col);
                        double acc = 0.0;
                        if (use_packed)
                        {
                            const T* dim_col = &ard_packed[(dim * per_dim) + ((col * ((2U * n) - col - 1U)) / 2U)];
                            for (std::size_t row = col + 1U; row < n; ++row)
                            {
                                acc += static_cast<double>(w_rad_col[row]) * static_cast<double>(dim_col[row] * inv_l2);
                            }
                        }
                        else
                        {
                            const T* x_dim = x.col(dim).data();
                            const T x_col = x_dim[col];
                            for (std::size_t row = col + 1U; row < n; ++row)
                            {
                                const T diff = x_dim[row] - x_col;
                                acc +=
                                    static_cast<double>(w_rad_col[row]) * static_cast<double>((diff * diff) * inv_l2);
                            }
                        }
                        col_partials[col] = acc;
                    }
                };
                if (n >= 384U)
                {
                    linalg::parallel_for_lower_blocks(n, 0U, dim_contract);
                }
                else
                {
                    dim_contract(0U, n);
                }
                double dim_total = 0.0;
                for (std::size_t col = 0U; col < n; ++col)
                {
                    dim_total += col_partials[col];
                }
                out_grads[dim] = dim_total;
            }
            return {};
        }
        );
        if (!weight_res.has_value())
        {
            return std::unexpected(std::move(weight_res).error());
        }

        if (rq != nullptr && need_value)
        {
            // When both RQ alpha and leaf value are needed inside an unkept product, evaluate K_leaf into scratch
            // after all derivative contractions have consumed it.
            if (auto ev = use_packed
                    ? leaf_apply_ard_packed(leaf, ls, ard_packed, n, scratch, Triangle::Lower, exp_mode)
                    : leaf_apply_points(leaf, x, scratch, Triangle::Lower, exp_mode, ls);
                !ev.has_value())
            {
                return std::unexpected(std::move(ev).error());
            }
            const auto n_cols_s = static_cast<std::ptrdiff_t>(n);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 384U)
#endif
            for (std::ptrdiff_t col_s = 0; col_s < n_cols_s; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                const auto diag = static_cast<double>(weight(col, col)) * static_cast<double>(scratch(col, col));
                double off = 0.0;
                for (std::size_t row = col + 1U; row < n; ++row)
                {
                    off += static_cast<double>(weight(row, col)) * static_cast<double>(scratch(row, col));
                }
                col_partials[col] = diag + (2.0 * off);
            }
            double v_sum = 0.0;
            for (std::size_t col = 0U; col < n; ++col)
            {
                v_sum += col_partials[col];
            }
            leaf_val = v_sum;
        }
        return leaf_val;
    }

    template std::expected<double, GprError> leaf_ard_weighted_grads<double>(
        const KernelLeaf&,
        std::span<const double>,
        MatrixRef<double>,
        std::span<const double>,
        MatrixRef<double>,
        MatrixMut<double>,
        std::span<double>,
        std::span<double>,
        bool,
        KernelExp
    );
    template std::expected<double, GprError> leaf_ard_weighted_grads<float>(
        const KernelLeaf&,
        std::span<const double>,
        MatrixRef<float>,
        std::span<const float>,
        MatrixRef<float>,
        MatrixMut<float>,
        std::span<double>,
        std::span<double>,
        bool,
        KernelExp
    );

    /// @brief Evaluates all hyperparameters and optional value contraction of an isotropic or linear leaf in one pass.
    /// @details Shares distance/radial/exponential evaluations across gradient parameters and optional value
    /// contraction over disjoint columns, parallelizing when `col_partials.size() >= 3 * n` and `n >= 128`, then
    /// summing column partials in strict ascending order for OpenMP thread-count determinism.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] leaf Isotropic-distance or `LinearKernel` leaf variant.
    /// @param[in] x Column-major `(n, d)` coordinate matrix (required for `LinearKernel` or empty `dist`).
    /// @param[in] dist Optional precomputed `(n, n)` squared-distance matrix.
    /// @param[in] weight Symmetric `(n, n)` adjoint matrix (lower triangle read).
    /// @param[in,out] scratch Reusable `(n, n)` scratch buffer for columnwise coordinate distances or dot products.
    /// @param[in,out] col_partials Reduction buffer (`>= 3 * n` enables multi-term OpenMP column reductions).
    /// @param[out] out_grads Output span sized to `leaf_num_params(leaf)`.
    /// @param[in] need_value True to also return $\langle \text{weight}, K_{\text{leaf}} \rangle_F$.
    /// @param[in] exp_mode Exponential evaluation backend (`Accurate` or `FastApprox`).
    /// @return $\langle \text{weight}, K_{\text{leaf}} \rangle_F$ if `need_value`, else `0.0`, or validation error.
    /// @retval GprErrorCode::UnsupportedKernelOperation When `leaf` is not isotropic-distance or `LinearKernel`.
    /// @retval GprErrorCode::EmptyInput When `weight`, required `x`, or supplied `dist` is empty or null.
    /// @retval GprErrorCode::LengthMismatch When shapes or `out_grads.size()` mismatch.
    /// @retval GprErrorCode::WorkspaceTooSmall When `scratch` or `col_partials` is undersized.
    template <KernelScalarConcept T>
    std::expected<double, GprError> leaf_iso_weighted_grads(
        const KernelLeaf& leaf,
        MatrixRef<T> x,
        MatrixRef<T> dist,
        MatrixRef<T> weight,
        MatrixMut<T> scratch,
        std::span<double> col_partials,
        std::span<double> out_grads,
        bool need_value,
        KernelExp exp_mode
    )
    {
        const auto* rbf = std::get_if<RbfKernel>(&leaf);
        const auto* mat = std::get_if<MaternKernel>(&leaf);
        const auto* per = std::get_if<PeriodicKernel>(&leaf);
        const auto* rq = std::get_if<RationalQuadraticKernel>(&leaf);
        const auto* lin = std::get_if<LinearKernel>(&leaf);
        if (rbf == nullptr && mat == nullptr && per == nullptr && rq == nullptr && lin == nullptr)
        {
            return std::unexpected(
                GprError::unsupported_kernel_operation("leaf is not supported by single-pass isotropic contraction")
            );
        }
        if (weight.is_empty() || weight.data() == nullptr)
        {
            return std::unexpected(GprError::empty_input());
        }
        const std::size_t n = weight.rows();
        if (weight.cols() != n)
        {
            return std::unexpected(GprError::length_mismatch("weight matrix must be square"));
        }
        const std::size_t np = leaf_num_params(leaf);
        if (out_grads.size() != np)
        {
            return std::unexpected(GprError::length_mismatch("gradient output span length mismatch"));
        }
        if (scratch.rows() != n || scratch.cols() != n || col_partials.size() < n)
        {
            return std::unexpected(GprError::workspace_too_small());
        }
        const bool use_dist = (lin == nullptr) && !dist.is_empty();
        if (use_dist)
        {
            if (dist.data() == nullptr)
            {
                return std::unexpected(GprError::empty_input());
            }
            if (dist.rows() != n || dist.cols() != n)
            {
                return std::unexpected(GprError::length_mismatch("distance matrix dimension mismatch"));
            }
        }
        else
        {
            if (x.is_empty() || x.data() == nullptr)
            {
                return std::unexpected(GprError::empty_input());
            }
            if (x.rows() != n)
            {
                return std::unexpected(GprError::length_mismatch("point matrix row count mismatch"));
            }
        }

        // Precompute leaf constants outside column loops.
        const T rbf_l2 = (rbf != nullptr) ? static_cast<T>(rbf->lengthscale() * rbf->lengthscale()) : T{1};
        const T mat_scale = (mat != nullptr) ? (matern_scale<T>(mat->nu()) / static_cast<T>(mat->lengthscale())) : T{1};
        const MaternNu mat_nu = (mat != nullptr) ? mat->nu() : MaternNu::OneHalf;
        const T per_l2 = (per != nullptr) ? static_cast<T>(per->lengthscale() * per->lengthscale()) : T{1};
        const T per_pi_p = (per != nullptr) ? (std::numbers::pi_v<T> / static_cast<T>(per->period())) : T{1};
        const T rq_alpha = (rq != nullptr) ? static_cast<T>(rq->alpha()) : T{1};
        const T rq_denom =
            (rq != nullptr) ? (T{2} * rq_alpha * static_cast<T>(rq->lengthscale() * rq->lengthscale())) : T{1};
        const T lin_var = (lin != nullptr) ? static_cast<T>(lin->variance()) : T{1};
        const std::size_t n_dims = use_dist ? 0U : x.cols();
        const bool has_param1 = (np >= 2U);

        // Process column `col` independently, reading either cached `dist` or computing slices in `scratch`.
        auto process_col = [&](std::size_t col, double& out_g0, double& out_g1, double& out_val) noexcept
        {
            const std::size_t len = n - col;
            const T* const w_col = &weight(0U, col);
            if (lin != nullptr)
            {
                T* const dot_col = &scratch(col, col);
                detail::set_dot_slice(dot_col, x.col(0U).data() + col, x(col, 0U), len);
                for (std::size_t dim = 1U; dim < n_dims; ++dim)
                {
                    const auto col_vec = x.col(dim);
                    detail::add_dot_slice(dot_col, col_vec.data() + col, col_vec[col], len);
                }
                const double g_diag = static_cast<double>(w_col[col]) * static_cast<double>(dot_col[0] * lin_var);
                double g_off = 0.0;
                for (std::size_t r = 1U; r < len; ++r)
                {
                    g_off += static_cast<double>(w_col[col + r]) * static_cast<double>(dot_col[r] * lin_var);
                }
                out_g0 = g_diag + (2.0 * g_off);
                out_g1 = 0.0;
                out_val = need_value ? out_g0 : 0.0;
                return;
            }

            const T* d2_col = nullptr;
            if (use_dist)
            {
                d2_col = &dist(0U, col);
            }
            else
            {
                T* const dist_slice = &scratch(col, col);
                detail::set_scaled_sq_diff_slice(dist_slice, x.col(0U).data() + col, x(col, 0U), T{1}, len);
                for (std::size_t dim = 1U; dim < n_dims; ++dim)
                {
                    const auto col_vec = x.col(dim);
                    detail::add_scaled_sq_diff_slice(dist_slice, col_vec.data() + col, col_vec[col], T{1}, len);
                }
                scratch(col, col) = T{0};
                d2_col = &scratch(0U, col);
            }

            // Evaluate the diagonal entry explicitly only if caller-supplied `dist` contains positive diagonal d^2.
            const auto w_diag = static_cast<double>(w_col[col]);
            double val_acc = (need_value && !use_dist) ? w_diag : 0.0;
            double g0_diag = 0.0;
            double g1_diag = 0.0;
            double g0_off = 0.0;
            double g1_off = 0.0;
            double val_off = 0.0;

            if (rbf != nullptr)
            {
                const T inv_2l2 = T{1} / (T{2} * rbf_l2);
                if (use_dist)
                {
                    const T d2_0 = d2_col[col];
                    if (d2_0 > T{0})
                    {
                        const T z0 = d2_0 * inv_2l2;
                        const ExpJet<T> j0 = eval_kernel_exp_jet(-z0, exp_mode);
                        g0_diag = w_diag * static_cast<double>((T{2} * z0) * j0.d1);
                        if (need_value)
                        {
                            val_acc = w_diag * static_cast<double>(j0.v);
                        }
                    }
                    else if (need_value)
                    {
                        val_acc = w_diag;
                    }
                }
                for (std::size_t row = col + 1U; row < n; ++row)
                {
                    const T d2 = d2_col[row];
                    const T clamped = d2 < T{0} ? T{0} : d2;
                    const T z = clamped * inv_2l2;
                    const ExpJet<T> jet = eval_kernel_exp_jet(-z, exp_mode);
                    const auto w_rc = static_cast<double>(w_col[row]);
                    g0_off += w_rc * static_cast<double>((T{2} * z) * jet.d1);
                    if (need_value)
                    {
                        val_off += w_rc * static_cast<double>(jet.v);
                    }
                }
            }
            else if (mat != nullptr)
            {
                if (use_dist)
                {
                    const T d2_0 = d2_col[col];
                    if (d2_0 > T{0})
                    {
                        const MaternJetVals<T> m0 = eval_matern_jet_vals(mat_scale * std::sqrt(d2_0), mat_nu, exp_mode);
                        g0_diag = w_diag * static_cast<double>(m0.g1);
                        if (need_value)
                        {
                            val_acc = w_diag * static_cast<double>(m0.val);
                        }
                    }
                    else if (need_value)
                    {
                        val_acc = w_diag;
                    }
                }
                for (std::size_t row = col + 1U; row < n; ++row)
                {
                    const T d2 = d2_col[row];
                    const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
                    const MaternJetVals<T> mj = eval_matern_jet_vals(mat_scale * r, mat_nu, exp_mode);
                    const auto w_rc = static_cast<double>(w_col[row]);
                    g0_off += w_rc * static_cast<double>(mj.g1);
                    if (need_value)
                    {
                        val_off += w_rc * static_cast<double>(mj.val);
                    }
                }
            }
            else if (per != nullptr)
            {
                if (use_dist)
                {
                    const T d2_0 = d2_col[col];
                    if (d2_0 > T{0})
                    {
                        const T phi0 = per_pi_p * std::sqrt(d2_0);
                        const T s0 = std::sin(phi0);
                        const T c0 = std::cos(phi0);
                        const ExpJet<T> j0 = eval_kernel_exp_jet(-(T{2} * s0 * s0) / per_l2, exp_mode);
                        g0_diag = w_diag * static_cast<double>(((T{4} * s0 * s0) / per_l2) * j0.d1);
                        g1_diag = w_diag * static_cast<double>(((T{4} * phi0 * s0 * c0) / per_l2) * j0.d1);
                        if (need_value)
                        {
                            val_acc = w_diag * static_cast<double>(j0.v);
                        }
                    }
                    else if (need_value)
                    {
                        val_acc = w_diag;
                    }
                }
                for (std::size_t row = col + 1U; row < n; ++row)
                {
                    const T d2 = d2_col[row];
                    const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
                    const T phi = per_pi_p * r;
                    const T s = std::sin(phi);
                    const T c = std::cos(phi);
                    const ExpJet<T> jet = eval_kernel_exp_jet(-(T{2} * s * s) / per_l2, exp_mode);
                    const auto w_rc = static_cast<double>(w_col[row]);
                    g0_off += w_rc * static_cast<double>(((T{4} * s * s) / per_l2) * jet.d1);
                    g1_off += w_rc * static_cast<double>(((T{4} * phi * s * c) / per_l2) * jet.d1);
                    if (need_value)
                    {
                        val_off += w_rc * static_cast<double>(jet.v);
                    }
                }
            }
            else
            {
                if (use_dist)
                {
                    const T d2_0 = d2_col[col];
                    if (d2_0 > T{0})
                    {
                        const T w0 = d2_0 / rq_denom;
                        const T b0 = T{1} + w0;
                        const T l0 = std::log1p(w0);
                        const ExpJet<T> j0 = eval_kernel_exp_jet(-rq_alpha * l0, exp_mode);
                        g0_diag = w_diag * static_cast<double>(((T{2} * rq_alpha * w0) / b0) * j0.d1);
                        g1_diag = w_diag * static_cast<double>((rq_alpha * ((w0 / b0) - l0)) * j0.d1);
                        if (need_value)
                        {
                            val_acc = w_diag * static_cast<double>(j0.v);
                        }
                    }
                    else if (need_value)
                    {
                        val_acc = w_diag;
                    }
                }
                for (std::size_t row = col + 1U; row < n; ++row)
                {
                    const T d2 = d2_col[row];
                    const T clamped = d2 < T{0} ? T{0} : d2;
                    const T w = clamped / rq_denom;
                    const T b = T{1} + w;
                    const T l1p = std::log1p(w);
                    const ExpJet<T> jet = eval_kernel_exp_jet(-rq_alpha * l1p, exp_mode);
                    const auto w_rc = static_cast<double>(w_col[row]);
                    g0_off += w_rc * static_cast<double>(((T{2} * rq_alpha * w) / b) * jet.d1);
                    g1_off += w_rc * static_cast<double>((rq_alpha * ((w / b) - l1p)) * jet.d1);
                    if (need_value)
                    {
                        val_off += w_rc * static_cast<double>(jet.v);
                    }
                }
            }

            out_g0 = g0_diag + (2.0 * g0_off);
            out_g1 = g1_diag + (2.0 * g1_off);
            out_val = val_acc + (2.0 * val_off);
        };

        double sum_g0 = 0.0;
        double sum_g1 = 0.0;
        double sum_val = 0.0;
        if (col_partials.size() >= 3U * n)
        {
            double* const g0_part = col_partials.data();
            double* const g1_part = col_partials.data() + n;
            double* const val_part = col_partials.data() + (2U * n);
            auto run_range = [&](std::size_t col_begin, std::size_t col_end)
            {
                for (std::size_t col = col_begin; col < col_end; ++col)
                {
                    process_col(col, g0_part[col], g1_part[col], val_part[col]);
                }
            };
            if (n >= 128U)
            {
                linalg::parallel_for_lower_blocks(n, 0U, run_range);
            }
            else
            {
                run_range(0U, n);
            }
            // Fold per-column partials in ascending order so OpenMP results remain bit-identical across threads.
            for (std::size_t col = 0U; col < n; ++col)
            {
                sum_g0 += g0_part[col];
                if (has_param1)
                {
                    sum_g1 += g1_part[col];
                }
                if (need_value)
                {
                    sum_val += val_part[col];
                }
            }
        }
        else
        {
            for (std::size_t col = 0U; col < n; ++col)
            {
                double c0 = 0.0;
                double c1 = 0.0;
                double cv = 0.0;
                process_col(col, c0, c1, cv);
                sum_g0 += c0;
                if (has_param1)
                {
                    sum_g1 += c1;
                }
                if (need_value)
                {
                    sum_val += cv;
                }
            }
        }

        out_grads[0] = sum_g0;
        if (has_param1)
        {
            out_grads[1] = sum_g1;
        }
        return need_value ? sum_val : 0.0;
    }

    template std::expected<double, GprError> leaf_iso_weighted_grads<double>(
        const KernelLeaf&,
        MatrixRef<double>,
        MatrixRef<double>,
        MatrixRef<double>,
        MatrixMut<double>,
        std::span<double>,
        std::span<double>,
        bool,
        KernelExp
    );
    template std::expected<double, GprError> leaf_iso_weighted_grads<float>(
        const KernelLeaf&,
        MatrixRef<float>,
        MatrixRef<float>,
        MatrixRef<float>,
        MatrixMut<float>,
        std::span<double>,
        std::span<double>,
        bool,
        KernelExp
    );

}  // namespace cxxgpr::detail
