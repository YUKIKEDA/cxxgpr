// SPDX-License-Identifier: Apache-2.0
/// @file distance.cpp
/// @brief Supplied squared-distance slots, validation/repair stores, and compiled distance kernels.
/// @details Implements `ScalarDistance`, `ArdDistance`, `DistanceKernel`, `DistanceSource`, `CheckedDistanceStore`,
/// and `CompiledDistanceKernel`, including single-pass validation and `tidy(rel_tol)` repair, packed ARD lower
/// triangles, and analytical evaluation, gradients, and Hessians.

#include "cxxgpr/kernel/distance.hpp"

#include "cxxgpr/core/checked.hpp"
#include "cxxgpr/linalg/parallel.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "core/noinline.hpp"
#include "kernel/distance_desc.hpp"
#include "kernel/distance_store.hpp"
#include "kernel/exp_simd.hpp"
#include "kernel/leaf_eval.hpp"

/// @brief Root namespace for the cxxgpr library.
namespace cxxgpr
{

    namespace
    {

        /// @brief Process-wide atomic counter generating unique non-zero `SlotId` identifiers.
        /// @details Incremented with relaxed memory order because slot IDs only need uniqueness across slots, not
        /// cross-thread memory synchronization.
        std::atomic<std::uint64_t> g_next_slot_id{1};

        /// @brief Computes the column-major packed lower-triangular index for symmetric entry `(row, col)`.
        /// @details Normalizes `(r, c) = (max(row, col), min(row, col))` and maps to `c*n - c*(c-1)/2 + (r - c)`.
        /// @param[in] n Square matrix side length.
        /// @param[in] row Row coordinate.
        /// @param[in] col Column coordinate.
        /// @return Zero-based packed lower-triangular offset.
        [[nodiscard]] constexpr std::size_t packed_lower_index(std::size_t n, std::size_t row, std::size_t col) noexcept
        {
            const std::size_t r = std::max(row, col);
            const std::size_t c = std::min(row, col);
            const std::size_t tri = (c > 0U) ? ((c * (c - 1U)) / 2U) : 0U;
            return ((c * n) - tri) + (r - c);
        }

        /// @brief Accumulates `dst += src` over the active matrix region `uplo`.
        /// @details Loops over column-major slices restricted to `Lower`, `Upper`, or `Full`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in,out] dst Destination matrix modified in place.
        /// @param[in] src Summand matrix view.
        /// @param[in] uplo Active matrix region (`Lower`, `Upper`, or `Full`).
        template <KernelScalarConcept T> void add_into(MatrixMut<T> dst, MatrixRef<T> src, Triangle uplo) noexcept
        {
            const std::size_t n_rows = dst.rows();
            const std::size_t n_cols = dst.cols();
            if (uplo == Triangle::Lower && n_rows == n_cols)
            {
                for (std::size_t col = 0; col < n_cols; ++col)
                {
                    for (std::size_t row = col; row < n_rows; ++row)
                    {
                        dst(row, col) += src(row, col);
                    }
                }
                return;
            }
            if (uplo == Triangle::Upper && n_rows == n_cols)
            {
                for (std::size_t col = 0; col < n_cols; ++col)
                {
                    for (std::size_t row = 0; row <= col; ++row)
                    {
                        dst(row, col) += src(row, col);
                    }
                }
                return;
            }
            for (std::size_t col = 0; col < n_cols; ++col)
            {
                for (std::size_t row = 0; row < n_rows; ++row)
                {
                    dst(row, col) += src(row, col);
                }
            }
        }

        /// @brief Multiplies `dst *= src` element-wise over the active matrix region `uplo`.
        /// @details Loops over column-major slices restricted to `Lower`, `Upper`, or `Full`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in,out] dst Destination matrix scaled in place.
        /// @param[in] src Multiplier matrix view.
        /// @param[in] uplo Active matrix region (`Lower`, `Upper`, or `Full`).
        template <KernelScalarConcept T> void mul_into(MatrixMut<T> dst, MatrixRef<T> src, Triangle uplo) noexcept
        {
            const std::size_t n_rows = dst.rows();
            const std::size_t n_cols = dst.cols();
            if (uplo == Triangle::Lower && n_rows == n_cols)
            {
                for (std::size_t col = 0; col < n_cols; ++col)
                {
                    for (std::size_t row = col; row < n_rows; ++row)
                    {
                        dst(row, col) *= src(row, col);
                    }
                }
                return;
            }
            if (uplo == Triangle::Upper && n_rows == n_cols)
            {
                for (std::size_t col = 0; col < n_cols; ++col)
                {
                    for (std::size_t row = 0; row <= col; ++row)
                    {
                        dst(row, col) *= src(row, col);
                    }
                }
                return;
            }
            for (std::size_t col = 0; col < n_cols; ++col)
            {
                for (std::size_t row = 0; row < n_rows; ++row)
                {
                    dst(row, col) *= src(row, col);
                }
            }
        }

        /// @brief Mirrors strictly lower entries of a square column-major matrix into its upper triangle when `Full`.
        /// @details No-op for `Triangle::Lower` and `Triangle::Upper`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in,out] out Target square matrix view.
        /// @param[in] uplo Active matrix region (`Full` triggers mirroring).
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

        /// @brief Allocation-free view of per-dimension `extra_scale / (ls[k]^2)` ARD weights.
        /// @details Reads from a stack-allocated precomputed buffer for common dimensions (`d <= 256`) and falls back
        /// to on-the-fly evaluation for arbitrarily large `d`, guaranteeing zero heap allocations for every dimension.
        /// @tparam T Storage floating-point scalar type.
        template <KernelScalarConcept T> struct ArdWeightView
        {
            /// @brief Stack-backed precomputed weights when `d <= 256`, empty otherwise.
            std::span<const T> precomputed;
            /// @brief Source lengthscales used for on-the-fly fallback when `precomputed` is empty.
            std::span<const double> ls;
            /// @brief Additional scalar factor folded into each inverse-squared weight.
            T extra_scale{T{1}};

            /// @brief Reports whether this view is empty.
            /// @return True when neither precomputed weights nor source lengthscales are set.
            [[nodiscard]] bool empty() const noexcept { return precomputed.empty() && ls.empty(); }

            /// @brief Returns the number of per-dimension weights.
            /// @return Dimension count.
            [[nodiscard]] std::size_t size() const noexcept
            {
                return !precomputed.empty() ? precomputed.size() : ls.size();
            }

            /// @brief Returns the inverse-squared weight for dimension `dim`.
            /// @param[in] dim Feature dimension index.
            /// @return `extra_scale / static_cast<T>(ls[dim] * ls[dim])`.
            [[nodiscard]] T operator[](std::size_t dim) const noexcept
            {
                return !precomputed.empty() ? precomputed[dim] : (extra_scale / static_cast<T>(ls[dim] * ls[dim]));
            }
        };

        /// @brief Invokes `fn(ArdWeightView<T> weights)` with `extra_scale / (ls[k]^2)` weights without heap
        /// allocation.
        /// @details Precomputes into a stack array for `d <= 256` and evaluates on the fly for `d > 256`, keeping
        /// every ARD dimension strictly allocation-free.
        /// @tparam T Storage floating-point scalar type.
        /// @tparam Fn Callable accepting `ArdWeightView<T>`.
        /// @param[in] ls Per-dimension lengthscale span.
        /// @param[in] extra_scale Additional scalar factor folded into each inverse-squared weight.
        /// @param[in] fn Callback consuming the inverse-squared lengthscale weight view.
        /// @return Result returned by `fn`.
        template <KernelScalarConcept T, typename Fn>
        decltype(auto) with_ard_inv_ell2(std::span<const double> ls, T extra_scale, Fn&& fn)
        {
            constexpr std::size_t k_stack_cap = 256U;
            if (ls.size() <= k_stack_cap)
            {
                std::array<T, k_stack_cap> buf{};
                for (std::size_t d = 0U; d < ls.size(); ++d)
                {
                    buf[d] = extra_scale / static_cast<T>(ls[d] * ls[d]);
                }
                return fn(
                    ArdWeightView<T>{
                        .precomputed = std::span<const T>(buf.data(), ls.size()),
                        .ls = {},
                        .extra_scale = T{1},
                    }
                );
            }
            return fn(ArdWeightView<T>{.precomputed = {}, .ls = ls, .extra_scale = extra_scale});
        }

        /// @brief Applies the Rational Quadratic radial transform $\exp(-\alpha \log(1 + w))$ in place along a slice.
        /// @details Expects `col_ptr[k]` to hold $w_k \ge 0$, clamps negative rounding noise to 0, evaluates `log1p`,
        /// and dispatches exponential evaluation to `detail::exp_scaled_nonpos_in_place`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in,out] col_ptr Contiguous column slice storing scaled squared distances `w`.
        /// @param[in] len Number of elements in the column slice.
        /// @param[in] alpha Rational quadratic shape parameter $\alpha > 0$.
        /// @param[in] exp_mode Exponential evaluation backend.
        template <KernelScalarConcept T>
        void transform_rq_slice_in_place(T* col_ptr, std::size_t len, T alpha, KernelExp exp_mode) noexcept
        {
            for (std::size_t k = 0U; k < len; ++k)
            {
                const T w = col_ptr[k] < T{0} ? T{0} : col_ptr[k];
                col_ptr[k] = std::log1p(w);
            }
            detail::exp_scaled_nonpos_in_place(col_ptr, len, -alpha, exp_mode);
        }

        /// @brief Visits column slices `(col, row_begin, row_end)` over `Lower`/`Full` or `Upper` in parallel.
        /// @details Uses area-balanced `linalg::parallel_for_lower_blocks` (`n >= 128U`) for `Lower` and `Full`
        /// (processing lower slices before mirroring), and static OpenMP scheduling (`n >= 128U`) for `Upper`.
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

        /// @brief Visits coordinates `(row, col)` of an `n x n` matrix region in column-major order.
        /// @details Uses area-balanced `linalg::parallel_for_lower_blocks` (`n >= 128U`) for `Lower`, and static OpenMP
        /// column parallelism (`n >= 128U`) for `Upper` and `Full`.
        /// @tparam T Storage floating-point scalar type.
        /// @tparam Fn Callable invoked as `fn(row, col)`.
        /// @param[in] n Square matrix side length.
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

        /// @brief Validates that `scratch` matches `out` shape and does not alias `out`.
        /// @details Returns `WorkspaceTooSmall` on dimension mismatch and `InvalidConfig` on pointer aliasing.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] out Reference output matrix view.
        /// @param[in] scratch Mutable scratch buffer view.
        /// @return Empty expected on success, or workspace/aliasing error.
        /// @retval GprErrorCode::WorkspaceTooSmall If `scratch` shape differs from `out`.
        /// @retval GprErrorCode::InvalidConfig If `scratch.data() == out.data()`.
        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> require_scratch(MatrixRef<T> out, MatrixMut<T> scratch)
        {
            if (scratch.rows() != out.rows() || scratch.cols() != out.cols())
            {
                return std::unexpected(GprError::workspace_too_small());
            }
            if (scratch.data() == out.data())
            {
                return std::unexpected(GprError::invalid_config("scratch buffer must not alias output"));
            }
            return {};
        }

        /// @brief Returns the squared Matern radial scaling factor $2\nu$ in `double`.
        /// @details Maps `OneHalf -> 1.0`, `ThreeHalves -> 3.0`, and `FiveHalves -> 5.0`.
        /// @param[in] nu Matern smoothness parameter.
        /// @return Squared radial factor $2\nu$.
        [[nodiscard]] constexpr double matern_scale_sq_d(MaternNu nu) noexcept
        {
            switch (nu)
            {
                case MaternNu::OneHalf:
                    return 1.0;
                case MaternNu::ThreeHalves:
                    return 3.0;
                case MaternNu::FiveHalves:
                    return 5.0;
            }
            return 1.0;
        }

        /// @brief Evaluates the scalar Matern radial kernel value at scaled radius $u$.
        /// @details Returns `1` at $u \le 0$ and evaluates the smoothness polynomial times `eval_kernel_exp(-u,
        /// exp_mode)` for positive $u$.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] u Scaled radius $u = \sqrt{2\nu\, s^2}$.
        /// @param[in] nu Matern smoothness parameter.
        /// @param[in] exp_mode Exponential evaluation backend.
        /// @return Matern kernel value.
        template <KernelScalarConcept T> [[nodiscard]] T eval_matern_u_d(T u, MaternNu nu, KernelExp exp_mode) noexcept
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

        /// @brief Evaluates the Matern radial log-lengthscale derivative factor $-u k'(u)$.
        /// @details Returns `0` for $u \le 0$ (continuous zero-distance limit) and propagates `ExpJet<T>` derivatives
        /// under `FastApprox`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] u Scaled ARD radius $u = \sqrt{2\nu\, s^2}$.
        /// @param[in] nu Matern smoothness parameter.
        /// @param[in] exp_mode Exponential evaluation backend.
        /// @return Radial gradient factor $-u k'(u)$.
        template <KernelScalarConcept T>
        [[nodiscard]] T eval_matern_grad_g1_d(T u, MaternNu nu, KernelExp exp_mode) noexcept
        {
            // Zero-distance limit of radial log-lengthscale derivative is 0.
            if (u <= T{0})
            {
                return T{0};
            }
            if (exp_mode == KernelExp::Accurate)
            {
                const T e = std::exp(-u);
                switch (nu)
                {
                    case MaternNu::OneHalf:
                        return u * e;
                    case MaternNu::ThreeHalves:
                        return (u * u) * e;
                    case MaternNu::FiveHalves:
                        return (((u * u) * (T{1} + u)) / T{3}) * e;
                }
                return T{0};
            }
            // Differentiate polynomial * exp_approx(-u) via ExpJet of exp(-u).
            const ExpJet<T> jet = eval_kernel_exp_jet(-u, exp_mode);
            switch (nu)
            {
                case MaternNu::OneHalf:
                    return u * jet.d1;
                case MaternNu::ThreeHalves:
                    return u * (((T{1} + u) * jet.d1) - jet.v);
                case MaternNu::FiveHalves:
                    return u * (((T{1} + u + ((u * u) / T{3})) * jet.d1) - ((T{1} + ((T{2} * u) / T{3})) * jet.v));
            }
            return T{0};
        }

        /// @brief Evaluates the ARD Matern second-derivative radial cross coefficient $u^2 k''(u) - u k'(u)$.
        /// @details Returns `0` for $u \le 0$ and supports both `Accurate` exact closed forms and `FastApprox`
        /// `ExpJet<T>` propagation.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] u Scaled ARD radius $u = \sqrt{2\nu\, s^2}$.
        /// @param[in] nu Matern smoothness parameter.
        /// @param[in] exp_mode Exponential evaluation backend.
        /// @return Radial cross coefficient for ARD Matern Hessians.
        template <KernelScalarConcept T>
        [[nodiscard]] T eval_matern_ard_cross_d(T u, MaternNu nu, KernelExp exp_mode) noexcept
        {
            if (u <= T{0})
            {
                return T{0};
            }
            if (exp_mode == KernelExp::Accurate)
            {
                const T e = std::exp(-u);
                switch (nu)
                {
                    case MaternNu::OneHalf:
                        return u * (u + T{1}) * e;
                    case MaternNu::ThreeHalves:
                        return (u * u * u) * e;
                    case MaternNu::FiveHalves:
                        return ((u * u * u * u) / T{3}) * e;
                }
                return T{0};
            }
            // Combine first and second radial derivatives: u^2 * k''(u) - u * k'(u).
            const ExpJet<T> jet = eval_kernel_exp_jet(-u, exp_mode);
            switch (nu)
            {
                case MaternNu::OneHalf:
                    return u * ((u * jet.d2) + jet.d1);
                case MaternNu::ThreeHalves:
                {
                    const T d2k = ((T{1} + u) * jet.d2) - (T{2} * jet.d1);
                    const T dk = jet.v - ((T{1} + u) * jet.d1);
                    return (u * u * d2k) - (u * dk);
                }
                case MaternNu::FiveHalves:
                {
                    const T poly = T{1} + u + ((u * u) / T{3});
                    const T dpoly = T{1} + ((T{2} * u) / T{3});
                    const T d2k = (poly * jet.d2) - (T{2} * dpoly * jet.d1) + ((T{2} / T{3}) * jet.v);
                    const T dk = (dpoly * jet.v) - (poly * jet.d1);
                    return (u * u * d2k) - (u * dk);
                }
            }
            return T{0};
        }

        /// @brief Fills a full symmetric square squared-distance matrix from point coordinates `x`.
        /// @details Zeroes the diagonal and mirrors strictly lower entries into the upper triangle for coordinate
        /// `CustomKernel` leaves mixed inside `DistanceKernel<WithPoints>`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] x Input point matrix of shape `(n, d)`.
        /// @param[out] dist Output square squared-distance matrix of shape `(n, n)`.
        template <KernelScalarConcept T> void fill_square_sq_dist(MatrixRef<T> x, MatrixMut<T> dist) noexcept
        {
            const std::size_t n = x.rows();
            const std::size_t d = x.cols();
            if (n == 0U || d == 0U)
            {
                return;
            }
            for_each_triangle_col(
                n, Triangle::Full, [&](std::size_t col, std::size_t row_begin, std::size_t row_end)
            {
                const std::size_t len = row_end - row_begin;
                if (len == 0U)
                {
                    return;
                }
                T* col_ptr = &dist(row_begin, col);
                detail::set_scaled_sq_diff_slice(col_ptr, x.col(0U).data() + row_begin, x(col, 0U), T{1}, len);
                for (std::size_t k = 1U; k < d; ++k)
                {
                    detail::add_scaled_sq_diff_slice(col_ptr, x.col(k).data() + row_begin, x(col, k), T{1}, len);
                }
                dist(col, col) = T{0};
            }
            );
            mirror_lower_if_full(dist, Triangle::Full);
        }

        /// @brief Fills a rectangular cross squared-distance matrix from coordinate matrices `x1` and `x2`.
        /// @details Vectorizes column slices along feature dimensions via `set_scaled_sq_diff_slice` and
        /// `add_scaled_sq_diff_slice`, parallelizing across columns when `(n1 * n2) >= 65536U`.
        /// @tparam T Storage floating-point scalar type.
        /// @param[in] x1 First point matrix of shape `(n1, d)`.
        /// @param[in] x2 Second point matrix of shape `(n2, d)`.
        /// @param[out] dist Output rectangular squared-distance matrix of shape `(n1, n2)`.
        template <KernelScalarConcept T>
        void fill_cross_sq_dist(MatrixRef<T> x1, MatrixRef<T> x2, MatrixMut<T> dist) noexcept
        {
            const std::size_t n1 = x1.rows();
            const std::size_t n2 = x2.rows();
            const std::size_t d = x1.cols();
            if (n1 == 0U || n2 == 0U || d == 0U)
            {
                return;
            }
            const auto n2_s = static_cast<std::ptrdiff_t>(n2);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if ((n1 * n2) >= 65536U)
#endif
            for (std::ptrdiff_t col_s = 0; col_s < n2_s; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                T* col_ptr = &dist(0U, col);
                detail::set_scaled_sq_diff_slice(col_ptr, x1.col(0U).data(), x2(col, 0U), T{1}, n1);
                for (std::size_t k = 1U; k < d; ++k)
                {
                    detail::add_scaled_sq_diff_slice(col_ptr, x1.col(k).data(), x2(col, k), T{1}, n1);
                }
            }
        }

        /// @brief Writes the log-space parameters of one leaf at offset, in a stack frame of its own.
        /// @details Keeps the std::expected temporary of the leaf query out of the frames of the recursive walk.
        /// @param[in] leaf Kernel leaf to query.
        /// @param[out] out Full flattened output parameter span.
        /// @param[in] offset Index in out of the first parameter of leaf.
        /// @param[out] status Set to the error of the leaf query when it fails.
        /// @return Offset immediately following the parameters of leaf.
        CXXGPR_NOINLINE std::size_t fill_leaf_params(
            const KernelLeaf& leaf, std::span<double> out, std::size_t offset, std::expected<void, GprError>& status
        )
        {
            const std::size_t count = detail::leaf_num_params(leaf);
            status = detail::leaf_get_params(leaf, out.subspan(offset, count));
            return offset + count;
        }

        /// @brief Writes the open intervals of one leaf at offset, in a stack frame of its own.
        /// @details Keeps the std::expected temporary of the leaf query out of the frames of the recursive walk.
        /// @param[in] leaf Kernel leaf to query.
        /// @param[out] out Full flattened output interval span.
        /// @param[in] offset Index in out of the first interval of leaf.
        /// @param[out] status Set to the error of the leaf query when it fails.
        /// @return Offset immediately following the intervals of leaf.
        CXXGPR_NOINLINE std::size_t fill_leaf_bounds(
            const KernelLeaf& leaf, std::span<Interval> out, std::size_t offset, std::expected<void, GprError>& status
        )
        {
            const std::size_t count = detail::leaf_num_params(leaf);
            status = detail::leaf_bounds_into(leaf, out.subspan(offset, count));
            return offset + count;
        }

    }  // namespace

    // --- Internal tree representation for DistanceKernel ---

    /// @brief Distance-kernel AST leaf bound to a `ScalarDistance` slot.
    /// @details Couples an isotropic or custom `KernelLeaf` to the `SlotId` providing its scalar squared distances.
    struct SuppliedScalarNode
    {
        /// @brief Identifier of the bound `ScalarDistance` slot.
        SlotId slot;
        /// @brief Built-in isotropic or custom kernel leaf definition.
        KernelLeaf leaf;
    };

    /// @brief Distance-kernel AST leaf bound to an `ArdDistance` slot.
    /// @details Couples an ARD `KernelLeaf` to its `SlotId` and validated feature dimensionality `dims`.
    struct SuppliedArdNode
    {
        /// @brief Identifier of the bound `ArdDistance` slot.
        SlotId slot;
        /// @brief Expected feature dimensionality of the supplied per-dimension distance blocks.
        std::size_t dims = 0;
        /// @brief ARD kernel leaf definition (`RbfArdKernel`, `MaternArdKernel`, or `RationalQuadraticArdKernel`).
        KernelLeaf leaf;
    };

    /// @brief Distance-kernel AST leaf evaluated from point coordinates (or parameter-free `Constant`/`White`).
    /// @details Appears directly in `DistanceOnly` for `ConstantKernel`/`WhiteKernel` and in `WithPoints` for any leaf
    /// promoted from `KernelSpec`.
    struct CoordLeafNode
    {
        /// @brief Kernel leaf definition.
        KernelLeaf leaf;
    };

    /// @brief Internal persistent AST node shared immutably by `DistanceKernel<Mode>`.
    /// @details Holds either a supplied scalar/ARD leaf, a coordinate leaf, or a binary `Sum`/`Product` node.
    /// @tparam Mode Distance kernel mode tag (`DistanceOnly` or `WithPoints`).
    template <DistanceModeConcept Mode> struct DistanceKernel<Mode>::InternalNode
    {
        /// @brief Binary addition node combining `lhs + rhs`.
        struct Sum
        {
            /// @brief Left-hand summand subtree.
            mutable std::shared_ptr<const InternalNode> lhs;
            /// @brief Right-hand summand subtree.
            mutable std::shared_ptr<const InternalNode> rhs;
        };

        /// @brief Binary multiplication node combining `lhs * rhs`.
        struct Product
        {
            /// @brief Left-hand factor subtree.
            mutable std::shared_ptr<const InternalNode> lhs;
            /// @brief Right-hand factor subtree.
            mutable std::shared_ptr<const InternalNode> rhs;
        };

        /// @brief Node payload variant holding a leaf or binary operator.
        std::variant<SuppliedScalarNode, SuppliedArdNode, CoordLeafNode, Sum, Product> data;
    };

    /// @brief Destroys composite subtrees iteratively so deep distance kernel trees do not overflow the stack.
    /// @tparam Mode Distance kernel mode tag (`DistanceOnly` or `WithPoints`).
    template <DistanceModeConcept Mode> DistanceKernel<Mode>::~DistanceKernel() noexcept
    {
        if (!root_ || root_.use_count() != 1L)
        {
            return;
        }
        try
        {
            std::vector<std::shared_ptr<const InternalNode>> work;
            work.push_back(std::move(root_));
            auto push_unique = [&work](std::shared_ptr<const InternalNode>& child)
            {
                if (child && child.use_count() == 1L)
                {
                    work.push_back(std::move(child));
                }
                else
                {
                    child.reset();
                }
            };
            while (!work.empty())
            {
                auto cur = std::move(work.back());
                work.pop_back();
                if (const auto* s = std::get_if<typename InternalNode::Sum>(&cur->data))
                {
                    push_unique(s->lhs);
                    push_unique(s->rhs);
                }
                else if (const auto* p = std::get_if<typename InternalNode::Product>(&cur->data))
                {
                    push_unique(p->lhs);
                    push_unique(p->rhs);
                }
            }
        }
        catch (...)
        {
            // Best-effort iterative teardown; remaining links fall back to default member destruction.
            return;
        }
    }

    /// @brief Friend helper encapsulating recursive `DistanceKernel<Mode>` AST inspection and transformation.
    /// @details Isolates internal tree manipulation (`promote_node`, `from_coord_spec`, parameter/bound/slot walks, and
    /// operator builders) from public header exposure.
    struct DistanceKernelAccess
    {
        /// @brief Shorthand alias for `DistanceKernel<Mode>::InternalNode`.
        /// @tparam Mode Distance kernel mode tag (`DistanceOnly` or `WithPoints`).
        template <DistanceModeConcept Mode> using Node = DistanceKernel<Mode>::InternalNode;

        /// @brief Converts a `DistanceOnly` AST node into a `Node<Mode>` tree.
        /// @details Copies directly when `Mode` is `DistanceOnly` and recursively reconstructs
        /// `InternalNode<WithPoints>` nodes when promoting to `WithPoints`.
        /// @tparam Mode Target distance kernel mode tag.
        /// @param[in] src Source `DistanceOnly` AST node.
        /// @return Shared immutable pointer to the promoted `Node<Mode>`.
        template <DistanceModeConcept Mode>
        static std::shared_ptr<const Node<Mode>> promote_node(const DistanceKernel<DistanceOnly>::InternalNode& src)
        {
            if constexpr (std::same_as<Mode, DistanceOnly>)
            {
                return std::make_shared<const Node<Mode>>(src);
            }
            else
            {
                if (const auto* s = std::get_if<SuppliedScalarNode>(&src.data))
                {
                    return std::make_shared<const Node<WithPoints>>(Node<WithPoints>{.data = *s});
                }
                if (const auto* a = std::get_if<SuppliedArdNode>(&src.data))
                {
                    return std::make_shared<const Node<WithPoints>>(Node<WithPoints>{.data = *a});
                }
                if (const auto* c = std::get_if<CoordLeafNode>(&src.data))
                {
                    return std::make_shared<const Node<WithPoints>>(Node<WithPoints>{.data = *c});
                }
                if (const auto* sum = std::get_if<DistanceKernel<DistanceOnly>::InternalNode::Sum>(&src.data))
                {
                    auto lhs_ptr = promote_node<WithPoints>(*sum->lhs);
                    auto rhs_ptr = promote_node<WithPoints>(*sum->rhs);
                    Node<WithPoints>::Sum snode{.lhs = std::move(lhs_ptr), .rhs = std::move(rhs_ptr)};
                    return std::make_shared<const Node<WithPoints>>(Node<WithPoints>{.data = std::move(snode)});
                }
                const auto& prod = std::get<DistanceKernel<DistanceOnly>::InternalNode::Product>(src.data);
                auto lhs_ptr = promote_node<WithPoints>(*prod.lhs);
                auto rhs_ptr = promote_node<WithPoints>(*prod.rhs);
                Node<WithPoints>::Product pnode{.lhs = std::move(lhs_ptr), .rhs = std::move(rhs_ptr)};
                return std::make_shared<const Node<WithPoints>>(Node<WithPoints>{.data = std::move(pnode)});
            }
        }

        /// @brief Converts a coordinate `KernelSpec` tree into `Node<WithPoints>` coordinate-leaf nodes.
        /// @details Preserves binary `Sum` and `Product` topology and depth-first left-to-right ordering.
        /// @param[in] spec Source coordinate-based `KernelSpec`.
        /// @return Shared immutable pointer to the equivalent `Node<WithPoints>` subtree.
        static std::shared_ptr<const Node<WithPoints>> from_coord_spec(const KernelSpec& spec)
        {
            if (const auto* leaf = std::get_if<KernelLeaf>(&spec.node_))
            {
                return std::make_shared<const Node<WithPoints>>(Node<WithPoints>{.data = CoordLeafNode{.leaf = *leaf}});
            }
            if (const auto* sum = std::get_if<KernelSpec::SumNode>(&spec.node_))
            {
                auto lhs_ptr = from_coord_spec(*sum->lhs);
                auto rhs_ptr = from_coord_spec(*sum->rhs);
                Node<WithPoints>::Sum snode{.lhs = std::move(lhs_ptr), .rhs = std::move(rhs_ptr)};
                return std::make_shared<const Node<WithPoints>>(Node<WithPoints>{.data = std::move(snode)});
            }
            const auto& prod = std::get<KernelSpec::ProductNode>(spec.node_);
            auto lhs_ptr = from_coord_spec(*prod.lhs);
            auto rhs_ptr = from_coord_spec(*prod.rhs);
            Node<WithPoints>::Product pnode{.lhs = std::move(lhs_ptr), .rhs = std::move(rhs_ptr)};
            return std::make_shared<const Node<WithPoints>>(Node<WithPoints>{.data = std::move(pnode)});
        }

        /// @brief Recursively counts total log-space hyperparameters under `node`.
        /// @details Sums leaf parameter counts in depth-first left-to-right order.
        /// @tparam Mode Distance kernel mode tag.
        /// @param[in] node AST node to inspect.
        /// @return Total hyperparameter count under `node`.
        template <DistanceModeConcept Mode> static std::size_t count_params(const Node<Mode>& node) noexcept
        {
            if (const auto* s = std::get_if<SuppliedScalarNode>(&node.data))
            {
                return detail::leaf_num_params(s->leaf);
            }
            if (const auto* a = std::get_if<SuppliedArdNode>(&node.data))
            {
                return detail::leaf_num_params(a->leaf);
            }
            if (const auto* c = std::get_if<CoordLeafNode>(&node.data))
            {
                return detail::leaf_num_params(c->leaf);
            }
            if (const auto* sum = std::get_if<typename Node<Mode>::Sum>(&node.data))
            {
                return count_params<Mode>(*sum->lhs) + count_params<Mode>(*sum->rhs);
            }
            if (const auto* prod = std::get_if<typename Node<Mode>::Product>(&node.data))
            {
                return count_params<Mode>(*prod->lhs) + count_params<Mode>(*prod->rhs);
            }
            return 0;
        }

        /// @brief Recursively writes flattened log-space hyperparameters under `node` into `out`.
        /// @details Traverses leaves depth-first left-to-right, short-circuiting if `status` records an error.
        /// @tparam Mode Distance kernel mode tag.
        /// @param[in] node AST node to inspect.
        /// @param[out] out Destination span of log-space hyperparameters.
        /// @param[in] offset Starting element offset inside `out`.
        /// @param[in,out] status Cumulative validation status updated on leaf read errors.
        /// @return Updated offset after consuming `node`'s hyperparameters.
        template <DistanceModeConcept Mode>
        static std::size_t fill_params(
            const Node<Mode>& node, std::span<double> out, std::size_t offset, std::expected<void, GprError>& status
        )
        {
            if (!status.has_value())
            {
                return offset;
            }
            const KernelLeaf* leaf = nullptr;
            if (const auto* s = std::get_if<SuppliedScalarNode>(&node.data))
            {
                leaf = &s->leaf;
            }
            else if (const auto* a = std::get_if<SuppliedArdNode>(&node.data))
            {
                leaf = &a->leaf;
            }
            else if (const auto* c = std::get_if<CoordLeafNode>(&node.data))
            {
                leaf = &c->leaf;
            }
            if (leaf != nullptr)
            {
                return fill_leaf_params(*leaf, out, offset, status);
            }
            if (const auto* sum = std::get_if<typename Node<Mode>::Sum>(&node.data))
            {
                offset = fill_params<Mode>(*sum->lhs, out, offset, status);
                return fill_params<Mode>(*sum->rhs, out, offset, status);
            }
            if (const auto* prod = std::get_if<typename Node<Mode>::Product>(&node.data))
            {
                offset = fill_params<Mode>(*prod->lhs, out, offset, status);
                return fill_params<Mode>(*prod->rhs, out, offset, status);
            }
            return offset;
        }

        /// @brief Recursively writes flattened open hyperparameter intervals under `node` into `out`.
        /// @details Traverses leaves depth-first left-to-right, short-circuiting if `status` records an error.
        /// @tparam Mode Distance kernel mode tag.
        /// @param[in] node AST node to inspect.
        /// @param[out] out Destination span of `Interval` bounds.
        /// @param[in] offset Starting element offset inside `out`.
        /// @param[in,out] status Cumulative validation status updated on leaf read errors.
        /// @return Updated offset after consuming `node`'s intervals.
        template <DistanceModeConcept Mode>
        static std::size_t fill_bounds(
            const Node<Mode>& node, std::span<Interval> out, std::size_t offset, std::expected<void, GprError>& status
        )
        {
            if (!status.has_value())
            {
                return offset;
            }
            const KernelLeaf* leaf = nullptr;
            if (const auto* s = std::get_if<SuppliedScalarNode>(&node.data))
            {
                leaf = &s->leaf;
            }
            else if (const auto* a = std::get_if<SuppliedArdNode>(&node.data))
            {
                leaf = &a->leaf;
            }
            else if (const auto* c = std::get_if<CoordLeafNode>(&node.data))
            {
                leaf = &c->leaf;
            }
            if (leaf != nullptr)
            {
                return fill_leaf_bounds(*leaf, out, offset, status);
            }
            if (const auto* sum = std::get_if<typename Node<Mode>::Sum>(&node.data))
            {
                offset = fill_bounds<Mode>(*sum->lhs, out, offset, status);
                return fill_bounds<Mode>(*sum->rhs, out, offset, status);
            }
            if (const auto* prod = std::get_if<typename Node<Mode>::Product>(&node.data))
            {
                offset = fill_bounds<Mode>(*prod->lhs, out, offset, status);
                return fill_bounds<Mode>(*prod->rhs, out, offset, status);
            }
            return offset;
        }

        /// @brief Persistently constructs a new `Node<Mode>` tree with updated log-space parameters.
        /// @details Clones modified leaf nodes without mutating the existing AST so parameter validation failures leave
        /// the source kernel untouched.
        /// @tparam Mode Distance kernel mode tag.
        /// @param[in] node Source AST node to rebuild.
        /// @param[in] params Log-space parameter span for the entire kernel.
        /// @param[in,out] offset Running offset advanced across leaves in depth-first left-to-right order.
        /// @return Newly allocated AST root on success, or bounds/parameter error.
        template <DistanceModeConcept Mode>
        static std::expected<std::shared_ptr<const Node<Mode>>, GprError> with_updated_params(
            const Node<Mode>& node, std::span<const double> params, std::size_t& offset
        )
        {
            std::expected<void, GprError> status;
            auto updated = update_params_node<Mode>(node, params, offset, status);
            if (updated == nullptr)
            {
                return std::unexpected(std::move(status).error());
            }
            return updated;
        }

        /// @brief Copies one leaf node with its log-space parameters replaced, in a stack frame of its own.
        /// @details Clones the leaf before mutating it, so that a rejected slice leaves the source tree untouched.
        /// @tparam Mode Distance kernel mode tag.
        /// @param[in] node Source leaf node: a supplied scalar, supplied ARD, or coordinate leaf.
        /// @param[in] params Log-space parameter span for the entire kernel.
        /// @param[in,out] offset Index in params of the first parameter of the leaf; advanced past them on success.
        /// @param[out] status Set to the error of the leaf when it rejects its slice.
        /// @return Newly allocated leaf node, or a null pointer when status holds the error.
        template <DistanceModeConcept Mode>
        CXXGPR_NOINLINE static std::shared_ptr<const Node<Mode>> update_leaf_node(
            const Node<Mode>& node,
            std::span<const double> params,
            std::size_t& offset,
            std::expected<void, GprError>& status
        )
        {
            if (const auto* s = std::get_if<SuppliedScalarNode>(&node.data))
            {
                KernelLeaf copy = s->leaf;
                const std::size_t count = detail::leaf_num_params(copy);
                if (auto res = detail::leaf_set_params(copy, params.subspan(offset, count)); !res.has_value())
                {
                    status = std::unexpected(std::move(res).error());
                    return nullptr;
                }
                offset += count;
                return std::make_shared<const Node<Mode>>(Node<Mode>{
                    .data = SuppliedScalarNode{.slot = s->slot, .leaf = std::move(copy)}
                });
            }
            if (const auto* a = std::get_if<SuppliedArdNode>(&node.data))
            {
                KernelLeaf copy = a->leaf;
                const std::size_t count = detail::leaf_num_params(copy);
                if (auto res = detail::leaf_set_params(copy, params.subspan(offset, count)); !res.has_value())
                {
                    status = std::unexpected(std::move(res).error());
                    return nullptr;
                }
                offset += count;
                return std::make_shared<const Node<Mode>>(Node<Mode>{
                    .data = SuppliedArdNode{.slot = a->slot, .dims = a->dims, .leaf = std::move(copy)}
                });
            }
            // The caller passes only leaves, so the remaining alternative is a coordinate leaf.
            KernelLeaf copy = std::get<CoordLeafNode>(node.data).leaf;
            const std::size_t count = detail::leaf_num_params(copy);
            if (auto res = detail::leaf_set_params(copy, params.subspan(offset, count)); !res.has_value())
            {
                status = std::unexpected(std::move(res).error());
                return nullptr;
            }
            offset += count;
            return std::make_shared<const Node<Mode>>(Node<Mode>{.data = CoordLeafNode{.leaf = std::move(copy)}});
        }

        /// @brief Joins two updated subtrees into a sum or product node, in a stack frame of its own.
        /// @tparam Mode Distance kernel mode tag.
        /// @param[in] sum True for a sum node, false for a product node.
        /// @param[in] lhs Updated left subtree.
        /// @param[in] rhs Updated right subtree.
        /// @return Newly allocated composite node.
        template <DistanceModeConcept Mode>
        CXXGPR_NOINLINE static std::shared_ptr<const Node<Mode>> join_updated_nodes(
            bool sum, std::shared_ptr<const Node<Mode>> lhs, std::shared_ptr<const Node<Mode>> rhs
        )
        {
            if (sum)
            {
                return std::make_shared<const Node<Mode>>(Node<Mode>{
                    .data = typename Node<Mode>::Sum{.lhs = std::move(lhs), .rhs = std::move(rhs)}
                });
            }
            return std::make_shared<const Node<Mode>>(Node<Mode>{
                .data = typename Node<Mode>::Product{.lhs = std::move(lhs), .rhs = std::move(rhs)}
            });
        }

        /// @brief Recursively constructs a new `Node<Mode>` tree with updated log-space parameters.
        /// @details A nesting level keeps only its two updated subtrees on the stack: the leaf update, the join, and
        /// the error run in frames of their own.
        /// @tparam Mode Distance kernel mode tag.
        /// @param[in] node Source AST node to rebuild.
        /// @param[in] params Log-space parameter span for the entire kernel.
        /// @param[in,out] offset Running offset advanced across leaves in depth-first left-to-right order.
        /// @param[out] status Set to the error of the first leaf that rejects its slice.
        /// @return Newly allocated subtree, or a null pointer exactly when status holds the error.
        template <DistanceModeConcept Mode>
        static std::shared_ptr<const Node<Mode>> update_params_node(
            const Node<Mode>& node,
            std::span<const double> params,
            std::size_t& offset,
            std::expected<void, GprError>& status
        )
        {
            const auto* sum = std::get_if<typename Node<Mode>::Sum>(&node.data);
            const auto* prod = std::get_if<typename Node<Mode>::Product>(&node.data);
            if (sum == nullptr && prod == nullptr)
            {
                return update_leaf_node<Mode>(node, params, offset, status);
            }
            const Node<Mode>& lhs_node = (sum != nullptr) ? *sum->lhs : *prod->lhs;
            const Node<Mode>& rhs_node = (sum != nullptr) ? *sum->rhs : *prod->rhs;
            auto lhs = update_params_node<Mode>(lhs_node, params, offset, status);
            if (lhs == nullptr)
            {
                return nullptr;
            }
            auto rhs = update_params_node<Mode>(rhs_node, params, offset, status);
            if (rhs == nullptr)
            {
                return nullptr;
            }
            return join_updated_nodes<Mode>(sum != nullptr, std::move(lhs), std::move(rhs));
        }

        /// @brief Collects depth-first left-to-right `ParameterBinding` descriptors across `node`.
        /// @details Assigns monotonically increasing `leaf_id` and global parameter `index` values.
        /// @tparam Mode Distance kernel mode tag.
        /// @param[in] node AST node to traverse.
        /// @param[in,out] next_leaf_id Running leaf counter incremented at each leaf.
        /// @param[in,out] next_param_idx Running global parameter index counter.
        /// @param[in,out] out Output vector receiving `ParameterBinding` entries.
        template <DistanceModeConcept Mode>
        static void collect_bindings(
            const Node<Mode>& node,
            std::size_t& next_leaf_id,
            std::size_t& next_param_idx,
            std::vector<ParameterBinding>& out
        )
        {
            const KernelLeaf* leaf = nullptr;
            if (const auto* s = std::get_if<SuppliedScalarNode>(&node.data))
            {
                leaf = &s->leaf;
            }
            else if (const auto* a = std::get_if<SuppliedArdNode>(&node.data))
            {
                leaf = &a->leaf;
            }
            else if (const auto* c = std::get_if<CoordLeafNode>(&node.data))
            {
                leaf = &c->leaf;
            }
            if (leaf != nullptr)
            {
                const std::size_t count = detail::leaf_num_params(*leaf);
                const std::size_t leaf_id = next_leaf_id++;
                for (std::size_t local = 0; local < count; ++local)
                {
                    out.push_back(
                        ParameterBinding{
                            .index = next_param_idx++,
                            .leaf_id = leaf_id,
                            .local_index = local,
                        }
                    );
                }
                return;
            }
            if (const auto* sum = std::get_if<typename Node<Mode>::Sum>(&node.data))
            {
                collect_bindings<Mode>(*sum->lhs, next_leaf_id, next_param_idx, out);
                collect_bindings<Mode>(*sum->rhs, next_leaf_id, next_param_idx, out);
                return;
            }
            if (const auto* prod = std::get_if<typename Node<Mode>::Product>(&node.data))
            {
                collect_bindings<Mode>(*prod->lhs, next_leaf_id, next_param_idx, out);
                collect_bindings<Mode>(*prod->rhs, next_leaf_id, next_param_idx, out);
            }
        }

        /// @brief Collects unique supplied distance slots in depth-first first-appearance order.
        /// @details Deduplicates shared `SlotId` references and assigns consecutive per-kind `at` indices separately
        /// for scalar and ARD slots.
        /// @tparam Mode Distance kernel mode tag.
        /// @param[in] node AST node to traverse.
        /// @param[in,out] out Output list of unique `SlotDescriptor` entries.
        /// @param[in,out] next_scalar Running scalar slot index incremented on first appearance of a scalar slot.
        /// @param[in,out] next_ard Running ARD slot index incremented on first appearance of an ARD slot.
        template <DistanceModeConcept Mode>
        static void collect_slots(
            const Node<Mode>& node, std::vector<SlotDescriptor>& out, std::size_t& next_scalar, std::size_t& next_ard
        )
        {
            if (const auto* s = std::get_if<SuppliedScalarNode>(&node.data))
            {
                // Deduplicate repeated bindings of the same slot so one table serves multiple leaves.
                for (const auto& existing : out)
                {
                    if (existing.id == s->slot)
                    {
                        return;
                    }
                }
                out.push_back(SlotDescriptor{.id = s->slot, .ard_dims = std::nullopt, .at = next_scalar++});
                return;
            }
            if (const auto* a = std::get_if<SuppliedArdNode>(&node.data))
            {
                for (const auto& existing : out)
                {
                    if (existing.id == a->slot)
                    {
                        return;
                    }
                }
                out.push_back(SlotDescriptor{.id = a->slot, .ard_dims = a->dims, .at = next_ard++});
                return;
            }
            if (const auto* sum = std::get_if<typename Node<Mode>::Sum>(&node.data))
            {
                collect_slots<Mode>(*sum->lhs, out, next_scalar, next_ard);
                collect_slots<Mode>(*sum->rhs, out, next_scalar, next_ard);
                return;
            }
            if (const auto* prod = std::get_if<typename Node<Mode>::Product>(&node.data))
            {
                collect_slots<Mode>(*prod->lhs, out, next_scalar, next_ard);
                collect_slots<Mode>(*prod->rhs, out, next_scalar, next_ard);
            }
        }

        /// @brief Constructs an additive `DistanceKernel<Mode>` combining `lhs + rhs`.
        /// @details Wraps `lhs.root_` and `rhs.root_` in an immutable `Sum` node.
        /// @tparam Mode Distance kernel mode tag.
        /// @param[in] lhs Left-hand distance kernel summand.
        /// @param[in] rhs Right-hand distance kernel summand.
        /// @return Composite additive `DistanceKernel<Mode>`.
        template <DistanceModeConcept Mode>
        static DistanceKernel<Mode> make_sum(DistanceKernel<Mode> lhs, DistanceKernel<Mode> rhs)
        {
            return DistanceKernel<Mode>(std::make_shared<const Node<Mode>>(Node<Mode>{
                .data = typename Node<Mode>::Sum{.lhs = std::move(lhs.root_), .rhs = std::move(rhs.root_)}
            }));
        }

        /// @brief Constructs a multiplicative `DistanceKernel<Mode>` combining `lhs * rhs`.
        /// @details Wraps `lhs.root_` and `rhs.root_` in an immutable `Product` node.
        /// @tparam Mode Distance kernel mode tag.
        /// @param[in] lhs Left-hand distance kernel factor.
        /// @param[in] rhs Right-hand distance kernel factor.
        /// @return Composite multiplicative `DistanceKernel<Mode>`.
        template <DistanceModeConcept Mode>
        static DistanceKernel<Mode> make_product(DistanceKernel<Mode> lhs, DistanceKernel<Mode> rhs)
        {
            return DistanceKernel<Mode>(std::make_shared<const Node<Mode>>(Node<Mode>{
                .data = typename Node<Mode>::Product{.lhs = std::move(lhs.root_), .rhs = std::move(rhs.root_)}
            }));
        }

        /// @brief Recursively describes one AST node for persistence.
        /// @tparam Mode Distance kernel mode tag.
        /// @param[in] node AST node to describe.
        /// @param[in] slots Slots of the whole kernel, used to turn slot identifiers into ordinals.
        /// @return Description of the subtree.
        template <DistanceModeConcept Mode>
        static std::shared_ptr<const detail::DistanceNodeDesc> describe_node(
            const Node<Mode>& node, const std::vector<SlotDescriptor>& slots
        )
        {
            auto out = std::make_shared<detail::DistanceNodeDesc>();
            auto ordinal = [&slots](SlotId id) -> std::size_t
            {
                for (std::size_t i = 0; i < slots.size(); ++i)
                {
                    if (slots[i].id == id)
                    {
                        return i;
                    }
                }
                return 0;
            };
            if (const auto* s = std::get_if<SuppliedScalarNode>(&node.data))
            {
                out->kind = detail::DistanceNodeDesc::Kind::Supplied;
                out->slot = ordinal(s->slot);
                out->leaf = s->leaf;
            }
            else if (const auto* a = std::get_if<SuppliedArdNode>(&node.data))
            {
                out->kind = detail::DistanceNodeDesc::Kind::Supplied;
                out->slot = ordinal(a->slot);
                out->leaf = a->leaf;
            }
            else if (const auto* c = std::get_if<CoordLeafNode>(&node.data))
            {
                out->kind = detail::DistanceNodeDesc::Kind::Coord;
                out->leaf = c->leaf;
            }
            else if (const auto* sum = std::get_if<typename Node<Mode>::Sum>(&node.data))
            {
                out->kind = detail::DistanceNodeDesc::Kind::Sum;
                out->lhs = describe_node<Mode>(*sum->lhs, slots);
                out->rhs = describe_node<Mode>(*sum->rhs, slots);
            }
            else
            {
                const auto& prod = std::get<typename Node<Mode>::Product>(node.data);
                out->kind = detail::DistanceNodeDesc::Kind::Product;
                out->lhs = describe_node<Mode>(*prod.lhs, slots);
                out->rhs = describe_node<Mode>(*prod.rhs, slots);
            }
            return out;
        }

        /// @brief Describes a whole kernel for persistence.
        /// @tparam Mode Distance kernel mode tag.
        /// @param[in] kernel Kernel to describe.
        /// @return Description of the tree and of its slots.
        template <DistanceModeConcept Mode>
        static detail::DistanceKernelDesc describe(const DistanceKernel<Mode>& kernel)
        {
            const auto slots = kernel.slots();
            detail::DistanceKernelDesc desc;
            desc.root = describe_node<Mode>(*kernel.root_, slots);
            for (const auto& s : slots)
            {
                desc.slots.push_back(detail::DistanceSlotDesc{.ard_dims = s.ard_dims});
            }
            return desc;
        }
    };

    // --- ScalarDistance & ArdDistance ---

    /// @brief Creates a new `ScalarDistance` slot with a process-unique identifier.
    /// @details Atomically increments `g_next_slot_id`.
    ScalarDistance::ScalarDistance() : id_(SlotId(g_next_slot_id.fetch_add(1U, std::memory_order_relaxed))) {}

    /// @brief Binds an isotropic `RbfKernel` leaf to this scalar squared-distance slot.
    /// @details Produces a `DistanceOnly` kernel expression reading scalar slot `id_`.
    /// @param[in] leaf Isotropic RBF kernel definition.
    /// @return `DistanceKernel<DistanceOnly>` node bound to this slot.
    DistanceKernel<DistanceOnly> ScalarDistance::rbf(RbfKernel leaf) const
    {
        return DistanceKernel<DistanceOnly>(std::make_shared<const DistanceKernel<DistanceOnly>::InternalNode>(
            DistanceKernel<DistanceOnly>::InternalNode{
                .data = SuppliedScalarNode{.slot = id_, .leaf = leaf}
        }
        ));
    }

    /// @brief Binds an isotropic `MaternKernel` leaf to this scalar squared-distance slot.
    /// @details Produces a `DistanceOnly` kernel expression reading scalar slot `id_`.
    /// @param[in] leaf Isotropic Matern kernel definition.
    /// @return `DistanceKernel<DistanceOnly>` node bound to this slot.
    DistanceKernel<DistanceOnly> ScalarDistance::matern(MaternKernel leaf) const
    {
        return DistanceKernel<DistanceOnly>(std::make_shared<const DistanceKernel<DistanceOnly>::InternalNode>(
            DistanceKernel<DistanceOnly>::InternalNode{
                .data = SuppliedScalarNode{.slot = id_, .leaf = leaf}
        }
        ));
    }

    /// @brief Binds an isotropic `PeriodicKernel` leaf to this scalar squared-distance slot.
    /// @details Produces a `DistanceOnly` kernel expression reading scalar slot `id_`.
    /// @param[in] leaf Isotropic periodic kernel definition.
    /// @return `DistanceKernel<DistanceOnly>` node bound to this slot.
    DistanceKernel<DistanceOnly> ScalarDistance::periodic(PeriodicKernel leaf) const
    {
        return DistanceKernel<DistanceOnly>(std::make_shared<const DistanceKernel<DistanceOnly>::InternalNode>(
            DistanceKernel<DistanceOnly>::InternalNode{
                .data = SuppliedScalarNode{.slot = id_, .leaf = leaf}
        }
        ));
    }

    /// @brief Binds an isotropic `RationalQuadraticKernel` leaf to this scalar squared-distance slot.
    /// @details Produces a `DistanceOnly` kernel expression reading scalar slot `id_`.
    /// @param[in] leaf Isotropic rational-quadratic kernel definition.
    /// @return `DistanceKernel<DistanceOnly>` node bound to this slot.
    DistanceKernel<DistanceOnly> ScalarDistance::rational_quadratic(RationalQuadraticKernel leaf) const
    {
        return DistanceKernel<DistanceOnly>(std::make_shared<const DistanceKernel<DistanceOnly>::InternalNode>(
            DistanceKernel<DistanceOnly>::InternalNode{
                .data = SuppliedScalarNode{.slot = id_, .leaf = leaf}
        }
        ));
    }

    /// @brief Binds a user-defined `CustomKernel` leaf to this scalar squared-distance slot.
    /// @details Produces a `DistanceOnly` kernel expression passing scalar slot `id_` to `CustomKernel`.
    /// @param[in] leaf Custom kernel wrapper.
    /// @return `DistanceKernel<DistanceOnly>` node bound to this slot.
    DistanceKernel<DistanceOnly> ScalarDistance::custom(CustomKernel leaf) const
    {
        return DistanceKernel<DistanceOnly>(std::make_shared<const DistanceKernel<DistanceOnly>::InternalNode>(
            DistanceKernel<DistanceOnly>::InternalNode{
                .data = SuppliedScalarNode{.slot = id_, .leaf = std::move(leaf)}
        }
        ));
    }

    /// @brief Creates an `ArdDistance` slot expecting `dims` per-dimension distance blocks.
    /// @details Rejects `dims == 0` with `GprErrorCode::EmptyInput`.
    /// @param[in] dims Number of ARD distance dimensions (`>= 1`).
    /// @return Constructed `ArdDistance` slot or `GprErrorCode::EmptyInput`.
    /// @retval GprErrorCode::EmptyInput If `dims == 0`.
    std::expected<ArdDistance, GprError> ArdDistance::create(std::size_t dims)
    {
        if (dims == 0U)
        {
            return std::unexpected(GprError::empty_input());
        }
        return ArdDistance(SlotId(g_next_slot_id.fetch_add(1U, std::memory_order_relaxed)), dims);
    }

    /// @brief Binds an `RbfArdKernel` leaf whose lengthscale count matches `dims()`.
    /// @details Returns `GprErrorCode::DimensionMismatch` if `leaf.num_params() != dims_`.
    /// @param[in] leaf ARD RBF kernel leaf definition.
    /// @return `DistanceKernel<DistanceOnly>` bound to this ARD slot, or dimension mismatch error.
    /// @retval GprErrorCode::DimensionMismatch If `leaf.num_params() != dims()`.
    std::expected<DistanceKernel<DistanceOnly>, GprError> ArdDistance::rbf_ard(RbfArdKernel leaf) const
    {
        if (leaf.num_params() != dims_)
        {
            return std::unexpected(GprError::dimension_mismatch(leaf.num_params(), dims_));
        }
        return DistanceKernel<DistanceOnly>(std::make_shared<const DistanceKernel<DistanceOnly>::InternalNode>(
            DistanceKernel<DistanceOnly>::InternalNode{
                .data = SuppliedArdNode{.slot = id_, .dims = dims_, .leaf = std::move(leaf)}
        }
        ));
    }

    /// @brief Binds a `MaternArdKernel` leaf whose lengthscale count matches `dims()`.
    /// @details Returns `GprErrorCode::DimensionMismatch` if `leaf.num_params() != dims_`.
    /// @param[in] leaf ARD Matern kernel leaf definition.
    /// @return `DistanceKernel<DistanceOnly>` bound to this ARD slot, or dimension mismatch error.
    /// @retval GprErrorCode::DimensionMismatch If `leaf.num_params() != dims()`.
    std::expected<DistanceKernel<DistanceOnly>, GprError> ArdDistance::matern_ard(MaternArdKernel leaf) const
    {
        if (leaf.num_params() != dims_)
        {
            return std::unexpected(GprError::dimension_mismatch(leaf.num_params(), dims_));
        }
        return DistanceKernel<DistanceOnly>(std::make_shared<const DistanceKernel<DistanceOnly>::InternalNode>(
            DistanceKernel<DistanceOnly>::InternalNode{
                .data = SuppliedArdNode{.slot = id_, .dims = dims_, .leaf = std::move(leaf)}
        }
        ));
    }

    /// @brief Binds a `RationalQuadraticArdKernel` leaf whose lengthscale count matches `dims()`.
    /// @details Returns `GprErrorCode::DimensionMismatch` if `leaf.lengthscales().size() != dims_`.
    /// @param[in] leaf ARD rational-quadratic kernel leaf definition.
    /// @return `DistanceKernel<DistanceOnly>` bound to this ARD slot, or dimension mismatch error.
    /// @retval GprErrorCode::DimensionMismatch If `leaf.lengthscales().size() != dims()`.
    std::expected<DistanceKernel<DistanceOnly>, GprError> ArdDistance::rational_quadratic_ard(
        RationalQuadraticArdKernel leaf
    ) const
    {
        const std::size_t leaf_dims = leaf.lengthscales().size();
        if (leaf_dims != dims_)
        {
            return std::unexpected(GprError::dimension_mismatch(leaf_dims, dims_));
        }
        return DistanceKernel<DistanceOnly>(std::make_shared<const DistanceKernel<DistanceOnly>::InternalNode>(
            DistanceKernel<DistanceOnly>::InternalNode{
                .data = SuppliedArdNode{.slot = id_, .dims = dims_, .leaf = std::move(leaf)}
        }
        ));
    }

    // --- DistanceKernel<Mode> ---

    /// @brief Constructs a `DistanceKernel<Mode>` wrapping a `ConstantKernel` leaf.
    /// @details Constant leaves read neither supplied distances nor coordinates.
    /// @tparam Mode Distance kernel mode tag.
    /// @param[in] leaf Constant kernel amplitude leaf.
    template <DistanceModeConcept Mode>
    DistanceKernel<Mode>::DistanceKernel(ConstantKernel leaf) :
        root_(std::make_shared<const InternalNode>(InternalNode{.data = CoordLeafNode{.leaf = leaf}}))
    {
    }

    /// @brief Constructs a `DistanceKernel<Mode>` wrapping a `WhiteKernel` leaf.
    /// @details White-noise leaves contribute only to square training Gram diagonals.
    /// @tparam Mode Distance kernel mode tag.
    /// @param[in] leaf White-noise kernel leaf.
    template <DistanceModeConcept Mode>
    DistanceKernel<Mode>::DistanceKernel(WhiteKernel leaf) :
        root_(std::make_shared<const InternalNode>(InternalNode{.data = CoordLeafNode{.leaf = leaf}}))
    {
    }

    /// @brief Promotes a `DistanceKernel<DistanceOnly>` into a `DistanceKernel<WithPoints>` tree.
    /// @details Recursively rebuilds the shared AST using `WithPoints` nodes.
    /// @tparam Mode Target mode tag (constrained to `WithPoints`).
    /// @param[in] other Source `DistanceOnly` kernel expression.
    template <DistanceModeConcept Mode>
    DistanceKernel<Mode>::DistanceKernel(const DistanceKernel<DistanceOnly>& other)
        requires std::same_as<Mode, WithPoints>
        : root_(DistanceKernelAccess::promote_node<WithPoints>(*other.root_))
    {
    }

    /// @brief Constructs a `DistanceKernel<WithPoints>` from a coordinate-based `KernelSpec`.
    /// @details Converts all leaves into `CoordLeafNode` entries evaluated from point coordinates.
    /// @tparam Mode Target mode tag (constrained to `WithPoints`).
    /// @param[in] coord_spec Source coordinate `KernelSpec` expression tree.
    template <DistanceModeConcept Mode>
    DistanceKernel<Mode>::DistanceKernel(const KernelSpec& coord_spec)
        requires std::same_as<Mode, WithPoints>
        : root_(DistanceKernelAccess::from_coord_spec(coord_spec))
    {
    }

    /// @brief Returns a `DistanceKernel<WithPoints>` view or promotion of `*this`.
    /// @details Copies directly when already `WithPoints`, or invokes the promotion constructor when `DistanceOnly`.
    /// @tparam Mode Distance kernel mode tag.
    /// @return Equivalent `DistanceKernel<WithPoints>`.
    template <DistanceModeConcept Mode> DistanceKernel<WithPoints> DistanceKernel<Mode>::with_points() const
    {
        if constexpr (std::same_as<Mode, WithPoints>)
        {
            return *this;
        }
        else
        {
            return DistanceKernel<WithPoints>(*this);
        }
    }

    /// @brief Returns the total number of log-space hyperparameters in the expression tree.
    /// @details Sums leaf parameter counts via `DistanceKernelAccess::count_params`.
    /// @tparam Mode Distance kernel mode tag.
    /// @return Total hyperparameter count.
    template <DistanceModeConcept Mode> std::size_t DistanceKernel<Mode>::num_params() const noexcept
    {
        return DistanceKernelAccess::count_params<Mode>(*root_);
    }

    /// @brief Writes flattened log-space hyperparameters in depth-first left-to-right order into `out`.
    /// @details Validates `out.size() == num_params()` before walking the tree.
    /// @tparam Mode Distance kernel mode tag.
    /// @param[out] out Destination hyperparameter span.
    /// @return Empty expected on success, or length mismatch error.
    /// @retval GprErrorCode::LengthMismatch If `out.size() != num_params()`.
    template <DistanceModeConcept Mode>
    std::expected<void, GprError> DistanceKernel<Mode>::get_params(std::span<double> out) const
    {
        if (auto cnt = require_count(out.size(), num_params(), "out"); !cnt.has_value())
        {
            return cnt;
        }
        std::expected<void, GprError> status{};
        DistanceKernelAccess::fill_params<Mode>(*root_, out, 0, status);
        return status;
    }

    /// @brief Atomically updates all leaf parameters from log-space `params`.
    /// @details Constructs an updated persistent AST first and commits `root_` only when every leaf validates.
    /// @tparam Mode Distance kernel mode tag.
    /// @param[in] params Log-space hyperparameter span of length `num_params()`.
    /// @return Empty expected on success, or length/bounds error leaving `*this` unchanged.
    /// @retval GprErrorCode::LengthMismatch If `params.size() != num_params()`.
    /// @retval GprErrorCode::InvalidHyperparameter If any value lies outside its open parameter interval.
    template <DistanceModeConcept Mode>
    std::expected<void, GprError> DistanceKernel<Mode>::set_params(std::span<const double> params)
    {
        if (auto cnt = require_count(params.size(), num_params(), "params"); !cnt.has_value())
        {
            return cnt;
        }
        std::size_t offset = 0;
        auto next = DistanceKernelAccess::with_updated_params<Mode>(*root_, params, offset);
        if (!next.has_value())
        {
            return std::unexpected(std::move(next).error());
        }
        root_ = std::move(*next);
        return {};
    }

    /// @brief Writes open hyperparameter intervals in depth-first left-to-right order into `out`.
    /// @details Validates `out.size() == num_params()` before collecting intervals.
    /// @tparam Mode Distance kernel mode tag.
    /// @param[out] out Destination span of `Interval` bounds.
    /// @return Empty expected on success, or length mismatch error.
    /// @retval GprErrorCode::LengthMismatch If `out.size() != num_params()`.
    template <DistanceModeConcept Mode>
    std::expected<void, GprError> DistanceKernel<Mode>::bounds_into(std::span<Interval> out) const
    {
        if (auto cnt = require_count(out.size(), num_params(), "out"); !cnt.has_value())
        {
            return cnt;
        }
        std::expected<void, GprError> status{};
        DistanceKernelAccess::fill_bounds<Mode>(*root_, out, 0, status);
        return status;
    }

    /// @brief Returns flattened parameter-to-leaf bindings in depth-first left-to-right order.
    /// @details Walks the immutable AST via `DistanceKernelAccess::collect_bindings`.
    /// @tparam Mode Distance kernel mode tag.
    /// @return Vector of `ParameterBinding` descriptors.
    template <DistanceModeConcept Mode> std::vector<ParameterBinding> DistanceKernel<Mode>::parameter_bindings() const
    {
        std::vector<ParameterBinding> out;
        out.reserve(num_params());
        std::size_t next_leaf = 0;
        std::size_t next_param = 0;
        DistanceKernelAccess::collect_bindings<Mode>(*root_, next_leaf, next_param, out);
        return out;
    }

    /// @brief Returns unique distance slot descriptors referenced by this tree in depth-first first-appearance order.
    /// @details Assigns separate sequential `at` indices for scalar and ARD slots.
    /// @tparam Mode Distance kernel mode tag.
    /// @return Vector of unique `SlotDescriptor` records.
    template <DistanceModeConcept Mode> std::vector<SlotDescriptor> DistanceKernel<Mode>::slots() const
    {
        std::vector<SlotDescriptor> out;
        std::size_t next_scalar = 0;
        std::size_t next_ard = 0;
        DistanceKernelAccess::collect_slots<Mode>(*root_, out, next_scalar, next_ard);
        return out;
    }

    /// @brief Returns the count of unique `ScalarDistance` slots referenced by this tree.
    /// @details Filters `slots()` entries with `!ard_dims.has_value()`.
    /// @tparam Mode Distance kernel mode tag.
    /// @return Number of scalar distance slots.
    template <DistanceModeConcept Mode> std::size_t DistanceKernel<Mode>::num_scalar_slots() const
    {
        std::size_t count = 0;
        for (const auto& s : slots())
        {
            if (!s.ard_dims.has_value())
            {
                ++count;
            }
        }
        return count;
    }

    /// @brief Returns the count of unique `ArdDistance` slots referenced by this tree.
    /// @details Filters `slots()` entries with `ard_dims.has_value()`.
    /// @tparam Mode Distance kernel mode tag.
    /// @return Number of ARD distance slots.
    template <DistanceModeConcept Mode> std::size_t DistanceKernel<Mode>::num_ard_slots() const
    {
        std::size_t count = 0;
        for (const auto& s : slots())
        {
            if (s.ard_dims.has_value())
            {
                ++count;
            }
        }
        return count;
    }

    /// @brief Combines two `DistanceOnly` kernels by addition.
    /// @details Builds a composite `Sum` node in `DistanceOnly` mode.
    /// @param[in] lhs Left-hand summand kernel.
    /// @param[in] rhs Right-hand summand kernel.
    /// @return Additive `DistanceKernel<DistanceOnly>`.
    DistanceKernel<DistanceOnly> operator+(DistanceKernel<DistanceOnly> lhs, DistanceKernel<DistanceOnly> rhs)
    {
        return DistanceKernelAccess::make_sum<DistanceOnly>(std::move(lhs), std::move(rhs));
    }

    /// @brief Combines two `DistanceOnly` kernels by multiplication.
    /// @details Builds a composite `Product` node in `DistanceOnly` mode.
    /// @param[in] lhs Left-hand factor kernel.
    /// @param[in] rhs Right-hand factor kernel.
    /// @return Multiplicative `DistanceKernel<DistanceOnly>`.
    DistanceKernel<DistanceOnly> operator*(DistanceKernel<DistanceOnly> lhs, DistanceKernel<DistanceOnly> rhs)
    {
        return DistanceKernelAccess::make_product<DistanceOnly>(std::move(lhs), std::move(rhs));
    }

    /// @brief Combines two `WithPoints` kernels by addition.
    /// @details Builds a composite `Sum` node in `WithPoints` mode.
    /// @param[in] lhs Left-hand summand kernel.
    /// @param[in] rhs Right-hand summand kernel.
    /// @return Additive `DistanceKernel<WithPoints>`.
    DistanceKernel<WithPoints> operator+(DistanceKernel<WithPoints> lhs, DistanceKernel<WithPoints> rhs)
    {
        return DistanceKernelAccess::make_sum<WithPoints>(std::move(lhs), std::move(rhs));
    }

    /// @brief Combines two `WithPoints` kernels by multiplication.
    /// @details Builds a composite `Product` node in `WithPoints` mode.
    /// @param[in] lhs Left-hand factor kernel.
    /// @param[in] rhs Right-hand factor kernel.
    /// @return Multiplicative `DistanceKernel<WithPoints>`.
    DistanceKernel<WithPoints> operator*(DistanceKernel<WithPoints> lhs, DistanceKernel<WithPoints> rhs)
    {
        return DistanceKernelAccess::make_product<WithPoints>(std::move(lhs), std::move(rhs));
    }

    /// @brief Combines a `DistanceOnly` kernel and a coordinate `KernelSpec` by addition in `WithPoints` mode.
    /// @details Promotes `lhs` and wraps `rhs` before summing.
    /// @param[in] lhs Supplied-distance summand kernel.
    /// @param[in] rhs Coordinate-based summand specification.
    /// @return Composite `DistanceKernel<WithPoints>`.
    DistanceKernel<WithPoints> operator+(const DistanceKernel<DistanceOnly>& lhs, const KernelSpec& rhs)
    {
        return DistanceKernel<WithPoints>(lhs) + DistanceKernel<WithPoints>(rhs);
    }

    /// @brief Combines a coordinate `KernelSpec` and a `DistanceOnly` kernel by addition in `WithPoints` mode.
    /// @details Wraps `lhs` and promotes `rhs` before summing.
    /// @param[in] lhs Coordinate-based summand specification.
    /// @param[in] rhs Supplied-distance summand kernel.
    /// @return Composite `DistanceKernel<WithPoints>`.
    DistanceKernel<WithPoints> operator+(const KernelSpec& lhs, const DistanceKernel<DistanceOnly>& rhs)
    {
        return DistanceKernel<WithPoints>(lhs) + DistanceKernel<WithPoints>(rhs);
    }

    /// @brief Combines a `DistanceOnly` kernel and a coordinate `KernelSpec` by multiplication in `WithPoints` mode.
    /// @details Promotes `lhs` and wraps `rhs` before multiplying.
    /// @param[in] lhs Supplied-distance factor kernel.
    /// @param[in] rhs Coordinate-based factor specification.
    /// @return Composite `DistanceKernel<WithPoints>`.
    DistanceKernel<WithPoints> operator*(const DistanceKernel<DistanceOnly>& lhs, const KernelSpec& rhs)
    {
        return DistanceKernel<WithPoints>(lhs) * DistanceKernel<WithPoints>(rhs);
    }

    /// @brief Combines a coordinate `KernelSpec` and a `DistanceOnly` kernel by multiplication in `WithPoints` mode.
    /// @details Wraps `lhs` and promotes `rhs` before multiplying.
    /// @param[in] lhs Coordinate-based factor specification.
    /// @param[in] rhs Supplied-distance factor kernel.
    /// @return Composite `DistanceKernel<WithPoints>`.
    DistanceKernel<WithPoints> operator*(const KernelSpec& lhs, const DistanceKernel<DistanceOnly>& rhs)
    {
        return DistanceKernel<WithPoints>(lhs) * DistanceKernel<WithPoints>(rhs);
    }

    // --- DistanceSource ---

    /// @brief Appends a scalar squared-distance table binding for `slot`.
    /// @details Validation of slot compatibility, duplicates, shapes, and values occurs during `prepare_square` or
    /// `prepare_cross`.
    /// @param[in] slot Scalar slot identifier.
    /// @param[in] d2 Supplied squared-distance matrix view (`n x n` for square, `n x m` for cross).
    /// @return Reference to `*this` for fluent chaining.
    DistanceSource& DistanceSource::bind_scalar(SlotId slot, MatrixRef<double> d2)
    {
        scalars_.push_back(ScalarEntry{.slot = slot, .d2 = d2});
        return *this;
    }

    /// @brief Appends per-dimension ARD squared-distance tables for `slot`.
    /// @details Copies the span of `MatrixRef<double>` views; validation and packing occur during `prepare_*`.
    /// @param[in] slot ARD slot identifier.
    /// @param[in] d2_per_dim Span of per-dimension squared-distance matrix views.
    /// @return Reference to `*this` for fluent chaining.
    DistanceSource& DistanceSource::bind_ard(SlotId slot, std::span<const MatrixRef<double>> d2_per_dim)
    {
        ards_.push_back(
            ArdEntry{.slot = slot, .d2_per_dim = std::vector<MatrixRef<double>>(d2_per_dim.begin(), d2_per_dim.end())}
        );
        return *this;
    }

    /// @brief Enables repair of minor negative, non-zero diagonal, and asymmetry violations within `rel_tol`.
    /// @details Stores `rel_tol` for scaling against `max_abs` in `DistanceStoreBuilder`; negative or non-finite values
    /// are rejected when building a `CheckedDistanceStore`.
    /// @param[in] rel_tol Non-negative relative tolerance threshold.
    /// @return Reference to `*this` for fluent chaining.
    DistanceSource& DistanceSource::tidy(double rel_tol)
    {
        tidy_rel_tol_ = rel_tol;
        return *this;
    }

    // --- CheckedDistanceStore<T> ---

    /// @brief Internal storage backing a validated (and optionally repaired) `CheckedDistanceStore<T>`.
    /// @details Holds slot metadata, shape flags, dense scalar squared-distance tables, and packed lower-triangular
    /// (for square) or dense column-major (for cross) per-dimension ARD blocks.
    /// @tparam T Storage floating-point scalar type.
    template <KernelScalarConcept T> struct CheckedDistanceStore<T>::Impl
    {
        /// @brief Ordered slot descriptors of the `CompiledDistanceKernel` that built this store.
        std::vector<SlotDescriptor> kernel_slots;
        /// @brief Row count of every stored distance block.
        std::size_t n_rows = 0;
        /// @brief Column count of every stored distance block.
        std::size_t n_cols = 0;
        /// @brief True for square training distance stores, false for rectangular cross stores.
        bool square = true;
        // Per scalar slot: dense n_rows x n_cols column-major data.
        /// @brief Dense column-major squared-distance tables indexed by scalar slot `at`.
        std::vector<std::vector<T>> scalar_tables;
        // Per ARD slot (when square): packed lower triangles m * per_dim + p.
        // Per ARD slot (when cross): dims * n_rows * n_cols dense blocks.
        /// @brief Feature dimensionality stored per ARD slot indexed by ARD slot `at`.
        std::vector<std::size_t> ard_dims;
        /// @brief Packed lower-triangular (square) or dense block (cross) ARD squared-distance arrays.
        std::vector<std::vector<T>> ard_data;
    };

    /// @brief Constructs a `CheckedDistanceStore<T>` from its validated internal state.
    /// @details Internal constructor called by `DistanceStoreBuilder::build`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] impl Owning pointer to validated distance store state.
    template <KernelScalarConcept T>
    CheckedDistanceStore<T>::CheckedDistanceStore(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl))
    {
    }

    /// @brief Destroys the `CheckedDistanceStore<T>` and its internal buffers.
    /// @details Defined out-of-line where `CheckedDistanceStore<T>::Impl` is complete.
    /// @tparam T Storage floating-point scalar type.
    template <KernelScalarConcept T> CheckedDistanceStore<T>::~CheckedDistanceStore() = default;

    /// @brief Move-constructs a `CheckedDistanceStore<T>`, transferring the validated tables.
    /// @details Defined out-of-line where `CheckedDistanceStore<T>::Impl` is complete so the defaulted move can
    /// destroy and relocate the owning pointer.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] other Store instance to move from.
    template <KernelScalarConcept T>
    CheckedDistanceStore<T>::CheckedDistanceStore(CheckedDistanceStore&& other) noexcept = default;

    /// @brief Move-assigns a `CheckedDistanceStore<T>`, transferring the validated tables.
    /// @details Defined out-of-line where `CheckedDistanceStore<T>::Impl` is complete.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] other Store instance to move from.
    /// @return Reference to this store.
    template <KernelScalarConcept T>
    CheckedDistanceStore<T>& CheckedDistanceStore<T>::operator=(CheckedDistanceStore&& other) noexcept = default;

    /// @brief Copy-constructs a `CheckedDistanceStore<T>` with an independent deep copy of stored tables.
    /// @details Duplicates `Impl` vectors including scalar and ARD tables.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] other Store instance to copy.
    template <KernelScalarConcept T>
    CheckedDistanceStore<T>::CheckedDistanceStore(const CheckedDistanceStore& other) :
        impl_(std::make_unique<Impl>(*other.impl_))
    {
    }

    /// @brief Copy-assigns a `CheckedDistanceStore<T>` with an independent deep copy of stored tables.
    /// @details Guards against self-assignment before cloning `other.impl_`.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] other Store instance to copy.
    /// @return Reference to `*this`.
    template <KernelScalarConcept T>
    CheckedDistanceStore<T>& CheckedDistanceStore<T>::operator=(const CheckedDistanceStore& other)
    {
        if (this != &other)
        {
            impl_ = std::make_unique<Impl>(*other.impl_);
        }
        return *this;
    }

    /// @brief Returns the row count of each stored squared-distance block.
    /// @details Equals training point count for square stores or first point set size for cross stores.
    /// @tparam T Storage floating-point scalar type.
    /// @return Row count `n_rows`.
    template <KernelScalarConcept T> std::size_t CheckedDistanceStore<T>::rows() const noexcept
    {
        return impl_->n_rows;
    }

    /// @brief Returns the column count of each stored squared-distance block.
    /// @details Equals training point count for square stores or query point set size for cross stores.
    /// @tparam T Storage floating-point scalar type.
    /// @return Column count `n_cols`.
    template <KernelScalarConcept T> std::size_t CheckedDistanceStore<T>::cols() const noexcept
    {
        return impl_->n_cols;
    }

    /// @brief Returns true if this store holds square training distance blocks.
    /// @details Distinguishes `prepare_square` outputs from rectangular `prepare_cross` outputs.
    /// @tparam T Storage floating-point scalar type.
    /// @return True when square training store, false when rectangular cross store.
    template <KernelScalarConcept T> bool CheckedDistanceStore<T>::is_square() const noexcept
    {
        return impl_->square;
    }

    // --- CompiledDistanceKernel<T>::Impl ---

    /// @brief PIMPL state for a compiled mixed supplied-distance and coordinate kernel.
    /// @details Stores flattened additive and multiplicative AST nodes, parameter bindings, distance slot descriptors,
    /// and cached ARD lengthscales to support zero-allocation evaluation, gradients, and Hessians.
    /// @tparam T Scalar type (@c double or @c float).
    template <KernelScalarConcept T> struct CompiledDistanceKernel<T>::Impl
    {
        /// @brief Compiled reference to a scalar supplied-distance leaf.
        struct SuppliedScalarRef
        {
            /// @brief Index of this leaf in @c Impl::leaves and @c Impl::cached_ls.
            std::size_t leaf_idx = 0;
            /// @brief Packed slot index inside a @c CheckedDistanceStore scalar slab array.
            std::size_t slot_at = 0;
            /// @brief Inclusive start index in the flattened hyperparameter vector.
            std::size_t param_begin = 0;
            /// @brief Number of hyperparameters owned by this leaf.
            std::size_t param_count = 0;
        };

        /// @brief Compiled reference to a per-dimension supplied-ARD-distance leaf.
        struct SuppliedArdRef
        {
            /// @brief Index of this leaf in @c Impl::leaves and @c Impl::cached_ls.
            std::size_t leaf_idx = 0;
            /// @brief Packed slot index inside a @c CheckedDistanceStore ARD slab array.
            std::size_t slot_at = 0;
            /// @brief Number of ARD dimensions required by this slot and leaf.
            std::size_t dims = 0;
            /// @brief Inclusive start index in the flattened hyperparameter vector.
            std::size_t param_begin = 0;
            /// @brief Number of hyperparameters owned by this leaf.
            std::size_t param_count = 0;
        };

        /// @brief Compiled reference to a coordinate-only kernel leaf.
        struct CoordLeafRef
        {
            /// @brief Index of this leaf in @c Impl::leaves and @c Impl::cached_ls.
            std::size_t leaf_idx = 0;
            /// @brief Inclusive start index in the flattened hyperparameter vector.
            std::size_t param_begin = 0;
            /// @brief Number of hyperparameters owned by this leaf.
            std::size_t param_count = 0;
        };

        /// @brief Forward declaration of a compiled additive term node.
        struct SumTerm;
        /// @brief Forward declaration of a compiled multiplicative term node.
        struct ProductTerm;
        /// @brief Recursive variant representing any leaf reference or composite sum/product term.
        using Term = std::variant<SuppliedScalarRef, SuppliedArdRef, CoordLeafRef, SumTerm, ProductTerm>;

        /// @brief Flattened additive node summing its child terms.
        struct SumTerm
        {
            /// @brief Child subterms summed together.
            std::vector<Term> terms;
            /// @brief Inclusive start index of parameters contained anywhere inside this sum subtree.
            std::size_t param_begin = 0;
            /// @brief Exclusive end index of parameters contained anywhere inside this sum subtree.
            std::size_t param_end = 0;
        };

        /// @brief Flattened multiplicative node multiplying its child terms.
        struct ProductTerm
        {
            /// @brief Child subterms multiplied together.
            std::vector<Term> terms;
            /// @brief Inclusive start index of parameters contained anywhere inside this product subtree.
            std::size_t param_begin = 0;
            /// @brief Exclusive end index of parameters contained anywhere inside this product subtree.
            std::size_t param_end = 0;
        };

        /// @brief Flattened leaf kernel instances ordered by pre-order AST traversal.
        std::vector<KernelLeaf> leaves;
        /// @brief Per-leaf cached ARD lengthscale vectors (empty for non-ARD leaves) to avoid allocation in loops.
        std::vector<std::vector<double>> cached_ls;
        /// @brief Mapping from flattened hyperparameter indices to leaf index and local parameter offset.
        std::vector<ParameterBinding> bindings;
        /// @brief Deduplicated supplied-distance slot descriptors required by this compiled kernel.
        std::vector<SlotDescriptor> slots;
        /// @brief Count of distinct scalar distance slots referenced by the kernel.
        std::size_t num_scalar_slots = 0;
        /// @brief Count of distinct ARD distance slots referenced by the kernel.
        std::size_t num_ard_slots = 0;
        /// @brief Root term of the compiled and associatively flattened kernel expression tree.
        Term root;
        /// @brief True when at least one coordinate-dependent leaf requires non-null input point matrices.
        bool has_coord_points = false;
        /// @brief Maximum scratch nesting depth required when evaluating nested sums and products.
        std::size_t nested_depth = 0;
        /// @brief Exponential evaluation precision mode shared by @c Rbf, @c Matern52, and @c Matern32 terms.
        KernelExp exp_mode = KernelExp::Accurate;

        /// @brief Default-constructs an empty compiled distance implementation.
        Impl() = default;

        /// @brief Copy-constructs a compiled distance implementation iteratively so deep ASTs stay stack-bounded.
        /// @param[in] other Source compiled distance implementation.
        Impl(const Impl& other) :
            leaves(other.leaves),
            cached_ls(other.cached_ls),
            bindings(other.bindings),
            slots(other.slots),
            num_scalar_slots(other.num_scalar_slots),
            num_ard_slots(other.num_ard_slots),
            root(clone_term_iterative(other.root)),
            has_coord_points(other.has_coord_points),
            nested_depth(other.nested_depth),
            exp_mode(other.exp_mode)
        {
        }

        /// @brief Move-constructs a compiled distance implementation.
        Impl(Impl&&) noexcept = default;

        /// @brief Copy-assigns a compiled distance implementation iteratively.
        /// @param[in] other Source compiled distance implementation.
        /// @return Reference to *this.
        Impl& operator=(const Impl& other)
        {
            if (this == &other)
            {
                return *this;
            }
            Impl copy(other);
            *this = std::move(copy);
            return *this;
        }

        /// @brief Move-assigns a compiled distance implementation.
        /// @return Reference to *this.
        Impl& operator=(Impl&&) noexcept = default;

        /// @brief Destroys the compiled distance AST iteratively to keep stack depth O(1) on deep trees.
        ~Impl() noexcept { destroy_term_iterative(root); }

        /// @brief Iteratively deep-copies a compiled distance Term tree without recursive calls.
        /// @param[in] src Root compiled term to clone.
        /// @return Cloned compiled term.
        [[nodiscard]] static Term clone_term_iterative(const Term& src)
        {
            if (is_leaf_term(src))
            {
                return src;
            }

            struct Frame
            {
                const std::vector<Term>* src_terms;
                std::size_t idx;
                bool is_sum;
                std::size_t param_begin;
                std::size_t param_end;
                std::vector<Term> dst_terms;
            };

            std::vector<Frame> stack;
            auto push_composite = [&stack](const Term& t)
            {
                if (const auto* s = std::get_if<SumTerm>(&t))
                {
                    stack.push_back(
                        Frame{
                            .src_terms = &s->terms,
                            .idx = 0U,
                            .is_sum = true,
                            .param_begin = s->param_begin,
                            .param_end = s->param_end,
                            .dst_terms = {},
                        }
                    );
                    stack.back().dst_terms.reserve(s->terms.size());
                }
                else
                {
                    const auto& p = std::get<ProductTerm>(t);
                    stack.push_back(
                        Frame{
                            .src_terms = &p.terms,
                            .idx = 0U,
                            .is_sum = false,
                            .param_begin = p.param_begin,
                            .param_end = p.param_end,
                            .dst_terms = {},
                        }
                    );
                    stack.back().dst_terms.reserve(p.terms.size());
                }
            };
            push_composite(src);
            while (true)
            {
                auto& top = stack.back();
                if (top.idx >= top.src_terms->size())
                {
                    Term finished = top.is_sum ? Term(
                                                     SumTerm{
                                                         .terms = std::move(top.dst_terms),
                                                         .param_begin = top.param_begin,
                                                         .param_end = top.param_end,
                                                     }
                                                 )
                                               : Term(
                                                     ProductTerm{
                                                         .terms = std::move(top.dst_terms),
                                                         .param_begin = top.param_begin,
                                                         .param_end = top.param_end,
                                                     }
                                                 );
                    stack.pop_back();
                    if (stack.empty())
                    {
                        return finished;
                    }
                    stack.back().dst_terms.push_back(std::move(finished));
                    continue;
                }
                const Term& child = (*top.src_terms)[top.idx++];
                if (is_leaf_term(child))
                {
                    top.dst_terms.push_back(child);
                }
                else
                {
                    push_composite(child);
                }
            }
        }

        /// @brief Iteratively destroys nested SumTerm/ProductTerm subtrees without recursive calls.
        /// @param[in,out] target Root compiled term to drain.
        static void destroy_term_iterative(Term& target) noexcept
        {
            std::vector<Term> pending;
            try
            {
                pending.push_back(std::move(target));
                while (!pending.empty())
                {
                    Term cur = std::move(pending.back());
                    pending.pop_back();
                    if (auto* s = std::get_if<SumTerm>(&cur))
                    {
                        for (auto& child : s->terms)
                        {
                            pending.push_back(std::move(child));
                        }
                        s->terms.clear();
                    }
                    else if (auto* p = std::get_if<ProductTerm>(&cur))
                    {
                        for (auto& child : p->terms)
                        {
                            pending.push_back(std::move(child));
                        }
                        p->terms.clear();
                    }
                }
            }
            catch (...)
            {
                // Fallback to bounded cleanup if OOM occurs while draining; noexcept boundary must not throw.
                return;
            }
        }

        /// @brief Synchronizes @c cached_ls with the current ARD lengthscale vectors stored in @c leaves.
        void refresh_cached_ls()
        {
            cached_ls.resize(leaves.size());
            for (std::size_t idx = 0; idx < leaves.size(); ++idx)
            {
                if (const auto* k = std::get_if<RbfArdKernel>(&leaves[idx]))
                {
                    cached_ls[idx] = k->lengthscales();
                }
                else if (const auto* km = std::get_if<MaternArdKernel>(&leaves[idx]))
                {
                    cached_ls[idx] = km->lengthscales();
                }
                else if (const auto* krq = std::get_if<RationalQuadraticArdKernel>(&leaves[idx]))
                {
                    cached_ls[idx] = krq->lengthscales();
                }
                else
                {
                    cached_ls[idx].clear();
                }
            }
        }

        /// @brief Returns the half-open global parameter index range @c [begin, end) owned by @p term.
        /// @param[in] term Compiled AST term to inspect.
        /// @return Pair @c {begin, end} of flattened hyperparameter indices.
        static std::pair<std::size_t, std::size_t> param_range(const Term& term) noexcept
        {
            if (const auto* s = std::get_if<SuppliedScalarRef>(&term))
            {
                return {s->param_begin, s->param_begin + s->param_count};
            }
            if (const auto* a = std::get_if<SuppliedArdRef>(&term))
            {
                return {a->param_begin, a->param_begin + a->param_count};
            }
            if (const auto* c = std::get_if<CoordLeafRef>(&term))
            {
                return {c->param_begin, c->param_begin + c->param_count};
            }
            if (const auto* sum = std::get_if<SumTerm>(&term))
            {
                return {sum->param_begin, sum->param_end};
            }
            if (const auto* prod = std::get_if<ProductTerm>(&term))
            {
                return {prod->param_begin, prod->param_end};
            }
            return {0, 0};
        }

        /// @brief Checks whether flattened hyperparameter index @p param_idx belongs to @p term.
        /// @param[in] term Compiled AST term to inspect.
        /// @param[in] param_idx Global flattened parameter index.
        /// @return @c true if @p param_idx lies inside @p term's half-open parameter range.
        static bool contains_param(const Term& term, std::size_t param_idx) noexcept
        {
            const auto [begin, end] = param_range(term);
            return param_idx >= begin && param_idx < end;
        }

        /// @brief Reports whether @p term represents a leaf node rather than a composite sum or product.
        /// @param[in] term Compiled AST term to test.
        /// @return @c true if @p term is a supplied-scalar, supplied-ARD, or coordinate leaf reference.
        static bool is_leaf_term(const Term& term) noexcept
        {
            return std::holds_alternative<SuppliedScalarRef>(term) || std::holds_alternative<SuppliedArdRef>(term)
                || std::holds_alternative<CoordLeafRef>(term);
        }

        /// @brief Computes the nested temporary-buffer stack depth needed to evaluate @p term.
        /// @details Leaf terms evaluate directly into the destination buffer and require zero nested depth.
        /// @param[in] term Compiled AST term to analyze.
        /// @return Minimum scratch buffer depth required for nested evaluation of @p term.
        static std::size_t compute_nested_depth(const Term& term) noexcept
        {
            struct DepthFrame
            {
                const Term* node;
                std::size_t depth;
            };

            std::vector<DepthFrame> work;
            std::size_t max_depth = 0U;
            try
            {
                work.push_back(DepthFrame{.node = &term, .depth = 0U});
                while (!work.empty())
                {
                    const DepthFrame cur = work.back();
                    work.pop_back();
                    const std::vector<Term>* children = nullptr;
                    if (const auto* s = std::get_if<SumTerm>(cur.node))
                    {
                        children = &s->terms;
                    }
                    else if (const auto* p = std::get_if<ProductTerm>(cur.node))
                    {
                        children = &p->terms;
                    }
                    else
                    {
                        continue;
                    }
                    for (const auto& child : *children)
                    {
                        if (!is_leaf_term(child))
                        {
                            const std::size_t child_depth = cur.depth + 1U;
                            max_depth = std::max(max_depth, child_depth);
                            work.push_back(DepthFrame{.node = &child, .depth = child_depth});
                        }
                    }
                }
            }
            catch (...)
            {
                return max_depth;
            }
            return max_depth;
        }

        /// @brief Looks up the packed @c at index assigned to supplied-distance slot @p id.
        /// @param[in] slot_list Kernel slot descriptors sorted by scalar/ARD packing order.
        /// @param[in] id Slot identifier to locate.
        /// @return Packed slab offset @c desc.at, or zero if not found.
        static std::size_t find_slot_at(const std::vector<SlotDescriptor>& slot_list, SlotId id) noexcept
        {
            for (const auto& desc : slot_list)
            {
                if (desc.id == id)
                {
                    return desc.at;
                }
            }
            return 0;
        }

        /// @brief Iteratively compiles a distance-kernel AST node into a flattened @c Term tree.
        /// @tparam Mode Distance kernel mode tag.
        /// @param[in] root_node Root AST node to compile.
        /// @param[in,out] out Target @c Impl receiving leaf state, bindings, and coordinate flags.
        /// @param[in,out] next_param_idx Running global parameter counter incremented as leaves are visited.
        /// @return Compiled @c Term corresponding to @p root_node.
        template <DistanceModeConcept Mode>
        static Term compile_node(
            const DistanceKernel<Mode>::InternalNode& root_node, Impl& out, std::size_t& next_param_idx
        )
        {
            using Node = DistanceKernel<Mode>::InternalNode;
            enum class FrameKind : std::uint8_t
            {
                Sum,
                Product,
            };

            struct CompileFrame
            {
                FrameKind kind;
                std::vector<Term> terms;
                std::size_t param_begin;
            };

            struct WorkItem
            {
                const Node* node{nullptr};
                bool close_frame{false};
            };

            std::vector<CompileFrame> frames;
            std::vector<WorkItem> work;
            std::vector<const Node*> flatten;
            Term compiled_root;

            auto emit_term = [&frames, &compiled_root](Term&& t)
            {
                if (frames.empty())
                {
                    compiled_root = std::move(t);
                }
                else
                {
                    frames.back().terms.push_back(std::move(t));
                }
            };

            work.push_back(WorkItem{.node = &root_node, .close_frame = false});
            while (!work.empty())
            {
                const WorkItem item = work.back();
                work.pop_back();

                if (item.close_frame)
                {
                    CompileFrame done = std::move(frames.back());
                    frames.pop_back();
                    if (done.kind == FrameKind::Sum)
                    {
                        emit_term(
                            SumTerm{
                                .terms = std::move(done.terms),
                                .param_begin = done.param_begin,
                                .param_end = next_param_idx
                            }
                        );
                    }
                    else
                    {
                        emit_term(
                            ProductTerm{
                                .terms = std::move(done.terms),
                                .param_begin = done.param_begin,
                                .param_end = next_param_idx
                            }
                        );
                    }
                    continue;
                }

                const Node& cur = *item.node;
                if (const auto* s = std::get_if<SuppliedScalarNode>(&cur.data))
                {
                    const std::size_t leaf_idx = out.leaves.size();
                    const std::size_t p_begin = next_param_idx;
                    const std::size_t p_count = detail::leaf_num_params(s->leaf);
                    out.leaves.push_back(s->leaf);
                    for (std::size_t local = 0; local < p_count; ++local)
                    {
                        out.bindings.push_back(
                            ParameterBinding{.index = next_param_idx++, .leaf_id = leaf_idx, .local_index = local}
                        );
                    }
                    emit_term(
                        SuppliedScalarRef{
                            .leaf_idx = leaf_idx,
                            .slot_at = find_slot_at(out.slots, s->slot),
                            .param_begin = p_begin,
                            .param_count = p_count
                        }
                    );
                    continue;
                }
                if (const auto* a = std::get_if<SuppliedArdNode>(&cur.data))
                {
                    const std::size_t leaf_idx = out.leaves.size();
                    const std::size_t p_begin = next_param_idx;
                    const std::size_t p_count = detail::leaf_num_params(a->leaf);
                    out.leaves.push_back(a->leaf);
                    for (std::size_t local = 0; local < p_count; ++local)
                    {
                        out.bindings.push_back(
                            ParameterBinding{.index = next_param_idx++, .leaf_id = leaf_idx, .local_index = local}
                        );
                    }
                    emit_term(
                        SuppliedArdRef{
                            .leaf_idx = leaf_idx,
                            .slot_at = find_slot_at(out.slots, a->slot),
                            .dims = a->dims,
                            .param_begin = p_begin,
                            .param_count = p_count
                        }
                    );
                    continue;
                }
                if (const auto* c = std::get_if<CoordLeafNode>(&cur.data))
                {
                    const std::size_t leaf_idx = out.leaves.size();
                    const std::size_t p_begin = next_param_idx;
                    const std::size_t p_count = detail::leaf_num_params(c->leaf);
                    out.leaves.push_back(c->leaf);
                    // Constant/White leaves do not access coordinates even in WithPoints trees;
                    // only point-reading coordinate leaves require non-null X matrices at runtime.
                    const CoordMode leaf_mode = detail::leaf_coord_mode(c->leaf);
                    if (leaf_mode != CoordMode::Either)
                    {
                        out.has_coord_points = true;
                    }
                    for (std::size_t local = 0; local < p_count; ++local)
                    {
                        out.bindings.push_back(
                            ParameterBinding{.index = next_param_idx++, .leaf_id = leaf_idx, .local_index = local}
                        );
                    }
                    emit_term(CoordLeafRef{.leaf_idx = leaf_idx, .param_begin = p_begin, .param_count = p_count});
                    continue;
                }

                const bool is_sum = std::holds_alternative<typename Node::Sum>(cur.data);
                frames.push_back(
                    CompileFrame{
                        .kind = is_sum ? FrameKind::Sum : FrameKind::Product,
                        .terms = {},
                        .param_begin = next_param_idx,
                    }
                );
                work.push_back(WorkItem{.node = nullptr, .close_frame = true});

                // Flatten nested binary sums/products into work in reverse order so popping visits left-to-right.
                flatten.push_back(&cur);
                while (!flatten.empty())
                {
                    const Node* f = flatten.back();
                    flatten.pop_back();
                    if (is_sum)
                    {
                        if (const auto* sum = std::get_if<typename Node::Sum>(&f->data))
                        {
                            flatten.push_back(sum->lhs.get());
                            flatten.push_back(sum->rhs.get());
                            continue;
                        }
                    }
                    else
                    {
                        if (const auto* prod = std::get_if<typename Node::Product>(&f->data))
                        {
                            flatten.push_back(prod->lhs.get());
                            flatten.push_back(prod->rhs.get());
                            continue;
                        }
                    }
                    work.push_back(WorkItem{.node = f, .close_frame = false});
                }
            }
            return compiled_root;
        }

        // --- Validation helpers for evaluation ---

        /// @brief Validates that @p store matches this kernel's slot descriptors, square/cross mode, and output shape.
        /// @param[in] store Pre-validated supplied-distance store.
        /// @param[in] expect_square @c true when requiring a square training store, @c false for a cross store.
        /// @param[in] out Output matrix whose row and column counts must match @p store.
        /// @return Empty expected value on success, or a structured @c GprError describing the mismatch.
        [[nodiscard]] std::expected<void, GprError> validate_store(
            const CheckedDistanceStore<T>& store, bool expect_square, MatrixRef<T> out
        ) const
        {
            if (store.impl_->kernel_slots != slots)
            {
                return std::unexpected(
                    GprError::unsupported_kernel_operation("distance store does not match this kernel's slots")
                );
            }
            if (store.impl_->square != expect_square)
            {
                return std::unexpected(
                    GprError::unsupported_kernel_operation(
                        expect_square ? "expected square training distance store"
                                      : "expected rectangular cross distance store"
                    )
                );
            }
            if (store.rows() == 0 || store.cols() == 0)
            {
                return std::unexpected(GprError::empty_input());
            }
            if (out.rows() != store.rows() || out.cols() != store.cols())
            {
                return std::unexpected(GprError::length_mismatch("output matrix shape mismatch"));
            }
            return {};
        }

        /// @brief Checks that coordinate ARD leaves match the feature dimension @p x_dim.
        /// @details Supplied-ARD leaves are intentionally skipped because their dimension counts correspond to their
        /// dedicated @c CheckedDistanceStore slot rather than coordinate columns.
        /// @param[in] x_dim Number of columns in the supplied coordinate matrix.
        /// @return Empty expected value on success, or @c GprError::dimension_mismatch on mismatch.
        [[nodiscard]] std::expected<void, GprError> validate_ard_coord_dims(std::size_t x_dim) const
        {
            for (std::size_t idx = 0; idx < leaves.size(); ++idx)
            {
                // Check coordinate ARD leaves only (SuppliedArd leaves read their own distance slot).
                if (is_coord_leaf_idx(root, idx))
                {
                    const auto& leaf = leaves[idx];
                    std::size_t expected_dim = 0;
                    if (const auto* k = std::get_if<RbfArdKernel>(&leaf))
                    {
                        expected_dim = k->num_params();
                    }
                    else if (const auto* km = std::get_if<MaternArdKernel>(&leaf))
                    {
                        expected_dim = km->num_params();
                    }
                    else if (const auto* krq = std::get_if<RationalQuadraticArdKernel>(&leaf))
                    {
                        expected_dim = krq->num_params() - 1U;
                    }
                    if (expected_dim > 0 && x_dim != expected_dim)
                    {
                        return std::unexpected(GprError::dimension_mismatch(x_dim, expected_dim));
                    }
                }
            }
            return {};
        }

        /// @brief Reports whether leaf index @p target_idx is bound to a coordinate leaf node in @p term.
        /// @param[in] term Compiled AST subtree to search.
        /// @param[in] target_idx Leaf index inside @c Impl::leaves.
        /// @return @c true if @p target_idx appears as a @c CoordLeafRef inside @p term.
        static bool is_coord_leaf_idx(const Term& term, std::size_t target_idx) noexcept
        {
            if (const auto* c = std::get_if<CoordLeafRef>(&term))
            {
                return c->leaf_idx == target_idx;
            }
            if (const auto* s = std::get_if<SumTerm>(&term))
            {
                for (const auto& child : s->terms)
                {
                    if (is_coord_leaf_idx(child, target_idx))
                    {
                        return true;
                    }
                }
            }
            else if (const auto* p = std::get_if<ProductTerm>(&term))
            {
                for (const auto& child : p->terms)
                {
                    if (is_coord_leaf_idx(child, target_idx))
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        /// @brief Validates optional coordinate matrix @p x against kernel requirements and @p expected_rows.
        /// @details When @c has_coord_points is false, @p x may be omitted; if provided, its shape is still checked.
        /// When @c has_coord_points is true, @p x is mandatory, non-empty, row-matched, ARD-dimension checked, and
        /// finite.
        /// @param[in] x Optional coordinate matrix passed to evaluation/differentiation.
        /// @param[in] expected_rows Row (or column) count dictated by the paired distance store.
        /// @return Empty expected value on success, or an error describing missing, mismatched, or non-finite input.
        [[nodiscard]] std::expected<void, GprError> validate_optional_points(
            std::optional<MatrixRef<T>> x, std::size_t expected_rows
        ) const
        {
            if (!has_coord_points)
            {
                if (x.has_value())
                {
                    if (x->rows() == 0 || x->cols() == 0)
                    {
                        return std::unexpected(GprError::empty_input());
                    }
                    if (x->rows() != expected_rows)
                    {
                        return std::unexpected(GprError::length_mismatch("point row count mismatch"));
                    }
                }
                return {};
            }
            if (!x.has_value())
            {
                return std::unexpected(
                    GprError::unsupported_kernel_operation(
                        "WithPoints distance kernel with coordinate leaves requires point matrix"
                    )
                );
            }
            if (x->rows() == 0 || x->cols() == 0)
            {
                return std::unexpected(GprError::empty_input());
            }
            if (x->rows() != expected_rows)
            {
                return std::unexpected(GprError::length_mismatch("point row count mismatch"));
            }
            if (auto ard = validate_ard_coord_dims(x->cols()); !ard.has_value())
            {
                return ard;
            }
            for (std::size_t col = 0; col < x->cols(); ++col)
            {
                for (std::size_t row = 0; row < x->rows(); ++row)
                {
                    if (!std::isfinite((*x)(row, col)))
                    {
                        return std::unexpected(GprError::non_finite_input());
                    }
                }
            }
            return {};
        }

        // --- ARD supplied evaluation / grad / hess ---

        /// @brief Computes ARD-scaled squared distance $s^2$ for entry @c (row, col) using precomputed inverse-squared
        /// weights.
        /// @details Avoids inner-loop floating-point divisions across `(n, n)` packed lower-triangular slabs.
        /// @param[in] packed Packed lower-triangular per-dimension squared distance buffer.
        /// @param[in] n Square matrix order.
        /// @param[in] per_dim Number of packed entries per dimension (@c n*(n+1)/2).
        /// @param[in] weights Precomputed inverse-squared lengthscales $1 / \ell_m^2$.
        /// @param[in] row Matrix row index.
        /// @param[in] col Matrix column index.
        /// @return Total ARD-weighted squared distance $s^2$.
        [[nodiscard]] static T ard_scaled_sq_packed_weights(
            std::span<const T> packed,
            std::size_t n,
            std::size_t per_dim,
            ArdWeightView<T> weights,
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
            for (std::size_t m = 0; m < weights.size(); ++m)
            {
                acc += packed[(m * per_dim) + p] * weights[m];
            }
            return acc;
        }

        /// @brief Accumulates packed ARD lower column slices into `out` and applies `slice_fn`.
        /// @details Requires `uplo` to be `Lower` or `Full` so packed storage is unit-stride contiguous within each
        /// column. Mirrors lower entries to upper triangle when `Full`.
        /// @tparam SliceFn Callable invoked as `slice_fn(col_ptr, len)`.
        /// @param[in] packed Packed lower-triangular per-dimension squared distance buffer.
        /// @param[in] n Square matrix order.
        /// @param[in] per_dim Number of packed entries per dimension.
        /// @param[in] weights Precomputed per-dimension weights.
        /// @param[out] out Destination Gram matrix.
        /// @param[in] uplo Triangle mode (`Lower` or `Full`).
        /// @param[in] slice_fn In-place transformation applied to each populated lower column slice.
        template <typename SliceFn>
        void eval_supplied_ard_lower_cols(
            std::span<const T> packed,
            std::size_t n,
            std::size_t per_dim,
            ArdWeightView<T> weights,
            MatrixMut<T> out,
            Triangle uplo,
            SliceFn&& slice_fn
        ) const
        {
            if (n == 0U || weights.empty())
            {
                return;
            }
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
                detail::scale_set_slice(col_ptr, &packed[col_packed_base], weights[0U], len);
                for (std::size_t m = 1U; m < weights.size(); ++m)
                {
                    detail::axpy_slice(col_ptr, &packed[(m * per_dim) + col_packed_base], weights[m], len);
                }
                out(col, col) = T{0};
                slice_fn(col_ptr, len);
            }
            );
            mirror_lower_if_full(out, uplo);
        }

        /// @brief Evaluates a supplied-ARD leaf over the specified triangle of a square @c CheckedDistanceStore.
        /// @param[in] ref Compiled supplied-ARD leaf reference.
        /// @param[in] store Pre-validated square supplied-distance store containing packed ARD slabs.
        /// @param[out] out Output matrix receiving kernel entries on @p uplo.
        /// @param[in] uplo Triangle mode specifying which entries to populate.
        void eval_supplied_ard_square(
            const SuppliedArdRef& ref, const CheckedDistanceStore<T>& store, MatrixMut<T> out, Triangle uplo
        ) const
        {
            const std::size_t n = store.rows();
            const std::size_t per_dim = (n * (n + 1U)) / 2U;
            const std::span<const T> packed(store.impl_->ard_data[ref.slot_at]);
            const std::span<const double> ls(cached_ls[ref.leaf_idx]);
            const auto& leaf = leaves[ref.leaf_idx];
            if (std::holds_alternative<RbfArdKernel>(leaf))
            {
                with_ard_inv_ell2<T>(
                    ls, T{1}, [&](ArdWeightView<T> weights)
                {
                    if (uplo != Triangle::Upper)
                    {
                        eval_supplied_ard_lower_cols(
                            packed, n, per_dim, weights, out, uplo, [this](T* col_ptr, std::size_t len) noexcept
                        { detail::exp_scaled_nonpos_in_place(col_ptr, len, T{-0.5}, exp_mode); }
                        );
                        return;
                    }
                    visit_triangle<T>(
                        n, uplo, [&](std::size_t row, std::size_t col)
                    {
                        const T s2 = ard_scaled_sq_packed_weights(packed, n, per_dim, weights, row, col);
                        out(row, col) = eval_kernel_exp(-T{0.5} * s2, exp_mode);
                    }
                    );
                }
                );
                return;
            }
            if (const auto* mat = std::get_if<MaternArdKernel>(&leaf))
            {
                const T c = static_cast<T>(matern_scale_sq_d(mat->nu()));
                const MaternNu nu = mat->nu();
                with_ard_inv_ell2<T>(
                    ls, T{1}, [&](ArdWeightView<T> weights)
                {
                    if (uplo != Triangle::Upper)
                    {
                        eval_supplied_ard_lower_cols(
                            packed, n, per_dim, weights, out, uplo, [this, c, nu](T* col_ptr, std::size_t len) noexcept
                        { detail::matern_s2_in_place(col_ptr, len, c, nu, exp_mode); }
                        );
                        return;
                    }
                    visit_triangle<T>(
                        n, uplo, [&](std::size_t row, std::size_t col)
                    {
                        const T u2 = c * ard_scaled_sq_packed_weights(packed, n, per_dim, weights, row, col);
                        const T u = u2 > T{0} ? std::sqrt(u2) : T{0};
                        out(row, col) = eval_matern_u_d(u, nu, exp_mode);
                    }
                    );
                }
                );
                return;
            }
            const auto& rq = std::get<RationalQuadraticArdKernel>(leaf);
            const T alpha = static_cast<T>(rq.alpha());
            const T rq_scale = T{1} / (T{2} * alpha);
            with_ard_inv_ell2<T>(
                ls, rq_scale, [&](ArdWeightView<T> weights)
            {
                if (uplo != Triangle::Upper)
                {
                    eval_supplied_ard_lower_cols(
                        packed, n, per_dim, weights, out, uplo, [this, alpha](T* col_ptr, std::size_t len) noexcept
                    { transform_rq_slice_in_place(col_ptr, len, alpha, exp_mode); }
                    );
                    return;
                }
                visit_triangle<T>(
                    n, uplo, [&](std::size_t row, std::size_t col)
                {
                    const T w = ard_scaled_sq_packed_weights(packed, n, per_dim, weights, row, col);
                    out(row, col) = eval_kernel_exp(-alpha * std::log1p(w), exp_mode);
                }
                );
            }
            );
        }

        /// @brief Evaluates a supplied-ARD leaf over a rectangular cross @c CheckedDistanceStore.
        /// @param[in] ref Compiled supplied-ARD leaf reference.
        /// @param[in] store Pre-validated cross supplied-distance store containing per-dimension blocks.
        /// @param[out] out Output matrix receiving cross-covariance values.
        void eval_supplied_ard_cross(
            const SuppliedArdRef& ref, const CheckedDistanceStore<T>& store, MatrixMut<T> out
        ) const
        {
            const std::size_t n_rows = store.rows();
            const std::size_t n_cols = store.cols();
            if (n_rows == 0U || n_cols == 0U)
            {
                return;
            }
            const std::size_t block_len = n_rows * n_cols;
            const std::span<const T> blocks(store.impl_->ard_data[ref.slot_at]);
            const std::span<const double> ls(cached_ls[ref.leaf_idx]);
            const auto& leaf = leaves[ref.leaf_idx];
            auto run_cross = [&](ArdWeightView<T> weights, auto&& slice_fn)
            {
                const auto n_cols_s = static_cast<std::ptrdiff_t>(n_cols);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (block_len >= 65536U)
#endif
                for (std::ptrdiff_t col_s = 0; col_s < n_cols_s; ++col_s)
                {
                    const auto col = static_cast<std::size_t>(col_s);
                    const std::size_t col_offset = col * n_rows;
                    T* col_ptr = &out(0U, col);
                    detail::scale_set_slice(col_ptr, &blocks[col_offset], weights[0U], n_rows);
                    for (std::size_t m = 1U; m < weights.size(); ++m)
                    {
                        detail::axpy_slice(col_ptr, &blocks[(m * block_len) + col_offset], weights[m], n_rows);
                    }
                    slice_fn(col_ptr, n_rows);
                }
            };
            if (std::holds_alternative<RbfArdKernel>(leaf))
            {
                with_ard_inv_ell2<T>(
                    ls, T{1}, [&](ArdWeightView<T> weights)
                {
                    run_cross(weights, [this](T* col_ptr, std::size_t len) noexcept {
                        detail::exp_scaled_nonpos_in_place(col_ptr, len, T{-0.5}, exp_mode);
                    });
                }
                );
                return;
            }
            if (const auto* mat = std::get_if<MaternArdKernel>(&leaf))
            {
                const T c = static_cast<T>(matern_scale_sq_d(mat->nu()));
                const MaternNu nu = mat->nu();
                with_ard_inv_ell2<T>(
                    ls, T{1}, [&](ArdWeightView<T> weights)
                {
                    run_cross(weights, [this, c, nu](T* col_ptr, std::size_t len) noexcept {
                        detail::matern_s2_in_place(col_ptr, len, c, nu, exp_mode);
                    });
                }
                );
                return;
            }
            const auto& rq = std::get<RationalQuadraticArdKernel>(leaf);
            const T alpha = static_cast<T>(rq.alpha());
            const T rq_scale = T{1} / (T{2} * alpha);
            with_ard_inv_ell2<T>(
                ls, rq_scale, [&](ArdWeightView<T> weights)
            {
                run_cross(weights, [this, alpha](T* col_ptr, std::size_t len) noexcept {
                    transform_rq_slice_in_place(col_ptr, len, alpha, exp_mode);
                });
            }
            );
        }

        /// @brief Computes the first log-parameter derivative of a supplied-ARD leaf over @p uplo.
        /// @details Differentiates with respect to $\theta_p = \log \ell_p$ (or $\log \alpha$ for
        /// @c RationalQuadraticArdKernel when @p local_idx equals the lengthscale count).
        /// @param[in] ref Compiled supplied-ARD leaf reference.
        /// @param[in] store Pre-validated square supplied-distance store.
        /// @param[out] d_k Output matrix receiving the log-theta derivative on @p uplo.
        /// @param[in] local_idx Leaf-local parameter index.
        /// @param[in] uplo Triangle mode specifying which entries to write.
        void eval_supplied_ard_grad(
            const SuppliedArdRef& ref,
            const CheckedDistanceStore<T>& store,
            MatrixMut<T> d_k,
            std::size_t local_idx,
            Triangle uplo
        ) const
        {
            const std::size_t n = store.rows();
            const std::size_t per_dim = (n * (n + 1U)) / 2U;
            const std::span<const T> packed(store.impl_->ard_data[ref.slot_at]);
            const std::span<const double> ls(cached_ls[ref.leaf_idx]);
            const auto& leaf = leaves[ref.leaf_idx];
            if (std::holds_alternative<RbfArdKernel>(leaf))
            {
                // For RBF ARD k = exp(-0.5 * sum_m s_m^2), d k / d(log l_p) = s_p^2 * d(exp(-0.5 * s^2)).
                with_ard_inv_ell2<T>(
                    ls, T{1}, [&](ArdWeightView<T> weights)
                {
                    const T inv_l2 = weights[local_idx];
                    visit_triangle<T>(
                        n, uplo, [&](std::size_t row, std::size_t col)
                    {
                        const std::size_t p = packed_lower_index(n, row, col);
                        const T s2 = ard_scaled_sq_packed_weights(packed, n, per_dim, weights, row, col);
                        const T sp2 = packed[(local_idx * per_dim) + p] * inv_l2;
                        d_k(row, col) = sp2 * eval_kernel_exp_jet(-T{0.5} * s2, exp_mode).d1;
                    }
                    );
                }
                );
                return;
            }
            if (const auto* mat = std::get_if<MaternArdKernel>(&leaf))
            {
                // Write u = sqrt(c * s^2) and w_p = (c * s_p^2) / u^2; then d u / d(log l_p) = -u * w_p,
                // so d k / d(log l_p) = g1(u) * w_p where g1(u) = -u * dk/du absorbs the 1/u singularity.
                const T c = static_cast<T>(matern_scale_sq_d(mat->nu()));
                const MaternNu nu = mat->nu();
                with_ard_inv_ell2<T>(
                    ls, T{1}, [&](ArdWeightView<T> weights)
                {
                    const T inv_l2 = weights[local_idx];
                    visit_triangle<T>(
                        n, uplo, [&](std::size_t row, std::size_t col)
                    {
                        const std::size_t p = packed_lower_index(n, row, col);
                        const T u2 = c * ard_scaled_sq_packed_weights(packed, n, per_dim, weights, row, col);
                        if (u2 <= T{0})
                        {
                            d_k(row, col) = T{0};
                            return;
                        }
                        const T u = std::sqrt(u2);
                        const T wp = (c * packed[(local_idx * per_dim) + p] * inv_l2) / u2;
                        d_k(row, col) = eval_matern_grad_g1_d(u, nu, exp_mode) * wp;
                    }
                    );
                }
                );
                return;
            }
            const auto& rq = std::get<RationalQuadraticArdKernel>(leaf);
            const T alpha = static_cast<T>(rq.alpha());
            const std::size_t alpha_idx = ls.size();
            with_ard_inv_ell2<T>(
                ls, T{1}, [&](ArdWeightView<T> weights)
            {
                const T inv_l2 = (local_idx < alpha_idx) ? weights[local_idx] : T{0};
                visit_triangle<T>(
                    n, uplo, [&](std::size_t row, std::size_t col)
                {
                    const std::size_t p = packed_lower_index(n, row, col);
                    const T s2 = ard_scaled_sq_packed_weights(packed, n, per_dim, weights, row, col);
                    const T w = s2 / (T{2} * alpha);
                    const T b = T{1} + w;
                    const T d1_exp = eval_kernel_exp_jet(-alpha * std::log1p(w), exp_mode).d1;
                    if (local_idx < alpha_idx)
                    {
                        // d(-alpha * log(1 + w)) / d(log l_p) = s_p^2 / (1 + w).
                        const T sp2 = packed[(local_idx * per_dim) + p] * inv_l2;
                        d_k(row, col) = (sp2 / b) * d1_exp;
                    }
                    else
                    {
                        // d(-alpha * log(1 + w)) / d(log alpha) = alpha * (w / (1 + w) - log(1 + w)).
                        d_k(row, col) = alpha * ((w / b) - std::log1p(w)) * d1_exp;
                    }
                }
                );
            }
            );
        }

        /// @brief Computes the second log-parameter derivative of a supplied-ARD leaf over @p uplo.
        /// @details Evaluates exact mixed or diagonal Hessians with respect to leaf-local log-parameters (@p i, @p j)
        /// using the exponential derivative jet @c ExpJet for consistent @c KernelExp precision.
        /// @param[in] ref Compiled supplied-ARD leaf reference.
        /// @param[in] store Pre-validated square supplied-distance store.
        /// @param[out] d2_k Output matrix receiving $\partial^2 K / \partial \theta_i \partial \theta_j$ on @p uplo.
        /// @param[in] i First leaf-local parameter index.
        /// @param[in] j Second leaf-local parameter index.
        /// @param[in] uplo Triangle mode specifying which entries to write.
        void eval_supplied_ard_hess(
            const SuppliedArdRef& ref,
            const CheckedDistanceStore<T>& store,
            MatrixMut<T> d2_k,
            std::size_t i,
            std::size_t j,
            Triangle uplo
        ) const
        {
            const std::size_t n = store.rows();
            const std::size_t per_dim = (n * (n + 1U)) / 2U;
            const std::span<const T> packed(store.impl_->ard_data[ref.slot_at]);
            const std::span<const double> ls(cached_ls[ref.leaf_idx]);
            const auto& leaf = leaves[ref.leaf_idx];
            if (std::holds_alternative<RbfArdKernel>(leaf))
            {
                with_ard_inv_ell2<T>(
                    ls, T{1}, [&](ArdWeightView<T> weights)
                {
                    const T inv_li2 = weights[i];
                    const T inv_lj2 = weights[j];
                    visit_triangle<T>(
                        n, uplo, [&](std::size_t row, std::size_t col)
                    {
                        const std::size_t p = packed_lower_index(n, row, col);
                        const T s2 = ard_scaled_sq_packed_weights(packed, n, per_dim, weights, row, col);
                        const ExpJet<T> jet = eval_kernel_exp_jet(-T{0.5} * s2, exp_mode);
                        const T si2 = packed[(i * per_dim) + p] * inv_li2;
                        if (i == j)
                        {
                            // Diagonal second derivative includes -2*s_i^2*jet.d1 from d(s_i^2)/d(log l_i) = -2*s_i^2.
                            d2_k(row, col) = (si2 * si2 * jet.d2) - (T{2} * si2 * jet.d1);
                        }
                        else
                        {
                            const T sj2 = packed[(j * per_dim) + p] * inv_lj2;
                            d2_k(row, col) = si2 * sj2 * jet.d2;
                        }
                    }
                    );
                }
                );
                return;
            }
            if (const auto* mat = std::get_if<MaternArdKernel>(&leaf))
            {
                const T c = static_cast<T>(matern_scale_sq_d(mat->nu()));
                const MaternNu nu = mat->nu();
                with_ard_inv_ell2<T>(
                    ls, T{1}, [&](ArdWeightView<T> weights)
                {
                    const T inv_li2 = weights[i];
                    const T inv_lj2 = weights[j];
                    visit_triangle<T>(
                        n, uplo, [&](std::size_t row, std::size_t col)
                    {
                        const std::size_t p = packed_lower_index(n, row, col);
                        const T u2 = c * ard_scaled_sq_packed_weights(packed, n, per_dim, weights, row, col);
                        if (u2 <= T{0})
                        {
                            d2_k(row, col) = T{0};
                            return;
                        }
                        const T u = std::sqrt(u2);
                        const T wi = (c * packed[(i * per_dim) + p] * inv_li2) / u2;
                        const T wj = (c * packed[(j * per_dim) + p] * inv_lj2) / u2;
                        // Mixed Matérn ARD Hessian splits into cross-dimension term plus diagonal -2*g1(u)*w_i.
                        T val = eval_matern_ard_cross_d(u, nu, exp_mode) * wi * wj;
                        if (i == j)
                        {
                            val -= T{2} * eval_matern_grad_g1_d(u, nu, exp_mode) * wi;
                        }
                        d2_k(row, col) = val;
                    }
                    );
                }
                );
                return;
            }
            const auto& rq = std::get<RationalQuadraticArdKernel>(leaf);
            const T alpha = static_cast<T>(rq.alpha());
            const std::size_t alpha_idx = ls.size();
            with_ard_inv_ell2<T>(
                ls, T{1}, [&](ArdWeightView<T> weights)
            {
                visit_triangle<T>(
                    n, uplo, [&](std::size_t row, std::size_t col)
                {
                    const std::size_t p = packed_lower_index(n, row, col);
                    const T s2 = ard_scaled_sq_packed_weights(packed, n, per_dim, weights, row, col);
                    const T w = s2 / (T{2} * alpha);
                    const T b = T{1} + w;
                    const ExpJet<T> jet = eval_kernel_exp_jet(-alpha * std::log1p(w), exp_mode);
                    const T g_alpha = alpha * ((w / b) - std::log1p(w));
                    if (i == alpha_idx && j == alpha_idx)
                    {
                        // d^2 f / d(log alpha)^2 = g_alpha + alpha * w^2 / b^2, chained through exp_jet.
                        d2_k(row, col) =
                            ((g_alpha * g_alpha) * jet.d2) + ((g_alpha + ((alpha * w * w) / (b * b))) * jet.d1);
                    }
                    else if (i == alpha_idx || j == alpha_idx)
                    {
                        // Mixed (log alpha, log l_k) derivative: df/d(log l_k) * d_alpha(f) + w*df/(b*b).
                        const std::size_t ls_idx = (i == alpha_idx) ? j : i;
                        const T s_ls2 = packed[(ls_idx * per_dim) + p] * weights[ls_idx];
                        d2_k(row, col) = (s_ls2 / b) * ((g_alpha * jet.d2) + ((w / b) * jet.d1));
                    }
                    else
                    {
                        // Lengthscale-lengthscale Hessian mixes outer chain, 1/alpha term, and diagonal -2*s_i^2/b.
                        const T si2 = packed[(i * per_dim) + p] * weights[i];
                        const T sj2 = packed[(j * per_dim) + p] * weights[j];
                        const T sq_term = (si2 * sj2) / (b * b);
                        T val = (sq_term * jet.d2) + ((sq_term / alpha) * jet.d1);
                        if (i == j)
                        {
                            val -= ((T{2} * si2) / b) * jet.d1;
                        }
                        d2_k(row, col) = val;
                    }
                }
                );
            }
            );
        }

        // --- Recursive tree evaluation ---

        /// @brief Verifies that @p nested holds at least @p level_idx + 1 buffers sized at least @p rows by @p cols.
        /// @param[in,out] nested Preallocated nested evaluation scratch pool.
        /// @param[in] level_idx Zero-based nesting level requested.
        /// @param[in] rows Required row count.
        /// @param[in] cols Required column count.
        /// @return Empty expected value on success, or @c GprError::workspace_too_small if capacity is insufficient.
        [[nodiscard]] std::expected<void, GprError> ensure_nested(
            NestedBuffers<T>& nested, std::size_t level_idx, std::size_t rows, std::size_t cols
        ) const
        {
            if (nested.depth() <= level_idx || nested.rows() < rows || nested.cols() < cols)
            {
                return std::unexpected(GprError::workspace_too_small());
            }
            return {};
        }

        /// @brief Writes WorkspaceTooSmall into @p err out-of-line so recursive frames construct no @c GprError.
        /// @param[out] err Destination error slot.
        /// @return Always false.
        [[nodiscard]] static CXXGPR_NOINLINE bool fail_workspace_too_small(std::optional<GprError>& err) noexcept
        {
            err = GprError::workspace_too_small();
            return false;
        }

        /// @brief Writes UnsupportedKernelOperation for an empty composite into @p err out-of-line.
        /// @param[out] err Destination error slot.
        /// @return Always false.
        [[nodiscard]] static CXXGPR_NOINLINE bool fail_empty_composite(std::optional<GprError>& err)
        {
            err = GprError::unsupported_kernel_operation("composite kernel has no terms");
            return false;
        }

        /// @brief Writes IndexOutOfRange for a parameter index into @p err out-of-line.
        /// @param[out] err Destination error slot.
        /// @return Always false.
        [[nodiscard]] static CXXGPR_NOINLINE bool fail_param_out_of_range(std::optional<GprError>& err)
        {
            err = GprError::index_out_of_range("kernel parameter index out of range");
            return false;
        }

        /// @brief Checks whether @p nested can serve @p level_idx without constructing @c std::expected.
        /// @param[in,out] nested Caller-supplied nested buffer pool.
        /// @param[in] level_idx Required zero-based buffer level index.
        /// @param[in] rows Required row capacity.
        /// @param[in] cols Required column capacity.
        /// @param[out] err Populated with WorkspaceTooSmall on failure.
        /// @return True when @p nested suffices, false otherwise.
        [[nodiscard]] bool ensure_nested_ok(
            NestedBuffers<T>& nested,
            std::size_t level_idx,
            std::size_t rows,
            std::size_t cols,
            std::optional<GprError>& err
        ) const noexcept
        {
            if (nested.depth() <= level_idx || nested.rows() < rows || nested.cols() < cols)
            {
                return fail_workspace_too_small(err);
            }
            return true;
        }

        /// @brief Captures a `std::expected<void, GprError>` into @p err out-of-line.
        /// @param[in] res Expected result to inspect.
        /// @param[out] err Populated with error on failure.
        /// @return True on success, false after writing @p err.
        [[nodiscard]] static CXXGPR_NOINLINE bool capture_expected(
            std::expected<void, GprError>&& res, std::optional<GprError>& err
        ) noexcept
        {
            if (res.has_value())
            {
                return true;
            }
            err = std::move(res).error();
            return false;
        }

        /// @brief Evaluates one square leaf term out-of-line so recursive frames stay free of `std::expected`.
        /// @param[in] term Leaf AST node.
        /// @param[in] store Pre-validated square supplied-distance store.
        /// @param[in] x Optional coordinate matrix.
        /// @param[out] out Destination matrix view.
        /// @param[in] uplo Active matrix triangle.
        /// @param[in,out] nested Multi-level workspace for @c CustomKernel distance buffers.
        /// @param[in] level_idx Current nesting depth inside @p nested.
        /// @param[out] err Populated with the error on failure.
        /// @return True on success, false after writing @p err.
        [[nodiscard]] CXXGPR_NOINLINE bool eval_square_leaf_step(
            const Term& term,
            const CheckedDistanceStore<T>& store,
            std::optional<MatrixRef<T>> x,
            MatrixMut<T> out,
            Triangle uplo,
            NestedBuffers<T>& nested,
            std::size_t level_idx,
            std::optional<GprError>& err
        ) const
        {
            if (const auto* s = std::get_if<SuppliedScalarRef>(&term))
            {
                const MatrixRef<T> dist(store.impl_->scalar_tables[s->slot_at].data(), store.rows(), store.cols());
                return capture_expected(detail::leaf_apply_dist(leaves[s->leaf_idx], dist, out, uplo, exp_mode), err);
            }
            if (const auto* a = std::get_if<SuppliedArdRef>(&term))
            {
                eval_supplied_ard_square(*a, store, out, uplo);
                return true;
            }
            const auto& c = std::get<CoordLeafRef>(term);
            if (x.has_value())
            {
                const KernelLeaf& kl = leaves[c.leaf_idx];
                if (const auto* custom = std::get_if<CustomKernel>(&kl))
                {
                    if (!ensure_nested_ok(nested, level_idx, x->rows(), x->rows(), err))
                    {
                        return false;
                    }
                    auto dist_buf = nested.level(level_idx, x->rows(), x->rows());
                    fill_square_sq_dist(*x, dist_buf);
                    return capture_expected(custom->template term<T>().apply(dist_buf.as_ref(), out, uplo), err);
                }
                return capture_expected(
                    detail::leaf_apply_points(kl, *x, out, uplo, exp_mode, cached_ls[c.leaf_idx]), err
                );
            }
            if (const auto* ck = std::get_if<ConstantKernel>(&leaves[c.leaf_idx]))
            {
                const T val = static_cast<T>(ck->constant());
                visit_triangle<T>(store.rows(), uplo, [&](std::size_t row, std::size_t col) { out(row, col) = val; });
                return true;
            }
            const auto& wk = std::get<WhiteKernel>(leaves[c.leaf_idx]);
            const T var = static_cast<T>(wk.variance());
            visit_triangle<T>(store.rows(), uplo, [&](std::size_t row, std::size_t col) {
                out(row, col) = (row == col) ? var : T{0};
            });
            return true;
        }

        /// @brief Invariant context for `eval_term_square_rec` to keep recursive stack frames minimal.
        struct SquareEvalCtx
        {
            /// @brief Pre-validated square supplied-distance store.
            const CheckedDistanceStore<T>& store;
            /// @brief Optional coordinate matrix.
            std::optional<MatrixRef<T>> x;
            /// @brief Active matrix triangle.
            Triangle uplo;
            /// @brief Multi-level workspace for nested composite nodes.
            NestedBuffers<T>& nested;
            /// @brief Out-of-line error slot.
            std::optional<GprError>& err;
        };

        /// @brief Recursive body of `eval_term_square` with minimal per-level stack frame size.
        /// @param[in] term Compiled AST node to evaluate.
        /// @param[out] out Destination matrix populated on @p ctx.uplo.
        /// @param[in,out] scratch Leaf scratch matrix.
        /// @param[in] level_idx Current recursion depth into @p ctx.nested.
        /// @param[in] ctx Invariant square evaluation context.
        /// @return True on success, false after writing @p ctx.err.
        [[nodiscard]] bool eval_term_square_rec(
            const Term& term,
            const MatrixMut<T>& out,
            const MatrixMut<T>& scratch,
            std::size_t level_idx,
            const SquareEvalCtx& ctx
        ) const
        {
            if (is_leaf_term(term))
            {
                return eval_square_leaf_step(term, ctx.store, ctx.x, out, ctx.uplo, ctx.nested, level_idx, ctx.err);
            }
            const bool is_sum = std::holds_alternative<SumTerm>(term);
            const auto& terms = is_sum ? std::get<SumTerm>(term).terms : std::get<ProductTerm>(term).terms;
            if (terms.empty())
            {
                return fail_empty_composite(ctx.err);
            }
            if (!eval_term_square_rec(terms[0], out, scratch, level_idx, ctx))
            {
                return false;
            }
            for (std::size_t k = 1; k < terms.size(); ++k)
            {
                if (is_leaf_term(terms[k]))
                {
                    if (!eval_term_square_rec(terms[k], scratch, out, level_idx, ctx))
                    {
                        return false;
                    }
                    if (is_sum)
                    {
                        add_into(out, scratch.as_ref(), ctx.uplo);
                    }
                    else
                    {
                        mul_into(out, scratch.as_ref(), ctx.uplo);
                    }
                }
                else
                {
                    if (!ensure_nested_ok(ctx.nested, level_idx, out.rows(), out.cols(), ctx.err))
                    {
                        return false;
                    }
                    const auto buf = ctx.nested.level(level_idx, out.rows(), out.cols());
                    if (!eval_term_square_rec(terms[k], buf, scratch, level_idx + 1U, ctx))
                    {
                        return false;
                    }
                    if (is_sum)
                    {
                        add_into(out, buf.as_ref(), ctx.uplo);
                    }
                    else
                    {
                        mul_into(out, buf.as_ref(), ctx.uplo);
                    }
                }
            }
            return true;
        }

        /// @brief Recursively evaluates compiled term @p term on a square @c CheckedDistanceStore over @p uplo.
        /// @param[in] term Compiled AST node to evaluate.
        /// @param[in] store Pre-validated square supplied-distance store.
        /// @param[in] x Optional coordinate matrix.
        /// @param[out] out Destination matrix populated on @p uplo.
        /// @param[in] uplo Triangle mode specifying which entries to compute.
        /// @param[in,out] scratch Leaf scratch matrix.
        /// @param[in,out] nested Multi-level workspace for nested composite nodes.
        /// @param[in] level_idx Current recursion depth into @p nested.
        /// @return Empty expected value on success, or a structured @c GprError on evaluation failure.
        [[nodiscard]] std::expected<void, GprError> eval_term_square(
            const Term& term,
            const CheckedDistanceStore<T>& store,
            std::optional<MatrixRef<T>> x,
            MatrixMut<T> out,
            Triangle uplo,
            MatrixMut<T> scratch,
            NestedBuffers<T>& nested,
            std::size_t level_idx
        ) const
        {
            std::optional<GprError> err;
            const SquareEvalCtx ctx{
                .store = store,
                .x = x,
                .uplo = uplo,
                .nested = nested,
                .err = err,
            };
            if (!eval_term_square_rec(term, out, scratch, level_idx, ctx))
            {
                return std::unexpected(
                    err.has_value() ? std::move(*err)
                                    : GprError::unsupported_kernel_operation("kernel evaluation failed")
                );
            }
            return {};
        }

        /// @brief Evaluates one cross leaf term out-of-line so recursive frames stay free of `std::expected`.
        /// @param[in] term Leaf AST node.
        /// @param[in] store Pre-validated cross supplied-distance store.
        /// @param[in] x1 Optional left coordinate matrix.
        /// @param[in] x2 Optional right coordinate matrix.
        /// @param[out] out Destination cross-covariance matrix view.
        /// @param[in,out] nested Multi-level workspace for @c CustomKernel distances.
        /// @param[in] level_idx Current recursion depth into @p nested.
        /// @param[out] err Populated with the error on failure.
        /// @return True on success, false after writing @p err.
        [[nodiscard]] CXXGPR_NOINLINE bool eval_cross_leaf_step(
            const Term& term,
            const CheckedDistanceStore<T>& store,
            std::optional<MatrixRef<T>> x1,
            std::optional<MatrixRef<T>> x2,
            MatrixMut<T> out,
            NestedBuffers<T>& nested,
            std::size_t level_idx,
            std::optional<GprError>& err
        ) const
        {
            if (const auto* s = std::get_if<SuppliedScalarRef>(&term))
            {
                const MatrixRef<T> dist(store.impl_->scalar_tables[s->slot_at].data(), store.rows(), store.cols());
                return capture_expected(detail::leaf_apply_cross_dist(leaves[s->leaf_idx], dist, out, exp_mode), err);
            }
            if (const auto* a = std::get_if<SuppliedArdRef>(&term))
            {
                eval_supplied_ard_cross(*a, store, out);
                return true;
            }
            const auto& c = std::get<CoordLeafRef>(term);
            if (x1.has_value() && x2.has_value())
            {
                const KernelLeaf& kl = leaves[c.leaf_idx];
                if (const auto* custom = std::get_if<CustomKernel>(&kl))
                {
                    if (!ensure_nested_ok(nested, level_idx, x1->rows(), x2->rows(), err))
                    {
                        return false;
                    }
                    auto dist_buf = nested.level(level_idx, x1->rows(), x2->rows());
                    fill_cross_sq_dist(*x1, *x2, dist_buf);
                    return capture_expected(custom->template term<T>().apply_cross(dist_buf.as_ref(), out), err);
                }
                return capture_expected(
                    detail::leaf_apply_cross_points(kl, *x1, *x2, out, exp_mode, cached_ls[c.leaf_idx]), err
                );
            }
            const T val = std::holds_alternative<ConstantKernel>(leaves[c.leaf_idx])
                ? static_cast<T>(std::get<ConstantKernel>(leaves[c.leaf_idx]).constant())
                : T{0};
            for (std::size_t col = 0; col < out.cols(); ++col)
            {
                for (std::size_t row = 0; row < out.rows(); ++row)
                {
                    out(row, col) = val;
                }
            }
            return true;
        }

        /// @brief Invariant context for `eval_term_cross_rec` to keep recursive stack frames minimal.
        struct CrossEvalCtx
        {
            /// @brief Pre-validated cross supplied-distance store.
            const CheckedDistanceStore<T>& store;
            /// @brief Optional left coordinate matrix.
            std::optional<MatrixRef<T>> x1;
            /// @brief Optional right coordinate matrix.
            std::optional<MatrixRef<T>> x2;
            /// @brief Multi-level workspace for nested composite nodes.
            NestedBuffers<T>& nested;
            /// @brief Out-of-line error slot.
            std::optional<GprError>& err;
        };

        /// @brief Recursive body of `eval_term_cross` with minimal per-level stack frame size.
        /// @param[in] term Compiled AST node to evaluate.
        /// @param[out] out Destination cross-covariance matrix view.
        /// @param[in,out] scratch Leaf scratch matrix.
        /// @param[in] level_idx Current recursion depth into @p ctx.nested.
        /// @param[in] ctx Invariant cross evaluation context.
        /// @return True on success, false after writing @p ctx.err.
        [[nodiscard]] bool eval_term_cross_rec(
            const Term& term,
            const MatrixMut<T>& out,
            const MatrixMut<T>& scratch,
            std::size_t level_idx,
            const CrossEvalCtx& ctx
        ) const
        {
            if (is_leaf_term(term))
            {
                return eval_cross_leaf_step(term, ctx.store, ctx.x1, ctx.x2, out, ctx.nested, level_idx, ctx.err);
            }
            const bool is_sum = std::holds_alternative<SumTerm>(term);
            const auto& terms = is_sum ? std::get<SumTerm>(term).terms : std::get<ProductTerm>(term).terms;
            if (terms.empty())
            {
                return fail_empty_composite(ctx.err);
            }
            if (!eval_term_cross_rec(terms[0], out, scratch, level_idx, ctx))
            {
                return false;
            }
            for (std::size_t k = 1; k < terms.size(); ++k)
            {
                if (is_leaf_term(terms[k]))
                {
                    if (!eval_term_cross_rec(terms[k], scratch, out, level_idx, ctx))
                    {
                        return false;
                    }
                    if (is_sum)
                    {
                        add_into(out, scratch.as_ref(), Triangle::Full);
                    }
                    else
                    {
                        mul_into(out, scratch.as_ref(), Triangle::Full);
                    }
                }
                else
                {
                    if (!ensure_nested_ok(ctx.nested, level_idx, out.rows(), out.cols(), ctx.err))
                    {
                        return false;
                    }
                    const auto buf = ctx.nested.level(level_idx, out.rows(), out.cols());
                    if (!eval_term_cross_rec(terms[k], buf, scratch, level_idx + 1U, ctx))
                    {
                        return false;
                    }
                    if (is_sum)
                    {
                        add_into(out, buf.as_ref(), Triangle::Full);
                    }
                    else
                    {
                        mul_into(out, buf.as_ref(), Triangle::Full);
                    }
                }
            }
            return true;
        }

        /// @brief Recursively evaluates compiled term @p term on a rectangular cross @c CheckedDistanceStore.
        /// @param[in] term Compiled AST node to evaluate.
        /// @param[in] store Pre-validated cross supplied-distance store.
        /// @param[in] x1 Optional left coordinate matrix.
        /// @param[in] x2 Optional right coordinate matrix.
        /// @param[out] out Destination cross-covariance matrix view.
        /// @param[in,out] scratch Leaf scratch matrix.
        /// @param[in,out] nested Multi-level workspace.
        /// @param[in] level_idx Current recursion depth into @p nested.
        /// @return Empty expected value on success, or a structured @c GprError on evaluation failure.
        [[nodiscard]] std::expected<void, GprError> eval_term_cross(
            const Term& term,
            const CheckedDistanceStore<T>& store,
            std::optional<MatrixRef<T>> x1,
            std::optional<MatrixRef<T>> x2,
            MatrixMut<T> out,
            MatrixMut<T> scratch,
            NestedBuffers<T>& nested,
            std::size_t level_idx
        ) const
        {
            std::optional<GprError> err;
            const CrossEvalCtx ctx{
                .store = store,
                .x1 = x1,
                .x2 = x2,
                .nested = nested,
                .err = err,
            };
            if (!eval_term_cross_rec(term, out, scratch, level_idx, ctx))
            {
                return std::unexpected(
                    err.has_value() ? std::move(*err)
                                    : GprError::unsupported_kernel_operation("kernel evaluation failed")
                );
            }
            return {};
        }

        /// @brief Evaluates one distance-kernel diagonal leaf out-of-line so recursive frames stay minimal.
        /// @param[in] term Leaf AST node.
        /// @param[in] x Optional coordinate matrix.
        /// @param[out] out Output diagonal span.
        /// @param[out] err Populated with the error on failure.
        /// @return True on success, false after writing @p err.
        [[nodiscard]] CXXGPR_NOINLINE bool fill_diag_leaf_step(
            const Term& term, const std::optional<MatrixRef<T>>& x, std::span<T> out, std::optional<GprError>& err
        ) const
        {
            if (std::holds_alternative<SuppliedScalarRef>(term) || std::holds_alternative<SuppliedArdRef>(term))
            {
                std::fill(out.begin(), out.end(), T{1});
                return true;
            }
            const auto& c = std::get<CoordLeafRef>(term);
            if (x.has_value())
            {
                return capture_expected(
                    detail::leaf_fill_diag(leaves[c.leaf_idx], *x, out, cached_ls[c.leaf_idx]), err
                );
            }
            if (const auto* ck = std::get_if<ConstantKernel>(&leaves[c.leaf_idx]))
            {
                std::fill(out.begin(), out.end(), static_cast<T>(ck->constant()));
                return true;
            }
            const auto& wk = std::get<WhiteKernel>(leaves[c.leaf_idx]);
            std::fill(out.begin(), out.end(), static_cast<T>(wk.variance()));
            return true;
        }

        /// @brief Invariant context for `eval_fill_diag_rec` to keep recursive stack frames minimal.
        struct FillDiagCtx
        {
            /// @brief Optional coordinate matrix for coordinate-based leaves.
            std::optional<MatrixRef<T>> x;
            /// @brief Flat scratch vector partitioned by `level_idx * out.size()`.
            std::vector<T>& scratch_vec;
            /// @brief Out-of-line error slot.
            std::optional<GprError>& err;
        };

        /// @brief Recursive body of `eval_fill_diag` with minimal per-level stack frame size.
        /// @param[in] term Compiled AST node to evaluate along the diagonal.
        /// @param[out] out Output diagonal span of length `n`.
        /// @param[in] level_idx Current recursion depth into `ctx.scratch_vec`.
        /// @param[in] ctx Invariant diagonal evaluation context.
        /// @return True on success, false after writing `ctx.err`.
        [[nodiscard]] bool eval_fill_diag_rec(
            const Term& term, std::span<T> out, std::size_t level_idx, const FillDiagCtx& ctx
        ) const
        {
            if (is_leaf_term(term))
            {
                return fill_diag_leaf_step(term, ctx.x, out, ctx.err);
            }
            const bool is_sum = std::holds_alternative<SumTerm>(term);
            const auto& terms = is_sum ? std::get<SumTerm>(term).terms : std::get<ProductTerm>(term).terms;
            if (terms.empty())
            {
                return fail_empty_composite(ctx.err);
            }
            if (!eval_fill_diag_rec(terms[0], out, level_idx, ctx))
            {
                return false;
            }
            if (terms.size() == 1U)
            {
                return true;
            }
            const std::size_t slice_offset = level_idx * out.size();
            if (ctx.scratch_vec.size() < slice_offset + out.size())
            {
                ctx.scratch_vec.resize(slice_offset + out.size());
            }
            for (std::size_t k = 1; k < terms.size(); ++k)
            {
                const std::span<T> sub(ctx.scratch_vec.data() + slice_offset, out.size());
                if (!eval_fill_diag_rec(terms[k], sub, level_idx + 1U, ctx))
                {
                    return false;
                }
                for (std::size_t idx = 0; idx < out.size(); ++idx)
                {
                    if (is_sum)
                    {
                        out[idx] += sub[idx];
                    }
                    else
                    {
                        out[idx] *= sub[idx];
                    }
                }
            }
            return true;
        }

        /// @brief Recursively evaluates the diagonal $k(x_i, x_i)$ of compiled term @p term into @p out.
        /// @param[in] term Compiled AST node.
        /// @param[in] n Number of diagonal entries.
        /// @param[in] x Optional coordinate matrix for coordinate-based leaves.
        /// @param[out] out Output diagonal span of length @p n.
        /// @param[in,out] scratch_vec Flat scratch vector partitioned by @p level_idx * @p out.size().
        /// @param[in] level_idx Current recursion depth into @p scratch_vec.
        /// @return Empty expected value on success, or a structured @c GprError on evaluation failure.
        [[nodiscard]] std::expected<void, GprError> eval_fill_diag(
            const Term& term,
            std::size_t /*n*/,
            std::optional<MatrixRef<T>> x,
            std::span<T> out,
            std::vector<T>& scratch_vec,
            std::size_t level_idx
        ) const
        {
            std::optional<GprError> err;
            const FillDiagCtx ctx{
                .x = x,
                .scratch_vec = scratch_vec,
                .err = err,
            };
            if (!eval_fill_diag_rec(term, out, level_idx, ctx))
            {
                return std::unexpected(
                    err.has_value() ? std::move(*err)
                                    : GprError::unsupported_kernel_operation("kernel evaluation failed")
                );
            }
            return {};
        }

        /// @brief Evaluates one leaf gradient out-of-line so recursive frames stay free of `std::expected`.
        /// @param[in] term Leaf AST node.
        /// @param[in] store Pre-validated square supplied-distance store.
        /// @param[in] x Optional coordinate matrix.
        /// @param[out] d_k Output derivative matrix.
        /// @param[in] param_idx Global flattened hyperparameter index.
        /// @param[in] uplo Triangle mode.
        /// @param[in,out] nested Multi-level workspace.
        /// @param[in] level_idx Current recursion depth into @p nested.
        /// @param[out] err Populated with the error on failure.
        /// @return True on success, false after writing @p err.
        [[nodiscard]] CXXGPR_NOINLINE bool grad_leaf_step(
            const Term& term,
            const CheckedDistanceStore<T>& store,
            std::optional<MatrixRef<T>> x,
            MatrixMut<T> d_k,
            std::size_t param_idx,
            Triangle uplo,
            NestedBuffers<T>& nested,
            std::size_t level_idx,
            std::optional<GprError>& err
        ) const
        {
            if (const auto* s = std::get_if<SuppliedScalarRef>(&term))
            {
                const MatrixRef<T> dist(store.impl_->scalar_tables[s->slot_at].data(), store.rows(), store.cols());
                return capture_expected(
                    detail::leaf_grad_dist(leaves[s->leaf_idx], dist, d_k, param_idx - s->param_begin, uplo, exp_mode),
                    err
                );
            }
            if (const auto* a = std::get_if<SuppliedArdRef>(&term))
            {
                eval_supplied_ard_grad(*a, store, d_k, param_idx - a->param_begin, uplo);
                return true;
            }
            const auto& c = std::get<CoordLeafRef>(term);
            if (x.has_value())
            {
                const KernelLeaf& kl = leaves[c.leaf_idx];
                if (const auto* custom = std::get_if<CustomKernel>(&kl))
                {
                    if (!ensure_nested_ok(nested, level_idx, x->rows(), x->rows(), err))
                    {
                        return false;
                    }
                    auto dist_buf = nested.level(level_idx, x->rows(), x->rows());
                    fill_square_sq_dist(*x, dist_buf);
                    return capture_expected(
                        custom->template term<T>().grad(dist_buf.as_ref(), d_k, param_idx - c.param_begin, uplo), err
                    );
                }
                return capture_expected(
                    detail::leaf_grad_points(
                        kl, *x, d_k, param_idx - c.param_begin, uplo, exp_mode, cached_ls[c.leaf_idx]
                    ),
                    err
                );
            }
            if (const auto* ck = std::get_if<ConstantKernel>(&leaves[c.leaf_idx]))
            {
                const T val = static_cast<T>(ck->constant());
                visit_triangle<T>(store.rows(), uplo, [&](std::size_t row, std::size_t col) { d_k(row, col) = val; });
                return true;
            }
            const auto& wk = std::get<WhiteKernel>(leaves[c.leaf_idx]);
            const T var = static_cast<T>(wk.variance());
            visit_triangle<T>(store.rows(), uplo, [&](std::size_t row, std::size_t col) {
                d_k(row, col) = (row == col) ? var : T{0};
            });
            return true;
        }

        /// @brief Invariant context for `eval_grad_rec` to keep recursive stack frames minimal.
        struct GradEvalCtx
        {
            /// @brief Pre-validated square supplied-distance store.
            const CheckedDistanceStore<T>& store;
            /// @brief Optional coordinate matrix.
            std::optional<MatrixRef<T>> x;
            /// @brief Active matrix triangle.
            Triangle uplo;
            /// @brief First scratch matrix.
            MatrixMut<T> scratch_a;
            /// @brief Second scratch matrix.
            MatrixMut<T> scratch_b;
            /// @brief Multi-level workspace.
            NestedBuffers<T>& nested;
            /// @brief Out-of-line error slot.
            std::optional<GprError>& err;
        };

        /// @brief Recursive body of `eval_grad` with minimal per-level stack frame size.
        /// @param[in] term Compiled AST node containing @p param_idx.
        /// @param[out] d_k Output derivative matrix.
        /// @param[in] param_idx Global flattened hyperparameter index.
        /// @param[in] level_idx Current recursion depth into @p ctx.nested.
        /// @param[in] ctx Invariant gradient evaluation context.
        /// @return True on success, false after writing @p ctx.err.
        [[nodiscard]] bool eval_grad_rec(
            const Term& term,
            const MatrixMut<T>& d_k,
            std::size_t param_idx,
            std::size_t level_idx,
            const GradEvalCtx& ctx
        ) const
        {
            if (is_leaf_term(term))
            {
                return grad_leaf_step(term, ctx.store, ctx.x, d_k, param_idx, ctx.uplo, ctx.nested, level_idx, ctx.err);
            }
            if (const auto* s = std::get_if<SumTerm>(&term))
            {
                // Linearity of differentiation: only the single addend owning param_idx contributes.
                for (const auto& child : s->terms)
                {
                    if (contains_param(child, param_idx))
                    {
                        return eval_grad_rec(child, d_k, param_idx, level_idx, ctx);
                    }
                }
                return fail_param_out_of_range(ctx.err);
            }
            const auto& p = std::get<ProductTerm>(term);
            std::size_t owner = p.terms.size();
            for (std::size_t m = 0; m < p.terms.size(); ++m)
            {
                if (contains_param(p.terms[m], param_idx))
                {
                    owner = m;
                    break;
                }
            }
            if (owner == p.terms.size())
            {
                return fail_param_out_of_range(ctx.err);
            }
            // Product rule: differentiate the owning factor and scale by all other factor values.
            if (!eval_grad_rec(p.terms[owner], d_k, param_idx, level_idx, ctx))
            {
                return false;
            }
            const SquareEvalCtx sctx{
                .store = ctx.store,
                .x = ctx.x,
                .uplo = ctx.uplo,
                .nested = ctx.nested,
                .err = ctx.err,
            };
            for (std::size_t m = 0; m < p.terms.size(); ++m)
            {
                if (m == owner)
                {
                    continue;
                }
                if (!eval_term_square_rec(p.terms[m], ctx.scratch_a, ctx.scratch_b, level_idx, sctx))
                {
                    return false;
                }
                mul_into(d_k, ctx.scratch_a.as_ref(), ctx.uplo);
            }
            return true;
        }

        /// @brief Recursively computes the first log-parameter derivative matrix.
        /// @param[in] term Compiled AST node containing @p param_idx.
        /// @param[in] store Pre-validated square supplied-distance store.
        /// @param[in] x Optional coordinate matrix.
        /// @param[out] d_k Output derivative matrix.
        /// @param[in] param_idx Global flattened hyperparameter index.
        /// @param[in] uplo Triangle mode.
        /// @param[in,out] scratch_a First scratch matrix.
        /// @param[in,out] scratch_b Second scratch matrix.
        /// @param[in,out] nested Multi-level workspace.
        /// @param[in] level_idx Current recursion depth into @p nested.
        /// @return Empty expected value on success, or a structured @c GprError on evaluation failure.
        [[nodiscard]] std::expected<void, GprError> eval_grad(
            const Term& term,
            const CheckedDistanceStore<T>& store,
            std::optional<MatrixRef<T>> x,
            MatrixMut<T> d_k,
            std::size_t param_idx,
            Triangle uplo,
            MatrixMut<T> scratch_a,
            MatrixMut<T> scratch_b,
            NestedBuffers<T>& nested,
            std::size_t level_idx
        ) const
        {
            std::optional<GprError> err;
            const GradEvalCtx ctx{
                .store = store,
                .x = x,
                .uplo = uplo,
                .scratch_a = scratch_a,
                .scratch_b = scratch_b,
                .nested = nested,
                .err = err,
            };
            if (!eval_grad_rec(term, d_k, param_idx, level_idx, ctx))
            {
                return std::unexpected(
                    err.has_value() ? std::move(*err)
                                    : GprError::unsupported_kernel_operation("kernel evaluation failed")
                );
            }
            return {};
        }

        /// @brief Evaluates one leaf Hessian out-of-line so recursive frames stay free of `std::expected`.
        /// @param[in] term Leaf AST node.
        /// @param[in] store Pre-validated square supplied-distance store.
        /// @param[in] x Optional coordinate matrix.
        /// @param[out] d2_k Output second-derivative matrix.
        /// @param[in] i First global hyperparameter index.
        /// @param[in] j Second global hyperparameter index.
        /// @param[in] uplo Triangle mode.
        /// @param[in,out] nested Multi-level workspace.
        /// @param[in] level_idx Current recursion depth into @p nested.
        /// @param[out] err Populated with the error on failure.
        /// @return True on success, false after writing @p err.
        [[nodiscard]] CXXGPR_NOINLINE bool hess_leaf_step(
            const Term& term,
            const CheckedDistanceStore<T>& store,
            std::optional<MatrixRef<T>> x,
            MatrixMut<T> d2_k,
            std::size_t i,
            std::size_t j,
            Triangle uplo,
            NestedBuffers<T>& nested,
            std::size_t level_idx,
            std::optional<GprError>& err
        ) const
        {
            if (const auto* s = std::get_if<SuppliedScalarRef>(&term))
            {
                const MatrixRef<T> dist(store.impl_->scalar_tables[s->slot_at].data(), store.rows(), store.cols());
                return capture_expected(
                    detail::leaf_hess_dist(
                        leaves[s->leaf_idx], dist, d2_k, i - s->param_begin, j - s->param_begin, uplo, exp_mode
                    ),
                    err
                );
            }
            if (const auto* a = std::get_if<SuppliedArdRef>(&term))
            {
                eval_supplied_ard_hess(*a, store, d2_k, i - a->param_begin, j - a->param_begin, uplo);
                return true;
            }
            const auto& c = std::get<CoordLeafRef>(term);
            if (x.has_value())
            {
                const KernelLeaf& kl = leaves[c.leaf_idx];
                if (const auto* custom = std::get_if<CustomKernel>(&kl))
                {
                    if (!ensure_nested_ok(nested, level_idx, x->rows(), x->rows(), err))
                    {
                        return false;
                    }
                    auto dist_buf = nested.level(level_idx, x->rows(), x->rows());
                    fill_square_sq_dist(*x, dist_buf);
                    return capture_expected(
                        custom->template term<T>().hess(
                            dist_buf.as_ref(), d2_k, i - c.param_begin, j - c.param_begin, uplo
                        ),
                        err
                    );
                }
                return capture_expected(
                    detail::leaf_hess_points(
                        kl, *x, d2_k, i - c.param_begin, j - c.param_begin, uplo, exp_mode, cached_ls[c.leaf_idx]
                    ),
                    err
                );
            }
            if (const auto* ck = std::get_if<ConstantKernel>(&leaves[c.leaf_idx]))
            {
                const T val = static_cast<T>(ck->constant());
                visit_triangle<T>(store.rows(), uplo, [&](std::size_t row, std::size_t col) { d2_k(row, col) = val; });
                return true;
            }
            const auto& wk = std::get<WhiteKernel>(leaves[c.leaf_idx]);
            const T var = static_cast<T>(wk.variance());
            visit_triangle<T>(store.rows(), uplo, [&](std::size_t row, std::size_t col) {
                d2_k(row, col) = (row == col) ? var : T{0};
            });
            return true;
        }

        /// @brief Invariant context for `eval_hess_rec` to keep recursive stack frames minimal.
        struct HessEvalCtx
        {
            /// @brief Pre-validated square supplied-distance store.
            const CheckedDistanceStore<T>& store;
            /// @brief Optional coordinate matrix.
            std::optional<MatrixRef<T>> x;
            /// @brief First global hyperparameter index.
            std::size_t i;
            /// @brief Second global hyperparameter index.
            std::size_t j;
            /// @brief Active matrix triangle.
            Triangle uplo;
            /// @brief First scratch matrix.
            MatrixMut<T> scratch_a;
            /// @brief Second scratch matrix.
            MatrixMut<T> scratch_b;
            /// @brief Third scratch matrix.
            MatrixMut<T> scratch_c;
            /// @brief Multi-level workspace.
            NestedBuffers<T>& nested;
            /// @brief Out-of-line error slot.
            std::optional<GprError>& err;
        };

        /// @brief Recursive body of `eval_hess` with minimal per-level stack frame size.
        /// @param[in] term Compiled AST node containing @p ctx.i and @p ctx.j.
        /// @param[out] d2_k Output second-derivative matrix.
        /// @param[in] level_idx Current recursion depth into @p ctx.nested.
        /// @param[in] ctx Invariant Hessian evaluation context.
        /// @return True on success, false after writing @p ctx.err.
        [[nodiscard]] bool eval_hess_rec(
            const Term& term, const MatrixMut<T>& d2_k, std::size_t level_idx, const HessEvalCtx& ctx
        ) const
        {
            if (is_leaf_term(term))
            {
                return hess_leaf_step(
                    term, ctx.store, ctx.x, d2_k, ctx.i, ctx.j, ctx.uplo, ctx.nested, level_idx, ctx.err
                );
            }
            if (const auto* s = std::get_if<SumTerm>(&term))
            {
                for (const auto& child : s->terms)
                {
                    const bool has_i = contains_param(child, ctx.i);
                    const bool has_j = contains_param(child, ctx.j);
                    if (has_i && has_j)
                    {
                        return eval_hess_rec(child, d2_k, level_idx, ctx);
                    }
                    if (has_i || has_j)
                    {
                        // Distinct addends in a sum depend on disjoint parameters, so mixed second partials vanish.
                        visit_triangle<T>(ctx.store.rows(), ctx.uplo, [&](std::size_t row, std::size_t col) {
                            d2_k(row, col) = T{0};
                        });
                        return true;
                    }
                }
                return fail_param_out_of_range(ctx.err);
            }
            const auto& p = std::get<ProductTerm>(term);
            std::size_t owner_i = p.terms.size();
            std::size_t owner_j = p.terms.size();
            for (std::size_t m = 0; m < p.terms.size(); ++m)
            {
                if (contains_param(p.terms[m], ctx.i))
                {
                    owner_i = m;
                }
                if (contains_param(p.terms[m], ctx.j))
                {
                    owner_j = m;
                }
            }
            if (owner_i == p.terms.size() || owner_j == p.terms.size())
            {
                return fail_param_out_of_range(ctx.err);
            }
            const SquareEvalCtx sctx{
                .store = ctx.store,
                .x = ctx.x,
                .uplo = ctx.uplo,
                .nested = ctx.nested,
                .err = ctx.err,
            };
            if (owner_i == owner_j)
            {
                // Both parameters belong to one factor: compute factor Hessian and multiply all sibling values.
                if (!eval_hess_rec(p.terms[owner_i], d2_k, level_idx, ctx))
                {
                    return false;
                }
                for (std::size_t m = 0; m < p.terms.size(); ++m)
                {
                    if (m == owner_i)
                    {
                        continue;
                    }
                    if (!eval_term_square_rec(p.terms[m], ctx.scratch_b, ctx.scratch_c, level_idx, sctx))
                    {
                        return false;
                    }
                    mul_into(d2_k, ctx.scratch_b.as_ref(), ctx.uplo);
                }
                return true;
            }
            // Mixed product-rule: (d factor_i / d theta_i) * (d factor_j / d theta_j) * prod_{m != i,j} factor_m.
            const GradEvalCtx gctx{
                .store = ctx.store,
                .x = ctx.x,
                .uplo = ctx.uplo,
                .scratch_a = ctx.scratch_b,
                .scratch_b = ctx.scratch_c,
                .nested = ctx.nested,
                .err = ctx.err,
            };
            if (!eval_grad_rec(p.terms[owner_i], d2_k, ctx.i, level_idx, gctx))
            {
                return false;
            }
            if (!eval_grad_rec(p.terms[owner_j], ctx.scratch_a, ctx.j, level_idx, gctx))
            {
                return false;
            }
            mul_into(d2_k, ctx.scratch_a.as_ref(), ctx.uplo);
            for (std::size_t m = 0; m < p.terms.size(); ++m)
            {
                if (m == owner_i || m == owner_j)
                {
                    continue;
                }
                if (!eval_term_square_rec(p.terms[m], ctx.scratch_b, ctx.scratch_c, level_idx, sctx))
                {
                    return false;
                }
                mul_into(d2_k, ctx.scratch_b.as_ref(), ctx.uplo);
            }
            return true;
        }

        /// @brief Recursively computes the second log-parameter derivative matrix.
        /// @param[in] term Compiled AST node containing @p i and @p j.
        /// @param[in] store Pre-validated square supplied-distance store.
        /// @param[in] x Optional coordinate matrix.
        /// @param[out] d2_k Output second-derivative matrix.
        /// @param[in] i First global hyperparameter index.
        /// @param[in] j Second global hyperparameter index.
        /// @param[in] uplo Triangle mode.
        /// @param[in,out] scratch_a First scratch matrix.
        /// @param[in,out] scratch_b Second scratch matrix.
        /// @param[in,out] scratch_c Third scratch matrix.
        /// @param[in,out] nested Multi-level workspace.
        /// @param[in] level_idx Current recursion depth into @p nested.
        /// @return Empty expected value on success, or a structured @c GprError on evaluation failure.
        [[nodiscard]] std::expected<void, GprError> eval_hess(
            const Term& term,
            const CheckedDistanceStore<T>& store,
            std::optional<MatrixRef<T>> x,
            MatrixMut<T> d2_k,
            std::size_t i,
            std::size_t j,
            Triangle uplo,
            MatrixMut<T> scratch_a,
            MatrixMut<T> scratch_b,
            MatrixMut<T> scratch_c,
            NestedBuffers<T>& nested,
            std::size_t level_idx
        ) const
        {
            std::optional<GprError> err;
            const HessEvalCtx ctx{
                .store = store,
                .x = x,
                .i = i,
                .j = j,
                .uplo = uplo,
                .scratch_a = scratch_a,
                .scratch_b = scratch_b,
                .scratch_c = scratch_c,
                .nested = nested,
                .err = err,
            };
            if (!eval_hess_rec(term, d2_k, level_idx, ctx))
            {
                return std::unexpected(
                    err.has_value() ? std::move(*err)
                                    : GprError::unsupported_kernel_operation("kernel evaluation failed")
                );
            }
            return {};
        }
    };

    // --- DistanceStoreBuilder ---

    /// @brief Internal helper aggregating validation, tidy repair, and packed allocation for @c CheckedDistanceStore.
    /// @details Enforces slot correspondence, finiteness, strict or @c tidy_rel_tol_-bounded non-negativity and
    /// symmetry, overflow-checked scalar narrowing to @c T, and cache-friendly packing.
    struct DistanceStoreBuilder
    {
        /// @brief Tile dimension (rows and columns per block) used during pairwise symmetry validation.
        static constexpr std::size_t k_pair_tile = 64U;

        /// @brief Narrows finite @c double value @p val to @p T and reports overflow with coordinates on failure.
        /// @tparam T Target storage scalar type (@c double or @c float).
        /// @param[in] val Repaired non-negative squared distance value to cast.
        /// @param[in] slot Optional packed slot index for diagnostic error context.
        /// @param[in] dim Optional ARD dimension index for diagnostic error context.
        /// @param[in] row Matrix row index where @p val resides.
        /// @param[in] col Matrix column index where @p val resides.
        /// @return Cast value of type @p T on success, or @c GprError::invalid_distance on non-finite overflow.
        template <KernelScalarConcept T>
        [[nodiscard]] static std::expected<T, GprError> checked_cast(
            double val,
            std::optional<std::size_t> slot,
            std::optional<std::size_t> dim,
            std::size_t row,
            std::size_t col
        )
        {
            const auto cast = static_cast<T>(val);
            if (!std::isfinite(cast))
            {
                return std::unexpected(
                    GprError::invalid_distance(slot, dim, std::pair{row, col}, "value overflows storage scalar range")
                );
            }
            return cast;
        }

        /// @brief Validates a square pairwise squared-distance matrix and appends it (packed or full) to @p dst.
        /// @details Performs a finiteness and max-magnitude scan to set @c tol = *tidy_tol * max_abs when tidy mode is
        /// enabled, validates diagonal-zero, non-negativity, and symmetry in @c k_pair_tile blocks, repairs
        /// tiny negative or asymmetric entries via midpoint averaging and zero clamping, and writes either packed
        /// lower-triangular storage (@p pack_lower = @c true) or mirrored column-major storage (@p pack_lower =
        /// @c false).
        /// @tparam T Target storage scalar type (@c double or @c float).
        /// @param[in] src Input square squared-distance matrix view.
        /// @param[in] tidy_tol Optional relative tolerance enabling small negative/diagonal/asymmetry repair.
        /// @param[in] slot_idx Packed slot index for error diagnostics.
        /// @param[in] dim_idx Optional ARD dimension index for error diagnostics.
        /// @param[in] pack_lower If @c true, append packed lower triangle; otherwise resize and mirror full @c n x n.
        /// @param[in,out] dst Destination storage vector populated with validated scalar entries.
        /// @return Empty expected value on success, or @c GprError describing the invalid entry or overflow.
        template <KernelScalarConcept T>
        [[nodiscard]] static std::expected<void, GprError> validate_and_pack_square_block(
            MatrixRef<double> src,
            std::optional<double> tidy_tol,
            std::optional<std::size_t> slot_idx,
            std::optional<std::size_t> dim_idx,
            bool pack_lower,
            std::vector<T>& dst
        )
        {
            const std::size_t n = src.rows();
            // Pass 1: verify finiteness across all n*n entries and capture max_abs to scale tidy_tol.
            double max_abs = 0.0;
            for (std::size_t col = 0; col < n; ++col)
            {
                for (std::size_t row = 0; row < n; ++row)
                {
                    const double v = src(row, col);
                    if (!std::isfinite(v))
                    {
                        return std::unexpected(
                            GprError::invalid_distance(slot_idx, dim_idx, std::pair{row, col}, "distance is not finite")
                        );
                    }
                    const double av = std::abs(v);
                    max_abs = std::max(av, max_abs);
                }
            }
            const double tol = tidy_tol.has_value() ? (*tidy_tol * max_abs) : 0.0;

            if (!tidy_tol.has_value())
            {
                // Strict mode: require exact zero diagonal, exact non-negativity, and bitwise symmetry a == b.
                for (std::size_t col = 0; col < n; ++col)
                {
                    const double diag = src(col, col);
                    if (diag < 0.0)
                    {
                        return std::unexpected(
                            GprError::invalid_distance(slot_idx, dim_idx, std::pair{col, col}, "distance is negative")
                        );
                    }
                    if (diag != 0.0)
                    {
                        return std::unexpected(
                            GprError::invalid_distance(slot_idx, dim_idx, std::pair{col, col}, "diagonal must be zero")
                        );
                    }
                }
                // Walk strict lower triangle in k_pair_tile x k_pair_tile blocks for cache-local transpose reads.
                for (std::size_t j0 = 0; j0 < n; j0 += k_pair_tile)
                {
                    const std::size_t j1 = std::min(j0 + k_pair_tile, n);
                    for (std::size_t i0 = j0; i0 < n; i0 += k_pair_tile)
                    {
                        const std::size_t i1 = std::min(i0 + k_pair_tile, n);
                        for (std::size_t j = j0; j < j1; ++j)
                        {
                            for (std::size_t i = std::max(i0, j + 1U); i < i1; ++i)
                            {
                                const double a = src(i, j);
                                const double b = src(j, i);
                                if (a < 0.0)
                                {
                                    return std::unexpected(
                                        GprError::invalid_distance(
                                            slot_idx, dim_idx, std::pair{i, j}, "distance is negative"
                                        )
                                    );
                                }
                                if (b < 0.0)
                                {
                                    return std::unexpected(
                                        GprError::invalid_distance(
                                            slot_idx, dim_idx, std::pair{j, i}, "distance is negative"
                                        )
                                    );
                                }
                                if (a != b)
                                {
                                    return std::unexpected(
                                        GprError::invalid_distance(
                                            slot_idx, dim_idx, std::pair{i, j},
                                            "square distance matrix is not symmetric"
                                        )
                                    );
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                // Tidy mode: reject violations larger than tol = tidy_rel_tol * max_abs before clamping/symmetrizing.
                for (std::size_t col = 0; col < n; ++col)
                {
                    for (std::size_t row = 0; row < n; ++row)
                    {
                        const double v = src(row, col);
                        if (v < -tol)
                        {
                            return std::unexpected(
                                GprError::invalid_distance(
                                    slot_idx, dim_idx, std::pair{row, col}, "negative distance exceeds tidy tolerance"
                                )
                            );
                        }
                    }
                    if (std::abs(src(col, col)) > tol)
                    {
                        return std::unexpected(
                            GprError::invalid_distance(
                                slot_idx, dim_idx, std::pair{col, col}, "nonzero diagonal exceeds tidy tolerance"
                            )
                        );
                    }
                }
                for (std::size_t j0 = 0; j0 < n; j0 += k_pair_tile)
                {
                    const std::size_t j1 = std::min(j0 + k_pair_tile, n);
                    for (std::size_t i0 = j0; i0 < n; i0 += k_pair_tile)
                    {
                        const std::size_t i1 = std::min(i0 + k_pair_tile, n);
                        for (std::size_t j = j0; j < j1; ++j)
                        {
                            for (std::size_t i = std::max(i0, j + 1U); i < i1; ++i)
                            {
                                const double gap = std::abs(src(i, j) - src(j, i));
                                if (gap > tol)
                                {
                                    return std::unexpected(
                                        GprError::invalid_distance(
                                            slot_idx, dim_idx, std::pair{i, j}, "asymmetry exceeds tidy tolerance"
                                        )
                                    );
                                }
                            }
                        }
                    }
                }
            }

            // Force diagonal to zero, average off-diagonal pairs a + 0.5*(b - a) for midpoint accuracy,
            // and clamp any values in [-tol, 0) up to 0 before checked narrowing cast to T.
            if (pack_lower)
            {
                const auto per_dim_exp = packed_len(n);
                if (!per_dim_exp.has_value())
                {
                    return std::unexpected(std::move(per_dim_exp).error());
                }
                const std::size_t per_dim = *per_dim_exp;
                const std::size_t base_offset = dst.size();
                dst.resize(base_offset + per_dim);
                for (std::size_t j = 0; j < n; ++j)
                {
                    const std::size_t diag_p = packed_lower_index(n, j, j);
                    dst[base_offset + diag_p] = T{0};
                    for (std::size_t i = j + 1U; i < n; ++i)
                    {
                        const double a = src(i, j);
                        const double b = src(j, i);
                        const double mean = (a == b) ? a : (a + (0.5 * (b - a)));
                        const double repaired = std::max(0.0, mean);
                        auto cast = checked_cast<T>(repaired, slot_idx, dim_idx, i, j);
                        if (!cast.has_value())
                        {
                            return std::unexpected(std::move(cast).error());
                        }
                        dst[base_offset + packed_lower_index(n, i, j)] = *cast;
                    }
                }
            }
            else
            {
                const auto len_exp = column_major_len(n, n);
                if (!len_exp.has_value())
                {
                    return std::unexpected(std::move(len_exp).error());
                }
                dst.resize(*len_exp);
                for (std::size_t j = 0; j < n; ++j)
                {
                    dst[(j * n) + j] = T{0};
                    for (std::size_t i = j + 1U; i < n; ++i)
                    {
                        const double a = src(i, j);
                        const double b = src(j, i);
                        const double mean = (a == b) ? a : (a + (0.5 * (b - a)));
                        const double repaired = std::max(0.0, mean);
                        auto cast = checked_cast<T>(repaired, slot_idx, dim_idx, i, j);
                        if (!cast.has_value())
                        {
                            return std::unexpected(std::move(cast).error());
                        }
                        dst[(j * n) + i] = *cast;
                        dst[(i * n) + j] = *cast;
                    }
                }
            }
            return {};
        }

        /// @brief Validates a rectangular cross squared-distance block and appends its column-major entries to @p dst.
        /// @details Enforces finiteness across all entries and checks non-negativity strictly or within
        /// @c -(*tidy_tol * max_abs), clamping tiny negative values to zero before narrowing to @p T.
        /// @tparam T Target storage scalar type (@c double or @c float).
        /// @param[in] src Input rectangular squared-distance matrix view.
        /// @param[in] tidy_tol Optional relative tolerance permitting clamping of tiny negative entries.
        /// @param[in] slot_idx Packed slot index for error diagnostics.
        /// @param[in] dim_idx Optional ARD dimension index for error diagnostics.
        /// @param[in,out] dst Destination column-major buffer to which validated entries are appended.
        /// @return Empty expected value on success, or @c GprError describing invalid entries or overflow.
        template <KernelScalarConcept T>
        [[nodiscard]] static std::expected<void, GprError> validate_and_copy_rect_block(
            MatrixRef<double> src,
            std::optional<double> tidy_tol,
            std::optional<std::size_t> slot_idx,
            std::optional<std::size_t> dim_idx,
            std::vector<T>& dst
        )
        {
            const std::size_t n_rows = src.rows();
            const std::size_t n_cols = src.cols();
            // Pass 1: check finiteness across all entries and record max_abs for relative tidy tolerance.
            double max_abs = 0.0;
            for (std::size_t col = 0; col < n_cols; ++col)
            {
                for (std::size_t row = 0; row < n_rows; ++row)
                {
                    const double v = src(row, col);
                    if (!std::isfinite(v))
                    {
                        return std::unexpected(
                            GprError::invalid_distance(slot_idx, dim_idx, std::pair{row, col}, "distance is not finite")
                        );
                    }
                    const double av = std::abs(v);
                    max_abs = std::max(av, max_abs);
                }
            }
            const double tol = tidy_tol.has_value() ? (*tidy_tol * max_abs) : 0.0;
            const std::size_t base_offset = dst.size();
            const auto len_exp = column_major_len(n_rows, n_cols);
            if (!len_exp.has_value())
            {
                return std::unexpected(std::move(len_exp).error());
            }
            dst.resize(base_offset + *len_exp);
            // Pass 2: validate non-negativity (strict or within -tol), clamp [-tol, 0) to 0, and narrow to T.
            for (std::size_t col = 0; col < n_cols; ++col)
            {
                for (std::size_t row = 0; row < n_rows; ++row)
                {
                    const double v = src(row, col);
                    if (!tidy_tol.has_value() && v < 0.0)
                    {
                        return std::unexpected(
                            GprError::invalid_distance(slot_idx, dim_idx, std::pair{row, col}, "distance is negative")
                        );
                    }
                    if (tidy_tol.has_value() && v < -tol)
                    {
                        return std::unexpected(
                            GprError::invalid_distance(
                                slot_idx, dim_idx, std::pair{row, col}, "negative distance exceeds tidy tolerance"
                            )
                        );
                    }
                    const double repaired = std::max(0.0, v);
                    auto cast = checked_cast<T>(repaired, slot_idx, dim_idx, row, col);
                    if (!cast.has_value())
                    {
                        return std::unexpected(std::move(cast).error());
                    }
                    dst[base_offset + (col * n_rows) + row] = *cast;
                }
            }
            return {};
        }

        /// @brief Returns the slot descriptor of @p kernel carrying @p id, or nullptr for a foreign slot.
        /// @param[in] kslots Slot descriptors of the compiled kernel.
        /// @param[in] id Slot identifier supplied by the caller.
        /// @return Pointer into @p kslots, or nullptr when no slot carries @p id.
        [[nodiscard]] static const SlotDescriptor* find_desc(const std::vector<SlotDescriptor>& kslots, SlotId id)
        {
            for (const auto& d : kslots)
            {
                if (d.id == id)
                {
                    return &d;
                }
            }
            return nullptr;
        }

        /// @brief Counts how many supplied scalar and ARD entries of @p source bind slot @p id.
        /// @param[in] source Unchecked supply.
        /// @param[in] id Slot identifier.
        /// @return Number of entries binding @p id (scalar and ARD entries together).
        [[nodiscard]] static std::size_t count_bindings(const DistanceSource& source, SlotId id) noexcept
        {
            std::size_t count = 0;
            for (const auto& e : source.scalars_)
            {
                count += (e.slot == id) ? 1U : 0U;
            }
            for (const auto& e : source.ards_)
            {
                count += (e.slot == id) ? 1U : 0U;
            }
            return count;
        }

        /// @brief Checks the structure of @p source against @p kernel without reading any table value.
        /// @details Applies, in this order: the @c tidy tolerance (InvalidConfig); for every scalar entry and then
        /// every ARD entry, a slot of another kernel (UnsupportedKernelOperation), a slot of the other kind
        /// (DistanceSlot NotRead) and a duplicate binding (DistanceSlot Duplicate); a missing slot (DistanceSlot
        /// Missing); a kernel without slots (UnsupportedKernelOperation); an empty table (EmptyInput); a size product
        /// that overflows (SizeOverflow); and tables of different shapes or a non-square training table
        /// (LengthMismatch). The number of per-dimension ARD tables is checked separately by check_ard_counts(). The
        /// check allocates nothing.
        /// @tparam T Storage scalar type of the compiled kernel.
        /// @param[in] kernel Compiled kernel defining the expected slots.
        /// @param[in] source Unchecked supply.
        /// @param[in] is_square True for a training (square) supply, false for a cross supply.
        /// @return Common table shape, or the first violated GprError.
        template <KernelScalarConcept T>
        [[nodiscard]] static std::expected<detail::DistanceTableShape, GprError> check_structure(
            const CompiledDistanceKernel<T>& kernel, const DistanceSource& source, bool is_square
        )
        {
            if (source.tidy_rel_tol_.has_value())
            {
                const double tol = *source.tidy_rel_tol_;
                if (!std::isfinite(tol) || tol < 0.0)
                {
                    return std::unexpected(GprError::invalid_config("tidy relative tolerance must be finite and >= 0"));
                }
            }

            const auto& kslots = kernel.slots();
            for (const auto& entry : source.scalars_)
            {
                const auto* desc = find_desc(kslots, entry.slot);
                if (desc == nullptr)
                {
                    return std::unexpected(
                        GprError::unsupported_kernel_operation("distance source supplies a slot from another kernel")
                    );
                }
                if (desc->ard_dims.has_value())
                {
                    return std::unexpected(GprError::distance_slot(SlotErrorKind::NotRead, desc->at));
                }
                if (count_bindings(source, entry.slot) > 1U)
                {
                    return std::unexpected(GprError::distance_slot(SlotErrorKind::Duplicate, desc->at));
                }
            }
            for (const auto& entry : source.ards_)
            {
                const auto* desc = find_desc(kslots, entry.slot);
                if (desc == nullptr)
                {
                    return std::unexpected(
                        GprError::unsupported_kernel_operation("distance source supplies a slot from another kernel")
                    );
                }
                if (!desc->ard_dims.has_value())
                {
                    return std::unexpected(GprError::distance_slot(SlotErrorKind::NotRead, desc->at));
                }
                if (count_bindings(source, entry.slot) > 1U)
                {
                    return std::unexpected(GprError::distance_slot(SlotErrorKind::Duplicate, desc->at));
                }
            }
            for (const auto& desc : kslots)
            {
                if (count_bindings(source, desc.id) == 0U)
                {
                    return std::unexpected(GprError::distance_slot(SlotErrorKind::Missing, desc.at));
                }
            }
            if (kslots.empty())
            {
                return std::unexpected(GprError::unsupported_kernel_operation("kernel has no supplied distance slots"));
            }

            // Every table must be non-empty before any shape comparison.
            for (const auto& entry : source.scalars_)
            {
                if (entry.d2.rows() == 0U || entry.d2.cols() == 0U)
                {
                    return std::unexpected(GprError::empty_input());
                }
            }
            for (const auto& entry : source.ards_)
            {
                if (entry.d2_per_dim.empty())
                {
                    return std::unexpected(GprError::empty_input());
                }
                for (const auto& blk : entry.d2_per_dim)
                {
                    if (blk.rows() == 0U || blk.cols() == 0U)
                    {
                        return std::unexpected(GprError::empty_input());
                    }
                }
            }

            const MatrixRef<double>& first =
                source.scalars_.empty() ? source.ards_[0].d2_per_dim[0] : source.scalars_[0].d2;
            const detail::DistanceTableShape shape{.rows = first.rows(), .cols = first.cols()};

            // Size products of the stored blocks, using the dimension counts the kernel expects.
            const auto block_len = is_square ? packed_len(shape.rows) : column_major_len(shape.rows, shape.cols);
            if (!block_len.has_value())
            {
                return std::unexpected(block_len.error());
            }
            if (auto dense = column_major_len(shape.rows, shape.cols); !dense.has_value())
            {
                return std::unexpected(std::move(dense).error());
            }
            for (const auto& desc : kslots)
            {
                if (desc.ard_dims.has_value())
                {
                    if (auto total = checked_mul(*block_len, *desc.ard_dims); !total.has_value())
                    {
                        return std::unexpected(std::move(total).error());
                    }
                }
            }

            if (is_square && shape.rows != shape.cols)
            {
                return std::unexpected(
                    GprError::length_mismatch("square distance matrix must have equal rows and cols")
                );
            }
            for (const auto& entry : source.scalars_)
            {
                if (entry.d2.rows() != shape.rows || entry.d2.cols() != shape.cols)
                {
                    return std::unexpected(GprError::length_mismatch("scalar distance block shape mismatch"));
                }
            }
            for (const auto& entry : source.ards_)
            {
                for (const auto& blk : entry.d2_per_dim)
                {
                    if (blk.rows() != shape.rows || blk.cols() != shape.cols)
                    {
                        return std::unexpected(GprError::length_mismatch("ARD distance block shape mismatch"));
                    }
                }
            }
            return shape;
        }

        /// @brief Checks that every ARD entry of @p source supplies as many per-dimension tables as its slot declares.
        /// @tparam T Storage scalar type of the compiled kernel.
        /// @param[in] kernel Compiled kernel defining the expected slots.
        /// @param[in] source Supply whose structure check_structure() accepted.
        /// @return Empty expected, or DimensionMismatch for the first ARD entry with a wrong table count.
        template <KernelScalarConcept T>
        [[nodiscard]] static std::expected<void, GprError> check_ard_counts(
            const CompiledDistanceKernel<T>& kernel, const DistanceSource& source
        )
        {
            for (const auto& entry : source.ards_)
            {
                const auto* desc = find_desc(kernel.slots(), entry.slot);
                const std::size_t expected_dims = (desc != nullptr) ? desc->ard_dims.value_or(0U) : 0U;
                if (entry.d2_per_dim.size() != expected_dims)
                {
                    return std::unexpected(GprError::dimension_mismatch(entry.d2_per_dim.size(), expected_dims));
                }
            }
            return {};
        }

        /// @brief Validates the table values of @p source and packs them into @p impl, reusing its capacity.
        /// @details Requires that check_structure() and check_ard_counts() accepted @p source. Every destination
        /// vector is cleared and refilled, so a store whose capacity already covers the shape is refilled without
        /// allocating. On failure @p impl holds partial data and must be refilled before use.
        /// @tparam T Storage scalar type.
        /// @param[in] kernel Compiled kernel defining the expected slots.
        /// @param[in] source Supply to validate and pack.
        /// @param[in] shape Common table shape returned by check_structure().
        /// @param[in] is_square True for a training (square) supply, false for a cross supply.
        /// @param[in,out] impl Store state refilled in place.
        /// @return Empty expected, or InvalidDistance for the first invalid entry.
        template <KernelScalarConcept T>
        [[nodiscard]] static std::expected<void, GprError> pack_values(
            const CompiledDistanceKernel<T>& kernel,
            const DistanceSource& source,
            detail::DistanceTableShape shape,
            bool is_square,
            CheckedDistanceStore<T>::Impl& impl
        )
        {
            const auto& kslots = kernel.slots();
            impl.kernel_slots = kslots;
            impl.n_rows = shape.rows;
            impl.n_cols = shape.cols;
            impl.square = is_square;
            impl.scalar_tables.resize(kernel.impl_->num_scalar_slots);
            impl.ard_dims.resize(kernel.impl_->num_ard_slots, 0U);
            impl.ard_data.resize(kernel.impl_->num_ard_slots);
            for (auto& table : impl.scalar_tables)
            {
                table.clear();
            }
            for (auto& table : impl.ard_data)
            {
                table.clear();
            }

            for (const auto& entry : source.scalars_)
            {
                const auto* desc = find_desc(kslots, entry.slot);
                if (is_square)
                {
                    if (auto res = validate_and_pack_square_block<T>(
                            entry.d2, source.tidy_rel_tol_, desc->at, std::nullopt, false, impl.scalar_tables[desc->at]
                        );
                        !res.has_value())
                    {
                        return std::unexpected(std::move(res).error());
                    }
                }
                else
                {
                    if (auto res = validate_and_copy_rect_block<T>(
                            entry.d2, source.tidy_rel_tol_, desc->at, std::nullopt, impl.scalar_tables[desc->at]
                        );
                        !res.has_value())
                    {
                        return std::unexpected(std::move(res).error());
                    }
                }
            }

            for (const auto& entry : source.ards_)
            {
                const auto* desc = find_desc(kslots, entry.slot);
                const std::size_t dims = entry.d2_per_dim.size();
                impl.ard_dims[desc->at] = dims;
                // check_structure() verified that these products do not overflow.
                const std::size_t block = is_square ? packed_len(shape.rows).value_or(0U)
                                                    : column_major_len(shape.rows, shape.cols).value_or(0U);
                impl.ard_data[desc->at].reserve(block * dims);
                for (std::size_t dim = 0; dim < dims; ++dim)
                {
                    const auto& blk = entry.d2_per_dim[dim];
                    if (is_square)
                    {
                        if (auto res = validate_and_pack_square_block<T>(
                                blk, source.tidy_rel_tol_, desc->at, dim, true, impl.ard_data[desc->at]
                            );
                            !res.has_value())
                        {
                            return std::unexpected(std::move(res).error());
                        }
                    }
                    else
                    {
                        if (auto res = validate_and_copy_rect_block<T>(
                                blk, source.tidy_rel_tol_, desc->at, dim, impl.ard_data[desc->at]
                            );
                            !res.has_value())
                        {
                            return std::unexpected(std::move(res).error());
                        }
                    }
                }
            }
            return {};
        }

        /// @brief Validates @p source against @p kernel's slot descriptors and constructs a @c CheckedDistanceStore.
        /// @details Runs check_structure(), check_ard_counts() and pack_values() in this order, so every structural
        /// and shape error is reported before any table value is read.
        /// @tparam T Target storage scalar type (@c double or @c float).
        /// @param[in] kernel Compiled mixed distance kernel defining expected slot signatures.
        /// @param[in] source Unchecked builder source supplying pairwise scalar and per-dimension ARD blocks.
        /// @param[in] is_square @c true for training square distance stores, @c false for rectangular cross stores.
        /// @return Immutable @c CheckedDistanceStore on success, or a structured @c GprError explaining the failure.
        template <KernelScalarConcept T>
        [[nodiscard]] static std::expected<CheckedDistanceStore<T>, GprError> build(
            const CompiledDistanceKernel<T>& kernel, const DistanceSource& source, bool is_square
        )
        {
            auto shape = check_structure(kernel, source, is_square);
            if (!shape.has_value())
            {
                return std::unexpected(std::move(shape).error());
            }
            if (auto counts = check_ard_counts(kernel, source); !counts.has_value())
            {
                return std::unexpected(std::move(counts).error());
            }
            return build_shaped(kernel, source, *shape, is_square);
        }

        /// @brief Validates the values of @p source into @p store, reusing the store's storage when present.
        /// @details The caller has already run check_structure() and check_ard_counts() on @p source. The first
        /// call creates the store; later calls refill it in place and allocate only when the shape grows.
        /// @tparam T Storage scalar type.
        /// @param[in] kernel Compiled kernel defining the expected slots.
        /// @param[in] source Supply to validate and pack.
        /// @param[in] shape Common table shape returned by check_structure().
        /// @param[in] is_square True for a training (square) supply, false for a cross supply.
        /// @param[in,out] store Reusable store; holds partial data after a failure.
        /// @return Pointer to the refilled store (never null), or InvalidDistance for the first invalid entry.
        template <KernelScalarConcept T>
        [[nodiscard]] static std::expected<const CheckedDistanceStore<T>*, GprError> build_into(
            const CompiledDistanceKernel<T>& kernel,
            const DistanceSource& source,
            detail::DistanceTableShape shape,
            bool is_square,
            std::optional<CheckedDistanceStore<T>>& store
        )
        {
            if (!store.has_value())
            {
                store = CheckedDistanceStore<T>(std::make_unique<typename CheckedDistanceStore<T>::Impl>());
            }
            auto& filled = *store;
            if (auto packed = pack_values(kernel, source, shape, is_square, *filled.impl_); !packed.has_value())
            {
                return std::unexpected(std::move(packed).error());
            }
            return &filled;
        }

        /// @brief Validates the values of @p source into a new store after the structure checks passed.
        /// @tparam T Storage scalar type.
        /// @param[in] kernel Compiled kernel defining the expected slots.
        /// @param[in] source Supply to validate and pack.
        /// @param[in] shape Common table shape returned by check_structure().
        /// @param[in] is_square True for a training (square) supply, false for a cross supply.
        /// @return New store, or InvalidDistance for the first invalid entry.
        template <KernelScalarConcept T>
        [[nodiscard]] static std::expected<CheckedDistanceStore<T>, GprError> build_shaped(
            const CompiledDistanceKernel<T>& kernel,
            const DistanceSource& source,
            detail::DistanceTableShape shape,
            bool is_square
        )
        {
            auto impl = std::make_unique<typename CheckedDistanceStore<T>::Impl>();
            if (auto packed = pack_values(kernel, source, shape, is_square, *impl); !packed.has_value())
            {
                return std::unexpected(std::move(packed).error());
            }
            return CheckedDistanceStore<T>(std::move(impl));
        }

        /// @brief Checks the column count of a point matrix against the coordinate leaves of @p kernel.
        /// @tparam T Storage scalar type.
        /// @param[in] kernel Compiled kernel.
        /// @param[in] cols Number of point columns.
        /// @return Empty expected, or DimensionMismatch when an ARD coordinate leaf needs another dimension.
        template <KernelScalarConcept T>
        [[nodiscard]] static std::expected<void, GprError> check_point_cols(
            const CompiledDistanceKernel<T>& kernel, std::size_t cols
        )
        {
            if (!kernel.impl_->has_coord_points)
            {
                return {};
            }
            return kernel.impl_->validate_ard_coord_dims(cols);
        }

        /// @brief Evaluates the kernel diagonal into @p out using the caller's grow-only scratch vector.
        /// @details Same contract as CompiledDistanceKernel::fill_diag(); @p scratch grows to the nested size and is
        /// reused, so repeated calls of the same or a smaller @p n allocate nothing.
        /// @tparam T Storage scalar type.
        /// @param[in] kernel Compiled kernel.
        /// @param[in] n Number of diagonal entries.
        /// @param[in] x Optional point matrix of @p n rows.
        /// @param[out] out Destination of length @p n.
        /// @param[in,out] scratch Grow-only scratch storage.
        /// @return Empty expected, or GprError.
        template <KernelScalarConcept T>
        [[nodiscard]] static std::expected<void, GprError> fill_diag_into(
            const CompiledDistanceKernel<T>& kernel,
            std::size_t n,
            std::optional<MatrixRef<T>> x,
            std::span<T> out,
            std::vector<T>& scratch
        )
        {
            if (n == 0U)
            {
                return std::unexpected(GprError::empty_input());
            }
            if (auto cnt = require_count(out.size(), n, "out"); !cnt.has_value())
            {
                return cnt;
            }
            if (auto vp = kernel.impl_->validate_optional_points(x, n); !vp.has_value())
            {
                return vp;
            }
            const auto needed = checked_mul(kernel.impl_->nested_depth + 1U, n);
            if (!needed.has_value())
            {
                return std::unexpected(needed.error());
            }
            if (scratch.size() < *needed)
            {
                scratch.resize(*needed);
            }
            return kernel.impl_->eval_fill_diag(kernel.impl_->root, n, x, out, scratch, 0);
        }
    };

    // --- CompiledDistanceKernel<T> ---

    /// @brief Constructs a @c CompiledDistanceKernel from its compiled PIMPL state.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @param[in] impl Ownership handle to compiled kernel state.
    template <KernelScalarConcept T>
    CompiledDistanceKernel<T>::CompiledDistanceKernel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl))
    {
    }

    /// @brief Destroys the @c CompiledDistanceKernel instance.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    template <KernelScalarConcept T> CompiledDistanceKernel<T>::~CompiledDistanceKernel() = default;

    /// @brief Move-constructs a `CompiledDistanceKernel<T>`.
    /// @details Transfers ownership of the internal `Impl` pointer without copying leaves or slot descriptors.
    /// Defined out-of-line where `CompiledDistanceKernel<T>::Impl` is complete so the defaulted move can
    /// destroy and relocate the owning pointer.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] other Source compiled distance kernel moved from.
    template <KernelScalarConcept T>
    CompiledDistanceKernel<T>::CompiledDistanceKernel(CompiledDistanceKernel&& other) noexcept = default;

    /// @brief Move-assigns a `CompiledDistanceKernel<T>`.
    /// @details Transfers ownership of the internal `Impl` pointer; defined out-of-line where
    /// `CompiledDistanceKernel<T>::Impl` is complete.
    /// @tparam T Storage floating-point scalar type.
    /// @param[in] other Source compiled distance kernel moved from.
    /// @return Reference to `*this`.
    template <KernelScalarConcept T>
    CompiledDistanceKernel<T>& CompiledDistanceKernel<T>::operator=(CompiledDistanceKernel&& other) noexcept = default;

    /// @brief Deep-copies a @c CompiledDistanceKernel by cloning its compiled PIMPL state.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @param[in] other Compiled kernel instance to copy.
    template <KernelScalarConcept T>
    CompiledDistanceKernel<T>::CompiledDistanceKernel(const CompiledDistanceKernel& other) :
        impl_(std::make_unique<Impl>(*other.impl_))
    {
    }

    /// @brief Assigns this compiled kernel by deep-copying @p other's PIMPL state.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @param[in] other Compiled kernel instance to copy from.
    /// @return Reference to @c *this.
    template <KernelScalarConcept T>
    CompiledDistanceKernel<T>& CompiledDistanceKernel<T>::operator=(const CompiledDistanceKernel& other)
    {
        if (this != &other)
        {
            impl_ = std::make_unique<Impl>(*other.impl_);
        }
        return *this;
    }

    /// @brief Compiles this symbolic distance-kernel expression into flattened evaluation IR.
    /// @details Collects slot descriptors, assigns contiguous leaf hyperparameter bindings via pre-order traversal,
    /// computes required nested scratch depth (adding one level if any coordinate @c CustomKernel leaf needs
    /// materialized squared distances), and caches ARD lengthscale vectors.
    /// @tparam Mode Distance mode tag (@c DistanceOnly or @c WithPoints).
    /// @tparam T Target scalar evaluation precision (@c double or @c float).
    /// @return Immutable-topology @c CompiledDistanceKernel ready for workspace-reusing evaluation.
    template <DistanceModeConcept Mode>
    template <KernelScalarConcept T>
    CompiledDistanceKernel<T> DistanceKernel<Mode>::compile() const
    {
        auto impl = std::make_unique<typename CompiledDistanceKernel<T>::Impl>();
        impl->slots = slots();
        impl->num_scalar_slots = num_scalar_slots();
        impl->num_ard_slots = num_ard_slots();
        std::size_t next_param = 0;
        impl->root = CompiledDistanceKernel<T>::Impl::template compile_node<Mode>(*root_, *impl, next_param);
        // A coordinate CustomKernel materializes an n x n squared-distance matrix in the deepest nested buffer.
        bool has_custom_coord = false;
        if (impl->has_coord_points)
        {
            for (const auto& leaf : impl->leaves)
            {
                if (std::holds_alternative<CustomKernel>(leaf))
                {
                    has_custom_coord = true;
                    break;
                }
            }
        }
        impl->nested_depth =
            CompiledDistanceKernel<T>::Impl::compute_nested_depth(impl->root) + (has_custom_coord ? 1U : 0U);
        impl->refresh_cached_ls();
        return CompiledDistanceKernel<T>(std::move(impl));
    }

    /// @brief Returns the exponential evaluation mode configured on this compiled kernel.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @return Active @c KernelExp precision mode.
    template <KernelScalarConcept T> KernelExp CompiledDistanceKernel<T>::kernel_exp() const noexcept
    {
        return impl_->exp_mode;
    }

    /// @brief Sets the exponential evaluation mode applied to @c Rbf, @c Matern52, and @c Matern32 leaves.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @param[in] mode New @c KernelExp evaluation precision mode.
    template <KernelScalarConcept T> void CompiledDistanceKernel<T>::set_kernel_exp(KernelExp mode) noexcept
    {
        impl_->exp_mode = mode;
    }

    /// @brief Returns the total number of flattened hyperparameters across all kernel leaves.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @return Total hyperparameter count.
    template <KernelScalarConcept T> std::size_t CompiledDistanceKernel<T>::num_params() const noexcept
    {
        return impl_->bindings.size();
    }

    /// @brief Returns the number of flattened leaf kernels in this compiled expression.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @return Total leaf count.
    template <KernelScalarConcept T> std::size_t CompiledDistanceKernel<T>::num_leaves() const noexcept
    {
        return impl_->leaves.size();
    }

    /// @brief Reports whether evaluation requires non-null point coordinate matrices.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @return @c true when coordinate-dependent leaves are present (excluding @c Constant and @c White).
    template <KernelScalarConcept T> bool CompiledDistanceKernel<T>::requires_points() const noexcept
    {
        return impl_->has_coord_points;
    }

    /// @brief Returns the minimum @c NestedBuffers depth needed for allocation-free evaluation.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @return Required nested scratch matrix count.
    template <KernelScalarConcept T> std::size_t CompiledDistanceKernel<T>::nested_depth() const noexcept
    {
        return impl_->nested_depth;
    }

    /// @brief Allocates a reusable @c NestedBuffers scratch pool sized for matrices up to @p rows by @p cols.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @param[in] rows Maximum row capacity needed per scratch buffer.
    /// @param[in] cols Maximum column capacity needed per scratch buffer.
    /// @return Pre-sized @c NestedBuffers workspace matching @c nested_depth().
    template <KernelScalarConcept T>
    NestedBuffers<T> CompiledDistanceKernel<T>::make_nested_buffers(std::size_t rows, std::size_t cols) const
    {
        return NestedBuffers<T>(impl_->nested_depth, rows, cols);
    }

    /// @brief Copies all flattened leaf hyperparameters in @c parameter_bindings() order into @p out.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @param[out] out Destination span of length equal to @c num_params().
    /// @return Empty expected value on success, or @c GprError on length mismatch.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledDistanceKernel<T>::get_params(std::span<double> out) const
    {
        if (auto cnt = require_count(out.size(), impl_->bindings.size(), "out"); !cnt.has_value())
        {
            return cnt;
        }
        std::size_t offset = 0;
        for (const auto& leaf : impl_->leaves)
        {
            const std::size_t count = detail::leaf_num_params(leaf);
            if (auto res = detail::leaf_get_params(leaf, out.subspan(offset, count)); !res.has_value())
            {
                return res;
            }
            offset += count;
        }
        return {};
    }

    /// @brief Atomically validates and updates all leaf hyperparameters from @p params.
    /// @details Applies updates to a temporary leaf vector before committing so invalid values leave @c *this
    /// unchanged, then refreshes cached ARD lengthscales.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @param[in] params Input parameter span of length equal to @c num_params().
    /// @return Empty expected value on success, or @c GprError on length mismatch or out-of-domain parameters.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledDistanceKernel<T>::set_params(std::span<const double> params)
    {
        if (auto cnt = require_count(params.size(), impl_->bindings.size(), "params"); !cnt.has_value())
        {
            return cnt;
        }
        // Copy-validate all leaves before committing so validation errors preserve strong exception/state safety.
        std::vector<KernelLeaf> updated = impl_->leaves;
        std::size_t offset = 0;
        for (auto& leaf : updated)
        {
            const std::size_t count = detail::leaf_num_params(leaf);
            if (auto res = detail::leaf_set_params(leaf, params.subspan(offset, count)); !res.has_value())
            {
                return res;
            }
            offset += count;
        }
        impl_->leaves = std::move(updated);
        impl_->refresh_cached_ls();
        return {};
    }

    /// @brief Writes per-parameter optimization interval bounds in @c parameter_bindings() order into @p out.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @param[out] out Destination span of @c Interval bounds of length @c num_params().
    /// @return Empty expected value on success, or @c GprError on span length mismatch.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledDistanceKernel<T>::bounds_into(std::span<Interval> out) const
    {
        if (auto cnt = require_count(out.size(), impl_->bindings.size(), "out"); !cnt.has_value())
        {
            return cnt;
        }
        std::size_t offset = 0;
        for (const auto& leaf : impl_->leaves)
        {
            const std::size_t count = detail::leaf_num_params(leaf);
            if (auto res = detail::leaf_bounds_into(leaf, out.subspan(offset, count)); !res.has_value())
            {
                return res;
            }
            offset += count;
        }
        return {};
    }

    /// @brief Returns metadata binding each global hyperparameter index to its leaf and local index.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @return Const reference to the ordered parameter binding vector.
    template <KernelScalarConcept T>
    const std::vector<ParameterBinding>& CompiledDistanceKernel<T>::parameter_bindings() const noexcept
    {
        return impl_->bindings;
    }

    /// @brief Returns the supplied-distance slot descriptors required by this compiled kernel.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @return Const reference to the kernel's slot descriptors.
    template <KernelScalarConcept T>
    const std::vector<SlotDescriptor>& CompiledDistanceKernel<T>::slots() const noexcept
    {
        return impl_->slots;
    }

    /// @brief Validates and packs square training distance matrices in @p source into a @c CheckedDistanceStore.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @param[in] source Unchecked supplied-distance matrices matching this kernel's slots.
    /// @return Validated square @c CheckedDistanceStore on success, or @c GprError on validation failure.
    template <KernelScalarConcept T>
    std::expected<CheckedDistanceStore<T>, GprError> CompiledDistanceKernel<T>::prepare_square(
        const DistanceSource& source
    ) const
    {
        return DistanceStoreBuilder::build<T>(*this, source, true);
    }

    /// @brief Validates and copies rectangular cross distance matrices in @p source into a @c CheckedDistanceStore.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @param[in] source Unchecked cross supplied-distance matrices matching this kernel's slots.
    /// @return Validated rectangular @c CheckedDistanceStore on success, or @c GprError on validation failure.
    template <KernelScalarConcept T>
    std::expected<CheckedDistanceStore<T>, GprError> CompiledDistanceKernel<T>::prepare_cross(
        const DistanceSource& source
    ) const
    {
        return DistanceStoreBuilder::build<T>(*this, source, false);
    }

    /// @brief Evaluates the symmetric Gram matrix over @p uplo using caller-supplied workspace buffers.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @param[in] store Pre-validated square @c CheckedDistanceStore matching this kernel's slots.
    /// @param[in] x Optional point coordinate matrix (required when @c requires_points() is true).
    /// @param[out] out Output @c n x n Gram matrix view populated on @p uplo.
    /// @param[in] uplo Triangle selection controlling written matrix entries.
    /// @param[in,out] scratch Non-aliasing @c n x n temporary buffer matching @p out shape.
    /// @param[in,out] nested Caller-owned nested evaluation scratch pool.
    /// @return Empty expected value on success, or a structured @c GprError on shape/workspace/point mismatch.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledDistanceKernel<T>::apply(
        const CheckedDistanceStore<T>& store,
        std::optional<MatrixRef<T>> x,
        MatrixMut<T> out,
        Triangle uplo,
        MatrixMut<T> scratch,
        NestedBuffers<T>& nested
    ) const
    {
        if (auto sc = require_scratch(out.as_ref(), scratch); !sc.has_value())
        {
            return sc;
        }
        if (auto vs = impl_->validate_store(store, true, out.as_ref()); !vs.has_value())
        {
            return vs;
        }
        if (auto vp = impl_->validate_optional_points(x, store.rows()); !vp.has_value())
        {
            return vp;
        }
        return impl_->eval_term_square(impl_->root, store, x, out, uplo, scratch, nested, 0);
    }

    /// @brief Evaluates the symmetric Gram matrix over @p uplo using an internally allocated @c NestedBuffers pool.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @param[in] store Pre-validated square @c CheckedDistanceStore matching this kernel's slots.
    /// @param[in] x Optional point coordinate matrix (required when @c requires_points() is true).
    /// @param[out] out Output @c n x n Gram matrix view populated on @p uplo.
    /// @param[in] uplo Triangle selection controlling written matrix entries.
    /// @param[in,out] scratch Non-aliasing @c n x n temporary buffer matching @p out shape.
    /// @return Empty expected value on success, or a structured @c GprError on validation failure.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledDistanceKernel<T>::apply(
        const CheckedDistanceStore<T>& store,
        std::optional<MatrixRef<T>> x,
        MatrixMut<T> out,
        Triangle uplo,
        MatrixMut<T> scratch
    ) const
    {
        auto nested = make_nested_buffers(out.rows(), out.cols());
        return apply(store, x, out, uplo, scratch, nested);
    }

    /// @brief Evaluates the rectangular cross-covariance matrix using caller-supplied workspace buffers.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @param[in] store Pre-validated rectangular cross @c CheckedDistanceStore matching this kernel's slots.
    /// @param[in] x1 Optional left coordinate matrix of row count @c store.rows().
    /// @param[in] x2 Optional right coordinate matrix of row count @c store.cols().
    /// @param[out] out Output @c store.rows() x @c store.cols() matrix view receiving kernel entries.
    /// @param[in,out] scratch Non-aliasing temporary matrix matching @p out shape.
    /// @param[in,out] nested Caller-owned nested evaluation scratch pool.
    /// @return Empty expected value on success, or @c GprError on dimension, shape, or workspace mismatch.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledDistanceKernel<T>::apply_cross(
        const CheckedDistanceStore<T>& store,
        std::optional<MatrixRef<T>> x1,
        std::optional<MatrixRef<T>> x2,
        MatrixMut<T> out,
        MatrixMut<T> scratch,
        NestedBuffers<T>& nested
    ) const
    {
        if (auto sc = require_scratch(out.as_ref(), scratch); !sc.has_value())
        {
            return sc;
        }
        if (auto vs = impl_->validate_store(store, false, out.as_ref()); !vs.has_value())
        {
            return vs;
        }
        if (auto vp1 = impl_->validate_optional_points(x1, store.rows()); !vp1.has_value())
        {
            return vp1;
        }
        if (auto vp2 = impl_->validate_optional_points(x2, store.cols()); !vp2.has_value())
        {
            return vp2;
        }
        if (x1.has_value() && x2.has_value() && x1->cols() != x2->cols())
        {
            return std::unexpected(GprError::dimension_mismatch(x2->cols(), x1->cols()));
        }
        return impl_->eval_term_cross(impl_->root, store, x1, x2, out, scratch, nested, 0);
    }

    /// @brief Evaluates the rectangular cross-covariance matrix using an internally allocated @c NestedBuffers pool.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @param[in] store Pre-validated rectangular cross @c CheckedDistanceStore matching this kernel's slots.
    /// @param[in] x1 Optional left coordinate matrix of row count @c store.rows().
    /// @param[in] x2 Optional right coordinate matrix of row count @c store.cols().
    /// @param[out] out Output @c store.rows() x @c store.cols() matrix view receiving kernel entries.
    /// @param[in,out] scratch Non-aliasing temporary matrix matching @p out shape.
    /// @return Empty expected value on success, or @c GprError on validation failure.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledDistanceKernel<T>::apply_cross(
        const CheckedDistanceStore<T>& store,
        std::optional<MatrixRef<T>> x1,
        std::optional<MatrixRef<T>> x2,
        MatrixMut<T> out,
        MatrixMut<T> scratch
    ) const
    {
        auto nested = make_nested_buffers(out.rows(), out.cols());
        return apply_cross(store, x1, x2, out, scratch, nested);
    }

    /// @brief Evaluates the kernel diagonal vector $\mathrm{diag}(K(X, X))$ for @p n points.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @param[in] n Number of diagonal entries to compute (must be non-zero).
    /// @param[in] x Optional point coordinate matrix of @p n rows when coordinate leaves require it.
    /// @param[out] out Destination span of length @p n receiving self-kernel variances.
    /// @return Empty expected value on success, or @c GprError on empty input, shape mismatch, or missing points.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledDistanceKernel<T>::fill_diag(
        std::size_t n, std::optional<MatrixRef<T>> x, std::span<T> out
    ) const
    {
        std::vector<T> scratch_vec;
        return DistanceStoreBuilder::fill_diag_into(*this, n, x, out, scratch_vec);
    }

    /// @brief Computes the first log-hyperparameter derivative matrix over @p uplo using caller workspace.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @param[in] store Pre-validated square @c CheckedDistanceStore matching this kernel's slots.
    /// @param[in] x Optional point coordinate matrix (required when @c requires_points() is true).
    /// @param[out] d_k Output @c n x n derivative matrix populated on @p uplo.
    /// @param[in] param_idx Global flattened hyperparameter index to differentiate with respect to.
    /// @param[in] uplo Triangle selection controlling written matrix entries.
    /// @param[in,out] scratch_a First non-aliasing @c n x n temporary buffer.
    /// @param[in,out] scratch_b Second non-aliasing @c n x n temporary buffer distinct from @p scratch_a.
    /// @param[in,out] nested Caller-owned nested evaluation scratch pool.
    /// @return Empty expected value on success, or @c GprError on aliasing, range, or validation failure.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledDistanceKernel<T>::grad(
        const CheckedDistanceStore<T>& store,
        std::optional<MatrixRef<T>> x,
        MatrixMut<T> d_k,
        std::size_t param_idx,
        Triangle uplo,
        MatrixMut<T> scratch_a,
        MatrixMut<T> scratch_b,
        NestedBuffers<T>& nested
    ) const
    {
        if (auto sc_a = require_scratch(d_k.as_ref(), scratch_a); !sc_a.has_value())
        {
            return sc_a;
        }
        if (auto sc_b = require_scratch(d_k.as_ref(), scratch_b); !sc_b.has_value())
        {
            return sc_b;
        }
        if (scratch_a.data() == scratch_b.data())
        {
            return std::unexpected(GprError::invalid_config("scratch buffers must not alias"));
        }
        if (auto vs = impl_->validate_store(store, true, d_k.as_ref()); !vs.has_value())
        {
            return vs;
        }
        if (auto vp = impl_->validate_optional_points(x, store.rows()); !vp.has_value())
        {
            return vp;
        }
        if (param_idx >= impl_->bindings.size())
        {
            return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
        }
        return impl_->eval_grad(impl_->root, store, x, d_k, param_idx, uplo, scratch_a, scratch_b, nested, 0);
    }

    /// @brief Computes the first log-hyperparameter derivative matrix using an internally allocated nested pool.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @param[in] store Pre-validated square @c CheckedDistanceStore matching this kernel's slots.
    /// @param[in] x Optional point coordinate matrix (required when @c requires_points() is true).
    /// @param[out] d_k Output @c n x n derivative matrix populated on @p uplo.
    /// @param[in] param_idx Global flattened hyperparameter index to differentiate with respect to.
    /// @param[in] uplo Triangle selection controlling written matrix entries.
    /// @param[in,out] scratch_a First non-aliasing @c n x n temporary buffer.
    /// @param[in,out] scratch_b Second non-aliasing @c n x n temporary buffer distinct from @p scratch_a.
    /// @return Empty expected value on success, or @c GprError on validation failure.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledDistanceKernel<T>::grad(
        const CheckedDistanceStore<T>& store,
        std::optional<MatrixRef<T>> x,
        MatrixMut<T> d_k,
        std::size_t param_idx,
        Triangle uplo,
        MatrixMut<T> scratch_a,
        MatrixMut<T> scratch_b
    ) const
    {
        auto nested = make_nested_buffers(d_k.rows(), d_k.cols());
        return grad(store, x, d_k, param_idx, uplo, scratch_a, scratch_b, nested);
    }

    /// @brief Computes the second log-hyperparameter derivative matrix over @p uplo using caller workspace.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @param[in] store Pre-validated square @c CheckedDistanceStore matching this kernel's slots.
    /// @param[in] x Optional point coordinate matrix (required when @c requires_points() is true).
    /// @param[out] d2_k Output @c n x n Hessian entry matrix populated on @p uplo.
    /// @param[in] i First global flattened hyperparameter index.
    /// @param[in] j Second global flattened hyperparameter index.
    /// @param[in] uplo Triangle selection controlling written matrix entries.
    /// @param[in,out] scratch_a First non-aliasing @c n x n temporary buffer.
    /// @param[in,out] scratch_b Second non-aliasing @c n x n temporary buffer pairwise disjoint from @p scratch_a and
    /// @p scratch_c.
    /// @param[in,out] scratch_c Third non-aliasing @c n x n temporary buffer pairwise disjoint from @p scratch_a and
    /// @p scratch_b.
    /// @param[in,out] nested Caller-owned nested evaluation scratch pool.
    /// @return Empty expected value on success, or @c GprError on aliasing, range, or validation failure.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledDistanceKernel<T>::hess(
        const CheckedDistanceStore<T>& store,
        std::optional<MatrixRef<T>> x,
        MatrixMut<T> d2_k,
        std::size_t i,
        std::size_t j,
        Triangle uplo,
        MatrixMut<T> scratch_a,
        MatrixMut<T> scratch_b,
        MatrixMut<T> scratch_c,
        NestedBuffers<T>& nested
    ) const
    {
        if (auto sc_a = require_scratch(d2_k.as_ref(), scratch_a); !sc_a.has_value())
        {
            return sc_a;
        }
        if (auto sc_b = require_scratch(d2_k.as_ref(), scratch_b); !sc_b.has_value())
        {
            return sc_b;
        }
        if (auto sc_c = require_scratch(d2_k.as_ref(), scratch_c); !sc_c.has_value())
        {
            return sc_c;
        }
        if (scratch_a.data() == scratch_b.data() || scratch_a.data() == scratch_c.data()
            || scratch_b.data() == scratch_c.data())
        {
            return std::unexpected(GprError::invalid_config("scratch buffers must not alias"));
        }
        if (auto vs = impl_->validate_store(store, true, d2_k.as_ref()); !vs.has_value())
        {
            return vs;
        }
        if (auto vp = impl_->validate_optional_points(x, store.rows()); !vp.has_value())
        {
            return vp;
        }
        if (i >= impl_->bindings.size() || j >= impl_->bindings.size())
        {
            return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
        }
        return impl_->eval_hess(impl_->root, store, x, d2_k, i, j, uplo, scratch_a, scratch_b, scratch_c, nested, 0);
    }

    /// @brief Computes the second log-hyperparameter derivative matrix using an internally allocated nested pool.
    /// @tparam T Scalar evaluation type (@c double or @c float).
    /// @param[in] store Pre-validated square @c CheckedDistanceStore matching this kernel's slots.
    /// @param[in] x Optional point coordinate matrix (required when @c requires_points() is true).
    /// @param[out] d2_k Output @c n x n Hessian entry matrix populated on @p uplo.
    /// @param[in] i First global flattened hyperparameter index.
    /// @param[in] j Second global flattened hyperparameter index.
    /// @param[in] uplo Triangle selection controlling written matrix entries.
    /// @param[in,out] scratch_a First non-aliasing @c n x n temporary buffer.
    /// @param[in,out] scratch_b Second non-aliasing @c n x n temporary buffer.
    /// @param[in,out] scratch_c Third non-aliasing @c n x n temporary buffer.
    /// @return Empty expected value on success, or @c GprError on validation failure.
    template <KernelScalarConcept T>
    std::expected<void, GprError> CompiledDistanceKernel<T>::hess(
        const CheckedDistanceStore<T>& store,
        std::optional<MatrixRef<T>> x,
        MatrixMut<T> d2_k,
        std::size_t i,
        std::size_t j,
        Triangle uplo,
        MatrixMut<T> scratch_a,
        MatrixMut<T> scratch_b,
        MatrixMut<T> scratch_c
    ) const
    {
        auto nested = make_nested_buffers(d2_k.rows(), d2_k.cols());
        return hess(store, x, d2_k, i, j, uplo, scratch_a, scratch_b, scratch_c, nested);
    }

    /// @brief Friend helper that reads the validated tables of a CheckedDistanceStore for persistence.
    struct DistanceStoreExport
    {
        /// @brief Packs the lower triangle of every slot table.
        /// @tparam T Storage scalar of the store.
        /// @param[in] store Square store.
        /// @return Packed arrays in slot order.
        template <KernelScalarConcept T> static std::vector<std::vector<T>> packed(const CheckedDistanceStore<T>& store)
        {
            const auto& impl = *store.impl_;
            const std::size_t n = impl.n_rows;
            const std::size_t packed_len = n * (n + 1U) / 2U;
            std::vector<std::vector<T>> out;
            out.reserve(impl.kernel_slots.size());
            for (const auto& slot : impl.kernel_slots)
            {
                if (slot.ard_dims.has_value())
                {
                    out.push_back(impl.ard_data[slot.at]);
                    continue;
                }
                const auto& dense = impl.scalar_tables[slot.at];
                std::vector<T> packed_table(packed_len);
                std::size_t pos = 0;
                for (std::size_t j = 0; j < n; ++j)
                {
                    // Column j contributes rows j..n-1, so positions follow j*n - j*(j-1)/2 + (i-j).
                    for (std::size_t i = j; i < n; ++i)
                    {
                        packed_table[pos++] = dense[(j * n) + i];
                    }
                }
                out.push_back(std::move(packed_table));
            }
            return out;
        }
    };

    /// @brief Describes the tree and the slots of a DistanceKernel.
    /// @tparam C Distance mode.
    /// @param[in] kernel Kernel to describe.
    /// @return Structural description.
    template <DistanceModeConcept C>
    detail::DistanceKernelDesc detail::describe_distance_kernel(const DistanceKernel<C>& kernel)
    {
        return DistanceKernelAccess::describe<C>(kernel);
    }

    /// @brief Exports the validated training tables of a store as lower-triangular packed arrays, one per slot.
    /// @tparam T Storage scalar of the store.
    /// @param[in] store Square store built by CompiledDistanceKernel::prepare_square.
    /// @return Packed arrays in slot order.
    template <KernelScalarConcept T>
    std::vector<std::vector<T>> detail::export_packed_distance_tables(const CheckedDistanceStore<T>& store)
    {
        return DistanceStoreExport::packed<T>(store);
    }

    namespace
    {

        /// @brief Builds a persistence configuration error.
        /// @param[in] message Description of the problem.
        /// @return GprError of kind PersistFailed(Config).
        GprError distance_config_error(std::string_view message)
        {
            return GprError::persist_failed(PersistErrorKind::Config, std::string(message));
        }

        /// @brief Slot handles issued for one rebuilt kernel.
        struct IssuedSlots
        {
            /// @brief Scalar handles by slot ordinal (engaged where the slot is scalar).
            std::vector<std::optional<ScalarDistance>> scalar;
            /// @brief ARD handles by slot ordinal (engaged where the slot is ARD).
            std::vector<std::optional<ArdDistance>> ard;
        };

        /// @brief Maps an ARD binding failure to a configuration error.
        /// @param[in] result Result of binding an ARD leaf.
        /// @return The kernel, or a configuration error.
        std::expected<DistanceKernel<DistanceOnly>, GprError> ard_bound(
            std::expected<DistanceKernel<DistanceOnly>, GprError> result
        )
        {
            if (!result.has_value())
            {
                return std::unexpected(distance_config_error("ARD leaf does not match the slot dimensions"));
            }
            return result;
        }

        /// @brief Binds a supplied leaf to its freshly issued slot.
        /// @param[in] node Described supplied leaf.
        /// @param[in] slots Issued slot handles.
        /// @return Single-leaf DistanceOnly kernel, or GprError when the leaf does not fit the slot kind.
        std::expected<DistanceKernel<DistanceOnly>, GprError> bind_supplied(
            const detail::DistanceNodeDesc& node, const IssuedSlots& slots
        )
        {
            if (!node.leaf.has_value() || node.slot >= slots.scalar.size())
            {
                return std::unexpected(distance_config_error("distance leaf refers to a missing slot"));
            }
            const KernelLeaf& leaf = *node.leaf;
            if (const auto& sc = slots.scalar[node.slot]; sc.has_value())
            {
                if (const auto* k = std::get_if<RbfKernel>(&leaf))
                {
                    return sc->rbf(*k);
                }
                if (const auto* k = std::get_if<MaternKernel>(&leaf))
                {
                    return sc->matern(*k);
                }
                if (const auto* k = std::get_if<PeriodicKernel>(&leaf))
                {
                    return sc->periodic(*k);
                }
                if (const auto* k = std::get_if<RationalQuadraticKernel>(&leaf))
                {
                    return sc->rational_quadratic(*k);
                }
                if (const auto* k = std::get_if<CustomKernel>(&leaf))
                {
                    return sc->custom(*k);
                }
                return std::unexpected(distance_config_error("leaf kind does not fit a scalar distance slot"));
            }
            const auto& ar = slots.ard[node.slot];
            if (!ar.has_value())
            {
                return std::unexpected(distance_config_error("distance leaf refers to a missing slot"));
            }
            if (const auto* k = std::get_if<RbfArdKernel>(&leaf))
            {
                return ard_bound(ar->rbf_ard(*k));
            }
            if (const auto* k = std::get_if<MaternArdKernel>(&leaf))
            {
                return ard_bound(ar->matern_ard(*k));
            }
            if (const auto* k = std::get_if<RationalQuadraticArdKernel>(&leaf))
            {
                return ard_bound(ar->rational_quadratic_ard(*k));
            }
            return std::unexpected(distance_config_error("leaf kind does not fit an ARD distance slot"));
        }

        /// @brief Rebuilds one described leaf: a supplied leaf or a coordinate leaf.
        /// @tparam C Distance mode.
        /// @param[in] node Described leaf node.
        /// @param[in] slots Issued slot handles.
        /// @return Rebuilt single-leaf kernel, or GprError when the leaf does not fit the mode or its slot.
        template <DistanceModeConcept C>
        std::expected<DistanceKernel<C>, GprError> build_leaf(
            const detail::DistanceNodeDesc& node, const IssuedSlots& slots
        )
        {
            using Kind = detail::DistanceNodeDesc::Kind;
            if (node.kind == Kind::Supplied)
            {
                auto leaf = bind_supplied(node, slots);
                if (!leaf.has_value())
                {
                    return std::unexpected(std::move(leaf).error());
                }
                if constexpr (std::same_as<C, WithPoints>)
                {
                    return DistanceKernel<WithPoints>(*leaf);
                }
                else
                {
                    return std::move(*leaf);
                }
            }
            // The caller passes only leaves, so the remaining kind is a coordinate leaf.
            if (!node.leaf.has_value())
            {
                return std::unexpected(distance_config_error("coordinate leaf is missing"));
            }
            if constexpr (std::same_as<C, WithPoints>)
            {
                return DistanceKernel<WithPoints>(KernelSpec(*node.leaf));
            }
            else
            {
                if (const auto* c = std::get_if<ConstantKernel>(&*node.leaf))
                {
                    return DistanceKernel<DistanceOnly>(*c);
                }
                if (const auto* w = std::get_if<WhiteKernel>(&*node.leaf))
                {
                    return DistanceKernel<DistanceOnly>(*w);
                }
                return std::unexpected(
                    distance_config_error("a distance-only model cannot contain a coordinate-dependent leaf")
                );
            }
        }

        /// @brief Rebuilds one described leaf into out, in a stack frame of its own.
        /// @tparam C Distance mode.
        /// @param[in] node Described leaf node.
        /// @param[in] slots Issued slot handles.
        /// @param[out] out Destination of the rebuilt kernel.
        /// @param[out] status Set to the error when the leaf does not fit the mode or its slot.
        template <DistanceModeConcept C>
        CXXGPR_NOINLINE void build_leaf_into(
            const detail::DistanceNodeDesc& node,
            const IssuedSlots& slots,
            std::optional<DistanceKernel<C>>& out,
            std::expected<void, GprError>& status
        )
        {
            auto leaf = build_leaf<C>(node, slots);
            if (!leaf.has_value())
            {
                status = std::unexpected(std::move(leaf).error());
                return;
            }
            out.emplace(std::move(*leaf));
        }

        /// @brief Records that a described composite node lacks an operand, in a stack frame of its own.
        /// @param[out] status Set to the configuration error.
        CXXGPR_NOINLINE void fail_missing_operand(std::expected<void, GprError>& status)
        {
            status = std::unexpected(distance_config_error("composite node is missing an operand"));
        }

        /// @brief Replaces the rebuilt left subtree in lhs with its sum or product with rhs, in a stack frame of its
        /// own.
        /// @tparam C Distance mode.
        /// @param[in] sum True for a sum node, false for a product node.
        /// @param[in,out] lhs Rebuilt left subtree on entry, the composite kernel on return.
        /// @param[in] rhs Rebuilt right subtree, moved from.
        template <DistanceModeConcept C>
        CXXGPR_NOINLINE void join_built(
            bool sum, std::optional<DistanceKernel<C>>& lhs, std::optional<DistanceKernel<C>>&& rhs
        )
        {
            // The caller joins only after both subtrees were rebuilt, so both hold a value here.
            if (lhs.has_value() && rhs.has_value())
            {
                lhs = sum ? (std::move(*lhs) + std::move(*rhs)) : (std::move(*lhs) * std::move(*rhs));
            }
        }

        /// @brief Recursively rebuilds one described subtree into out.
        /// @details A nesting level keeps only the rebuilt right subtree on the stack, the left one being rebuilt
        /// straight into out: leaf binding, joining, and error construction run in frames of their own.
        /// @tparam C Distance mode.
        /// @param[in] node Described subtree.
        /// @param[in] slots Issued slot handles.
        /// @param[out] out Destination of the rebuilt kernel.
        /// @param[out] status Set to the first error: a leaf that does not fit the mode or its slot, or a composite
        /// without an operand.
        template <DistanceModeConcept C>
        void build_node_into(
            const detail::DistanceNodeDesc& node,
            const IssuedSlots& slots,
            std::optional<DistanceKernel<C>>& out,
            std::expected<void, GprError>& status
        )
        {
            using Kind = detail::DistanceNodeDesc::Kind;
            if (node.kind == Kind::Supplied || node.kind == Kind::Coord)
            {
                build_leaf_into<C>(node, slots, out, status);
                return;
            }
            if (!node.lhs || !node.rhs)
            {
                fail_missing_operand(status);
                return;
            }
            build_node_into<C>(*node.lhs, slots, out, status);
            if (!status.has_value())
            {
                return;
            }
            std::optional<DistanceKernel<C>> rhs;
            build_node_into<C>(*node.rhs, slots, rhs, status);
            if (!status.has_value())
            {
                return;
            }
            join_built<C>(node.kind == Kind::Sum, out, std::move(rhs));
        }

        /// @brief Rebuilds one described subtree.
        /// @tparam C Distance mode.
        /// @param[in] node Described subtree.
        /// @param[in] slots Issued slot handles.
        /// @return Rebuilt kernel, or GprError when a leaf does not fit the mode or its slot.
        template <DistanceModeConcept C>
        std::expected<DistanceKernel<C>, GprError> build_node(
            const detail::DistanceNodeDesc& node, const IssuedSlots& slots
        )
        {
            std::optional<DistanceKernel<C>> out;
            std::expected<void, GprError> status;
            build_node_into<C>(node, slots, out, status);
            if (out.has_value())
            {
                return std::move(*out);
            }
            return std::unexpected(std::move(status).error());
        }

    }  // namespace

    /// @brief Rebuilds a DistanceKernel from its description, issuing a fresh slot for every described slot.
    /// @tparam C Distance mode.
    /// @param[in] desc Structural description.
    /// @return The kernel, or GprError when the description is inconsistent with the mode or the slots.
    /// @retval GprErrorCode::PersistFailed When a leaf does not fit its slot or the mode.
    template <DistanceModeConcept C>
    std::expected<DistanceKernel<C>, GprError> detail::build_distance_kernel(const DistanceKernelDesc& desc)
    {
        if (!desc.root)
        {
            return std::unexpected(distance_config_error("distance kernel has no root"));
        }
        IssuedSlots slots;
        slots.scalar.resize(desc.slots.size());
        slots.ard.resize(desc.slots.size());
        for (std::size_t i = 0; i < desc.slots.size(); ++i)
        {
            if (const auto ard_dims = desc.slots[i].ard_dims; ard_dims.has_value())
            {
                auto ard = ArdDistance::create(*ard_dims);
                if (!ard.has_value())
                {
                    return std::unexpected(distance_config_error("ARD slot needs at least one dimension"));
                }
                slots.ard[i] = *ard;
            }
            else
            {
                slots.scalar[i] = ScalarDistance();
            }
        }
        auto kernel = build_node<C>(*desc.root, slots);
        if (!kernel.has_value())
        {
            return kernel;
        }
        // The rebuilt kernel must expose exactly the slots that were described, in the same order.
        const auto rebuilt = kernel->slots();
        if (rebuilt.size() != desc.slots.size())
        {
            return std::unexpected(distance_config_error("distance slots do not match the kernel tree"));
        }
        for (std::size_t i = 0; i < rebuilt.size(); ++i)
        {
            if (rebuilt[i].ard_dims != desc.slots[i].ard_dims)
            {
                return std::unexpected(distance_config_error("distance slots do not match the kernel tree"));
            }
            // The tables d2.<i> are bound to slots()[i], the i-th slot in depth-first first-appearance order, so the
            // slot numbers of the tree must appear in that canonical order; a permuted numbering would silently pair
            // each table with another leaf.
            std::optional<SlotId> issued;
            if (const auto& ar = slots.ard[i]; ar.has_value())
            {
                issued = ar->slot();
            }
            else if (const auto& sc = slots.scalar[i]; sc.has_value())
            {
                issued = sc->slot();
            }
            if (!issued.has_value() || rebuilt[i].id != *issued)
            {
                return std::unexpected(
                    distance_config_error("distance slot numbers are not in depth-first first-appearance order")
                );
            }
        }
        return kernel;
    }

    /// @brief Checks the structure of a supply; see distance_store.hpp.
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled kernel.
    /// @param[in] source Unchecked supply.
    /// @param[in] is_square True for a training supply.
    /// @return Common table shape, or GprError.
    template <KernelScalarConcept T>
    std::expected<detail::DistanceTableShape, GprError> detail::check_distance_structure(
        const CompiledDistanceKernel<T>& kernel, const DistanceSource& source, bool is_square
    )
    {
        return DistanceStoreBuilder::check_structure(kernel, source, is_square);
    }

    /// @brief Checks the ARD table counts of a supply; see distance_store.hpp.
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled kernel.
    /// @param[in] source Supply accepted by the structure check.
    /// @return Empty expected, or DimensionMismatch.
    template <KernelScalarConcept T>
    std::expected<void, GprError> detail::check_distance_ard_counts(
        const CompiledDistanceKernel<T>& kernel, const DistanceSource& source
    )
    {
        return DistanceStoreBuilder::check_ard_counts(kernel, source);
    }

    /// @brief Checks point columns against the coordinate leaves; see distance_store.hpp.
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled kernel.
    /// @param[in] cols Number of point columns.
    /// @return Empty expected, or DimensionMismatch.
    template <KernelScalarConcept T>
    std::expected<void, GprError> detail::check_distance_point_cols(
        const CompiledDistanceKernel<T>& kernel, std::size_t cols
    )
    {
        return DistanceStoreBuilder::check_point_cols(kernel, cols);
    }

    /// @brief Validates table values into a reusable store; see distance_store.hpp.
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled kernel.
    /// @param[in] source Supply to validate.
    /// @param[in] shape Shape from the structure check.
    /// @param[in] is_square True for a square store.
    /// @param[in,out] store Reusable store.
    /// @return Pointer to the refilled store, or InvalidDistance.
    template <KernelScalarConcept T>
    std::expected<const CheckedDistanceStore<T>*, GprError> detail::fill_distance_store(
        const CompiledDistanceKernel<T>& kernel,
        const DistanceSource& source,
        DistanceTableShape shape,
        bool is_square,
        std::optional<CheckedDistanceStore<T>>& store
    )
    {
        return DistanceStoreBuilder::build_into(kernel, source, shape, is_square, store);
    }

    /// @brief Validates table values into a new store; see distance_store.hpp.
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled kernel.
    /// @param[in] source Supply to validate.
    /// @param[in] shape Shape from the structure check.
    /// @param[in] is_square True for a square store.
    /// @return New store, or InvalidDistance.
    template <KernelScalarConcept T>
    std::expected<CheckedDistanceStore<T>, GprError> detail::make_distance_store(
        const CompiledDistanceKernel<T>& kernel, const DistanceSource& source, DistanceTableShape shape, bool is_square
    )
    {
        return DistanceStoreBuilder::build_shaped(kernel, source, shape, is_square);
    }

    /// @brief Evaluates the kernel diagonal with reusable scratch; see distance_store.hpp.
    /// @tparam T Storage scalar type.
    /// @param[in] kernel Compiled kernel.
    /// @param[in] n Number of diagonal entries.
    /// @param[in] x Optional point matrix.
    /// @param[out] out Destination of length n.
    /// @param[in,out] scratch Grow-only scratch.
    /// @return Empty expected, or GprError.
    template <KernelScalarConcept T>
    std::expected<void, GprError> detail::fill_distance_diag(
        const CompiledDistanceKernel<T>& kernel,
        std::size_t n,
        std::optional<MatrixRef<T>> x,
        std::span<T> out,
        std::vector<T>& scratch
    )
    {
        return DistanceStoreBuilder::fill_diag_into(kernel, n, x, out, scratch);
    }

    template class DistanceKernel<DistanceOnly>;
    template class DistanceKernel<WithPoints>;
    template CompiledDistanceKernel<double> DistanceKernel<DistanceOnly>::compile<double>() const;
    template CompiledDistanceKernel<float> DistanceKernel<DistanceOnly>::compile<float>() const;
    template CompiledDistanceKernel<double> DistanceKernel<WithPoints>::compile<double>() const;
    template CompiledDistanceKernel<float> DistanceKernel<WithPoints>::compile<float>() const;
    template detail::DistanceKernelDesc detail::describe_distance_kernel<DistanceOnly>(
        const DistanceKernel<DistanceOnly>&
    );
    template detail::DistanceKernelDesc detail::describe_distance_kernel<WithPoints>(const DistanceKernel<WithPoints>&);
    template std::expected<DistanceKernel<DistanceOnly>, GprError> detail::build_distance_kernel<DistanceOnly>(
        const detail::DistanceKernelDesc&
    );
    template std::expected<DistanceKernel<WithPoints>, GprError> detail::build_distance_kernel<WithPoints>(
        const detail::DistanceKernelDesc&
    );
    template std::vector<std::vector<double>> detail::export_packed_distance_tables<double>(
        const CheckedDistanceStore<double>&
    );
    template std::vector<std::vector<float>> detail::export_packed_distance_tables<float>(
        const CheckedDistanceStore<float>&
    );
    template class CheckedDistanceStore<double>;
    template class CheckedDistanceStore<float>;
    template class CompiledDistanceKernel<double>;
    template class CompiledDistanceKernel<float>;
    template std::expected<detail::DistanceTableShape, GprError> detail::check_distance_structure<double>(
        const CompiledDistanceKernel<double>&, const DistanceSource&, bool
    );
    template std::expected<detail::DistanceTableShape, GprError> detail::check_distance_structure<float>(
        const CompiledDistanceKernel<float>&, const DistanceSource&, bool
    );
    template std::expected<void, GprError> detail::check_distance_ard_counts<double>(
        const CompiledDistanceKernel<double>&, const DistanceSource&
    );
    template std::expected<void, GprError> detail::check_distance_ard_counts<float>(
        const CompiledDistanceKernel<float>&, const DistanceSource&
    );
    template std::expected<void, GprError> detail::check_distance_point_cols<double>(
        const CompiledDistanceKernel<double>&, std::size_t
    );
    template std::expected<void, GprError> detail::check_distance_point_cols<float>(
        const CompiledDistanceKernel<float>&, std::size_t
    );
    template std::expected<CheckedDistanceStore<double>, GprError> detail::make_distance_store<double>(
        const CompiledDistanceKernel<double>&, const DistanceSource&, DistanceTableShape, bool
    );
    template std::expected<const CheckedDistanceStore<double>*, GprError> detail::fill_distance_store<double>(
        const CompiledDistanceKernel<double>&,
        const DistanceSource&,
        DistanceTableShape,
        bool,
        std::optional<CheckedDistanceStore<double>>&
    );
    template std::expected<CheckedDistanceStore<float>, GprError> detail::make_distance_store<float>(
        const CompiledDistanceKernel<float>&, const DistanceSource&, DistanceTableShape, bool
    );
    template std::expected<const CheckedDistanceStore<float>*, GprError> detail::fill_distance_store<float>(
        const CompiledDistanceKernel<float>&,
        const DistanceSource&,
        DistanceTableShape,
        bool,
        std::optional<CheckedDistanceStore<float>>&
    );
    template std::expected<void, GprError> detail::fill_distance_diag<double>(
        const CompiledDistanceKernel<double>&,
        std::size_t,
        std::optional<MatrixRef<double>>,
        std::span<double>,
        std::vector<double>&
    );
    template std::expected<void, GprError> detail::fill_distance_diag<float>(
        const CompiledDistanceKernel<float>&,
        std::size_t,
        std::optional<MatrixRef<float>>,
        std::span<float>,
        std::vector<float>&
    );

}  // namespace cxxgpr
