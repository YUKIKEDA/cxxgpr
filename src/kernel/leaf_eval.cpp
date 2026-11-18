// SPDX-License-Identifier: Apache-2.0
/// @file leaf_eval.cpp
/// @brief Leaf-level kernel evaluation, first derivatives, and analytical Hessians.
/// @details Implements Gram evaluation, diagonal evaluation, log-hyperparameter gradients, and second-order Hessians
/// for all built-in and custom kernel leaves over squared Euclidean distances, point coordinates, and packed ARD
/// squared differences.

#include "kernel/leaf_eval.hpp"

#include "cxxgpr/core/checked.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <utility>

#ifdef _OPENMP
#    include <omp.h>
#endif

/// @brief Internal kernel leaf evaluation and differentiation utilities.
namespace cxxgpr::detail
{

    namespace
    {

        /// @brief Visits elements of a square matrix triangle in column-parallel order.
        /// @details Distributes columns statically across OpenMP threads when the matrix size reaches 64 so each
        /// worker writes disjoint columns without synchronization.
        /// @tparam T Storage floating-point scalar type.
        /// @tparam Fn Callable invoked as `fn(row, col)` for each selected coordinate.
        /// @param n Matrix side length.
        /// @param uplo Active matrix region (`Lower`, `Upper`, or `Full`).
        /// @param fn Element visitor callback.
        template <KernelScalarConcept T, typename Fn> void visit_triangle(std::size_t n, Triangle uplo, Fn&& fn)
        {
            const auto n_cols = static_cast<std::ptrdiff_t>(n);
            // Parallelize across columns so each OpenMP worker owns disjoint column-major slices.
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 64U)
#endif
            for (std::ptrdiff_t col_s = 0; col_s < n_cols; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                const std::size_t row_begin = (uplo == Triangle::Lower) ? col : 0U;
                const std::size_t row_end = (uplo == Triangle::Upper) ? (col + 1U) : n;
                for (std::size_t row = row_begin; row < row_end; ++row)
                {
                    fn(row, col);
                }
            }
        }

        /// @brief Validates dimensions and finiteness of a square squared-distance matrix and output view.
        /// @details Enforces empty-input, squareness, shape-match, and active-triangle finiteness checks in
        /// deterministic order before leaf evaluation.
        /// @tparam T Storage floating-point scalar type.
        /// @param dist Input squared Euclidean distance matrix.
        /// @param out Target output matrix view.
        /// @param uplo Triangle of `dist` and `out` that will be read and written.
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
                for (std::size_t row = r_begin; row < r_end; ++row)
                {
                    if (!std::isfinite(dist(row, col)))
                    {
                        return std::unexpected(GprError::non_finite_input());
                    }
                }
            }
            return {};
        }

        /// @brief Validates dimensions and finiteness of a rectangular cross squared-distance matrix and output view.
        /// @details Checks non-emptiness, exact shape agreement with `out`, and finiteness of all entries in `dist`.
        /// @tparam T Storage floating-point scalar type.
        /// @param dist Input rectangular squared-distance matrix.
        /// @param out Target cross-covariance output matrix view.
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
            for (std::size_t col = 0; col < dist.cols(); ++col)
            {
                for (std::size_t row = 0; row < dist.rows(); ++row)
                {
                    if (!std::isfinite(dist(row, col)))
                    {
                        return std::unexpected(GprError::non_finite_input());
                    }
                }
            }
            return {};
        }

        /// @brief Validates that a point-coordinate matrix is non-empty and finite.
        /// @details Iterates in column-major order for contiguous memory access.
        /// @tparam T Storage floating-point scalar type.
        /// @param x Input point matrix of shape `(n_points, n_dims)`.
        /// @return Empty expected on success, or an input validation error.
        /// @retval GprErrorCode::EmptyInput If `x` has zero rows or columns.
        /// @retval GprErrorCode::NonFiniteInput If any coordinate in `x` is NaN or infinite.
        template <KernelScalarConcept T> [[nodiscard]] std::expected<void, GprError> validate_points(MatrixRef<T> x)
        {
            if (x.rows() == 0 || x.cols() == 0)
            {
                return std::unexpected(GprError::empty_input());
            }
            // Scan in column-major order matching MatrixRef storage layout.
            for (std::size_t col = 0; col < x.cols(); ++col)
            {
                for (std::size_t row = 0; row < x.rows(); ++row)
                {
                    if (!std::isfinite(x(row, col)))
                    {
                        return std::unexpected(GprError::non_finite_input());
                    }
                }
            }
            return {};
        }

        /// @brief Validates a point matrix and square output buffer for self-covariance or derivative evaluation.
        /// @details Checks non-emptiness, verifies `out` is `x.rows() x x.rows()`, and delegates finiteness checks.
        /// @tparam T Storage floating-point scalar type.
        /// @param x Input point matrix of shape `(n, d)`.
        /// @param out Target square output matrix view of shape `(n, n)`.
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
        /// @param x1 First point matrix of shape `(n1, d)`.
        /// @param x2 Second point matrix of shape `(n2, d)`.
        /// @param out Target cross-covariance matrix view of shape `(n1, n2)`.
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
        /// @param x1 First point coordinate matrix.
        /// @param row1 Row index in `x1`.
        /// @param x2 Second point coordinate matrix.
        /// @param row2 Row index in `x2`.
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
        /// @param x Input point matrix of shape `(n, d)`.
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
        /// @param x1 First point matrix of shape `(n1, d)`.
        /// @param x2 Second point matrix of shape `(n2, d)`.
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
        /// @param x1 First point coordinate matrix.
        /// @param row1 Row index in `x1`.
        /// @param x2 Second point coordinate matrix.
        /// @param row2 Row index in `x2`.
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
        /// @param x1 First point coordinate matrix.
        /// @param row1 Row index in `x1`.
        /// @param x2 Second point coordinate matrix.
        /// @param row2 Row index in `x2`.
        /// @param ls Per-dimension lengthscale vector in user units.
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

        /// @brief Invokes a callback with per-dimension lengthscales without heap allocation when cached.
        /// @details Uses `cached_ls` directly when non-empty (hot path) and falls back to `k.lengthscales()` only for
        /// standalone uncached calls.
        /// @tparam KernelT ARD kernel leaf type exposing `.lengthscales()`.
        /// @tparam Fn Callable accepting `std::span<const double>` and returning `std::expected<void, GprError>`.
        /// @param k ARD kernel leaf instance.
        /// @param cached_ls Pre-cached lengthscale span (empty if uncached).
        /// @param fn Callback consuming the lengthscale span.
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
        /// @param nu Matern smoothness parameter.
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
        /// @param nu Matern smoothness parameter.
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
        /// @param d2 Squared Euclidean distance.
        /// @param inv_two_ell2 Precomputed factor $1 / (2 \ell^2)$.
        /// @param exp_mode Exponential evaluation backend.
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
        /// @param u Scaled radius $u = \sqrt{2\nu}\, r / \ell$ (isotropic) or $\sqrt{2\nu\, s^2}$ (ARD).
        /// @param nu Matern smoothness parameter (`OneHalf`, `ThreeHalves`, or `FiveHalves`).
        /// @param exp_mode Exponential evaluation backend.
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
        /// @param u Scaled radius $u \ge 0$.
        /// @param nu Matern smoothness parameter.
        /// @param exp_mode Exponential evaluation backend.
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
        /// @param u Scaled radius $u \ge 0$.
        /// @param nu Matern smoothness parameter.
        /// @param exp_mode Exponential evaluation backend.
        /// @return First radial log-lengthscale derivative factor.
        template <KernelScalarConcept T>
        [[nodiscard]] T eval_matern_grad_g1(T u, MaternNu nu, KernelExp exp_mode) noexcept
        {
            return eval_matern_jet_vals(u, nu, exp_mode).g1;
        }

        /// @brief Evaluates the isotropic Matern second derivative $\partial^2 k / \partial (\log \ell)^2$.
        /// @details Delegates to `eval_matern_jet_vals` and returns $u k'(u) + u^2 k''(u)$.
        /// @tparam T Storage floating-point scalar type.
        /// @param u Scaled radius $u \ge 0$.
        /// @param nu Matern smoothness parameter.
        /// @param exp_mode Exponential evaluation backend.
        /// @return Isotropic Matern second log-lengthscale derivative.
        template <KernelScalarConcept T>
        [[nodiscard]] T eval_matern_iso_hess(T u, MaternNu nu, KernelExp exp_mode) noexcept
        {
            return eval_matern_jet_vals(u, nu, exp_mode).iso_hess;
        }

        /// @brief Evaluates the ARD Matern second-derivative radial cross coefficient $u^2 k''(u) - u k'(u)$.
        /// @details Multiplied by dimension weights $w_i w_j$ in `leaf_hess_points` and `leaf_hess_ard_packed`.
        /// @tparam T Storage floating-point scalar type.
        /// @param u Scaled ARD radius $u = \sqrt{2\nu\, s^2}$.
        /// @param nu Matern smoothness parameter.
        /// @param exp_mode Exponential evaluation backend.
        /// @return Radial cross coefficient for ARD Matern Hessians.
        template <KernelScalarConcept T>
        [[nodiscard]] T eval_matern_ard_cross_coeff(T u, MaternNu nu, KernelExp exp_mode) noexcept
        {
            return eval_matern_jet_vals(u, nu, exp_mode).ard_cross;
        }

        /// @brief Evaluates the isotropic periodic kernel from squared distance $d^2$.
        /// @details Computes $k = \exp(-2 \sin^2(\pi r / p) / \ell^2)$ with $r = \sqrt{\max(d^2, 0)}$.
        /// @tparam T Storage floating-point scalar type.
        /// @param d2 Squared Euclidean distance.
        /// @param ell Lengthscale parameter $\ell > 0$.
        /// @param period Period parameter $p > 0$.
        /// @param exp_mode Exponential evaluation backend.
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
        /// @param d2 Squared Euclidean distance.
        /// @param ell Lengthscale parameter $\ell > 0$.
        /// @param alpha Scale-mixture shape parameter $\alpha > 0$.
        /// @param exp_mode Exponential evaluation backend.
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
        /// @param dist Input squared-distance matrix.
        /// @param out Target square output matrix.
        /// @param uplo Active matrix triangle to fill.
        /// @param elem_fn Element evaluation functor.
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
        /// @param dist Input rectangular squared-distance matrix.
        /// @param out Target rectangular output matrix.
        /// @param elem_fn Element evaluation functor.
        template <KernelScalarConcept T, typename ElemFn>
        void write_cross_dist(MatrixRef<T> dist, MatrixMut<T> out, ElemFn&& elem_fn)
        {
            const std::size_t n_rows = dist.rows();
            const std::size_t n_cols = dist.cols();
            const auto cols_s = static_cast<std::ptrdiff_t>(n_cols);
            // Parallelize across disjoint columns to preserve column-major cache locality and avoid write races.
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n_cols >= 64U || (n_rows * n_cols) >= 4096U)
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
        /// @param x Input point matrix.
        /// @param out Target square output matrix.
        /// @param uplo Active matrix triangle to fill.
        /// @param elem_fn Element evaluation functor.
        template <KernelScalarConcept T, typename ElemFn>
        void write_square_points(MatrixRef<T> x, MatrixMut<T> out, Triangle uplo, ElemFn&& elem_fn)
        {
            visit_triangle<T>(x.rows(), uplo, [&](std::size_t row, std::size_t col) {
                out(row, col) = elem_fn(row, col);
            });
        }

        /// @brief Populates a rectangular cross output matrix from two point matrices via `elem_fn`.
        /// @details Parallelizes across columns with OpenMP when column count or matrix area crosses threshold.
        /// @tparam T Storage floating-point scalar type.
        /// @tparam ElemFn Callable invoked as `elem_fn(row, col)`.
        /// @param x1 First point matrix of shape `(n1, d)`.
        /// @param x2 Second point matrix of shape `(n2, d)`.
        /// @param out Target rectangular output matrix of shape `(n1, n2)`.
        /// @param elem_fn Element evaluation functor.
        template <KernelScalarConcept T, typename ElemFn>
        void write_cross_points(MatrixRef<T> x1, MatrixRef<T> x2, MatrixMut<T> out, ElemFn&& elem_fn)
        {
            const std::size_t n_rows = x1.rows();
            const std::size_t n_cols = x2.rows();
            const auto cols_s = static_cast<std::ptrdiff_t>(n_cols);
            // Parallelize over columns so each thread writes contiguous column-major memory.
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n_cols >= 64U || (n_rows * n_cols) >= 4096U)
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

        /// @brief Bounds-checks a local kernel hyperparameter index against `count`.
        /// @details Returns `GprErrorCode::IndexOutOfRange` when `idx >= count`.
        /// @param idx Requested local hyperparameter index.
        /// @param count Total parameter count for the leaf.
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
        /// @param x_dim Column count of the input point matrix.
        /// @param expected Lengthscale count expected by the ARD leaf.
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
        /// @param leaf Kernel leaf variant to evaluate.
        /// @param dist Square squared-distance matrix.
        /// @param out Target square Gram matrix.
        /// @param uplo Active matrix triangle to populate.
        /// @param exp_mode Exponential evaluation backend.
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
                    const T inv_2l2 = T{1} / (T{2} * ell * ell);
                    write_square_dist(dist, out, uplo, [inv_2l2, exp_mode](T d2, std::size_t, std::size_t) {
                        return eval_rbf_d2(d2, inv_2l2, exp_mode);
                    });
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, MaternKernel>)
                {
                    const T scale = matern_scale<T>(k.nu()) / static_cast<T>(k.lengthscale());
                    const MaternNu nu = k.nu();
                    write_square_dist(
                        dist, out, uplo, [scale, nu, exp_mode](T d2, std::size_t, std::size_t)
                    {
                        const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
                        return eval_matern_u(scale * r, nu, exp_mode);
                    }
                    );
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, PeriodicKernel>)
                {
                    const T ell = static_cast<T>(k.lengthscale());
                    const T period = static_cast<T>(k.period());
                    write_square_dist(dist, out, uplo, [ell, period, exp_mode](T d2, std::size_t, std::size_t) {
                        return eval_periodic_d2(d2, ell, period, exp_mode);
                    });
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, RationalQuadraticKernel>)
                {
                    const T ell = static_cast<T>(k.lengthscale());
                    const T alpha = static_cast<T>(k.alpha());
                    write_square_dist(dist, out, uplo, [ell, alpha, exp_mode](T d2, std::size_t, std::size_t) {
                        return eval_rq_d2(d2, ell, alpha, exp_mode);
                    });
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
        /// @param leaf Kernel leaf variant to evaluate.
        /// @param x Input point matrix of shape `(n, d)`.
        /// @param out Target square Gram matrix of shape `(n, n)`.
        /// @param uplo Active matrix triangle to populate.
        /// @param exp_mode Exponential evaluation backend.
        /// @param cached_ls Pre-cached ARD lengthscales, or empty to read from `leaf`.
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
                    const T inv_2l2 = T{1} / (T{2} * ell * ell);
                    write_square_points(x, out, uplo, [x, inv_2l2, exp_mode](std::size_t row, std::size_t col) {
                        return eval_rbf_d2(sq_dist_points(x, row, x, col), inv_2l2, exp_mode);
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
                        write_square_points(x, out, uplo, [x, ls, exp_mode](std::size_t row, std::size_t col) {
                            return eval_kernel_exp(-T{0.5} * ard_scaled_sq(x, row, x, col, ls), exp_mode);
                        });
                        return std::expected<void, GprError>{};
                    }
                    );
                }
                else if constexpr (std::is_same_v<LeafT, MaternKernel>)
                {
                    const T scale = matern_scale<T>(k.nu()) / static_cast<T>(k.lengthscale());
                    const MaternNu nu = k.nu();
                    write_square_points(
                        x, out, uplo, [x, scale, nu, exp_mode](std::size_t row, std::size_t col)
                    {
                        const T d2 = sq_dist_points(x, row, x, col);
                        const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
                        return eval_matern_u(scale * r, nu, exp_mode);
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
                        const T c = matern_scale_sq<T>(k.nu());
                        const MaternNu nu = k.nu();
                        write_square_points(
                            x, out, uplo, [x, ls, c, nu, exp_mode](std::size_t row, std::size_t col)
                        {
                            const T u2 = c * ard_scaled_sq(x, row, x, col, ls);
                            const T u = u2 > T{0} ? std::sqrt(u2) : T{0};
                            return eval_matern_u(u, nu, exp_mode);
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
                    write_square_points(x, out, uplo, [x, ell, period, exp_mode](std::size_t row, std::size_t col) {
                        return eval_periodic_d2(sq_dist_points(x, row, x, col), ell, period, exp_mode);
                    });
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, RationalQuadraticKernel>)
                {
                    const T ell = static_cast<T>(k.lengthscale());
                    const T alpha = static_cast<T>(k.alpha());
                    write_square_points(x, out, uplo, [x, ell, alpha, exp_mode](std::size_t row, std::size_t col) {
                        return eval_rq_d2(sq_dist_points(x, row, x, col), ell, alpha, exp_mode);
                    });
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
                        write_square_points(
                            x, out, uplo, [x, ls, alpha, exp_mode](std::size_t row, std::size_t col)
                        {
                            const T w = ard_scaled_sq(x, row, x, col, ls) / (T{2} * alpha);
                            return eval_kernel_exp(-alpha * std::log1p(w), exp_mode);
                        }
                        );
                        return std::expected<void, GprError>{};
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
                    write_square_points(x, out, uplo, [x, var](std::size_t row, std::size_t col) {
                        return var * dot_points(x, row, x, col);
                    });
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
        /// @param leaf Kernel leaf variant to evaluate.
        /// @param dist Rectangular squared-distance matrix of shape `(n1, n2)`.
        /// @param out Target cross-covariance matrix of shape `(n1, n2)`.
        /// @param exp_mode Exponential evaluation backend.
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
                    const T inv_2l2 = T{1} / (T{2} * ell * ell);
                    write_cross_dist(dist, out, [inv_2l2, exp_mode](T d2, std::size_t, std::size_t) {
                        return eval_rbf_d2(d2, inv_2l2, exp_mode);
                    });
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, MaternKernel>)
                {
                    const T scale = matern_scale<T>(k.nu()) / static_cast<T>(k.lengthscale());
                    const MaternNu nu = k.nu();
                    write_cross_dist(
                        dist, out, [scale, nu, exp_mode](T d2, std::size_t, std::size_t)
                    {
                        const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
                        return eval_matern_u(scale * r, nu, exp_mode);
                    }
                    );
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, PeriodicKernel>)
                {
                    const T ell = static_cast<T>(k.lengthscale());
                    const T period = static_cast<T>(k.period());
                    write_cross_dist(dist, out, [ell, period, exp_mode](T d2, std::size_t, std::size_t) {
                        return eval_periodic_d2(d2, ell, period, exp_mode);
                    });
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, RationalQuadraticKernel>)
                {
                    const T ell = static_cast<T>(k.lengthscale());
                    const T alpha = static_cast<T>(k.alpha());
                    write_cross_dist(dist, out, [ell, alpha, exp_mode](T d2, std::size_t, std::size_t) {
                        return eval_rq_d2(d2, ell, alpha, exp_mode);
                    });
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
        /// @param leaf Kernel leaf variant to evaluate.
        /// @param x1 First point matrix of shape `(n1, d)`.
        /// @param x2 Second point matrix of shape `(n2, d)`.
        /// @param out Target cross-covariance matrix of shape `(n1, n2)`.
        /// @param exp_mode Exponential evaluation backend.
        /// @param cached_ls Pre-cached ARD lengthscales, or empty to read from `leaf`.
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
                    const T inv_2l2 = T{1} / (T{2} * ell * ell);
                    write_cross_points(x1, x2, out, [x1, x2, inv_2l2, exp_mode](std::size_t row, std::size_t col) {
                        return eval_rbf_d2(sq_dist_points(x1, row, x2, col), inv_2l2, exp_mode);
                    });
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
                        write_cross_points(x1, x2, out, [x1, x2, ls, exp_mode](std::size_t row, std::size_t col) {
                            return eval_kernel_exp(-T{0.5} * ard_scaled_sq(x1, row, x2, col, ls), exp_mode);
                        });
                        return std::expected<void, GprError>{};
                    }
                    );
                }
                else if constexpr (std::is_same_v<LeafT, MaternKernel>)
                {
                    const T scale = matern_scale<T>(k.nu()) / static_cast<T>(k.lengthscale());
                    const MaternNu nu = k.nu();
                    write_cross_points(
                        x1, x2, out, [x1, x2, scale, nu, exp_mode](std::size_t row, std::size_t col)
                    {
                        const T d2 = sq_dist_points(x1, row, x2, col);
                        const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
                        return eval_matern_u(scale * r, nu, exp_mode);
                    }
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
                        write_cross_points(
                            x1, x2, out, [x1, x2, ls, c, nu, exp_mode](std::size_t row, std::size_t col)
                        {
                            const T u2 = c * ard_scaled_sq(x1, row, x2, col, ls);
                            const T u = u2 > T{0} ? std::sqrt(u2) : T{0};
                            return eval_matern_u(u, nu, exp_mode);
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
                    write_cross_points(x1, x2, out, [x1, x2, ell, period, exp_mode](std::size_t row, std::size_t col) {
                        return eval_periodic_d2(sq_dist_points(x1, row, x2, col), ell, period, exp_mode);
                    });
                    return {};
                }
                else if constexpr (std::is_same_v<LeafT, RationalQuadraticKernel>)
                {
                    const T ell = static_cast<T>(k.lengthscale());
                    const T alpha = static_cast<T>(k.alpha());
                    write_cross_points(x1, x2, out, [x1, x2, ell, alpha, exp_mode](std::size_t row, std::size_t col) {
                        return eval_rq_d2(sq_dist_points(x1, row, x2, col), ell, alpha, exp_mode);
                    });
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
                        write_cross_points(
                            x1, x2, out, [x1, x2, ls, alpha, exp_mode](std::size_t row, std::size_t col)
                        {
                            const T w = ard_scaled_sq(x1, row, x2, col, ls) / (T{2} * alpha);
                            return eval_kernel_exp(-alpha * std::log1p(w), exp_mode);
                        }
                        );
                        return std::expected<void, GprError>{};
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
                    write_cross_points(x1, x2, out, [x1, x2, var](std::size_t row, std::size_t col) {
                        return var * dot_points(x1, row, x2, col);
                    });
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
        /// @param d2 Squared Euclidean distance.
        /// @param ell Lengthscale parameter $\ell > 0$.
        /// @param exp_mode Exponential evaluation backend.
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
        /// @param d2 Squared Euclidean distance.
        /// @param ell Lengthscale parameter $\ell > 0$.
        /// @param exp_mode Exponential evaluation backend.
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
        /// @param d2 Squared Euclidean distance.
        /// @param ell Lengthscale parameter $\ell > 0$.
        /// @param period Period parameter $p > 0$.
        /// @param idx Local hyperparameter index (`0` for $\log \ell$, `1` for $\log p$).
        /// @param exp_mode Exponential evaluation backend.
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
        /// @param d2 Squared Euclidean distance.
        /// @param ell Lengthscale parameter $\ell > 0$.
        /// @param period Period parameter $p > 0$.
        /// @param i First local hyperparameter index (`0` or `1`).
        /// @param j Second local hyperparameter index (`0` or `1`).
        /// @param exp_mode Exponential evaluation backend.
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
        /// @param w Normalized squared distance $d^2 / (2\alpha \ell^2)$.
        /// @param alpha Scale-mixture shape parameter $\alpha > 0$.
        /// @param idx Local hyperparameter index.
        /// @param alpha_idx Index of $\log \alpha$ (`1` for isotropic RQ).
        /// @param exp_mode Exponential evaluation backend.
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
        /// @param w Normalized squared distance $d^2 / (2\alpha \ell^2)$.
        /// @param alpha Scale-mixture shape parameter $\alpha > 0$.
        /// @param i First local hyperparameter index (`0` for $\log\ell$, `1` for $\log\alpha$).
        /// @param j Second local hyperparameter index (`0` for $\log\ell$, `1` for $\log\alpha$).
        /// @param exp_mode Exponential evaluation backend.
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
    /// @param leaf Kernel leaf variant.
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
    /// @param leaf Kernel leaf variant.
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
    /// @param leaf Kernel leaf variant.
    /// @param out Destination span of length `leaf_num_params(leaf)`.
    /// @return Empty expected on success, or length mismatch error.
    /// @retval GprErrorCode::LengthMismatch If `out.size()` differs from `leaf_num_params(leaf)`.
    std::expected<void, GprError> leaf_get_params(const KernelLeaf& leaf, std::span<double> out)
    {
        return std::visit([out](const auto& k) { return k.get_params(out); }, leaf);
    }

    /// @brief Updates a leaf's hyperparameters from log-space `params`.
    /// @details Dispatches to `set_params` of the underlying leaf variant; leaves `leaf` unchanged on error.
    /// @param leaf Kernel leaf variant to update.
    /// @param params Log-space parameter span of length `leaf_num_params(leaf)`.
    /// @return Empty expected on success, or length/finitedness/bounds error.
    /// @retval GprErrorCode::LengthMismatch If `params.size()` differs from `leaf_num_params(leaf)`.
    /// @retval GprErrorCode::InvalidHyperparameter If any value lies outside its open parameter interval.
    std::expected<void, GprError> leaf_set_params(KernelLeaf& leaf, std::span<const double> params)
    {
        return std::visit([params](auto& k) { return k.set_params(params); }, leaf);
    }

    /// @brief Copies a leaf's open hyperparameter intervals into `out`.
    /// @details Dispatches to `bounds_into` of the underlying leaf variant.
    /// @param leaf Kernel leaf variant.
    /// @param out Destination span of intervals of length `leaf_num_params(leaf)`.
    /// @return Empty expected on success, or length mismatch error.
    /// @retval GprErrorCode::LengthMismatch If `out.size()` differs from `leaf_num_params(leaf)`.
    std::expected<void, GprError> leaf_bounds_into(const KernelLeaf& leaf, std::span<Interval> out)
    {
        return std::visit([out](const auto& k) { return k.bounds_into(out); }, leaf);
    }

    /// @brief Evaluates a square Gram matrix triangle for `leaf` from squared distances.
    /// @details Public wrapper around `eval_leaf_dist`.
    /// @tparam T Storage floating-point scalar type.
    /// @param leaf Kernel leaf variant.
    /// @param dist Input square squared-distance matrix.
    /// @param out Output Gram matrix view.
    /// @param uplo Active matrix triangle to fill.
    /// @param exp_mode Exponential evaluation backend.
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
    /// @param leaf Kernel leaf variant.
    /// @param x Input point matrix of shape `(n, d)`.
    /// @param out Output square Gram matrix of shape `(n, n)`.
    /// @param uplo Active matrix triangle to fill.
    /// @param exp_mode Exponential evaluation backend.
    /// @param cached_ls Pre-cached ARD lengthscales, or empty to query `leaf`.
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
    /// @param leaf Kernel leaf variant.
    /// @param dist Input rectangular squared-distance matrix.
    /// @param out Output cross-covariance matrix.
    /// @param exp_mode Exponential evaluation backend.
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
    /// @param leaf Kernel leaf variant.
    /// @param x1 First point matrix of shape `(n1, d)`.
    /// @param x2 Second point matrix of shape `(n2, d)`.
    /// @param out Output cross-covariance matrix of shape `(n1, n2)`.
    /// @param exp_mode Exponential evaluation backend.
    /// @param cached_ls Pre-cached ARD lengthscales, or empty to query `leaf`.
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
    /// @param leaf Kernel leaf variant.
    /// @param x Input point matrix of shape `(n, d)`.
    /// @param out Output diagonal span of length `n`.
    /// @param cached_ls Pre-cached ARD lengthscales, or empty to inspect `leaf`.
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
    /// @param leaf Kernel leaf variant.
    /// @param dist Square squared-distance matrix.
    /// @param d_k Output derivative matrix view.
    /// @param param_idx Local hyperparameter index within `leaf`.
    /// @param uplo Active matrix triangle to write.
    /// @param exp_mode Exponential evaluation backend.
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
    /// @param leaf Kernel leaf variant.
    /// @param x Input point matrix of shape `(n, d)`.
    /// @param d_k Output derivative matrix of shape `(n, n)`.
    /// @param param_idx Local hyperparameter index within `leaf`.
    /// @param uplo Active matrix triangle to write.
    /// @param exp_mode Exponential evaluation backend.
    /// @param cached_ls Pre-cached ARD lengthscales, or empty to query `leaf`.
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
                    // For ARD RBF, d(-0.5 * s^2)/d(log ell_p) = ((x_r1,p - x_r2,p) / ell_p)^2.
                    const T ell_p = static_cast<T>(ls[param_idx]);
                    write_square_points(
                        x, d_k, uplo, [x, ls, param_idx, ell_p, exp_mode](std::size_t row, std::size_t col)
                    {
                        const std::size_t r1 = row;
                        const std::size_t r2 = col;
                        const T s2 = ard_scaled_sq(x, r1, x, r2, ls);
                        const T diff = (x(r1, param_idx) - x(r2, param_idx)) / ell_p;
                        return (diff * diff) * eval_kernel_exp_jet(-T{0.5} * s2, exp_mode).d1;
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
                    const T ell_p = static_cast<T>(ls[param_idx]);
                    write_square_points(
                        x, d_k, uplo, [x, ls, c, nu, param_idx, ell_p, exp_mode](std::size_t row, std::size_t col)
                    {
                        const std::size_t r1 = row;
                        const std::size_t r2 = col;
                        const T u2 = c * ard_scaled_sq(x, r1, x, r2, ls);
                        if (u2 <= T{0})
                        {
                            return T{0};
                        }
                        const T u = std::sqrt(u2);
                        const T diff = (x(r1, param_idx) - x(r2, param_idx)) / ell_p;
                        const T w_p = (c * diff * diff) / u2;
                        return eval_matern_grad_g1(u, nu, exp_mode) * w_p;
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
                    write_square_points(
                        x, d_k, uplo, [x, ls, alpha, param_idx, alpha_idx, exp_mode](std::size_t row, std::size_t col)
                    {
                        const std::size_t r1 = row;
                        const std::size_t r2 = col;
                        const T s2 = ard_scaled_sq(x, r1, x, r2, ls);
                        const T w = s2 / (T{2} * alpha);
                        const T b = T{1} + w;
                        const T d1_exp = eval_kernel_exp_jet(-alpha * std::log1p(w), exp_mode).d1;
                        if (param_idx < alpha_idx)
                        {
                            const T ell_p = static_cast<T>(ls[param_idx]);
                            const T diff = (x(r1, param_idx) - x(r2, param_idx)) / ell_p;
                            return ((diff * diff) / b) * d1_exp;
                        }
                        return alpha * ((w / b) - std::log1p(w)) * d1_exp;
                    }
                    );
                    return std::expected<void, GprError>{};
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
    /// @param leaf Kernel leaf variant.
    /// @param dist Square squared-distance matrix.
    /// @param d2_k Output second-derivative matrix view.
    /// @param i First local hyperparameter index within `leaf`.
    /// @param j Second local hyperparameter index within `leaf`.
    /// @param uplo Active matrix triangle to write.
    /// @param exp_mode Exponential evaluation backend.
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
    /// @param leaf Kernel leaf variant.
    /// @param x Input point matrix of shape `(n, d)`.
    /// @param d2_k Output second-derivative matrix of shape `(n, n)`.
    /// @param i First local hyperparameter index within `leaf`.
    /// @param j Second local hyperparameter index within `leaf`.
    /// @param uplo Active matrix triangle to write.
    /// @param exp_mode Exponential evaluation backend.
    /// @param cached_ls Pre-cached ARD lengthscales, or empty to query `leaf`.
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
        /// @param n Square matrix side length.
        /// @param row Row coordinate.
        /// @param col Column coordinate.
        /// @return Linear offset into a packed lower-triangular block of length $n(n + 1)/2$.
        [[nodiscard]] inline std::size_t packed_lower_index(std::size_t n, std::size_t row, std::size_t col) noexcept
        {
            const std::size_t r = row >= col ? row : col;
            const std::size_t c = row >= col ? col : row;
            return ((c * ((2U * n) - c - 1U)) / 2U) + r;
        }

        /// @brief Computes ARD-scaled squared distance $s^2$ from packed per-dimension squared-difference tables.
        /// @details Short-circuits diagonal entries (`row == col`) to zero and accumulates
        /// `packed[dim * per_dim + p] / ell_dim^2` across all dimensions.
        /// @tparam T Storage floating-point scalar type.
        /// @param packed Packed lower-triangular squared differences laid out as `dims x per_dim`.
        /// @param n Number of points.
        /// @param per_dim Stride between consecutive feature dimension blocks, equal to $n(n + 1)/2$.
        /// @param ls Per-dimension lengthscale values in user units.
        /// @param row Row point index.
        /// @param col Column point index.
        /// @return ARD-scaled squared distance $s^2$.
        template <KernelScalarConcept T>
        [[nodiscard]] T ard_scaled_sq_packed(
            std::span<const T> packed,
            std::size_t n,
            std::size_t per_dim,
            std::span<const double> ls,
            std::size_t row,
            std::size_t col
        ) noexcept
        {
            // Short-circuit self-distance on the diagonal without reading packed storage.
            if (row == col)
            {
                return T{0};
            }
            const std::size_t p = packed_lower_index(n, row, col);
            T acc = T{0};
            for (std::size_t dim = 0; dim < ls.size(); ++dim)
            {
                const T ell = static_cast<T>(ls[dim]);
                acc += packed[(dim * per_dim) + p] / (ell * ell);
            }
            return acc;
        }

        /// @brief Evaluates and writes a square symmetric matrix triangle or full matrix in column-parallel order.
        /// @details When `uplo == Triangle::Full`, computes each strictly lower entry once and mirrors it to
        /// `(col, row)` so symmetric packed ARD kernels execute half as many transcendental evaluations.
        /// @tparam T Storage floating-point scalar type.
        /// @tparam Fn Callable invoked as `fn(row, col)` with `row >= col`.
        /// @param n Square matrix side length.
        /// @param out Target square matrix view.
        /// @param uplo Active matrix region (`Lower`, `Upper`, or `Full`).
        /// @param fn Element evaluation functor.
        template <KernelScalarConcept T, typename Fn>
        void write_square_packed(std::size_t n, MatrixMut<T> out, Triangle uplo, Fn&& fn) noexcept
        {
            const auto n_cols = static_cast<std::ptrdiff_t>(n);
            if (uplo == Triangle::Lower)
            {
                // Parallelize across columns when filling only the lower triangle.
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 64U)
#endif
                for (std::ptrdiff_t col_s = 0; col_s < n_cols; ++col_s)
                {
                    const auto col = static_cast<std::size_t>(col_s);
                    for (std::size_t row = col; row < n; ++row)
                    {
                        out(row, col) = fn(row, col);
                    }
                }
                return;
            }
            if (uplo == Triangle::Upper)
            {
                // Parallelize across columns when filling only the upper triangle.
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 64U)
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
            // For Full output, evaluate each lower-triangular pair once per column and mirror to the upper triangle.
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (n >= 64U)
#endif
            for (std::ptrdiff_t col_s = 0; col_s < n_cols; ++col_s)
            {
                const auto col = static_cast<std::size_t>(col_s);
                out(col, col) = fn(col, col);
                for (std::size_t row = col + 1U; row < n; ++row)
                {
                    const std::size_t r1 = row;
                    const std::size_t r2 = col;
                    const T val = fn(r1, r2);
                    out(r1, r2) = val;
                    out(r2, r1) = val;
                }
            }
        }

    }  // namespace

    /// @brief Evaluates an ARD kernel leaf from precomputed packed per-dimension squared differences.
    /// @details Supports `RbfArdKernel`, `MaternArdKernel`, and `RationalQuadraticArdKernel`, rejecting any other
    /// leaf variant with `GprErrorCode::UnsupportedKernelOperation`.
    /// @tparam T Storage floating-point scalar type.
    /// @param leaf Kernel leaf variant (must be an ARD leaf).
    /// @param ls Per-dimension lengthscale values in user units.
    /// @param ard_packed Packed lower-triangular squared-difference array of length `ls.size() * n * (n + 1) / 2`.
    /// @param n Number of training points.
    /// @param out Target square Gram matrix of shape `(n, n)`.
    /// @param uplo Active matrix region to write.
    /// @param exp_mode Exponential evaluation backend.
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
            write_square_packed(
                n, out, uplo, [ard_packed, n, per_dim, ls, exp_mode](std::size_t row, std::size_t col)
            {
                const T s2 = ard_scaled_sq_packed(ard_packed, n, per_dim, ls, row, col);
                return eval_kernel_exp(-T{0.5} * s2, exp_mode);
            }
            );
            return {};
        }
        if (const auto* mat = std::get_if<MaternArdKernel>(&leaf))
        {
            const T c = matern_scale_sq<T>(mat->nu());
            const MaternNu nu = mat->nu();
            write_square_packed(
                n, out, uplo, [ard_packed, n, per_dim, ls, c, nu, exp_mode](std::size_t row, std::size_t col)
            {
                const T u2 = c * ard_scaled_sq_packed(ard_packed, n, per_dim, ls, row, col);
                const T u = u2 > T{0} ? std::sqrt(u2) : T{0};
                return eval_matern_u(u, nu, exp_mode);
            }
            );
            return {};
        }
        if (const auto* rq = std::get_if<RationalQuadraticArdKernel>(&leaf))
        {
            const T alpha = static_cast<T>(rq->alpha());
            write_square_packed(
                n, out, uplo, [ard_packed, n, per_dim, ls, alpha, exp_mode](std::size_t row, std::size_t col)
            {
                const T s2 = ard_scaled_sq_packed(ard_packed, n, per_dim, ls, row, col);
                const T w = s2 / (T{2} * alpha);
                return eval_kernel_exp(-alpha * std::log1p(w), exp_mode);
            }
            );
            return {};
        }
        return std::unexpected(GprError::unsupported_kernel_operation("leaf does not consume packed ARD distances"));
    }

    /// @brief Evaluates the first log-hyperparameter derivative of an ARD leaf from packed squared differences.
    /// @details Uses precomputed per-dimension packed blocks `ard_packed` to compute $\partial K / \partial \theta_p$
    /// for `RbfArdKernel`, `MaternArdKernel`, and `RationalQuadraticArdKernel` without reading `X`.
    /// @tparam T Storage floating-point scalar type.
    /// @param leaf Kernel leaf variant (must be an ARD leaf).
    /// @param ls Per-dimension lengthscale values in user units.
    /// @param ard_packed Packed lower-triangular squared-difference array.
    /// @param n Number of training points.
    /// @param d_k Target square derivative matrix of shape `(n, n)`.
    /// @param param_idx Local hyperparameter index within `leaf`.
    /// @param uplo Active matrix region to write.
    /// @param exp_mode Exponential evaluation backend.
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
            const T ell_p = static_cast<T>(ls[param_idx]);
            const T inv_l2 = T{1} / (ell_p * ell_p);
            write_square_packed(
                n, d_k, uplo,
                [ard_packed, n, per_dim, ls, param_idx, inv_l2, exp_mode](std::size_t row, std::size_t col)
            {
                const std::size_t p = packed_lower_index(n, row, col);
                const T s2 = ard_scaled_sq_packed(ard_packed, n, per_dim, ls, row, col);
                const T sp2 = ard_packed[(param_idx * per_dim) + p] * inv_l2;
                return sp2 * eval_kernel_exp_jet(-T{0.5} * s2, exp_mode).d1;
            }
            );
            return {};
        }
        if (const auto* mat = std::get_if<MaternArdKernel>(&leaf))
        {
            // For Matern-ARD: dk/d(log ell_p) = g1(u) * (2*nu * dx_p^2 / ell_p^2) / u^2, with limit 0 at u <= 0.
            const T c = matern_scale_sq<T>(mat->nu());
            const MaternNu nu = mat->nu();
            const T ell_p = static_cast<T>(ls[param_idx]);
            const T inv_l2 = T{1} / (ell_p * ell_p);
            write_square_packed(
                n, d_k, uplo,
                [ard_packed, n, per_dim, ls, c, nu, param_idx, inv_l2, exp_mode](std::size_t row, std::size_t col)
            {
                const std::size_t p = packed_lower_index(n, row, col);
                const T u2 = c * ard_scaled_sq_packed(ard_packed, n, per_dim, ls, row, col);
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
        if (const auto* rq = std::get_if<RationalQuadraticArdKernel>(&leaf))
        {
            // For RQ-ARD: indices < ls.size() differentiate wrt log(ell_p); index == ls.size() differentiates alpha.
            const T alpha = static_cast<T>(rq->alpha());
            const std::size_t alpha_idx = ls.size();
            write_square_packed(
                n, d_k, uplo,
                [ard_packed, n, per_dim, ls, alpha, param_idx, alpha_idx, exp_mode](std::size_t row, std::size_t col)
            {
                const std::size_t p = packed_lower_index(n, row, col);
                const T s2 = ard_scaled_sq_packed(ard_packed, n, per_dim, ls, row, col);
                const T w = s2 / (T{2} * alpha);
                const T b = T{1} + w;
                const T d1_exp = eval_kernel_exp_jet(-alpha * std::log1p(w), exp_mode).d1;
                if (param_idx < alpha_idx)
                {
                    const T ell_p = static_cast<T>(ls[param_idx]);
                    const T sp2 = ard_packed[(param_idx * per_dim) + p] / (ell_p * ell_p);
                    return (sp2 / b) * d1_exp;
                }
                return alpha * ((w / b) - std::log1p(w)) * d1_exp;
            }
            );
            return {};
        }
        return std::unexpected(GprError::unsupported_kernel_operation("leaf does not consume packed ARD distances"));
    }

    /// @brief Evaluates the second log-hyperparameter derivative of an ARD leaf from packed squared differences.
    /// @details Computes $\partial^2 K / (\partial \theta_i \partial \theta_j)$ for `RbfArdKernel`, `MaternArdKernel`,
    /// and `RationalQuadraticArdKernel` (including lengthscale-alpha cross terms) using packed ARD distances.
    /// @tparam T Storage floating-point scalar type.
    /// @param leaf Kernel leaf variant (must be an ARD leaf).
    /// @param ls Per-dimension lengthscale values in user units.
    /// @param ard_packed Packed lower-triangular squared-difference array.
    /// @param n Number of training points.
    /// @param d2_k Target square second-derivative matrix of shape `(n, n)`.
    /// @param i First local hyperparameter index within `leaf`.
    /// @param j Second local hyperparameter index within `leaf`.
    /// @param uplo Active matrix region to write.
    /// @param exp_mode Exponential evaluation backend.
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
            const T inv_li2 = T{1} / static_cast<T>(ls[i] * ls[i]);
            const T inv_lj2 = T{1} / static_cast<T>(ls[j] * ls[j]);
            write_square_packed(
                n, d2_k, uplo,
                [ard_packed, n, per_dim, ls, i, j, inv_li2, inv_lj2, exp_mode](std::size_t row, std::size_t col)
            {
                const std::size_t p = packed_lower_index(n, row, col);
                const T s2 = ard_scaled_sq_packed(ard_packed, n, per_dim, ls, row, col);
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
        if (const auto* mat = std::get_if<MaternArdKernel>(&leaf))
        {
            // For Matern-ARD: ard_cross * w_i * w_j - 2 * delta_ij * g1 * w_i, returning 0 at u <= 0.
            const T c = matern_scale_sq<T>(mat->nu());
            const MaternNu nu = mat->nu();
            const T inv_li2 = T{1} / static_cast<T>(ls[i] * ls[i]);
            const T inv_lj2 = T{1} / static_cast<T>(ls[j] * ls[j]);
            write_square_packed(
                n, d2_k, uplo,
                [ard_packed, n, per_dim, ls, c, nu, i, j, inv_li2, inv_lj2, exp_mode](std::size_t row, std::size_t col)
            {
                const std::size_t p = packed_lower_index(n, row, col);
                const T u2 = c * ard_scaled_sq_packed(ard_packed, n, per_dim, ls, row, col);
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
        if (const auto* rq = std::get_if<RationalQuadraticArdKernel>(&leaf))
        {
            // For RQ-ARD: evaluate (alpha, alpha), mixed (alpha, ell_k), and (ell_i, ell_j) blocks.
            const T alpha = static_cast<T>(rq->alpha());
            const std::size_t alpha_idx = ls.size();
            write_square_packed(
                n, d2_k, uplo,
                [ard_packed, n, per_dim, ls, alpha, i, j, alpha_idx, exp_mode](std::size_t row, std::size_t col)
            {
                const std::size_t p = packed_lower_index(n, row, col);
                const T s2 = ard_scaled_sq_packed(ard_packed, n, per_dim, ls, row, col);
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
                    const T ell_ls = static_cast<T>(ls[ls_idx]);
                    const T s_ls2 = ard_packed[(ls_idx * per_dim) + p] / (ell_ls * ell_ls);
                    return (s_ls2 / b) * ((g_alpha * jet.d2) + ((w / b) * jet.d1));
                }
                const T ell_i = static_cast<T>(ls[i]);
                const T ell_j = static_cast<T>(ls[j]);
                const T si2 = ard_packed[(i * per_dim) + p] / (ell_i * ell_i);
                const T sj2 = ard_packed[(j * per_dim) + p] / (ell_j * ell_j);
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

}  // namespace cxxgpr::detail
