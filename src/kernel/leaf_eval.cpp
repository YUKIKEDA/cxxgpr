// SPDX-License-Identifier: Apache-2.0
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

namespace cxxgpr::detail
{

    namespace
    {

        template <KernelScalarConcept T, typename Fn> void visit_triangle(std::size_t n, Triangle uplo, Fn&& fn)
        {
            const auto n_cols = static_cast<std::ptrdiff_t>(n);
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

        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> validate_square_dist(
            MatrixRef<T> dist, MatrixMut<T> out, Triangle uplo
        )
        {
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

        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> validate_cross_dist(MatrixRef<T> dist, MatrixMut<T> out)
        {
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

        template <KernelScalarConcept T> [[nodiscard]] std::expected<void, GprError> validate_points(MatrixRef<T> x)
        {
            if (x.rows() == 0 || x.cols() == 0)
            {
                return std::unexpected(GprError::empty_input());
            }
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

        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> validate_cross_points(
            MatrixRef<T> x1, MatrixRef<T> x2, MatrixMut<T> out
        )
        {
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

        template <KernelScalarConcept T>
        [[nodiscard]] T sq_dist_points(MatrixRef<T> x1, std::size_t row1, MatrixRef<T> x2, std::size_t row2) noexcept
        {
            T acc = T{0};
            for (std::size_t d = 0; d < x1.cols(); ++d)
            {
                const T diff = x1(row1, d) - x2(row2, d);
                acc += diff * diff;
            }
            return acc;
        }

        template <KernelScalarConcept T> [[nodiscard]] std::vector<T> make_square_dist_buf(MatrixRef<T> x)
        {
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

        template <KernelScalarConcept T>
        [[nodiscard]] std::vector<T> make_cross_dist_buf(MatrixRef<T> x1, MatrixRef<T> x2)
        {
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

        template <KernelScalarConcept T>
        [[nodiscard]] T dot_points(MatrixRef<T> x1, std::size_t row1, MatrixRef<T> x2, std::size_t row2) noexcept
        {
            T acc = T{0};
            for (std::size_t d = 0; d < x1.cols(); ++d)
            {
                acc += x1(row1, d) * x2(row2, d);
            }
            return acc;
        }

        template <KernelScalarConcept T>
        [[nodiscard]] T ard_scaled_sq(
            MatrixRef<T> x1, std::size_t row1, MatrixRef<T> x2, std::size_t row2, std::span<const double> ls
        ) noexcept
        {
            T acc = T{0};
            for (std::size_t d = 0; d < ls.size(); ++d)
            {
                const T ell = static_cast<T>(ls[d]);
                const T diff = (x1(row1, d) - x2(row2, d)) / ell;
                acc += diff * diff;
            }
            return acc;
        }

        template <typename KernelT, typename Fn>
        [[nodiscard]] std::expected<void, GprError> with_ard_ls(
            const KernelT& k, std::span<const double> cached_ls, Fn&& fn
        )
        {
            if (!cached_ls.empty())
            {
                return fn(cached_ls);
            }
            const auto fallback = k.lengthscales();
            return fn(std::span<const double>(fallback));
        }

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

        template <KernelScalarConcept T> [[nodiscard]] T eval_rbf_d2(T d2, T inv_two_ell2, KernelExp exp_mode) noexcept
        {
            const T clamped = d2 < T{0} ? T{0} : d2;
            return eval_kernel_exp(-clamped * inv_two_ell2, exp_mode);
        }

        template <KernelScalarConcept T> struct MaternJetVals
        {
            T val = T{0};
            T g1 = T{0};
            T iso_hess = T{0};
            T ard_cross = T{0};
        };

        template <KernelScalarConcept T>
        [[nodiscard]] MaternJetVals<T> eval_matern_jet_vals(T u, MaternNu nu, KernelExp exp_mode) noexcept
        {
            if (u <= T{0})
            {
                return {.val = T{1}, .g1 = T{0}, .iso_hess = T{0}, .ard_cross = T{0}};
            }
            if (exp_mode == KernelExp::Accurate)
            {
                const T e = std::exp(-u);
                switch (nu)
                {
                    case MaternNu::OneHalf:
                        return {.val = e, .g1 = u * e, .iso_hess = u * (u - T{1}) * e, .ard_cross = u * (u + T{1}) * e};
                    case MaternNu::ThreeHalves:
                        return {
                            .val = (T{1} + u) * e,
                            .g1 = (u * u) * e,
                            .iso_hess = (u * u) * (u - T{2}) * e,
                            .ard_cross = (u * u * u) * e
                        };
                    case MaternNu::FiveHalves:
                        return {
                            .val = (T{1} + u + ((u * u) / T{3})) * e,
                            .g1 = (((u * u) * (T{1} + u)) / T{3}) * e,
                            .iso_hess = (((u * u) * ((u * u) - (T{2} * u) - T{2})) / T{3}) * e,
                            .ard_cross = ((u * u * u * u) / T{3}) * e
                        };
                }
            }
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

        template <KernelScalarConcept T>
        [[nodiscard]] T eval_matern_grad_g1(T u, MaternNu nu, KernelExp exp_mode) noexcept
        {
            return eval_matern_jet_vals(u, nu, exp_mode).g1;
        }

        template <KernelScalarConcept T>
        [[nodiscard]] T eval_matern_iso_hess(T u, MaternNu nu, KernelExp exp_mode) noexcept
        {
            return eval_matern_jet_vals(u, nu, exp_mode).iso_hess;
        }

        template <KernelScalarConcept T>
        [[nodiscard]] T eval_matern_ard_cross_coeff(T u, MaternNu nu, KernelExp exp_mode) noexcept
        {
            return eval_matern_jet_vals(u, nu, exp_mode).ard_cross;
        }

        template <KernelScalarConcept T>
        [[nodiscard]] T eval_periodic_d2(T d2, T ell, T period, KernelExp exp_mode) noexcept
        {
            const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
            const T phi = (std::numbers::pi_v<T> * r) / period;
            const T s = std::sin(phi);
            const T z = (T{2} * s * s) / (ell * ell);
            return eval_kernel_exp(-z, exp_mode);
        }

        template <KernelScalarConcept T> [[nodiscard]] T eval_rq_d2(T d2, T ell, T alpha, KernelExp exp_mode) noexcept
        {
            const T clamped = d2 < T{0} ? T{0} : d2;
            const T w = clamped / (T{2} * alpha * ell * ell);
            return eval_kernel_exp(-alpha * std::log1p(w), exp_mode);
        }

        template <KernelScalarConcept T, typename ElemFn>
        void write_square_dist(MatrixRef<T> dist, MatrixMut<T> out, Triangle uplo, ElemFn&& elem_fn)
        {
            visit_triangle<T>(dist.rows(), uplo, [&](std::size_t row, std::size_t col) {
                out(row, col) = elem_fn(dist(row, col), row, col);
            });
        }

        template <KernelScalarConcept T, typename ElemFn>
        void write_cross_dist(MatrixRef<T> dist, MatrixMut<T> out, ElemFn&& elem_fn)
        {
            const std::size_t n_rows = dist.rows();
            const std::size_t n_cols = dist.cols();
            const auto cols_s = static_cast<std::ptrdiff_t>(n_cols);
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

        template <KernelScalarConcept T, typename ElemFn>
        void write_square_points(MatrixRef<T> x, MatrixMut<T> out, Triangle uplo, ElemFn&& elem_fn)
        {
            visit_triangle<T>(x.rows(), uplo, [&](std::size_t row, std::size_t col) {
                out(row, col) = elem_fn(row, col);
            });
        }

        template <KernelScalarConcept T, typename ElemFn>
        void write_cross_points(MatrixRef<T> x1, MatrixRef<T> x2, MatrixMut<T> out, ElemFn&& elem_fn)
        {
            const std::size_t n_rows = x1.rows();
            const std::size_t n_cols = x2.rows();
            const auto cols_s = static_cast<std::ptrdiff_t>(n_cols);
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

        [[nodiscard]] std::expected<void, GprError> require_param_index(std::size_t idx, std::size_t count)
        {
            if (idx < count)
            {
                return {};
            }
            return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
        }

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

        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> eval_leaf_dist(
            const KernelLeaf& leaf, MatrixRef<T> dist, MatrixMut<T> out, Triangle uplo, KernelExp exp_mode
        )
        {
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
                    return std::unexpected(
                        GprError::unsupported_kernel_operation(
                            "coordinate-dependent leaf cannot be evaluated from squared distances"
                        )
                    );
                }
            }, leaf
            );
        }

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
                    const T var = static_cast<T>(k.variance());
                    write_square_points(x, out, uplo, [var](std::size_t row, std::size_t col) {
                        return row == col ? var : T{0};
                    });
                    return {};
                }
            }, leaf
            );
        }

        template <KernelScalarConcept T>
        [[nodiscard]] std::expected<void, GprError> eval_leaf_cross_dist(
            const KernelLeaf& leaf, MatrixRef<T> dist, MatrixMut<T> out, KernelExp exp_mode
        )
        {
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
                    write_cross_points(x1, x2, out, [](std::size_t, std::size_t) { return T{0}; });
                    return {};
                }
            }, leaf
            );
        }

        template <KernelScalarConcept T> [[nodiscard]] T rbf_grad_d2(T d2, T ell, KernelExp exp_mode) noexcept
        {
            const T clamped = d2 < T{0} ? T{0} : d2;
            const T l2 = ell * ell;
            const T z = clamped / (T{2} * l2);
            const ExpJet<T> jet = eval_kernel_exp_jet(-z, exp_mode);
            return T{2} * z * jet.d1;
        }

        template <KernelScalarConcept T> [[nodiscard]] T rbf_hess_d2(T d2, T ell, KernelExp exp_mode) noexcept
        {
            const T clamped = d2 < T{0} ? T{0} : d2;
            const T l2 = ell * ell;
            const T z = clamped / (T{2} * l2);
            const ExpJet<T> jet = eval_kernel_exp_jet(-z, exp_mode);
            return T{4} * z * ((z * jet.d2) - jet.d1);
        }

        template <KernelScalarConcept T>
        [[nodiscard]] T periodic_grad_d2(T d2, T ell, T period, std::size_t idx, KernelExp exp_mode) noexcept
        {
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

        template <KernelScalarConcept T>
        [[nodiscard]] T periodic_hess_d2(
            T d2, T ell, T period, std::size_t i, std::size_t j, KernelExp exp_mode
        ) noexcept
        {
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

        template <KernelScalarConcept T>
        [[nodiscard]] T rq_grad_w(T w, T alpha, std::size_t idx, std::size_t alpha_idx, KernelExp exp_mode) noexcept
        {
            const T b = T{1} + w;
            const ExpJet<T> jet = eval_kernel_exp_jet(-alpha * std::log1p(w), exp_mode);
            if (idx < alpha_idx)
            {
                return ((T{2} * alpha * w) / b) * jet.d1;
            }
            return alpha * ((w / b) - std::log1p(w)) * jet.d1;
        }

        template <KernelScalarConcept T>
        [[nodiscard]] T rq_hess_iso(T w, T alpha, std::size_t i, std::size_t j, KernelExp exp_mode) noexcept
        {
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

    std::expected<void, GprError> leaf_get_params(const KernelLeaf& leaf, std::span<double> out)
    {
        return std::visit([out](const auto& k) { return k.get_params(out); }, leaf);
    }

    std::expected<void, GprError> leaf_set_params(KernelLeaf& leaf, std::span<const double> params)
    {
        return std::visit([params](auto& k) { return k.set_params(params); }, leaf);
    }

    std::expected<void, GprError> leaf_bounds_into(const KernelLeaf& leaf, std::span<Interval> out)
    {
        return std::visit([out](const auto& k) { return k.bounds_into(out); }, leaf);
    }

    template <KernelScalarConcept T>
    std::expected<void, GprError> leaf_apply_dist(
        const KernelLeaf& leaf, MatrixRef<T> dist, MatrixMut<T> out, Triangle uplo, KernelExp exp_mode
    )
    {
        return eval_leaf_dist(leaf, dist, out, uplo, exp_mode);
    }

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

    template <KernelScalarConcept T>
    std::expected<void, GprError> leaf_apply_cross_dist(
        const KernelLeaf& leaf, MatrixRef<T> dist, MatrixMut<T> out, KernelExp exp_mode
    )
    {
        return eval_leaf_cross_dist(leaf, dist, out, exp_mode);
    }

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

    template <KernelScalarConcept T>
    std::expected<void, GprError> leaf_fill_diag(
        const KernelLeaf& leaf, MatrixRef<T> x, std::span<T> out, std::span<const double> cached_ls
    )
    {
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
                std::fill(out.begin(), out.end(), T{1});
                return {};
            }
        }, leaf
        );
    }

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
                const T c = static_cast<T>(k.constant());
                write_square_dist(dist, d_k, uplo, [c](T, std::size_t, std::size_t) { return c; });
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, WhiteKernel>)
            {
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
                const T c = static_cast<T>(k.constant());
                write_square_dist(dist, d2_k, uplo, [c](T, std::size_t, std::size_t) { return c; });
                return {};
            }
            else if constexpr (std::is_same_v<LeafT, WhiteKernel>)
            {
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

        [[nodiscard]] inline std::size_t packed_lower_index(std::size_t n, std::size_t row, std::size_t col) noexcept
        {
            const std::size_t r = row >= col ? row : col;
            const std::size_t c = row >= col ? col : row;
            return ((c * ((2U * n) - c - 1U)) / 2U) + r;
        }

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

        template <KernelScalarConcept T, typename Fn>
        void write_square_packed(std::size_t n, MatrixMut<T> out, Triangle uplo, Fn&& fn) noexcept
        {
            const auto n_cols = static_cast<std::ptrdiff_t>(n);
            if (uplo == Triangle::Lower)
            {
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
