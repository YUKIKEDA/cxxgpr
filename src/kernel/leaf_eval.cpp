// SPDX-License-Identifier: Apache-2.0
#include "kernel/leaf_eval.hpp"

#include "cxxgpr/core/checked.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>

namespace cxxgpr::detail {

namespace {

template <KernelScalarConcept T, typename Fn>
void visit_triangle(std::size_t n, Triangle uplo, Fn&& fn) {
    switch (uplo) {
    case Triangle::Lower:
        for (std::size_t col = 0; col < n; ++col) {
            for (std::size_t row = col; row < n; ++row) {
                fn(row, col);
            }
        }
        return;
    case Triangle::Upper:
        for (std::size_t col = 0; col < n; ++col) {
            for (std::size_t row = 0; row <= col; ++row) {
                fn(row, col);
            }
        }
        return;
    case Triangle::Full:
        for (std::size_t col = 0; col < n; ++col) {
            for (std::size_t row = 0; row < n; ++row) {
                fn(row, col);
            }
        }
        return;
    }
}

template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError> validate_square_dist(MatrixRef<T> dist,
                                                                 MatrixMut<T> out, Triangle uplo) {
    if (dist.rows() == 0 || dist.cols() == 0) {
        return std::unexpected(GprError::empty_input());
    }
    if (dist.rows() != dist.cols()) {
        return std::unexpected(
            GprError::length_mismatch("distance matrix for square evaluation must be square"));
    }
    if (out.rows() != dist.rows() || out.cols() != dist.cols()) {
        return std::unexpected(GprError::length_mismatch("output matrix shape mismatch"));
    }
    const std::size_t n = dist.rows();
    bool ok = true;
    visit_triangle<T>(n, uplo, [&](std::size_t row, std::size_t col) {
        if (!std::isfinite(dist(row, col))) {
            ok = false;
        }
    });
    if (!ok) {
        return std::unexpected(GprError::non_finite_input());
    }
    return {};
}

template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError> validate_cross_dist(MatrixRef<T> dist,
                                                                MatrixMut<T> out) {
    if (dist.rows() == 0 || dist.cols() == 0) {
        return std::unexpected(GprError::empty_input());
    }
    if (out.rows() != dist.rows() || out.cols() != dist.cols()) {
        return std::unexpected(GprError::length_mismatch("output matrix shape mismatch"));
    }
    for (std::size_t col = 0; col < dist.cols(); ++col) {
        for (std::size_t row = 0; row < dist.rows(); ++row) {
            if (!std::isfinite(dist(row, col))) {
                return std::unexpected(GprError::non_finite_input());
            }
        }
    }
    return {};
}

template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError> validate_points(MatrixRef<T> x) {
    if (x.rows() == 0 || x.cols() == 0) {
        return std::unexpected(GprError::empty_input());
    }
    for (std::size_t col = 0; col < x.cols(); ++col) {
        for (std::size_t row = 0; row < x.rows(); ++row) {
            if (!std::isfinite(x(row, col))) {
                return std::unexpected(GprError::non_finite_input());
            }
        }
    }
    return {};
}

template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError> validate_square_points(MatrixRef<T> x,
                                                                   MatrixMut<T> out) {
    if (x.rows() == 0 || x.cols() == 0) {
        return std::unexpected(GprError::empty_input());
    }
    if (out.rows() != x.rows() || out.cols() != x.rows()) {
        return std::unexpected(GprError::length_mismatch("output matrix shape mismatch"));
    }
    return validate_points(x);
}

template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError> validate_cross_points(MatrixRef<T> x1, MatrixRef<T> x2,
                                                                  MatrixMut<T> out) {
    if (x1.rows() == 0 || x1.cols() == 0 || x2.rows() == 0 || x2.cols() == 0) {
        return std::unexpected(GprError::empty_input());
    }
    if (out.rows() != x1.rows() || out.cols() != x2.rows()) {
        return std::unexpected(GprError::length_mismatch("output matrix shape mismatch"));
    }
    if (x1.cols() != x2.cols()) {
        return std::unexpected(GprError::dimension_mismatch(x2.cols(), x1.cols()));
    }
    if (auto v = validate_points(x1); !v.has_value()) {
        return v;
    }
    return validate_points(x2);
}

template <KernelScalarConcept T>
[[nodiscard]] T sq_dist_points(MatrixRef<T> x1, std::size_t row1, MatrixRef<T> x2,
                               std::size_t row2) noexcept {
    T acc = T{0};
    for (std::size_t d = 0; d < x1.cols(); ++d) {
        const T diff = x1(row1, d) - x2(row2, d);
        acc += diff * diff;
    }
    return acc;
}

template <KernelScalarConcept T>
[[nodiscard]] T dot_points(MatrixRef<T> x1, std::size_t row1, MatrixRef<T> x2,
                           std::size_t row2) noexcept {
    T acc = T{0};
    for (std::size_t d = 0; d < x1.cols(); ++d) {
        acc += x1(row1, d) * x2(row2, d);
    }
    return acc;
}

template <KernelScalarConcept T>
[[nodiscard]] T ard_scaled_sq(MatrixRef<T> x1, std::size_t row1, MatrixRef<T> x2, std::size_t row2,
                              std::span<const double> ls) noexcept {
    T acc = T{0};
    for (std::size_t d = 0; d < ls.size(); ++d) {
        const T ell = static_cast<T>(ls[d]);
        const T diff = (x1(row1, d) - x2(row2, d)) / ell;
        acc += diff * diff;
    }
    return acc;
}

template <KernelScalarConcept T> [[nodiscard]] constexpr T matern_scale(MaternNu nu) noexcept {
    switch (nu) {
    case MaternNu::OneHalf:
        return T{1};
    case MaternNu::ThreeHalves:
        return std::numbers::sqrt3_v<T>;
    case MaternNu::FiveHalves:
        return static_cast<T>(2.23606797749978969641);
    }
    return T{1};
}

template <KernelScalarConcept T> [[nodiscard]] constexpr T matern_scale_sq(MaternNu nu) noexcept {
    switch (nu) {
    case MaternNu::OneHalf:
        return T{1};
    case MaternNu::ThreeHalves:
        return T{3};
    case MaternNu::FiveHalves:
        return T{5};
    }
    return T{1};
}

template <KernelScalarConcept T> [[nodiscard]] T eval_rbf_d2(T d2, T inv_two_ell2) noexcept {
    const T clamped = d2 < T{0} ? T{0} : d2;
    return std::exp(-clamped * inv_two_ell2);
}

template <KernelScalarConcept T> [[nodiscard]] T eval_matern_u(T u, MaternNu nu) noexcept {
    if (u <= T{0}) {
        return T{1};
    }
    const T e = std::exp(-u);
    switch (nu) {
    case MaternNu::OneHalf:
        return e;
    case MaternNu::ThreeHalves:
        return (T{1} + u) * e;
    case MaternNu::FiveHalves:
        return (T{1} + u + (u * u) / T{3}) * e;
    }
    return e;
}

template <KernelScalarConcept T> [[nodiscard]] T eval_matern_grad_g1(T u, MaternNu nu) noexcept {
    if (u <= T{0}) {
        return T{0};
    }
    const T e = std::exp(-u);
    switch (nu) {
    case MaternNu::OneHalf:
        return u * e;
    case MaternNu::ThreeHalves:
        return (u * u) * e;
    case MaternNu::FiveHalves:
        return (((u * u) * (T{1} + u)) / T{3}) * e;
    }
    return T{0};
}

template <KernelScalarConcept T> [[nodiscard]] T eval_matern_iso_hess(T u, MaternNu nu) noexcept {
    if (u <= T{0}) {
        return T{0};
    }
    const T e = std::exp(-u);
    switch (nu) {
    case MaternNu::OneHalf:
        return u * (u - T{1}) * e;
    case MaternNu::ThreeHalves:
        return (u * u) * (u - T{2}) * e;
    case MaternNu::FiveHalves:
        return (((u * u) * ((u * u) - (T{2} * u) - T{2})) / T{3}) * e;
    }
    return T{0};
}

template <KernelScalarConcept T>
[[nodiscard]] T eval_matern_ard_cross_coeff(T u, MaternNu nu) noexcept {
    if (u <= T{0}) {
        return T{0};
    }
    const T e = std::exp(-u);
    switch (nu) {
    case MaternNu::OneHalf:
        return u * (u + T{1}) * e;
    case MaternNu::ThreeHalves:
        return (u * u * u) * e;
    case MaternNu::FiveHalves:
        return ((u * u * u * u) / T{3}) * e;
    }
    return T{0};
}

template <KernelScalarConcept T> [[nodiscard]] T eval_periodic_d2(T d2, T ell, T period) noexcept {
    const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
    const T phi = (std::numbers::pi_v<T> * r) / period;
    const T s = std::sin(phi);
    const T z = (T{2} * s * s) / (ell * ell);
    return std::exp(-z);
}

template <KernelScalarConcept T> [[nodiscard]] T eval_rq_d2(T d2, T ell, T alpha) noexcept {
    const T clamped = d2 < T{0} ? T{0} : d2;
    const T w = clamped / (T{2} * alpha * ell * ell);
    return std::exp(-alpha * std::log1p(w));
}

template <KernelScalarConcept T, typename ElemFn>
void write_square_dist(MatrixRef<T> dist, MatrixMut<T> out, Triangle uplo, ElemFn&& elem_fn) {
    visit_triangle<T>(dist.rows(), uplo, [&](std::size_t row, std::size_t col) {
        out(row, col) = elem_fn(dist(row, col), row, col);
    });
}

template <KernelScalarConcept T, typename ElemFn>
void write_cross_dist(MatrixRef<T> dist, MatrixMut<T> out, ElemFn&& elem_fn) {
    for (std::size_t col = 0; col < dist.cols(); ++col) {
        for (std::size_t row = 0; row < dist.rows(); ++row) {
            out(row, col) = elem_fn(dist(row, col), row, col);
        }
    }
}

template <KernelScalarConcept T, typename ElemFn>
void write_square_points(MatrixRef<T> x, MatrixMut<T> out, Triangle uplo, ElemFn&& elem_fn) {
    visit_triangle<T>(x.rows(), uplo,
                      [&](std::size_t row, std::size_t col) { out(row, col) = elem_fn(row, col); });
}

template <KernelScalarConcept T, typename ElemFn>
void write_cross_points(MatrixRef<T> x1, MatrixRef<T> x2, MatrixMut<T> out, ElemFn&& elem_fn) {
    for (std::size_t col = 0; col < x2.rows(); ++col) {
        for (std::size_t row = 0; row < x1.rows(); ++row) {
            out(row, col) = elem_fn(row, col);
        }
    }
}

[[nodiscard]] std::expected<void, GprError> require_param_index(std::size_t idx,
                                                                std::size_t count) {
    if (idx < count) {
        return {};
    }
    return std::unexpected(GprError::index_out_of_range("kernel parameter index out of range"));
}

template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError> check_ard_dim(std::size_t x_dim, std::size_t expected) {
    if (x_dim != expected) {
        return std::unexpected(GprError::dimension_mismatch(x_dim, expected));
    }
    return {};
}

// --- Dispatch helpers for each operation ---

template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError>
eval_leaf_dist(const KernelLeaf& leaf, MatrixRef<T> dist, MatrixMut<T> out, Triangle uplo) {
    if (auto v = validate_square_dist(dist, out, uplo); !v.has_value()) {
        return v;
    }
    return std::visit(
        [&](const auto& k) -> std::expected<void, GprError> {
            using LeafT = std::decay_t<decltype(k)>;
            if constexpr (std::is_same_v<LeafT, RbfKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                const T inv_2l2 = T{1} / (T{2} * ell * ell);
                write_square_dist(dist, out, uplo, [inv_2l2](T d2, std::size_t, std::size_t) {
                    return eval_rbf_d2(d2, inv_2l2);
                });
                return {};
            } else if constexpr (std::is_same_v<LeafT, MaternKernel>) {
                const T scale = matern_scale<T>(k.nu()) / static_cast<T>(k.lengthscale());
                const MaternNu nu = k.nu();
                write_square_dist(dist, out, uplo, [scale, nu](T d2, std::size_t, std::size_t) {
                    const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
                    return eval_matern_u(scale * r, nu);
                });
                return {};
            } else if constexpr (std::is_same_v<LeafT, PeriodicKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                const T period = static_cast<T>(k.period());
                write_square_dist(dist, out, uplo, [ell, period](T d2, std::size_t, std::size_t) {
                    return eval_periodic_d2(d2, ell, period);
                });
                return {};
            } else if constexpr (std::is_same_v<LeafT, RationalQuadraticKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                const T alpha = static_cast<T>(k.alpha());
                write_square_dist(dist, out, uplo, [ell, alpha](T d2, std::size_t, std::size_t) {
                    return eval_rq_d2(d2, ell, alpha);
                });
                return {};
            } else if constexpr (std::is_same_v<LeafT, ConstantKernel>) {
                const T c = static_cast<T>(k.constant());
                write_square_dist(dist, out, uplo, [c](T, std::size_t, std::size_t) { return c; });
                return {};
            } else if constexpr (std::is_same_v<LeafT, WhiteKernel>) {
                const T v = static_cast<T>(k.variance());
                write_square_dist(dist, out, uplo, [v](T, std::size_t row, std::size_t col) {
                    return row == col ? v : T{0};
                });
                return {};
            } else {
                return std::unexpected(GprError::unsupported_kernel_operation(
                    "coordinate-dependent leaf cannot be evaluated from squared distances"));
            }
        },
        leaf);
}

template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError> eval_leaf_points(const KernelLeaf& leaf, MatrixRef<T> x,
                                                             MatrixMut<T> out, Triangle uplo) {
    if (auto v = validate_square_points(x, out); !v.has_value()) {
        return v;
    }
    return std::visit(
        [&](const auto& k) -> std::expected<void, GprError> {
            using LeafT = std::decay_t<decltype(k)>;
            if constexpr (std::is_same_v<LeafT, RbfKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                const T inv_2l2 = T{1} / (T{2} * ell * ell);
                write_square_points(x, out, uplo, [x, inv_2l2](std::size_t row, std::size_t col) {
                    return eval_rbf_d2(sq_dist_points(x, row, x, col), inv_2l2);
                });
                return {};
            } else if constexpr (std::is_same_v<LeafT, RbfArdKernel>) {
                const auto ls = k.lengthscales();
                if (auto d = check_ard_dim<T>(x.cols(), ls.size()); !d.has_value()) {
                    return d;
                }
                write_square_points(x, out, uplo, [x, &ls](std::size_t row, std::size_t col) {
                    return std::exp(-T{0.5} * ard_scaled_sq(x, row, x, col, ls));
                });
                return {};
            } else if constexpr (std::is_same_v<LeafT, MaternKernel>) {
                const T scale = matern_scale<T>(k.nu()) / static_cast<T>(k.lengthscale());
                const MaternNu nu = k.nu();
                write_square_points(x, out, uplo, [x, scale, nu](std::size_t row, std::size_t col) {
                    const T d2 = sq_dist_points(x, row, x, col);
                    const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
                    return eval_matern_u(scale * r, nu);
                });
                return {};
            } else if constexpr (std::is_same_v<LeafT, MaternArdKernel>) {
                const auto ls = k.lengthscales();
                if (auto d = check_ard_dim<T>(x.cols(), ls.size()); !d.has_value()) {
                    return d;
                }
                const T c = matern_scale_sq<T>(k.nu());
                const MaternNu nu = k.nu();
                write_square_points(x, out, uplo,
                                    [x, &ls, c, nu](std::size_t row, std::size_t col) {
                                        const T u2 = c * ard_scaled_sq(x, row, x, col, ls);
                                        const T u = u2 > T{0} ? std::sqrt(u2) : T{0};
                                        return eval_matern_u(u, nu);
                                    });
                return {};
            } else if constexpr (std::is_same_v<LeafT, PeriodicKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                const T period = static_cast<T>(k.period());
                write_square_points(
                    x, out, uplo, [x, ell, period](std::size_t row, std::size_t col) {
                        return eval_periodic_d2(sq_dist_points(x, row, x, col), ell, period);
                    });
                return {};
            } else if constexpr (std::is_same_v<LeafT, RationalQuadraticKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                const T alpha = static_cast<T>(k.alpha());
                write_square_points(
                    x, out, uplo, [x, ell, alpha](std::size_t row, std::size_t col) {
                        return eval_rq_d2(sq_dist_points(x, row, x, col), ell, alpha);
                    });
                return {};
            } else if constexpr (std::is_same_v<LeafT, RationalQuadraticArdKernel>) {
                const auto ls = k.lengthscales();
                if (auto d = check_ard_dim<T>(x.cols(), ls.size()); !d.has_value()) {
                    return d;
                }
                const T alpha = static_cast<T>(k.alpha());
                write_square_points(
                    x, out, uplo, [x, &ls, alpha](std::size_t row, std::size_t col) {
                        const T w = ard_scaled_sq(x, row, x, col, ls) / (T{2} * alpha);
                        return std::exp(-alpha * std::log1p(w));
                    });
                return {};
            } else if constexpr (std::is_same_v<LeafT, ConstantKernel>) {
                const T c = static_cast<T>(k.constant());
                write_square_points(x, out, uplo, [c](std::size_t, std::size_t) { return c; });
                return {};
            } else if constexpr (std::is_same_v<LeafT, LinearKernel>) {
                const T var = static_cast<T>(k.variance());
                write_square_points(x, out, uplo, [x, var](std::size_t row, std::size_t col) {
                    return var * dot_points(x, row, x, col);
                });
                return {};
            } else {
                const T var = static_cast<T>(k.variance());
                write_square_points(x, out, uplo, [var](std::size_t row, std::size_t col) {
                    return row == col ? var : T{0};
                });
                return {};
            }
        },
        leaf);
}

template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError>
eval_leaf_cross_dist(const KernelLeaf& leaf, MatrixRef<T> dist, MatrixMut<T> out) {
    if (auto v = validate_cross_dist(dist, out); !v.has_value()) {
        return v;
    }
    return std::visit(
        [&](const auto& k) -> std::expected<void, GprError> {
            using LeafT = std::decay_t<decltype(k)>;
            if constexpr (std::is_same_v<LeafT, RbfKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                const T inv_2l2 = T{1} / (T{2} * ell * ell);
                write_cross_dist(dist, out, [inv_2l2](T d2, std::size_t, std::size_t) {
                    return eval_rbf_d2(d2, inv_2l2);
                });
                return {};
            } else if constexpr (std::is_same_v<LeafT, MaternKernel>) {
                const T scale = matern_scale<T>(k.nu()) / static_cast<T>(k.lengthscale());
                const MaternNu nu = k.nu();
                write_cross_dist(dist, out, [scale, nu](T d2, std::size_t, std::size_t) {
                    const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
                    return eval_matern_u(scale * r, nu);
                });
                return {};
            } else if constexpr (std::is_same_v<LeafT, PeriodicKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                const T period = static_cast<T>(k.period());
                write_cross_dist(dist, out, [ell, period](T d2, std::size_t, std::size_t) {
                    return eval_periodic_d2(d2, ell, period);
                });
                return {};
            } else if constexpr (std::is_same_v<LeafT, RationalQuadraticKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                const T alpha = static_cast<T>(k.alpha());
                write_cross_dist(dist, out, [ell, alpha](T d2, std::size_t, std::size_t) {
                    return eval_rq_d2(d2, ell, alpha);
                });
                return {};
            } else if constexpr (std::is_same_v<LeafT, ConstantKernel>) {
                const T c = static_cast<T>(k.constant());
                write_cross_dist(dist, out, [c](T, std::size_t, std::size_t) { return c; });
                return {};
            } else if constexpr (std::is_same_v<LeafT, WhiteKernel>) {
                write_cross_dist(dist, out, [](T, std::size_t, std::size_t) { return T{0}; });
                return {};
            } else {
                return std::unexpected(GprError::unsupported_kernel_operation(
                    "coordinate-dependent leaf cannot be evaluated from squared distances"));
            }
        },
        leaf);
}

template <KernelScalarConcept T>
[[nodiscard]] std::expected<void, GprError>
eval_leaf_cross_points(const KernelLeaf& leaf, MatrixRef<T> x1, MatrixRef<T> x2, MatrixMut<T> out) {
    if (auto v = validate_cross_points(x1, x2, out); !v.has_value()) {
        return v;
    }
    return std::visit(
        [&](const auto& k) -> std::expected<void, GprError> {
            using LeafT = std::decay_t<decltype(k)>;
            if constexpr (std::is_same_v<LeafT, RbfKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                const T inv_2l2 = T{1} / (T{2} * ell * ell);
                write_cross_points(
                    x1, x2, out, [x1, x2, inv_2l2](std::size_t row, std::size_t col) {
                        return eval_rbf_d2(sq_dist_points(x1, row, x2, col), inv_2l2);
                    });
                return {};
            } else if constexpr (std::is_same_v<LeafT, RbfArdKernel>) {
                const auto ls = k.lengthscales();
                if (auto d = check_ard_dim<T>(x1.cols(), ls.size()); !d.has_value()) {
                    return d;
                }
                write_cross_points(x1, x2, out, [x1, x2, &ls](std::size_t row, std::size_t col) {
                    return std::exp(-T{0.5} * ard_scaled_sq(x1, row, x2, col, ls));
                });
                return {};
            } else if constexpr (std::is_same_v<LeafT, MaternKernel>) {
                const T scale = matern_scale<T>(k.nu()) / static_cast<T>(k.lengthscale());
                const MaternNu nu = k.nu();
                write_cross_points(x1, x2, out,
                                   [x1, x2, scale, nu](std::size_t row, std::size_t col) {
                                       const T d2 = sq_dist_points(x1, row, x2, col);
                                       const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
                                       return eval_matern_u(scale * r, nu);
                                   });
                return {};
            } else if constexpr (std::is_same_v<LeafT, MaternArdKernel>) {
                const auto ls = k.lengthscales();
                if (auto d = check_ard_dim<T>(x1.cols(), ls.size()); !d.has_value()) {
                    return d;
                }
                const T c = matern_scale_sq<T>(k.nu());
                const MaternNu nu = k.nu();
                write_cross_points(x1, x2, out,
                                   [x1, x2, &ls, c, nu](std::size_t row, std::size_t col) {
                                       const T u2 = c * ard_scaled_sq(x1, row, x2, col, ls);
                                       const T u = u2 > T{0} ? std::sqrt(u2) : T{0};
                                       return eval_matern_u(u, nu);
                                   });
                return {};
            } else if constexpr (std::is_same_v<LeafT, PeriodicKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                const T period = static_cast<T>(k.period());
                write_cross_points(
                    x1, x2, out, [x1, x2, ell, period](std::size_t row, std::size_t col) {
                        return eval_periodic_d2(sq_dist_points(x1, row, x2, col), ell, period);
                    });
                return {};
            } else if constexpr (std::is_same_v<LeafT, RationalQuadraticKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                const T alpha = static_cast<T>(k.alpha());
                write_cross_points(
                    x1, x2, out, [x1, x2, ell, alpha](std::size_t row, std::size_t col) {
                        return eval_rq_d2(sq_dist_points(x1, row, x2, col), ell, alpha);
                    });
                return {};
            } else if constexpr (std::is_same_v<LeafT, RationalQuadraticArdKernel>) {
                const auto ls = k.lengthscales();
                if (auto d = check_ard_dim<T>(x1.cols(), ls.size()); !d.has_value()) {
                    return d;
                }
                const T alpha = static_cast<T>(k.alpha());
                write_cross_points(
                    x1, x2, out, [x1, x2, &ls, alpha](std::size_t row, std::size_t col) {
                        const T w = ard_scaled_sq(x1, row, x2, col, ls) / (T{2} * alpha);
                        return std::exp(-alpha * std::log1p(w));
                    });
                return {};
            } else if constexpr (std::is_same_v<LeafT, ConstantKernel>) {
                const T c = static_cast<T>(k.constant());
                write_cross_points(x1, x2, out, [c](std::size_t, std::size_t) { return c; });
                return {};
            } else if constexpr (std::is_same_v<LeafT, LinearKernel>) {
                const T var = static_cast<T>(k.variance());
                write_cross_points(x1, x2, out, [x1, x2, var](std::size_t row, std::size_t col) {
                    return var * dot_points(x1, row, x2, col);
                });
                return {};
            } else {
                write_cross_points(x1, x2, out, [](std::size_t, std::size_t) { return T{0}; });
                return {};
            }
        },
        leaf);
}

template <KernelScalarConcept T> [[nodiscard]] T rbf_grad_d2(T d2, T ell) noexcept {
    const T clamped = d2 < T{0} ? T{0} : d2;
    const T l2 = ell * ell;
    const T z = clamped / (T{2} * l2);
    return T{2} * z * std::exp(-z);
}

template <KernelScalarConcept T> [[nodiscard]] T rbf_hess_d2(T d2, T ell) noexcept {
    const T clamped = d2 < T{0} ? T{0} : d2;
    const T l2 = ell * ell;
    const T z = clamped / (T{2} * l2);
    return T{4} * z * (z - T{1}) * std::exp(-z);
}

template <KernelScalarConcept T>
[[nodiscard]] T periodic_grad_d2(T d2, T ell, T period, std::size_t idx) noexcept {
    const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
    const T phi = (std::numbers::pi_v<T> * r) / period;
    const T s = std::sin(phi);
    const T c = std::cos(phi);
    const T l2 = ell * ell;
    const T k = std::exp(-(T{2} * s * s) / l2);
    if (idx == 0) {
        return ((T{4} * s * s) / l2) * k;
    }
    return ((T{4} * phi * s * c) / l2) * k;
}

template <KernelScalarConcept T>
[[nodiscard]] T periodic_hess_d2(T d2, T ell, T period, std::size_t i, std::size_t j) noexcept {
    const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
    const T phi = (std::numbers::pi_v<T> * r) / period;
    const T s = std::sin(phi);
    const T c = std::cos(phi);
    const T l2 = ell * ell;
    const T k = std::exp(-(T{2} * s * s) / l2);
    const T a = (T{4} * s * s) / l2;
    const T b = (T{4} * phi * s * c) / l2;
    if (i == 0 && j == 0) {
        return a * (a - T{2}) * k;
    }
    if (i != j) {
        return b * (a - T{2}) * k;
    }
    const T cos2phi = (c * c) - (s * s);
    return ((b * b) - b - ((T{4} * phi * phi * cos2phi) / l2)) * k;
}

template <KernelScalarConcept T>
[[nodiscard]] T rq_grad_w(T w, T alpha, std::size_t idx, std::size_t alpha_idx) noexcept {
    const T b = T{1} + w;
    const T k = std::exp(-alpha * std::log1p(w));
    if (idx < alpha_idx) {
        return ((T{2} * alpha * w) / b) * k;
    }
    return alpha * ((w / b) - std::log1p(w)) * k;
}

template <KernelScalarConcept T>
[[nodiscard]] T rq_hess_iso(T w, T alpha, std::size_t i, std::size_t j) noexcept {
    const T b = T{1} + w;
    const T k = std::exp(-alpha * std::log1p(w));
    const T g0 = (T{2} * alpha * w) / b;
    const T g1 = alpha * ((w / b) - std::log1p(w));
    if (i == 0 && j == 0) {
        return ((g0 * g0) - ((T{4} * alpha * w) / (b * b))) * k;
    }
    if (i != j) {
        return ((g0 * g1) + ((T{2} * alpha * w * w) / (b * b))) * k;
    }
    return ((g1 * g1) + g1 + ((alpha * w * w) / (b * b))) * k;
}

} // namespace

CoordMode leaf_coord_mode(const KernelLeaf& leaf) noexcept {
    if (std::holds_alternative<ConstantKernel>(leaf) || std::holds_alternative<WhiteKernel>(leaf)) {
        return CoordMode::Either;
    }
    if (std::holds_alternative<RbfArdKernel>(leaf) ||
        std::holds_alternative<MaternArdKernel>(leaf) ||
        std::holds_alternative<RationalQuadraticArdKernel>(leaf) ||
        std::holds_alternative<LinearKernel>(leaf)) {
        return CoordMode::Points;
    }
    return CoordMode::Dist;
}

std::size_t leaf_num_params(const KernelLeaf& leaf) noexcept {
    if (const auto* k = std::get_if<RbfArdKernel>(&leaf)) {
        return k->num_params();
    }
    if (const auto* k = std::get_if<MaternArdKernel>(&leaf)) {
        return k->num_params();
    }
    if (std::holds_alternative<PeriodicKernel>(leaf)) {
        return PeriodicKernel::num_params();
    }
    if (std::holds_alternative<RationalQuadraticKernel>(leaf)) {
        return RationalQuadraticKernel::num_params();
    }
    if (const auto* k = std::get_if<RationalQuadraticArdKernel>(&leaf)) {
        return k->num_params();
    }
    return 1U;
}

std::expected<void, GprError> leaf_get_params(const KernelLeaf& leaf, std::span<double> out) {
    return std::visit([out](const auto& k) { return k.get_params(out); }, leaf);
}

std::expected<void, GprError> leaf_set_params(KernelLeaf& leaf, std::span<const double> params) {
    return std::visit([params](auto& k) { return k.set_params(params); }, leaf);
}

std::expected<void, GprError> leaf_bounds_into(const KernelLeaf& leaf, std::span<Interval> out) {
    return std::visit([out](const auto& k) { return k.bounds_into(out); }, leaf);
}

template <KernelScalarConcept T>
std::expected<void, GprError> leaf_apply_dist(const KernelLeaf& leaf, MatrixRef<T> dist,
                                              MatrixMut<T> out, Triangle uplo) {
    return eval_leaf_dist(leaf, dist, out, uplo);
}

template <KernelScalarConcept T>
std::expected<void, GprError> leaf_apply_points(const KernelLeaf& leaf, MatrixRef<T> x,
                                                MatrixMut<T> out, Triangle uplo) {
    return eval_leaf_points(leaf, x, out, uplo);
}

template <KernelScalarConcept T>
std::expected<void, GprError> leaf_apply_cross_dist(const KernelLeaf& leaf, MatrixRef<T> dist,
                                                    MatrixMut<T> out) {
    return eval_leaf_cross_dist(leaf, dist, out);
}

template <KernelScalarConcept T>
std::expected<void, GprError> leaf_apply_cross_points(const KernelLeaf& leaf, MatrixRef<T> x1,
                                                      MatrixRef<T> x2, MatrixMut<T> out) {
    return eval_leaf_cross_points(leaf, x1, x2, out);
}

template <KernelScalarConcept T>
std::expected<void, GprError> leaf_fill_diag(const KernelLeaf& leaf, MatrixRef<T> x,
                                             std::span<T> out) {
    if (auto v = validate_points(x); !v.has_value()) {
        return v;
    }
    if (auto cnt = require_count(out.size(), x.rows(), "out"); !cnt.has_value()) {
        return std::unexpected(std::move(cnt).error());
    }
    return std::visit(
        [&](const auto& k) -> std::expected<void, GprError> {
            using LeafT = std::decay_t<decltype(k)>;
            if constexpr (std::is_same_v<LeafT, RbfArdKernel> ||
                          std::is_same_v<LeafT, MaternArdKernel> ||
                          std::is_same_v<LeafT, RationalQuadraticArdKernel>) {
                if (auto d = check_ard_dim<T>(x.cols(), k.lengthscales().size()); !d.has_value()) {
                    return d;
                }
                std::fill(out.begin(), out.end(), T{1});
                return {};
            } else if constexpr (std::is_same_v<LeafT, ConstantKernel>) {
                std::fill(out.begin(), out.end(), static_cast<T>(k.constant()));
                return {};
            } else if constexpr (std::is_same_v<LeafT, WhiteKernel>) {
                std::fill(out.begin(), out.end(), static_cast<T>(k.variance()));
                return {};
            } else if constexpr (std::is_same_v<LeafT, LinearKernel>) {
                const T var = static_cast<T>(k.variance());
                for (std::size_t row = 0; row < x.rows(); ++row) {
                    out[row] = var * dot_points(x, row, x, row);
                }
                return {};
            } else {
                std::fill(out.begin(), out.end(), T{1});
                return {};
            }
        },
        leaf);
}

template <KernelScalarConcept T>
std::expected<void, GprError> leaf_grad_dist(const KernelLeaf& leaf, MatrixRef<T> dist,
                                             MatrixMut<T> d_k, std::size_t param_idx,
                                             Triangle uplo) {
    if (auto v = validate_square_dist(dist, d_k, uplo); !v.has_value()) {
        return v;
    }
    if (auto idx_ok = require_param_index(param_idx, leaf_num_params(leaf)); !idx_ok.has_value()) {
        return idx_ok;
    }
    return std::visit(
        [&](const auto& k) -> std::expected<void, GprError> {
            using LeafT = std::decay_t<decltype(k)>;
            if constexpr (std::is_same_v<LeafT, RbfKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                write_square_dist(dist, d_k, uplo, [ell](T d2, std::size_t, std::size_t) {
                    return rbf_grad_d2(d2, ell);
                });
                return {};
            } else if constexpr (std::is_same_v<LeafT, MaternKernel>) {
                const T scale = matern_scale<T>(k.nu()) / static_cast<T>(k.lengthscale());
                const MaternNu nu = k.nu();
                write_square_dist(dist, d_k, uplo, [scale, nu](T d2, std::size_t, std::size_t) {
                    const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
                    return eval_matern_grad_g1(scale * r, nu);
                });
                return {};
            } else if constexpr (std::is_same_v<LeafT, PeriodicKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                const T period = static_cast<T>(k.period());
                write_square_dist(dist, d_k, uplo,
                                  [ell, period, param_idx](T d2, std::size_t, std::size_t) {
                                      return periodic_grad_d2(d2, ell, period, param_idx);
                                  });
                return {};
            } else if constexpr (std::is_same_v<LeafT, RationalQuadraticKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                const T alpha = static_cast<T>(k.alpha());
                write_square_dist(dist, d_k, uplo,
                                  [ell, alpha, param_idx](T d2, std::size_t, std::size_t) {
                                      const T clamped = d2 < T{0} ? T{0} : d2;
                                      const T w = clamped / (T{2} * alpha * ell * ell);
                                      return rq_grad_w(w, alpha, param_idx, 1U);
                                  });
                return {};
            } else if constexpr (std::is_same_v<LeafT, ConstantKernel>) {
                const T c = static_cast<T>(k.constant());
                write_square_dist(dist, d_k, uplo, [c](T, std::size_t, std::size_t) { return c; });
                return {};
            } else if constexpr (std::is_same_v<LeafT, WhiteKernel>) {
                const T v = static_cast<T>(k.variance());
                write_square_dist(dist, d_k, uplo, [v](T, std::size_t row, std::size_t col) {
                    return row == col ? v : T{0};
                });
                return {};
            } else {
                return std::unexpected(GprError::unsupported_kernel_operation(
                    "coordinate-dependent leaf gradient requires points"));
            }
        },
        leaf);
}

template <KernelScalarConcept T>
std::expected<void, GprError> leaf_grad_points(const KernelLeaf& leaf, MatrixRef<T> x,
                                               MatrixMut<T> d_k, std::size_t param_idx,
                                               Triangle uplo) {
    if (auto v = validate_square_points(x, d_k); !v.has_value()) {
        return v;
    }
    if (auto idx_ok = require_param_index(param_idx, leaf_num_params(leaf)); !idx_ok.has_value()) {
        return idx_ok;
    }
    return std::visit(
        [&](const auto& k) -> std::expected<void, GprError> {
            using LeafT = std::decay_t<decltype(k)>;
            if constexpr (std::is_same_v<LeafT, RbfKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                write_square_points(x, d_k, uplo, [x, ell](std::size_t row, std::size_t col) {
                    return rbf_grad_d2(sq_dist_points(x, row, x, col), ell);
                });
                return {};
            } else if constexpr (std::is_same_v<LeafT, RbfArdKernel>) {
                const auto ls = k.lengthscales();
                if (auto d = check_ard_dim<T>(x.cols(), ls.size()); !d.has_value()) {
                    return d;
                }
                const T ell_p = static_cast<T>(ls[param_idx]);
                write_square_points(
                    x, d_k, uplo, [x, &ls, param_idx, ell_p](std::size_t row, std::size_t col) {
                        const std::size_t r1 = row;
                        const std::size_t r2 = col;
                        const T s2 = ard_scaled_sq(x, r1, x, r2, ls);
                        const T diff = (x(r1, param_idx) - x(r2, param_idx)) / ell_p;
                        return (diff * diff) * std::exp(-T{0.5} * s2);
                    });
                return {};
            } else if constexpr (std::is_same_v<LeafT, MaternKernel>) {
                const T scale = matern_scale<T>(k.nu()) / static_cast<T>(k.lengthscale());
                const MaternNu nu = k.nu();
                write_square_points(x, d_k, uplo, [x, scale, nu](std::size_t row, std::size_t col) {
                    const T d2 = sq_dist_points(x, row, x, col);
                    const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
                    return eval_matern_grad_g1(scale * r, nu);
                });
                return {};
            } else if constexpr (std::is_same_v<LeafT, MaternArdKernel>) {
                const auto ls = k.lengthscales();
                if (auto d = check_ard_dim<T>(x.cols(), ls.size()); !d.has_value()) {
                    return d;
                }
                const T c = matern_scale_sq<T>(k.nu());
                const MaternNu nu = k.nu();
                const T ell_p = static_cast<T>(ls[param_idx]);
                write_square_points(
                    x, d_k, uplo,
                    [x, &ls, c, nu, param_idx, ell_p](std::size_t row, std::size_t col) {
                        const std::size_t r1 = row;
                        const std::size_t r2 = col;
                        const T u2 = c * ard_scaled_sq(x, r1, x, r2, ls);
                        if (u2 <= T{0}) {
                            return T{0};
                        }
                        const T u = std::sqrt(u2);
                        const T diff = (x(r1, param_idx) - x(r2, param_idx)) / ell_p;
                        const T w_p = (c * diff * diff) / u2;
                        return eval_matern_grad_g1(u, nu) * w_p;
                    });
                return {};
            } else if constexpr (std::is_same_v<LeafT, PeriodicKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                const T period = static_cast<T>(k.period());
                write_square_points(x, d_k, uplo,
                                    [x, ell, period, param_idx](std::size_t row, std::size_t col) {
                                        return periodic_grad_d2(sq_dist_points(x, row, x, col), ell,
                                                                period, param_idx);
                                    });
                return {};
            } else if constexpr (std::is_same_v<LeafT, RationalQuadraticKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                const T alpha = static_cast<T>(k.alpha());
                write_square_points(
                    x, d_k, uplo, [x, ell, alpha, param_idx](std::size_t row, std::size_t col) {
                        const T w = sq_dist_points(x, row, x, col) / (T{2} * alpha * ell * ell);
                        return rq_grad_w(w, alpha, param_idx, 1U);
                    });
                return {};
            } else if constexpr (std::is_same_v<LeafT, RationalQuadraticArdKernel>) {
                const auto ls = k.lengthscales();
                if (auto d = check_ard_dim<T>(x.cols(), ls.size()); !d.has_value()) {
                    return d;
                }
                const T alpha = static_cast<T>(k.alpha());
                const std::size_t alpha_idx = ls.size();
                write_square_points(
                    x, d_k, uplo,
                    [x, &ls, alpha, param_idx, alpha_idx](std::size_t row, std::size_t col) {
                        const std::size_t r1 = row;
                        const std::size_t r2 = col;
                        const T s2 = ard_scaled_sq(x, r1, x, r2, ls);
                        const T w = s2 / (T{2} * alpha);
                        const T b = T{1} + w;
                        const T k_val = std::exp(-alpha * std::log1p(w));
                        if (param_idx < alpha_idx) {
                            const T ell_p = static_cast<T>(ls[param_idx]);
                            const T diff = (x(r1, param_idx) - x(r2, param_idx)) / ell_p;
                            return ((diff * diff) / b) * k_val;
                        }
                        return alpha * ((w / b) - std::log1p(w)) * k_val;
                    });
                return {};
            } else if constexpr (std::is_same_v<LeafT, ConstantKernel>) {
                const T c = static_cast<T>(k.constant());
                write_square_points(x, d_k, uplo, [c](std::size_t, std::size_t) { return c; });
                return {};
            } else if constexpr (std::is_same_v<LeafT, LinearKernel>) {
                const T var = static_cast<T>(k.variance());
                write_square_points(x, d_k, uplo, [x, var](std::size_t row, std::size_t col) {
                    return var * dot_points(x, row, x, col);
                });
                return {};
            } else {
                const T var = static_cast<T>(k.variance());
                write_square_points(x, d_k, uplo, [var](std::size_t row, std::size_t col) {
                    return row == col ? var : T{0};
                });
                return {};
            }
        },
        leaf);
}

template <KernelScalarConcept T>
std::expected<void, GprError> leaf_hess_dist(const KernelLeaf& leaf, MatrixRef<T> dist,
                                             MatrixMut<T> d2_k, std::size_t i, std::size_t j,
                                             Triangle uplo) {
    if (auto v = validate_square_dist(dist, d2_k, uplo); !v.has_value()) {
        return v;
    }
    const std::size_t np = leaf_num_params(leaf);
    if (auto idx_i = require_param_index(i, np); !idx_i.has_value()) {
        return idx_i;
    }
    if (auto idx_j = require_param_index(j, np); !idx_j.has_value()) {
        return idx_j;
    }
    return std::visit(
        [&](const auto& k) -> std::expected<void, GprError> {
            using LeafT = std::decay_t<decltype(k)>;
            if constexpr (std::is_same_v<LeafT, RbfKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                write_square_dist(dist, d2_k, uplo, [ell](T d2, std::size_t, std::size_t) {
                    return rbf_hess_d2(d2, ell);
                });
                return {};
            } else if constexpr (std::is_same_v<LeafT, MaternKernel>) {
                const T scale = matern_scale<T>(k.nu()) / static_cast<T>(k.lengthscale());
                const MaternNu nu = k.nu();
                write_square_dist(dist, d2_k, uplo, [scale, nu](T d2, std::size_t, std::size_t) {
                    const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
                    return eval_matern_iso_hess(scale * r, nu);
                });
                return {};
            } else if constexpr (std::is_same_v<LeafT, PeriodicKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                const T period = static_cast<T>(k.period());
                write_square_dist(dist, d2_k, uplo,
                                  [ell, period, i, j](T d2, std::size_t, std::size_t) {
                                      return periodic_hess_d2(d2, ell, period, i, j);
                                  });
                return {};
            } else if constexpr (std::is_same_v<LeafT, RationalQuadraticKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                const T alpha = static_cast<T>(k.alpha());
                write_square_dist(dist, d2_k, uplo,
                                  [ell, alpha, i, j](T d2, std::size_t, std::size_t) {
                                      const T clamped = d2 < T{0} ? T{0} : d2;
                                      const T w = clamped / (T{2} * alpha * ell * ell);
                                      return rq_hess_iso(w, alpha, i, j);
                                  });
                return {};
            } else if constexpr (std::is_same_v<LeafT, ConstantKernel>) {
                const T c = static_cast<T>(k.constant());
                write_square_dist(dist, d2_k, uplo, [c](T, std::size_t, std::size_t) { return c; });
                return {};
            } else if constexpr (std::is_same_v<LeafT, WhiteKernel>) {
                const T v = static_cast<T>(k.variance());
                write_square_dist(dist, d2_k, uplo, [v](T, std::size_t row, std::size_t col) {
                    return row == col ? v : T{0};
                });
                return {};
            } else {
                return std::unexpected(GprError::unsupported_kernel_operation(
                    "coordinate-dependent leaf Hessian requires points"));
            }
        },
        leaf);
}

template <KernelScalarConcept T>
std::expected<void, GprError> leaf_hess_points(const KernelLeaf& leaf, MatrixRef<T> x,
                                               MatrixMut<T> d2_k, std::size_t i, std::size_t j,
                                               Triangle uplo) {
    if (auto v = validate_square_points(x, d2_k); !v.has_value()) {
        return v;
    }
    const std::size_t np = leaf_num_params(leaf);
    if (auto idx_i = require_param_index(i, np); !idx_i.has_value()) {
        return idx_i;
    }
    if (auto idx_j = require_param_index(j, np); !idx_j.has_value()) {
        return idx_j;
    }
    return std::visit(
        [&](const auto& k) -> std::expected<void, GprError> {
            using LeafT = std::decay_t<decltype(k)>;
            if constexpr (std::is_same_v<LeafT, RbfKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                write_square_points(x, d2_k, uplo, [x, ell](std::size_t row, std::size_t col) {
                    return rbf_hess_d2(sq_dist_points(x, row, x, col), ell);
                });
                return {};
            } else if constexpr (std::is_same_v<LeafT, RbfArdKernel>) {
                const auto ls = k.lengthscales();
                if (auto d = check_ard_dim<T>(x.cols(), ls.size()); !d.has_value()) {
                    return d;
                }
                const T ell_i = static_cast<T>(ls[i]);
                const T ell_j = static_cast<T>(ls[j]);
                write_square_points(x, d2_k, uplo,
                                    [x, &ls, i, j, ell_i, ell_j](std::size_t row, std::size_t col) {
                                        const T s2 = ard_scaled_sq(x, row, x, col, ls);
                                        const T kv = std::exp(-T{0.5} * s2);
                                        const T di = (x(row, i) - x(col, i)) / ell_i;
                                        const T si2 = di * di;
                                        if (i == j) {
                                            return si2 * (si2 - T{2}) * kv;
                                        }
                                        const T dj = (x(row, j) - x(col, j)) / ell_j;
                                        return si2 * (dj * dj) * kv;
                                    });
                return {};
            } else if constexpr (std::is_same_v<LeafT, MaternKernel>) {
                const T scale = matern_scale<T>(k.nu()) / static_cast<T>(k.lengthscale());
                const MaternNu nu = k.nu();
                write_square_points(x, d2_k, uplo,
                                    [x, scale, nu](std::size_t row, std::size_t col) {
                                        const T d2 = sq_dist_points(x, row, x, col);
                                        const T r = d2 > T{0} ? std::sqrt(d2) : T{0};
                                        return eval_matern_iso_hess(scale * r, nu);
                                    });
                return {};
            } else if constexpr (std::is_same_v<LeafT, MaternArdKernel>) {
                const auto ls = k.lengthscales();
                if (auto d = check_ard_dim<T>(x.cols(), ls.size()); !d.has_value()) {
                    return d;
                }
                const T c = matern_scale_sq<T>(k.nu());
                const MaternNu nu = k.nu();
                const T ell_i = static_cast<T>(ls[i]);
                const T ell_j = static_cast<T>(ls[j]);
                write_square_points(
                    x, d2_k, uplo,
                    [x, &ls, c, nu, i, j, ell_i, ell_j](std::size_t row, std::size_t col) {
                        const T u2 = c * ard_scaled_sq(x, row, x, col, ls);
                        if (u2 <= T{0}) {
                            return T{0};
                        }
                        const T u = std::sqrt(u2);
                        const T di = (x(row, i) - x(col, i)) / ell_i;
                        const T wi = (c * di * di) / u2;
                        const T dj = (x(row, j) - x(col, j)) / ell_j;
                        const T wj = (c * dj * dj) / u2;
                        T val = eval_matern_ard_cross_coeff(u, nu) * wi * wj;
                        if (i == j) {
                            val -= T{2} * eval_matern_grad_g1(u, nu) * wi;
                        }
                        return val;
                    });
                return {};
            } else if constexpr (std::is_same_v<LeafT, PeriodicKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                const T period = static_cast<T>(k.period());
                write_square_points(
                    x, d2_k, uplo, [x, ell, period, i, j](std::size_t row, std::size_t col) {
                        return periodic_hess_d2(sq_dist_points(x, row, x, col), ell, period, i, j);
                    });
                return {};
            } else if constexpr (std::is_same_v<LeafT, RationalQuadraticKernel>) {
                const T ell = static_cast<T>(k.lengthscale());
                const T alpha = static_cast<T>(k.alpha());
                write_square_points(
                    x, d2_k, uplo, [x, ell, alpha, i, j](std::size_t row, std::size_t col) {
                        const T w = sq_dist_points(x, row, x, col) / (T{2} * alpha * ell * ell);
                        return rq_hess_iso(w, alpha, i, j);
                    });
                return {};
            } else if constexpr (std::is_same_v<LeafT, RationalQuadraticArdKernel>) {
                const auto ls = k.lengthscales();
                if (auto d = check_ard_dim<T>(x.cols(), ls.size()); !d.has_value()) {
                    return d;
                }
                const T alpha = static_cast<T>(k.alpha());
                const std::size_t alpha_idx = ls.size();
                write_square_points(
                    x, d2_k, uplo,
                    [x, &ls, alpha, i, j, alpha_idx](std::size_t row, std::size_t col) {
                        const T s2 = ard_scaled_sq(x, row, x, col, ls);
                        const T w = s2 / (T{2} * alpha);
                        const T b = T{1} + w;
                        const T kv = std::exp(-alpha * std::log1p(w));
                        const T g_alpha = alpha * ((w / b) - std::log1p(w));
                        if (i == alpha_idx && j == alpha_idx) {
                            return ((g_alpha * g_alpha) + g_alpha + ((alpha * w * w) / (b * b))) *
                                   kv;
                        }
                        const std::size_t r1 = row;
                        const std::size_t r2 = col;
                        if (i == alpha_idx || j == alpha_idx) {
                            const std::size_t ls_idx = (i == alpha_idx) ? j : i;
                            const T diff =
                                (x(r1, ls_idx) - x(r2, ls_idx)) / static_cast<T>(ls[ls_idx]);
                            return ((diff * diff) / b) * (g_alpha + (w / b)) * kv;
                        }
                        const T di = (x(r1, i) - x(r2, i)) / static_cast<T>(ls[i]);
                        const T dj = (x(r1, j) - x(r2, j)) / static_cast<T>(ls[j]);
                        const T si2 = di * di;
                        const T sj2 = dj * dj;
                        T val = (((alpha + T{1}) * si2 * sj2) / (alpha * b * b)) * kv;
                        if (i == j) {
                            val -= ((T{2} * si2) / b) * kv;
                        }
                        return val;
                    });
                return {};
            } else if constexpr (std::is_same_v<LeafT, ConstantKernel>) {
                const T c = static_cast<T>(k.constant());
                write_square_points(x, d2_k, uplo, [c](std::size_t, std::size_t) { return c; });
                return {};
            } else if constexpr (std::is_same_v<LeafT, LinearKernel>) {
                const T var = static_cast<T>(k.variance());
                write_square_points(x, d2_k, uplo, [x, var](std::size_t row, std::size_t col) {
                    return var * dot_points(x, row, x, col);
                });
                return {};
            } else {
                const T var = static_cast<T>(k.variance());
                write_square_points(x, d2_k, uplo, [var](std::size_t row, std::size_t col) {
                    return row == col ? var : T{0};
                });
                return {};
            }
        },
        leaf);
}

template std::expected<void, GprError> leaf_apply_dist<double>(const KernelLeaf&, MatrixRef<double>,
                                                               MatrixMut<double>, Triangle);
template std::expected<void, GprError> leaf_apply_dist<float>(const KernelLeaf&, MatrixRef<float>,
                                                              MatrixMut<float>, Triangle);
template std::expected<void, GprError>
leaf_apply_points<double>(const KernelLeaf&, MatrixRef<double>, MatrixMut<double>, Triangle);
template std::expected<void, GprError> leaf_apply_points<float>(const KernelLeaf&, MatrixRef<float>,
                                                                MatrixMut<float>, Triangle);
template std::expected<void, GprError>
leaf_apply_cross_dist<double>(const KernelLeaf&, MatrixRef<double>, MatrixMut<double>);
template std::expected<void, GprError>
leaf_apply_cross_dist<float>(const KernelLeaf&, MatrixRef<float>, MatrixMut<float>);
template std::expected<void, GprError> leaf_apply_cross_points<double>(const KernelLeaf&,
                                                                       MatrixRef<double>,
                                                                       MatrixRef<double>,
                                                                       MatrixMut<double>);
template std::expected<void, GprError> leaf_apply_cross_points<float>(const KernelLeaf&,
                                                                      MatrixRef<float>,
                                                                      MatrixRef<float>,
                                                                      MatrixMut<float>);
template std::expected<void, GprError> leaf_fill_diag<double>(const KernelLeaf&, MatrixRef<double>,
                                                              std::span<double>);
template std::expected<void, GprError> leaf_fill_diag<float>(const KernelLeaf&, MatrixRef<float>,
                                                             std::span<float>);
template std::expected<void, GprError> leaf_grad_dist<double>(const KernelLeaf&, MatrixRef<double>,
                                                              MatrixMut<double>, std::size_t,
                                                              Triangle);
template std::expected<void, GprError>
leaf_grad_dist<float>(const KernelLeaf&, MatrixRef<float>, MatrixMut<float>, std::size_t, Triangle);
template std::expected<void, GprError> leaf_grad_points<double>(const KernelLeaf&,
                                                                MatrixRef<double>,
                                                                MatrixMut<double>, std::size_t,
                                                                Triangle);
template std::expected<void, GprError> leaf_grad_points<float>(const KernelLeaf&, MatrixRef<float>,
                                                               MatrixMut<float>, std::size_t,
                                                               Triangle);
template std::expected<void, GprError> leaf_hess_dist<double>(const KernelLeaf&, MatrixRef<double>,
                                                              MatrixMut<double>, std::size_t,
                                                              std::size_t, Triangle);
template std::expected<void, GprError> leaf_hess_dist<float>(const KernelLeaf&, MatrixRef<float>,
                                                             MatrixMut<float>, std::size_t,
                                                             std::size_t, Triangle);
template std::expected<void, GprError> leaf_hess_points<double>(const KernelLeaf&,
                                                                MatrixRef<double>,
                                                                MatrixMut<double>, std::size_t,
                                                                std::size_t, Triangle);
template std::expected<void, GprError> leaf_hess_points<float>(const KernelLeaf&, MatrixRef<float>,
                                                               MatrixMut<float>, std::size_t,
                                                               std::size_t, Triangle);

} // namespace cxxgpr::detail
