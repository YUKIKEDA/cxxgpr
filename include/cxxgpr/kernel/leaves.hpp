// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/interval.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

namespace cxxgpr {

/// Smoothness parameter nu for Matern and ARD Matern kernels.
enum class MaternNu : std::uint8_t {
    OneHalf,
    ThreeHalves,
    FiveHalves,
};

/// Isotropic squared-exponential (RBF) kernel: exp(-d^2 / (2 * lengthscale^2)).
class RbfKernel {
public:
    /// Constructs an RBF kernel with positive lengthscale inside Interval::default_positive().
    [[nodiscard]] static std::expected<RbfKernel, GprError> create(double lengthscale);

    /// Constructs an RBF kernel with lengthscale strictly inside the given positive interval.
    [[nodiscard]] static std::expected<RbfKernel, GprError> create(double lengthscale,
                                                                   Interval interval);

    /// Returns the number of hyperparameters (always 1).
    [[nodiscard]] static constexpr std::size_t num_params() noexcept {
        return 1;
    }

    /// Returns the lengthscale value in user units.
    [[nodiscard]] double lengthscale() const noexcept {
        return lengthscale_.value();
    }

    /// Returns the open interval bounding the lengthscale.
    [[nodiscard]] Interval interval() const noexcept {
        return lengthscale_.interval();
    }

    /// Writes log(lengthscale) into out[0].
    [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

    /// Updates lengthscale from log-space params[0], failing atomically if out of bounds.
    [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

    /// Writes the open interval into out[0].
    [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

    /// Checks value equality.
    [[nodiscard]] friend constexpr bool operator==(const RbfKernel& lhs,
                                                   const RbfKernel& rhs) noexcept = default;

private:
    explicit constexpr RbfKernel(BoundedParam lengthscale) noexcept : lengthscale_(lengthscale) {}

    BoundedParam lengthscale_;
};

/// ARD squared-exponential (RBF) kernel with one lengthscale per input dimension.
class RbfArdKernel {
public:
    /// Constructs an ARD RBF kernel with positive lengthscales inside Interval::default_positive().
    [[nodiscard]] static std::expected<RbfArdKernel, GprError>
    create(std::span<const double> lengthscales);

    /// Constructs an ARD RBF kernel with per-dimension open intervals.
    [[nodiscard]] static std::expected<RbfArdKernel, GprError>
    create(std::span<const double> lengthscales, std::span<const Interval> intervals);

    /// Returns the number of hyperparameters (equals input feature dimensionality).
    [[nodiscard]] std::size_t num_params() const noexcept {
        return lengthscales_.size();
    }

    /// Returns the lengthscale values in user units.
    [[nodiscard]] std::vector<double> lengthscales() const;

    /// Returns the open intervals bounding each lengthscale.
    [[nodiscard]] std::vector<Interval> intervals() const;

    /// Writes log(lengthscales) into out.
    [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

    /// Atomically updates all lengthscales from log-space parameters.
    [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

    /// Writes per-dimension open intervals into out.
    [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

    /// Checks value equality.
    [[nodiscard]] friend bool operator==(const RbfArdKernel& lhs,
                                         const RbfArdKernel& rhs) noexcept = default;

private:
    explicit RbfArdKernel(std::vector<BoundedParam> lengthscales) noexcept
        : lengthscales_(std::move(lengthscales)) {}

    std::vector<BoundedParam> lengthscales_;
};

/// Isotropic Matern kernel for nu in {1/2, 3/2, 5/2}.
class MaternKernel {
public:
    /// Constructs a Matern kernel with lengthscale inside Interval::default_positive().
    [[nodiscard]] static std::expected<MaternKernel, GprError> create(double lengthscale,
                                                                      MaternNu nu);

    /// Constructs a Matern kernel with lengthscale inside the given positive interval.
    [[nodiscard]] static std::expected<MaternKernel, GprError>
    create(double lengthscale, MaternNu nu, Interval interval);

    /// Returns the number of hyperparameters (always 1).
    [[nodiscard]] static constexpr std::size_t num_params() noexcept {
        return 1;
    }

    /// Returns the lengthscale value in user units.
    [[nodiscard]] double lengthscale() const noexcept {
        return lengthscale_.value();
    }

    /// Returns the smoothness order nu.
    [[nodiscard]] constexpr MaternNu nu() const noexcept {
        return nu_;
    }

    /// Returns the open interval bounding the lengthscale.
    [[nodiscard]] Interval interval() const noexcept {
        return lengthscale_.interval();
    }

    /// Writes log(lengthscale) into out[0].
    [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

    /// Updates lengthscale from log-space params[0], failing atomically if out of bounds.
    [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

    /// Writes the open interval into out[0].
    [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

    /// Checks value equality.
    [[nodiscard]] friend constexpr bool operator==(const MaternKernel& lhs,
                                                   const MaternKernel& rhs) noexcept = default;

private:
    constexpr MaternKernel(BoundedParam lengthscale, MaternNu nu) noexcept
        : lengthscale_(lengthscale), nu_(nu) {}

    BoundedParam lengthscale_;
    MaternNu nu_{MaternNu::FiveHalves};
};

/// ARD Matern kernel with per-dimension lengthscales for nu in {1/2, 3/2, 5/2}.
class MaternArdKernel {
public:
    /// Constructs an ARD Matern kernel with lengthscales inside Interval::default_positive().
    [[nodiscard]] static std::expected<MaternArdKernel, GprError>
    create(std::span<const double> lengthscales, MaternNu nu);

    /// Constructs an ARD Matern kernel with per-dimension open intervals.
    [[nodiscard]] static std::expected<MaternArdKernel, GprError>
    create(std::span<const double> lengthscales, MaternNu nu, std::span<const Interval> intervals);

    /// Returns the number of hyperparameters (equals input feature dimensionality).
    [[nodiscard]] std::size_t num_params() const noexcept {
        return lengthscales_.size();
    }

    /// Returns the lengthscale values in user units.
    [[nodiscard]] std::vector<double> lengthscales() const;

    /// Returns the smoothness order nu.
    [[nodiscard]] constexpr MaternNu nu() const noexcept {
        return nu_;
    }

    /// Returns the open intervals bounding each lengthscale.
    [[nodiscard]] std::vector<Interval> intervals() const;

    /// Writes log(lengthscales) into out.
    [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

    /// Atomically updates all lengthscales from log-space parameters.
    [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

    /// Writes per-dimension open intervals into out.
    [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

    /// Checks value equality.
    [[nodiscard]] friend bool operator==(const MaternArdKernel& lhs,
                                         const MaternArdKernel& rhs) noexcept = default;

private:
    MaternArdKernel(std::vector<BoundedParam> lengthscales, MaternNu nu) noexcept
        : lengthscales_(std::move(lengthscales)), nu_(nu) {}

    std::vector<BoundedParam> lengthscales_;
    MaternNu nu_{MaternNu::FiveHalves};
};

/// Isotropic periodic kernel: exp(-2 * sin^2(pi * r / period) / lengthscale^2).
class PeriodicKernel {
public:
    /// Constructs a periodic kernel with default positive intervals.
    [[nodiscard]] static std::expected<PeriodicKernel, GprError> create(double lengthscale,
                                                                        double period);

    /// Constructs a periodic kernel with explicit intervals for lengthscale and period.
    [[nodiscard]] static std::expected<PeriodicKernel, GprError>
    create(double lengthscale, double period, Interval lengthscale_interval,
           Interval period_interval);

    /// Returns the number of hyperparameters (always 2: log lengthscale, log period).
    [[nodiscard]] static constexpr std::size_t num_params() noexcept {
        return 2;
    }

    /// Returns the lengthscale value in user units.
    [[nodiscard]] double lengthscale() const noexcept {
        return lengthscale_.value();
    }

    /// Returns the period value in user units.
    [[nodiscard]] double period() const noexcept {
        return period_.value();
    }

    /// Returns the open interval bounding the lengthscale.
    [[nodiscard]] Interval lengthscale_interval() const noexcept {
        return lengthscale_.interval();
    }

    /// Returns the open interval bounding the period.
    [[nodiscard]] Interval period_interval() const noexcept {
        return period_.interval();
    }

    /// Writes [log(lengthscale), log(period)] into out.
    [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

    /// Atomically updates lengthscale and period from log-space parameters.
    [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

    /// Writes [lengthscale_interval, period_interval] into out.
    [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

    /// Checks value equality.
    [[nodiscard]] friend constexpr bool operator==(const PeriodicKernel& lhs,
                                                   const PeriodicKernel& rhs) noexcept = default;

private:
    constexpr PeriodicKernel(BoundedParam lengthscale, BoundedParam period) noexcept
        : lengthscale_(lengthscale), period_(period) {}

    BoundedParam lengthscale_;
    BoundedParam period_;
};

/// Isotropic rational-quadratic kernel: (1 + d^2 / (2 * alpha * lengthscale^2))^(-alpha).
class RationalQuadraticKernel {
public:
    /// Constructs an isotropic RQ kernel with default positive intervals.
    [[nodiscard]] static std::expected<RationalQuadraticKernel, GprError> create(double lengthscale,
                                                                                 double alpha);

    /// Constructs an isotropic RQ kernel with explicit intervals for lengthscale and alpha.
    [[nodiscard]] static std::expected<RationalQuadraticKernel, GprError>
    create(double lengthscale, double alpha, Interval lengthscale_interval,
           Interval alpha_interval);

    /// Returns the number of hyperparameters (always 2: log lengthscale, log alpha).
    [[nodiscard]] static constexpr std::size_t num_params() noexcept {
        return 2;
    }

    /// Returns the lengthscale value in user units.
    [[nodiscard]] double lengthscale() const noexcept {
        return lengthscale_.value();
    }

    /// Returns the scale-mixture shape parameter alpha in user units.
    [[nodiscard]] double alpha() const noexcept {
        return alpha_.value();
    }

    /// Returns the open interval bounding the lengthscale.
    [[nodiscard]] Interval lengthscale_interval() const noexcept {
        return lengthscale_.interval();
    }

    /// Returns the open interval bounding alpha.
    [[nodiscard]] Interval alpha_interval() const noexcept {
        return alpha_.interval();
    }

    /// Writes [log(lengthscale), log(alpha)] into out.
    [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

    /// Atomically updates lengthscale and alpha from log-space parameters.
    [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

    /// Writes [lengthscale_interval, alpha_interval] into out.
    [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

    /// Checks value equality.
    [[nodiscard]] friend constexpr bool
    operator==(const RationalQuadraticKernel& lhs,
               const RationalQuadraticKernel& rhs) noexcept = default;

private:
    constexpr RationalQuadraticKernel(BoundedParam lengthscale, BoundedParam alpha) noexcept
        : lengthscale_(lengthscale), alpha_(alpha) {}

    BoundedParam lengthscale_;
    BoundedParam alpha_;
};

/// ARD rational-quadratic kernel with per-dimension lengthscales and shape parameter alpha.
class RationalQuadraticArdKernel {
public:
    /// Constructs an ARD RQ kernel with default positive intervals.
    [[nodiscard]] static std::expected<RationalQuadraticArdKernel, GprError>
    create(std::span<const double> lengthscales, double alpha);

    /// Constructs an ARD RQ kernel with explicit intervals for lengthscales and alpha.
    [[nodiscard]] static std::expected<RationalQuadraticArdKernel, GprError>
    create(std::span<const double> lengthscales, double alpha,
           std::span<const Interval> lengthscale_intervals, Interval alpha_interval);

    /// Returns the number of hyperparameters (dims + 1).
    [[nodiscard]] std::size_t num_params() const noexcept {
        return lengthscales_.size() + 1;
    }

    /// Returns the lengthscale values in user units.
    [[nodiscard]] std::vector<double> lengthscales() const;

    /// Returns the scale-mixture shape parameter alpha in user units.
    [[nodiscard]] double alpha() const noexcept {
        return alpha_.value();
    }

    /// Returns the open intervals bounding each lengthscale.
    [[nodiscard]] std::vector<Interval> lengthscale_intervals() const;

    /// Returns the open interval bounding alpha.
    [[nodiscard]] Interval alpha_interval() const noexcept {
        return alpha_.interval();
    }

    /// Writes [log(lengthscales)..., log(alpha)] into out.
    [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

    /// Atomically updates lengthscales and alpha from log-space parameters.
    [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

    /// Writes [lengthscale_intervals..., alpha_interval] into out.
    [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

    /// Checks value equality.
    [[nodiscard]] friend bool operator==(const RationalQuadraticArdKernel& lhs,
                                         const RationalQuadraticArdKernel& rhs) noexcept = default;

private:
    RationalQuadraticArdKernel(std::vector<BoundedParam> lengthscales, BoundedParam alpha) noexcept
        : lengthscales_(std::move(lengthscales)), alpha_(alpha) {}

    std::vector<BoundedParam> lengthscales_;
    BoundedParam alpha_;
};

/// Constant kernel: k(x, x') = constant.
class ConstantKernel {
public:
    /// Constructs a constant kernel with value inside Interval::default_positive().
    [[nodiscard]] static std::expected<ConstantKernel, GprError> create(double constant);

    /// Constructs a constant kernel with value inside the given positive interval.
    [[nodiscard]] static std::expected<ConstantKernel, GprError> create(double constant,
                                                                        Interval interval);

    /// Returns the number of hyperparameters (always 1).
    [[nodiscard]] static constexpr std::size_t num_params() noexcept {
        return 1;
    }

    /// Returns the constant amplitude value in user units.
    [[nodiscard]] double constant() const noexcept {
        return constant_.value();
    }

    /// Returns the open interval bounding the constant value.
    [[nodiscard]] Interval interval() const noexcept {
        return constant_.interval();
    }

    /// Writes log(constant) into out[0].
    [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

    /// Updates constant from log-space params[0], failing atomically if out of bounds.
    [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

    /// Writes the open interval into out[0].
    [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

    /// Checks value equality.
    [[nodiscard]] friend constexpr bool operator==(const ConstantKernel& lhs,
                                                   const ConstantKernel& rhs) noexcept = default;

private:
    explicit constexpr ConstantKernel(BoundedParam constant) noexcept : constant_(constant) {}

    BoundedParam constant_;
};

/// Linear dot-product kernel: k(x, x') = variance * (x . x').
class LinearKernel {
public:
    /// Constructs a linear kernel with variance inside Interval::default_positive().
    [[nodiscard]] static std::expected<LinearKernel, GprError> create(double variance);

    /// Constructs a linear kernel with variance inside the given positive interval.
    [[nodiscard]] static std::expected<LinearKernel, GprError> create(double variance,
                                                                      Interval interval);

    /// Returns the number of hyperparameters (always 1).
    [[nodiscard]] static constexpr std::size_t num_params() noexcept {
        return 1;
    }

    /// Returns the variance multiplier in user units.
    [[nodiscard]] double variance() const noexcept {
        return variance_.value();
    }

    /// Returns the open interval bounding the variance multiplier.
    [[nodiscard]] Interval interval() const noexcept {
        return variance_.interval();
    }

    /// Writes log(variance) into out[0].
    [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

    /// Updates variance from log-space params[0], failing atomically if out of bounds.
    [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

    /// Writes the open interval into out[0].
    [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

    /// Checks value equality.
    [[nodiscard]] friend constexpr bool operator==(const LinearKernel& lhs,
                                                   const LinearKernel& rhs) noexcept = default;

private:
    explicit constexpr LinearKernel(BoundedParam variance) noexcept : variance_(variance) {}

    BoundedParam variance_;
};

/// Independent white-noise kernel: k(x_i, x_j) = variance * delta_{i, j} on training Gram only.
class WhiteKernel {
public:
    /// Constructs a white-noise kernel with variance inside Interval::default_positive().
    [[nodiscard]] static std::expected<WhiteKernel, GprError> create(double variance);

    /// Constructs a white-noise kernel with variance inside the given positive interval.
    [[nodiscard]] static std::expected<WhiteKernel, GprError> create(double variance,
                                                                     Interval interval);

    /// Returns the number of hyperparameters (always 1).
    [[nodiscard]] static constexpr std::size_t num_params() noexcept {
        return 1;
    }

    /// Returns the noise variance in user units.
    [[nodiscard]] double variance() const noexcept {
        return variance_.value();
    }

    /// Returns the open interval bounding the noise variance.
    [[nodiscard]] Interval interval() const noexcept {
        return variance_.interval();
    }

    /// Writes log(variance) into out[0].
    [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

    /// Updates variance from log-space params[0], failing atomically if out of bounds.
    [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

    /// Writes the open interval into out[0].
    [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

    /// Checks value equality.
    [[nodiscard]] friend constexpr bool operator==(const WhiteKernel& lhs,
                                                   const WhiteKernel& rhs) noexcept = default;

private:
    explicit constexpr WhiteKernel(BoundedParam variance) noexcept : variance_(variance) {}

    BoundedParam variance_;
};

} // namespace cxxgpr
