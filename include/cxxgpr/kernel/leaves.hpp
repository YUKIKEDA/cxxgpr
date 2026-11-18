// SPDX-License-Identifier: Apache-2.0
/// @file leaves.hpp
/// @brief Built-in covariance kernel leaf types and bounded hyperparameter descriptors.
/// @details Declares MaternNu and all 10 built-in positive-parameterized kernel leaf value types
/// (RbfKernel, RbfArdKernel, MaternKernel, MaternArdKernel, PeriodicKernel, RationalQuadraticKernel,
/// RationalQuadraticArdKernel, ConstantKernel, LinearKernel, and WhiteKernel) with unconstrained
/// log-space optimization accessors and atomic bounded updates.
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/interval.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

/// @brief Root namespace of the cxxgpr Gaussian Process Regression library.
namespace cxxgpr
{

    /// @brief Smoothness parameter nu for isotropic and ARD Matern kernels.
    /// @details Selects the half-integer order controlling sample-path differentiability.
    enum class MaternNu : std::uint8_t
    {
        /// @brief Exponential (Matern-1/2) kernel: k(r) = exp(-r / l).
        OneHalf,
        /// @brief Once-differentiable Matern-3/2 kernel: k(r) = (1 + sqrt(3) * r / l) * exp(-sqrt(3) * r / l).
        ThreeHalves,
        /// @brief Twice-differentiable Matern-5/2 kernel: k(r) = (1 + u + u^2 / 3) * exp(-u), u = sqrt(5) * r / l.
        FiveHalves,
    };

    /// @brief Isotropic squared-exponential (RBF) kernel: k(d) = exp(-d^2 / (2 * lengthscale^2)).
    /// @details Stores a strictly positive lengthscale parameter bounded by an open Interval and exposed
    /// in unconstrained log space as log(lengthscale).
    class RbfKernel
    {
    public:
        /// @brief Constructs an RBF kernel with lengthscale inside Interval::default_positive().
        /// @details Requires lengthscale strictly inside (1e-5, 1e5).
        /// @param lengthscale Positive lengthscale in user units.
        /// @return Validated RbfKernel or GprError if lengthscale is out of range.
        /// @retval GprErrorCode::InvalidInterval When lengthscale is non-finite or outside default bounds.
        [[nodiscard]] static std::expected<RbfKernel, GprError> create(double lengthscale);

        /// @brief Constructs an RBF kernel with lengthscale strictly inside the given positive interval.
        /// @details Validates that interval is strictly positive (0 < lo < hi) and contains lengthscale.
        /// @param lengthscale Positive lengthscale in user units.
        /// @param interval Strictly positive open interval bounding lengthscale.
        /// @return Validated RbfKernel or GprError on invalid interval or out-of-range value.
        /// @retval GprErrorCode::InvalidInterval When interval is invalid or does not contain lengthscale.
        [[nodiscard]] static std::expected<RbfKernel, GprError> create(double lengthscale, Interval interval);

        /// @brief Returns the number of hyperparameters.
        /// @details Always returns 1 for isotropic RBF.
        /// @return Parameter count (1).
        [[nodiscard]] static constexpr std::size_t num_params() noexcept { return 1; }

        /// @brief Returns the lengthscale value in user units.
        /// @details Stored value is guaranteed strictly positive and inside interval().
        /// @return Current lengthscale.
        [[nodiscard]] double lengthscale() const noexcept { return lengthscale_.value(); }

        /// @brief Returns the open interval bounding the lengthscale.
        /// @details Endpoints define strict positivity and optimization constraints.
        /// @return Bounding open Interval.
        [[nodiscard]] Interval interval() const noexcept { return lengthscale_.interval(); }

        /// @brief Writes log(lengthscale) into out[0].
        /// @details Exposes unconstrained log-space parameterization for optimization.
        /// @param out Destination span of length 1.
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
        [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

        /// @brief Updates lengthscale from log-space params[0], failing atomically if out of bounds.
        /// @details Evaluates exp(params[0]) and leaves state unchanged on validation failure.
        /// @param params Input span of length 1 containing log(lengthscale).
        /// @return Empty expected on success or GprError on invalid count, non-finite input, or bound violation.
        /// @retval GprErrorCode::LengthMismatch When params.size() != 1.
        /// @retval GprErrorCode::NonFiniteInput When params[0] is NaN or infinite.
        /// @retval GprErrorCode::InvalidInterval When exp(params[0]) overflows or falls outside interval().
        [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

        /// @brief Writes the open interval into out[0].
        /// @details Populates one Interval in user units corresponding to parameter index 0.
        /// @param out Destination span of length 1.
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
        [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

        /// @brief Checks value equality between two RbfKernel instances.
        /// @details Compares stored lengthscale value and bounding interval.
        /// @param lhs Left-hand RbfKernel operand.
        /// @param rhs Right-hand RbfKernel operand.
        /// @return True if both kernels share identical lengthscale and interval.
        [[nodiscard]] friend constexpr bool operator==(const RbfKernel& lhs, const RbfKernel& rhs) noexcept = default;

    private:
        /// @brief Constructs an RbfKernel from an already-validated positive lengthscale parameter.
        /// @details Invoked by create() factories after positive interval validation.
        /// @param lengthscale Validated bounded lengthscale parameter.
        explicit constexpr RbfKernel(BoundedParam lengthscale) noexcept : lengthscale_(lengthscale) {}

        /// @brief Strictly positive lengthscale parameter with its open bounding interval.
        BoundedParam lengthscale_;
    };

    /// @brief ARD squared-exponential (RBF) kernel: k(x, x') = exp(-0.5 * sum_k ((x_k - x'_k) / l_k)^2).
    /// @details Maintains one strictly positive BoundedParam lengthscale per input feature dimension
    /// and updates all dimensions atomically in log space.
    class RbfArdKernel
    {
    public:
        /// @brief Constructs an ARD RBF kernel with positive lengthscales inside Interval::default_positive().
        /// @details Requires lengthscales to be non-empty with every entry strictly inside (1e-5, 1e5).
        /// @param lengthscales Per-dimension positive lengthscales in user units.
        /// @return Validated RbfArdKernel or GprError on empty or out-of-range inputs.
        /// @retval GprErrorCode::EmptyInput When lengthscales is empty.
        /// @retval GprErrorCode::InvalidInterval When any lengthscale is non-finite or outside default bounds.
        [[nodiscard]] static std::expected<RbfArdKernel, GprError> create(std::span<const double> lengthscales);

        /// @brief Constructs an ARD RBF kernel with per-dimension open intervals.
        /// @details Requires non-empty matching spans and strictly positive valid intervals for each dimension.
        /// @param lengthscales Per-dimension positive lengthscales in user units.
        /// @param intervals Per-dimension strictly positive open intervals.
        /// @return Validated RbfArdKernel or GprError on empty, mismatched, or invalid inputs.
        /// @retval GprErrorCode::EmptyInput When lengthscales is empty.
        /// @retval GprErrorCode::LengthMismatch When intervals.size() != lengthscales.size().
        /// @retval GprErrorCode::InvalidInterval When any interval or value fails strict positivity/containment.
        [[nodiscard]] static std::expected<RbfArdKernel, GprError> create(
            std::span<const double> lengthscales, std::span<const Interval> intervals
        );

        /// @brief Returns the number of hyperparameters.
        /// @details Equals the input feature dimensionality.
        /// @return Number of lengthscale parameters.
        [[nodiscard]] std::size_t num_params() const noexcept { return lengthscales_.size(); }

        /// @brief Returns the lengthscale values in user units.
        /// @details Allocates and returns a vector ordered by feature dimension.
        /// @return Vector of current positive lengthscale values.
        [[nodiscard]] std::vector<double> lengthscales() const;

        /// @brief Returns the open intervals bounding each lengthscale.
        /// @details Order matches lengthscales() and log-space parameter slots.
        /// @return Vector of bounding open Interval descriptors.
        [[nodiscard]] std::vector<Interval> intervals() const;

        /// @brief Writes log(lengthscales) into out.
        /// @details Populates out[k] = log(lengthscale_k) for k in [0, num_params()).
        /// @param out Destination span of length num_params().
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
        [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

        /// @brief Atomically updates all lengthscales from log-space parameters.
        /// @details Validates every entry into a temporary vector before replacing internal state so any
        /// failure leaves all lengthscales unchanged.
        /// @param params Input span of log-space lengthscales of length num_params().
        /// @return Empty expected on success or GprError on invalid count, non-finite value, or bound error.
        /// @retval GprErrorCode::LengthMismatch When params.size() != num_params().
        /// @retval GprErrorCode::NonFiniteInput When any element of params is NaN or infinite.
        /// @retval GprErrorCode::InvalidInterval When any exp(params[k]) overflows or leaves its interval.
        [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

        /// @brief Writes per-dimension open intervals into out.
        /// @details Copies interval_k into out[k] for each dimension k.
        /// @param out Destination span of length num_params().
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
        [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

        /// @brief Checks value equality between two RbfArdKernel instances.
        /// @details Compares all per-dimension lengthscale values and intervals elementwise.
        /// @param lhs Left-hand RbfArdKernel operand.
        /// @param rhs Right-hand RbfArdKernel operand.
        /// @return True if both kernels have identical dimensions, lengthscales, and intervals.
        [[nodiscard]] friend bool operator==(const RbfArdKernel& lhs, const RbfArdKernel& rhs) noexcept = default;

    private:
        /// @brief Constructs an RbfArdKernel from validated bounded lengthscale parameters.
        /// @details Invoked by create() factories after per-dimension validation succeeds.
        /// @param lengthscales Non-empty vector of validated positive BoundedParam lengthscales.
        explicit RbfArdKernel(std::vector<BoundedParam> lengthscales) noexcept : lengthscales_(std::move(lengthscales))
        {
        }

        /// @brief Per-dimension strictly positive lengthscale parameters.
        std::vector<BoundedParam> lengthscales_;
    };

    /// @brief Isotropic Matern kernel for nu in {1/2, 3/2, 5/2}.
    /// @details Evaluates k(d) via the closed-form Matern polynomial-exponential representation in
    /// u = sqrt(2 * nu) * d / lengthscale with a single positive lengthscale hyperparameter.
    class MaternKernel
    {
    public:
        /// @brief Constructs a Matern kernel with lengthscale inside Interval::default_positive().
        /// @details Smoothness order nu is fixed at construction and not an optimizable parameter.
        /// @param lengthscale Positive lengthscale in user units.
        /// @param nu Discrete Matern smoothness order.
        /// @return Validated MaternKernel or GprError if lengthscale is out of range.
        /// @retval GprErrorCode::InvalidInterval When lengthscale is non-finite or outside default bounds.
        [[nodiscard]] static std::expected<MaternKernel, GprError> create(double lengthscale, MaternNu nu);

        /// @brief Constructs a Matern kernel with lengthscale inside the given positive interval.
        /// @details Validates that interval is strictly positive (0 < lo < hi) and contains lengthscale.
        /// @param lengthscale Positive lengthscale in user units.
        /// @param nu Discrete Matern smoothness order.
        /// @param interval Strictly positive open interval bounding lengthscale.
        /// @return Validated MaternKernel or GprError on invalid interval or out-of-range value.
        /// @retval GprErrorCode::InvalidInterval When interval is invalid or does not contain lengthscale.
        [[nodiscard]] static std::expected<MaternKernel, GprError> create(
            double lengthscale, MaternNu nu, Interval interval
        );

        /// @brief Returns the number of hyperparameters.
        /// @details Always returns 1 (smoothness order nu is discrete and fixed).
        /// @return Parameter count (1).
        [[nodiscard]] static constexpr std::size_t num_params() noexcept { return 1; }

        /// @brief Returns the lengthscale value in user units.
        /// @details Stored value is strictly positive and inside interval().
        /// @return Current lengthscale.
        [[nodiscard]] double lengthscale() const noexcept { return lengthscale_.value(); }

        /// @brief Returns the discrete smoothness order nu.
        /// @details Selects Matern-1/2, 3/2, or 5/2 polynomial coefficients.
        /// @return Configured MaternNu enum value.
        [[nodiscard]] constexpr MaternNu nu() const noexcept { return nu_; }

        /// @brief Returns the open interval bounding the lengthscale.
        /// @details Defines strict positivity and optimization constraints.
        /// @return Bounding open Interval.
        [[nodiscard]] Interval interval() const noexcept { return lengthscale_.interval(); }

        /// @brief Writes log(lengthscale) into out[0].
        /// @details Exposes unconstrained log-space parameterization for optimization.
        /// @param out Destination span of length 1.
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
        [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

        /// @brief Updates lengthscale from log-space params[0], failing atomically if out of bounds.
        /// @details Leaves internal state unchanged when validation fails.
        /// @param params Input span of length 1 containing log(lengthscale).
        /// @return Empty expected on success or GprError on invalid count, non-finite input, or bound error.
        /// @retval GprErrorCode::LengthMismatch When params.size() != 1.
        /// @retval GprErrorCode::NonFiniteInput When params[0] is NaN or infinite.
        /// @retval GprErrorCode::InvalidInterval When exp(params[0]) overflows or falls outside interval().
        [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

        /// @brief Writes the open interval into out[0].
        /// @details Populates the bounding Interval for parameter index 0.
        /// @param out Destination span of length 1.
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
        [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

        /// @brief Checks value equality between two MaternKernel instances.
        /// @details Compares lengthscale, bounding interval, and smoothness order nu.
        /// @param lhs Left-hand MaternKernel operand.
        /// @param rhs Right-hand MaternKernel operand.
        /// @return True if lengthscale, interval, and nu match.
        [[nodiscard]] friend constexpr bool operator==(
            const MaternKernel& lhs, const MaternKernel& rhs
        ) noexcept = default;

    private:
        /// @brief Constructs a MaternKernel from validated lengthscale and smoothness order.
        /// @details Invoked by create() factories after positive interval validation.
        /// @param lengthscale Validated bounded lengthscale parameter.
        /// @param nu Discrete Matern smoothness order.
        constexpr MaternKernel(BoundedParam lengthscale, MaternNu nu) noexcept : lengthscale_(lengthscale), nu_(nu) {}

        /// @brief Strictly positive lengthscale parameter with its open bounding interval.
        BoundedParam lengthscale_;
        /// @brief Fixed half-integer smoothness order nu in {1/2, 3/2, 5/2}.
        MaternNu nu_{MaternNu::FiveHalves};
    };

    /// @brief ARD Matern kernel with per-dimension lengthscales for nu in {1/2, 3/2, 5/2}.
    /// @details Evaluates the Matern function on u = sqrt(2 * nu * sum_k ((x_k - x'_k) / l_k)^2)
    /// using one positive lengthscale per feature dimension.
    class MaternArdKernel
    {
    public:
        /// @brief Constructs an ARD Matern kernel with lengthscales inside Interval::default_positive().
        /// @details Requires non-empty lengthscales all inside (1e-5, 1e5).
        /// @param lengthscales Per-dimension positive lengthscales in user units.
        /// @param nu Discrete Matern smoothness order.
        /// @return Validated MaternArdKernel or GprError on empty or out-of-range inputs.
        /// @retval GprErrorCode::EmptyInput When lengthscales is empty.
        /// @retval GprErrorCode::InvalidInterval When any lengthscale is non-finite or outside default bounds.
        [[nodiscard]] static std::expected<MaternArdKernel, GprError> create(
            std::span<const double> lengthscales, MaternNu nu
        );

        /// @brief Constructs an ARD Matern kernel with per-dimension open intervals.
        /// @details Validates strictly positive intervals matching lengthscales in length.
        /// @param lengthscales Per-dimension positive lengthscales in user units.
        /// @param nu Discrete Matern smoothness order.
        /// @param intervals Per-dimension strictly positive open intervals.
        /// @return Validated MaternArdKernel or GprError on empty, mismatched, or invalid inputs.
        /// @retval GprErrorCode::EmptyInput When lengthscales is empty.
        /// @retval GprErrorCode::LengthMismatch When intervals.size() != lengthscales.size().
        /// @retval GprErrorCode::InvalidInterval When any interval or value fails validation.
        [[nodiscard]] static std::expected<MaternArdKernel, GprError> create(
            std::span<const double> lengthscales, MaternNu nu, std::span<const Interval> intervals
        );

        /// @brief Returns the number of hyperparameters.
        /// @details Equals the input feature dimensionality.
        /// @return Number of lengthscale parameters.
        [[nodiscard]] std::size_t num_params() const noexcept { return lengthscales_.size(); }

        /// @brief Returns the lengthscale values in user units.
        /// @details Allocates and returns a vector ordered by feature dimension.
        /// @return Vector of current positive lengthscale values.
        [[nodiscard]] std::vector<double> lengthscales() const;

        /// @brief Returns the discrete smoothness order nu.
        /// @details Selects Matern-1/2, 3/2, or 5/2 polynomial coefficients.
        /// @return Configured MaternNu enum value.
        [[nodiscard]] constexpr MaternNu nu() const noexcept { return nu_; }

        /// @brief Returns the open intervals bounding each lengthscale.
        /// @details Order matches lengthscales() and log-space parameter slots.
        /// @return Vector of bounding open Interval descriptors.
        [[nodiscard]] std::vector<Interval> intervals() const;

        /// @brief Writes log(lengthscales) into out.
        /// @details Populates out[k] = log(lengthscale_k) for k in [0, num_params()).
        /// @param out Destination span of length num_params().
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
        [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

        /// @brief Atomically updates all lengthscales from log-space parameters.
        /// @details Validates all values into a staging vector before committing so partial failures
        /// leave the kernel unchanged.
        /// @param params Input span of log-space lengthscales of length num_params().
        /// @return Empty expected on success or GprError on invalid count, non-finite value, or bound error.
        /// @retval GprErrorCode::LengthMismatch When params.size() != num_params().
        /// @retval GprErrorCode::NonFiniteInput When any element of params is NaN or infinite.
        /// @retval GprErrorCode::InvalidInterval When any exp(params[k]) overflows or leaves its interval.
        [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

        /// @brief Writes per-dimension open intervals into out.
        /// @details Copies each dimension's bounding Interval into out.
        /// @param out Destination span of length num_params().
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
        [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

        /// @brief Checks value equality between two MaternArdKernel instances.
        /// @details Compares lengthscale vectors and smoothness order nu.
        /// @param lhs Left-hand MaternArdKernel operand.
        /// @param rhs Right-hand MaternArdKernel operand.
        /// @return True if dimensions, lengthscales, intervals, and nu match.
        [[nodiscard]] friend bool operator==(const MaternArdKernel& lhs, const MaternArdKernel& rhs) noexcept = default;

    private:
        /// @brief Constructs a MaternArdKernel from validated lengthscales and smoothness order.
        /// @details Invoked by create() factories after per-dimension validation succeeds.
        /// @param lengthscales Non-empty vector of validated BoundedParam lengthscales.
        /// @param nu Discrete Matern smoothness order.
        MaternArdKernel(std::vector<BoundedParam> lengthscales, MaternNu nu) noexcept :
            lengthscales_(std::move(lengthscales)),
            nu_(nu)
        {
        }

        /// @brief Per-dimension strictly positive lengthscale parameters.
        std::vector<BoundedParam> lengthscales_;
        /// @brief Fixed half-integer smoothness order nu in {1/2, 3/2, 5/2}.
        MaternNu nu_{MaternNu::FiveHalves};
    };

    /// @brief Isotropic periodic kernel: k(d) = exp(-2 * sin^2(pi * d / period) / lengthscale^2).
    /// @details Parameterizes lengthscale at parameter index 0 and period at parameter index 1 in
    /// unconstrained log space.
    class PeriodicKernel
    {
    public:
        /// @brief Constructs a periodic kernel with default positive intervals.
        /// @details Requires both lengthscale and period strictly inside Interval::default_positive().
        /// @param lengthscale Positive lengthscale in user units.
        /// @param period Positive spatial period in user units.
        /// @return Validated PeriodicKernel or GprError on out-of-range parameters.
        /// @retval GprErrorCode::InvalidInterval When lengthscale or period falls outside default bounds.
        [[nodiscard]] static std::expected<PeriodicKernel, GprError> create(double lengthscale, double period);

        /// @brief Constructs a periodic kernel with explicit intervals for lengthscale and period.
        /// @details Both intervals must be strictly positive (0 < lo < hi) and contain their values.
        /// @param lengthscale Positive lengthscale in user units.
        /// @param period Positive spatial period in user units.
        /// @param lengthscale_interval Strictly positive open interval for lengthscale.
        /// @param period_interval Strictly positive open interval for period.
        /// @return Validated PeriodicKernel or GprError on invalid interval or out-of-range parameter.
        /// @retval GprErrorCode::InvalidInterval When either interval or value fails validation.
        [[nodiscard]] static std::expected<PeriodicKernel, GprError> create(
            double lengthscale, double period, Interval lengthscale_interval, Interval period_interval
        );

        /// @brief Returns the number of hyperparameters.
        /// @details Always returns 2 corresponding to [log(lengthscale), log(period)].
        /// @return Parameter count (2).
        [[nodiscard]] static constexpr std::size_t num_params() noexcept { return 2; }

        /// @brief Returns the lengthscale value in user units.
        /// @details Stored value lies strictly inside lengthscale_interval().
        /// @return Current lengthscale.
        [[nodiscard]] double lengthscale() const noexcept { return lengthscale_.value(); }

        /// @brief Returns the period value in user units.
        /// @details Stored value lies strictly inside period_interval().
        /// @return Current period.
        [[nodiscard]] double period() const noexcept { return period_.value(); }

        /// @brief Returns the open interval bounding the lengthscale.
        /// @details Defines strict positivity and optimization constraints for parameter index 0.
        /// @return Bounding open Interval for lengthscale.
        [[nodiscard]] Interval lengthscale_interval() const noexcept { return lengthscale_.interval(); }

        /// @brief Returns the open interval bounding the period.
        /// @details Defines strict positivity and optimization constraints for parameter index 1.
        /// @return Bounding open Interval for period.
        [[nodiscard]] Interval period_interval() const noexcept { return period_.interval(); }

        /// @brief Writes [log(lengthscale), log(period)] into out.
        /// @details Populates out[0] with log(lengthscale) and out[1] with log(period).
        /// @param out Destination span of length 2.
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != 2.
        [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

        /// @brief Atomically updates lengthscale and period from log-space parameters.
        /// @details Validates both params[0] and params[1] before updating either member.
        /// @param params Input span of length 2 ordered [log(lengthscale), log(period)].
        /// @return Empty expected on success or GprError on count mismatch, non-finite input, or bound error.
        /// @retval GprErrorCode::LengthMismatch When params.size() != 2.
        /// @retval GprErrorCode::NonFiniteInput When any parameter is NaN or infinite.
        /// @retval GprErrorCode::InvalidInterval When exp(param) overflows or leaves its open interval.
        [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

        /// @brief Writes [lengthscale_interval, period_interval] into out.
        /// @details Populates out[0] and out[1] in parameter order.
        /// @param out Destination span of length 2.
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != 2.
        [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

        /// @brief Checks value equality between two PeriodicKernel instances.
        /// @details Compares lengthscale and period values along with their bounding intervals.
        /// @param lhs Left-hand PeriodicKernel operand.
        /// @param rhs Right-hand PeriodicKernel operand.
        /// @return True if both parameters and intervals match.
        [[nodiscard]] friend constexpr bool operator==(
            const PeriodicKernel& lhs, const PeriodicKernel& rhs
        ) noexcept = default;

    private:
        /// @brief Constructs a PeriodicKernel from validated lengthscale and period parameters.
        /// @details Invoked by create() factories after positive interval validation.
        /// @param lengthscale Validated bounded lengthscale parameter.
        /// @param period Validated bounded period parameter.
        constexpr PeriodicKernel(BoundedParam lengthscale, BoundedParam period) noexcept :
            lengthscale_(lengthscale),
            period_(period)
        {
        }

        /// @brief Strictly positive lengthscale parameter (index 0).
        BoundedParam lengthscale_;
        /// @brief Strictly positive period parameter (index 1).
        BoundedParam period_;
    };

    /// @brief Isotropic rational-quadratic kernel: k(d) = (1 + d^2 / (2 * alpha * lengthscale^2))^(-alpha).
    /// @details Scale mixture of RBF kernels parameterized by [log(lengthscale), log(alpha)].
    class RationalQuadraticKernel
    {
    public:
        /// @brief Constructs an isotropic RQ kernel with default positive intervals.
        /// @details Requires both lengthscale and alpha strictly inside Interval::default_positive().
        /// @param lengthscale Positive lengthscale in user units.
        /// @param alpha Positive scale-mixture shape parameter.
        /// @return Validated RationalQuadraticKernel or GprError on out-of-range parameters.
        /// @retval GprErrorCode::InvalidInterval When lengthscale or alpha falls outside default bounds.
        [[nodiscard]] static std::expected<RationalQuadraticKernel, GprError> create(double lengthscale, double alpha);

        /// @brief Constructs an isotropic RQ kernel with explicit intervals for lengthscale and alpha.
        /// @details Both intervals must be strictly positive (0 < lo < hi) and contain their values.
        /// @param lengthscale Positive lengthscale in user units.
        /// @param alpha Positive scale-mixture shape parameter.
        /// @param lengthscale_interval Strictly positive open interval for lengthscale.
        /// @param alpha_interval Strictly positive open interval for alpha.
        /// @return Validated RationalQuadraticKernel or GprError on invalid interval or value.
        /// @retval GprErrorCode::InvalidInterval When either interval or value fails validation.
        [[nodiscard]] static std::expected<RationalQuadraticKernel, GprError> create(
            double lengthscale, double alpha, Interval lengthscale_interval, Interval alpha_interval
        );

        /// @brief Returns the number of hyperparameters.
        /// @details Always returns 2 corresponding to [log(lengthscale), log(alpha)].
        /// @return Parameter count (2).
        [[nodiscard]] static constexpr std::size_t num_params() noexcept { return 2; }

        /// @brief Returns the lengthscale value in user units.
        /// @details Stored value lies strictly inside lengthscale_interval().
        /// @return Current lengthscale.
        [[nodiscard]] double lengthscale() const noexcept { return lengthscale_.value(); }

        /// @brief Returns the scale-mixture shape parameter alpha in user units.
        /// @details Stored value lies strictly inside alpha_interval().
        /// @return Current shape parameter alpha.
        [[nodiscard]] double alpha() const noexcept { return alpha_.value(); }

        /// @brief Returns the open interval bounding the lengthscale.
        /// @details Defines strict positivity and optimization constraints for parameter index 0.
        /// @return Bounding open Interval for lengthscale.
        [[nodiscard]] Interval lengthscale_interval() const noexcept { return lengthscale_.interval(); }

        /// @brief Returns the open interval bounding alpha.
        /// @details Defines strict positivity and optimization constraints for parameter index 1.
        /// @return Bounding open Interval for alpha.
        [[nodiscard]] Interval alpha_interval() const noexcept { return alpha_.interval(); }

        /// @brief Writes [log(lengthscale), log(alpha)] into out.
        /// @details Populates out[0] with log(lengthscale) and out[1] with log(alpha).
        /// @param out Destination span of length 2.
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != 2.
        [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

        /// @brief Atomically updates lengthscale and alpha from log-space parameters.
        /// @details Validates both parameters before mutating either member.
        /// @param params Input span of length 2 ordered [log(lengthscale), log(alpha)].
        /// @return Empty expected on success or GprError on count mismatch, non-finite input, or bound error.
        /// @retval GprErrorCode::LengthMismatch When params.size() != 2.
        /// @retval GprErrorCode::NonFiniteInput When any parameter is NaN or infinite.
        /// @retval GprErrorCode::InvalidInterval When exp(param) overflows or leaves its open interval.
        [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

        /// @brief Writes [lengthscale_interval, alpha_interval] into out.
        /// @details Populates out[0] and out[1] in parameter order.
        /// @param out Destination span of length 2.
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != 2.
        [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

        /// @brief Checks value equality between two RationalQuadraticKernel instances.
        /// @details Compares lengthscale, alpha, and their bounding intervals.
        /// @param lhs Left-hand RationalQuadraticKernel operand.
        /// @param rhs Right-hand RationalQuadraticKernel operand.
        /// @return True if both parameters and intervals match.
        [[nodiscard]] friend constexpr bool operator==(
            const RationalQuadraticKernel& lhs, const RationalQuadraticKernel& rhs
        ) noexcept = default;

    private:
        /// @brief Constructs a RationalQuadraticKernel from validated lengthscale and alpha parameters.
        /// @details Invoked by create() factories after positive interval validation.
        /// @param lengthscale Validated bounded lengthscale parameter.
        /// @param alpha Validated bounded alpha parameter.
        constexpr RationalQuadraticKernel(BoundedParam lengthscale, BoundedParam alpha) noexcept :
            lengthscale_(lengthscale),
            alpha_(alpha)
        {
        }

        /// @brief Strictly positive lengthscale parameter (index 0).
        BoundedParam lengthscale_;
        /// @brief Strictly positive scale-mixture shape parameter alpha (index 1).
        BoundedParam alpha_;
    };

    /// @brief ARD rational-quadratic kernel with per-dimension lengthscales and shape parameter alpha.
    /// @details Evaluates k(x, x') = (1 + (sum_k ((x_k - x'_k) / l_k)^2) / (2 * alpha))^(-alpha) with
    /// dims + 1 hyperparameters ordered [log(lengthscales)..., log(alpha)].
    class RationalQuadraticArdKernel
    {
    public:
        /// @brief Constructs an ARD RQ kernel with default positive intervals.
        /// @details Requires non-empty lengthscales and alpha all strictly inside Interval::default_positive().
        /// @param lengthscales Per-dimension positive lengthscales in user units.
        /// @param alpha Positive scale-mixture shape parameter.
        /// @return Validated RationalQuadraticArdKernel or GprError on empty or out-of-range input.
        /// @retval GprErrorCode::EmptyInput When lengthscales is empty.
        /// @retval GprErrorCode::InvalidInterval When any lengthscale or alpha falls outside default bounds.
        [[nodiscard]] static std::expected<RationalQuadraticArdKernel, GprError> create(
            std::span<const double> lengthscales, double alpha
        );

        /// @brief Constructs an ARD RQ kernel with explicit intervals for lengthscales and alpha.
        /// @details Validates strictly positive intervals and matching span lengths for lengthscales.
        /// @param lengthscales Per-dimension positive lengthscales in user units.
        /// @param alpha Positive scale-mixture shape parameter.
        /// @param lengthscale_intervals Strictly positive open intervals for each lengthscale.
        /// @param alpha_interval Strictly positive open interval for alpha.
        /// @return Validated RationalQuadraticArdKernel or GprError on empty, mismatched, or invalid inputs.
        /// @retval GprErrorCode::EmptyInput When lengthscales is empty.
        /// @retval GprErrorCode::LengthMismatch When lengthscale_intervals.size() != lengthscales.size().
        /// @retval GprErrorCode::InvalidInterval When any interval or value fails validation.
        [[nodiscard]] static std::expected<RationalQuadraticArdKernel, GprError> create(
            std::span<const double> lengthscales,
            double alpha,
            std::span<const Interval> lengthscale_intervals,
            Interval alpha_interval
        );

        /// @brief Returns the number of hyperparameters.
        /// @details Equals feature dimensionality plus one for alpha (dims + 1).
        /// @return Total parameter count (lengthscales.size() + 1).
        [[nodiscard]] std::size_t num_params() const noexcept { return lengthscales_.size() + 1; }

        /// @brief Returns the lengthscale values in user units.
        /// @details Allocates and returns a vector ordered by feature dimension.
        /// @return Vector of current positive lengthscales.
        [[nodiscard]] std::vector<double> lengthscales() const;

        /// @brief Returns the scale-mixture shape parameter alpha in user units.
        /// @details Stored value lies strictly inside alpha_interval().
        /// @return Current shape parameter alpha.
        [[nodiscard]] double alpha() const noexcept { return alpha_.value(); }

        /// @brief Returns the open intervals bounding each lengthscale.
        /// @details Order matches lengthscales() and the leading dims parameter slots.
        /// @return Vector of bounding open Interval descriptors for lengthscales.
        [[nodiscard]] std::vector<Interval> lengthscale_intervals() const;

        /// @brief Returns the open interval bounding alpha.
        /// @details Corresponds to the trailing parameter slot at index num_params() - 1.
        /// @return Bounding open Interval for alpha.
        [[nodiscard]] Interval alpha_interval() const noexcept { return alpha_.interval(); }

        /// @brief Writes [log(lengthscales)..., log(alpha)] into out.
        /// @details Populates out[0..dims-1] with log lengthscales and out[dims] with log(alpha).
        /// @param out Destination span of length num_params().
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
        [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

        /// @brief Atomically updates lengthscales and alpha from log-space parameters.
        /// @details Validates all dims + 1 parameters before committing any change to lengthscales_ or alpha_.
        /// @param params Input span of length num_params() ordered [log(lengthscales)..., log(alpha)].
        /// @return Empty expected on success or GprError on count mismatch, non-finite input, or bound error.
        /// @retval GprErrorCode::LengthMismatch When params.size() != num_params().
        /// @retval GprErrorCode::NonFiniteInput When any parameter is NaN or infinite.
        /// @retval GprErrorCode::InvalidInterval When any exp(param) overflows or leaves its open interval.
        [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

        /// @brief Writes [lengthscale_intervals..., alpha_interval] into out.
        /// @details Populates out[0..dims-1] and out[dims] in parameter order.
        /// @param out Destination span of length num_params().
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != num_params().
        [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

        /// @brief Checks value equality between two RationalQuadraticArdKernel instances.
        /// @details Compares per-dimension lengthscales, alpha, and all bounding intervals.
        /// @param lhs Left-hand RationalQuadraticArdKernel operand.
        /// @param rhs Right-hand RationalQuadraticArdKernel operand.
        /// @return True if all parameters and intervals match.
        [[nodiscard]] friend bool operator==(
            const RationalQuadraticArdKernel& lhs, const RationalQuadraticArdKernel& rhs
        ) noexcept = default;

    private:
        /// @brief Constructs a RationalQuadraticArdKernel from validated lengthscales and alpha.
        /// @details Invoked by create() factories after per-dimension and alpha validation succeed.
        /// @param lengthscales Non-empty vector of validated BoundedParam lengthscales.
        /// @param alpha Validated bounded alpha parameter.
        RationalQuadraticArdKernel(std::vector<BoundedParam> lengthscales, BoundedParam alpha) noexcept :
            lengthscales_(std::move(lengthscales)),
            alpha_(alpha)
        {
        }

        /// @brief Per-dimension strictly positive lengthscale parameters (indices [0, dims)).
        std::vector<BoundedParam> lengthscales_;
        /// @brief Strictly positive scale-mixture shape parameter alpha (index dims).
        BoundedParam alpha_;
    };

    /// @brief Constant kernel: k(x, x') = constant.
    /// @details Models a positive output signal variance multiplier or constant prior offset in log space.
    class ConstantKernel
    {
    public:
        /// @brief Constructs a constant kernel with value inside Interval::default_positive().
        /// @details Requires constant strictly inside (1e-5, 1e5).
        /// @param constant Positive constant amplitude in user units.
        /// @return Validated ConstantKernel or GprError if constant is out of range.
        /// @retval GprErrorCode::InvalidInterval When constant is non-finite or outside default bounds.
        [[nodiscard]] static std::expected<ConstantKernel, GprError> create(double constant);

        /// @brief Constructs a constant kernel with value inside the given positive interval.
        /// @details Validates that interval is strictly positive (0 < lo < hi) and contains constant.
        /// @param constant Positive constant amplitude in user units.
        /// @param interval Strictly positive open interval bounding constant.
        /// @return Validated ConstantKernel or GprError on invalid interval or out-of-range value.
        /// @retval GprErrorCode::InvalidInterval When interval is invalid or does not contain constant.
        [[nodiscard]] static std::expected<ConstantKernel, GprError> create(double constant, Interval interval);

        /// @brief Returns the number of hyperparameters.
        /// @details Always returns 1.
        /// @return Parameter count (1).
        [[nodiscard]] static constexpr std::size_t num_params() noexcept { return 1; }

        /// @brief Returns the constant amplitude value in user units.
        /// @details Stored value lies strictly inside interval().
        /// @return Current constant amplitude.
        [[nodiscard]] double constant() const noexcept { return constant_.value(); }

        /// @brief Returns the open interval bounding the constant value.
        /// @details Defines strict positivity and optimization constraints.
        /// @return Bounding open Interval.
        [[nodiscard]] Interval interval() const noexcept { return constant_.interval(); }

        /// @brief Writes log(constant) into out[0].
        /// @details Exposes unconstrained log-space parameterization for optimization.
        /// @param out Destination span of length 1.
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
        [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

        /// @brief Updates constant from log-space params[0], failing atomically if out of bounds.
        /// @details Evaluates exp(params[0]) and leaves state unchanged on validation failure.
        /// @param params Input span of length 1 containing log(constant).
        /// @return Empty expected on success or GprError on count mismatch, non-finite input, or bound error.
        /// @retval GprErrorCode::LengthMismatch When params.size() != 1.
        /// @retval GprErrorCode::NonFiniteInput When params[0] is NaN or infinite.
        /// @retval GprErrorCode::InvalidInterval When exp(params[0]) overflows or falls outside interval().
        [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

        /// @brief Writes the open interval into out[0].
        /// @details Populates the bounding Interval for parameter index 0.
        /// @param out Destination span of length 1.
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
        [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

        /// @brief Checks value equality between two ConstantKernel instances.
        /// @details Compares constant value and bounding interval.
        /// @param lhs Left-hand ConstantKernel operand.
        /// @param rhs Right-hand ConstantKernel operand.
        /// @return True if constant and interval match.
        [[nodiscard]] friend constexpr bool operator==(
            const ConstantKernel& lhs, const ConstantKernel& rhs
        ) noexcept = default;

    private:
        /// @brief Constructs a ConstantKernel from a validated bounded constant parameter.
        /// @details Invoked by create() factories after positive interval validation.
        /// @param constant Validated bounded constant parameter.
        explicit constexpr ConstantKernel(BoundedParam constant) noexcept : constant_(constant) {}

        /// @brief Strictly positive constant amplitude parameter with its open bounding interval.
        BoundedParam constant_;
    };

    /// @brief Linear dot-product kernel: k(x, x') = variance * (x . x').
    /// @details Models linear trends in input coordinates with a strictly positive variance multiplier.
    class LinearKernel
    {
    public:
        /// @brief Constructs a linear kernel with variance inside Interval::default_positive().
        /// @details Requires variance strictly inside (1e-5, 1e5).
        /// @param variance Positive slope variance multiplier in user units.
        /// @return Validated LinearKernel or GprError if variance is out of range.
        /// @retval GprErrorCode::InvalidInterval When variance is non-finite or outside default bounds.
        [[nodiscard]] static std::expected<LinearKernel, GprError> create(double variance);

        /// @brief Constructs a linear kernel with variance inside the given positive interval.
        /// @details Validates that interval is strictly positive (0 < lo < hi) and contains variance.
        /// @param variance Positive slope variance multiplier in user units.
        /// @param interval Strictly positive open interval bounding variance.
        /// @return Validated LinearKernel or GprError on invalid interval or out-of-range value.
        /// @retval GprErrorCode::InvalidInterval When interval is invalid or does not contain variance.
        [[nodiscard]] static std::expected<LinearKernel, GprError> create(double variance, Interval interval);

        /// @brief Returns the number of hyperparameters.
        /// @details Always returns 1.
        /// @return Parameter count (1).
        [[nodiscard]] static constexpr std::size_t num_params() noexcept { return 1; }

        /// @brief Returns the variance multiplier in user units.
        /// @details Stored value lies strictly inside interval().
        /// @return Current variance multiplier.
        [[nodiscard]] double variance() const noexcept { return variance_.value(); }

        /// @brief Returns the open interval bounding the variance multiplier.
        /// @details Defines strict positivity and optimization constraints.
        /// @return Bounding open Interval.
        [[nodiscard]] Interval interval() const noexcept { return variance_.interval(); }

        /// @brief Writes log(variance) into out[0].
        /// @details Exposes unconstrained log-space parameterization for optimization.
        /// @param out Destination span of length 1.
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
        [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

        /// @brief Updates variance from log-space params[0], failing atomically if out of bounds.
        /// @details Evaluates exp(params[0]) and leaves state unchanged on validation failure.
        /// @param params Input span of length 1 containing log(variance).
        /// @return Empty expected on success or GprError on count mismatch, non-finite input, or bound error.
        /// @retval GprErrorCode::LengthMismatch When params.size() != 1.
        /// @retval GprErrorCode::NonFiniteInput When params[0] is NaN or infinite.
        /// @retval GprErrorCode::InvalidInterval When exp(params[0]) overflows or falls outside interval().
        [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

        /// @brief Writes the open interval into out[0].
        /// @details Populates the bounding Interval for parameter index 0.
        /// @param out Destination span of length 1.
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
        [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

        /// @brief Checks value equality between two LinearKernel instances.
        /// @details Compares variance multiplier and bounding interval.
        /// @param lhs Left-hand LinearKernel operand.
        /// @param rhs Right-hand LinearKernel operand.
        /// @return True if variance and interval match.
        [[nodiscard]] friend constexpr bool operator==(
            const LinearKernel& lhs, const LinearKernel& rhs
        ) noexcept = default;

    private:
        /// @brief Constructs a LinearKernel from a validated bounded variance parameter.
        /// @details Invoked by create() factories after positive interval validation.
        /// @param variance Validated bounded variance multiplier parameter.
        explicit constexpr LinearKernel(BoundedParam variance) noexcept : variance_(variance) {}

        /// @brief Strictly positive variance multiplier parameter with its open bounding interval.
        BoundedParam variance_;
    };

    /// @brief Independent white-noise kernel: k(x_i, x_j) = variance * delta_{i, j} on training Gram only.
    /// @details Contributes variance to the diagonal of square training Gram matrices and training-diagonal
    /// queries, and contributes 0 to rectangular cross-covariances between distinct point sets.
    class WhiteKernel
    {
    public:
        /// @brief Constructs a white-noise kernel with variance inside Interval::default_positive().
        /// @details Requires variance strictly inside (1e-5, 1e5).
        /// @param variance Positive white-noise variance in user units.
        /// @return Validated WhiteKernel or GprError if variance is out of range.
        /// @retval GprErrorCode::InvalidInterval When variance is non-finite or outside default bounds.
        [[nodiscard]] static std::expected<WhiteKernel, GprError> create(double variance);

        /// @brief Constructs a white-noise kernel with variance inside the given positive interval.
        /// @details Validates that interval is strictly positive (0 < lo < hi) and contains variance.
        /// @param variance Positive white-noise variance in user units.
        /// @param interval Strictly positive open interval bounding variance.
        /// @return Validated WhiteKernel or GprError on invalid interval or out-of-range value.
        /// @retval GprErrorCode::InvalidInterval When interval is invalid or does not contain variance.
        [[nodiscard]] static std::expected<WhiteKernel, GprError> create(double variance, Interval interval);

        /// @brief Returns the number of hyperparameters.
        /// @details Always returns 1.
        /// @return Parameter count (1).
        [[nodiscard]] static constexpr std::size_t num_params() noexcept { return 1; }

        /// @brief Returns the noise variance in user units.
        /// @details Stored value lies strictly inside interval().
        /// @return Current white-noise variance.
        [[nodiscard]] double variance() const noexcept { return variance_.value(); }

        /// @brief Returns the open interval bounding the noise variance.
        /// @details Defines strict positivity and optimization constraints.
        /// @return Bounding open Interval.
        [[nodiscard]] Interval interval() const noexcept { return variance_.interval(); }

        /// @brief Writes log(variance) into out[0].
        /// @details Exposes unconstrained log-space parameterization for optimization.
        /// @param out Destination span of length 1.
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
        [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

        /// @brief Updates variance from log-space params[0], failing atomically if out of bounds.
        /// @details Evaluates exp(params[0]) and leaves state unchanged on validation failure.
        /// @param params Input span of length 1 containing log(variance).
        /// @return Empty expected on success or GprError on count mismatch, non-finite input, or bound error.
        /// @retval GprErrorCode::LengthMismatch When params.size() != 1.
        /// @retval GprErrorCode::NonFiniteInput When params[0] is NaN or infinite.
        /// @retval GprErrorCode::InvalidInterval When exp(params[0]) overflows or falls outside interval().
        [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

        /// @brief Writes the open interval into out[0].
        /// @details Populates the bounding Interval for parameter index 0.
        /// @param out Destination span of length 1.
        /// @return Empty expected on success or GprError on span length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
        [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

        /// @brief Checks value equality between two WhiteKernel instances.
        /// @details Compares noise variance and bounding interval.
        /// @param lhs Left-hand WhiteKernel operand.
        /// @param rhs Right-hand WhiteKernel operand.
        /// @return True if variance and interval match.
        [[nodiscard]] friend constexpr bool operator==(
            const WhiteKernel& lhs, const WhiteKernel& rhs
        ) noexcept = default;

    private:
        /// @brief Constructs a WhiteKernel from a validated bounded noise variance parameter.
        /// @details Invoked by create() factories after positive interval validation.
        /// @param variance Validated bounded white-noise variance parameter.
        explicit constexpr WhiteKernel(BoundedParam variance) noexcept : variance_(variance) {}

        /// @brief Strictly positive white-noise variance parameter with its open bounding interval.
        BoundedParam variance_;
    };

}  // namespace cxxgpr
