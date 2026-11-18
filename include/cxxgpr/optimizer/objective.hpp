// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/interval.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

namespace cxxgpr::optimizer
{

    /// Base optimization objective operating in unconstrained log-theta parameter space, with open
    /// parameter bounds provided in user space via fill_intervals().
    class Objective
    {
    public:
        Objective() = default;
        Objective(const Objective&) = default;
        Objective& operator=(const Objective&) = default;
        Objective(Objective&&) noexcept = default;
        Objective& operator=(Objective&&) noexcept = default;
        virtual ~Objective() = default;

        /// Returns the number of scalar log-space parameters.
        [[nodiscard]] virtual std::size_t dim() const noexcept = 0;

        /// Evaluates the scalar objective at log-space parameter vector params.
        [[nodiscard]] virtual std::expected<double, GprError> value(std::span<const double> params) = 0;

        /// Evaluates the scalar objective when only changed_indices in params changed relative to
        /// the previous evaluation; defaults to value(params).
        [[nodiscard]] virtual std::expected<double, GprError> value_at_changes(
            std::span<const double> params, [[maybe_unused]] std::span<const std::size_t> changed_indices
        )
        {
            return value(params);
        }

        /// Fills out with the open Interval (lo, hi) in user units for each parameter.
        [[nodiscard]] virtual std::expected<void, GprError> fill_intervals(std::span<Interval> out) const = 0;
    };

    /// Objective additionally providing exact first-order analytical gradients in log-theta space.
    class Differentiable : public Objective
    {
    public:
        /// Writes the gradient wrt log-space params into out_grad.
        [[nodiscard]] virtual std::expected<void, GprError> gradient_into(
            std::span<const double> params, std::span<double> out_grad
        ) = 0;

        /// Evaluates both objective value and gradient into out_grad in a single pass.
        [[nodiscard]] virtual std::expected<double, GprError> value_and_gradient_into(
            std::span<const double> params, std::span<double> out_grad
        ) = 0;
    };

    /// Differentiable objective additionally providing second-order Hessian matrices in log-theta
    /// space.
    class TwiceDifferentiable : public Differentiable
    {
    public:
        /// Writes the symmetric dim x dim Hessian matrix (row/col-major flattened) into out_hess.
        [[nodiscard]] virtual std::expected<void, GprError> hessian_into(
            std::span<const double> params, std::span<double> out_hess
        ) = 0;

        /// Evaluates value, gradient, and Hessian together; defaults to value_and_gradient_into +
        /// hessian_into.
        [[nodiscard]] virtual std::expected<double, GprError> value_gradient_hessian_into(
            std::span<const double> params, std::span<double> out_grad, std::span<double> out_hess
        )
        {
            auto val = value_and_gradient_into(params, out_grad);
            if (!val.has_value())
            {
                return val;
            }
            if (auto h = hessian_into(params, out_hess); !h.has_value())
            {
                return std::unexpected(std::move(h).error());
            }
            return *val;
        }
    };

    /// Outcome returned by an optimizer run in log-theta space.
    struct OptimizerResult
    {
        /// Best log-space parameter vector found.
        std::vector<double> params;
        /// Objective value at params.
        double value{0.0};
        /// Number of optimization iterations performed.
        std::size_t iterations{0};
        /// True if the optimizer satisfied its convergence criterion.
        bool converged{false};

        /// Checks value equality between two OptimizerResult instances.
        [[nodiscard]] bool operator==(const OptimizerResult& other) const noexcept = default;
    };

    /// Configuration for deterministic multi-start optimization restarts.
    struct RestartConfig
    {
        /// Number of additional random starting points beyond the caller-supplied initial point.
        std::size_t count{0};
        /// Seed for Xoshiro256++ deterministic restart point sampling.
        std::uint64_t seed{0};

        /// Checks value equality between two RestartConfig instances.
        [[nodiscard]] constexpr bool operator==(const RestartConfig& other) const noexcept = default;
    };

    /// Compile-time traits describing optimizer behavior.
    template <typename O> struct OptimizerTraits
    {
        /// True when the optimizer evaluates single-coordinate proposals via value_at_changes().
        static constexpr bool uses_change_indices = O::uses_change_indices;
    };

}  // namespace cxxgpr::optimizer
