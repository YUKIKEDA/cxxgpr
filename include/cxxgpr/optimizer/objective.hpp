// SPDX-License-Identifier: Apache-2.0
/// @file objective.hpp
/// @brief Abstract hyperparameter optimization objective hierarchy, result types, and traits.
/// @details Defines Objective, Differentiable, and TwiceDifferentiable operating in log-theta space
/// alongside open user-space Interval bounds, plus OptimizerResult and RestartConfig.
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/interval.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

/// @brief Hyperparameter optimization interfaces, coordinate transforms, and solver implementations.
namespace cxxgpr::optimizer
{

    /// @brief Base optimization objective operating in log-theta parameter space with user-unit open bounds.
    /// @details Exposes parameter dimension, scalar objective evaluation, optional incremental evaluation
    /// for single-coordinate steps, and open parameter intervals via fill_intervals().
    class Objective
    {
    public:
        /// @brief Default-constructs an objective interface instance.
        /// @details Provided for derived objective classes.
        Objective() = default;
        /// @brief Copy-constructs an objective interface instance.
        /// @details Defaults copy construction for derived objectives.
        Objective(const Objective&) = default;
        /// @brief Copy-assigns an objective interface instance.
        /// @details Defaults copy assignment for derived objectives.
        /// @return Reference to this Objective.
        Objective& operator=(const Objective&) = default;
        /// @brief Move-constructs an objective interface instance.
        /// @details Defaults move construction for derived objectives.
        Objective(Objective&&) noexcept = default;
        /// @brief Move-assigns an objective interface instance.
        /// @details Defaults move assignment for derived objectives.
        /// @return Reference to this Objective.
        Objective& operator=(Objective&&) noexcept = default;
        /// @brief Virtual destructor for polymorphic objective implementations.
        /// @details Allows safe deletion of derived objective objects via base pointers or references.
        virtual ~Objective() = default;

        /// @brief Returns the number of scalar log-space parameters.
        /// @details Defines the expected length of parameter, gradient, and interval spans.
        /// @return Objective parameter count.
        [[nodiscard]] virtual std::size_t dim() const noexcept = 0;

        /// @brief Evaluates the scalar objective at log-space parameter vector params.
        /// @details Implementations evaluate the negative marginal log-likelihood or custom loss at params.
        /// @param[in] params Log-space parameter vector of length dim().
        /// @return Finite objective value on success, or GprError if evaluation fails.
        [[nodiscard]] virtual std::expected<double, GprError> value(std::span<const double> params) = 0;

        /// @brief Evaluates the scalar objective when only changed_indices in params changed since the last step.
        /// @details Optimizers supporting incremental coordinate proposals (such as FastSimulatedAnnealing)
        /// supply changed_indices so objectives backed by LeafCache recompute only dirty leaves; defaults
        /// to calling value(params).
        /// @param[in] params Candidate log-space parameter vector of length dim().
        /// @param[in] changed_indices Non-empty list of parameter indices modified relative to the previous call.
        /// @return Finite objective value on success, or GprError if evaluation fails.
        [[nodiscard]] virtual std::expected<double, GprError> value_at_changes(
            std::span<const double> params, [[maybe_unused]] std::span<const std::size_t> changed_indices
        )
        {
            // Fall back to full objective evaluation when derived classes do not cache per-leaf state.
            return value(params);
        }

        /// @brief Fills out with the open Interval (lo, hi) in user units for each parameter.
        /// @details Optimizer LogBounds transforms each interval via (log(lo), log(hi)) into unconstrained
        /// logit coordinates; implementations must enforce out.size() == dim().
        /// @param[out] out Destination span of length dim() receiving open user-space parameter intervals.
        /// @return Empty expected on success, or GprError on length mismatch or invalid bounds.
        [[nodiscard]] virtual std::expected<void, GprError> fill_intervals(std::span<Interval> out) const = 0;
    };

    /// @brief Objective additionally providing exact first-order analytical gradients in log-theta space.
    /// @details Required by gradient-based first-order solvers such as Lbfgs.
    class Differentiable : public Objective
    {
    public:
        /// @brief Writes the analytical gradient with respect to log-space params into out_grad.
        /// @details Both params and out_grad must have length dim().
        /// @param[in] params Log-space parameter vector of length dim().
        /// @param[out] out_grad Destination span of length dim() receiving d(value)/d(params).
        /// @return Empty expected on success, or GprError if evaluation or validation fails.
        [[nodiscard]] virtual std::expected<void, GprError> gradient_into(
            std::span<const double> params, std::span<double> out_grad
        ) = 0;

        /// @brief Evaluates both objective value and gradient into out_grad in a single pass.
        /// @details Shares the Cholesky factorization between objective value and gradient evaluation.
        /// @param[in] params Log-space parameter vector of length dim().
        /// @param[out] out_grad Destination span of length dim() receiving d(value)/d(params).
        /// @return Scalar objective value on success, or GprError if evaluation or validation fails.
        [[nodiscard]] virtual std::expected<double, GprError> value_and_gradient_into(
            std::span<const double> params, std::span<double> out_grad
        ) = 0;
    };

    /// @brief Differentiable objective additionally providing second-order Hessian matrices in log-theta space.
    /// @details Required by second-order Newton/trust-region solvers such as TrustRegion.
    class TwiceDifferentiable : public Differentiable
    {
    public:
        /// @brief Writes the symmetric dim x dim Hessian matrix (row/col-major flattened) into out_hess.
        /// @details Entry (i, j) is stored at (i * dim()) + j (and symmetrically at (j * dim()) + i).
        /// @param[in] params Log-space parameter vector of length dim().
        /// @param[out] out_hess Destination span of length dim() * dim() receiving d^2(value)/(dtheta_i dtheta_j).
        /// @return Empty expected on success, or GprError if evaluation or validation fails.
        [[nodiscard]] virtual std::expected<void, GprError> hessian_into(
            std::span<const double> params, std::span<double> out_hess
        ) = 0;

        /// @brief Evaluates value, gradient, and Hessian together; defaults to sequential calls.
        /// @details Evaluates value_and_gradient_into(params, out_grad) first and then hessian_into(params, out_hess);
        /// derived GPR objectives override this to reuse a single Cholesky factor across all three outputs.
        /// @param[in] params Log-space parameter vector of length dim().
        /// @param[out] out_grad Destination span of length dim() receiving the log-space gradient.
        /// @param[out] out_hess Destination span of length dim() * dim() receiving the symmetric Hessian.
        /// @return Scalar objective value on success, or GprError if gradient or Hessian evaluation fails.
        [[nodiscard]] virtual std::expected<double, GprError> value_gradient_hessian_into(
            std::span<const double> params, std::span<double> out_grad, std::span<double> out_hess
        )
        {
            // Default implementation delegates sequentially to value+gradient and Hessian evaluations.
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

    /// @brief Outcome returned by an optimizer run in log-theta space.
    /// @details Holds the best feasible log-space parameter vector, corresponding objective value,
    /// completed iteration count, and convergence status.
    struct OptimizerResult
    {
        /// @brief Best log-space parameter vector found.
        std::vector<double> params;
        /// @brief Objective value at params.
        double value{0.0};
        /// @brief Number of optimization iterations performed.
        std::size_t iterations{0};
        /// @brief True if the optimizer satisfied its convergence criterion.
        bool converged{false};

        /// @brief Checks value equality between two OptimizerResult instances.
        /// @details Compares params, value, iterations, and converged.
        /// @param[in] other OptimizerResult instance to compare against.
        /// @return True if all fields compare equal.
        [[nodiscard]] bool operator==(const OptimizerResult& other) const noexcept = default;
    };

    /// @brief Configuration for deterministic multi-start optimization restarts.
    /// @details Controls how many extra log-uniform random starting points are explored using
    /// Xoshiro256PlusPlus seeded from seed.
    struct RestartConfig
    {
        /// @brief Number of additional random starting points beyond the caller-supplied initial point.
        std::size_t count{0};
        /// @brief Seed for Xoshiro256++ deterministic restart point sampling.
        std::uint64_t seed{0};

        /// @brief Checks value equality between two RestartConfig instances.
        /// @details Compares both count and seed.
        /// @param[in] other RestartConfig instance to compare against.
        /// @return True if both restart configuration fields match.
        [[nodiscard]] constexpr bool operator==(const RestartConfig& other) const noexcept = default;
    };

}  // namespace cxxgpr::optimizer
