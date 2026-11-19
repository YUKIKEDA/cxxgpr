// SPDX-License-Identifier: Apache-2.0
/// @file bounds.hpp
/// @brief Log-space parameter bounds, midpoint unit-Jacobian logit transforms, and evaluation cache.
/// @details Declares LogBounds for mapping open positive intervals (lo, hi) between constrained
/// log-space theta in (log(lo), log(hi)) and unconstrained z in R^d with d(theta)/dz = 1 at the
/// midpoint, plus EvalCache for avoiding duplicate Differentiable evaluations at identical points.
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/interval.hpp"
#include "cxxgpr/core/rng.hpp"
#include "cxxgpr/optimizer/objective.hpp"

#include <cstddef>
#include <expected>
#include <span>
#include <vector>

/// @brief Hyperparameter optimization interfaces, coordinate transforms, and solver implementations.
namespace cxxgpr::optimizer
{

    /// @brief Log-space open parameter bounds (log(lo), log(hi)) with midpoint unit-Jacobian logit mapping.
    /// @details Maps strictly interior log-theta coordinates to and from unconstrained z in R^d via
    /// theta_i = lo_i + width_i * sigmoid(4 * z_i / width_i), so d(theta_i)/dz_i = 1 at the midpoint
    /// z_i = 0, and provides first- and second-order chain-rule helpers plus log-uniform restart sampling.
    class LogBounds
    {
    public:
        /// @brief Builds log-space bounds from Objective::fill_intervals(), requiring 0 < lo < hi.
        /// @details Queries obj.dim(), fills user-space Interval bounds, and delegates to from_intervals().
        /// @param[in] obj Optimization objective providing open user-space parameter intervals.
        /// @return Validated LogBounds, or GprError if obj is zero-dimensional or has invalid bounds.
        /// @retval GprErrorCode::EmptyInput When obj.dim() == 0.
        /// @retval GprErrorCode::InvalidInterval When any interval violates 0 < lo < hi or finite logs.
        [[nodiscard]] static std::expected<LogBounds, GprError> from_objective(const Objective& obj);

        /// @brief Builds log-space bounds from explicit user-space open intervals (0 < lo < hi).
        /// @details Validates finiteness, strict positivity lo > 0, ordering lo < hi, and finite
        /// logarithms before constructing LogBounds.
        /// @param[in] intervals Span of open user-space intervals (lo, hi).
        /// @return Validated LogBounds, or GprError if intervals is empty or any bound is invalid.
        /// @retval GprErrorCode::EmptyInput When intervals is empty.
        /// @retval GprErrorCode::InvalidInterval When any interval violates 0 < lo < hi or finite logs.
        [[nodiscard]] static std::expected<LogBounds, GprError> from_intervals(std::span<const Interval> intervals);

        /// @brief Returns the parameter dimension.
        /// @details Matches the number of lower and upper log-space bounds.
        /// @return Number of bounded parameters.
        [[nodiscard]] std::size_t dim() const noexcept { return lo_log_.size(); }

        /// @brief Returns the exclusive lower bounds in log-theta space.
        /// @details Stores log(lo_i) for each parameter i in [0, dim()).
        /// @return Immutable span of lower log-space bounds.
        [[nodiscard]] std::span<const double> lo_log() const noexcept { return lo_log_; }

        /// @brief Returns the exclusive upper bounds in log-theta space.
        /// @details Stores log(hi_i) for each parameter i in [0, dim()).
        /// @return Immutable span of upper log-space bounds.
        [[nodiscard]] std::span<const double> hi_log() const noexcept { return hi_log_; }

        /// @brief Validates that theta is finite, has length dim(), and lies strictly inside (lo_log, hi_log).
        /// @details Checks non-emptiness, length parity with dim(), finiteness, and open interval containment.
        /// @param[in] theta Candidate log-space parameter vector.
        /// @return Empty expected on success, or GprError on invalid count, non-finite value, or bound breach.
        /// @retval GprErrorCode::EmptyInput When theta is empty.
        /// @retval GprErrorCode::LengthMismatch When theta.size() != dim().
        /// @retval GprErrorCode::NonFiniteInput When any entry of theta is NaN or +/-Inf.
        /// @retval GprErrorCode::InvalidInterval When any theta[i] lies outside (lo_log_[i], hi_log_[i]).
        [[nodiscard]] std::expected<void, GprError> validate_log_params(std::span<const double> theta) const;

        /// @brief Maps constrained log-space theta into unconstrained z with d(theta)/dz = 1 at the midpoint.
        /// @details Validates theta strictly inside (lo_log, hi_log) and applies
        /// z_i = 0.25 * width_i * log(p_i / (1 - p_i)) where p_i = (theta_i - lo_log_i) / width_i.
        /// @param[in] theta Strictly interior constrained log-space vector of length dim().
        /// @param[out] out_z Destination span of length dim() receiving unconstrained logit coordinates.
        /// @return Empty expected on success, or GprError on invalid theta or out_z length mismatch.
        /// @retval GprErrorCode::EmptyInput When theta is empty.
        /// @retval GprErrorCode::LengthMismatch When theta.size() != dim() or out_z.size() != dim().
        /// @retval GprErrorCode::NonFiniteInput When theta contains NaN or +/-Inf.
        /// @retval GprErrorCode::InvalidInterval When any theta[i] lies outside (lo_log_[i], hi_log_[i]).
        [[nodiscard]] std::expected<void, GprError> to_unconstrained(
            std::span<const double> theta, std::span<double> out_z
        ) const;

        /// @brief Maps unconstrained z back into strictly interior log-space theta.
        /// @details Evaluates clamped stable sigmoid s_i = sigmoid(4 * z_i / width_i) and clamps
        /// lo_log_i + width_i * s_i to (nextafter(lo, hi), nextafter(hi, lo)).
        /// @param[in] z Unconstrained logit coordinate vector of length dim().
        /// @param[out] out_theta Destination span of length dim() receiving interior log-space theta.
        void to_constrained(std::span<const double> z, std::span<double> out_theta) const noexcept;

        /// @brief Maps unconstrained z to theta and computes first derivatives d(theta_i)/dz_i.
        /// @details Evaluates out_theta[i] and d1[i] = 4 * s_i * (1 - s_i), equal to 1 at z_i = 0.
        /// @param[in] z Unconstrained logit coordinate vector of length dim().
        /// @param[out] out_theta Destination span of length dim() receiving interior log-space theta.
        /// @param[out] out_dtheta_dz Destination span of length dim() receiving Jacobian diagonal d(theta)/dz.
        void to_constrained_with_jacobian(
            std::span<const double> z, std::span<double> out_theta, std::span<double> out_dtheta_dz
        ) const noexcept;

        /// @brief Maps unconstrained z to theta and computes first and second derivatives d(theta)/dz and
        /// d^2(theta)/dz^2.
        /// @details Computes out_theta[i], d1[i] = 4 * s_i * (1 - s_i), and
        /// d2[i] = (16 / width_i) * s_i * (1 - s_i) * (1 - 2 * s_i).
        /// @param[in] z Unconstrained logit coordinate vector of length dim().
        /// @param[out] out_theta Destination span of length dim() receiving interior log-space theta.
        /// @param[out] out_d1 Destination span of length dim() receiving first derivatives d(theta_i)/dz_i.
        /// @param[out] out_d2 Destination span of length dim() receiving second derivatives d^2(theta_i)/dz_i^2.
        void to_constrained_with_derivatives(
            std::span<const double> z, std::span<double> out_theta, std::span<double> out_d1, std::span<double> out_d2
        ) const noexcept;

        /// @brief Applies the first-order chain rule g_z[i] = g_theta[i] * d1[i].
        /// @details Scales log-theta gradient entries by the diagonal Jacobian d(theta)/dz.
        /// @param[in] g_theta Gradient with respect to constrained log-space theta (length dim()).
        /// @param[in] d1 First derivatives d(theta_i)/dz_i (length dim()).
        /// @param[out] out_g_z Destination span receiving gradient with respect to unconstrained z.
        static void chain_gradient(
            std::span<const double> g_theta, std::span<const double> d1, std::span<double> out_g_z
        ) noexcept;

        /// @brief Applies the second-order chain rule H_z = J^T * H_theta * J + diag(g_theta * d2).
        /// @details Transforms the dim x dim log-theta Hessian into unconstrained logit coordinates,
        /// adding the coordinate-transform curvature term g_theta[i] * d2[i] on the diagonal.
        /// @param[in] h_theta Flattened dim x dim Hessian in log-theta space.
        /// @param[in] g_theta Gradient in log-theta space (length dim()).
        /// @param[in] d1 First derivatives d(theta_i)/dz_i (length dim()).
        /// @param[in] d2 Second derivatives d^2(theta_i)/dz_i^2 (length dim()).
        /// @param[out] out_h_z Destination flattened dim x dim span receiving Hessian in unconstrained z space.
        static void chain_hessian(
            std::span<const double> h_theta,
            std::span<const double> g_theta,
            std::span<const double> d1,
            std::span<const double> d2,
            std::span<double> out_h_z
        ) noexcept;

        /// @brief Samples a deterministic log-uniform point strictly inside (lo_log, hi_log).
        /// @details Draws u ~ open_unit() from rng and maps lo_log_i + width_i * u clamped strictly inside.
        /// @param[in,out] rng Deterministic Xoshiro256PlusPlus generator state.
        /// @param[out] out_theta Destination span of length dim() receiving sampled log-space parameters.
        void sample_log_uniform(Xoshiro256PlusPlus& rng, std::span<double> out_theta) const noexcept;

    private:
        /// @brief Constructs LogBounds from pre-validated lower and upper log-space bound vectors.
        /// @details Private so public factories guarantee finiteness and strict lo_log < hi_log ordering.
        /// @param[in] lo_log Lower bounds log(lo) for each parameter.
        /// @param[in] hi_log Upper bounds log(hi) for each parameter.
        LogBounds(std::vector<double> lo_log, std::vector<double> hi_log) noexcept :
            lo_log_(std::move(lo_log)),
            hi_log_(std::move(hi_log))
        {
        }

        /// @brief Exclusive lower bounds in log-theta space.
        std::vector<double> lo_log_;
        /// @brief Exclusive upper bounds in log-theta space.
        std::vector<double> hi_log_;
    };

    /// @brief Single-entry evaluation cache avoiding redundant value/gradient evaluations at identical points.
    /// @details Caches the most recent unconstrained point z, scalar value, and chained gradient g_z for
    /// reuse between Moré–Thuente line-search step acceptance and the start of the next outer iteration.
    class EvalCache
    {
    public:
        /// @brief Clears any cached evaluation.
        /// @details Resets valid_ to false without releasing internal vector capacity.
        void clear() noexcept { valid_ = false; }

        /// @brief Evaluates (or retrieves cached) objective value and gradient at unconstrained z.
        /// @details Checks bitwise/exact equality against last_z_; on a miss, validates finiteness of z,
        /// maps z -> (theta, d1), invokes obj.value_and_gradient_into(), validates finite value/gradient,
        /// chains gradient into out_grad_z, and updates the cache entry.
        /// @param[in,out] obj Differentiable objective evaluated in log-theta space.
        /// @param[in] bounds Log-space bounds governing the z <-> theta transform.
        /// @param[in] z Unconstrained logit parameter vector of length bounds.dim().
        /// @param[in,out] theta_buf Caller-supplied scratch span of length bounds.dim() for constrained theta.
        /// @param[in,out] d1_buf Caller-supplied scratch span of length bounds.dim() for Jacobian d(theta)/dz.
        /// @param[in,out] grad_theta_buf Caller-supplied scratch span of length bounds.dim() for log-theta gradient.
        /// @param[out] out_grad_z Destination span of length bounds.dim() receiving unconstrained gradient g_z.
        /// @return Objective scalar value at z, or GprError if evaluation fails or yields non-finite outputs.
        /// @retval GprErrorCode::NonFiniteInput When z, objective value, or any gradient component is non-finite.
        [[nodiscard]] std::expected<double, GprError> eval_unconstrained(
            Differentiable& obj,
            const LogBounds& bounds,
            std::span<const double> z,
            std::span<double> theta_buf,
            std::span<double> d1_buf,
            std::span<double> grad_theta_buf,
            std::span<double> out_grad_z
        );

    private:
        /// @brief True when last_z_, last_val_, and last_grad_z_ hold a valid cached evaluation.
        bool valid_{false};
        /// @brief Cached unconstrained parameter coordinates from the last successful evaluation.
        std::vector<double> last_z_;
        /// @brief Cached objective scalar value at last_z_.
        double last_val_{0.0};
        /// @brief Cached unconstrained gradient g_z at last_z_.
        std::vector<double> last_grad_z_;
    };

}  // namespace cxxgpr::optimizer
