// SPDX-License-Identifier: Apache-2.0
/// @file bounds.cpp
/// @brief Implementations of LogBounds midpoint unit-Jacobian logit transforms and EvalCache.
/// @details Implements validated open log-space bounds, overflow-safe logit/sigmoid mappings,
/// first- and second-order chain rules, deterministic log-uniform sampling, and cached evaluation.

#include "cxxgpr/optimizer/bounds.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace cxxgpr::optimizer
{
    namespace
    {

        /// @brief Magnitude clamp applied to scaled logit inputs before exp() to avoid overflow.
        constexpr double k_logit_clamp = 36.0;

        /// @brief Evaluates the logistic sigmoid 1 / (1 + exp(-s)) safely without floating-point overflow.
        /// @details Clamps s to [-k_logit_clamp, k_logit_clamp] and branches on the sign of s so exp()
        /// always receives a non-positive argument.
        /// @param s Input real value.
        /// @return Logistic sigmoid value strictly inside (0, 1).
        [[nodiscard]] double stable_sigmoid(double s) noexcept
        {
            // Clamp extreme arguments and choose the negative-exponent branch to prevent overflow.
            const double clamped = std::clamp(s, -k_logit_clamp, k_logit_clamp);
            if (clamped >= 0.0)
            {
                const double e = std::exp(-clamped);
                return 1.0 / (1.0 + e);
            }
            const double e = std::exp(clamped);
            return e / (1.0 + e);
        }

        /// @brief Clamps val strictly inside the open interval (lo, hi) using adjacent representable doubles.
        /// @details Prevents round-off in sigmoid scaling from touching endpoints lo or hi.
        /// @param val Candidate scalar inside or near (lo, hi).
        /// @param lo Exclusive lower bound.
        /// @param hi Exclusive upper bound.
        /// @return Value clamped to [std::nextafter(lo, hi), std::nextafter(hi, lo)].
        [[nodiscard]] double clamp_strict_interior(double val, double lo, double hi) noexcept
        {
            // Step inward by 1 ULP from both endpoints so open-interval checks always hold.
            const double min_val = std::nextafter(lo, hi);
            const double max_val = std::nextafter(hi, lo);
            return std::clamp(val, min_val, max_val);
        }

    }  // namespace

    /// @brief Builds log-space bounds from Objective::fill_intervals(), requiring 0 < lo < hi.
    /// @details Queries obj.dim(), fills user-space intervals, and delegates to from_intervals().
    /// @param obj Optimization objective providing open user-space parameter intervals.
    /// @return Validated LogBounds, or GprError if obj is zero-dimensional or has invalid bounds.
    /// @retval GprErrorCode::EmptyInput When obj.dim() == 0.
    /// @retval GprErrorCode::InvalidInterval When any interval violates 0 < lo < hi or finite logs.
    std::expected<LogBounds, GprError> LogBounds::from_objective(const Objective& obj)
    {
        const std::size_t d = obj.dim();
        if (d == 0U)
        {
            return std::unexpected(GprError::empty_input());
        }
        std::vector<Interval> intervals(d, Interval::default_positive());
        if (auto res = obj.fill_intervals(intervals); !res.has_value())
        {
            return std::unexpected(std::move(res).error());
        }
        return from_intervals(intervals);
    }

    /// @brief Builds log-space bounds from explicit user-space open intervals (0 < lo < hi).
    /// @details Verifies non-emptiness, finiteness, strict positivity, and finite ordered log bounds.
    /// @param intervals Span of open user-space intervals (lo, hi).
    /// @return Validated LogBounds, or GprError if intervals is empty or any bound is invalid.
    /// @retval GprErrorCode::EmptyInput When intervals is empty.
    /// @retval GprErrorCode::InvalidInterval When any interval violates 0 < lo < hi or finite logs.
    std::expected<LogBounds, GprError> LogBounds::from_intervals(std::span<const Interval> intervals)
    {
        if (intervals.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        std::vector<double> lo_log;
        std::vector<double> hi_log;
        lo_log.reserve(intervals.size());
        hi_log.reserve(intervals.size());
        // Convert positive user-space bounds (lo, hi) to ordered finite log-space bounds.
        for (const auto& iv : intervals)
        {
            if (!std::isfinite(iv.lo()) || !std::isfinite(iv.hi()) || iv.lo() <= 0.0 || iv.lo() >= iv.hi())
            {
                return std::unexpected(GprError::invalid_interval_bounds(iv.lo(), iv.hi()));
            }
            const double l = std::log(iv.lo());
            const double u = std::log(iv.hi());
            if (!std::isfinite(l) || !std::isfinite(u) || l >= u)
            {
                return std::unexpected(GprError::invalid_interval_bounds(iv.lo(), iv.hi()));
            }
            lo_log.push_back(l);
            hi_log.push_back(u);
        }
        return LogBounds(std::move(lo_log), std::move(hi_log));
    }

    /// @brief Validates that theta is finite, has length dim(), and lies strictly inside (lo_log, hi_log).
    /// @details Checks empty span, dimension mismatch, non-finite values, and strict open-interval containment.
    /// @param theta Candidate log-space parameter vector.
    /// @return Empty expected on success, or GprError on invalid count, non-finite entry, or bound breach.
    /// @retval GprErrorCode::EmptyInput When theta is empty.
    /// @retval GprErrorCode::LengthMismatch When theta.size() != dim().
    /// @retval GprErrorCode::NonFiniteInput When any entry of theta is NaN or +/-Inf.
    /// @retval GprErrorCode::InvalidInterval When any theta[i] lies outside (lo_log_[i], hi_log_[i]).
    std::expected<void, GprError> LogBounds::validate_log_params(std::span<const double> theta) const
    {
        const std::size_t d = dim();
        if (theta.empty())
        {
            return std::unexpected(GprError::empty_input());
        }
        if (theta.size() != d)
        {
            return std::unexpected(
                GprError::length_mismatch("initial parameter length must match objective dimension")
            );
        }
        for (std::size_t i = 0; i < d; ++i)
        {
            if (!std::isfinite(theta[i]))
            {
                return std::unexpected(GprError::non_finite_input());
            }
            if (theta[i] <= lo_log_[i] || theta[i] >= hi_log_[i])
            {
                return std::unexpected(GprError::invalid_interval_out_of_range(theta[i], lo_log_[i], hi_log_[i]));
            }
        }
        return {};
    }

    /// @brief Maps constrained log-space theta into unconstrained z with d(theta)/dz = 1 at the midpoint.
    /// @details Validates theta inside open bounds, checks out_z length, and applies
    /// z_i = 0.25 * width_i * log(p_i / (1 - p_i)) for normalized coordinate p_i in (0, 1).
    /// @param theta Strictly interior constrained log-space vector of length dim().
    /// @param out_z Destination span of length dim() receiving unconstrained logit coordinates.
    /// @return Empty expected on success, or GprError on invalid theta or out_z length mismatch.
    /// @retval GprErrorCode::EmptyInput When theta is empty.
    /// @retval GprErrorCode::LengthMismatch When theta.size() != dim() or out_z.size() != dim().
    /// @retval GprErrorCode::NonFiniteInput When theta contains NaN or +/-Inf.
    /// @retval GprErrorCode::InvalidInterval When any theta[i] lies outside (lo_log_[i], hi_log_[i]).
    std::expected<void, GprError> LogBounds::to_unconstrained(
        std::span<const double> theta, std::span<double> out_z
    ) const
    {
        if (auto valid = validate_log_params(theta); !valid.has_value())
        {
            return valid;
        }
        const std::size_t d = dim();
        if (out_z.size() != d)
        {
            return std::unexpected(GprError::length_mismatch("unconstrained buffer length must match dimension"));
        }
        // Apply midpoint unit-Jacobian logit: z = (w / 4) * log((theta - lo) / (hi - theta)).
        for (std::size_t i = 0; i < d; ++i)
        {
            const double width = hi_log_[i] - lo_log_[i];
            const double p = (theta[i] - lo_log_[i]) / width;
            out_z[i] = 0.25 * width * std::log(p / (1.0 - p));
        }
        return {};
    }

    /// @brief Maps unconstrained z back into strictly interior log-space theta.
    /// @details Evaluates lo_log_i + width_i * sigmoid(4 * z_i / width_i) clamped to strict interior.
    /// @param z Unconstrained logit coordinate vector of length dim().
    /// @param out_theta Destination span of length dim() receiving interior log-space theta.
    void LogBounds::to_constrained(std::span<const double> z, std::span<double> out_theta) const noexcept
    {
        const std::size_t d = dim();
        for (std::size_t i = 0; i < d; ++i)
        {
            const double width = hi_log_[i] - lo_log_[i];
            const double sig = stable_sigmoid((4.0 * z[i]) / width);
            out_theta[i] = clamp_strict_interior(lo_log_[i] + (width * sig), lo_log_[i], hi_log_[i]);
        }
    }

    /// @brief Maps unconstrained z to theta and computes first derivatives d(theta_i)/dz_i.
    /// @details Computes theta_i and Jacobian diagonal d1_i = 4 * s_i * (1 - s_i) (unit at midpoint z_i = 0).
    /// @param z Unconstrained logit coordinate vector of length dim().
    /// @param out_theta Destination span of length dim() receiving interior log-space theta.
    /// @param out_dtheta_dz Destination span of length dim() receiving d(theta_i)/dz_i.
    void LogBounds::to_constrained_with_jacobian(
        std::span<const double> z, std::span<double> out_theta, std::span<double> out_dtheta_dz
    ) const noexcept
    {
        const std::size_t d = dim();
        for (std::size_t i = 0; i < d; ++i)
        {
            const double width = hi_log_[i] - lo_log_[i];
            const double sig = stable_sigmoid((4.0 * z[i]) / width);
            out_theta[i] = clamp_strict_interior(lo_log_[i] + (width * sig), lo_log_[i], hi_log_[i]);
            // Since theta = lo + w * sig(4 * z / w), d(theta)/dz = 4 * sig * (1 - sig) (= 1 at z = 0).
            out_dtheta_dz[i] = 4.0 * sig * (1.0 - sig);
        }
    }

    /// @brief Maps unconstrained z to theta and computes first and second derivatives d(theta)/dz and d^2(theta)/dz^2.
    /// @details Computes theta_i, d1_i = 4 * s_i * (1 - s_i), and d2_i = (16 / width_i) * s_i * (1 - s_i) * (1 - 2 *
    /// s_i).
    /// @param z Unconstrained logit coordinate vector of length dim().
    /// @param out_theta Destination span of length dim() receiving interior log-space theta.
    /// @param out_d1 Destination span of length dim() receiving first derivatives d(theta_i)/dz_i.
    /// @param out_d2 Destination span of length dim() receiving second derivatives d^2(theta_i)/dz_i^2.
    void LogBounds::to_constrained_with_derivatives(
        std::span<const double> z, std::span<double> out_theta, std::span<double> out_d1, std::span<double> out_d2
    ) const noexcept
    {
        const std::size_t d = dim();
        for (std::size_t i = 0; i < d; ++i)
        {
            const double width = hi_log_[i] - lo_log_[i];
            const double sig = stable_sigmoid((4.0 * z[i]) / width);
            out_theta[i] = clamp_strict_interior(lo_log_[i] + (width * sig), lo_log_[i], hi_log_[i]);
            // Propagate first and second derivatives of sig(4 * z / width) for Hessian chain rule.
            const double sig_om = sig * (1.0 - sig);
            out_d1[i] = 4.0 * sig_om;
            out_d2[i] = (16.0 / width) * sig_om * (1.0 - (2.0 * sig));
        }
    }

    /// @brief Applies the first-order chain rule g_z[i] = g_theta[i] * d1[i].
    /// @details Multiplies each constrained gradient entry by the diagonal Jacobian factor d(theta_i)/dz_i.
    /// @param g_theta Gradient with respect to constrained log-space theta (length dim()).
    /// @param d1 First derivatives d(theta_i)/dz_i (length dim()).
    /// @param out_g_z Destination span receiving gradient with respect to unconstrained z.
    void LogBounds::chain_gradient(
        std::span<const double> g_theta, std::span<const double> d1, std::span<double> out_g_z
    ) noexcept
    {
        const std::size_t d = g_theta.size();
        for (std::size_t i = 0; i < d; ++i)
        {
            out_g_z[i] = g_theta[i] * d1[i];
        }
    }

    /// @brief Applies the second-order chain rule H_z = J^T * H_theta * J + diag(g_theta * d2).
    /// @details Scales entry (row, col) by d1[row] * d1[col] and adds g_theta[row] * d2[row] on the diagonal.
    /// @param h_theta Flattened dim x dim Hessian in log-theta space.
    /// @param g_theta Gradient in log-theta space (length dim()).
    /// @param d1 First derivatives d(theta_i)/dz_i (length dim()).
    /// @param d2 Second derivatives d^2(theta_i)/dz_i^2 (length dim()).
    /// @param out_h_z Destination flattened dim x dim span receiving Hessian in unconstrained z space.
    void LogBounds::chain_hessian(
        std::span<const double> h_theta,
        std::span<const double> g_theta,
        std::span<const double> d1,
        std::span<const double> d2,
        std::span<double> out_h_z
    ) noexcept
    {
        const std::size_t d = g_theta.size();
        // Compute H_z(row, col) = H_theta(row, col) * d1[row] * d1[col] + delta(row, col) * g_theta[row] * d2[row].
        for (std::size_t col = 0; col < d; ++col)
        {
            for (std::size_t row = 0; row < d; ++row)
            {
                double val = h_theta[(col * d) + row] * d1[row] * d1[col];
                if (row == col)
                {
                    val += g_theta[row] * d2[row];
                }
                out_h_z[(col * d) + row] = val;
            }
        }
    }

    /// @brief Samples a deterministic log-uniform point strictly inside (lo_log, hi_log).
    /// @details Uses rng.open_unit() in (0, 1) and clamps to strict interior of each log-space interval.
    /// @param rng Deterministic Xoshiro256PlusPlus generator state.
    /// @param out_theta Destination span of length dim() receiving sampled log-space parameters.
    void LogBounds::sample_log_uniform(Xoshiro256PlusPlus& rng, std::span<double> out_theta) const noexcept
    {
        const std::size_t d = dim();
        for (std::size_t i = 0; i < d; ++i)
        {
            const double u = rng.open_unit();
            const double width = hi_log_[i] - lo_log_[i];
            out_theta[i] = clamp_strict_interior(lo_log_[i] + (width * u), lo_log_[i], hi_log_[i]);
        }
    }

    /// @brief Evaluates (or retrieves cached) objective value and gradient at unconstrained z.
    /// @details Returns the cached value and gradient when valid_ is true and z matches last_z_; otherwise
    /// checks finiteness of z, maps z to theta and Jacobian d1, calls obj.value_and_gradient_into(), checks
    /// output finiteness, chains g_z = g_theta * d1, and caches the result.
    /// @param obj Differentiable objective evaluated in log-theta space.
    /// @param bounds Log-space bounds governing the z <-> theta transform.
    /// @param z Unconstrained logit parameter vector of length bounds.dim().
    /// @param theta_buf Caller-supplied scratch span of length bounds.dim() for constrained theta.
    /// @param d1_buf Caller-supplied scratch span of length bounds.dim() for Jacobian d(theta)/dz.
    /// @param grad_theta_buf Caller-supplied scratch span of length bounds.dim() for log-theta gradient.
    /// @param out_grad_z Destination span of length bounds.dim() receiving unconstrained gradient g_z.
    /// @return Objective scalar value at z, or GprError if evaluation fails or yields non-finite outputs.
    /// @retval GprErrorCode::NonFiniteInput When z, objective value, or any gradient component is non-finite.
    std::expected<double, GprError> EvalCache::eval_unconstrained(
        Differentiable& obj,
        const LogBounds& bounds,
        std::span<const double> z,
        std::span<double> theta_buf,
        std::span<double> d1_buf,
        std::span<double> grad_theta_buf,
        std::span<double> out_grad_z
    )
    {
        const std::size_t d = bounds.dim();
        // Return cached value and unconstrained gradient immediately on exact coordinate repeat.
        if (valid_ && last_z_.size() == d && std::equal(z.begin(), z.end(), last_z_.begin()))
        {
            std::ranges::copy(last_grad_z_, out_grad_z.begin());
            return last_val_;
        }

        // Reject non-finite unconstrained coordinates before transforming into bounded log space.
        for (const double val : z)
        {
            if (!std::isfinite(val))
            {
                return std::unexpected(GprError::non_finite_input());
            }
        }
        bounds.to_constrained_with_jacobian(z, theta_buf, d1_buf);
        auto val = obj.value_and_gradient_into(theta_buf, grad_theta_buf);
        if (!val.has_value())
        {
            return val;
        }
        if (!std::isfinite(*val))
        {
            return std::unexpected(GprError::non_finite_input());
        }
        for (const double g : grad_theta_buf)
        {
            if (!std::isfinite(g))
            {
                return std::unexpected(GprError::non_finite_input());
            }
        }
        LogBounds::chain_gradient(grad_theta_buf, d1_buf, out_grad_z);

        // Record z, scalar value, and chained gradient for reuse on line-search acceptance.
        last_z_.assign(z.begin(), z.end());
        last_grad_z_.assign(out_grad_z.begin(), out_grad_z.end());
        last_val_ = *val;
        valid_ = true;
        return *val;
    }

}  // namespace cxxgpr::optimizer
