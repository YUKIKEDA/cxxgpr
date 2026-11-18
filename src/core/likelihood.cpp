// SPDX-License-Identifier: Apache-2.0
/// @file likelihood.cpp
/// @brief Implementation of GaussianLikelihood construction, parameter mapping, and diagonal ops.
/// @details Validates log-space and user-space observation noise variance conversions and applies
/// noise diagonal updates and derivatives.

#include "cxxgpr/core/likelihood.hpp"

#include "cxxgpr/core/checked.hpp"

#include <cmath>
#include <expected>

namespace cxxgpr
{

    namespace
    {

        /// @brief Safely exponentiates log(sigma_n^2) while rejecting non-finite, overflowing, or underflowing inputs.
        /// @details Distinguishes invalid log inputs from overflow to +Inf and underflow to 0.0 before
        /// checking interval bounds.
        /// @param log_noise_variance Candidate natural-log noise variance theta.
        /// @return Strictly positive finite noise variance exp(theta), or GprError.
        /// @retval GprErrorCode::InvalidNoiseVariance When log_noise_variance is non-finite, exp
        /// overflows, or exp underflows to zero.
        std::expected<double, GprError> exp_validated_log_noise_variance(double log_noise_variance)
        {
            // Reject NaN/Inf log values before calling std::exp.
            if (!std::isfinite(log_noise_variance))
            {
                return std::unexpected(GprError::invalid_noise_variance("log noise variance must be finite"));
            }
            const double variance = std::exp(log_noise_variance);
            // Catch IEEE-754 overflow to +Inf and underflow to 0.0 separately for clear diagnostics.
            if (!std::isfinite(variance))
            {
                return std::unexpected(
                    GprError::invalid_noise_variance("noise variance overflowed to a non-finite value")
                );
            }
            if (variance <= 0.0)
            {
                return std::unexpected(GprError::invalid_noise_variance("noise variance underflowed to zero"));
            }
            return variance;
        }

    }  // namespace

    /// @brief Builds a likelihood from a positive finite observation noise variance inside the default bounds.
    /// @details Enforces finiteness and positivity checks before testing Interval::default_positive().
    /// @param noise_variance Candidate observation noise variance sigma_n^2.
    /// @return Validated GaussianLikelihood, or GprError on invalid input.
    /// @retval GprErrorCode::InvalidNoiseVariance When noise_variance is non-finite or <= 0.
    /// @retval GprErrorCode::InvalidInterval When noise_variance lies outside Interval::default_positive().
    std::expected<GaussianLikelihood, GprError> GaussianLikelihood::create(double noise_variance)
    {
        // Report InvalidNoiseVariance for non-finite or non-positive inputs prior to interval bounds checking.
        if (!std::isfinite(noise_variance))
        {
            return std::unexpected(GprError::invalid_noise_variance("noise variance must be finite"));
        }
        if (noise_variance <= 0.0)
        {
            return std::unexpected(GprError::invalid_noise_variance("noise variance must be positive"));
        }
        auto bounded = BoundedParam::default_positive(noise_variance);
        if (!bounded)
        {
            return std::unexpected(std::move(bounded.error()));
        }
        return GaussianLikelihood{*bounded};
    }

    /// @brief Builds a likelihood from log(sigma_n^2) inside the default bounds.
    /// @details Exponentiates via exp_validated_log_noise_variance() and validates against default bounds.
    /// @param log_noise_variance Candidate log noise variance theta = log(sigma_n^2).
    /// @return Validated GaussianLikelihood, or GprError on invalid input.
    /// @retval GprErrorCode::InvalidNoiseVariance When log_noise_variance fails finiteness or exp() checks.
    /// @retval GprErrorCode::InvalidInterval When exp(log_noise_variance) lies outside Interval::default_positive().
    std::expected<GaussianLikelihood, GprError> GaussianLikelihood::from_log_noise_variance(double log_noise_variance)
    {
        // Exponentiate with overflow/underflow checks first, then enforce default open interval bounds.
        auto variance = exp_validated_log_noise_variance(log_noise_variance);
        if (!variance)
        {
            return std::unexpected(std::move(variance.error()));
        }
        auto bounded = BoundedParam::default_positive(*variance);
        if (!bounded)
        {
            return std::unexpected(std::move(bounded.error()));
        }
        return GaussianLikelihood{*bounded};
    }

    /// @brief Returns a copy with updated noise-variance bounds after verifying the current value stays inside.
    /// @details Delegates interval checking to BoundedParam::with_interval().
    /// @param bounds Replacement open interval for sigma_n^2.
    /// @return Updated GaussianLikelihood, or GprError if current variance falls outside bounds.
    /// @retval GprErrorCode::InvalidInterval When noise_variance() is outside bounds.
    std::expected<GaussianLikelihood, GprError> GaussianLikelihood::with_bounds(Interval bounds) const
    {
        // Reject new bounds that exclude the current stored noise variance.
        auto updated = noise_variance_.with_interval(bounds);
        if (!updated)
        {
            return std::unexpected(std::move(updated.error()));
        }
        return GaussianLikelihood{*updated};
    }

    /// @brief Writes log(sigma_n^2) into a length-1 output span without modifying out on length error.
    /// @details Validates out.size() == 1 prior to writing.
    /// @param out Destination span of length 1.
    /// @return Empty expected on success, or GprError on length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
    std::expected<void, GprError> GaussianLikelihood::get_params(std::span<double> out) const
    {
        if (auto ok = require_count(out.size(), 1, "likelihood parameter"); !ok)
        {
            return ok;
        }
        out[0] = log_noise_variance();
        return {};
    }

    /// @brief Updates theta from a length-1 span atomically; leaves *this unchanged on error.
    /// @details Checks length, exponentiates params[0], and verifies bounds() before assigning noise_variance_.
    /// @param params Input span of length 1 holding candidate log(sigma_n^2).
    /// @return Empty expected on success, or GprError on validation failure.
    /// @retval GprErrorCode::LengthMismatch When params.size() != 1.
    /// @retval GprErrorCode::InvalidNoiseVariance When params[0] is non-finite or exp() overflows/underflows.
    /// @retval GprErrorCode::InvalidInterval When exp(params[0]) lies outside bounds().
    std::expected<void, GprError> GaussianLikelihood::set_params(std::span<const double> params)
    {
        // Validate length, log-to-variance conversion, and open bounds before mutating noise_variance_.
        if (auto ok = require_count(params.size(), 1, "likelihood parameter"); !ok)
        {
            return ok;
        }
        auto variance = exp_validated_log_noise_variance(params[0]);
        if (!variance)
        {
            return std::unexpected(std::move(variance.error()));
        }
        auto updated = BoundedParam::create(*variance, bounds());
        if (!updated)
        {
            return std::unexpected(std::move(updated.error()));
        }
        noise_variance_ = *updated;
        return {};
    }

    /// @brief Writes the noise-variance Interval into a length-1 output span.
    /// @details Validates out.size() == 1 prior to writing.
    /// @param out Destination span of length 1.
    /// @return Empty expected on success, or GprError on length mismatch.
    /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
    std::expected<void, GprError> GaussianLikelihood::bounds_into(std::span<Interval> out) const
    {
        if (auto ok = require_count(out.size(), 1, "likelihood parameter"); !ok)
        {
            return ok;
        }
        out[0] = bounds();
        return {};
    }

    /// @brief Adds sigma_n^2 to each diagonal entry of a kernel matrix diagonal.
    /// @details Updates k_diag[i] += sigma_n^2 in place.
    /// @param k_diag Mutable span over the kernel diagonal elements.
    void GaussianLikelihood::add_noise_diag(std::span<double> k_diag) const noexcept
    {
        // Cache scalar noise_variance() outside the loop to add sigma_n^2 uniformly to every diagonal entry.
        const double noise = noise_variance();
        for (double& entry : k_diag)
        {
            entry += noise;
        }
    }

    /// @brief Writes the diagonal of dK/dtheta_i (= sigma_n^2 for i = 0) into dk_diag.
    /// @details Sets every element of dk_diag to sigma_n^2 when param_idx == 0.
    /// @param dk_diag Mutable span receiving the diagonal gradient entries.
    /// @param param_idx Parameter index within the likelihood (must be 0).
    /// @return Empty expected on success, or GprError if param_idx != 0.
    /// @retval GprErrorCode::IndexOutOfRange When param_idx != 0.
    std::expected<void, GprError> GaussianLikelihood::noise_grad_diag(
        std::span<double> dk_diag, std::size_t param_idx
    ) const
    {
        if (param_idx != 0)
        {
            return std::unexpected(GprError::index_out_of_range("likelihood has a single parameter at index 0"));
        }
        // Since theta = log(sigma_n^2), d(sigma_n^2)/dtheta = exp(theta) = sigma_n^2.
        const double dkd_theta = noise_variance();
        for (double& entry : dk_diag)
        {
            entry = dkd_theta;
        }
        return {};
    }

}  // namespace cxxgpr
