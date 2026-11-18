// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/core/likelihood.hpp"

#include "cxxgpr/core/checked.hpp"

#include <cmath>
#include <expected>

namespace cxxgpr
{

    namespace
    {

        std::expected<double, GprError> exp_validated_log_noise_variance(double log_noise_variance)
        {
            if (!std::isfinite(log_noise_variance))
            {
                return std::unexpected(GprError::invalid_noise_variance("log noise variance must be finite"));
            }
            const double variance = std::exp(log_noise_variance);
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

    std::expected<GaussianLikelihood, GprError> GaussianLikelihood::create(double noise_variance)
    {
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

    std::expected<GaussianLikelihood, GprError> GaussianLikelihood::from_log_noise_variance(double log_noise_variance)
    {
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

    std::expected<GaussianLikelihood, GprError> GaussianLikelihood::with_bounds(Interval bounds) const
    {
        auto updated = noise_variance_.with_interval(bounds);
        if (!updated)
        {
            return std::unexpected(std::move(updated.error()));
        }
        return GaussianLikelihood{*updated};
    }

    std::expected<void, GprError> GaussianLikelihood::get_params(std::span<double> out) const
    {
        if (auto ok = require_count(out.size(), 1, "likelihood parameter"); !ok)
        {
            return ok;
        }
        out[0] = log_noise_variance();
        return {};
    }

    std::expected<void, GprError> GaussianLikelihood::set_params(std::span<const double> params)
    {
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

    std::expected<void, GprError> GaussianLikelihood::bounds_into(std::span<Interval> out) const
    {
        if (auto ok = require_count(out.size(), 1, "likelihood parameter"); !ok)
        {
            return ok;
        }
        out[0] = bounds();
        return {};
    }

    void GaussianLikelihood::add_noise_diag(std::span<double> k_diag) const noexcept
    {
        const double noise = noise_variance();
        for (double& entry : k_diag)
        {
            entry += noise;
        }
    }

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
