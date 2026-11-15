// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/interval.hpp"

#include <cstddef>
#include <expected>
#include <span>

namespace cxxgpr {

/// Gaussian observation noise likelihood parameterized in log space as theta = log(sigma_n^2).
class GaussianLikelihood {
public:
    /// Builds a likelihood from a positive finite observation noise variance inside the default
    /// bounds.
    [[nodiscard]] static std::expected<GaussianLikelihood, GprError> create(double noise_variance);

    /// Builds a likelihood from log(sigma_n^2) inside the default bounds.
    [[nodiscard]] static std::expected<GaussianLikelihood, GprError>
    from_log_noise_variance(double log_noise_variance);

    /// Returns the observation noise variance sigma_n^2 = exp(theta).
    [[nodiscard]] constexpr double noise_variance() const noexcept {
        return noise_variance_.value();
    }

    /// Returns the log-noise parameter theta = log(sigma_n^2).
    [[nodiscard]] double log_noise_variance() const noexcept {
        return noise_variance_.ln();
    }

    /// Returns the open interval bounding sigma_n^2 in user units.
    [[nodiscard]] constexpr Interval bounds() const noexcept {
        return noise_variance_.interval();
    }

    /// Returns a copy with updated noise-variance bounds after verifying the current value stays
    /// inside.
    [[nodiscard]] std::expected<GaussianLikelihood, GprError> with_bounds(Interval bounds) const;

    /// Returns the number of likelihood optimizer parameters (always 1).
    [[nodiscard]] constexpr std::size_t num_params() const noexcept {
        return 1;
    }

    /// Writes log(sigma_n^2) into a length-1 output span without modifying out on length error.
    [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

    /// Updates theta from a length-1 span atomically; leaves *this unchanged on error.
    [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

    /// Writes the noise-variance Interval into a length-1 output span.
    [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

    /// Adds sigma_n^2 to each diagonal entry of a kernel matrix diagonal.
    void add_noise_diag(std::span<double> k_diag) const noexcept;

    /// Writes the diagonal of dK/dtheta_i (= sigma_n^2 for i = 0) into dk_diag.
    [[nodiscard]] std::expected<void, GprError> noise_grad_diag(std::span<double> dk_diag,
                                                                std::size_t param_idx) const;

    /// Checks value equality between two GaussianLikelihood instances.
    [[nodiscard]] constexpr bool
    operator==(const GaussianLikelihood& other) const noexcept = default;

private:
    explicit constexpr GaussianLikelihood(BoundedParam noise_variance) noexcept
        : noise_variance_(noise_variance) {}

    BoundedParam noise_variance_;
};

} // namespace cxxgpr
