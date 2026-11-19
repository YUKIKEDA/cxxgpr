// SPDX-License-Identifier: Apache-2.0
/// @file likelihood.hpp
/// @brief Gaussian observation noise likelihood parameterization and diagonal helpers.
/// @details Models homoscedastic observation noise variance sigma_n^2 bounded within an open
/// Interval and parameterized for optimization as theta = log(sigma_n^2).

#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/interval.hpp"

#include <cstddef>
#include <expected>
#include <span>

/// @brief Top-level namespace for the cxxgpr Gaussian process regression library.
namespace cxxgpr
{

    /// @brief Gaussian observation noise likelihood parameterized in log space as theta = log(sigma_n^2).
    /// @details Enforces strict positivity and boundedness of noise variance while exposing the
    /// single log-noise parameter interface expected by hyperparameter optimizers.
    class GaussianLikelihood
    {
    public:
        /// @brief Builds a likelihood from a positive finite observation noise variance inside the default bounds.
        /// @details Validates finiteness and positivity first, then checks Interval::default_positive().
        /// @param[in] noise_variance Observation noise variance sigma_n^2 in user units.
        /// @return Validated GaussianLikelihood, or GprError on invalid variance.
        /// @retval GprErrorCode::InvalidNoiseVariance When noise_variance is non-finite or <= 0.
        /// @retval GprErrorCode::InvalidInterval When noise_variance lies outside Interval::default_positive().
        [[nodiscard]] static std::expected<GaussianLikelihood, GprError> create(double noise_variance);

        /// @brief Builds a likelihood from log(sigma_n^2) inside the default bounds.
        /// @details Exponentiates log_noise_variance safely, guarding overflow and underflow before
        /// checking Interval::default_positive().
        /// @param[in] log_noise_variance Natural logarithm theta = log(sigma_n^2).
        /// @return Validated GaussianLikelihood, or GprError on invalid log variance.
        /// @retval GprErrorCode::InvalidNoiseVariance When log_noise_variance is non-finite or
        /// exp() overflows/underflows.
        /// @retval GprErrorCode::InvalidInterval When exp(log_noise_variance) lies outside
        /// Interval::default_positive().
        [[nodiscard]] static std::expected<GaussianLikelihood, GprError> from_log_noise_variance(
            double log_noise_variance
        );

        /// @brief Returns the observation noise variance sigma_n^2 = exp(theta).
        /// @details Guaranteed finite and strictly inside bounds().
        /// @return Noise variance in user units.
        [[nodiscard]] constexpr double noise_variance() const noexcept { return noise_variance_.value(); }

        /// @brief Returns the log-noise parameter theta = log(sigma_n^2).
        /// @details Matches the optimizer parameter vector value at index 0.
        /// @return Natural logarithm of noise_variance().
        [[nodiscard]] double log_noise_variance() const noexcept { return noise_variance_.ln(); }

        /// @brief Returns the open interval bounding sigma_n^2 in user units.
        /// @details Defaults to Interval::default_positive() unless updated via with_bounds().
        /// @return Bounding Interval for noise_variance().
        [[nodiscard]] constexpr Interval bounds() const noexcept { return noise_variance_.interval(); }

        /// @brief Returns a copy with updated noise-variance bounds after verifying the current value stays inside.
        /// @details Leaves *this unchanged and returns an error if noise_variance() lies outside bounds.
        /// @param[in] bounds New open bounding interval for sigma_n^2.
        /// @return Updated GaussianLikelihood, or GprError if current variance is outside bounds.
        /// @retval GprErrorCode::InvalidInterval When noise_variance() is not strictly inside bounds.
        [[nodiscard]] std::expected<GaussianLikelihood, GprError> with_bounds(Interval bounds) const;

        /// @brief Returns the number of likelihood optimizer parameters (always 1).
        /// @details Represents the single scalar parameter theta = log(sigma_n^2).
        /// @return Always returns 1.
        [[nodiscard]] constexpr std::size_t num_params() const noexcept { return 1; }

        /// @brief Writes log(sigma_n^2) into a length-1 output span without modifying out on length error.
        /// @details Requires out.size() == 1.
        /// @param[out] out Destination span of length 1 receiving theta = log(sigma_n^2).
        /// @return Empty expected on success, or GprError on length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
        [[nodiscard]] std::expected<void, GprError> get_params(std::span<double> out) const;

        /// @brief Updates theta from a length-1 span atomically; leaves *this unchanged on error.
        /// @details Validates span length, finiteness/exp range, and current bounds() before committing.
        /// @param[in] params Input span of length 1 holding candidate log(sigma_n^2).
        /// @return Empty expected on success, or GprError if validation fails.
        /// @retval GprErrorCode::LengthMismatch When params.size() != 1.
        /// @retval GprErrorCode::InvalidNoiseVariance When params[0] is non-finite or
        /// exp(params[0]) overflows/underflows.
        /// @retval GprErrorCode::InvalidInterval When exp(params[0]) lies outside bounds().
        [[nodiscard]] std::expected<void, GprError> set_params(std::span<const double> params);

        /// @brief Writes the noise-variance Interval into a length-1 output span.
        /// @details Requires out.size() == 1; leaves out unmodified on length mismatch.
        /// @param[out] out Destination span of length 1 receiving bounds().
        /// @return Empty expected on success, or GprError on length mismatch.
        /// @retval GprErrorCode::LengthMismatch When out.size() != 1.
        [[nodiscard]] std::expected<void, GprError> bounds_into(std::span<Interval> out) const;

        /// @brief Adds sigma_n^2 to each diagonal entry of a kernel matrix diagonal.
        /// @details Performs an in-place elementwise addition entry += noise_variance() with zero allocation.
        /// @param[in,out] k_diag Mutable span over the n diagonal entries of the Gram matrix.
        void add_noise_diag(std::span<double> k_diag) const noexcept;

        /// @brief Writes the diagonal of dK/dtheta_i (= sigma_n^2 for i = 0) into dk_diag.
        /// @details Uses d(sigma_n^2)/d(log(sigma_n^2)) = sigma_n^2; rejects param_idx != 0.
        /// @param[out] dk_diag Mutable span receiving the n diagonal derivative entries.
        /// @param[in] param_idx Zero-based likelihood parameter index (must be 0).
        /// @return Empty expected on success, or GprError if param_idx is out of range.
        /// @retval GprErrorCode::IndexOutOfRange When param_idx != 0.
        [[nodiscard]] std::expected<void, GprError> noise_grad_diag(
            std::span<double> dk_diag, std::size_t param_idx
        ) const;

        /// @brief Checks value equality between two GaussianLikelihood instances.
        /// @details Compares underlying BoundedParam value and interval.
        /// @param[in] other GaussianLikelihood instance to compare against.
        /// @return True if noise variance and bounds match.
        [[nodiscard]] constexpr bool operator==(const GaussianLikelihood& other) const noexcept = default;

    private:
        /// @brief Constructs a GaussianLikelihood from a validated BoundedParam.
        /// @details Private so public factories guarantee valid positivity and interval invariants.
        /// @param[in] noise_variance Validated observation noise variance bounded parameter.
        explicit constexpr GaussianLikelihood(BoundedParam noise_variance) noexcept : noise_variance_(noise_variance) {}

        /// @brief Bounded observation noise variance sigma_n^2 in user units.
        BoundedParam noise_variance_;
    };

}  // namespace cxxgpr
