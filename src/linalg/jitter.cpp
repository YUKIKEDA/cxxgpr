// SPDX-License-Identifier: Apache-2.0
/// @file jitter.cpp
/// @brief Implementations of JitterSchedule and JitterPolicy validation and retry generation.
/// @details Implements geometric jitter progression, parameter validation, and retry schedule construction.

#include "cxxgpr/linalg/jitter.hpp"

#include <cmath>
#include <format>

namespace cxxgpr
{

    /// @brief Advances the schedule and returns the next retry jitter offset.
    /// @details Terminates immediately if remaining_ is zero, next_ is empty, or the candidate exceeds max_jitter_;
    /// otherwise returns the current jitter and updates next_ via current * multiplier_.
    /// @return Next diagonal jitter offset to attempt, or std::nullopt when the schedule is exhausted.
    std::optional<double> JitterSchedule::next() noexcept
    {
        // Stop immediately once the retry budget or candidate stream is exhausted.
        if (remaining_ == 0 || !next_.has_value())
        {
            return std::nullopt;
        }
        const double current = *next_;
        // Enforce the upper jitter cap and exhaust remaining retries if exceeded.
        if (current > max_jitter_)
        {
            remaining_ = 0;
            next_ = std::nullopt;
            return std::nullopt;
        }
        --remaining_;
        // Advance geometrically; mark the stream exhausted if scaling overflows to infinity.
        const double grown = current * multiplier_;
        if (std::isfinite(grown))
        {
            next_ = grown;
        }
        else
        {
            next_ = std::nullopt;
        }
        return current;
    }

    /// @brief Builds a Fixed jitter policy requiring a finite, non-negative offset.
    /// @details Rejects NaN, infinite, or negative offsets with GprErrorCode::InvalidConfig.
    /// @param jitter Non-negative diagonal jitter offset to try once if j = 0 fails.
    /// @return Validated JitterPolicy on success, or GprError on invalid input.
    /// @retval GprErrorCode::InvalidConfig When jitter is NaN, infinite, or negative.
    std::expected<JitterPolicy, GprError> JitterPolicy::fixed(double jitter)
    {
        // Reject non-finite or negative offsets so diagonal perturbations can never shrink eigenvalues.
        if (!std::isfinite(jitter) || jitter < 0.0)
        {
            return std::unexpected(
                GprError::invalid_config(std::format("jitter must be finite and non-negative, got {}", jitter))
            );
        }
        return JitterPolicy(FixedJitter(jitter));
    }

    /// @brief Builds an Adaptive jitter policy with geometric escalation parameters.
    /// @details Validates initial > 0, multiplier > 1, max_retries >= 1, and max_jitter >= initial with all
    /// floating-point arguments finite.
    /// @param initial Positive finite jitter offset for the first retry.
    /// @param multiplier Finite growth factor strictly greater than 1.0.
    /// @param max_retries Maximum number of retry attempts (must be at least 1).
    /// @param max_jitter Finite upper cap on retry jitter (must be at least initial).
    /// @return Validated JitterPolicy on success, or GprError on invalid configuration.
    /// @retval GprErrorCode::InvalidConfig When any parameter violates finiteness or range preconditions.
    std::expected<JitterPolicy, GprError> JitterPolicy::adaptive(
        double initial, double multiplier, std::size_t max_retries, double max_jitter
    )
    {
        // Validate parameters in logical order: initial -> multiplier -> max_retries -> max_jitter.
        if (!std::isfinite(initial) || initial <= 0.0)
        {
            return std::unexpected(
                GprError::invalid_config(
                    std::format("adaptive initial jitter must be finite and positive, got {}", initial)
                )
            );
        }
        if (!std::isfinite(multiplier) || multiplier <= 1.0)
        {
            return std::unexpected(
                GprError::invalid_config(
                    std::format("adaptive jitter multiplier must be finite and greater than 1, got {}", multiplier)
                )
            );
        }
        if (max_retries < 1)
        {
            return std::unexpected(GprError::invalid_config("adaptive jitter max_retries must be at least 1"));
        }
        if (!std::isfinite(max_jitter) || max_jitter < initial)
        {
            return std::unexpected(
                GprError::invalid_config(
                    std::format(
                        "adaptive max_jitter must be finite and at least initial ({}), got {}", initial, max_jitter
                    )
                )
            );
        }
        return JitterPolicy(AdaptiveJitter(initial, multiplier, max_retries, max_jitter));
    }

    /// @brief Creates the retry schedule evaluated after an initial j = 0 factorization fails.
    /// @details Returns an empty schedule for Fixed(0.0), a single-entry schedule for positive Fixed, or a geometric
    /// sequence for Adaptive.
    /// @return Stateful JitterSchedule initialized from this policy.
    JitterSchedule JitterPolicy::retry_schedule() const noexcept
    {
        if (kind_ == Kind::Fixed)
        {
            // Only positive fixed jitter schedules a single retry after j = 0 fails.
            if (fixed_.jitter() > 0.0)
            {
                return {fixed_.jitter(), 1.0, 1, fixed_.jitter()};
            }
            return {std::nullopt, 1.0, 0, 0.0};
        }
        // Construct geometric progression starting at initial_ up to max_retries_ and max_jitter_.
        return {adaptive_.initial(), adaptive_.multiplier(), adaptive_.max_retries(), adaptive_.max_jitter()};
    }

}  // namespace cxxgpr
