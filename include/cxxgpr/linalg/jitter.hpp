// SPDX-License-Identifier: Apache-2.0
/// @file jitter.hpp
/// @brief Cholesky diagonal jitter regularization policies and retry schedules.
/// @details Defines FixedJitter, AdaptiveJitter, JitterSchedule, and JitterPolicy to distinguish numerical Cholesky
/// stabilization from statistical observation noise.

#pragma once

#include "cxxgpr/core/error.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>

/// @brief Root namespace for cxxgpr Gaussian process regression components.
namespace cxxgpr
{

    /// @brief Non-negative diagonal jitter offset used by fixed jitter regularization.
    /// @details Stores a validated non-negative scalar offset applied only if unjittered Cholesky factorization fails.
    class FixedJitter
    {
    public:
        /// @brief Constructs a fixed jitter configuration with a zero offset.
        /// @details A zero offset causes Cholesky factorization to make no jittered retry after j = 0 fails.
        constexpr FixedJitter() noexcept = default;

        /// @brief Constructs a fixed jitter configuration from a validated non-negative offset.
        /// @details Callers outside JitterPolicy::fixed must ensure the offset is finite and non-negative.
        /// @param[in] jitter Non-negative diagonal jitter offset to add on retry.
        explicit constexpr FixedJitter(double jitter) noexcept : jitter_(jitter) {}

        /// @brief Returns the configured diagonal jitter offset.
        /// @details Returns 0.0 when jittered retries are disabled.
        /// @return Non-negative diagonal jitter offset.
        [[nodiscard]] constexpr double jitter() const noexcept { return jitter_; }

        /// @brief Checks value equality between two FixedJitter instances.
        /// @details Compares the underlying diagonal jitter offset for exact equality.
        /// @param[in] lhs Left-hand FixedJitter instance.
        /// @param[in] rhs Right-hand FixedJitter instance.
        /// @return True if both instances store equal jitter offsets.
        [[nodiscard]] friend constexpr bool operator==(
            const FixedJitter& lhs, const FixedJitter& rhs
        ) noexcept = default;

    private:
        /// @brief Non-negative diagonal offset applied during a fixed retry attempt.
        double jitter_ = 0.0;
    };

    /// @brief Geometric jitter escalation parameters used by adaptive jitter regularization.
    /// @details Encapsulates the initial offset, growth factor, retry budget, and upper cap for adaptive Cholesky
    /// retries.
    class AdaptiveJitter
    {
    public:
        /// @brief Constructs an adaptive jitter schedule configuration from validated parameters.
        /// @details Callers must obtain validated parameters through JitterPolicy::adaptive.
        /// @param[in] initial Positive finite jitter offset for the first retry attempt.
        /// @param[in] multiplier Multiplicative growth factor greater than 1.0 applied after each failed retry.
        /// @param[in] max_retries Maximum number of jittered retry attempts allowed (at least 1).
        /// @param[in] max_jitter Finite upper bound on generated retry jitter offsets (at least initial).
        constexpr AdaptiveJitter(double initial, double multiplier, std::size_t max_retries, double max_jitter) noexcept
            :
            initial_(initial),
            multiplier_(multiplier),
            max_retries_(max_retries),
            max_jitter_(max_jitter)
        {
        }

        /// @brief Returns the first retry jitter offset.
        /// @details Used immediately after unjittered Cholesky factorization fails.
        /// @return Positive initial retry jitter offset.
        [[nodiscard]] constexpr double initial() const noexcept { return initial_; }

        /// @brief Returns the multiplicative growth factor applied after each failed retry.
        /// @details Scales the previous jitter offset geometrically before the next retry attempt.
        /// @return Finite multiplier strictly greater than 1.0.
        [[nodiscard]] constexpr double multiplier() const noexcept { return multiplier_; }

        /// @brief Returns the maximum number of jittered retry attempts.
        /// @details Bounds the retry loop independently of the max_jitter cap.
        /// @return Maximum retry count (at least 1).
        [[nodiscard]] constexpr std::size_t max_retries() const noexcept { return max_retries_; }

        /// @brief Returns the upper bound on retry jitter offsets.
        /// @details Any candidate offset exceeding this threshold terminates the schedule early.
        /// @return Finite maximum jitter offset.
        [[nodiscard]] constexpr double max_jitter() const noexcept { return max_jitter_; }

        /// @brief Checks value equality between two AdaptiveJitter instances.
        /// @details Compares initial, multiplier, max_retries, and max_jitter fields for equality.
        /// @param[in] lhs Left-hand AdaptiveJitter instance.
        /// @param[in] rhs Right-hand AdaptiveJitter instance.
        /// @return True if all adaptive jitter parameters match.
        [[nodiscard]] friend constexpr bool operator==(
            const AdaptiveJitter& lhs, const AdaptiveJitter& rhs
        ) noexcept = default;

    private:
        /// @brief Positive initial retry jitter offset.
        double initial_ = 0.0;
        /// @brief Multiplicative growth factor applied across consecutive retries.
        double multiplier_ = 1.0;
        /// @brief Maximum number of retry attempts permitted.
        std::size_t max_retries_ = 0;
        /// @brief Inclusive upper bound on retry jitter values.
        double max_jitter_ = 0.0;
    };

    /// @brief Stateful generator yielding the sequence of retry jitter offsets after j = 0 fails.
    /// @details Produces consecutive retry jitter values while respecting remaining retry count, max_jitter, and finite
    /// arithmetic bounds.
    class JitterSchedule
    {
    public:
        /// @brief Constructs an exhausted retry schedule.
        /// @details Subsequent calls to next() return std::nullopt.
        constexpr JitterSchedule() noexcept = default;

        /// @brief Constructs a retry schedule with explicit progression state.
        /// @details Typically created by JitterPolicy::retry_schedule.
        /// @param[in] next Candidate jitter offset for the upcoming retry, or std::nullopt if exhausted.
        /// @param[in] multiplier Multiplicative growth factor applied to advance to the following offset.
        /// @param[in] remaining Remaining number of retries allowed by the budget.
        /// @param[in] max_jitter Inclusive upper bound beyond which candidate offsets are rejected.
        constexpr JitterSchedule(
            std::optional<double> next, double multiplier, std::size_t remaining, double max_jitter
        ) noexcept :
            next_(next),
            multiplier_(multiplier),
            remaining_(remaining),
            max_jitter_(max_jitter)
        {
        }

        /// @brief Advances the schedule and returns the next retry jitter offset.
        /// @details Stops and clears remaining state once the retry budget is exhausted, the candidate exceeds
        /// max_jitter, or multiplication overflows to non-finite.
        /// @return Next diagonal jitter offset to attempt, or std::nullopt when the schedule is exhausted.
        [[nodiscard]] std::optional<double> next() noexcept;

    private:
        /// @brief Next candidate jitter offset, or std::nullopt when exhausted.
        std::optional<double> next_;
        /// @brief Geometric growth factor applied after yielding a jitter value.
        double multiplier_ = 1.0;
        /// @brief Number of retry attempts remaining in the schedule.
        std::size_t remaining_ = 0;
        /// @brief Inclusive upper cap for yielded jitter values.
        double max_jitter_ = 0.0;
    };

    /// @brief Selects the numerical Cholesky stabilization strategy, distinct from observation noise.
    /// @details Represents either a fixed single-retry jitter offset or a geometrically escalating adaptive schedule
    /// evaluated only when j = 0 factorization fails.
    class JitterPolicy
    {
    public:
        /// @brief Categorizes the jitter retry strategy.
        /// @details Distinguishes fixed-offset single retry from geometric adaptive escalation.
        enum class Kind : std::uint8_t
        {
            /// @brief Tries j = 0 first and retries at most once with a fixed offset when jitter > 0.
            Fixed,
            /// @brief Tries j = 0 first and escalates geometrically within max_retries and max_jitter.
            Adaptive,
        };

        /// @brief Constructs the default policy: Fixed(0.0) with no retries.
        /// @details Factorizations using the default policy fail immediately if j = 0 is not positive definite.
        constexpr JitterPolicy() noexcept = default;

        /// @brief Builds a Fixed jitter policy requiring a finite, non-negative offset.
        /// @details Passing 0.0 creates a zero-retry policy that leaves the matrix unperturbed.
        /// @param[in] jitter Non-negative diagonal jitter offset to try once if j = 0 fails.
        /// @return Validated JitterPolicy on success, or GprError on invalid input.
        /// @retval GprErrorCode::InvalidConfig When jitter is NaN, infinite, or negative.
        [[nodiscard]] static std::expected<JitterPolicy, GprError> fixed(double jitter);

        /// @brief Builds an Adaptive jitter policy with geometric escalation parameters.
        /// @details Validates finiteness, positivity, growth factor, retry count, and cap before constructing the
        /// policy.
        /// @param[in] initial Positive finite jitter offset for the first retry.
        /// @param[in] multiplier Finite growth factor strictly greater than 1.0.
        /// @param[in] max_retries Maximum number of retry attempts (must be at least 1).
        /// @param[in] max_jitter Finite upper cap on retry jitter (must be at least initial).
        /// @return Validated JitterPolicy on success, or GprError on invalid configuration.
        /// @retval GprErrorCode::InvalidConfig When any parameter violates finiteness or range preconditions.
        [[nodiscard]] static std::expected<JitterPolicy, GprError> adaptive(
            double initial, double multiplier, std::size_t max_retries, double max_jitter
        );

        /// @brief Returns the active jitter strategy kind.
        /// @details Indicates whether fixed or adaptive parameters are active.
        /// @return Enumerator identifying the active jitter policy kind.
        [[nodiscard]] constexpr Kind kind() const noexcept { return kind_; }

        /// @brief Returns true if this policy uses a fixed retry offset.
        /// @details Equivalent to checking kind() == Kind::Fixed.
        /// @return True for fixed jitter policies, false otherwise.
        [[nodiscard]] constexpr bool is_fixed() const noexcept { return kind_ == Kind::Fixed; }

        /// @brief Returns true if this policy uses an adaptive escalation schedule.
        /// @details Equivalent to checking kind() == Kind::Adaptive.
        /// @return True for adaptive jitter policies, false otherwise.
        [[nodiscard]] constexpr bool is_adaptive() const noexcept { return kind_ == Kind::Adaptive; }

        /// @brief Returns the FixedJitter settings when kind() == Kind::Fixed.
        /// @details Provides typed access to the fixed jitter configuration without exposing inactive adaptive
        /// settings.
        /// @return Populated FixedJitter settings if fixed, or std::nullopt otherwise.
        [[nodiscard]] constexpr std::optional<FixedJitter> as_fixed() const noexcept
        {
            // Guard against reading the inactive fixed payload on an adaptive policy.
            if (kind_ != Kind::Fixed)
            {
                return std::nullopt;
            }
            return fixed_;
        }

        /// @brief Returns the AdaptiveJitter settings when kind() == Kind::Adaptive.
        /// @details Provides typed access to the adaptive jitter configuration without exposing inactive fixed
        /// settings.
        /// @return Populated AdaptiveJitter settings if adaptive, or std::nullopt otherwise.
        [[nodiscard]] constexpr std::optional<AdaptiveJitter> as_adaptive() const noexcept
        {
            // Guard against reading the inactive adaptive payload on a fixed policy.
            if (kind_ != Kind::Adaptive)
            {
                return std::nullopt;
            }
            return adaptive_;
        }

        /// @brief Creates the retry schedule evaluated after an initial j = 0 factorization fails.
        /// @details A Fixed(0.0) policy produces an immediately exhausted schedule; positive fixed yields one retry;
        /// adaptive yields up to max_retries geometrically scaled offsets.
        /// @return Stateful JitterSchedule initialized from this policy.
        [[nodiscard]] JitterSchedule retry_schedule() const noexcept;

        /// @brief Checks value equality between two JitterPolicy instances.
        /// @details Compares active kind and both stored configuration payloads.
        /// @param[in] lhs Left-hand JitterPolicy instance.
        /// @param[in] rhs Right-hand JitterPolicy instance.
        /// @return True if both policies have the same kind and configuration values.
        [[nodiscard]] friend constexpr bool operator==(
            const JitterPolicy& lhs, const JitterPolicy& rhs
        ) noexcept = default;

    private:
        /// @brief Constructs a fixed jitter policy from validated settings.
        /// @details Initializes kind_ to Kind::Fixed and stores fixed_cfg.
        /// @param[in] fixed_cfg Validated fixed jitter settings.
        constexpr explicit JitterPolicy(FixedJitter fixed_cfg) noexcept : kind_(Kind::Fixed), fixed_(fixed_cfg) {}

        /// @brief Constructs an adaptive jitter policy from validated settings.
        /// @details Initializes kind_ to Kind::Adaptive and stores adaptive_cfg.
        /// @param[in] adaptive_cfg Validated adaptive jitter settings.
        constexpr explicit JitterPolicy(AdaptiveJitter adaptive_cfg) noexcept :
            kind_(Kind::Adaptive),
            adaptive_(adaptive_cfg)
        {
        }

        /// @brief Discriminator selecting between Fixed and Adaptive policy modes.
        Kind kind_ = Kind::Fixed;
        /// @brief Stored settings when kind_ == Kind::Fixed.
        FixedJitter fixed_{0.0};
        /// @brief Stored settings when kind_ == Kind::Adaptive.
        AdaptiveJitter adaptive_{0.0, 1.0, 0, 0.0};
    };

}  // namespace cxxgpr
