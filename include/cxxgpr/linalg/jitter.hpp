// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>

namespace cxxgpr
{

    /// Non-negative diagonal jitter offset used by Fixed jitter regularization.
    class FixedJitter
    {
    public:
        /// Constructs a fixed jitter with zero offset.
        constexpr FixedJitter() noexcept = default;

        /// Constructs a fixed jitter with a validated non-negative offset.
        explicit constexpr FixedJitter(double jitter) noexcept : jitter_(jitter) {}

        /// Returns the diagonal jitter offset.
        [[nodiscard]] constexpr double jitter() const noexcept { return jitter_; }

        /// Checks value equality between two FixedJitter instances.
        [[nodiscard]] friend constexpr bool operator==(const FixedJitter&, const FixedJitter&) noexcept = default;

    private:
        double jitter_ = 0.0;
    };

    /// Geometric jitter escalation parameters used by Adaptive jitter regularization.
    class AdaptiveJitter
    {
    public:
        /// Constructs an adaptive jitter schedule from validated parameters.
        constexpr AdaptiveJitter(double initial, double multiplier, std::size_t max_retries, double max_jitter) noexcept
            :
            initial_(initial),
            multiplier_(multiplier),
            max_retries_(max_retries),
            max_jitter_(max_jitter)
        {
        }

        /// Returns the first retry jitter offset.
        [[nodiscard]] constexpr double initial() const noexcept { return initial_; }

        /// Returns the multiplicative growth factor applied after each failed retry.
        [[nodiscard]] constexpr double multiplier() const noexcept { return multiplier_; }

        /// Returns the maximum number of jittered retry attempts.
        [[nodiscard]] constexpr std::size_t max_retries() const noexcept { return max_retries_; }

        /// Returns the upper bound on retry jitter offsets.
        [[nodiscard]] constexpr double max_jitter() const noexcept { return max_jitter_; }

        /// Checks value equality between two AdaptiveJitter instances.
        [[nodiscard]] friend constexpr bool operator==(const AdaptiveJitter&, const AdaptiveJitter&) noexcept = default;

    private:
        double initial_ = 0.0;
        double multiplier_ = 1.0;
        std::size_t max_retries_ = 0;
        double max_jitter_ = 0.0;
    };

    /// Stateful generator yielding the sequence of retry jitter offsets after j = 0 fails.
    class JitterSchedule
    {
    public:
        /// Constructs an empty retry schedule.
        constexpr JitterSchedule() noexcept = default;

        /// Constructs a schedule with explicit state.
        constexpr JitterSchedule(
            std::optional<double> next, double multiplier, std::size_t remaining, double max_jitter
        ) noexcept :
            next_(next),
            multiplier_(multiplier),
            remaining_(remaining),
            max_jitter_(max_jitter)
        {
        }

        /// Returns the next retry jitter offset, or std::nullopt when the schedule is exhausted.
        [[nodiscard]] std::optional<double> next() noexcept;

    private:
        std::optional<double> next_;
        double multiplier_ = 1.0;
        std::size_t remaining_ = 0;
        double max_jitter_ = 0.0;
    };

    /// Selects the numerical Cholesky stabilization strategy, distinct from observation noise.
    class JitterPolicy
    {
    public:
        /// Categorizes the jitter retry strategy.
        enum class Kind : std::uint8_t
        {
            /// Tries j = 0 first and retries at most once with a fixed offset if j > 0.
            Fixed,
            /// Tries j = 0 first and escalates geometrically within max_retries and max_jitter.
            Adaptive,
        };

        /// Constructs the default policy: Fixed(0.0) with no retries.
        constexpr JitterPolicy() noexcept = default;

        /// Builds a Fixed jitter policy requiring a finite, non-negative offset.
        [[nodiscard]] static std::expected<JitterPolicy, GprError> fixed(double jitter);

        /// Builds an Adaptive jitter policy requiring initial > 0, multiplier > 1, max_retries >= 1,
        /// and max_jitter >= initial with all floating-point values finite.
        [[nodiscard]] static std::expected<JitterPolicy, GprError> adaptive(
            double initial, double multiplier, std::size_t max_retries, double max_jitter
        );

        /// Returns the active jitter strategy kind.
        [[nodiscard]] constexpr Kind kind() const noexcept { return kind_; }

        /// Returns true if this policy uses a Fixed retry offset.
        [[nodiscard]] constexpr bool is_fixed() const noexcept { return kind_ == Kind::Fixed; }

        /// Returns true if this policy uses an Adaptive escalation schedule.
        [[nodiscard]] constexpr bool is_adaptive() const noexcept { return kind_ == Kind::Adaptive; }

        /// Returns the FixedJitter settings when kind() == Kind::Fixed.
        [[nodiscard]] constexpr std::optional<FixedJitter> as_fixed() const noexcept
        {
            if (kind_ != Kind::Fixed)
            {
                return std::nullopt;
            }
            return fixed_;
        }

        /// Returns the AdaptiveJitter settings when kind() == Kind::Adaptive.
        [[nodiscard]] constexpr std::optional<AdaptiveJitter> as_adaptive() const noexcept
        {
            if (kind_ != Kind::Adaptive)
            {
                return std::nullopt;
            }
            return adaptive_;
        }

        /// Creates the retry schedule evaluated if the initial j = 0 factorization fails.
        [[nodiscard]] JitterSchedule retry_schedule() const noexcept;

        /// Checks value equality between two JitterPolicy instances.
        [[nodiscard]] friend constexpr bool operator==(const JitterPolicy&, const JitterPolicy&) noexcept = default;

    private:
        constexpr explicit JitterPolicy(FixedJitter fixed_cfg) noexcept : kind_(Kind::Fixed), fixed_(fixed_cfg) {}

        constexpr explicit JitterPolicy(AdaptiveJitter adaptive_cfg) noexcept :
            kind_(Kind::Adaptive),
            adaptive_(adaptive_cfg)
        {
        }

        Kind kind_ = Kind::Fixed;
        FixedJitter fixed_{0.0};
        AdaptiveJitter adaptive_{0.0, 1.0, 0, 0.0};
    };

}  // namespace cxxgpr
