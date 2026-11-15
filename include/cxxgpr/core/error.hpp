// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace cxxgpr {

/// Error categories returned by cxxgpr API boundaries.
enum class GprErrorCode : std::uint8_t {
    EmptyInput,
    SizeOverflow,
    LengthMismatch,
    DimensionMismatch,
    IndexOutOfRange,
    InsufficientData,
    NonFiniteInput,
    NonFiniteKernelValue,
    InvalidHyperparameter,
    InvalidNoiseVariance,
    InvalidInterval,
    InvalidConfig,
    InvalidDistance,
    DistanceSlot,
    UnsupportedKernelOperation,
    CholeskyFailed,
    NonPositiveDefiniteMatrix,
    OptimizationNotConverged,
    WorkspaceTooSmall,
    PersistFailed,
    UnsupportedPersistVersion,
};

/// Call site of a failed Cholesky factorization.
enum class CholeskyStage : std::uint8_t {
    Fit,
    Predict,
};

/// Reason a supplied distance source does not match a kernel's slots.
enum class SlotErrorKind : std::uint8_t {
    NotRead,
    Duplicate,
    Missing,
};

/// Category of a persistence save or load failure.
enum class PersistErrorKind : std::uint8_t {
    Io,
    Config,
    Tensor,
    InvalidPersistId,
    NotPersistable,
    UnregisteredId,
    WrongModel,
};

/// Category of an open interval or bounded parameter error.
enum class IntervalErrorKind : std::uint8_t {
    InvalidBounds,
    OutOfRange,
};

/// Value-type error returned via std::expected across all cxxgpr public APIs.
class GprError {
public:
    /// Constructs an EmptyInput error.
    [[nodiscard]] static GprError empty_input();

    /// Constructs a SizeOverflow error.
    [[nodiscard]] static GprError size_overflow();

    /// Constructs a LengthMismatch error with details.
    [[nodiscard]] static GprError length_mismatch(std::string reason);

    /// Constructs a DimensionMismatch error recording actual and expected feature dimensions.
    [[nodiscard]] static GprError dimension_mismatch(std::size_t x_dim, std::size_t expected_dim);

    /// Constructs an IndexOutOfRange error with details.
    [[nodiscard]] static GprError index_out_of_range(std::string reason);

    /// Constructs an InsufficientData error recording provided and minimum required point counts.
    [[nodiscard]] static GprError insufficient_data(std::size_t n_points, std::size_t min_points);

    /// Constructs a NonFiniteInput error.
    [[nodiscard]] static GprError non_finite_input();

    /// Constructs a NonFiniteKernelValue error.
    [[nodiscard]] static GprError non_finite_kernel_value();

    /// Constructs an InvalidHyperparameter error with details.
    [[nodiscard]] static GprError invalid_hyperparameter(std::string reason);

    /// Constructs an InvalidNoiseVariance error with details.
    [[nodiscard]] static GprError invalid_noise_variance(std::string reason);

    /// Constructs an InvalidInterval error for invalid interval bounds (lo, hi).
    [[nodiscard]] static GprError invalid_interval_bounds(double lo, double hi);

    /// Constructs an InvalidInterval error for a parameter value outside (lo, hi).
    [[nodiscard]] static GprError invalid_interval_out_of_range(double value, double lo, double hi);

    /// Constructs an InvalidConfig error with details.
    [[nodiscard]] static GprError invalid_config(std::string reason);

    /// Constructs an InvalidDistance error identifying slot, ARD dimension, matrix pair, and cause.
    [[nodiscard]] static GprError
    invalid_distance(std::optional<std::size_t> slot, std::optional<std::size_t> dim,
                     std::optional<std::pair<std::size_t, std::size_t>> pair, std::string reason);

    /// Constructs a DistanceSlot error for a slot matching mismatch.
    [[nodiscard]] static GprError distance_slot(SlotErrorKind kind,
                                                std::optional<std::size_t> slot);

    /// Constructs an UnsupportedKernelOperation error with details.
    [[nodiscard]] static GprError unsupported_kernel_operation(std::string reason);

    /// Constructs a CholeskyFailed error recording the last jitter attempted, matrix size, and
    /// stage.
    [[nodiscard]] static GprError cholesky_failed(double jitter, std::size_t matrix_size,
                                                  CholeskyStage stage);

    /// Constructs a NonPositiveDefiniteMatrix error.
    [[nodiscard]] static GprError non_positive_definite_matrix();

    /// Constructs an OptimizationNotConverged error recording completed iterations.
    [[nodiscard]] static GprError optimization_not_converged(std::size_t iterations);

    /// Constructs a WorkspaceTooSmall error.
    [[nodiscard]] static GprError workspace_too_small();

    /// Constructs a PersistFailed error identifying the failure kind and reason.
    [[nodiscard]] static GprError persist_failed(PersistErrorKind kind, std::string reason);

    /// Constructs an UnsupportedPersistVersion error recording found and supported versions.
    [[nodiscard]] static GprError unsupported_persist_version(std::uint32_t found,
                                                              std::uint32_t supported);

    /// Returns the error category code.
    [[nodiscard]] GprErrorCode code() const noexcept {
        return code_;
    }

    /// Returns a human-readable diagnostic message.
    [[nodiscard]] std::string_view message() const noexcept {
        return dynamic_message_.empty() ? static_message_ : std::string_view{dynamic_message_};
    }

    /// Returns the actual feature column count for DimensionMismatch, if applicable.
    [[nodiscard]] std::optional<std::size_t> x_dim() const noexcept {
        return x_dim_;
    }

    /// Returns the expected feature dimension for DimensionMismatch, if applicable.
    [[nodiscard]] std::optional<std::size_t> expected_dim() const noexcept {
        return expected_dim_;
    }

    /// Returns the supplied point count for InsufficientData, if applicable.
    [[nodiscard]] std::optional<std::size_t> n_points() const noexcept {
        return n_points_;
    }

    /// Returns the minimum required point count for InsufficientData, if applicable.
    [[nodiscard]] std::optional<std::size_t> min_points() const noexcept {
        return min_points_;
    }

    /// Returns the last diagonal jitter applied for CholeskyFailed, if applicable.
    [[nodiscard]] std::optional<double> jitter() const noexcept {
        return jitter_;
    }

    /// Returns the square matrix order for CholeskyFailed, if applicable.
    [[nodiscard]] std::optional<std::size_t> matrix_size() const noexcept {
        return matrix_size_;
    }

    /// Returns the call site stage for CholeskyFailed, if applicable.
    [[nodiscard]] std::optional<CholeskyStage> cholesky_stage() const noexcept {
        return cholesky_stage_;
    }

    /// Returns the completed iteration count for OptimizationNotConverged, if applicable.
    [[nodiscard]] std::optional<std::size_t> iterations() const noexcept {
        return iterations_;
    }

    /// Returns the interval error subtype for InvalidInterval, if applicable.
    [[nodiscard]] std::optional<IntervalErrorKind> interval_error_kind() const noexcept {
        return interval_error_kind_;
    }

    /// Returns the out-of-range value for InvalidInterval::OutOfRange, if applicable.
    [[nodiscard]] std::optional<double> interval_value() const noexcept {
        return interval_value_;
    }

    /// Returns the lower interval bound for InvalidInterval, if applicable.
    [[nodiscard]] std::optional<double> interval_lo() const noexcept {
        return interval_lo_;
    }

    /// Returns the upper interval bound for InvalidInterval, if applicable.
    [[nodiscard]] std::optional<double> interval_hi() const noexcept {
        return interval_hi_;
    }

    /// Returns the offending distance slot index for distance errors, if known.
    [[nodiscard]] std::optional<std::size_t> distance_slot() const noexcept {
        return distance_slot_;
    }

    /// Returns the offending ARD dimension index for InvalidDistance, if known.
    [[nodiscard]] std::optional<std::size_t> distance_dim() const noexcept {
        return distance_dim_;
    }

    /// Returns the offending matrix entry pair (row, col) for InvalidDistance, if known.
    [[nodiscard]] std::optional<std::pair<std::size_t, std::size_t>>
    distance_pair() const noexcept {
        return distance_pair_;
    }

    /// Returns the cause description for InvalidDistance, if applicable.
    [[nodiscard]] std::optional<std::string_view> distance_reason() const noexcept {
        if (code_ != GprErrorCode::InvalidDistance) {
            return std::nullopt;
        }
        return std::string_view{distance_reason_};
    }

    /// Returns the slot error subtype for DistanceSlot, if applicable.
    [[nodiscard]] std::optional<SlotErrorKind> slot_error_kind() const noexcept {
        return slot_error_kind_;
    }

    /// Returns the persistence failure subtype for PersistFailed, if applicable.
    [[nodiscard]] std::optional<PersistErrorKind> persist_error_kind() const noexcept {
        return persist_error_kind_;
    }

    /// Returns the discovered persist format version for UnsupportedPersistVersion, if applicable.
    [[nodiscard]] std::optional<std::uint32_t> found_version() const noexcept {
        return found_version_;
    }

    /// Returns the supported persist format version for UnsupportedPersistVersion, if applicable.
    [[nodiscard]] std::optional<std::uint32_t> supported_version() const noexcept {
        return supported_version_;
    }

    /// Names the distance slot on an InvalidDistance error if not already set; otherwise returns
    /// *this.
    [[nodiscard]] GprError with_slot_if_unset(std::size_t slot_place) const;

    /// Names the ARD dimension on an InvalidDistance error if not already set; otherwise returns
    /// *this.
    [[nodiscard]] GprError with_dim_if_unset(std::size_t dim_place) const;

    /// Checks structural equality between two GprError instances.
    [[nodiscard]] bool operator==(const GprError& other) const = default;

private:
    GprError(GprErrorCode code, std::string_view static_message) noexcept
        : code_(code), static_message_(static_message) {}

    GprError(GprErrorCode code, std::string dynamic_message) noexcept
        : code_(code), dynamic_message_(std::move(dynamic_message)) {}

    GprErrorCode code_;
    std::string_view static_message_;
    std::string dynamic_message_;

    std::optional<std::size_t> x_dim_;
    std::optional<std::size_t> expected_dim_;
    std::optional<std::size_t> n_points_;
    std::optional<std::size_t> min_points_;
    std::optional<double> jitter_;
    std::optional<std::size_t> matrix_size_;
    std::optional<CholeskyStage> cholesky_stage_;
    std::optional<std::size_t> iterations_;
    std::optional<IntervalErrorKind> interval_error_kind_;
    std::optional<double> interval_value_;
    std::optional<double> interval_lo_;
    std::optional<double> interval_hi_;
    std::optional<std::size_t> distance_slot_;
    std::optional<std::size_t> distance_dim_;
    std::optional<std::pair<std::size_t, std::size_t>> distance_pair_;
    std::string distance_reason_;
    std::optional<SlotErrorKind> slot_error_kind_;
    std::optional<PersistErrorKind> persist_error_kind_;
    std::optional<std::uint32_t> found_version_;
    std::optional<std::uint32_t> supported_version_;
};

} // namespace cxxgpr
