// SPDX-License-Identifier: Apache-2.0
/// @file error.hpp
/// @brief Error categories, diagnostic enums, and the GprError value type.
/// @details Defines non-throwing error payloads returned via std::expected across all cxxgpr
/// library boundaries, including structured metadata for dimensions, Cholesky failures,
/// intervals, distance slots, and persistence diagnostics.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

/// @brief Top-level namespace for the cxxgpr Gaussian process regression library.
namespace cxxgpr
{

    /// @brief Error categories returned by cxxgpr API boundaries.
    /// @details Covers input validation, parameter bounds, distance matrix checking, linear
    /// algebra failures, optimization status, workspace capacity, and model persistence.
    enum class GprErrorCode : std::uint8_t
    {
        /// @brief Input matrix, target vector, or buffer span is empty or has a zero extent.
        EmptyInput,
        /// @brief Dimension or packed buffer size calculation overflowed std::size_t.
        SizeOverflow,
        /// @brief Supplied vector or matrix buffer length does not match expected dimensions.
        LengthMismatch,
        /// @brief Feature matrix column count does not match expected kernel dimensionality.
        DimensionMismatch,
        /// @brief Parameter, distance slot, or element index is out of valid bounds.
        IndexOutOfRange,
        /// @brief Supplied training point count is below the operation minimum.
        InsufficientData,
        /// @brief Input coordinates, targets, or scalars contain NaN or +/-Inf.
        NonFiniteInput,
        /// @brief Kernel evaluation produced a NaN or +/-Inf matrix entry.
        NonFiniteKernelValue,
        /// @brief Kernel hyperparameter value or vector failed validation.
        InvalidHyperparameter,
        /// @brief Observation noise variance or log-noise variance failed validation.
        InvalidNoiseVariance,
        /// @brief Open interval bounds are invalid or a bounded parameter lies outside its interval.
        InvalidInterval,
        /// @brief Model, optimizer, or kernel configuration contains an invalid setting.
        InvalidConfig,
        /// @brief Supplied squared-distance matrix violated symmetry, diagonal, non-negativity, or shape rules.
        InvalidDistance,
        /// @brief Supplied distance sources do not match the distance slots required by the kernel.
        DistanceSlot,
        /// @brief Kernel does not support the requested operation (e.g., feature evaluation on distance-only kernels).
        UnsupportedKernelOperation,
        /// @brief Cholesky factorization failed even after applying the full jitter ladder.
        CholeskyFailed,
        /// @brief Matrix failed a strict positive-definiteness check without jitter recovery.
        NonPositiveDefiniteMatrix,
        /// @brief Hyperparameter optimizer reached its iteration limit without converging.
        OptimizationNotConverged,
        /// @brief Caller-provided workspace or scratch buffer is smaller than required.
        WorkspaceTooSmall,
        /// @brief Model persistence save or load failed due to I/O, JSON, tensor, or registry errors.
        PersistFailed,
        /// @brief Persisted artifact format version is not supported by this library build.
        UnsupportedPersistVersion,
    };

    /// @brief Call site of a failed Cholesky factorization.
    /// @details Identifies whether Cholesky failure occurred during model fitting or prediction.
    enum class CholeskyStage : std::uint8_t
    {
        /// @brief Factorization failed while decomposing the noisy training Gram matrix.
        Fit,
        /// @brief Factorization failed during prediction or posterior covariance evaluation.
        Predict,
    };

    /// @brief Reason a supplied distance source does not match a kernel's slots.
    /// @details Used with GprErrorCode::DistanceSlot to distinguish unused, duplicated, or absent slots.
    enum class SlotErrorKind : std::uint8_t
    {
        /// @brief Supplied distance source targets a slot index that the kernel does not read.
        NotRead,
        /// @brief Two or more distance sources were supplied for the same slot index.
        Duplicate,
        /// @brief A slot required by the kernel has no supplied distance source.
        Missing,
    };

    /// @brief Category of a persistence save or load failure.
    /// @details Subclassifies GprErrorCode::PersistFailed errors for programmatic inspection.
    enum class PersistErrorKind : std::uint8_t
    {
        /// @brief Filesystem open, read, write, or metadata error.
        Io,
        /// @brief Persisted JSON configuration is malformed or violates schema constraints.
        Config,
        /// @brief Persisted safetensors header, dtype, shape, or payload is invalid.
        Tensor,
        /// @brief Custom kernel persistence identifier failed format validation.
        InvalidPersistId,
        /// @brief Model or custom kernel is not persistable.
        NotPersistable,
        /// @brief Persisted custom kernel identifier has no registered loader factory.
        UnregisteredId,
        /// @brief Artifact precision mode or model family does not match the target model type.
        WrongModel,
    };

    /// @brief Category of an open interval or bounded parameter error.
    /// @details Subclassifies GprErrorCode::InvalidInterval diagnostics.
    enum class IntervalErrorKind : std::uint8_t
    {
        /// @brief Interval bounds failed finiteness or strict ordering (lo < hi).
        InvalidBounds,
        /// @brief Parameter value is non-finite or outside the open interval (lo, hi).
        OutOfRange,
    };

    /// @brief Value-type error returned via std::expected across all cxxgpr public APIs.
    /// @details Stores an error category code, either a static literal message or owned dynamic
    /// message, and optional typed metadata corresponding to the specific error variant.
    class GprError
    {
    public:
        /// @brief Constructs an EmptyInput error.
        /// @details Uses a static diagnostic message with no heap allocation.
        /// @return GprError with code GprErrorCode::EmptyInput.
        [[nodiscard]] static GprError empty_input();

        /// @brief Constructs a SizeOverflow error.
        /// @details Uses a static diagnostic message with no heap allocation.
        /// @return GprError with code GprErrorCode::SizeOverflow.
        [[nodiscard]] static GprError size_overflow();

        /// @brief Constructs a LengthMismatch error with details.
        /// @details Formats the supplied mismatch description into the diagnostic message.
        /// @param reason Human-readable explanation of expected versus actual buffer length.
        /// @return GprError with code GprErrorCode::LengthMismatch.
        [[nodiscard]] static GprError length_mismatch(std::string reason);

        /// @brief Constructs a DimensionMismatch error recording actual and expected feature dimensions.
        /// @details Populates x_dim() and expected_dim() alongside a formatted diagnostic message.
        /// @param x_dim Supplied number of feature columns.
        /// @param expected_dim Required number of feature columns.
        /// @return GprError with code GprErrorCode::DimensionMismatch.
        [[nodiscard]] static GprError dimension_mismatch(std::size_t x_dim, std::size_t expected_dim);

        /// @brief Constructs an IndexOutOfRange error with details.
        /// @details Formats the supplied out-of-range reason into the diagnostic message.
        /// @param reason Human-readable explanation of the invalid index.
        /// @return GprError with code GprErrorCode::IndexOutOfRange.
        [[nodiscard]] static GprError index_out_of_range(std::string reason);

        /// @brief Constructs an InsufficientData error recording provided and minimum required point counts.
        /// @details Populates n_points() and min_points() alongside a formatted diagnostic message.
        /// @param n_points Number of training samples supplied.
        /// @param min_points Minimum number of training samples required.
        /// @return GprError with code GprErrorCode::InsufficientData.
        [[nodiscard]] static GprError insufficient_data(std::size_t n_points, std::size_t min_points);

        /// @brief Constructs a NonFiniteInput error.
        /// @details Uses a static diagnostic message indicating NaN or Inf input values.
        /// @return GprError with code GprErrorCode::NonFiniteInput.
        [[nodiscard]] static GprError non_finite_input();

        /// @brief Constructs a NonFiniteKernelValue error.
        /// @details Uses a static diagnostic message indicating non-finite kernel outputs.
        /// @return GprError with code GprErrorCode::NonFiniteKernelValue.
        [[nodiscard]] static GprError non_finite_kernel_value();

        /// @brief Constructs an InvalidHyperparameter error with details.
        /// @details Formats the supplied hyperparameter validation reason into the message.
        /// @param reason Human-readable description of the hyperparameter violation.
        /// @return GprError with code GprErrorCode::InvalidHyperparameter.
        [[nodiscard]] static GprError invalid_hyperparameter(std::string reason);

        /// @brief Constructs an InvalidNoiseVariance error with details.
        /// @details Formats the supplied observation noise violation reason into the message.
        /// @param reason Human-readable description of the invalid noise variance.
        /// @return GprError with code GprErrorCode::InvalidNoiseVariance.
        [[nodiscard]] static GprError invalid_noise_variance(std::string reason);

        /// @brief Constructs an InvalidInterval error for invalid interval bounds (lo, hi).
        /// @details Records IntervalErrorKind::InvalidBounds, interval_lo(), and interval_hi().
        /// @param lo Candidate lower bound.
        /// @param hi Candidate upper bound.
        /// @return GprError with code GprErrorCode::InvalidInterval.
        [[nodiscard]] static GprError invalid_interval_bounds(double lo, double hi);

        /// @brief Constructs an InvalidInterval error for a parameter value outside (lo, hi).
        /// @details Records IntervalErrorKind::OutOfRange, interval_value(), interval_lo(), and interval_hi().
        /// @param value Candidate parameter value.
        /// @param lo Exclusive lower bound.
        /// @param hi Exclusive upper bound.
        /// @return GprError with code GprErrorCode::InvalidInterval.
        [[nodiscard]] static GprError invalid_interval_out_of_range(double value, double lo, double hi);

        /// @brief Constructs an InvalidConfig error with details.
        /// @details Formats the configuration error description into the message.
        /// @param reason Human-readable explanation of the invalid configuration setting.
        /// @return GprError with code GprErrorCode::InvalidConfig.
        [[nodiscard]] static GprError invalid_config(std::string reason);

        /// @brief Constructs an InvalidDistance error identifying slot, ARD dimension, matrix pair, and cause.
        /// @details Stores all provided location metadata so outer wrappers can fill unset slot or dimension indices.
        /// @param slot Optional distance slot index where the violation occurred.
        /// @param dim Optional ARD dimension index within the distance source.
        /// @param pair Optional (row, col) matrix entry coordinates of the offending value.
        /// @param reason Human-readable cause of the distance validation failure.
        /// @return GprError with code GprErrorCode::InvalidDistance.
        [[nodiscard]] static GprError invalid_distance(
            std::optional<std::size_t> slot,
            std::optional<std::size_t> dim,
            std::optional<std::pair<std::size_t, std::size_t>> pair,
            std::string reason
        );

        /// @brief Constructs a DistanceSlot error for a slot matching mismatch.
        /// @details Populates slot_error_kind() and optional distance_slot().
        /// @param kind Specific slot matching failure category.
        /// @param slot Optional offending slot index.
        /// @return GprError with code GprErrorCode::DistanceSlot.
        [[nodiscard]] static GprError distance_slot(SlotErrorKind kind, std::optional<std::size_t> slot);

        /// @brief Constructs an UnsupportedKernelOperation error with details.
        /// @details Formats the unsupported operation explanation into the diagnostic message.
        /// @param reason Human-readable explanation of why the kernel cannot perform the operation.
        /// @return GprError with code GprErrorCode::UnsupportedKernelOperation.
        [[nodiscard]] static GprError unsupported_kernel_operation(std::string reason);

        /// @brief Constructs a CholeskyFailed error recording the last jitter attempted, matrix size, and stage.
        /// @details Populates jitter(), matrix_size(), and cholesky_stage().
        /// @param jitter Final diagonal jitter value already attempted before failure.
        /// @param matrix_size Order of the square matrix whose factorization failed.
        /// @param stage Model lifecycle stage at which factorization failed.
        /// @return GprError with code GprErrorCode::CholeskyFailed.
        [[nodiscard]] static GprError cholesky_failed(double jitter, std::size_t matrix_size, CholeskyStage stage);

        /// @brief Constructs a NonPositiveDefiniteMatrix error.
        /// @details Uses a static diagnostic message for strict definiteness checks without jitter recovery.
        /// @return GprError with code GprErrorCode::NonPositiveDefiniteMatrix.
        [[nodiscard]] static GprError non_positive_definite_matrix();

        /// @brief Constructs an OptimizationNotConverged error recording completed iterations.
        /// @details Populates iterations() alongside a formatted diagnostic message.
        /// @param iterations Number of optimization iterations completed before stopping.
        /// @return GprError with code GprErrorCode::OptimizationNotConverged.
        [[nodiscard]] static GprError optimization_not_converged(std::size_t iterations);

        /// @brief Constructs a WorkspaceTooSmall error.
        /// @details Uses a static diagnostic message indicating insufficient caller buffer capacity.
        /// @return GprError with code GprErrorCode::WorkspaceTooSmall.
        [[nodiscard]] static GprError workspace_too_small();

        /// @brief Constructs a PersistFailed error identifying the failure kind and reason.
        /// @details Populates persist_error_kind() alongside a formatted diagnostic message.
        /// @param kind Persistence failure sub-category.
        /// @param reason Human-readable failure description.
        /// @return GprError with code GprErrorCode::PersistFailed.
        [[nodiscard]] static GprError persist_failed(PersistErrorKind kind, std::string reason);

        /// @brief Constructs an UnsupportedPersistVersion error recording found and supported versions.
        /// @details Populates found_version() and supported_version().
        /// @param found Format version integer read from the persisted artifact.
        /// @param supported Format version integer supported by this library build.
        /// @return GprError with code GprErrorCode::UnsupportedPersistVersion.
        [[nodiscard]] static GprError unsupported_persist_version(std::uint32_t found, std::uint32_t supported);

        /// @brief Returns the error category code.
        /// @details Identifies which GprErrorCode variant this error represents.
        /// @return Error category enum value.
        [[nodiscard]] GprErrorCode code() const noexcept { return code_; }

        /// @brief Returns a human-readable diagnostic message.
        /// @details Prefers owned dynamic message storage when non-empty, otherwise returns the static literal.
        /// @return Non-owning view into the diagnostic message string.
        [[nodiscard]] std::string_view message() const noexcept
        {
            // Return owned dynamic text when present; static errors store only a literal view.
            return dynamic_message_.empty() ? static_message_ : std::string_view{dynamic_message_};
        }

        /// @brief Returns the actual feature column count for DimensionMismatch, if applicable.
        /// @details Only populated when constructed via dimension_mismatch().
        /// @return Supplied feature column count, or std::nullopt.
        [[nodiscard]] std::optional<std::size_t> x_dim() const noexcept { return x_dim_; }

        /// @brief Returns the expected feature dimension for DimensionMismatch, if applicable.
        /// @details Only populated when constructed via dimension_mismatch().
        /// @return Expected feature dimension, or std::nullopt.
        [[nodiscard]] std::optional<std::size_t> expected_dim() const noexcept { return expected_dim_; }

        /// @brief Returns the supplied point count for InsufficientData, if applicable.
        /// @details Only populated when constructed via insufficient_data().
        /// @return Supplied training sample count, or std::nullopt.
        [[nodiscard]] std::optional<std::size_t> n_points() const noexcept { return n_points_; }

        /// @brief Returns the minimum required point count for InsufficientData, if applicable.
        /// @details Only populated when constructed via insufficient_data().
        /// @return Required minimum training sample count, or std::nullopt.
        [[nodiscard]] std::optional<std::size_t> min_points() const noexcept { return min_points_; }

        /// @brief Returns the last diagonal jitter applied for CholeskyFailed, if applicable.
        /// @details Only populated when constructed via cholesky_failed().
        /// @return Last attempted jitter value, or std::nullopt.
        [[nodiscard]] std::optional<double> jitter() const noexcept { return jitter_; }

        /// @brief Returns the square matrix order for CholeskyFailed, if applicable.
        /// @details Only populated when constructed via cholesky_failed().
        /// @return Square matrix dimension n, or std::nullopt.
        [[nodiscard]] std::optional<std::size_t> matrix_size() const noexcept { return matrix_size_; }

        /// @brief Returns the call site stage for CholeskyFailed, if applicable.
        /// @details Only populated when constructed via cholesky_failed().
        /// @return Cholesky call site stage, or std::nullopt.
        [[nodiscard]] std::optional<CholeskyStage> cholesky_stage() const noexcept { return cholesky_stage_; }

        /// @brief Returns the completed iteration count for OptimizationNotConverged, if applicable.
        /// @details Only populated when constructed via optimization_not_converged().
        /// @return Completed optimizer iteration count, or std::nullopt.
        [[nodiscard]] std::optional<std::size_t> iterations() const noexcept { return iterations_; }

        /// @brief Returns the interval error subtype for InvalidInterval, if applicable.
        /// @details Set by invalid_interval_bounds() and invalid_interval_out_of_range().
        /// @return Interval error subtype, or std::nullopt.
        [[nodiscard]] std::optional<IntervalErrorKind> interval_error_kind() const noexcept
        {
            return interval_error_kind_;
        }

        /// @brief Returns the out-of-range value for InvalidInterval::OutOfRange, if applicable.
        /// @details Only populated when constructed via invalid_interval_out_of_range().
        /// @return Candidate parameter value outside the interval, or std::nullopt.
        [[nodiscard]] std::optional<double> interval_value() const noexcept { return interval_value_; }

        /// @brief Returns the lower interval bound for InvalidInterval, if applicable.
        /// @details Set by invalid_interval_bounds() and invalid_interval_out_of_range().
        /// @return Lower interval bound, or std::nullopt.
        [[nodiscard]] std::optional<double> interval_lo() const noexcept { return interval_lo_; }

        /// @brief Returns the upper interval bound for InvalidInterval, if applicable.
        /// @details Set by invalid_interval_bounds() and invalid_interval_out_of_range().
        /// @return Upper interval bound, or std::nullopt.
        [[nodiscard]] std::optional<double> interval_hi() const noexcept { return interval_hi_; }

        /// @brief Returns the offending distance slot index for distance errors, if known.
        /// @details May be populated on InvalidDistance or DistanceSlot errors.
        /// @return Offending distance slot index, or std::nullopt.
        [[nodiscard]] std::optional<std::size_t> distance_slot() const noexcept { return distance_slot_; }

        /// @brief Returns the offending ARD dimension index for InvalidDistance, if known.
        /// @details Set directly at construction or populated via with_dim_if_unset().
        /// @return Offending ARD dimension index, or std::nullopt.
        [[nodiscard]] std::optional<std::size_t> distance_dim() const noexcept { return distance_dim_; }

        /// @brief Returns the offending matrix entry pair (row, col) for InvalidDistance, if known.
        /// @details Populated when an element-level distance check fails at a specific matrix entry.
        /// @return Zero-based (row, col) index pair, or std::nullopt.
        [[nodiscard]] std::optional<std::pair<std::size_t, std::size_t>> distance_pair() const noexcept
        {
            return distance_pair_;
        }

        /// @brief Returns the cause description for InvalidDistance, if applicable.
        /// @details Returns std::nullopt whenever code() != GprErrorCode::InvalidDistance.
        /// @return Distance failure cause string view, or std::nullopt.
        [[nodiscard]] std::optional<std::string_view> distance_reason() const noexcept
        {
            // Only expose distance_reason_ for InvalidDistance errors so other codes return nullopt.
            if (code_ != GprErrorCode::InvalidDistance)
            {
                return std::nullopt;
            }
            return std::string_view{distance_reason_};
        }

        /// @brief Returns the slot error subtype for DistanceSlot, if applicable.
        /// @details Only populated when constructed via distance_slot().
        /// @return Distance slot mismatch subtype, or std::nullopt.
        [[nodiscard]] std::optional<SlotErrorKind> slot_error_kind() const noexcept { return slot_error_kind_; }

        /// @brief Returns the persistence failure subtype for PersistFailed, if applicable.
        /// @details Only populated when constructed via persist_failed().
        /// @return Persistence failure subtype, or std::nullopt.
        [[nodiscard]] std::optional<PersistErrorKind> persist_error_kind() const noexcept
        {
            return persist_error_kind_;
        }

        /// @brief Returns the discovered persist format version for UnsupportedPersistVersion, if applicable.
        /// @details Only populated when constructed via unsupported_persist_version().
        /// @return Format version found in artifact, or std::nullopt.
        [[nodiscard]] std::optional<std::uint32_t> found_version() const noexcept { return found_version_; }

        /// @brief Returns the supported persist format version for UnsupportedPersistVersion, if applicable.
        /// @details Only populated when constructed via unsupported_persist_version().
        /// @return Format version supported by this build, or std::nullopt.
        [[nodiscard]] std::optional<std::uint32_t> supported_version() const noexcept { return supported_version_; }

        /// @brief Names the distance slot on an InvalidDistance error if not already set; otherwise returns *this.
        /// @details Rebuilds the formatted message to include slot_place while preserving dim, pair, and reason.
        /// @param slot_place Distance slot index to attach if distance_slot() is empty.
        /// @return Updated GprError with slot set, or an unchanged copy of *this.
        [[nodiscard]] GprError with_slot_if_unset(std::size_t slot_place) const;

        /// @brief Names the ARD dimension on an InvalidDistance error if not already set; otherwise returns *this.
        /// @details Rebuilds the formatted message to include dim_place while preserving slot, pair, and reason.
        /// @param dim_place ARD dimension index to attach if distance_dim() is empty.
        /// @return Updated GprError with dimension set, or an unchanged copy of *this.
        [[nodiscard]] GprError with_dim_if_unset(std::size_t dim_place) const;

        /// @brief Checks structural equality between two GprError instances.
        /// @details Compares error code, message fields, and all optional metadata fields.
        /// @param other GprError instance to compare against.
        /// @return True if all fields compare equal.
        [[nodiscard]] bool operator==(const GprError& other) const = default;

    private:
        /// @brief Constructs a GprError backed by a static string literal without allocating.
        /// @details Leaves dynamic_message_ empty so message() reads static_message_.
        /// @param code Error category code.
        /// @param static_message Static-storage diagnostic string view.
        GprError(GprErrorCode code, std::string_view static_message) noexcept :
            code_(code),
            static_message_(static_message)
        {
        }

        /// @brief Constructs a GprError owning a dynamically formatted diagnostic string.
        /// @details Leaves static_message_ empty so message() reads dynamic_message_.
        /// @param code Error category code.
        /// @param dynamic_message Owned diagnostic message string.
        GprError(GprErrorCode code, std::string dynamic_message) noexcept :
            code_(code),
            dynamic_message_(std::move(dynamic_message))
        {
        }

        /// @brief Error category discriminant.
        GprErrorCode code_;
        /// @brief Non-owning view of a static-storage diagnostic message when dynamic_message_ is empty.
        std::string_view static_message_;
        /// @brief Owned diagnostic string for formatted error variants.
        std::string dynamic_message_;

        /// @brief Actual feature column count for DimensionMismatch.
        std::optional<std::size_t> x_dim_;
        /// @brief Expected feature dimension for DimensionMismatch.
        std::optional<std::size_t> expected_dim_;
        /// @brief Supplied training sample count for InsufficientData.
        std::optional<std::size_t> n_points_;
        /// @brief Minimum required sample count for InsufficientData.
        std::optional<std::size_t> min_points_;
        /// @brief Last attempted diagonal jitter for CholeskyFailed.
        std::optional<double> jitter_;
        /// @brief Matrix order n for CholeskyFailed.
        std::optional<std::size_t> matrix_size_;
        /// @brief Factorization lifecycle stage for CholeskyFailed.
        std::optional<CholeskyStage> cholesky_stage_;
        /// @brief Completed iteration count for OptimizationNotConverged.
        std::optional<std::size_t> iterations_;
        /// @brief Subtype tag for InvalidInterval.
        std::optional<IntervalErrorKind> interval_error_kind_;
        /// @brief Out-of-range candidate value for InvalidInterval::OutOfRange.
        std::optional<double> interval_value_;
        /// @brief Lower interval endpoint for InvalidInterval.
        std::optional<double> interval_lo_;
        /// @brief Upper interval endpoint for InvalidInterval.
        std::optional<double> interval_hi_;
        /// @brief Offending distance slot index for InvalidDistance or DistanceSlot.
        std::optional<std::size_t> distance_slot_;
        /// @brief Offending ARD dimension index for InvalidDistance.
        std::optional<std::size_t> distance_dim_;
        /// @brief Offending (row, col) matrix coordinates for InvalidDistance.
        std::optional<std::pair<std::size_t, std::size_t>> distance_pair_;
        /// @brief Underlying cause text preserved separately for InvalidDistance enrichment.
        std::string distance_reason_;
        /// @brief Subtype tag for DistanceSlot.
        std::optional<SlotErrorKind> slot_error_kind_;
        /// @brief Subtype tag for PersistFailed.
        std::optional<PersistErrorKind> persist_error_kind_;
        /// @brief Discovered persistence format version for UnsupportedPersistVersion.
        std::optional<std::uint32_t> found_version_;
        /// @brief Supported persistence format version for UnsupportedPersistVersion.
        std::optional<std::uint32_t> supported_version_;
    };

}  // namespace cxxgpr
