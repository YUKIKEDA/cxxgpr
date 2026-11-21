// SPDX-License-Identifier: Apache-2.0
/// @file error.cpp
/// @brief Implementation of GprError factories, diagnostic formatting, and metadata enrichment.
/// @details Formats diagnostic strings and populates optional structured payloads for every
/// GprErrorCode variant.

#include "cxxgpr/core/error.hpp"

#include <format>
#include <string>
#include <string_view>
#include <utility>

namespace cxxgpr
{

    namespace
    {

        /// @brief Maps a CholeskyStage enum value to a human-readable stage name.
        /// @details Used when formatting CholeskyFailed diagnostic messages.
        /// @param[in] stage Cholesky call site stage.
        /// @return Stage label string ("Fit", "Predict", or "Unknown").
        constexpr std::string_view stage_label(CholeskyStage stage) noexcept
        {
            switch (stage)
            {
                case CholeskyStage::Fit:
                    return "Fit";
                case CholeskyStage::Predict:
                    return "Predict";
            }
            // Defensive fallback if an unenumerated value is cast to CholeskyStage.
            return "Unknown";
        }

        /// @brief Maps a SlotErrorKind enum value to a diagnostic description fragment.
        /// @details Explains slot mismatch semantics inside DistanceSlot error messages.
        /// @param[in] kind Distance slot mismatch subtype.
        /// @return Descriptive clause for the slot error kind.
        constexpr std::string_view slot_kind_message(SlotErrorKind kind) noexcept
        {
            switch (kind)
            {
                case SlotErrorKind::NotRead:
                    return "a supplied source names a slot the kernel does not read";
                case SlotErrorKind::Duplicate:
                    return "two sources were supplied for one distance slot";
                case SlotErrorKind::Missing:
                    return "a kernel slot has no supplied distance source";
            }
            // Defensive fallback for unenumerated SlotErrorKind values.
            return "unknown distance slot error";
        }

        /// @brief Maps a PersistErrorKind enum value to a short snake_case tag.
        /// @details Embedded in PersistFailed messages for concise error identification.
        /// @param[in] kind Persistence failure subtype.
        /// @return Short category identifier string.
        constexpr std::string_view persist_kind_label(PersistErrorKind kind) noexcept
        {
            switch (kind)
            {
                case PersistErrorKind::Io:
                    return "io";
                case PersistErrorKind::Config:
                    return "config";
                case PersistErrorKind::Tensor:
                    return "tensor";
                case PersistErrorKind::InvalidPersistId:
                    return "invalid_persist_id";
                case PersistErrorKind::NotPersistable:
                    return "not_persistable";
                case PersistErrorKind::UnregisteredId:
                    return "unregistered_id";
                case PersistErrorKind::WrongModel:
                    return "wrong_model";
            }
            // Defensive fallback for unenumerated PersistErrorKind values.
            return "unknown";
        }

        /// @brief Formats optional slot, ARD dimension, and entry coordinates into a location clause.
        /// @details Produces fragments such as " in slot 0, dimension 1 at (2, 3)" depending on which
        /// optional fields are present.
        /// @param[in] slot Optional distance slot index.
        /// @param[in] dim Optional ARD dimension index.
        /// @param[in] pair Optional (row, col) matrix coordinates.
        /// @return Formatted location suffix, or an empty string if all fields are std::nullopt.
        std::string format_distance_place(
            std::optional<std::size_t> slot,
            std::optional<std::size_t> dim,
            std::optional<std::pair<std::size_t, std::size_t>> pair
        )
        {
            std::string out;
            bool has_location = false;
            // Append slot, dimension, and matrix coordinates only when populated.
            if (slot.has_value())
            {
                out += std::format(" in slot {}", *slot);
                has_location = true;
            }
            if (dim.has_value())
            {
                out += has_location ? std::format(", dimension {}", *dim) : std::format(" in dimension {}", *dim);
            }
            if (pair.has_value())
            {
                out += std::format(" at ({}, {})", pair->first, pair->second);
            }
            return out;
        }

        /// @brief Formats the full diagnostic message for an InvalidDistance error.
        /// @details Combines optional location coordinates from format_distance_place() with reason.
        /// @param[in] slot Optional distance slot index.
        /// @param[in] dim Optional ARD dimension index.
        /// @param[in] pair Optional (row, col) matrix coordinates.
        /// @param[in] reason Underlying validation failure cause.
        /// @return Complete formatted error string.
        std::string format_invalid_distance_message(
            std::optional<std::size_t> slot,
            std::optional<std::size_t> dim,
            std::optional<std::pair<std::size_t, std::size_t>> pair,
            std::string_view reason
        )
        {
            return std::format("invalid squared distance{}: {}", format_distance_place(slot, dim, pair), reason);
        }

    }  // namespace

    /// @brief Constructs an EmptyInput error.
    /// @details Stores a static string literal without dynamic allocation.
    /// @return GprError with code GprErrorCode::EmptyInput.
    GprError GprError::empty_input()
    {
        return {GprErrorCode::EmptyInput, std::string_view{"input is empty"}};
    }

    /// @brief Constructs a SizeOverflow error.
    /// @details Stores a static string literal without dynamic allocation.
    /// @return GprError with code GprErrorCode::SizeOverflow.
    GprError GprError::size_overflow()
    {
        return {GprErrorCode::SizeOverflow, std::string_view{"size overflows std::size_t"}};
    }

    /// @brief Constructs a LengthMismatch error with details.
    /// @details Prefixes reason with "length mismatch: ".
    /// @param[in] reason Description of expected versus actual length.
    /// @return GprError with code GprErrorCode::LengthMismatch.
    GprError GprError::length_mismatch(std::string reason)
    {
        return {GprErrorCode::LengthMismatch, std::format("length mismatch: {}", reason)};
    }

    /// @brief Constructs a DimensionMismatch error recording actual and expected feature dimensions.
    /// @details Formats the mismatch message and populates x_dim_ and expected_dim_.
    /// @param[in] x_dim Supplied number of feature columns.
    /// @param[in] expected_dim Required number of feature columns.
    /// @return GprError with code GprErrorCode::DimensionMismatch.
    GprError GprError::dimension_mismatch(std::size_t x_dim, std::size_t expected_dim)
    {
        // Format diagnostic text and retain both column counts for programmatic inspection.
        GprError err(
            GprErrorCode::DimensionMismatch,
            std::format("input dimension mismatch: X.ncols()={}, expected {}", x_dim, expected_dim)
        );
        err.x_dim_ = x_dim;
        err.expected_dim_ = expected_dim;
        return err;
    }

    /// @brief Constructs an IndexOutOfRange error with details.
    /// @details Prefixes reason with "index out of range: ".
    /// @param[in] reason Description of the out-of-range index access.
    /// @return GprError with code GprErrorCode::IndexOutOfRange.
    GprError GprError::index_out_of_range(std::string reason)
    {
        return {GprErrorCode::IndexOutOfRange, std::format("index out of range: {}", reason)};
    }

    /// @brief Constructs an InsufficientData error recording provided and minimum required point counts.
    /// @details Formats the sample count message and populates n_points_ and min_points_.
    /// @param[in] n_points Supplied training point count.
    /// @param[in] min_points Minimum required training point count.
    /// @return GprError with code GprErrorCode::InsufficientData.
    GprError GprError::insufficient_data(std::size_t n_points, std::size_t min_points)
    {
        // Store point counts alongside the message so callers can inspect the threshold.
        GprError err(
            GprErrorCode::InsufficientData,
            std::format("insufficient data: n={}, at least {} points required", n_points, min_points)
        );
        err.n_points_ = n_points;
        err.min_points_ = min_points;
        return err;
    }

    /// @brief Constructs a NonFiniteInput error.
    /// @details Stores a static string literal without dynamic allocation.
    /// @return GprError with code GprErrorCode::NonFiniteInput.
    GprError GprError::non_finite_input()
    {
        return {GprErrorCode::NonFiniteInput, std::string_view{"input contains a non-finite value (NaN/Inf)"}};
    }

    /// @brief Constructs a NonFiniteKernelValue error.
    /// @details Stores a static string literal without dynamic allocation.
    /// @return GprError with code GprErrorCode::NonFiniteKernelValue.
    GprError GprError::non_finite_kernel_value()
    {
        return {GprErrorCode::NonFiniteKernelValue, std::string_view{"kernel evaluation produced a non-finite value"}};
    }

    /// @brief Constructs an InvalidHyperparameter error with details.
    /// @details Prefixes reason with "invalid hyperparameter: ".
    /// @param[in] reason Description of the invalid hyperparameter.
    /// @return GprError with code GprErrorCode::InvalidHyperparameter.
    GprError GprError::invalid_hyperparameter(std::string reason)
    {
        return {GprErrorCode::InvalidHyperparameter, std::format("invalid hyperparameter: {}", reason)};
    }

    /// @brief Constructs an InvalidNoiseVariance error with details.
    /// @details Prefixes reason with "invalid observation noise variance: ".
    /// @param[in] reason Description of the invalid noise variance.
    /// @return GprError with code GprErrorCode::InvalidNoiseVariance.
    GprError GprError::invalid_noise_variance(std::string reason)
    {
        return {GprErrorCode::InvalidNoiseVariance, std::format("invalid observation noise variance: {}", reason)};
    }

    /// @brief Constructs an InvalidInterval error for invalid interval bounds (lo, hi).
    /// @details Populates interval_error_kind_, interval_lo_, and interval_hi_.
    /// @param[in] lo Candidate lower endpoint.
    /// @param[in] hi Candidate upper endpoint.
    /// @return GprError with code GprErrorCode::InvalidInterval.
    GprError GprError::invalid_interval_bounds(double lo, double hi)
    {
        // Tag as InvalidBounds and retain both endpoints for diagnostic reporting.
        GprError err(
            GprErrorCode::InvalidInterval,
            std::format("interval bounds must be finite and satisfy lo < hi (got lo={}, hi={})", lo, hi)
        );
        err.interval_error_kind_ = IntervalErrorKind::InvalidBounds;
        err.interval_lo_ = lo;
        err.interval_hi_ = hi;
        return err;
    }

    /// @brief Constructs an InvalidInterval error for a parameter value outside (lo, hi).
    /// @details Populates interval_error_kind_, interval_value_, interval_lo_, and interval_hi_.
    /// @param[in] value Candidate parameter value.
    /// @param[in] lo Exclusive lower bound.
    /// @param[in] hi Exclusive upper bound.
    /// @return GprError with code GprErrorCode::InvalidInterval.
    GprError GprError::invalid_interval_out_of_range(double value, double lo, double hi)
    {
        // Tag as OutOfRange and attach value together with interval bounds.
        GprError err(
            GprErrorCode::InvalidInterval, std::format("value {} is not strictly inside ({}, {})", value, lo, hi)
        );
        err.interval_error_kind_ = IntervalErrorKind::OutOfRange;
        err.interval_value_ = value;
        err.interval_lo_ = lo;
        err.interval_hi_ = hi;
        return err;
    }

    /// @brief Constructs an InvalidConfig error with details.
    /// @details Prefixes reason with "invalid configuration: ".
    /// @param[in] reason Description of the configuration error.
    /// @return GprError with code GprErrorCode::InvalidConfig.
    GprError GprError::invalid_config(std::string reason)
    {
        return {GprErrorCode::InvalidConfig, std::format("invalid configuration: {}", reason)};
    }

    /// @brief Constructs an InvalidDistance error identifying slot, ARD dimension, matrix pair, and cause.
    /// @details Formats the location-aware message and preserves distance_reason_ for later enrichment.
    /// @param[in] slot Optional distance slot index.
    /// @param[in] dim Optional ARD dimension index.
    /// @param[in] pair Optional (row, col) matrix coordinates.
    /// @param[in] reason Human-readable validation failure description.
    /// @return GprError with code GprErrorCode::InvalidDistance.
    GprError GprError::invalid_distance(
        std::optional<std::size_t> slot,
        std::optional<std::size_t> dim,
        std::optional<std::pair<std::size_t, std::size_t>> pair,
        std::string reason
    )
    {
        // Format message before moving reason into distance_reason_ so with_*_if_unset can rebuild it.
        GprError err(GprErrorCode::InvalidDistance, format_invalid_distance_message(slot, dim, pair, reason));
        err.distance_slot_ = slot;
        err.distance_dim_ = dim;
        err.distance_pair_ = pair;
        err.distance_reason_ = std::move(reason);
        return err;
    }

    /// @brief Constructs a DistanceSlot error for a slot matching mismatch.
    /// @details Populates slot_error_kind_ and distance_slot_ with a descriptive slot message.
    /// @param[in] kind Slot mismatch category.
    /// @param[in] slot Optional offending slot index.
    /// @return GprError with code GprErrorCode::DistanceSlot.
    GprError GprError::distance_slot(SlotErrorKind kind, std::optional<std::size_t> slot)
    {
        // Combine optional slot location with the canonical description for kind.
        GprError err(
            GprErrorCode::DistanceSlot,
            std::format(
                "distance slot mismatch{}: {}", format_distance_place(slot, std::nullopt, std::nullopt),
                slot_kind_message(kind)
            )
        );
        err.slot_error_kind_ = kind;
        err.distance_slot_ = slot;
        return err;
    }

    /// @brief Constructs an UnsupportedKernelOperation error with details.
    /// @details Prefixes reason with "unsupported kernel operation: ".
    /// @param[in] reason Description of the unsupported kernel operation.
    /// @return GprError with code GprErrorCode::UnsupportedKernelOperation.
    GprError GprError::unsupported_kernel_operation(std::string reason)
    {
        return {GprErrorCode::UnsupportedKernelOperation, std::format("unsupported kernel operation: {}", reason)};
    }

    /// @brief Constructs a CholeskyFailed error recording the last jitter attempted, matrix size, and stage.
    /// @details Populates jitter_, matrix_size_, and cholesky_stage_.
    /// @param[in] jitter Last diagonal jitter value applied prior to failure.
    /// @param[in] matrix_size Square matrix order n.
    /// @param[in] stage Factorization call site stage.
    /// @return GprError with code GprErrorCode::CholeskyFailed.
    GprError GprError::cholesky_failed(double jitter, std::size_t matrix_size, CholeskyStage stage)
    {
        // Record numeric jitter and matrix order alongside the lifecycle stage.
        GprError err(
            GprErrorCode::CholeskyFailed,
            std::format(
                "Cholesky factorization failed (stage={}, size={}, jitter={} already applied)", stage_label(stage),
                matrix_size, jitter
            )
        );
        err.jitter_ = jitter;
        err.matrix_size_ = matrix_size;
        err.cholesky_stage_ = stage;
        return err;
    }

    /// @brief Constructs a NonPositiveDefiniteMatrix error.
    /// @details Stores a static string literal without dynamic allocation.
    /// @return GprError with code GprErrorCode::NonPositiveDefiniteMatrix.
    GprError GprError::non_positive_definite_matrix()
    {
        return {GprErrorCode::NonPositiveDefiniteMatrix, std::string_view{"matrix is not positive semidefinite"}};
    }

    /// @brief Constructs an OptimizationNotConverged error recording completed iterations.
    /// @details Populates iterations_ alongside the convergence failure message.
    /// @param[in] iterations Number of completed optimizer iterations.
    /// @return GprError with code GprErrorCode::OptimizationNotConverged.
    GprError GprError::optimization_not_converged(std::size_t iterations)
    {
        // Preserve the iteration count reported by the user-defined optimizer that gave up.
        GprError err(
            GprErrorCode::OptimizationNotConverged,
            std::format("optimizer did not converge after {} iterations", iterations)
        );
        err.iterations_ = iterations;
        return err;
    }

    /// @brief Constructs a WorkspaceTooSmall error.
    /// @details Stores a static string literal without dynamic allocation.
    /// @return GprError with code GprErrorCode::WorkspaceTooSmall.
    GprError GprError::workspace_too_small()
    {
        return {GprErrorCode::WorkspaceTooSmall, std::string_view{"workspace capacity is insufficient"}};
    }

    /// @brief Constructs a PersistFailed error identifying the failure kind and reason.
    /// @details Populates persist_error_kind_ and formats the sub-category label into the message.
    /// @param[in] kind Persistence failure subtype.
    /// @param[in] reason Human-readable persistence failure description.
    /// @return GprError with code GprErrorCode::PersistFailed.
    GprError GprError::persist_failed(PersistErrorKind kind, std::string reason)
    {
        // Attach the PersistErrorKind subtype tag while formatting the message.
        GprError err(
            GprErrorCode::PersistFailed, std::format("persist failed ({}): {}", persist_kind_label(kind), reason)
        );
        err.persist_error_kind_ = kind;
        return err;
    }

    /// @brief Constructs an UnsupportedPersistVersion error recording found and supported versions.
    /// @details Populates found_version_ and supported_version_.
    /// @param[in] found Version number discovered in the persisted artifact.
    /// @param[in] supported Version number supported by this library build.
    /// @return GprError with code GprErrorCode::UnsupportedPersistVersion.
    GprError GprError::unsupported_persist_version(std::uint32_t found, std::uint32_t supported)
    {
        // Record both found and supported format versions for compatibility diagnostics.
        GprError err(
            GprErrorCode::UnsupportedPersistVersion,
            std::format("unsupported persist format version {}; this library reads version {}", found, supported)
        );
        err.found_version_ = found;
        err.supported_version_ = supported;
        return err;
    }

    /// @brief Names the distance slot on an InvalidDistance error if not already set; otherwise returns *this.
    /// @details Reconstructs the error via invalid_distance() so the formatted message includes the slot index.
    /// @param[in] slot_place Distance slot index to attach if distance_slot_ is unset.
    /// @return Updated GprError with slot set, or an unmodified copy of *this.
    GprError GprError::with_slot_if_unset(std::size_t slot_place) const
    {
        // Only enrich InvalidDistance errors whose slot has not yet been annotated by an inner validator.
        if (code_ != GprErrorCode::InvalidDistance || distance_slot_.has_value())
        {
            return *this;
        }
        return invalid_distance(slot_place, distance_dim_, distance_pair_, distance_reason_);
    }

    /// @brief Names the ARD dimension on an InvalidDistance error if not already set; otherwise returns *this.
    /// @details Reconstructs the error via invalid_distance() so the formatted message includes the ARD dimension.
    /// @param[in] dim_place ARD dimension index to attach if distance_dim_ is unset.
    /// @return Updated GprError with dimension set, or an unmodified copy of *this.
    GprError GprError::with_dim_if_unset(std::size_t dim_place) const
    {
        // Only enrich InvalidDistance errors whose ARD dimension index is still unset.
        if (code_ != GprErrorCode::InvalidDistance || distance_dim_.has_value())
        {
            return *this;
        }
        return invalid_distance(distance_slot_, dim_place, distance_pair_, distance_reason_);
    }

}  // namespace cxxgpr
