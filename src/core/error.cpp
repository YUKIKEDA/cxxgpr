// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/core/error.hpp"

#include <format>
#include <string>
#include <string_view>
#include <utility>

namespace cxxgpr {

namespace {

constexpr std::string_view stage_label(CholeskyStage stage) noexcept {
    switch (stage) {
    case CholeskyStage::Fit:
        return "Fit";
    case CholeskyStage::Predict:
        return "Predict";
    }
    return "Unknown";
}

constexpr std::string_view slot_kind_message(SlotErrorKind kind) noexcept {
    switch (kind) {
    case SlotErrorKind::NotRead:
        return "a supplied source names a slot the kernel does not read";
    case SlotErrorKind::Duplicate:
        return "two sources were supplied for one distance slot";
    case SlotErrorKind::Missing:
        return "a kernel slot has no supplied distance source";
    }
    return "unknown distance slot error";
}

constexpr std::string_view persist_kind_label(PersistErrorKind kind) noexcept {
    switch (kind) {
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
    return "unknown";
}

std::string format_distance_place(std::optional<std::size_t> slot, std::optional<std::size_t> dim,
                                  std::optional<std::pair<std::size_t, std::size_t>> pair) {
    std::string out;
    bool has_location = false;
    if (slot.has_value()) {
        out += std::format(" in slot {}", *slot);
        has_location = true;
    }
    if (dim.has_value()) {
        out += has_location ? std::format(", dimension {}", *dim)
                            : std::format(" in dimension {}", *dim);
    }
    if (pair.has_value()) {
        out += std::format(" at ({}, {})", pair->first, pair->second);
    }
    return out;
}

std::string format_invalid_distance_message(std::optional<std::size_t> slot,
                                            std::optional<std::size_t> dim,
                                            std::optional<std::pair<std::size_t, std::size_t>> pair,
                                            std::string_view reason) {
    return std::format("invalid squared distance{}: {}", format_distance_place(slot, dim, pair),
                       reason);
}

} // namespace

GprError GprError::empty_input() {
    return {GprErrorCode::EmptyInput, std::string_view{"input is empty"}};
}

GprError GprError::size_overflow() {
    return {GprErrorCode::SizeOverflow, std::string_view{"size overflows std::size_t"}};
}

GprError GprError::length_mismatch(std::string reason) {
    return {GprErrorCode::LengthMismatch, std::format("length mismatch: {}", reason)};
}

GprError GprError::dimension_mismatch(std::size_t x_dim, std::size_t expected_dim) {
    GprError err(
        GprErrorCode::DimensionMismatch,
        std::format("input dimension mismatch: X.ncols()={}, expected {}", x_dim, expected_dim));
    err.x_dim_ = x_dim;
    err.expected_dim_ = expected_dim;
    return err;
}

GprError GprError::index_out_of_range(std::string reason) {
    return {GprErrorCode::IndexOutOfRange, std::format("index out of range: {}", reason)};
}

GprError GprError::insufficient_data(std::size_t n_points, std::size_t min_points) {
    GprError err(
        GprErrorCode::InsufficientData,
        std::format("insufficient data: n={}, at least {} points required", n_points, min_points));
    err.n_points_ = n_points;
    err.min_points_ = min_points;
    return err;
}

GprError GprError::non_finite_input() {
    return {GprErrorCode::NonFiniteInput,
            std::string_view{"input contains a non-finite value (NaN/Inf)"}};
}

GprError GprError::non_finite_kernel_value() {
    return {GprErrorCode::NonFiniteKernelValue,
            std::string_view{"kernel evaluation produced a non-finite value"}};
}

GprError GprError::invalid_hyperparameter(std::string reason) {
    return {GprErrorCode::InvalidHyperparameter, std::format("invalid hyperparameter: {}", reason)};
}

GprError GprError::invalid_noise_variance(std::string reason) {
    return {GprErrorCode::InvalidNoiseVariance,
            std::format("invalid observation noise variance: {}", reason)};
}

GprError GprError::invalid_interval_bounds(double lo, double hi) {
    GprError err(
        GprErrorCode::InvalidInterval,
        std::format("interval bounds must be finite and satisfy lo < hi (got lo={}, hi={})", lo,
                    hi));
    err.interval_error_kind_ = IntervalErrorKind::InvalidBounds;
    err.interval_lo_ = lo;
    err.interval_hi_ = hi;
    return err;
}

GprError GprError::invalid_interval_out_of_range(double value, double lo, double hi) {
    GprError err(GprErrorCode::InvalidInterval,
                 std::format("value {} is not strictly inside ({}, {})", value, lo, hi));
    err.interval_error_kind_ = IntervalErrorKind::OutOfRange;
    err.interval_value_ = value;
    err.interval_lo_ = lo;
    err.interval_hi_ = hi;
    return err;
}

GprError GprError::invalid_config(std::string reason) {
    return {GprErrorCode::InvalidConfig, std::format("invalid configuration: {}", reason)};
}

GprError GprError::invalid_distance(std::optional<std::size_t> slot, std::optional<std::size_t> dim,
                                    std::optional<std::pair<std::size_t, std::size_t>> pair,
                                    std::string reason) {
    GprError err(GprErrorCode::InvalidDistance,
                 format_invalid_distance_message(slot, dim, pair, reason));
    err.distance_slot_ = slot;
    err.distance_dim_ = dim;
    err.distance_pair_ = pair;
    err.distance_reason_ = std::move(reason);
    return err;
}

GprError GprError::distance_slot(SlotErrorKind kind, std::optional<std::size_t> slot) {
    GprError err(GprErrorCode::DistanceSlot,
                 std::format("distance slot mismatch{}: {}",
                             format_distance_place(slot, std::nullopt, std::nullopt),
                             slot_kind_message(kind)));
    err.slot_error_kind_ = kind;
    err.distance_slot_ = slot;
    return err;
}

GprError GprError::unsupported_kernel_operation(std::string reason) {
    return {GprErrorCode::UnsupportedKernelOperation,
            std::format("unsupported kernel operation: {}", reason)};
}

GprError GprError::cholesky_failed(double jitter, std::size_t matrix_size, CholeskyStage stage) {
    GprError err(
        GprErrorCode::CholeskyFailed,
        std::format("Cholesky factorization failed (stage={}, size={}, jitter={} already applied)",
                    stage_label(stage), matrix_size, jitter));
    err.jitter_ = jitter;
    err.matrix_size_ = matrix_size;
    err.cholesky_stage_ = stage;
    return err;
}

GprError GprError::non_positive_definite_matrix() {
    return {GprErrorCode::NonPositiveDefiniteMatrix,
            std::string_view{"matrix is not positive semidefinite"}};
}

GprError GprError::optimization_not_converged(std::size_t iterations) {
    GprError err(GprErrorCode::OptimizationNotConverged,
                 std::format("optimizer did not converge after {} iterations", iterations));
    err.iterations_ = iterations;
    return err;
}

GprError GprError::workspace_too_small() {
    return {GprErrorCode::WorkspaceTooSmall,
            std::string_view{"workspace capacity is insufficient"}};
}

GprError GprError::persist_failed(PersistErrorKind kind, std::string reason) {
    GprError err(GprErrorCode::PersistFailed,
                 std::format("persist failed ({}): {}", persist_kind_label(kind), reason));
    err.persist_error_kind_ = kind;
    return err;
}

GprError GprError::unsupported_persist_version(std::uint32_t found, std::uint32_t supported) {
    GprError err(GprErrorCode::UnsupportedPersistVersion,
                 std::format("unsupported persist format version {}; this library reads version {}",
                             found, supported));
    err.found_version_ = found;
    err.supported_version_ = supported;
    return err;
}

GprError GprError::with_slot_if_unset(std::size_t slot_place) const {
    if (code_ != GprErrorCode::InvalidDistance || distance_slot_.has_value()) {
        return *this;
    }
    return invalid_distance(slot_place, distance_dim_, distance_pair_, distance_reason_);
}

} // namespace cxxgpr
