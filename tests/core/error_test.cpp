// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/core/error.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// Spec: REQ-02, REQ-11, DES-01, DES-02
namespace {

TEST(ErrorTest, AllErrorCodesConstructWithNonEmptyMessagesAndMetadata) {
    const std::vector<std::pair<cxxgpr::GprError, cxxgpr::GprErrorCode>> cases = {
        {cxxgpr::GprError::empty_input(), cxxgpr::GprErrorCode::EmptyInput},
        {cxxgpr::GprError::size_overflow(), cxxgpr::GprErrorCode::SizeOverflow},
        {cxxgpr::GprError::length_mismatch("expected 3 values, got 2"),
         cxxgpr::GprErrorCode::LengthMismatch},
        {cxxgpr::GprError::dimension_mismatch(3, 2), cxxgpr::GprErrorCode::DimensionMismatch},
        {cxxgpr::GprError::index_out_of_range("index 4 out of range 0..3"),
         cxxgpr::GprErrorCode::IndexOutOfRange},
        {cxxgpr::GprError::insufficient_data(1, 2), cxxgpr::GprErrorCode::InsufficientData},
        {cxxgpr::GprError::non_finite_input(), cxxgpr::GprErrorCode::NonFiniteInput},
        {cxxgpr::GprError::non_finite_kernel_value(), cxxgpr::GprErrorCode::NonFiniteKernelValue},
        {cxxgpr::GprError::invalid_hyperparameter("lengthscale must be positive"),
         cxxgpr::GprErrorCode::InvalidHyperparameter},
        {cxxgpr::GprError::invalid_noise_variance("noise variance must be positive"),
         cxxgpr::GprErrorCode::InvalidNoiseVariance},
        {cxxgpr::GprError::invalid_interval_bounds(1.0, 0.0),
         cxxgpr::GprErrorCode::InvalidInterval},
        {cxxgpr::GprError::invalid_interval_out_of_range(2.0, 0.0, 1.0),
         cxxgpr::GprErrorCode::InvalidInterval},
        {cxxgpr::GprError::invalid_config("max_iterations must be positive"),
         cxxgpr::GprErrorCode::InvalidConfig},
        {cxxgpr::GprError::invalid_distance(0, 1, std::pair<std::size_t, std::size_t>{2, 0},
                                            "-1 is negative"),
         cxxgpr::GprErrorCode::InvalidDistance},
        {cxxgpr::GprError::distance_slot(cxxgpr::SlotErrorKind::Duplicate, 0),
         cxxgpr::GprErrorCode::DistanceSlot},
        {cxxgpr::GprError::unsupported_kernel_operation("coordinates unsupported"),
         cxxgpr::GprErrorCode::UnsupportedKernelOperation},
        {cxxgpr::GprError::cholesky_failed(1e-6, 4, cxxgpr::CholeskyStage::Fit),
         cxxgpr::GprErrorCode::CholeskyFailed},
        {cxxgpr::GprError::non_positive_definite_matrix(),
         cxxgpr::GprErrorCode::NonPositiveDefiniteMatrix},
        {cxxgpr::GprError::optimization_not_converged(25),
         cxxgpr::GprErrorCode::OptimizationNotConverged},
        {cxxgpr::GprError::workspace_too_small(), cxxgpr::GprErrorCode::WorkspaceTooSmall},
        {cxxgpr::GprError::persist_failed(cxxgpr::PersistErrorKind::Tensor, "missing alpha"),
         cxxgpr::GprErrorCode::PersistFailed},
        {cxxgpr::GprError::unsupported_persist_version(2, 1),
         cxxgpr::GprErrorCode::UnsupportedPersistVersion},
    };

    for (const auto& [err, expected_code] : cases) {
        EXPECT_EQ(err.code(), expected_code);
        EXPECT_FALSE(err.message().empty());
    }
}

TEST(ErrorTest, StructuredPayloadAccessorsMatchConstructedValues) {
    const auto dim_err = cxxgpr::GprError::dimension_mismatch(3, 1);
    EXPECT_EQ(dim_err.x_dim(), 3U);
    EXPECT_EQ(dim_err.expected_dim(), 1U);
    EXPECT_EQ(dim_err.message(), "input dimension mismatch: X.ncols()=3, expected 1");
    EXPECT_FALSE(dim_err.n_points().has_value());

    const auto data_err = cxxgpr::GprError::insufficient_data(2, 5);
    EXPECT_EQ(data_err.n_points(), 2U);
    EXPECT_EQ(data_err.min_points(), 5U);
    EXPECT_EQ(data_err.message(), "insufficient data: n=2, at least 5 points required");

    const auto chol_err =
        cxxgpr::GprError::cholesky_failed(1e-6, 4, cxxgpr::CholeskyStage::Predict);
    EXPECT_EQ(chol_err.jitter(), 1e-6);
    EXPECT_EQ(chol_err.matrix_size(), 4U);
    EXPECT_EQ(chol_err.cholesky_stage(), cxxgpr::CholeskyStage::Predict);
    EXPECT_NE(chol_err.message().find("stage=Predict"), std::string::npos);
    EXPECT_NE(chol_err.message().find("size=4"), std::string::npos);

    const auto opt_err = cxxgpr::GprError::optimization_not_converged(100);
    EXPECT_EQ(opt_err.iterations(), 100U);

    const auto bounds_err = cxxgpr::GprError::invalid_interval_bounds(5.0, 2.0);
    EXPECT_EQ(bounds_err.interval_error_kind(), cxxgpr::IntervalErrorKind::InvalidBounds);
    EXPECT_FALSE(bounds_err.interval_value().has_value());
    EXPECT_EQ(bounds_err.interval_lo(), 5.0);
    EXPECT_EQ(bounds_err.interval_hi(), 2.0);

    const auto range_err = cxxgpr::GprError::invalid_interval_out_of_range(10.0, 1.0, 5.0);
    EXPECT_EQ(range_err.interval_error_kind(), cxxgpr::IntervalErrorKind::OutOfRange);
    EXPECT_EQ(range_err.interval_value(), 10.0);
    EXPECT_EQ(range_err.interval_lo(), 1.0);
    EXPECT_EQ(range_err.interval_hi(), 5.0);

    const auto persist_err =
        cxxgpr::GprError::persist_failed(cxxgpr::PersistErrorKind::WrongModel, "expected Gpr");
    EXPECT_EQ(persist_err.persist_error_kind(), cxxgpr::PersistErrorKind::WrongModel);
    EXPECT_EQ(persist_err.message(), "persist failed (wrong_model): expected Gpr");

    const auto ver_err = cxxgpr::GprError::unsupported_persist_version(2, 1);
    EXPECT_EQ(ver_err.found_version(), 2U);
    EXPECT_EQ(ver_err.supported_version(), 1U);
    EXPECT_EQ(ver_err.message(),
              "unsupported persist format version 2; this library reads version 1");
}

TEST(ErrorTest, DistanceSlotAndWithSlotIfUnsetBehaveConsistently) {
    const auto bare = cxxgpr::GprError::invalid_distance(
        std::nullopt, 2, std::pair<std::size_t, std::size_t>{3, 0}, "-1 is negative");
    EXPECT_EQ(bare.message(), "invalid squared distance in dimension 2 at (3, 0): -1 is negative");
    EXPECT_FALSE(bare.distance_slot().has_value());
    EXPECT_EQ(bare.distance_dim(), 2U);
    EXPECT_EQ(bare.distance_pair(), (std::pair<std::size_t, std::size_t>{3, 0}));

    const auto slotted = bare.with_slot_if_unset(1);
    EXPECT_EQ(slotted.distance_slot(), 1U);
    EXPECT_EQ(slotted.distance_dim(), 2U);
    EXPECT_EQ(slotted.distance_pair(), (std::pair<std::size_t, std::size_t>{3, 0}));
    EXPECT_EQ(slotted.message(),
              "invalid squared distance in slot 1, dimension 2 at (3, 0): -1 is negative");

    // Existing slot must not be overwritten.
    EXPECT_EQ(slotted.with_slot_if_unset(4), slotted);

    // Non-InvalidDistance errors must be preserved unchanged.
    const auto empty = cxxgpr::GprError::empty_input();
    EXPECT_EQ(empty.with_slot_if_unset(0), empty);

    const auto dup_slot = cxxgpr::GprError::distance_slot(cxxgpr::SlotErrorKind::Duplicate, 0);
    EXPECT_EQ(dup_slot.slot_error_kind(), cxxgpr::SlotErrorKind::Duplicate);
    EXPECT_EQ(dup_slot.distance_slot(), 0U);
    EXPECT_EQ(dup_slot.message(),
              "distance slot mismatch in slot 0: two sources were supplied for one distance slot");

    const auto not_read =
        cxxgpr::GprError::distance_slot(cxxgpr::SlotErrorKind::NotRead, std::nullopt);
    EXPECT_EQ(not_read.slot_error_kind(), cxxgpr::SlotErrorKind::NotRead);
    EXPECT_FALSE(not_read.distance_slot().has_value());
}

TEST(ErrorTest, EqualityComparesCodesAndPayloads) {
    EXPECT_EQ(cxxgpr::GprError::empty_input(), cxxgpr::GprError::empty_input());
    EXPECT_NE(cxxgpr::GprError::empty_input(), cxxgpr::GprError::size_overflow());
    EXPECT_NE(cxxgpr::GprError::dimension_mismatch(2, 1),
              cxxgpr::GprError::dimension_mismatch(3, 1));
    EXPECT_EQ(cxxgpr::GprError::invalid_noise_variance("bad"),
              cxxgpr::GprError::invalid_noise_variance("bad"));
}

} // namespace
