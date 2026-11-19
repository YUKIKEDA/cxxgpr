// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-01, REQ-02, REQ-05, DES-10
/// @file distance_test.cpp
/// @brief Unit and property tests for supplied-distance kernels, slots, tidy repair, and parity.
/// @details Verifies ScalarDistance/ArdDistance slot creation and deduplication, atomic set_params,
/// DistanceSource error handling and symmetric tidy() repair, numerical parity of DistanceKernel
/// against coordinate kernels for values/cross-covariance/gradients/Hessians, and zero allocations.

#include "cxxgpr/kernel/kernel.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

#include "allocation_audit.hpp"

namespace
{

    /// @brief Computes a column-major rectangular pairwise squared Euclidean distance matrix.
    /// @details Evaluates ||x1_i - x2_j||_2^2 for cross-covariance and square distance tests.
    /// @param[in] x1 Left point matrix of shape n1 x d.
    /// @param[in] x2 Right point matrix of shape n2 x d.
    /// @return Column-major n1 x n2 vector of squared Euclidean distances.
    std::vector<double> make_sq_dist_rect(cxxgpr::MatrixRef<double> x1, cxxgpr::MatrixRef<double> x2)
    {
        std::vector<double> d2(x1.rows() * x2.rows(), 0.0);
        // Accumulate squared differences across all feature coordinates per pair.
        for (std::size_t col = 0; col < x2.rows(); ++col)
        {
            for (std::size_t row = 0; row < x1.rows(); ++row)
            {
                double acc = 0.0;
                for (std::size_t d = 0; d < x1.cols(); ++d)
                {
                    const double diff = x1(row, d) - x2(col, d);
                    acc += diff * diff;
                }
                d2[(col * x1.rows()) + row] = acc;
            }
        }
        return d2;
    }

    /// @brief Computes per-dimension pairwise squared coordinate differences (x1_{i,dim} - x2_{j,dim})^2.
    /// @details Produces one column-major n1 x n2 difference slice for binding into an ArdDistance slot.
    /// @param[in] x1 Left point matrix of shape n1 x d.
    /// @param[in] x2 Right point matrix of shape n2 x d.
    /// @param[in] dim Zero-based feature dimension index.
    /// @return Column-major n1 x n2 vector of single-dimension squared differences.
    std::vector<double> make_dim_sq_diff_rect(
        cxxgpr::MatrixRef<double> x1, cxxgpr::MatrixRef<double> x2, std::size_t dim
    )
    {
        std::vector<double> d2(x1.rows() * x2.rows(), 0.0);
        for (std::size_t col = 0; col < x2.rows(); ++col)
        {
            const std::size_t r2 = col;
            for (std::size_t row = 0; row < x1.rows(); ++row)
            {
                const double diff = x1(row, dim) - x2(r2, dim);
                d2[(col * x1.rows()) + row] = diff * diff;
            }
        }
        return d2;
    }

    /// @brief Verifies distance slot creation, slot deduplication, and atomic set_params rollback.
    /// @details Checks EmptyInput for zero-dimension ArdDistance, DimensionMismatch on mismatched ARD
    /// leaf dimensions, unique slot merging across repeated scalar slot uses, and rollback on failure.
    TEST(DistanceKernelTest, SlotsBindingsAndAtomicSetParamsWork)
    {
        // Reject zero-dimension ARD slot construction and mismatched leaf dimensions.
        EXPECT_EQ(cxxgpr::ArdDistance::create(0U).error().code(), cxxgpr::GprErrorCode::EmptyInput);

        const cxxgpr::ScalarDistance s_slot;
        const auto a_slot = cxxgpr::ArdDistance::create(2U);
        ASSERT_TRUE(a_slot.has_value());

        const auto rbf = cxxgpr::RbfKernel::create(1.2);
        const auto m32 = cxxgpr::MaternKernel::create(0.9, cxxgpr::MaternNu::ThreeHalves);
        const auto rbf_ard = cxxgpr::RbfArdKernel::create(std::array<double, 2>{1.0, 2.0});
        const auto bad_ard = cxxgpr::RbfArdKernel::create(std::array<double, 3>{1.0, 2.0, 3.0});
        const auto c = cxxgpr::ConstantKernel::create(1.5);
        ASSERT_TRUE(rbf && m32 && rbf_ard && bad_ard && c);

        EXPECT_EQ(a_slot->rbf_ard(*bad_ard).error().code(), cxxgpr::GprErrorCode::DimensionMismatch);

        // Same scalar slot referenced twice yields one unique slot descriptor.
        auto dk = (cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(*c) * s_slot.rbf(*rbf)) + s_slot.matern(*m32)
            + *(a_slot->rbf_ard(*rbf_ard));
        EXPECT_EQ(dk.num_params(), 5U);
        EXPECT_EQ(dk.num_scalar_slots(), 1U);
        EXPECT_EQ(dk.num_ard_slots(), 1U);

        const auto slots = dk.slots();
        ASSERT_EQ(slots.size(), 2U);
        EXPECT_EQ(slots[0].id, s_slot.slot());
        EXPECT_FALSE(slots[0].ard_dims.has_value());
        EXPECT_EQ(slots[0].at, 0U);
        EXPECT_EQ(slots[1].id, a_slot->slot());
        EXPECT_EQ(slots[1].ard_dims, 2U);
        EXPECT_EQ(slots[1].at, 0U);

        // Atomic set_params leaves previous parameters untouched on failure.
        std::array<double, 5> orig{};
        ASSERT_TRUE(dk.get_params(orig).has_value());
        const std::array<double, 5> bad_params{
            std::log(3.0), std::numbers::ln2, std::log(1.5), std::log(2.5), std::log(1e6)
        };
        EXPECT_EQ(dk.set_params(bad_params).error().code(), cxxgpr::GprErrorCode::InvalidInterval);
        std::array<double, 5> after{};
        ASSERT_TRUE(dk.get_params(after).has_value());
        for (std::size_t i = 0; i < 5U; ++i)
        {
            EXPECT_DOUBLE_EQ(after[i], orig[i]);
        }
    }

    /// @brief Negative controls for invalid slots/distances and positive check for tidy() repair.
    /// @details Confirms missing slot, foreign slot, duplicate slot, strict rejection of asymmetry or
    /// nonzero diagonals, tidy() tolerance acceptance/rejection, and invalid tidy tolerance checks.
    TEST(DistanceKernelTest, RejectsInvalidDistancesSlotMismatchesAndSupportsTidyRepair)
    {
        const cxxgpr::ScalarDistance s1;
        const cxxgpr::ScalarDistance other_slot;
        const auto rbf = cxxgpr::RbfKernel::create(1.0);
        ASSERT_TRUE(rbf.has_value());

        const auto compiled = s1.rbf(*rbf).compile<double>();

        // Missing slot.
        cxxgpr::DistanceSource empty_src;
        const auto missing = compiled.prepare_square(empty_src);
        ASSERT_FALSE(missing.has_value());
        EXPECT_EQ(missing.error().code(), cxxgpr::GprErrorCode::DistanceSlot);

        // Slot from another kernel -> UnsupportedKernelOperation.
        std::array<double, 4> valid_d2{0.0, 1.0, 1.0, 0.0};
        cxxgpr::DistanceSource foreign_src;
        foreign_src.bind_scalar(other_slot.slot(), cxxgpr::MatrixRef<double>(valid_d2.data(), 2, 2));
        EXPECT_EQ(
            compiled.prepare_square(foreign_src).error().code(), cxxgpr::GprErrorCode::UnsupportedKernelOperation
        );

        // Duplicate slot -> DistanceSlot.
        cxxgpr::DistanceSource dup_src;
        dup_src.bind_scalar(s1.slot(), cxxgpr::MatrixRef<double>(valid_d2.data(), 2, 2))
            .bind_scalar(s1.slot(), cxxgpr::MatrixRef<double>(valid_d2.data(), 2, 2));
        EXPECT_EQ(compiled.prepare_square(dup_src).error().code(), cxxgpr::GprErrorCode::DistanceSlot);

        // Strict check rejects minor asymmetry and nonzero diagonal.
        std::array<double, 4> noisy_d2{1e-9, 1.0, 1.0 + 1e-9, -1e-10};
        cxxgpr::DistanceSource strict_src;
        strict_src.bind_scalar(s1.slot(), cxxgpr::MatrixRef<double>(noisy_d2.data(), 2, 2));
        const auto strict_err = compiled.prepare_square(strict_src);
        ASSERT_FALSE(strict_err.has_value());
        EXPECT_EQ(strict_err.error().code(), cxxgpr::GprErrorCode::InvalidDistance);

        // Tidy within 1e-6 repairs diagonal, negative, and mirror asymmetry to mean.
        cxxgpr::DistanceSource tidy_src;
        tidy_src.bind_scalar(s1.slot(), cxxgpr::MatrixRef<double>(noisy_d2.data(), 2, 2)).tidy(1e-6);
        const auto repaired = compiled.prepare_square(tidy_src);
        ASSERT_TRUE(repaired.has_value());

        // Tidy exceeding tolerance fails with InvalidDistance.
        cxxgpr::DistanceSource tight_src;
        tight_src.bind_scalar(s1.slot(), cxxgpr::MatrixRef<double>(noisy_d2.data(), 2, 2)).tidy(1e-12);
        EXPECT_EQ(compiled.prepare_square(tight_src).error().code(), cxxgpr::GprErrorCode::InvalidDistance);

        // Invalid tidy tolerance (negative / NaN) fails with InvalidConfig.
        cxxgpr::DistanceSource bad_tol_src;
        bad_tol_src.bind_scalar(s1.slot(), cxxgpr::MatrixRef<double>(valid_d2.data(), 2, 2)).tidy(-0.1);
        EXPECT_EQ(compiled.prepare_square(bad_tol_src).error().code(), cxxgpr::GprErrorCode::InvalidConfig);
    }

    /// @brief Verifies hybrid DistanceKernel + LinearKernel parity and zero-allocation hot path.
    /// @details Compares square Gram, cross-covariance, all parameter gradients, and all Hessians
    /// between a WithPoints supplied-distance composite and an equivalent coordinate tree, then
    /// audits zero heap allocations under NoAllocationScope.
    TEST(DistanceKernelTest, MatchesCoordinateKernelValuesGradHessAndZeroAllocHotPath)
    {
        // Build 3x2 training and 2x2 query point sets along with scalar and per-dim squared distances.
        const std::array<double, 6> x_buf{0.1, 1.2, 2.3, -0.4, 0.7, 1.5};
        const std::array<double, 4> x2_buf{0.5, 1.8, 0.2, -0.3};
        const cxxgpr::MatrixRef<double> x_ref(x_buf.data(), 3, 2);
        const cxxgpr::MatrixRef<double> x2_ref(x2_buf.data(), 2, 2);

        const auto scalar_sq = make_sq_dist_rect(x_ref, x_ref);
        const auto scalar_cross = make_sq_dist_rect(x_ref, x2_ref);
        const auto ard_d0 = make_dim_sq_diff_rect(x_ref, x_ref, 0);
        const auto ard_d1 = make_dim_sq_diff_rect(x_ref, x_ref, 1);
        const auto ard_c0 = make_dim_sq_diff_rect(x_ref, x2_ref, 0);
        const auto ard_c1 = make_dim_sq_diff_rect(x_ref, x2_ref, 1);

        const cxxgpr::ScalarDistance s_slot;
        const auto a_slot = cxxgpr::ArdDistance::create(2U);
        ASSERT_TRUE(a_slot.has_value());

        const auto c = cxxgpr::ConstantKernel::create(1.6);
        const auto rbf = cxxgpr::RbfKernel::create(1.2);
        const auto m52 = cxxgpr::MaternArdKernel::create(std::array<double, 2>{0.9, 1.5}, cxxgpr::MaternNu::FiveHalves);
        const auto rq_ard = cxxgpr::RationalQuadraticArdKernel::create(std::array<double, 2>{1.1, 1.8}, 2.0);
        const auto lin = cxxgpr::LinearKernel::create(0.7);
        const auto white = cxxgpr::WhiteKernel::create(0.05);
        ASSERT_TRUE(c && rbf && m52 && rq_ard && lin && white);

        // Coordinate reference tree.
        const cxxgpr::KernelSpec coord_spec = (cxxgpr::KernelSpec(*c) * cxxgpr::KernelSpec(*rbf))
            + cxxgpr::KernelSpec(*m52) + cxxgpr::KernelSpec(*rq_ard) + cxxgpr::KernelSpec(*lin)
            + cxxgpr::KernelSpec(*white);
        const auto coord_compiled = coord_spec.compile<double>();

        // Equivalent WithPoints supplied-distance + Linear coordinate tree.
        const auto dist_only = (cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(*c) * s_slot.rbf(*rbf))
            + *(a_slot->matern_ard(*m52)) + *(a_slot->rational_quadratic_ard(*rq_ard));
        const auto with_pts = dist_only + cxxgpr::KernelSpec(*lin) + cxxgpr::KernelSpec(*white);
        const auto dist_compiled = with_pts.compile<double>();
        EXPECT_TRUE(dist_compiled.requires_points());

        cxxgpr::DistanceSource sq_src;
        const std::array<cxxgpr::MatrixRef<double>, 2> ard_sq_refs{
            cxxgpr::MatrixRef<double>(ard_d0.data(), 3, 3),
            cxxgpr::MatrixRef<double>(ard_d1.data(), 3, 3),
        };
        sq_src.bind_scalar(s_slot.slot(), cxxgpr::MatrixRef<double>(scalar_sq.data(), 3, 3))
            .bind_ard(a_slot->slot(), ard_sq_refs);
        const auto sq_store = dist_compiled.prepare_square(sq_src);
        ASSERT_TRUE(sq_store.has_value());

        cxxgpr::DistanceSource cross_src;
        const std::array<cxxgpr::MatrixRef<double>, 2> ard_cross_refs{
            cxxgpr::MatrixRef<double>(ard_c0.data(), 3, 2),
            cxxgpr::MatrixRef<double>(ard_c1.data(), 3, 2),
        };
        cross_src.bind_scalar(s_slot.slot(), cxxgpr::MatrixRef<double>(scalar_cross.data(), 3, 2))
            .bind_ard(a_slot->slot(), ard_cross_refs);
        const auto cross_store = dist_compiled.prepare_cross(cross_src);
        ASSERT_TRUE(cross_store.has_value());

        std::vector<double> k_coord(9, 0.0);
        std::vector<double> k_dist(9, 0.0);
        std::vector<double> sa(9, 0.0);
        std::vector<double> sb(9, 0.0);
        std::vector<double> sc(9, 0.0);
        auto nested_coord = coord_compiled.make_nested_buffers(3, 3);
        auto nested_dist = dist_compiled.make_nested_buffers(3, 3);

        ASSERT_TRUE(coord_compiled
                .apply_points(
                    x_ref, cxxgpr::MatrixMut<double>(k_coord.data(), 3, 3), cxxgpr::Triangle::Lower,
                    cxxgpr::MatrixMut<double>(sa.data(), 3, 3), nested_coord
                )
                .has_value());
        ASSERT_TRUE(dist_compiled
                .apply(
                    *sq_store, x_ref, cxxgpr::MatrixMut<double>(k_dist.data(), 3, 3), cxxgpr::Triangle::Lower,
                    cxxgpr::MatrixMut<double>(sa.data(), 3, 3), nested_dist
                )
                .has_value());
        for (std::size_t col = 0; col < 3; ++col)
        {
            for (std::size_t row = col; row < 3; ++row)
            {
                EXPECT_NEAR(k_dist[(col * 3) + row], k_coord[(col * 3) + row], 1e-13);
            }
        }

        // Cross covariance match.
        std::vector<double> kc_coord(6, 0.0);
        std::vector<double> kc_dist(6, 0.0);
        std::vector<double> s_cross(6, 0.0);
        auto nc_coord = coord_compiled.make_nested_buffers(3, 2);
        auto nc_dist = dist_compiled.make_nested_buffers(3, 2);
        ASSERT_TRUE(coord_compiled
                .apply_cross_points(
                    x_ref, x2_ref, cxxgpr::MatrixMut<double>(kc_coord.data(), 3, 2),
                    cxxgpr::MatrixMut<double>(s_cross.data(), 3, 2), nc_coord
                )
                .has_value());
        ASSERT_TRUE(dist_compiled
                .apply_cross(
                    *cross_store, x_ref, x2_ref, cxxgpr::MatrixMut<double>(kc_dist.data(), 3, 2),
                    cxxgpr::MatrixMut<double>(s_cross.data(), 3, 2), nc_dist
                )
                .has_value());
        for (std::size_t idx = 0; idx < 6U; ++idx)
        {
            EXPECT_NEAR(kc_dist[idx], kc_coord[idx], 1e-13);
        }

        // Compare all gradients and Hessians against coordinate reference.
        const std::size_t np = coord_compiled.num_params();
        ASSERT_EQ(dist_compiled.num_params(), np);
        std::vector<double> g_coord(9, 0.0);
        std::vector<double> g_dist(9, 0.0);
        for (std::size_t p = 0; p < np; ++p)
        {
            ASSERT_TRUE(coord_compiled
                    .grad_points(
                        x_ref, cxxgpr::MatrixMut<double>(g_coord.data(), 3, 3), p, cxxgpr::Triangle::Upper,
                        cxxgpr::MatrixMut<double>(sa.data(), 3, 3), cxxgpr::MatrixMut<double>(sb.data(), 3, 3),
                        nested_coord
                    )
                    .has_value());
            ASSERT_TRUE(dist_compiled
                    .grad(
                        *sq_store, x_ref, cxxgpr::MatrixMut<double>(g_dist.data(), 3, 3), p, cxxgpr::Triangle::Upper,
                        cxxgpr::MatrixMut<double>(sa.data(), 3, 3), cxxgpr::MatrixMut<double>(sb.data(), 3, 3),
                        nested_dist
                    )
                    .has_value());
            for (std::size_t col = 0; col < 3; ++col)
            {
                for (std::size_t row = 0; row <= col; ++row)
                {
                    EXPECT_NEAR(g_dist[(col * 3) + row], g_coord[(col * 3) + row], 1e-13);
                }
            }
        }

        for (std::size_t i = 0; i < np; ++i)
        {
            for (std::size_t j = 0; j < np; ++j)
            {
                ASSERT_TRUE(coord_compiled
                        .hess_points(
                            x_ref, cxxgpr::MatrixMut<double>(g_coord.data(), 3, 3), i, j, cxxgpr::Triangle::Lower,
                            cxxgpr::MatrixMut<double>(sa.data(), 3, 3), cxxgpr::MatrixMut<double>(sb.data(), 3, 3),
                            cxxgpr::MatrixMut<double>(sc.data(), 3, 3), nested_coord
                        )
                        .has_value());
                ASSERT_TRUE(dist_compiled
                        .hess(
                            *sq_store, x_ref, cxxgpr::MatrixMut<double>(g_dist.data(), 3, 3), i, j,
                            cxxgpr::Triangle::Lower, cxxgpr::MatrixMut<double>(sa.data(), 3, 3),
                            cxxgpr::MatrixMut<double>(sb.data(), 3, 3), cxxgpr::MatrixMut<double>(sc.data(), 3, 3),
                            nested_dist
                        )
                        .has_value());
                for (std::size_t col = 0; col < 3; ++col)
                {
                    for (std::size_t row = col; row < 3; ++row)
                    {
                        EXPECT_NEAR(g_dist[(col * 3) + row], g_coord[(col * 3) + row], 1e-13);
                    }
                }
            }
        }

        // Zero-allocation audit on warmed-up evaluation / grad / hess.
        {
            const cxxgpr::test_support::NoAllocationScope audit;
            ASSERT_TRUE(dist_compiled
                    .apply(
                        *sq_store, x_ref, cxxgpr::MatrixMut<double>(k_dist.data(), 3, 3), cxxgpr::Triangle::Lower,
                        cxxgpr::MatrixMut<double>(sa.data(), 3, 3), nested_dist
                    )
                    .has_value());
            ASSERT_TRUE(dist_compiled
                    .grad(
                        *sq_store, x_ref, cxxgpr::MatrixMut<double>(g_dist.data(), 3, 3), 2U, cxxgpr::Triangle::Lower,
                        cxxgpr::MatrixMut<double>(sa.data(), 3, 3), cxxgpr::MatrixMut<double>(sb.data(), 3, 3),
                        nested_dist
                    )
                    .has_value());
            ASSERT_TRUE(dist_compiled
                    .hess(
                        *sq_store, x_ref, cxxgpr::MatrixMut<double>(g_dist.data(), 3, 3), 0U, 2U,
                        cxxgpr::Triangle::Lower, cxxgpr::MatrixMut<double>(sa.data(), 3, 3),
                        cxxgpr::MatrixMut<double>(sb.data(), 3, 3), cxxgpr::MatrixMut<double>(sc.data(), 3, 3),
                        nested_dist
                    )
                    .has_value());
            EXPECT_EQ(audit.delta_allocations(), 0U);
        }
    }

}  // namespace
