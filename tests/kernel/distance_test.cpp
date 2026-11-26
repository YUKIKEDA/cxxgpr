// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-01, REQ-02, REQ-05, DES-10
/// @file distance_test.cpp
/// @brief Unit and property tests for supplied-distance kernels, slots, tidy repair, and parity.
/// @details Verifies ScalarDistance/ArdDistance slot creation and deduplication, atomic set_params,
/// DistanceSource error handling and symmetric tidy() repair, numerical parity of DistanceKernel
/// against coordinate kernels for values/cross-covariance/gradients/Hessians, and zero allocations.

#include "cxxgpr/kernel/kernel.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "allocation_audit.hpp"
#include "small_stack.hpp"

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

    /// @brief Verifies a checked distance store can be moved out of std::expected and move-assigned.
    /// @details The move operations must transfer the validated tables: the moved-to store reports the same shape
    /// and the original (moved-from) store is not required to be usable. This compiles only when the move
    /// constructor and assignment are defined where the implementation type is complete.
    /// @tparam T Storage scalar type of the compiled distance kernel and its checked store.
    template <typename T> void expect_checked_distance_store_is_movable()
    {
        const cxxgpr::ScalarDistance slot;
        const auto rbf = cxxgpr::RbfKernel::create(1.0);
        ASSERT_TRUE(rbf.has_value());
        const auto compiled = slot.rbf(*rbf).template compile<T>();

        std::array<double, 9> d2{0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0};
        cxxgpr::DistanceSource src;
        src.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(d2.data(), 3, 3));
        std::array<double, 6> d2_cross{0.0, 1.0, 4.0, 1.0, 0.0, 1.0};
        cxxgpr::DistanceSource cross_src;
        cross_src.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(d2_cross.data(), 2, 3));

        auto prepared = compiled.prepare_square(src);
        ASSERT_TRUE(prepared.has_value());
        static_assert(std::is_nothrow_move_constructible_v<cxxgpr::CheckedDistanceStore<T>>);
        static_assert(std::is_nothrow_move_assignable_v<cxxgpr::CheckedDistanceStore<T>>);

        // Move-construct out of the std::expected.
        cxxgpr::CheckedDistanceStore<T> moved(std::move(*prepared));
        EXPECT_EQ(moved.rows(), 3U);
        EXPECT_EQ(moved.cols(), 3U);
        EXPECT_TRUE(moved.is_square());

        // Move-assign over a store of a different shape, so the assertions below fail unless the
        // assignment really replaced the 2 x 3 cross tables with the 3 x 3 square ones.
        auto second = compiled.prepare_cross(cross_src);
        ASSERT_TRUE(second.has_value());
        cxxgpr::CheckedDistanceStore<T> target(std::move(*second));
        EXPECT_EQ(target.rows(), 2U);
        EXPECT_FALSE(target.is_square());
        target = std::move(moved);
        EXPECT_EQ(target.rows(), 3U);
        EXPECT_EQ(target.cols(), 3U);
        EXPECT_TRUE(target.is_square());
    }

    /// @brief Verifies a compiled distance kernel can be moved out of compile(), move-assigned, and then used.
    /// @details Move construction and assignment must transfer the compiled tree: the moved-to kernel reports
    /// the source hyperparameters, validates a distance source, and evaluates the same Gram matrix as an
    /// untouched kernel compiled from the same tree. This compiles only when the move constructor and
    /// assignment are defined where the implementation type is complete.
    /// @tparam T Storage scalar type of the compiled distance kernel.
    template <typename T> void expect_compiled_distance_kernel_is_movable()
    {
        using Compiled = cxxgpr::CompiledDistanceKernel<T>;
        static_assert(std::is_nothrow_move_constructible_v<Compiled>);
        static_assert(std::is_nothrow_move_assignable_v<Compiled>);

        const cxxgpr::ScalarDistance slot;
        const auto narrow = cxxgpr::RbfKernel::create(0.5);
        const auto wide = cxxgpr::RbfKernel::create(2.0);
        ASSERT_TRUE(narrow.has_value());
        ASSERT_TRUE(wide.has_value());

        std::array<double, 9> d2{0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0};
        cxxgpr::DistanceSource src;
        src.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(d2.data(), 3, 3));

        // Evaluates the 3 x 3 Gram matrix of a compiled kernel from the bound squared distances.
        const auto gram = [&src](const Compiled& kernel)
        {
            std::array<T, 9> k{};
            const auto store = kernel.prepare_square(src);
            EXPECT_TRUE(store.has_value());
            if (store.has_value())
            {
                std::array<T, 9> scratch{};
                EXPECT_TRUE(kernel
                        .apply(
                            *store, std::nullopt, cxxgpr::MatrixMut<T>(k.data(), 3, 3), cxxgpr::Triangle::Full,
                            cxxgpr::MatrixMut<T>(scratch.data(), 3, 3)
                        )
                        .has_value());
            }
            return k;
        };

        // Reference values come from kernels that are never moved.
        const Compiled narrow_ref = slot.rbf(*narrow).template compile<T>();
        const Compiled wide_ref = slot.rbf(*wide).template compile<T>();
        const auto k_narrow = gram(narrow_ref);
        const auto k_wide = gram(wide_ref);
        ASSERT_NE(k_narrow, k_wide);
        // Closed form k = exp(-d2 / (2 l^2)) with l = 0.5; values are at most 1, so a few ulps of T suffice.
        const T tol = T{4} * std::numeric_limits<T>::epsilon();
        for (std::size_t i = 0; i < d2.size(); ++i)
        {
            EXPECT_NEAR(k_narrow.at(i), static_cast<T>(std::exp(-2.0 * d2.at(i))), tol);
        }
        ASSERT_EQ(narrow_ref.num_params(), 1U);
        std::array<double, 1> narrow_params{};
        ASSERT_TRUE(narrow_ref.get_params(narrow_params).has_value());

        // Move-construct out of the compile() result.
        Compiled source = slot.rbf(*narrow).template compile<T>();
        Compiled moved(std::move(source));
        EXPECT_EQ(moved.num_params(), 1U);
        EXPECT_EQ(moved.num_leaves(), 1U);
        std::array<double, 1> params{};
        ASSERT_TRUE(moved.get_params(params).has_value());
        EXPECT_EQ(params, narrow_params);
        EXPECT_EQ(gram(moved), k_narrow);

        // Move-assign over a kernel with a different lengthscale; the target must take the source state.
        Compiled target = slot.rbf(*wide).template compile<T>();
        EXPECT_EQ(gram(target), k_wide);
        target = std::move(moved);
        ASSERT_TRUE(target.get_params(params).has_value());
        EXPECT_EQ(params, narrow_params);
        EXPECT_EQ(gram(target), k_narrow);

        // The moved-to kernel stays fully usable: parameter updates apply to it.
        std::array<double, 1> wide_params{};
        ASSERT_TRUE(wide_ref.get_params(wide_params).has_value());
        ASSERT_TRUE(target.set_params(wide_params).has_value());
        EXPECT_EQ(gram(target), k_wide);
    }

    /// @brief Verifies CheckedDistanceStore move construction and assignment for both storage scalars.
    /// @details Runs the move checks for every explicitly instantiated scalar type (double and float).
    TEST(DistanceKernelTest, CheckedDistanceStoreIsMovable)
    {
        // Spec: DES-10
        {
            SCOPED_TRACE("T = double");
            expect_checked_distance_store_is_movable<double>();
        }
        {
            SCOPED_TRACE("T = float");
            expect_checked_distance_store_is_movable<float>();
        }
    }

    /// @brief Verifies CompiledDistanceKernel move construction and assignment for both storage scalars.
    /// @details Runs the move checks for every explicitly instantiated scalar type (double and float).
    TEST(DistanceKernelTest, CompiledDistanceKernelIsMovable)
    {
        // Spec: DES-10
        {
            SCOPED_TRACE("T = double");
            expect_compiled_distance_kernel_is_movable<double>();
        }
        {
            SCOPED_TRACE("T = float");
            expect_compiled_distance_kernel_is_movable<float>();
        }
    }

    /// @brief Verifies that prepare_square() and prepare_cross() report every shape error before any value error.
    /// @details A supply whose first table holds a NaN and whose later table is empty, of another shape, or has the
    /// wrong number of ARD dimensions must report EmptyInput, LengthMismatch, or DimensionMismatch, not
    /// InvalidDistance; with every shape valid, the NaN is reported.
    TEST(DistanceKernelTest, PrepareReportsShapeErrorsBeforeValueErrors)
    {
        // Spec: DES-02, DES-10, DES-21
        const cxxgpr::ScalarDistance s1;
        const cxxgpr::ScalarDistance s2;
        const auto ard = cxxgpr::ArdDistance::create(2);
        const auto rbf = cxxgpr::RbfKernel::create(1.0);
        const auto rbf_ard = cxxgpr::RbfArdKernel::create(std::array{1.0, 2.0});
        ASSERT_TRUE(ard.has_value());
        ASSERT_TRUE(rbf.has_value());
        ASSERT_TRUE(rbf_ard.has_value());
        const auto ard_leaf = ard->rbf_ard(*rbf_ard);
        ASSERT_TRUE(ard_leaf.has_value());
        const auto compiled = (s1.rbf(*rbf) * s2.rbf(*rbf) * *ard_leaf).compile<double>();

        const std::array<double, 4> nan_2x2{0.0, std::numeric_limits<double>::quiet_NaN(), 1.0, 0.0};
        const std::array<double, 4> ok_2x2{0.0, 1.0, 1.0, 0.0};
        const std::array<double, 9> ok_3x3{0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0};
        const std::array<cxxgpr::MatrixRef<double>, 2> ard_ok{
            cxxgpr::MatrixRef<double>(ok_2x2.data(), 2, 2), cxxgpr::MatrixRef<double>(ok_2x2.data(), 2, 2)
        };
        const std::array<cxxgpr::MatrixRef<double>, 1> ard_short{cxxgpr::MatrixRef<double>(ok_2x2.data(), 2, 2)};
        const std::array<cxxgpr::MatrixRef<double>, 2> ard_empty{
            cxxgpr::MatrixRef<double>(ok_2x2.data(), 2, 2), cxxgpr::MatrixRef<double>()
        };

        const auto code_square = [&](const cxxgpr::DistanceSource& src)
        {
            const auto store = compiled.prepare_square(src);
            return store.has_value() ? std::optional<cxxgpr::GprErrorCode>{} : store.error().code();
        };
        const auto code_cross = [&](const cxxgpr::DistanceSource& src)
        {
            const auto store = compiled.prepare_cross(src);
            return store.has_value() ? std::optional<cxxgpr::GprErrorCode>{} : store.error().code();
        };

        cxxgpr::DistanceSource empty_later;
        empty_later.bind_scalar(s1.slot(), cxxgpr::MatrixRef<double>(nan_2x2.data(), 2, 2))
            .bind_scalar(s2.slot(), cxxgpr::MatrixRef<double>(ok_2x2.data(), 2, 2))
            .bind_ard(ard->slot(), ard_empty);
        EXPECT_EQ(code_square(empty_later), cxxgpr::GprErrorCode::EmptyInput);
        EXPECT_EQ(code_cross(empty_later), cxxgpr::GprErrorCode::EmptyInput);

        cxxgpr::DistanceSource other_shape;
        other_shape.bind_scalar(s1.slot(), cxxgpr::MatrixRef<double>(nan_2x2.data(), 2, 2))
            .bind_scalar(s2.slot(), cxxgpr::MatrixRef<double>(ok_3x3.data(), 3, 3))
            .bind_ard(ard->slot(), ard_ok);
        EXPECT_EQ(code_square(other_shape), cxxgpr::GprErrorCode::LengthMismatch);
        EXPECT_EQ(code_cross(other_shape), cxxgpr::GprErrorCode::LengthMismatch);

        cxxgpr::DistanceSource wrong_dims;
        wrong_dims.bind_scalar(s1.slot(), cxxgpr::MatrixRef<double>(nan_2x2.data(), 2, 2))
            .bind_scalar(s2.slot(), cxxgpr::MatrixRef<double>(ok_2x2.data(), 2, 2))
            .bind_ard(ard->slot(), ard_short);
        EXPECT_EQ(code_square(wrong_dims), cxxgpr::GprErrorCode::DimensionMismatch);
        EXPECT_EQ(code_cross(wrong_dims), cxxgpr::GprErrorCode::DimensionMismatch);

        cxxgpr::DistanceSource values_only;
        values_only.bind_scalar(s1.slot(), cxxgpr::MatrixRef<double>(nan_2x2.data(), 2, 2))
            .bind_scalar(s2.slot(), cxxgpr::MatrixRef<double>(ok_2x2.data(), 2, 2))
            .bind_ard(ard->slot(), ard_ok);
        EXPECT_EQ(code_square(values_only), cxxgpr::GprErrorCode::InvalidDistance);
        EXPECT_EQ(code_cross(values_only), cxxgpr::GprErrorCode::InvalidDistance);

        cxxgpr::DistanceSource valid;
        valid.bind_scalar(s1.slot(), cxxgpr::MatrixRef<double>(ok_2x2.data(), 2, 2))
            .bind_scalar(s2.slot(), cxxgpr::MatrixRef<double>(ok_2x2.data(), 2, 2))
            .bind_ard(ard->slot(), ard_ok);
        EXPECT_FALSE(code_square(valid).has_value());
        EXPECT_FALSE(code_cross(valid).has_value());
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

    /// @brief Verifies depth-128 alternating sum/product DistanceKernel eval, grad, hess, and teardown fit in small
    /// stack.
    TEST(DistanceKernelTest, DeepAlternatingTreeEvalGradHessFitInSmallStack)
    {
        // Spec: REQ-05
        const bool ran = cxxgpr::test_support::run_with_stack_size(
            cxxgpr::test_support::deep_model_stack_bytes(), []()
        {
            const cxxgpr::ScalarDistance slot;
            auto tree = slot.rbf(cxxgpr::RbfKernel::create(1.1).value()).with_points();
            for (int depth = 1; depth < 128; ++depth)
            {
                const auto leaf = slot.rbf(cxxgpr::RbfKernel::create(0.9).value()).with_points();
                tree = (depth % 2 == 0) ? (std::move(tree) + leaf) : (std::move(tree) * leaf);
            }
            const auto compiled = tree.compile<double>();
            auto compiled_copy = compiled;
            compiled_copy = compiled;
            EXPECT_EQ(compiled_copy.num_params(), 128U);
            constexpr std::size_t n = 3U;
            const std::array<double, 6> x_raw{0.0, 0.5, 1.0, 0.2, 0.4, 0.8};
            const cxxgpr::MatrixRef<double> x(x_raw.data(), n, 2U, n);
            std::vector<double> diag(n, 0.0);
            ASSERT_TRUE(compiled.fill_diag(n, x, diag).has_value());
            const std::array<double, 9> d2_raw{0.0, 0.29, 1.36, 0.29, 0.0, 0.41, 1.36, 0.41, 0.0};
            cxxgpr::DistanceSource src;
            src.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(d2_raw.data(), n, n, n));
            const auto store = compiled.prepare_square(src);
            ASSERT_TRUE(store.has_value());

            std::vector<double> out(n * n, 0.0);
            std::vector<double> sa(n * n, 0.0);
            std::vector<double> sb(n * n, 0.0);
            std::vector<double> sc(n * n, 0.0);
            auto nested = compiled.make_nested_buffers(n, n);
            ASSERT_TRUE(compiled
                    .apply(
                        *store, x, cxxgpr::MatrixMut<double>(out.data(), n, n, n), cxxgpr::Triangle::Lower,
                        cxxgpr::MatrixMut<double>(sa.data(), n, n, n), nested
                    )
                    .has_value());
            ASSERT_TRUE(compiled
                    .grad(
                        *store, x, cxxgpr::MatrixMut<double>(out.data(), n, n, n), 0U, cxxgpr::Triangle::Lower,
                        cxxgpr::MatrixMut<double>(sa.data(), n, n, n), cxxgpr::MatrixMut<double>(sb.data(), n, n, n),
                        nested
                    )
                    .has_value());
            ASSERT_TRUE(compiled
                    .hess(
                        *store, x, cxxgpr::MatrixMut<double>(out.data(), n, n, n), 0U, 1U, cxxgpr::Triangle::Lower,
                        cxxgpr::MatrixMut<double>(sa.data(), n, n, n), cxxgpr::MatrixMut<double>(sb.data(), n, n, n),
                        cxxgpr::MatrixMut<double>(sc.data(), n, n, n), nested
                    )
                    .has_value());
            const auto cross_store = compiled.prepare_cross(src);
            ASSERT_TRUE(cross_store.has_value());
            ASSERT_TRUE(compiled
                    .apply_cross(
                        *cross_store, x, x, cxxgpr::MatrixMut<double>(out.data(), n, n, n),
                        cxxgpr::MatrixMut<double>(sa.data(), n, n, n), nested
                    )
                    .has_value());
        }
        );
        ASSERT_TRUE(ran);
    }

    /// @brief Verifies ConstantKernel folding in DistanceKernel matches coordinate KernelSpec for apply, cross, grad,
    /// and hess.
    /// @details Evaluates pure-distance `Constant * Constant * Rbf` and mixed `Constant * Rbf * Matern + Linear`
    /// across Lower, Upper, Full, and rectangular cross paths and asserts zero heap allocations on warmed buffers.
    TEST(DistanceKernelTest, ConstantProductFoldingMatchesCoordinateSpec)
    {
        // Spec: REQ-05, DES-10
        constexpr std::size_t n = 4U;
        constexpr std::size_t m = 3U;
        constexpr std::size_t d = 2U;

        const std::vector<double> x{0.0, 0.4, 1.1, -0.5, 0.2, -0.8, 0.3, 0.9};
        const std::vector<double> q{0.1, 0.7, -0.3, -0.2, 0.5, 1.2};

        auto sq_dist = [](const std::vector<double>& a, std::size_t na, const std::vector<double>& b, std::size_t nb)
        {
            std::vector<double> out(na * nb, 0.0);
            for (std::size_t j = 0; j < nb; ++j)
            {
                for (std::size_t i = 0; i < na; ++i)
                {
                    double acc = 0.0;
                    for (std::size_t dim = 0; dim < d; ++dim)
                    {
                        const double diff = a[(dim * na) + i] - b[(dim * nb) + j];
                        acc += diff * diff;
                    }
                    out[(j * na) + i] = acc;
                }
            }
            return out;
        };

        const auto d2_train = sq_dist(x, n, x, n);
        const auto d2_cross = sq_dist(x, n, q, m);
        const cxxgpr::ScalarDistance slot;

        cxxgpr::DistanceSource train_src;
        train_src.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(d2_train.data(), n, n));
        cxxgpr::DistanceSource cross_src;
        cross_src.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(d2_cross.data(), n, m));

        const auto c1 = cxxgpr::ConstantKernel::create(2.3).value();
        const auto c2 = cxxgpr::ConstantKernel::create(0.75).value();
        const auto rbf = cxxgpr::RbfKernel::create(1.15).value();
        const auto m52 = cxxgpr::MaternKernel::create(0.85, cxxgpr::MaternNu::FiveHalves).value();
        const auto lin = cxxgpr::LinearKernel::create(0.45).value();

        // Case 1: DistanceOnly Constant * Constant * Rbf.
        {
            const auto dist_k =
                (cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(c1) * cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(c2))
                * slot.rbf(rbf);
            const auto coord_k = (cxxgpr::KernelSpec(c1) * cxxgpr::KernelSpec(c2)) * cxxgpr::KernelSpec(rbf);
            const auto dist_comp = dist_k.compile<double>();
            const auto coord_comp = coord_k.compile<double>();

            const auto sq_store = dist_comp.prepare_square(train_src);
            const auto cx_store = dist_comp.prepare_cross(cross_src);
            ASSERT_TRUE(sq_store.has_value()) << sq_store.error().message();
            ASSERT_TRUE(cx_store.has_value()) << cx_store.error().message();

            auto dist_nested = dist_comp.make_nested_buffers(n, n);
            auto coord_nested = coord_comp.make_nested_buffers(n, n);
            std::vector<double> k_dist(n * n);
            std::vector<double> k_coord(n * n);
            std::vector<double> sa(n * n);
            std::vector<double> sb(n * n);
            std::vector<double> sc(n * n);
            const cxxgpr::MatrixRef<double> x_ref(x.data(), n, d);
            const cxxgpr::MatrixRef<double> q_ref(q.data(), m, d);

            for (const auto uplo : {cxxgpr::Triangle::Lower, cxxgpr::Triangle::Upper, cxxgpr::Triangle::Full})
            {
                std::ranges::fill(k_dist, -999.0);
                std::ranges::fill(k_coord, -999.0);
                ASSERT_TRUE(dist_comp
                        .apply(
                            *sq_store, std::nullopt, cxxgpr::MatrixMut<double>(k_dist.data(), n, n), uplo,
                            cxxgpr::MatrixMut<double>(sa.data(), n, n), dist_nested
                        )
                        .has_value());
                ASSERT_TRUE(coord_comp
                        .apply_points(
                            x_ref, cxxgpr::MatrixMut<double>(k_coord.data(), n, n), uplo,
                            cxxgpr::MatrixMut<double>(sa.data(), n, n), coord_nested
                        )
                        .has_value());
                for (std::size_t col = 0; col < n; ++col)
                {
                    for (std::size_t row = 0; row < n; ++row)
                    {
                        const bool check = (uplo == cxxgpr::Triangle::Full)
                            || (uplo == cxxgpr::Triangle::Lower && row >= col)
                            || (uplo == cxxgpr::Triangle::Upper && row <= col);
                        if (check)
                        {
                            EXPECT_NEAR(k_dist[(col * n) + row], k_coord[(col * n) + row], 1e-13);
                        }
                    }
                }
            }

            std::vector<double> cx_dist(n * m);
            std::vector<double> cx_coord(n * m);
            std::vector<double> cx_s(n * m);
            ASSERT_TRUE(dist_comp
                    .apply_cross(
                        *cx_store, std::nullopt, std::nullopt, cxxgpr::MatrixMut<double>(cx_dist.data(), n, m),
                        cxxgpr::MatrixMut<double>(cx_s.data(), n, m), dist_nested
                    )
                    .has_value());
            ASSERT_TRUE(coord_comp
                    .apply_cross_points(
                        x_ref, q_ref, cxxgpr::MatrixMut<double>(cx_coord.data(), n, m),
                        cxxgpr::MatrixMut<double>(cx_s.data(), n, m), coord_nested
                    )
                    .has_value());
            for (std::size_t k = 0; k < cx_dist.size(); ++k)
            {
                EXPECT_NEAR(cx_dist[k], cx_coord[k], 1e-13);
            }

            const std::size_t np = dist_comp.num_params();
            for (std::size_t p = 0; p < np; ++p)
            {
                ASSERT_TRUE(dist_comp
                        .grad(
                            *sq_store, std::nullopt, cxxgpr::MatrixMut<double>(k_dist.data(), n, n), p,
                            cxxgpr::Triangle::Lower, cxxgpr::MatrixMut<double>(sa.data(), n, n),
                            cxxgpr::MatrixMut<double>(sb.data(), n, n), dist_nested
                        )
                        .has_value());
                ASSERT_TRUE(coord_comp
                        .grad_points(
                            x_ref, cxxgpr::MatrixMut<double>(k_coord.data(), n, n), p, cxxgpr::Triangle::Lower,
                            cxxgpr::MatrixMut<double>(sa.data(), n, n), cxxgpr::MatrixMut<double>(sb.data(), n, n),
                            coord_nested
                        )
                        .has_value());
                for (std::size_t col = 0; col < n; ++col)
                {
                    for (std::size_t row = col; row < n; ++row)
                    {
                        EXPECT_NEAR(k_dist[(col * n) + row], k_coord[(col * n) + row], 1e-13);
                    }
                }
                for (std::size_t j = 0; j < np; ++j)
                {
                    ASSERT_TRUE(dist_comp
                            .hess(
                                *sq_store, std::nullopt, cxxgpr::MatrixMut<double>(k_dist.data(), n, n), p, j,
                                cxxgpr::Triangle::Lower, cxxgpr::MatrixMut<double>(sa.data(), n, n),
                                cxxgpr::MatrixMut<double>(sb.data(), n, n), cxxgpr::MatrixMut<double>(sc.data(), n, n),
                                dist_nested
                            )
                            .has_value());
                    ASSERT_TRUE(coord_comp
                            .hess_points(
                                x_ref, cxxgpr::MatrixMut<double>(k_coord.data(), n, n), p, j, cxxgpr::Triangle::Lower,
                                cxxgpr::MatrixMut<double>(sa.data(), n, n), cxxgpr::MatrixMut<double>(sb.data(), n, n),
                                cxxgpr::MatrixMut<double>(sc.data(), n, n), coord_nested
                            )
                            .has_value());
                    for (std::size_t col = 0; col < n; ++col)
                    {
                        for (std::size_t row = col; row < n; ++row)
                        {
                            EXPECT_NEAR(k_dist[(col * n) + row], k_coord[(col * n) + row], 1e-13);
                        }
                    }
                }
            }

            const cxxgpr::test_support::NoAllocationScope audit;
            ASSERT_TRUE(dist_comp
                    .apply(
                        *sq_store, std::nullopt, cxxgpr::MatrixMut<double>(k_dist.data(), n, n), cxxgpr::Triangle::Full,
                        cxxgpr::MatrixMut<double>(sa.data(), n, n), dist_nested
                    )
                    .has_value());
            ASSERT_TRUE(dist_comp
                    .apply_cross(
                        *cx_store, std::nullopt, std::nullopt, cxxgpr::MatrixMut<double>(cx_dist.data(), n, m),
                        cxxgpr::MatrixMut<double>(cx_s.data(), n, m), dist_nested
                    )
                    .has_value());
            ASSERT_TRUE(dist_comp
                    .grad(
                        *sq_store, std::nullopt, cxxgpr::MatrixMut<double>(k_dist.data(), n, n), 0U,
                        cxxgpr::Triangle::Lower, cxxgpr::MatrixMut<double>(sa.data(), n, n),
                        cxxgpr::MatrixMut<double>(sb.data(), n, n), dist_nested
                    )
                    .has_value());
            ASSERT_TRUE(dist_comp
                    .hess(
                        *sq_store, std::nullopt, cxxgpr::MatrixMut<double>(k_dist.data(), n, n), 0U, 2U,
                        cxxgpr::Triangle::Lower, cxxgpr::MatrixMut<double>(sa.data(), n, n),
                        cxxgpr::MatrixMut<double>(sb.data(), n, n), cxxgpr::MatrixMut<double>(sc.data(), n, n),
                        dist_nested
                    )
                    .has_value());
            EXPECT_EQ(audit.delta_allocations(), 0U);
        }

        // Case 2: WithPoints (Constant * Rbf * Matern) + Linear.
        {
            const auto dist_k = ((cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(c1) * slot.rbf(rbf)) * slot.matern(m52))
                + cxxgpr::KernelSpec(lin);
            const auto coord_k = ((cxxgpr::KernelSpec(c1) * cxxgpr::KernelSpec(rbf)) * cxxgpr::KernelSpec(m52))
                + cxxgpr::KernelSpec(lin);
            const auto dist_comp = dist_k.compile<double>();
            const auto coord_comp = coord_k.compile<double>();

            const auto sq_store = dist_comp.prepare_square(train_src);
            const auto cx_store = dist_comp.prepare_cross(cross_src);
            ASSERT_TRUE(sq_store.has_value());
            ASSERT_TRUE(cx_store.has_value());

            auto dist_nested = dist_comp.make_nested_buffers(n, n);
            auto coord_nested = coord_comp.make_nested_buffers(n, n);
            std::vector<double> k_dist(n * n);
            std::vector<double> k_coord(n * n);
            std::vector<double> sa(n * n);
            std::vector<double> sb(n * n);
            std::vector<double> sc(n * n);
            const cxxgpr::MatrixRef<double> x_ref(x.data(), n, d);
            const cxxgpr::MatrixRef<double> q_ref(q.data(), m, d);

            ASSERT_TRUE(dist_comp
                    .apply(
                        *sq_store, x_ref, cxxgpr::MatrixMut<double>(k_dist.data(), n, n), cxxgpr::Triangle::Full,
                        cxxgpr::MatrixMut<double>(sa.data(), n, n), dist_nested
                    )
                    .has_value());
            ASSERT_TRUE(coord_comp
                    .apply_points(
                        x_ref, cxxgpr::MatrixMut<double>(k_coord.data(), n, n), cxxgpr::Triangle::Full,
                        cxxgpr::MatrixMut<double>(sa.data(), n, n), coord_nested
                    )
                    .has_value());
            for (std::size_t idx = 0; idx < k_dist.size(); ++idx)
            {
                EXPECT_NEAR(k_dist[idx], k_coord[idx], 1e-13);
            }

            std::vector<double> cx_dist(n * m);
            std::vector<double> cx_coord(n * m);
            std::vector<double> cx_s(n * m);
            ASSERT_TRUE(dist_comp
                    .apply_cross(
                        *cx_store, x_ref, q_ref, cxxgpr::MatrixMut<double>(cx_dist.data(), n, m),
                        cxxgpr::MatrixMut<double>(cx_s.data(), n, m), dist_nested
                    )
                    .has_value());
            ASSERT_TRUE(coord_comp
                    .apply_cross_points(
                        x_ref, q_ref, cxxgpr::MatrixMut<double>(cx_coord.data(), n, m),
                        cxxgpr::MatrixMut<double>(cx_s.data(), n, m), coord_nested
                    )
                    .has_value());
            for (std::size_t idx = 0; idx < cx_dist.size(); ++idx)
            {
                EXPECT_NEAR(cx_dist[idx], cx_coord[idx], 1e-13);
            }

            const std::size_t np = dist_comp.num_params();
            for (std::size_t p = 0; p < np; ++p)
            {
                ASSERT_TRUE(dist_comp
                        .grad(
                            *sq_store, x_ref, cxxgpr::MatrixMut<double>(k_dist.data(), n, n), p,
                            cxxgpr::Triangle::Upper, cxxgpr::MatrixMut<double>(sa.data(), n, n),
                            cxxgpr::MatrixMut<double>(sb.data(), n, n), dist_nested
                        )
                        .has_value());
                ASSERT_TRUE(coord_comp
                        .grad_points(
                            x_ref, cxxgpr::MatrixMut<double>(k_coord.data(), n, n), p, cxxgpr::Triangle::Upper,
                            cxxgpr::MatrixMut<double>(sa.data(), n, n), cxxgpr::MatrixMut<double>(sb.data(), n, n),
                            coord_nested
                        )
                        .has_value());
                for (std::size_t col = 0; col < n; ++col)
                {
                    for (std::size_t row = 0; row <= col; ++row)
                    {
                        EXPECT_NEAR(k_dist[(col * n) + row], k_coord[(col * n) + row], 1e-13);
                    }
                }
                for (std::size_t j = 0; j < np; ++j)
                {
                    ASSERT_TRUE(dist_comp
                            .hess(
                                *sq_store, x_ref, cxxgpr::MatrixMut<double>(k_dist.data(), n, n), p, j,
                                cxxgpr::Triangle::Lower, cxxgpr::MatrixMut<double>(sa.data(), n, n),
                                cxxgpr::MatrixMut<double>(sb.data(), n, n), cxxgpr::MatrixMut<double>(sc.data(), n, n),
                                dist_nested
                            )
                            .has_value());
                    ASSERT_TRUE(coord_comp
                            .hess_points(
                                x_ref, cxxgpr::MatrixMut<double>(k_coord.data(), n, n), p, j, cxxgpr::Triangle::Lower,
                                cxxgpr::MatrixMut<double>(sa.data(), n, n), cxxgpr::MatrixMut<double>(sb.data(), n, n),
                                cxxgpr::MatrixMut<double>(sc.data(), n, n), coord_nested
                            )
                            .has_value());
                    for (std::size_t col = 0; col < n; ++col)
                    {
                        for (std::size_t row = col; row < n; ++row)
                        {
                            EXPECT_NEAR(k_dist[(col * n) + row], k_coord[(col * n) + row], 1e-13);
                        }
                    }
                }
            }
        }
    }

    /// @brief Verifies parallel DistanceStoreBuilder preserves exact error priority and float-overflow checks.
    /// @details Tests strict and tidy modes on n = 140 square and rectangular blocks where non-finite or negative
    /// entries appear after earlier violations, plus float scalar range overflow during packing.
    TEST(DistanceKernelTest, LargeBlockErrorPriorityAndFloatOverflow)
    {
        // Spec: REQ-05, DES-10
        constexpr std::size_t n = 140U;
        const cxxgpr::ScalarDistance slot;
        const auto dk = slot.rbf(cxxgpr::RbfKernel::create(1.0).value());
        const auto comp64 = dk.compile<double>();
        const auto comp32 = dk.compile<float>();

        auto make_valid_sq = []()
        {
            std::vector<double> d2(n * n, 1.0);
            for (std::size_t i = 0; i < n; ++i)
            {
                d2[(i * n) + i] = 0.0;
            }
            return d2;
        };

        // 1. Late NaN beats earlier nonzero diagonal, negative value, and asymmetry in strict & tidy modes.
        {
            auto d2 = make_valid_sq();
            d2[0U] = 1e-3;             // nonzero diag at (0, 0)
            d2[(1U * n) + 2U] = 1.5;   // asymmetric with (2, 1)
            d2[(3U * n) + 5U] = -0.2;  // earlier negative
            d2[(138U * n) + 139U] = std::numeric_limits<double>::quiet_NaN();

            cxxgpr::DistanceSource strict_src;
            strict_src.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(d2.data(), n, n));
            const auto err_strict = comp64.prepare_square(strict_src);
            ASSERT_FALSE(err_strict.has_value());
            EXPECT_EQ(err_strict.error().code(), cxxgpr::GprErrorCode::InvalidDistance);
            EXPECT_EQ(err_strict.error().distance_reason(), std::optional<std::string_view>{"distance is not finite"});

            cxxgpr::DistanceSource tidy_src;
            tidy_src.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(d2.data(), n, n)).tidy(1e-6);
            const auto err_tidy = comp64.prepare_square(tidy_src);
            ASSERT_FALSE(err_tidy.has_value());
            EXPECT_EQ(err_tidy.error().code(), cxxgpr::GprErrorCode::InvalidDistance);
            EXPECT_EQ(err_tidy.error().distance_reason(), std::optional<std::string_view>{"distance is not finite"});
        }

        // 2. Strict & tidy error priorities across diagonal, negative, and off-diagonal asymmetry.
        {
            // Strict: same-pair negative beats asymmetry on (139, 138) vs (138, 139).
            auto d2_pair = make_valid_sq();
            d2_pair[(138U * n) + 139U] = -0.5;
            d2_pair[(139U * n) + 138U] = 2.0;
            cxxgpr::DistanceSource strict_pair_src;
            strict_pair_src.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(d2_pair.data(), n, n));
            const auto err_strict_pair = comp64.prepare_square(strict_pair_src);
            ASSERT_FALSE(err_strict_pair.has_value());
            EXPECT_EQ(err_strict_pair.error().code(), cxxgpr::GprErrorCode::InvalidDistance);
            EXPECT_EQ(
                err_strict_pair.error().distance_reason(), std::optional<std::string_view>{"distance is negative"}
            );

            // Tidy: late negative at (139, 138) beats earlier asymmetry at (2, 1).
            auto d2_tidy = make_valid_sq();
            d2_tidy[(1U * n) + 2U] = 2.0;
            d2_tidy[(138U * n) + 139U] = -0.5;
            cxxgpr::DistanceSource tidy_src;
            tidy_src.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(d2_tidy.data(), n, n)).tidy(1e-6);
            const auto err_tidy = comp64.prepare_square(tidy_src);
            ASSERT_FALSE(err_tidy.has_value());
            EXPECT_EQ(err_tidy.error().code(), cxxgpr::GprErrorCode::InvalidDistance);
            EXPECT_EQ(
                err_tidy.error().distance_reason(),
                std::optional<std::string_view>{"negative distance exceeds tidy tolerance"}
            );

            // Strict & tidy: late nonzero diagonal at (n - 1, n - 1) beats earlier asymmetry at (2, 1).
            auto d2_diag = make_valid_sq();
            d2_diag[(1U * n) + 2U] = 2.0;
            d2_diag[((n - 1U) * n) + (n - 1U)] = 0.5;
            cxxgpr::DistanceSource diag_strict_src;
            diag_strict_src.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(d2_diag.data(), n, n));
            const auto diag_strict = comp64.prepare_square(diag_strict_src);
            ASSERT_FALSE(diag_strict.has_value());
            EXPECT_EQ(diag_strict.error().code(), cxxgpr::GprErrorCode::InvalidDistance);
            EXPECT_EQ(diag_strict.error().distance_reason(), std::optional<std::string_view>{"diagonal must be zero"});

            cxxgpr::DistanceSource diag_tidy_src;
            diag_tidy_src.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(d2_diag.data(), n, n)).tidy(1e-6);
            const auto diag_tidy = comp64.prepare_square(diag_tidy_src);
            ASSERT_FALSE(diag_tidy.has_value());
            EXPECT_EQ(diag_tidy.error().code(), cxxgpr::GprErrorCode::InvalidDistance);
            EXPECT_EQ(
                diag_tidy.error().distance_reason(),
                std::optional<std::string_view>{"nonzero diagonal exceeds tidy tolerance"}
            );
        }

        // 3. Rectangular prepare_cross: late Inf beats earlier negative in strict & tidy modes.
        {
            std::vector<double> rect(n * n, 1.0);
            rect[(1U * n) + 0U] = -0.5;
            rect[(138U * n) + 139U] = std::numeric_limits<double>::infinity();

            cxxgpr::DistanceSource strict_cx;
            strict_cx.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(rect.data(), n, n));
            const auto err_strict = comp64.prepare_cross(strict_cx);
            ASSERT_FALSE(err_strict.has_value());
            EXPECT_EQ(err_strict.error().code(), cxxgpr::GprErrorCode::InvalidDistance);
            EXPECT_EQ(err_strict.error().distance_reason(), std::optional<std::string_view>{"distance is not finite"});

            cxxgpr::DistanceSource tidy_cx;
            tidy_cx.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(rect.data(), n, n)).tidy(1e-6);
            const auto err_tidy = comp64.prepare_cross(tidy_cx);
            ASSERT_FALSE(err_tidy.has_value());
            EXPECT_EQ(err_tidy.error().code(), cxxgpr::GprErrorCode::InvalidDistance);
            EXPECT_EQ(err_tidy.error().distance_reason(), std::optional<std::string_view>{"distance is not finite"});
        }

        // 4. Float storage overflow in large square and cross blocks.
        {
            auto d2 = make_valid_sq();
            d2[(138U * n) + 139U] = 1e40;
            d2[(139U * n) + 138U] = 1e40;
            cxxgpr::DistanceSource sq_src;
            sq_src.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(d2.data(), n, n));
            const auto sq_err = comp32.prepare_square(sq_src);
            ASSERT_FALSE(sq_err.has_value());
            EXPECT_EQ(sq_err.error().code(), cxxgpr::GprErrorCode::InvalidDistance);
            EXPECT_EQ(
                sq_err.error().distance_reason(),
                std::optional<std::string_view>{"value overflows storage scalar range"}
            );

            std::vector<double> rect(n * n, 1.0);
            rect[(138U * n) + 139U] = 1e40;
            cxxgpr::DistanceSource cx_src;
            cx_src.bind_scalar(slot.slot(), cxxgpr::MatrixRef<double>(rect.data(), n, n));
            const auto cx_err = comp32.prepare_cross(cx_src);
            ASSERT_FALSE(cx_err.has_value());
            EXPECT_EQ(cx_err.error().code(), cxxgpr::GprErrorCode::InvalidDistance);
            EXPECT_EQ(
                cx_err.error().distance_reason(),
                std::optional<std::string_view>{"value overflows storage scalar range"}
            );
        }
    }

    /// @brief Minimal user-defined custom kernel term k(d2) = scale * exp(-d2) for scalar-slot fallback tests.
    /// @tparam T Evaluation scalar type (`double` or `float`).
    template <cxxgpr::KernelScalarConcept T> class SimpleExpTerm final : public cxxgpr::KernelTerm<T>
    {
    public:
        /// @brief Constructs the custom term with output scale @p scale.
        /// @param[in] scale Positive output scale parameter.
        explicit SimpleExpTerm(double scale = 1.5) : scale_(scale) {}

        /// @brief Clones this custom term.
        /// @return Owning copy.
        [[nodiscard]] std::unique_ptr<cxxgpr::KernelTerm<T>> clone() const override
        {
            return std::make_unique<SimpleExpTerm<T>>(scale_);
        }

        /// @brief Returns parameter count (1).
        /// @return Hyperparameter count.
        [[nodiscard]] std::size_t num_params() const noexcept override { return 1U; }

        /// @brief Reads log(scale) into @p out.
        /// @param[out] out Parameter destination span.
        /// @return Empty expected on success, or LengthMismatch.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> get_params(std::span<double> out) const override
        {
            if (out.size() != 1U)
            {
                return std::unexpected(cxxgpr::GprError::length_mismatch("out"));
            }
            out[0] = std::log(scale_);
            return {};
        }

        /// @brief Writes log(scale) from @p params.
        /// @param[in] params Log-space parameter span.
        /// @return Empty expected on success, or GprError on invalid value.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> set_params(std::span<const double> params) override
        {
            if (params.size() != 1U)
            {
                return std::unexpected(cxxgpr::GprError::length_mismatch("params"));
            }
            if (!std::isfinite(params[0]))
            {
                return std::unexpected(cxxgpr::GprError::non_finite_input());
            }
            const double next = std::exp(params[0]);
            if (!std::isfinite(next) || next < 1e-4 || next > 1e4)
            {
                return std::unexpected(cxxgpr::GprError::invalid_config("scale out of bounds"));
            }
            scale_ = next;
            return {};
        }

        /// @brief Populates parameter bounds [1e-4, 1e4] into @p out.
        /// @param[out] out Interval destination span.
        /// @return Empty expected on success, or GprError.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> bounds_into(std::span<cxxgpr::Interval> out) const override
        {
            if (out.size() != 1U)
            {
                return std::unexpected(cxxgpr::GprError::length_mismatch("out"));
            }
            auto b = cxxgpr::Interval::create(1e-4, 1e4);
            if (!b.has_value())
            {
                return std::unexpected(std::move(b).error());
            }
            out[0] = *b;
            return {};
        }

        /// @brief Evaluates scale * exp(-d2) across @p uplo.
        /// @param[in] dist Pairwise squared distance view.
        /// @param[out] out Output kernel matrix.
        /// @param[in] uplo Triangle mask.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> apply(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> out, cxxgpr::Triangle uplo
        ) const override
        {
            for (std::size_t col = 0; col < dist.cols(); ++col)
            {
                for (std::size_t row = 0; row < dist.rows(); ++row)
                {
                    if ((uplo == cxxgpr::Triangle::Lower && row < col)
                        || (uplo == cxxgpr::Triangle::Upper && row > col))
                    {
                        continue;
                    }
                    const T d2 = dist(row, col) < T{0} ? T{0} : dist(row, col);
                    out(row, col) = static_cast<T>(scale_) * std::exp(-d2);
                }
            }
            return {};
        }

        /// @brief Evaluates rectangular cross-covariance.
        /// @param[in] dist Pairwise squared distance view.
        /// @param[out] out Output kernel matrix.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> apply_cross(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> out
        ) const override
        {
            return apply(dist, out, cxxgpr::Triangle::Full);
        }

        /// @brief Fills diagonal entries with scale.
        /// @param[out] out Destination span.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> fill_diag(std::span<T> out) const override
        {
            std::fill(out.begin(), out.end(), static_cast<T>(scale_));
            return {};
        }

        /// @brief Computes d/d(log scale) = k.
        /// @param[in] dist Pairwise squared distance view.
        /// @param[out] d_k Gradient output matrix.
        /// @param[in] param_idx Parameter index.
        /// @param[in] uplo Triangle mask.
        /// @return Empty expected on success, or IndexOutOfRange.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> grad(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> d_k, std::size_t param_idx, cxxgpr::Triangle uplo
        ) const override
        {
            if (param_idx != 0U)
            {
                return std::unexpected(cxxgpr::GprError::index_out_of_range("param_idx"));
            }
            return apply(dist, d_k, uplo);
        }

        /// @brief Computes d2/d(log scale)^2 = k.
        /// @param[in] dist Pairwise squared distance view.
        /// @param[out] d2_k Hessian output matrix.
        /// @param[in] i First parameter index.
        /// @param[in] j Second parameter index.
        /// @param[in] uplo Triangle mask.
        /// @return Empty expected on success, or IndexOutOfRange.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> hess(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> d2_k, std::size_t i, std::size_t j, cxxgpr::Triangle uplo
        ) const override
        {
            if (i != 0U || j != 0U)
            {
                return std::unexpected(cxxgpr::GprError::index_out_of_range("param_idx"));
            }
            return apply(dist, d2_k, uplo);
        }

    private:
        /// @brief Linear output scale.
        double scale_;
    };

    /// @brief Verifies float distance table bindings, assume_valid() contract, packed vs dense parity, and
    /// CheckedDistanceStore $O(1)$ copyability and move ownership guarantees.
    TEST(DistanceKernelTest, FloatBindingsAssumeValidPackedParityAndStoreOwnership)
    {
        // Spec: DES-10, REQ-05
        static_assert(std::is_copy_constructible_v<cxxgpr::CheckedDistanceStore<double>>);
        static_assert(std::is_copy_assignable_v<cxxgpr::CheckedDistanceStore<double>>);
        static_assert(std::is_copy_constructible_v<cxxgpr::CheckedDistanceStore<float>>);
        static_assert(std::is_copy_assignable_v<cxxgpr::CheckedDistanceStore<float>>);
        static_assert(std::is_move_constructible_v<cxxgpr::CheckedDistanceStore<double>>);
        static_assert(std::is_move_assignable_v<cxxgpr::CheckedDistanceStore<double>>);

        constexpr std::size_t n = 4U;
        constexpr std::size_t m = 3U;
        const std::vector<float> x1_f{0.0F, 0.4F, 0.9F, 1.4F, 0.2F, 0.7F, 0.1F, 1.1F};
        const std::vector<float> x2_f{0.1F, 0.8F, 1.3F, 0.3F, 0.5F, 1.0F};
        const std::vector<double> x1_d(x1_f.begin(), x1_f.end());
        const std::vector<double> x2_d(x2_f.begin(), x2_f.end());
        const cxxgpr::MatrixRef<double> x1_ref_d(x1_d.data(), n, 2U);
        const cxxgpr::MatrixRef<double> x2_ref_d(x2_d.data(), m, 2U);

        const std::vector<double> sq_d = make_sq_dist_rect(x1_ref_d, x1_ref_d);
        const std::vector<double> cross_d = make_sq_dist_rect(x1_ref_d, x2_ref_d);
        std::vector<float> sq_f(sq_d.size());
        std::vector<float> cross_f(cross_d.size());
        for (std::size_t idx = 0; idx < sq_d.size(); ++idx)
        {
            sq_f[idx] = static_cast<float>(sq_d[idx]);
        }
        for (std::size_t idx = 0; idx < cross_d.size(); ++idx)
        {
            cross_f[idx] = static_cast<float>(cross_d[idx]);
        }
        std::vector<double> sq_widened(sq_f.begin(), sq_f.end());
        std::vector<double> cross_widened(cross_f.begin(), cross_f.end());

        const auto ard0_d = make_dim_sq_diff_rect(x1_ref_d, x1_ref_d, 0U);
        const auto ard1_d = make_dim_sq_diff_rect(x1_ref_d, x1_ref_d, 1U);
        std::vector<float> ard0_f(ard0_d.size());
        std::vector<float> ard1_f(ard1_d.size());
        for (std::size_t idx = 0; idx < ard0_d.size(); ++idx)
        {
            ard0_f[idx] = static_cast<float>(ard0_d[idx]);
            ard1_f[idx] = static_cast<float>(ard1_d[idx]);
        }
        const std::vector<double> ard0_w(ard0_f.begin(), ard0_f.end());
        const std::vector<double> ard1_w(ard1_f.begin(), ard1_f.end());

        const cxxgpr::ScalarDistance s_slot;
        const auto a_slot = cxxgpr::ArdDistance::create(2U).value();
        const auto dk = (s_slot.rbf(cxxgpr::RbfKernel::create(0.8).value())
                            * s_slot.matern(cxxgpr::MaternKernel::create(1.1, cxxgpr::MaternNu::FiveHalves).value()))
            + a_slot.rbf_ard(cxxgpr::RbfArdKernel::create(std::array<double, 2>{0.7, 1.2}).value()).value();

        const auto comp64 = dk.compile<double>();
        const auto comp32 = dk.compile<float>();

        cxxgpr::DistanceSource src_f;
        EXPECT_FALSE(src_f.assumes_valid());
        const std::array<cxxgpr::MatrixRef<float>, 2> ard_refs_f{
            cxxgpr::MatrixRef<float>(ard0_f.data(), n, n), cxxgpr::MatrixRef<float>(ard1_f.data(), n, n)
        };
        src_f.bind_scalar(s_slot.slot(), cxxgpr::MatrixRef<float>(sq_f.data(), n, n))
            .bind_ard(a_slot.slot(), ard_refs_f);

        cxxgpr::DistanceSource src_w;
        const std::array<cxxgpr::MatrixRef<double>, 2> ard_refs_w{
            cxxgpr::MatrixRef<double>(ard0_w.data(), n, n), cxxgpr::MatrixRef<double>(ard1_w.data(), n, n)
        };
        src_w.bind_scalar(s_slot.slot(), cxxgpr::MatrixRef<double>(sq_widened.data(), n, n))
            .bind_ard(a_slot.slot(), ard_refs_w);

        auto store64_f = comp64.prepare_square(src_f);
        auto store64_w = comp64.prepare_square(src_w);
        auto store32_f = comp32.prepare_square(src_f);
        auto store32_w = comp32.prepare_square(src_w);
        ASSERT_TRUE(store64_f.has_value());
        ASSERT_TRUE(store64_w.has_value());
        ASSERT_TRUE(store32_f.has_value());
        ASSERT_TRUE(store32_w.has_value());

        std::vector<double> scratch64(n * n, 0.0);
        std::vector<double> scratch64_b(n * n, 0.0);
        std::vector<float> scratch32(n * n, 0.0F);
        std::vector<double> k64_f(n * n, 0.0);
        std::vector<double> k64_w(n * n, 0.0);
        ASSERT_TRUE(comp64
                .apply(
                    *store64_f, std::nullopt, cxxgpr::MatrixMut<double>(k64_f.data(), n, n), cxxgpr::Triangle::Full,
                    cxxgpr::MatrixMut<double>(scratch64.data(), n, n)
                )
                .has_value());
        ASSERT_TRUE(comp64
                .apply(
                    *store64_w, std::nullopt, cxxgpr::MatrixMut<double>(k64_w.data(), n, n), cxxgpr::Triangle::Full,
                    cxxgpr::MatrixMut<double>(scratch64.data(), n, n)
                )
                .has_value());
        EXPECT_EQ(k64_f, k64_w);

        std::vector<float> k32_f(n * n, 0.0F);
        std::vector<float> k32_w(n * n, 0.0F);
        ASSERT_TRUE(comp32
                .apply(
                    *store32_f, std::nullopt, cxxgpr::MatrixMut<float>(k32_f.data(), n, n), cxxgpr::Triangle::Full,
                    cxxgpr::MatrixMut<float>(scratch32.data(), n, n)
                )
                .has_value());
        ASSERT_TRUE(comp32
                .apply(
                    *store32_w, std::nullopt, cxxgpr::MatrixMut<float>(k32_w.data(), n, n), cxxgpr::Triangle::Full,
                    cxxgpr::MatrixMut<float>(scratch32.data(), n, n)
                )
                .has_value());
        EXPECT_EQ(k32_f, k32_w);

        // Strict float rejection on negative entry and double > float::max overflow in compile<float>().
        std::vector<float> bad_sq_f = sq_f;
        bad_sq_f[1] = -0.25F;
        cxxgpr::DistanceSource bad_f_src;
        bad_f_src.bind_scalar(s_slot.slot(), cxxgpr::MatrixRef<float>(bad_sq_f.data(), n, n))
            .bind_ard(a_slot.slot(), ard_refs_f);
        EXPECT_EQ(comp64.prepare_square(bad_f_src).error().code(), cxxgpr::GprErrorCode::InvalidDistance);

        std::vector<double> huge_sq = sq_widened;
        huge_sq[(1U * n) + 0U] = 1e300;
        huge_sq[(0U * n) + 1U] = 1e300;
        cxxgpr::DistanceSource huge_src;
        huge_src.bind_scalar(s_slot.slot(), cxxgpr::MatrixRef<double>(huge_sq.data(), n, n))
            .bind_ard(a_slot.slot(), ard_refs_w);
        EXPECT_EQ(comp32.prepare_square(huge_src).error().code(), cxxgpr::GprErrorCode::InvalidDistance);

        // assume_valid(true) skips value checks on bad values but still enforces structural checks and forbids tidy().
        cxxgpr::DistanceSource av_src;
        av_src.bind_scalar(s_slot.slot(), cxxgpr::MatrixRef<float>(bad_sq_f.data(), n, n))
            .bind_ard(a_slot.slot(), ard_refs_f)
            .assume_valid(true);
        EXPECT_TRUE(av_src.assumes_valid());
        EXPECT_TRUE(comp64.prepare_square(av_src).has_value());

        cxxgpr::DistanceSource av_tidy = src_w;
        av_tidy.assume_valid(true).tidy(1e-9);
        EXPECT_EQ(comp64.prepare_square(av_tidy).error().code(), cxxgpr::GprErrorCode::InvalidConfig);

        cxxgpr::DistanceSource av_missing;
        av_missing.bind_scalar(s_slot.slot(), cxxgpr::MatrixRef<double>(sq_widened.data(), n, n)).assume_valid(true);
        EXPECT_EQ(comp64.prepare_square(av_missing).error().code(), cxxgpr::GprErrorCode::DistanceSlot);

        cxxgpr::DistanceSource av_rect;
        av_rect.bind_scalar(s_slot.slot(), cxxgpr::MatrixRef<double>(cross_widened.data(), n, m))
            .bind_ard(a_slot.slot(), ard_refs_w)
            .assume_valid(true);
        EXPECT_EQ(comp64.prepare_square(av_rect).error().code(), cxxgpr::GprErrorCode::LengthMismatch);

        // Packed scalar storage matches dense coordinate evaluation for built-in leaves, and CustomKernel falls back
        // to dense storage while rejecting an incompatible packed store.
        const cxxgpr::ScalarDistance leaf_slot;
        const auto all_builtin =
            (leaf_slot.rbf(cxxgpr::RbfKernel::create(0.8).value())
                + leaf_slot.matern(cxxgpr::MaternKernel::create(0.9, cxxgpr::MaternNu::ThreeHalves).value()))
            + (leaf_slot.periodic(cxxgpr::PeriodicKernel::create(1.1, 2.0).value())
                + leaf_slot.rational_quadratic(cxxgpr::RationalQuadraticKernel::create(0.7, 1.5).value()));
        const auto builtin_comp = all_builtin.compile<double>();
        cxxgpr::DistanceSource leaf_src;
        leaf_src.bind_scalar(leaf_slot.slot(), cxxgpr::MatrixRef<double>(sq_d.data(), n, n));
        auto builtin_store = builtin_comp.prepare_square(leaf_src);
        ASSERT_TRUE(builtin_store.has_value());
        std::vector<double> k_builtin(n * n, 0.0);
        ASSERT_TRUE(builtin_comp
                .apply(
                    *builtin_store, std::nullopt, cxxgpr::MatrixMut<double>(k_builtin.data(), n, n),
                    cxxgpr::Triangle::Full, cxxgpr::MatrixMut<double>(scratch64.data(), n, n)
                )
                .has_value());

        const auto coord_equiv =
            (cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.8).value())
                + cxxgpr::KernelSpec(cxxgpr::MaternKernel::create(0.9, cxxgpr::MaternNu::ThreeHalves).value()))
            + (cxxgpr::KernelSpec(cxxgpr::PeriodicKernel::create(1.1, 2.0).value())
                + cxxgpr::KernelSpec(cxxgpr::RationalQuadraticKernel::create(0.7, 1.5).value()));
        const auto coord_comp = coord_equiv.compile<double>();
        auto coord_nested = coord_comp.make_nested_buffers(n, n);
        std::vector<double> k_coord(n * n, 0.0);
        ASSERT_TRUE(coord_comp
                .apply_points(
                    x1_ref_d, cxxgpr::MatrixMut<double>(k_coord.data(), n, n), cxxgpr::Triangle::Full,
                    cxxgpr::MatrixMut<double>(scratch64.data(), n, n), coord_nested
                )
                .has_value());
        for (std::size_t idx = 0; idx < k_builtin.size(); ++idx)
        {
            EXPECT_NEAR(k_builtin[idx], k_coord[idx], 1e-12);
        }

        // CustomKernel on the same slot builds dense storage and works end-to-end; passing builtin_store (packed)
        // to custom_comp is rejected with UnsupportedKernelOperation.
        auto ck = cxxgpr::CustomKernel::create(
            std::make_unique<SimpleExpTerm<double>>(1.5), std::make_unique<SimpleExpTerm<float>>(1.5F)
        );
        ASSERT_TRUE(ck.has_value());
        const auto custom_dk = leaf_slot.custom(*ck);
        const auto custom_comp = custom_dk.compile<double>();
        auto custom_store = custom_comp.prepare_square(leaf_src);
        ASSERT_TRUE(custom_store.has_value());
        std::vector<double> k_custom(n * n, 0.0);
        ASSERT_TRUE(custom_comp
                .apply(
                    *custom_store, std::nullopt, cxxgpr::MatrixMut<double>(k_custom.data(), n, n),
                    cxxgpr::Triangle::Full, cxxgpr::MatrixMut<double>(scratch64.data(), n, n)
                )
                .has_value());
        for (std::size_t idx = 0; idx < n * n; ++idx)
        {
            EXPECT_NEAR(k_custom[idx], 1.5 * std::exp(-sq_d[idx]), 1e-13);
        }
        EXPECT_EQ(
            custom_comp
                .apply(
                    *builtin_store, std::nullopt, cxxgpr::MatrixMut<double>(k_custom.data(), n, n),
                    cxxgpr::Triangle::Full, cxxgpr::MatrixMut<double>(scratch64.data(), n, n)
                )
                .error()
                .code(),
            cxxgpr::GprErrorCode::UnsupportedKernelOperation
        );

        // Moved-from CheckedDistanceStore returns EmptyInput on apply/grad/hess/apply_cross.
        cxxgpr::CheckedDistanceStore<double> moved_to = std::move(*custom_store);
        EXPECT_EQ(
            custom_comp
                .apply(
                    *custom_store, std::nullopt, cxxgpr::MatrixMut<double>(k_custom.data(), n, n),
                    cxxgpr::Triangle::Full, cxxgpr::MatrixMut<double>(scratch64.data(), n, n)
                )
                .error()
                .code(),
            cxxgpr::GprErrorCode::EmptyInput
        );
        EXPECT_EQ(
            custom_comp
                .grad(
                    *custom_store, std::nullopt, cxxgpr::MatrixMut<double>(k_custom.data(), n, n), 0U,
                    cxxgpr::Triangle::Full, cxxgpr::MatrixMut<double>(scratch64.data(), n, n),
                    cxxgpr::MatrixMut<double>(scratch64_b.data(), n, n)
                )
                .error()
                .code(),
            cxxgpr::GprErrorCode::EmptyInput
        );
        EXPECT_EQ(
            custom_comp
                .apply_cross(
                    *custom_store, std::nullopt, std::nullopt, cxxgpr::MatrixMut<double>(k_custom.data(), n, n),
                    cxxgpr::MatrixMut<double>(scratch64.data(), n, n)
                )
                .error()
                .code(),
            cxxgpr::GprErrorCode::EmptyInput
        );
        EXPECT_TRUE(custom_comp
                .apply(
                    moved_to, std::nullopt, cxxgpr::MatrixMut<double>(k_custom.data(), n, n), cxxgpr::Triangle::Full,
                    cxxgpr::MatrixMut<double>(scratch64.data(), n, n)
                )
                .has_value());

        // Foreign store built with another slot returns UnsupportedKernelOperation.
        const cxxgpr::ScalarDistance foreign_slot;
        const auto foreign_comp = foreign_slot.rbf(cxxgpr::RbfKernel::create(0.8).value()).compile<double>();
        cxxgpr::DistanceSource foreign_src;
        foreign_src.bind_scalar(foreign_slot.slot(), cxxgpr::MatrixRef<double>(sq_d.data(), n, n));
        auto foreign_store = foreign_comp.prepare_square(foreign_src);
        ASSERT_TRUE(foreign_store.has_value());
        EXPECT_EQ(
            builtin_comp
                .apply(
                    *foreign_store, std::nullopt, cxxgpr::MatrixMut<double>(k_builtin.data(), n, n),
                    cxxgpr::Triangle::Full, cxxgpr::MatrixMut<double>(scratch64.data(), n, n)
                )
                .error()
                .code(),
            cxxgpr::GprErrorCode::UnsupportedKernelOperation
        );

        // O(1) copy construction and assignment share validated tables cleanly with independent lifetimes.
        cxxgpr::CheckedDistanceStore<double> copied = moved_to;
        cxxgpr::CheckedDistanceStore<double> assigned = *builtin_store;
        assigned = copied;
        moved_to = *builtin_store;
        copied = assigned;
        std::vector<double> k_copied(n * n, 0.0);
        ASSERT_TRUE(custom_comp
                .apply(
                    copied, std::nullopt, cxxgpr::MatrixMut<double>(k_copied.data(), n, n), cxxgpr::Triangle::Full,
                    cxxgpr::MatrixMut<double>(scratch64.data(), n, n)
                )
                .has_value());
        EXPECT_EQ(k_copied, k_custom);
        std::vector<double> k_assigned(n * n, 0.0);
        ASSERT_TRUE(custom_comp
                .apply(
                    assigned, std::nullopt, cxxgpr::MatrixMut<double>(k_assigned.data(), n, n), cxxgpr::Triangle::Full,
                    cxxgpr::MatrixMut<double>(scratch64.data(), n, n)
                )
                .has_value());
        EXPECT_EQ(k_assigned, k_custom);

        // Null-data MatrixRef with non-zero dims must return EmptyInput in both strict and assume_valid modes.
        for (const bool assume : {false, true})
        {
            cxxgpr::DistanceSource null_scalar_d;
            null_scalar_d.bind_scalar(s_slot.slot(), cxxgpr::MatrixRef<double>(nullptr, n, n))
                .bind_ard(a_slot.slot(), ard_refs_w)
                .assume_valid(assume);
            EXPECT_EQ(comp64.prepare_square(null_scalar_d).error().code(), cxxgpr::GprErrorCode::EmptyInput);

            const std::array<cxxgpr::MatrixRef<float>, 2U> ard_cross_null_refs_f{
                cxxgpr::MatrixRef<float>(ard0_f.data(), n, m), cxxgpr::MatrixRef<float>(ard1_f.data(), n, m)
            };
            cxxgpr::DistanceSource null_scalar_f;
            null_scalar_f.bind_scalar(s_slot.slot(), cxxgpr::MatrixRef<float>(nullptr, n, m))
                .bind_ard(a_slot.slot(), ard_cross_null_refs_f)
                .assume_valid(assume);
            EXPECT_EQ(comp32.prepare_cross(null_scalar_f).error().code(), cxxgpr::GprErrorCode::EmptyInput);

            const std::array<cxxgpr::MatrixRef<double>, 2U> null_ard_d{
                cxxgpr::MatrixRef<double>(ard0_w.data(), n, n), cxxgpr::MatrixRef<double>(nullptr, n, n)
            };
            cxxgpr::DistanceSource null_ard_src;
            null_ard_src.bind_scalar(s_slot.slot(), cxxgpr::MatrixRef<double>(sq_widened.data(), n, n))
                .bind_ard(a_slot.slot(), null_ard_d)
                .assume_valid(assume);
            EXPECT_EQ(comp64.prepare_square(null_ard_src).error().code(), cxxgpr::GprErrorCode::EmptyInput);
        }

        // Large tidy_tol * max_abs overflowing to +Inf must be rejected with InvalidDistance.
        std::vector<double> big_d2(n * n, 1e200);
        for (std::size_t idx = 0; idx < n; ++idx)
        {
            big_d2[(idx * n) + idx] = 0.0;
        }
        cxxgpr::DistanceSource overflow_tidy;
        overflow_tidy.bind_scalar(leaf_slot.slot(), cxxgpr::MatrixRef<double>(big_d2.data(), n, n)).tidy(1e200);
        const auto overflow_err = builtin_comp.prepare_square(overflow_tidy);
        ASSERT_FALSE(overflow_err.has_value());
        EXPECT_EQ(overflow_err.error().code(), cxxgpr::GprErrorCode::InvalidDistance);
        EXPECT_EQ(overflow_err.error().distance_reason(), std::optional<std::string_view>{"tidy tolerance overflow"});

        std::vector<double> big_cross(n * m, 1e200);
        cxxgpr::DistanceSource overflow_cross;
        overflow_cross.bind_scalar(leaf_slot.slot(), cxxgpr::MatrixRef<double>(big_cross.data(), n, m)).tidy(1e200);
        const auto overflow_cross_err = builtin_comp.prepare_cross(overflow_cross);
        ASSERT_FALSE(overflow_cross_err.has_value());
        EXPECT_EQ(overflow_cross_err.error().code(), cxxgpr::GprErrorCode::InvalidDistance);
        EXPECT_EQ(
            overflow_cross_err.error().distance_reason(), std::optional<std::string_view>{"tidy tolerance overflow"}
        );
    }

}  // namespace
