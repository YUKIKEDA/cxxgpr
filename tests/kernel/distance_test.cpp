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
        // Spec: REQ-05, REQ-12
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
            constexpr std::size_t n = 3U;
            const std::array<double, 6> x_raw{0.0, 0.5, 1.0, 0.2, 0.4, 0.8};
            const cxxgpr::MatrixRef<double> x(x_raw.data(), n, 2U, n);
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

}  // namespace
