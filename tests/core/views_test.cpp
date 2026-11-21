// SPDX-License-Identifier: Apache-2.0
// Spec: REQ-03, REQ-11, DES-01, DES-02, DES-08, DES-16
/// @file views_test.cpp
/// @brief Unit tests for column-major matrix views and prediction result container defaults.
/// @details Verifies MatrixRef and MatrixMut column-major indexing, strided subviews, out-of-bounds
/// and zero-column safety, the checked from_span() factories (pointer binding, every shape error, and
/// the validation order), Prediction/PredictiveCovariance/Triangle defaults, and zero-allocation
/// element traversal.

#include "cxxgpr/core/views.hpp"

#include "cxxgpr/core/error.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <expected>
#include <limits>
#include <span>
#include <string>
#include <type_traits>

#include "allocation_audit.hpp"

namespace
{

    /// @brief Verifies column-major element indexing, strided subviews, and empty/oob column guards.
    /// @details Confirms writes through a strided MatrixMut subview alias the backing buffer, checks
    /// MatrixRef conversion and col() spans, and verifies zero-column or out-of-range col() returns empty.
    TEST(ViewsTest, ColumnMajorIndexingAndStridedSubmatrixWorkCorrectly)
    {
        // Packed 3x2 column-major matrix:
        // col 0: [1, 2, 3], col 1: [4, 5, 6]
        std::array<double, 6> storage{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        const cxxgpr::MatrixMut<double> mat(storage.data(), 3U, 2U);

        EXPECT_FALSE(mat.is_empty());
        EXPECT_FALSE(mat.is_square());
        EXPECT_TRUE(mat.is_contiguous());
        EXPECT_EQ(mat.rows(), 3U);
        EXPECT_EQ(mat.cols(), 2U);
        EXPECT_EQ(mat.col_stride(), 3U);
        EXPECT_DOUBLE_EQ(mat(0, 0), 1.0);
        EXPECT_DOUBLE_EQ(mat(2, 0), 3.0);
        EXPECT_DOUBLE_EQ(mat(1, 1), 5.0);

        // Strided 2x2 bottom-row subview starting at row 1 with col_stride = 3.
        cxxgpr::MatrixMut<double> sub(&storage[1], 2U, 2U, 3U);
        EXPECT_TRUE(sub.is_square());
        EXPECT_FALSE(sub.is_contiguous());
        EXPECT_DOUBLE_EQ(sub(0, 0), 2.0);
        EXPECT_DOUBLE_EQ(sub(1, 0), 3.0);
        EXPECT_DOUBLE_EQ(sub(0, 1), 5.0);
        EXPECT_DOUBLE_EQ(sub(1, 1), 6.0);

        // Mutate via subview and confirm the backing storage updates at offset (1 + 3*1 + 1 = 5).
        sub(1, 1) = 42.0;
        EXPECT_DOUBLE_EQ(storage[5], 42.0);

        const cxxgpr::MatrixRef<double> ref = sub;
        EXPECT_DOUBLE_EQ(ref(1, 1), 42.0);
        const auto col1 = ref.col(1);
        ASSERT_EQ(col1.size(), 2U);
        EXPECT_DOUBLE_EQ(col1[0], 5.0);
        EXPECT_DOUBLE_EQ(col1[1], 42.0);

        const cxxgpr::MatrixRef<double> empty_view;
        EXPECT_TRUE(empty_view.is_empty());
        EXPECT_TRUE(empty_view.col(0).empty());

        // Non-null data with zero columns or out-of-range col_idx must return an empty span safely.
        const cxxgpr::MatrixRef<double> zero_cols_ref(storage.data(), 3U, 0U);
        EXPECT_TRUE(zero_cols_ref.is_empty());
        EXPECT_TRUE(zero_cols_ref.col(0).empty());

        const cxxgpr::MatrixMut<double> zero_cols_mut(storage.data(), 3U, 0U);
        EXPECT_TRUE(zero_cols_mut.is_empty());
        EXPECT_TRUE(zero_cols_mut.col(0).empty());

        EXPECT_TRUE(mat.as_ref().col(2).empty());
        EXPECT_TRUE(mat.as_ref().col(10).empty());
        EXPECT_TRUE(mat.col(2).empty());
        EXPECT_TRUE(mat.col(10).empty());
    }

    /// @brief Verifies PredictOptions, Triangle aliasing, Prediction<T>, and PredictiveCovariance<T> defaults.
    /// @details Confirms Observation variance is the default across options and result structs and that
    /// cxxgpr::kernel::Triangle aliases cxxgpr::Triangle.
    TEST(ViewsTest, PredictionTypesAndClosedEnumsUseSpecifiedDefaults)
    {
        // Confirm Observation is the default VarianceKind and Triangle aliases match across namespaces.
        EXPECT_EQ(cxxgpr::PredictOptions{}.variance_kind, cxxgpr::VarianceKind::Observation);
        EXPECT_NE(cxxgpr::Triangle::Lower, cxxgpr::Triangle::Upper);
        EXPECT_EQ(cxxgpr::kernel::Triangle::Full, cxxgpr::Triangle::Full);

        cxxgpr::Prediction<double> pred_d;
        EXPECT_EQ(pred_d.variance_kind, cxxgpr::VarianceKind::Observation);
        pred_d.mean = {1.0, 2.0};
        pred_d.variance = {0.1, 0.2};
        EXPECT_EQ(pred_d.mean.size(), 2U);

        cxxgpr::Prediction<float> pred_f;
        pred_f.variance_kind = cxxgpr::VarianceKind::Latent;
        EXPECT_EQ(pred_f.variance_kind, cxxgpr::VarianceKind::Latent);

        cxxgpr::PredictiveCovariance<double> cov;
        EXPECT_EQ(cov.n_points, 0U);
        EXPECT_EQ(cov.variance_kind, cxxgpr::VarianceKind::Observation);
    }

    /// @brief Verifies MatrixMut mutation, MatrixRef conversion, and col() iteration allocate zero heap memory.
    /// @details Traverses a 3x3 matrix inside AuditMode::CppAndEigen NoAllocationScope and checks the
    /// tracked allocation delta is zero and the accumulated sum equals 32.0.
    TEST(ViewsTest, ViewAccessAndColumnSlicingAllocateZero)
    {
        std::array<double, 9> buf{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        double acc = 0.0;
        std::size_t delta = 1;
        // Original sum is 45; changing element (2, 1) from 6 to -7 reduces sum by 13 to 32.
        {
            cxxgpr::test_support::NoAllocationScope scope(cxxgpr::test_support::AuditMode::CppAndEigen);
            cxxgpr::MatrixMut<double> mut(buf.data(), 3U, 3U);
            mut(2, 1) = -7.0;
            const cxxgpr::MatrixRef<double> ref = mut.as_ref();
            for (std::size_t col = 0; col < ref.cols(); ++col)
            {
                for (const double v : ref.col(col))
                {
                    acc += v;
                }
            }
            cxxgpr::test_support::touch_for_test(&acc);
            delta = scope.delta_allocations();
        }
        EXPECT_EQ(delta, 0U);
        EXPECT_DOUBLE_EQ(acc, 32.0);
    }

    /// @brief Maps a view type to its scalar type and the span type its from_span() factory accepts.
    /// @tparam View MatrixRef<T> or MatrixMut<T>.
    template <typename View> struct ViewTraits;

    /// @brief ViewTraits specialization for read-only views, whose factory takes std::span<const T>.
    /// @tparam T Matrix element scalar type.
    template <typename T> struct ViewTraits<cxxgpr::MatrixRef<T>>
    {
        /// @brief Matrix element scalar type.
        using Scalar = T;
        /// @brief Span type accepted by MatrixRef<T>::from_span().
        using Span = std::span<const T>;
    };

    /// @brief ViewTraits specialization for mutable views, whose factory takes std::span<T>.
    /// @tparam T Matrix element scalar type.
    template <typename T> struct ViewTraits<cxxgpr::MatrixMut<T>>
    {
        /// @brief Matrix element scalar type.
        using Scalar = T;
        /// @brief Span type accepted by MatrixMut<T>::from_span().
        using Span = std::span<T>;
    };

    /// @brief Backing buffer for the from_span() cases of one view type.
    /// @tparam View MatrixRef<T> or MatrixMut<T>.
    template <typename View> struct SpanBuffer
    {
        /// @brief Matrix element scalar type of View.
        using Scalar = typename ViewTraits<View>::Scalar;
        /// @brief Span type accepted by View::from_span().
        using Span = typename ViewTraits<View>::Span;

        /// @brief Values 1, 2, ..., 8, more than every shape used in the cases needs.
        std::array<Scalar, 8> values{
            Scalar{1}, Scalar{2}, Scalar{3}, Scalar{4}, Scalar{5}, Scalar{6}, Scalar{7}, Scalar{8}
        };

        /// @brief Returns a span over values starting at offset with the given length.
        /// @param[in] offset Index of the first element of the span inside values.
        /// @param[in] length Number of elements in the span.
        /// @return Span of View's accepted type over values[offset, offset + length).
        [[nodiscard]] Span span_at(std::size_t offset, std::size_t length) noexcept
        {
            return Span(values.data() + offset, length);
        }
    };

    /// @brief Expects that from_span() fails with the given error code.
    /// @tparam View MatrixRef<T> or MatrixMut<T>.
    /// @param[in] result Result of View::from_span().
    /// @param[in] code Expected error code.
    /// @param[in] label Case label printed on failure.
    template <typename View>
    void expect_from_span_error(
        const std::expected<View, cxxgpr::GprError>& result, cxxgpr::GprErrorCode code, const std::string& label
    )
    {
        if (result.has_value())
        {
            ADD_FAILURE() << label << ": from_span() accepted an invalid shape";
            return;
        }
        EXPECT_EQ(result.error().code(), code) << label << ": " << result.error().message();
    }

    /// @brief Checks that View::from_span() binds span.data(), the extents, and col_stride == rows exactly.
    /// @details Uses a span that starts at offset 1 of the buffer so a factory that bound another pointer, or
    /// a stride other than rows, reads different elements. Also covers the 1 x 1 boundary and, for MatrixMut,
    /// that writes through the view land in the span's memory.
    /// @tparam View MatrixRef<T> or MatrixMut<T>.
    /// @param[in] view_name Name of View printed on failure.
    template <typename View> void check_from_span_binds_span(const char* view_name)
    {
        SCOPED_TRACE(view_name);
        using Scalar = typename SpanBuffer<View>::Scalar;
        SpanBuffer<View> buf;

        const auto view = View::from_span(buf.span_at(1U, 6U), 3U, 2U);
        if (!view.has_value())
        {
            FAIL() << "from_span(6 values, 3 x 2) failed: " << view.error().message();
        }
        EXPECT_EQ(view->data(), buf.values.data() + 1);
        EXPECT_EQ(view->rows(), 3U);
        EXPECT_EQ(view->cols(), 2U);
        EXPECT_EQ(view->col_stride(), 3U);
        EXPECT_TRUE(view->is_contiguous());
        for (std::size_t col = 0; col < 2U; ++col)
        {
            for (std::size_t row = 0; row < 3U; ++row)
            {
                // Element (row, col) of the view is values[1 + col * 3 + row], i.e. the value 2 + col * 3 + row.
                EXPECT_EQ((*view)(row, col), static_cast<Scalar>(2U + (col * 3U) + row)) << row << ", " << col;
            }
        }
        EXPECT_EQ(view->col(1).data(), buf.values.data() + 4);
        EXPECT_EQ(view->col(1).size(), 3U);

        if constexpr (std::is_same_v<View, cxxgpr::MatrixMut<Scalar>>)
        {
            (*view)(2, 1) = Scalar{-1};
            EXPECT_EQ(buf.values[6], Scalar{-1});
        }

        const auto single = View::from_span(buf.span_at(7U, 1U), 1U, 1U);
        if (!single.has_value())
        {
            FAIL() << "from_span(1 value, 1 x 1) failed: " << single.error().message();
        }
        EXPECT_EQ(single->data(), buf.values.data() + 7);
        EXPECT_EQ((*single)(0, 0), Scalar{8});
    }

    /// @brief Checks that View::from_span() reports every shape violation with its own error code.
    /// @details One minimal input per condition: a null empty span, a non-null empty span, a zero row count,
    /// a zero column count, both counts zero, a product overflowing in either factor order, a span one element
    /// short, and a span one element long.
    /// @tparam View MatrixRef<T> or MatrixMut<T>.
    /// @param[in] view_name Name of View printed on failure.
    template <typename View> void check_from_span_rejects_each_violation(const char* view_name)
    {
        SCOPED_TRACE(view_name);
        using Span = typename SpanBuffer<View>::Span;
        constexpr std::size_t max_sz = std::numeric_limits<std::size_t>::max();
        SpanBuffer<View> buf;

        expect_from_span_error(View::from_span(Span{}, 3U, 2U), cxxgpr::GprErrorCode::EmptyInput, "null empty span");
        expect_from_span_error(
            View::from_span(buf.span_at(8U, 0U), 3U, 2U), cxxgpr::GprErrorCode::EmptyInput, "non-null empty span"
        );
        expect_from_span_error(
            View::from_span(buf.span_at(0U, 6U), 0U, 2U), cxxgpr::GprErrorCode::EmptyInput, "rows == 0"
        );
        expect_from_span_error(
            View::from_span(buf.span_at(0U, 6U), 3U, 0U), cxxgpr::GprErrorCode::EmptyInput, "cols == 0"
        );
        expect_from_span_error(
            View::from_span(buf.span_at(0U, 6U), 0U, 0U), cxxgpr::GprErrorCode::EmptyInput, "rows == cols == 0"
        );
        expect_from_span_error(
            View::from_span(buf.span_at(0U, 6U), max_sz, 2U), cxxgpr::GprErrorCode::SizeOverflow, "SIZE_MAX x 2"
        );
        expect_from_span_error(
            View::from_span(buf.span_at(0U, 6U), 2U, max_sz), cxxgpr::GprErrorCode::SizeOverflow, "2 x SIZE_MAX"
        );
        expect_from_span_error(
            View::from_span(buf.span_at(0U, 5U), 3U, 2U), cxxgpr::GprErrorCode::LengthMismatch, "short span"
        );
        expect_from_span_error(
            View::from_span(buf.span_at(0U, 7U), 3U, 2U), cxxgpr::GprErrorCode::LengthMismatch, "long span"
        );

        const auto short_span = View::from_span(buf.span_at(0U, 5U), 3U, 2U);
        if (!short_span.has_value())
        {
            EXPECT_EQ(std::string(short_span.error().message()), "length mismatch: expected 6 matrix values, got 5");
        }
    }

    /// @brief Checks that View::from_span() returns the earliest error when adjacent conditions hold together.
    /// @details EmptyInput outranks SizeOverflow (empty span, null or not, with overflowing extents) and
    /// LengthMismatch (zero extent with a non-empty span); SizeOverflow outranks LengthMismatch (overflowing
    /// extents can never match the span size).
    /// @tparam View MatrixRef<T> or MatrixMut<T>.
    /// @param[in] view_name Name of View printed on failure.
    template <typename View> void check_from_span_validation_order(const char* view_name)
    {
        SCOPED_TRACE(view_name);
        using Span = typename SpanBuffer<View>::Span;
        constexpr std::size_t max_sz = std::numeric_limits<std::size_t>::max();
        SpanBuffer<View> buf;

        expect_from_span_error(
            View::from_span(Span{}, max_sz, 2U), cxxgpr::GprErrorCode::EmptyInput, "null empty span + overflow"
        );
        expect_from_span_error(
            View::from_span(buf.span_at(8U, 0U), 2U, max_sz), cxxgpr::GprErrorCode::EmptyInput,
            "non-null empty span + overflow"
        );
        expect_from_span_error(
            View::from_span(buf.span_at(0U, 6U), 0U, max_sz), cxxgpr::GprErrorCode::EmptyInput,
            "rows == 0 + span length 6 != 0"
        );
        expect_from_span_error(
            View::from_span(buf.span_at(0U, 5U), max_sz, max_sz), cxxgpr::GprErrorCode::SizeOverflow,
            "overflow + span length mismatch"
        );
    }

    /// @brief Checks that a successful View::from_span() call allocates no heap memory.
    /// @details Builds a view inside AuditMode::CppAndEigen NoAllocationScope and checks the allocation delta.
    /// @tparam View MatrixRef<T> or MatrixMut<T>.
    /// @param[in] view_name Name of View printed on failure.
    template <typename View> void check_from_span_allocates_zero(const char* view_name)
    {
        SCOPED_TRACE(view_name);
        SpanBuffer<View> buf;
        bool ok = false;
        std::size_t delta = 1;
        {
            cxxgpr::test_support::NoAllocationScope scope(cxxgpr::test_support::AuditMode::CppAndEigen);
            const auto view = View::from_span(buf.span_at(0U, 8U), 4U, 2U);
            ok = view.has_value();
            cxxgpr::test_support::touch_for_test(&ok);
            delta = scope.delta_allocations();
        }
        EXPECT_TRUE(ok);
        EXPECT_EQ(delta, 0U);
    }

    /// @brief Verifies that from_span() binds the span's pointer and a tightly packed shape.
    /// @details Runs check_from_span_binds_span() for MatrixRef and MatrixMut over double and float.
    TEST(ViewsTest, FromSpanBindsSpanPointerAndPackedShape)
    {
        check_from_span_binds_span<cxxgpr::MatrixRef<double>>("MatrixRef<double>");
        check_from_span_binds_span<cxxgpr::MatrixMut<double>>("MatrixMut<double>");
        check_from_span_binds_span<cxxgpr::MatrixRef<float>>("MatrixRef<float>");
        check_from_span_binds_span<cxxgpr::MatrixMut<float>>("MatrixMut<float>");
    }

    /// @brief Verifies that from_span() rejects every shape violation with its own error code.
    /// @details Runs check_from_span_rejects_each_violation() for MatrixRef and MatrixMut over double and float.
    TEST(ViewsTest, FromSpanRejectsEachShapeViolation)
    {
        check_from_span_rejects_each_violation<cxxgpr::MatrixRef<double>>("MatrixRef<double>");
        check_from_span_rejects_each_violation<cxxgpr::MatrixMut<double>>("MatrixMut<double>");
        check_from_span_rejects_each_violation<cxxgpr::MatrixRef<float>>("MatrixRef<float>");
        check_from_span_rejects_each_violation<cxxgpr::MatrixMut<float>>("MatrixMut<float>");
    }

    /// @brief Verifies that from_span() applies EmptyInput, SizeOverflow, and LengthMismatch in this order.
    /// @details Runs check_from_span_validation_order() for MatrixRef and MatrixMut over double and float.
    TEST(ViewsTest, FromSpanReportsViolationsInValidationOrder)
    {
        check_from_span_validation_order<cxxgpr::MatrixRef<double>>("MatrixRef<double>");
        check_from_span_validation_order<cxxgpr::MatrixMut<double>>("MatrixMut<double>");
        check_from_span_validation_order<cxxgpr::MatrixRef<float>>("MatrixRef<float>");
        check_from_span_validation_order<cxxgpr::MatrixMut<float>>("MatrixMut<float>");
    }

    /// @brief Verifies that a successful from_span() call allocates no heap memory.
    /// @details Runs check_from_span_allocates_zero() for MatrixRef and MatrixMut over double and float.
    TEST(ViewsTest, FromSpanSuccessAllocatesZero)
    {
        check_from_span_allocates_zero<cxxgpr::MatrixRef<double>>("MatrixRef<double>");
        check_from_span_allocates_zero<cxxgpr::MatrixMut<double>>("MatrixMut<double>");
        check_from_span_allocates_zero<cxxgpr::MatrixRef<float>>("MatrixRef<float>");
        check_from_span_allocates_zero<cxxgpr::MatrixMut<float>>("MatrixMut<float>");
    }

}  // namespace
