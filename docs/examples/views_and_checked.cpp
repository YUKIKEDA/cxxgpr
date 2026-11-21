// SPDX-License-Identifier: Apache-2.0
// Matrix views, prediction structs, and checked arithmetic / validation helpers.
#include <cxxgpr/core/checked.hpp>
#include <cxxgpr/core/error.hpp>
#include <cxxgpr/core/views.hpp>

#include <array>
#include <cstddef>
#include <expected>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace
{
    bool expect_code(const char* label, const std::expected<void, cxxgpr::GprError>& result, cxxgpr::GprErrorCode code)
    {
        const bool good = !result.has_value() && result.error().code() == code;
        std::cout << label << ": " << (result ? "ok" : std::string(result.error().message())) << '\n';
        return good;
    }
}  // namespace

int main()
{
    bool ok = true;

    // [layout]
    // Row-major source data: one array per sample (3 samples, 2 features).
    const std::vector<std::array<double, 2>> samples{{1.0, 10.0}, {2.0, 20.0}, {3.0, 30.0}};
    // Column-major buffer: all values of feature 0, then all values of feature 1.
    std::vector<double> column_major(samples.size() * 2U);
    for (std::size_t row = 0; row < samples.size(); ++row)
    {
        for (std::size_t col = 0; col < 2U; ++col)
        {
            column_major[col * samples.size() + row] = samples[row][col];
        }
    }
    const cxxgpr::MatrixRef<double> x(std::span<const double>(column_major), samples.size(), 2U);
    std::cout << "x(1, 1) = " << x(1, 1) << ", buffer = " << column_major[0] << ' ' << column_major[1] << ' '
              << column_major[2] << " | " << column_major[3] << ' ' << column_major[4] << ' ' << column_major[5]
              << '\n';
    ok = ok && x(1, 1) == 20.0 && column_major[3] == 10.0;
    // [end]

    // [views]
    // A 3 x 2 column-major matrix: column 0 is {1, 2, 3}, column 1 is {4, 5, 6}.
    std::array<double, 6> storage{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const cxxgpr::MatrixRef<double> a(std::span<const double>(storage), 3U, 2U);
    std::cout << "a(2, 1) = " << a(2, 1) << ", rows = " << a.rows() << ", cols = " << a.cols()
              << ", col_stride = " << a.col_stride() << ", contiguous = " << std::boolalpha << a.is_contiguous()
              << '\n';
    std::cout << "column 1 has " << a.col(1).size() << " elements, first = " << a.col(1)[0] << '\n';

    // A 2 x 2 sub-block (rows 1..2) of the same buffer: same column stride 3, so it is not contiguous.
    const cxxgpr::MatrixRef<double> block(storage.data() + 1, 2U, 2U, 3U);
    std::cout << "block(0, 0) = " << block(0, 0) << ", block(1, 1) = " << block(1, 1)
              << ", contiguous = " << block.is_contiguous() << '\n';
    ok = ok && a(2, 1) == 6.0 && block(0, 0) == 2.0 && block(1, 1) == 6.0 && !block.is_contiguous();

    // A mutable view writes through; it converts implicitly to a read-only view.
    const cxxgpr::MatrixMut<double> m(storage.data(), 3U, 2U);
    m(0, 0) = 10.0;
    const cxxgpr::MatrixRef<double> as_read_only = m;
    std::cout << "after write: a(0, 0) = " << a(0, 0) << ", via MatrixMut::as_ref = " << m.as_ref()(0, 0)
              << ", via conversion = " << as_read_only(0, 0) << '\n';
    ok = ok && a(0, 0) == 10.0 && as_read_only(0, 0) == 10.0;

    const cxxgpr::MatrixRef<double> empty;
    std::cout << "default view: is_empty = " << empty.is_empty() << ", is_square = " << empty.is_square() << '\n';
    ok = ok && empty.is_empty() && empty.is_square() && a.col(5).empty();
    // [end]

    // [checked]
    const auto sum = cxxgpr::checked_add(2U, 3U);
    const auto prod = cxxgpr::checked_mul(1000U, 1000U);
    const auto len = cxxgpr::column_major_len(30U, 2U);
    const auto packed = cxxgpr::packed_len(4U);
    std::cout << "2 + 3 = " << *sum << ", 1000 * 1000 = " << *prod << ", 30 x 2 -> " << *len
              << " elements, packed_len(4) = " << *packed << '\n';
    ok = ok && *sum == 5U && *prod == 1000000U && *len == 60U && *packed == 10U;

    const std::size_t huge = std::numeric_limits<std::size_t>::max();
    const auto overflow = cxxgpr::checked_mul(huge, 2U);
    std::cout << "checked_mul(max, 2): " << overflow.error().message() << '\n';
    ok = ok && !overflow && overflow.error().code() == cxxgpr::GprErrorCode::SizeOverflow;
    ok = ok && !cxxgpr::checked_add(huge, 1U) && !cxxgpr::column_major_len(0U, 2U);
    // [end]

    // [validate]
    const std::array<double, 4> good{0.0, 1.0, 2.0, 3.0};
    const std::array<double, 4> bad{0.0, 1.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
    const std::array<double, 3> short_y{0.0, 1.0, 2.0};

    ok = ok && cxxgpr::require_nonempty(4U).has_value();
    ok = ok && expect_code("require_nonempty(0)", cxxgpr::require_nonempty(0U), cxxgpr::GprErrorCode::EmptyInput);
    ok = ok && expect_code("require_count(3, 4)", cxxgpr::require_count(3U, 4U, "y"), cxxgpr::GprErrorCode::LengthMismatch);
    ok = ok && expect_code("require_finite(bad)", cxxgpr::require_finite(std::span<const double>(bad)),
                           cxxgpr::GprErrorCode::NonFiniteInput);
    ok = ok && cxxgpr::require_finite(std::span<const double>(good)).has_value();
    ok = ok && expect_code("validate_matrix(2 x 3 over 4 values)", cxxgpr::validate_matrix(good, 2U, 3U, "X"),
                           cxxgpr::GprErrorCode::LengthMismatch);
    ok = ok && expect_code("validate_training(y too short)", cxxgpr::validate_training(good, 4U, 1U, short_y),
                           cxxgpr::GprErrorCode::LengthMismatch);
    ok = ok && expect_code("validate_training(wrong d)", cxxgpr::validate_training(good, 4U, 1U, good, 2U),
                           cxxgpr::GprErrorCode::DimensionMismatch);
    ok = ok && expect_code("validate_training(min_points = 5)",
                           cxxgpr::validate_training(good, 4U, 1U, good, std::nullopt, 5U),
                           cxxgpr::GprErrorCode::InsufficientData);
    ok = ok && expect_code("validate_distance_targets(n = 3)", cxxgpr::validate_distance_targets(3U, good),
                           cxxgpr::GprErrorCode::LengthMismatch);
    ok = ok && expect_code("validate_query(d = 2 expected)", cxxgpr::validate_query(good, 4U, 1U, 2U),
                           cxxgpr::GprErrorCode::DimensionMismatch);
    ok = ok && cxxgpr::validate_training(good, 4U, 1U, good).has_value() &&
         cxxgpr::validate_query(good, 4U, 1U, 1U).has_value();
    // [end]

    // [results]
    cxxgpr::Prediction<double> prediction;
    prediction.mean = {0.5, 1.5};
    prediction.variance = {0.04, 0.09};
    cxxgpr::PredictiveCovariance<double> joint;
    joint.n_points = 2U;
    joint.mean = {0.5, 1.5};
    joint.covariance = {0.04, 0.01, 0.01, 0.09};  // column-major 2 x 2: element (row, col) at col * 2 + row
    const cxxgpr::PredictOptions latent{.variance_kind = cxxgpr::VarianceKind::Latent};
    const cxxgpr::PredictOptions defaults;
    std::cout << "default options are Observation: "
              << (defaults.variance_kind == cxxgpr::VarianceKind::Observation) << ", options equal: "
              << (latent == defaults) << ", prediction == itself: " << (prediction == prediction)
              << ", Cov(0, 1) = " << joint.covariance[1 * joint.n_points + 0] << '\n';
    ok = ok && latent != defaults && prediction == prediction && joint.covariance[2] == 0.01;
    // [end]

    return ok ? 0 : 1;
}
