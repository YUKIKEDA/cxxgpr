// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/precision/refine.hpp"

#include "allocation_audit.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

// Spec: REQ-07, REQ-11, DES-02, DES-05
namespace {

template <typename Scalar>
bool factor_lower_llt(std::vector<Scalar>& a_col_major, std::size_t n) noexcept {
    for (std::size_t j = 0; j < n; ++j) {
        auto diag = static_cast<double>(a_col_major[(j * n) + j]);
        for (std::size_t k = 0; k < j; ++k) {
            const auto l_jk = static_cast<double>(a_col_major[(k * n) + j]);
            diag -= l_jk * l_jk;
        }
        if (!(diag > 0.0) || !std::isfinite(diag)) {
            return false;
        }
        const auto l_jj = static_cast<Scalar>(std::sqrt(diag));
        a_col_major[(j * n) + j] = l_jj;
        for (std::size_t i = j + 1; i < n; ++i) {
            auto sum = static_cast<double>(a_col_major[(j * n) + i]);
            for (std::size_t k = 0; k < j; ++k) {
                sum -= static_cast<double>(a_col_major[(k * n) + i]) *
                       static_cast<double>(a_col_major[(k * n) + j]);
            }
            a_col_major[(j * n) + i] = static_cast<Scalar>(sum / static_cast<double>(l_jj));
        }
    }
    return true;
}

template <typename Scalar>
void solve_lower_llt(std::span<const Scalar> l_col_major, std::size_t n,
                     std::span<Scalar> x) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        auto sum = static_cast<double>(x[i]);
        for (std::size_t k = 0; k < i; ++k) {
            sum -= static_cast<double>(l_col_major[(k * n) + i]) * static_cast<double>(x[k]);
        }
        x[i] = static_cast<Scalar>(sum / static_cast<double>(l_col_major[(i * n) + i]));
    }
    for (std::size_t rev = n; rev > 0; --rev) {
        const std::size_t i = rev - 1U;
        auto sum = static_cast<double>(x[i]);
        for (std::size_t k = i + 1; k < n; ++k) {
            sum -= static_cast<double>(l_col_major[(i * n) + k]) * static_cast<double>(x[k]);
        }
        x[i] = static_cast<Scalar>(sum / static_cast<double>(l_col_major[(i * n) + i]));
    }
}

double relative_inf_error(std::span<const double> got, std::span<const double> truth) noexcept {
    double max_diff = 0.0;
    double max_truth = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(got[i] - truth[i]));
        max_truth = std::max(max_truth, std::abs(truth[i]));
    }
    return max_diff / max_truth;
}

std::vector<double> make_grid_points(std::size_t n) {
    std::vector<double> pts(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        pts[i] = static_cast<double>(i) / static_cast<double>(n - 1U);
    }
    return pts;
}

std::vector<double> make_forrester_targets(std::span<const double> pts) {
    std::vector<double> y(pts.size(), 0.0);
    for (std::size_t i = 0; i < pts.size(); ++i) {
        const double z = (6.0 * pts[i]) - 2.0;
        y[i] = (z * z) * std::sin((12.0 * pts[i]) - 4.0);
    }
    return y;
}

double rbf_entry(double xi, double xj, double inv_two_ell2) noexcept {
    const double diff = xi - xj;
    return std::exp(-(diff * diff) * inv_two_ell2);
}

// Simulates both PromoteStorage and ReevaluateKernel (column block 32) on an n x n RBF system.
class RbfTestRefineSystem final : public cxxgpr::RefineSystem {
public:
    enum class Mode : std::uint8_t {
        PromoteExactF32,
        PromoteAgainstF64Rbf,
        ReevaluateBlockedF64,
    };

    RbfTestRefineSystem(std::span<const double> pts, std::span<const double> y, double ell,
                        double diag_offset, Mode mode)
        : n_(pts.size()), inv_two_ell2_(1.0 / (2.0 * ell * ell)), diag_offset_(diag_offset),
          mode_(mode), pts_(pts.begin(), pts.end()), y_(y.begin(), y.end()), a_f32_(n_ * n_, 0.0F),
          l_f32_(n_ * n_, 0.0F), l_f64_(n_ * n_, 0.0),
          k_block_f64_(n_ * cxxgpr::refinement_column_block, 0.0), row_abs_(n_, 0.0),
          row_sum_(n_, 0.0), rhs_f32_(n_, 0.0F), accept_resid_(n_, 0.0), fallback_rhs_(n_, 0.0) {
        // Fill only the lower triangle of a_f64, symmetrize with symmetrize_lower, then build
        // a_f32.
        std::vector<double> a_f64(n_ * n_, 0.0);
        for (std::size_t col = 0; col < n_; ++col) {
            a_f64[(col * n_) + col] = rbf_entry(pts_[col], pts_[col], inv_two_ell2_) + diag_offset_;
            for (std::size_t row = col + 1; row < n_; ++row) {
                a_f64[(col * n_) + row] = rbf_entry(pts_[row], pts_[col], inv_two_ell2_);
            }
        }
        cxxgpr::symmetrize_lower(cxxgpr::MatrixMut<double>{a_f64.data(), n_, n_});

        for (std::size_t idx = 0; idx < n_ * n_; ++idx) {
            a_f32_[idx] = static_cast<float>(a_f64[idx]);
        }
        if (mode_ == Mode::PromoteExactF32) {
            // Make the true f64 matrix identical to the promoted f32 matrix so PromoteStorage
            // accept check genuinely passes without fallback.
            for (std::size_t idx = 0; idx < n_ * n_; ++idx) {
                a_f64[idx] = static_cast<double>(a_f32_[idx]);
            }
        }

        l_f32_ = a_f32_;
        EXPECT_TRUE(factor_lower_llt(l_f32_, n_));
        l_f64_ = a_f64;
        EXPECT_TRUE(factor_lower_llt(l_f64_, n_));

        f64_truth_ = y_;
        solve_lower_llt<double>(l_f64_, n_, f64_truth_);

        for (std::size_t i = 0; i < n_; ++i) {
            rhs_f32_[i] = static_cast<float>(y_[i]);
        }
        solve_lower_llt<float>(l_f32_, n_, rhs_f32_);
        initial_w_.resize(n_);
        for (std::size_t i = 0; i < n_; ++i) {
            initial_w_[i] = static_cast<double>(rhs_f32_[i]);
        }
    }

    [[nodiscard]] const std::vector<double>& initial_w() const noexcept {
        return initial_w_;
    }

    [[nodiscard]] const std::vector<double>& truth_f64() const noexcept {
        return f64_truth_;
    }

    [[nodiscard]] std::span<const double> rhs() const noexcept override {
        return y_;
    }

    [[nodiscard]] std::expected<double, cxxgpr::GprError> residual(std::span<const double> w,
                                                                   std::span<double> r) override {
        if (mode_ == Mode::ReevaluateBlockedF64) {
            return reevaluate_blocked_residual(w, r);
        }
        const cxxgpr::MatrixRef<float> a_ref{a_f32_.data(), n_, n_};
        return cxxgpr::promoted_residual_into(a_ref, w, y_, r, row_abs_, row_sum_);
    }

    [[nodiscard]] std::expected<void, cxxgpr::GprError> correct(std::span<const double> r,
                                                                std::span<double> w) override {
        for (std::size_t i = 0; i < n_; ++i) {
            rhs_f32_[i] = static_cast<float>(r[i]);
        }
        solve_lower_llt<float>(l_f32_, n_, rhs_f32_);
        for (std::size_t i = 0; i < n_; ++i) {
            w[i] += static_cast<double>(rhs_f32_[i]);
        }
        return {};
    }

    [[nodiscard]] std::expected<bool, cxxgpr::GprError> accept(std::span<const double> w,
                                                               double tol) override {
        if (mode_ == Mode::ReevaluateBlockedF64) {
            return true;
        }
        if (mode_ == Mode::PromoteExactF32) {
            const cxxgpr::MatrixRef<float> a_ref{a_f32_.data(), n_, n_};
            auto b_inf =
                cxxgpr::promoted_residual_into(a_ref, w, y_, accept_resid_, row_abs_, row_sum_);
            if (!b_inf) {
                return std::unexpected(std::move(b_inf).error());
            }
            return cxxgpr::meets_relative_residual(*b_inf, w, y_, accept_resid_, tol);
        }
        auto b_inf = reevaluate_blocked_residual(w, accept_resid_);
        if (!b_inf) {
            return std::unexpected(std::move(b_inf).error());
        }
        return cxxgpr::meets_relative_residual(*b_inf, w, y_, accept_resid_, tol);
    }

    [[nodiscard]] std::expected<void, cxxgpr::GprError> fallback(std::span<double> w) override {
        std::ranges::copy(y_, fallback_rhs_.begin());
        solve_lower_llt<double>(l_f64_, n_, fallback_rhs_);
        std::ranges::copy(fallback_rhs_, w.begin());
        return {};
    }

private:
    std::expected<double, cxxgpr::GprError> reevaluate_blocked_residual(std::span<const double> w,
                                                                        std::span<double> r) {
        std::ranges::fill(row_abs_, 0.0);
        std::ranges::fill(row_sum_, 0.0);
        for (std::size_t start = 0; start < n_; start += cxxgpr::refinement_column_block) {
            const std::size_t len = std::min(cxxgpr::refinement_column_block, n_ - start);
            for (std::size_t jj = 0; jj < len; ++jj) {
                const std::size_t col = start + jj;
                for (std::size_t i = 0; i < n_; ++i) {
                    double val = rbf_entry(pts_[i], pts_[col], inv_two_ell2_);
                    if (i == col) {
                        val += diag_offset_;
                    }
                    k_block_f64_[(jj * n_) + i] = val;
                }
            }
            for (std::size_t jj = 0; jj < len; ++jj) {
                const std::size_t col = start + jj;
                const double w_j = w[col];
                for (std::size_t i = 0; i < n_; ++i) {
                    const double b_ij = k_block_f64_[(jj * n_) + i];
                    row_abs_[i] += std::abs(b_ij);
                    row_sum_[i] += b_ij * w_j;
                }
            }
        }
        for (std::size_t i = 0; i < n_; ++i) {
            r[i] = y_[i] - row_sum_[i];
        }
        return cxxgpr::inf_norm(row_abs_);
    }

    std::size_t n_;
    double inv_two_ell2_;
    double diag_offset_;
    Mode mode_;
    std::vector<double> pts_;
    std::vector<double> y_;
    std::vector<float> a_f32_;
    std::vector<float> l_f32_;
    std::vector<double> l_f64_;
    std::vector<double> k_block_f64_;
    std::vector<double> row_abs_;
    std::vector<double> row_sum_;
    std::vector<float> rhs_f32_;
    std::vector<double> accept_resid_;
    std::vector<double> fallback_rhs_;
    std::vector<double> initial_w_;
    std::vector<double> f64_truth_;
};

// Scriptable system for deterministic branch, recovery, stagnation, max-iter, zero-RHS, and error
// tests.
class ScriptedRefineSystem final : public cxxgpr::RefineSystem {
public:
    explicit ScriptedRefineSystem(std::vector<double> rhs_vec, std::vector<double> b_infs,
                                  std::vector<double> residual_norms)
        : rhs_(std::move(rhs_vec)), b_infs_(std::move(b_infs)),
          residual_norms_(std::move(residual_norms)) {}

    std::size_t residual_calls = 0;
    std::size_t correct_calls = 0;
    std::size_t accept_calls = 0;
    std::size_t fallback_calls = 0;

    bool accept_result = true;
    std::optional<cxxgpr::GprError> fail_residual;
    std::optional<cxxgpr::GprError> fail_correct;
    std::optional<cxxgpr::GprError> fail_accept;
    std::optional<cxxgpr::GprError> fail_fallback;
    std::size_t fail_residual_on_call = 0;
    std::vector<double> fallback_value{42.0, -7.0, 3.0};

    [[nodiscard]] std::span<const double> rhs() const noexcept override {
        return rhs_;
    }

    [[nodiscard]] std::expected<double, cxxgpr::GprError> residual(std::span<const double> /*w*/,
                                                                   std::span<double> r) override {
        const std::size_t idx = residual_calls++;
        if (fail_residual.has_value() && idx == fail_residual_on_call) {
            return std::unexpected(*fail_residual);
        }
        std::ranges::fill(r, 0.0);
        r[0] = idx < residual_norms_.size() ? residual_norms_[idx] : residual_norms_.back();
        return idx < b_infs_.size() ? b_infs_[idx] : b_infs_.back();
    }

    [[nodiscard]] std::expected<void, cxxgpr::GprError> correct(std::span<const double> r,
                                                                std::span<double> w) override {
        ++correct_calls;
        if (fail_correct.has_value()) {
            return std::unexpected(*fail_correct);
        }
        if (cxxgpr::inf_norm(r) == 0.0) {
            return {};
        }
        for (double& v : w) {
            v += 1e-6;
        }
        return {};
    }

    [[nodiscard]] std::expected<bool, cxxgpr::GprError> accept(std::span<const double> /*w*/,
                                                               double /*tol*/) override {
        ++accept_calls;
        if (fail_accept.has_value()) {
            return std::unexpected(*fail_accept);
        }
        return accept_result;
    }

    [[nodiscard]] std::expected<void, cxxgpr::GprError> fallback(std::span<double> w) override {
        ++fallback_calls;
        if (fail_fallback.has_value()) {
            return std::unexpected(*fail_fallback);
        }
        for (std::size_t i = 0; i < w.size(); ++i) {
            w[i] = i < fallback_value.size() ? fallback_value[i] : 0.0;
        }
        return {};
    }

private:
    std::vector<double> rhs_;
    std::vector<double> b_infs_;
    std::vector<double> residual_norms_;
};

TEST(PrecisionRefineTest, ConstantsAndNormHelpersMatchSpecification) {
    EXPECT_EQ(cxxgpr::max_refinement_corrections, 10U);
    EXPECT_DOUBLE_EQ(cxxgpr::refinement_stagnation_ratio, 0.9);
    EXPECT_EQ(cxxgpr::refinement_stagnation_steps, 2U);
    EXPECT_EQ(cxxgpr::refinement_column_block, 32U);
    EXPECT_DOUBLE_EQ(cxxgpr::refinement_tolerance(40U),
                     10.0 * 40.0 * std::numeric_limits<double>::epsilon());

    EXPECT_DOUBLE_EQ(cxxgpr::inf_norm({}), 0.0);
    const std::array<double, 4> vals{-1.5, 4.25, -7.5, 2.0};
    EXPECT_DOUBLE_EQ(cxxgpr::inf_norm(vals), 7.5);

    const std::array<double, 3> with_nan{1.0, std::numeric_limits<double>::quiet_NaN(), 2.0};
    EXPECT_TRUE(std::isnan(cxxgpr::inf_norm(with_nan)));

    // Symmetrize lower triangle on a padded stride-4 3x3 matrix.
    std::array<double, 12> mat_raw{1.0, -2.0, 3.0, 99.0, 0.0, 4.0, -5.0, 99.0, 0.0, 0.0, 6.0, 99.0};
    cxxgpr::symmetrize_lower(cxxgpr::MatrixMut<double>{mat_raw.data(), 3U, 3U, 4U});
    const cxxgpr::MatrixRef<double> mat_view{mat_raw.data(), 3U, 3U, 4U};
    EXPECT_DOUBLE_EQ(mat_view(0U, 1U), -2.0);
    EXPECT_DOUBLE_EQ(mat_view(0U, 2U), 3.0);
    EXPECT_DOUBLE_EQ(mat_view(1U, 2U), -5.0);
    EXPECT_DOUBLE_EQ(mat_raw[3], 99.0);

    // Non-square symmetrize_lower is a safe no-op.
    std::array<double, 6> rect{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    cxxgpr::symmetrize_lower(cxxgpr::MatrixMut<double>{rect.data(), 2U, 3U});
    EXPECT_EQ(rect, (std::array<double, 6>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));

    // meets_relative_residual guards zero denom, invalid tolerances, and non-finite inputs.
    const std::array<double, 2> w{1.0, -1.0};
    const std::array<double, 2> b{2.0, 0.0};
    const std::array<double, 2> r_small{1e-13, -1e-13};
    const std::array<double, 2> r_large{1e-4, 0.0};
    const std::array<double, 2> zero2{0.0, 0.0};
    EXPECT_TRUE(cxxgpr::meets_relative_residual(5.0, w, b, r_small, 1e-10));
    EXPECT_FALSE(cxxgpr::meets_relative_residual(5.0, w, b, r_large, 1e-10));
    EXPECT_FALSE(cxxgpr::meets_relative_residual(0.0, zero2, zero2, zero2, 1e-10));
    EXPECT_FALSE(cxxgpr::meets_relative_residual(-1.0, w, b, r_small, 1e-10));
    EXPECT_FALSE(cxxgpr::meets_relative_residual(std::numeric_limits<double>::infinity(), w, b,
                                                 r_small, 1e-10));
    EXPECT_FALSE(cxxgpr::meets_relative_residual(5.0, w, b, r_small, 0.0));
}

TEST(PrecisionRefineTest, PromotedResidualMatchesIndependentRowDotsBitExactAndZeroAlloc) {
    // 3x3 matrix with column stride 4 (padding element at indices 3, 7, 11).
    const std::array<float, 12> a_raw{
        2.5F,  -1.25F, 0.5F,   999.0F, // col 0
        0.5F,  3.25F,  -0.75F, 999.0F, // col 1
        -1.0F, 0.25F,  4.0F,   999.0F, // col 2
    };
    const cxxgpr::MatrixRef<float> a_ref{a_raw.data(), 3U, 3U, 4U};
    const std::array<double, 3> alpha{1.125, -0.75, 0.375};
    const std::array<double, 3> y{2.0, -3.5, 4.25};

    std::array<double, 3> expected_r{};
    double expected_b_inf = 0.0;
    for (std::size_t i = 0; i < 3U; ++i) {
        double row_sum = 0.0;
        double row_abs = 0.0;
        for (std::size_t j = 0; j < 3U; ++j) {
            const auto val = static_cast<double>(a_ref(i, j));
            row_sum += val * alpha[j];
            row_abs += std::abs(val);
        }
        expected_r[i] = y[i] - row_sum;
        expected_b_inf = std::max(expected_b_inf, row_abs);
    }

    // Fill scratch buffers with non-zero garbage to verify reuse clears prior contents.
    std::array<double, 4> row_abs_scratch{100.0, 200.0, 300.0, 400.0};
    std::array<double, 4> sum_scratch{-50.0, -60.0, -70.0, -80.0};
    std::array<double, 3> residual{};

    std::expected<double, cxxgpr::GprError> got = std::unexpected(cxxgpr::GprError::empty_input());
    {
        const cxxgpr::test_support::NoAllocationScope scope;
        got =
            cxxgpr::promoted_residual_into(a_ref, alpha, y, residual, row_abs_scratch, sum_scratch);
        EXPECT_EQ(scope.delta_allocations(), 0U);
    }

    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(std::bit_cast<std::uint64_t>(*got), std::bit_cast<std::uint64_t>(expected_b_inf));
    for (std::size_t i = 0; i < 3U; ++i) {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(residual[i]),
                  std::bit_cast<std::uint64_t>(expected_r[i]));
    }
    // Elements beyond n in oversized scratch remain untouched.
    EXPECT_DOUBLE_EQ(row_abs_scratch[3], 400.0);
    EXPECT_DOUBLE_EQ(sum_scratch[3], -80.0);

    std::array<double, 3> convenience_residual{};
    const auto conv_got = cxxgpr::promoted_residual(a_ref, alpha, y, convenience_residual);
    ASSERT_TRUE(conv_got.has_value());
    EXPECT_EQ(std::bit_cast<std::uint64_t>(*conv_got),
              std::bit_cast<std::uint64_t>(expected_b_inf));
    EXPECT_EQ(convenience_residual, residual);
}

TEST(PrecisionRefineTest, PromotedResidualRejectsInvalidShapesAndUndersizedScratch) {
    const std::array<float, 4> a2x2{1.0F, 0.0F, 0.0F, 1.0F};
    const cxxgpr::MatrixRef<float> ref_2x2{a2x2.data(), 2U, 2U};
    const std::array<double, 2> v2{1.0, 2.0};
    std::array<double, 2> s2{};
    std::array<double, 1> s1{};

    const auto empty_y = cxxgpr::promoted_residual_into(ref_2x2, v2, {}, s2, s2, s2);
    ASSERT_FALSE(empty_y.has_value());
    EXPECT_EQ(empty_y.error().code(), cxxgpr::GprErrorCode::EmptyInput);

    const cxxgpr::MatrixRef<float> null_a{nullptr, 2U, 2U};
    const auto empty_a = cxxgpr::promoted_residual_into(null_a, v2, v2, s2, s2, s2);
    ASSERT_FALSE(empty_a.has_value());
    EXPECT_EQ(empty_a.error().code(), cxxgpr::GprErrorCode::EmptyInput);

    const std::array<float, 6> a2x3{1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F};
    const auto shape_err = cxxgpr::promoted_residual_into(
        cxxgpr::MatrixRef<float>{a2x3.data(), 2U, 3U}, v2, v2, s2, s2, s2);
    ASSERT_FALSE(shape_err.has_value());
    EXPECT_EQ(shape_err.error().code(), cxxgpr::GprErrorCode::LengthMismatch);

    const std::array<double, 3> v3{1.0, 2.0, 3.0};
    const auto alpha_err = cxxgpr::promoted_residual_into(ref_2x2, v3, v2, s2, s2, s2);
    ASSERT_FALSE(alpha_err.has_value());
    EXPECT_EQ(alpha_err.error().code(), cxxgpr::GprErrorCode::LengthMismatch);

    const auto resid_err = cxxgpr::promoted_residual_into(ref_2x2, v2, v2, s1, s2, s2);
    ASSERT_FALSE(resid_err.has_value());
    EXPECT_EQ(resid_err.error().code(), cxxgpr::GprErrorCode::LengthMismatch);

    const auto ws_err1 = cxxgpr::promoted_residual_into(ref_2x2, v2, v2, s2, s1, s2);
    ASSERT_FALSE(ws_err1.has_value());
    EXPECT_EQ(ws_err1.error().code(), cxxgpr::GprErrorCode::WorkspaceTooSmall);

    const auto ws_err2 = cxxgpr::promoted_residual_into(ref_2x2, v2, v2, s2, s2, s1);
    ASSERT_FALSE(ws_err2.has_value());
    EXPECT_EQ(ws_err2.error().code(), cxxgpr::GprErrorCode::WorkspaceTooSmall);
}

TEST(PrecisionRefineTest, ConvergesForBothPromoteStorageAndReevaluateKernelWithZeroAlloc) {
    constexpr std::size_t n = 40U;
    const auto pts = make_grid_points(n);
    const auto y = make_forrester_targets(pts);
    const double tol = cxxgpr::refinement_tolerance(n);

    // PromoteStorage on float-exact SPD Gram matrix (n = 40, ell = 0.5, sigma_n^2 = 1e-2).
    {
        RbfTestRefineSystem sys{pts, y, 0.5, 1e-2, RbfTestRefineSystem::Mode::PromoteExactF32};
        const double initial_err = relative_inf_error(sys.initial_w(), sys.truth_f64());
        EXPECT_GT(initial_err, 1e-8);

        std::vector<double> w = sys.initial_w();
        std::vector<double> resid(n, 0.0);
        std::expected<cxxgpr::RefinementSummary, cxxgpr::GprError> summary =
            std::unexpected(cxxgpr::GprError::empty_input());
        {
            const cxxgpr::test_support::NoAllocationScope scope;
            summary = cxxgpr::refine_in_place(sys, w, resid);
            EXPECT_EQ(scope.delta_allocations(), 0U);
        }

        ASSERT_TRUE(summary.has_value());
        EXPECT_EQ(summary->status, cxxgpr::RefinementStatus::Converged);
        EXPECT_FALSE(summary->used_fallback());
        EXPECT_GT(summary->iterations, 0U);
        EXPECT_LT(summary->iterations, cxxgpr::max_refinement_corrections);
        EXPECT_LT(summary->final_relative_residual, tol);
        // Condition number kappa_inf(B) ~ 2.5e3 -> forward error bound kappa * tol ~ 2.2e-10;
        // fixed 1e-9 check proves massive refinement from the ~1e-6 float Cholesky initial solve.
        EXPECT_LT(relative_inf_error(w, sys.truth_f64()), 1e-9);
    }

    // ReevaluateKernel across multiple 32-column f64 blocks (40 = 32 + 8) with ell = 0.5,
    // diag_offset = 1e-3.
    {
        RbfTestRefineSystem sys{pts, y, 0.5, 1e-3, RbfTestRefineSystem::Mode::ReevaluateBlockedF64};
        const double initial_err = relative_inf_error(sys.initial_w(), sys.truth_f64());
        EXPECT_GT(initial_err, 1e-8);

        std::vector<double> w = sys.initial_w();
        std::vector<double> resid(n, 0.0);
        std::expected<cxxgpr::RefinementSummary, cxxgpr::GprError> summary =
            std::unexpected(cxxgpr::GprError::empty_input());
        {
            const cxxgpr::test_support::NoAllocationScope scope;
            summary = cxxgpr::refine_in_place(sys, w, resid);
            EXPECT_EQ(scope.delta_allocations(), 0U);
        }

        ASSERT_TRUE(summary.has_value());
        EXPECT_EQ(summary->status, cxxgpr::RefinementStatus::Converged);
        EXPECT_FALSE(summary->used_fallback());
        EXPECT_GT(summary->iterations, 0U);
        EXPECT_LT(summary->iterations, cxxgpr::max_refinement_corrections);
        EXPECT_LT(summary->final_relative_residual, tol);
        // Condition number kappa_inf(B) ~ 2.5e4 -> forward error bound ~ 2.2e-9; fixed 1e-9
        // holds because step-2 refinement overshoots tol to ~1e-12.
        EXPECT_LT(relative_inf_error(w, sys.truth_f64()), 1e-9);
    }
}

TEST(PrecisionRefineTest, FallsBackWithoutErrorWhenPromoteStorageFailsTrueF64AcceptCheck) {
    constexpr std::size_t n = 40U;
    const auto pts = make_grid_points(n);
    const auto y = make_forrester_targets(pts);

    // Transcendental RBF (ell = 0.5, diag = 1e-3): PromoteStorage converges on B_f32 yet
    // single-precision matrix rounding causes accept() on B_f64 to reject and fall back.
    {
        RbfTestRefineSystem sys{pts, y, 0.5, 1e-3, RbfTestRefineSystem::Mode::PromoteAgainstF64Rbf};
        std::vector<double> w = sys.initial_w();
        std::vector<double> resid(n, 0.0);

        std::expected<cxxgpr::RefinementSummary, cxxgpr::GprError> summary =
            std::unexpected(cxxgpr::GprError::empty_input());
        {
            const cxxgpr::test_support::NoAllocationScope scope;
            summary = cxxgpr::refine_in_place(sys, w, resid);
            EXPECT_EQ(scope.delta_allocations(), 0U);
        }

        ASSERT_TRUE(summary.has_value());
        EXPECT_TRUE(summary->used_fallback());
        EXPECT_EQ(summary->status, cxxgpr::RefinementStatus::RejectedByAccept);
        EXPECT_LT(relative_inf_error(w, sys.truth_f64()), 1e-11);

        const auto refined_vec = cxxgpr::refine(sys, sys.initial_w());
        ASSERT_TRUE(refined_vec.has_value());
        EXPECT_LT(relative_inf_error(*refined_vec, sys.truth_f64()), 1e-11);
    }

    // Ill-conditioned RBF (ell = 10.0, diag = 3e-5, kappa ~ 1.3e6): PromoteStorage must never
    // return an unrefined float solution or fail, falling back cleanly to f64 Cholesky accuracy.
    {
        RbfTestRefineSystem ill_sys{pts, y, 10.0, 3e-5,
                                    RbfTestRefineSystem::Mode::PromoteAgainstF64Rbf};
        std::vector<double> w = ill_sys.initial_w();
        std::vector<double> resid(n, 0.0);
        const auto summary = cxxgpr::refine_in_place(ill_sys, w, resid);
        ASSERT_TRUE(summary.has_value());
        EXPECT_TRUE(summary->used_fallback());
        EXPECT_LT(relative_inf_error(w, ill_sys.truth_f64()), 1e-10);
    }
}

TEST(PrecisionRefineTest, EarlyConvergenceStagnationRecoveryAndMaxIterationsBehaveExactly) {
    // Already converged at step 0: zero corrections, zero fallback.
    {
        ScriptedRefineSystem sys{{1.0, 0.0, 0.0}, {2.0}, {1e-16}};
        std::vector<double> w{1.0, 0.0, 0.0};
        std::vector<double> resid(3, 0.0);
        const auto summary = cxxgpr::refine_in_place(sys, w, resid);
        ASSERT_TRUE(summary.has_value());
        EXPECT_EQ(summary->status, cxxgpr::RefinementStatus::Converged);
        EXPECT_FALSE(summary->used_fallback());
        EXPECT_EQ(summary->iterations, 0U);
        EXPECT_EQ(sys.correct_calls, 0U);
        EXPECT_EQ(sys.accept_calls, 1U);
        EXPECT_EQ(sys.fallback_calls, 0U);
    }

    // Two consecutive steps with ratio > 0.9 trigger immediate stagnation fallback before
    // correct().
    {
        // step 0: r=1.0 -> correct (#1)
        // step 1: r=0.95 (ratio 0.95 > 0.9, streak=1) -> correct (#2)
        // step 2: r=0.9025 (ratio 0.95 > 0.9, streak=2 >= 2) -> fallback
        ScriptedRefineSystem sys{{10.0, 0.0, 0.0}, {2.0}, {1.0, 0.95, 0.9025, 0.85}};
        std::vector<double> w{1.0, 2.0, 3.0};
        std::vector<double> resid(3, 0.0);
        const auto summary = cxxgpr::refine_in_place(sys, w, resid);
        ASSERT_TRUE(summary.has_value());
        EXPECT_EQ(summary->status, cxxgpr::RefinementStatus::Stagnated);
        EXPECT_TRUE(summary->used_fallback());
        EXPECT_EQ(summary->iterations, 2U);
        EXPECT_EQ(sys.residual_calls, 3U);
        EXPECT_EQ(sys.correct_calls, 2U);
        EXPECT_EQ(sys.fallback_calls, 1U);
        EXPECT_EQ(w, (std::vector<double>{42.0, -7.0, 3.0}));
    }

    // Stagnation counter resets to zero when a ratio <= 0.9 is observed.
    {
        // r0 = 1.0
        // r1 = 0.95 (ratio 0.95 > 0.9 -> streak 1)
        // r2 = 0.76 (ratio 0.80 <= 0.9 -> streak reset to 0)
        // r3 = 0.722 (ratio 0.95 > 0.9 -> streak 1 only)
        // r4 = 1e-16 (converges)
        ScriptedRefineSystem sys{{10.0, 0.0, 0.0}, {2.0}, {1.0, 0.95, 0.76, 0.722, 1e-16}};
        std::vector<double> w{1.0, 2.0, 3.0};
        std::vector<double> resid(3, 0.0);
        const auto summary = cxxgpr::refine_in_place(sys, w, resid);
        ASSERT_TRUE(summary.has_value());
        EXPECT_EQ(summary->status, cxxgpr::RefinementStatus::Converged);
        EXPECT_FALSE(summary->used_fallback());
        EXPECT_EQ(summary->iterations, 4U);
        EXPECT_EQ(sys.fallback_calls, 0U);
    }

    // Steadily shrinking residuals (ratio = 0.5 <= 0.9) that stay above tol reach max corrections.
    {
        std::vector<double> norms;
        double current = 1.0;
        for (std::size_t k = 0; k < cxxgpr::max_refinement_corrections; ++k) {
            norms.push_back(current);
            current *= 0.5;
        }
        ScriptedRefineSystem sys{{10.0, 0.0, 0.0}, {2.0}, std::move(norms)};
        std::vector<double> w{1.0, 2.0, 3.0};
        std::vector<double> resid(3, 0.0);
        const auto summary = cxxgpr::refine_in_place(sys, w, resid);
        ASSERT_TRUE(summary.has_value());
        EXPECT_EQ(summary->status, cxxgpr::RefinementStatus::MaxCorrectionsReached);
        EXPECT_TRUE(summary->used_fallback());
        EXPECT_EQ(summary->iterations, 10U);
        EXPECT_EQ(sys.residual_calls, 10U);
        EXPECT_EQ(sys.correct_calls, 10U);
        EXPECT_EQ(sys.fallback_calls, 1U);
    }
}

TEST(PrecisionRefineTest, ZeroRhsAvoidsDivisionByZeroAndPropagatesSystemAndShapeErrors) {
    // Zero RHS and zero initial iterate: denom == 0.0, never divides by zero, reaches fallback.
    {
        ScriptedRefineSystem sys{{0.0, 0.0, 0.0}, {1.0}, {0.0}};
        std::vector<double> w{0.0, 0.0, 0.0};
        std::vector<double> resid(3, 0.0);
        const auto summary = cxxgpr::refine_in_place(sys, w, resid);
        ASSERT_TRUE(summary.has_value());
        EXPECT_EQ(summary->status, cxxgpr::RefinementStatus::MaxCorrectionsReached);
        EXPECT_TRUE(std::isfinite(summary->final_relative_residual));
        EXPECT_DOUBLE_EQ(summary->final_relative_residual, 0.0);
        EXPECT_EQ(sys.fallback_calls, 1U);
    }

    // Shape negative controls.
    {
        ScriptedRefineSystem empty_sys{{}, {1.0}, {0.0}};
        std::vector<double> empty_w;
        std::vector<double> resid(3, 0.0);
        const auto empty_err = cxxgpr::refine_in_place(empty_sys, empty_w, resid);
        ASSERT_FALSE(empty_err.has_value());
        EXPECT_EQ(empty_err.error().code(), cxxgpr::GprErrorCode::EmptyInput);

        const auto empty_wrap = cxxgpr::refine(empty_sys, {});
        ASSERT_FALSE(empty_wrap.has_value());
        EXPECT_EQ(empty_wrap.error().code(), cxxgpr::GprErrorCode::EmptyInput);

        ScriptedRefineSystem sys3{{1.0, 2.0, 3.0}, {1.0}, {1.0}};
        std::vector<double> w2{1.0, 2.0};
        const auto len_err = cxxgpr::refine_in_place(sys3, w2, resid);
        ASSERT_FALSE(len_err.has_value());
        EXPECT_EQ(len_err.error().code(), cxxgpr::GprErrorCode::LengthMismatch);

        std::vector<double> w3{1.0, 2.0, 3.0};
        std::vector<double> tiny_resid(2, 0.0);
        const auto ws_err = cxxgpr::refine_in_place(sys3, w3, tiny_resid);
        ASSERT_FALSE(ws_err.has_value());
        EXPECT_EQ(ws_err.error().code(), cxxgpr::GprErrorCode::WorkspaceTooSmall);
    }

    // Hook error propagation: residual, correct, accept, fallback.
    {
        ScriptedRefineSystem sys{{1.0, 2.0, 3.0}, {1.0}, {1.0}};
        sys.fail_residual = cxxgpr::GprError::non_finite_kernel_value();
        std::vector<double> w(3, 0.0);
        std::vector<double> resid(3, 0.0);
        const auto err = cxxgpr::refine_in_place(sys, w, resid);
        ASSERT_FALSE(err.has_value());
        EXPECT_EQ(err.error().code(), cxxgpr::GprErrorCode::NonFiniteKernelValue);
    }
    {
        ScriptedRefineSystem sys{{1.0, 2.0, 3.0}, {1.0}, {1.0}};
        sys.fail_correct =
            cxxgpr::GprError::cholesky_failed(0.0, 3U, cxxgpr::CholeskyStage::Predict);
        std::vector<double> w(3, 0.0);
        std::vector<double> resid(3, 0.0);
        const auto err = cxxgpr::refine_in_place(sys, w, resid);
        ASSERT_FALSE(err.has_value());
        EXPECT_EQ(err.error().code(), cxxgpr::GprErrorCode::CholeskyFailed);
    }
    {
        ScriptedRefineSystem sys{{1.0, 2.0, 3.0}, {2.0}, {1e-16}};
        sys.fail_accept = cxxgpr::GprError::non_finite_input();
        std::vector<double> w{1.0, 0.0, 0.0};
        std::vector<double> resid(3, 0.0);
        const auto err = cxxgpr::refine_in_place(sys, w, resid);
        ASSERT_FALSE(err.has_value());
        EXPECT_EQ(err.error().code(), cxxgpr::GprErrorCode::NonFiniteInput);
    }
    {
        ScriptedRefineSystem sys{{1.0, 2.0, 3.0}, {2.0}, {1e-16}};
        sys.accept_result = false;
        sys.fail_fallback =
            cxxgpr::GprError::cholesky_failed(1e-3, 3U, cxxgpr::CholeskyStage::Predict);
        std::vector<double> w{1.0, 0.0, 0.0};
        std::vector<double> resid(3, 0.0);
        const auto err = cxxgpr::refine_in_place(sys, w, resid);
        ASSERT_FALSE(err.has_value());
        EXPECT_EQ(err.error().code(), cxxgpr::GprErrorCode::CholeskyFailed);
        EXPECT_EQ(err.error().jitter(), 1e-3);
    }
}

} // namespace
