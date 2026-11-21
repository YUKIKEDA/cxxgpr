// SPDX-License-Identifier: Apache-2.0
// The iterative-refinement engine on its own: a custom RefineSystem over a float Cholesky factor.
#include <cxxgpr/linalg/cholesky.hpp>
#include <cxxgpr/linalg/dense.hpp>
#include <cxxgpr/precision/refine.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <span>
#include <vector>

namespace
{

    constexpr std::size_t n = 60;

    const char* to_string(cxxgpr::RefinementStatus status)
    {
        switch (status)
        {
            case cxxgpr::RefinementStatus::Converged:
                return "Converged";
            case cxxgpr::RefinementStatus::RejectedByAccept:
                return "RejectedByAccept";
            case cxxgpr::RefinementStatus::Stagnated:
                return "Stagnated";
            case cxxgpr::RefinementStatus::MaxCorrectionsReached:
                return "MaxCorrectionsReached";
        }
        return "?";
    }

    // [system]
    // How the residual b - B w is evaluated, and whether a converged iterate is double-checked.
    enum class Residual
    {
        StoredFloatMatrix,  // r = b - float(A) w: the matrix that the factor was built from
        StoredFloatChecked,  // same residual, but accept() re-checks the exact double matrix
        ExactDouble,         // r = b - A w with the double matrix
    };

    // A dense system A w = b whose approximate inverse is a float Cholesky factor.
    class FloatFactorSystem final : public cxxgpr::RefineSystem
    {
    public:
        FloatFactorSystem(
            cxxgpr::MatrixRef<double> a_f64, cxxgpr::MatrixRef<float> a_f32, cxxgpr::MatrixRef<float> l_f32,
            std::span<const double> b, Residual residual, double damping = 1.0
        ) :
            a_f64_(a_f64),
            a_f32_(a_f32),
            l_f32_(l_f32),
            b_(b),
            residual_(residual),
            damping_(damping),
            row_abs_(b.size()),
            sum_(b.size()),
            delta_(b.size())
        {
        }

        std::span<const double> rhs() const noexcept override { return b_; }

        // Writes r = b - B w and returns ||B||_inf for the chosen matrix B.
        std::expected<double, cxxgpr::GprError> residual(std::span<const double> w, std::span<double> r) override
        {
            if (residual_ != Residual::ExactDouble)
            {
                return cxxgpr::promoted_residual_into(a_f32_, w, b_, r, row_abs_, sum_);
            }
            return exact_residual(w, r);
        }

        // One refinement step: w += (L L^T)^-1 r, with the solve done in float.
        std::expected<void, cxxgpr::GprError> correct(std::span<const double> r, std::span<double> w) override
        {
            for (std::size_t i = 0; i < r.size(); ++i)
            {
                delta_[i] = static_cast<float>(r[i]);
            }
            if (auto solved = cxxgpr::linalg::solve_llt_in_place(l_f32_, std::span<float>(delta_)); !solved)
            {
                return solved;
            }
            for (std::size_t i = 0; i < w.size(); ++i)
            {
                w[i] += damping_ * static_cast<double>(delta_[i]);
            }
            return {};
        }

        // Only the verified variant overrides the default "always accept".
        std::expected<bool, cxxgpr::GprError> accept(std::span<const double> w, double tol) override
        {
            if (residual_ != Residual::StoredFloatChecked)
            {
                return RefineSystem::accept(w, tol);
            }
            std::vector<double> r(w.size());
            auto b_inf = exact_residual(w, r);
            if (!b_inf)
            {
                return std::unexpected(b_inf.error());
            }
            return cxxgpr::meets_relative_residual(*b_inf, w, b_, r, tol);
        }

        // Last resort: solve the exact double system with a double Cholesky factorization.
        std::expected<void, cxxgpr::GprError> fallback(std::span<double> w) override
        {
            std::vector<double> factor(a_f64_.rows() * a_f64_.cols());
            if (auto copied = cxxgpr::linalg::copy_lower_triangle(
                    a_f64_, cxxgpr::MatrixMut<double>(factor.data(), a_f64_.rows(), a_f64_.cols())
                );
                !copied)
            {
                return copied;
            }
            const cxxgpr::MatrixMut<double> l(factor.data(), a_f64_.rows(), a_f64_.cols());
            if (auto done = cxxgpr::linalg::cholesky_lower_in_place(l); !done)
            {
                return done;
            }
            std::ranges::copy(b_, w.begin());
            return cxxgpr::linalg::solve_llt_in_place(l.as_ref(), w);
        }

    private:
        std::expected<double, cxxgpr::GprError> exact_residual(std::span<const double> w, std::span<double> r)
        {
            if (auto product = cxxgpr::linalg::matvec_columns(a_f64_, w, std::span<double>(sum_)); !product)
            {
                return std::unexpected(product.error());
            }
            for (std::size_t i = 0; i < r.size(); ++i)
            {
                r[i] = b_[i] - sum_[i];
            }
            std::ranges::fill(row_abs_, 0.0);
            for (std::size_t col = 0; col < a_f64_.cols(); ++col)
            {
                for (std::size_t row = 0; row < a_f64_.rows(); ++row)
                {
                    row_abs_[row] += std::abs(a_f64_(row, col));
                }
            }
            return cxxgpr::inf_norm(row_abs_);
        }

        cxxgpr::MatrixRef<double> a_f64_;
        cxxgpr::MatrixRef<float> a_f32_;
        cxxgpr::MatrixRef<float> l_f32_;
        std::span<const double> b_;
        Residual residual_;
        double damping_;
        std::vector<double> row_abs_;
        std::vector<double> sum_;
        std::vector<float> delta_;
    };
    // [end]

    double relative_error(const std::vector<double>& a, const std::vector<double>& b)
    {
        double num = 0.0;
        double den = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            num = std::max(num, std::abs(a[i] - b[i]));
            den = std::max(den, std::abs(b[i]));
        }
        return num / den;
    }

}  // namespace

int main()
{
    using namespace cxxgpr;
    std::cout << std::setprecision(3) << std::scientific;

    // [setup]
    // A = RBF Gram matrix on [0, 3] plus noise, in double; the right-hand side is a smooth signal.
    std::vector<double> a(n * n);
    std::vector<double> b(n);
    for (std::size_t col = 0; col < n; ++col)
    {
        for (std::size_t row = 0; row < n; ++row)
        {
            const double d = 3.0 * (static_cast<double>(row) - static_cast<double>(col)) / static_cast<double>(n - 1);
            a[(col * n) + row] = std::exp(-0.5 * d * d / 0.25) + (row == col ? 1e-3 : 0.0);
        }
        b[col] = std::sin(6.0 * static_cast<double>(col) / static_cast<double>(n - 1));
    }
    const MatrixRef<double> a_f64(a.data(), n, n);

    // The float copy: round the lower triangle, mirror it, and factor a second copy of it.
    std::vector<float> a32(n * n);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        a32[i] = static_cast<float>(a[i]);
    }
    symmetrize_lower(MatrixMut<float>(a32.data(), n, n));
    std::vector<float> l32 = a32;
    if (auto r = linalg::cholesky_lower_in_place(MatrixMut<float>(l32.data(), n, n)); !r)
    {
        std::cerr << r.error().message() << '\n';
        return 1;
    }
    const MatrixRef<float> a_f32(a32.data(), n, n);
    const MatrixRef<float> l_f32(l32.data(), n, n);

    // Reference solution from a double factorization, and the plain float solve as the starting point.
    std::vector<double> factor64(n * n);
    (void)linalg::copy_lower_triangle(a_f64, MatrixMut<double>(factor64.data(), n, n));
    (void)linalg::cholesky_lower_in_place(MatrixMut<double>(factor64.data(), n, n));
    std::vector<double> reference = b;
    (void)linalg::solve_llt_in_place(MatrixRef<double>(factor64.data(), n, n), std::span<double>(reference));

    std::vector<float> start32(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        start32[i] = static_cast<float>(b[i]);
    }
    (void)linalg::solve_llt_in_place(l_f32, std::span<float>(start32));
    const std::vector<double> start(start32.begin(), start32.end());
    std::cout << "float solve only:   relative error " << relative_error(start, reference) << '\n';
    std::cout << "tolerance 10 n eps = " << refinement_tolerance(n) << ", at most " << max_refinement_corrections
              << " corrections, stagnation above ratio " << std::fixed << std::setprecision(1)
              << refinement_stagnation_ratio << " twice in a row\n"
              << std::scientific << std::setprecision(3);
    // [end]

    // [run]
    struct Case
    {
        const char* label;
        Residual residual;
        double damping;
        RefinementStatus expected;
    };
    const Case cases[] = {
        {"stored float matrix        ", Residual::StoredFloatMatrix, 1.0, RefinementStatus::Converged},
        {"stored float, checked      ", Residual::StoredFloatChecked, 1.0, RefinementStatus::RejectedByAccept},
        {"exact double residual      ", Residual::ExactDouble, 1.0, RefinementStatus::Converged},
        {"useless correction (x0)    ", Residual::ExactDouble, 0.0, RefinementStatus::Stagnated},
        {"weak correction (x0.2)     ", Residual::ExactDouble, 0.2, RefinementStatus::MaxCorrectionsReached},
    };
    bool ok = true;
    for (const Case& c : cases)
    {
        FloatFactorSystem system(a_f64, a_f32, l_f32, b, c.residual, c.damping);
        std::vector<double> w = start;
        std::vector<double> scratch(n);
        auto summary = refine_in_place(system, w, scratch);
        if (!summary)
        {
            std::cerr << summary.error().message() << '\n';
            return 1;
        }
        std::cout << c.label << to_string(summary->status) << ", corrections " << summary->iterations
                  << ", fallback " << summary->used_fallback() << ", relative residual "
                  << summary->final_relative_residual << ", error " << relative_error(w, reference) << '\n';
        ok = ok && summary->status == c.expected;
    }
    // [end]

    // [convenience]
    // refine() allocates the residual scratch itself and returns the refined vector.
    FloatFactorSystem exact(a_f64, a_f32, l_f32, b, Residual::ExactDouble);
    auto refined = refine(exact, start);
    if (!refined)
    {
        return 1;
    }

    // The stopping test used by the loop, applied by hand to the result.
    std::vector<double> r(n);
    const double a_inf = exact.residual(*refined, r).value();
    const double r_inf = inf_norm(r);
    const bool met = meets_relative_residual(a_inf, *refined, b, r, refinement_tolerance(n));
    std::cout << "refine(): ||r||_inf = " << r_inf << ", ||A||_inf = " << a_inf
              << ", tolerance met: " << met << ", error " << relative_error(*refined, reference) << '\n';

    // promoted_residual() is the same float-matrix residual that the loop uses in its first variant.
    std::vector<double> r_float(n);
    const double a32_inf = promoted_residual(a_f32, *refined, b, r_float).value();
    std::cout << "||float(A)||_inf = " << a32_inf << ", ||b - float(A) w||_inf = " << inf_norm(r_float) << '\n';
    ok = ok && met;
    // [end]
    return ok ? 0 : 1;
}
