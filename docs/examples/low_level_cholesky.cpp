// SPDX-License-Identifier: Apache-2.0
// Low-level linear algebra: in-place Cholesky, triangular solves, jitter retries, dense helpers
// and the deterministic parallel helpers.
#include <cxxgpr/linalg/cholesky.hpp>
#include <cxxgpr/linalg/dense.hpp>
#include <cxxgpr/linalg/jitter.hpp>
#include <cxxgpr/linalg/parallel.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace
{

    constexpr std::size_t n = 6;

    int fail(const char* what, const cxxgpr::GprError& error)
    {
        std::cerr << what << ": " << error.message() << '\n';
        return 1;
    }

}  // namespace

int main()
{
    using namespace cxxgpr;
    std::cout << std::setprecision(6);

    // A symmetric positive-definite matrix A = B B^T + I with a deterministic B, stored column-major.
    std::vector<double> a(n * n);
    for (std::size_t col = 0; col < n; ++col)
    {
        for (std::size_t row = 0; row < n; ++row)
        {
            double sum = (row == col) ? 1.0 : 0.0;
            for (std::size_t k = 0; k < 3; ++k)
            {
                const double b_rk = std::sin(static_cast<double>((row + 1) * (k + 2)));
                const double b_ck = std::sin(static_cast<double>((col + 1) * (k + 2)));
                sum += b_rk * b_ck;
            }
            a[(col * n) + row] = sum;
        }
    }
    const MatrixRef<double> a_full(a.data(), n, n);

    // [factor]
    // Factor a copy of the lower triangle. The strictly upper triangle is never read or written,
    // so a sentinel survives the call.
    std::vector<double> l(n * n, 777.0);
    if (auto r = linalg::copy_lower_triangle(a_full, MatrixMut<double>(l.data(), n, n)); !r)
    {
        return fail("copy_lower_triangle", r.error());
    }
    if (auto r = linalg::cholesky_lower_in_place(MatrixMut<double>(l.data(), n, n), CholeskyStage::Fit); !r)
    {
        return fail("cholesky_lower_in_place", r.error());
    }
    const MatrixRef<double> l_ref(l.data(), n, n);
    std::cout << "L(0,0) = " << l_ref(0, 0) << ", upper sentinel untouched: " << (l_ref(0, 1) == 777.0) << '\n';

    // Check L L^T = A on the lower triangle.
    double factor_error = 0.0;
    for (std::size_t col = 0; col < n; ++col)
    {
        for (std::size_t row = col; row < n; ++row)
        {
            double sum = 0.0;
            for (std::size_t k = 0; k <= col; ++k)
            {
                sum += l_ref(row, k) * l_ref(col, k);
            }
            factor_error = std::max(factor_error, std::abs(sum - a_full(row, col)));
        }
    }
    std::cout << "max |L L^T - A| = " << factor_error << '\n';
    if (factor_error > 1e-12)
    {
        return 1;
    }
    // [end]

    // [solve]
    std::vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        b[i] = static_cast<double>(i + 1);
    }
    std::vector<double> x = b;
    if (auto r = linalg::solve_llt_in_place(l_ref, std::span<double>(x)); !r)
    {
        return fail("solve_llt_in_place", r.error());
    }

    // Residual of A x = b through a column-major matrix-vector product.
    std::vector<double> ax(n);
    if (auto r = linalg::matvec_columns(a_full, std::span<const double>(x), std::span<double>(ax)); !r)
    {
        return fail("matvec_columns", r.error());
    }
    double residual = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        residual = std::max(residual, std::abs(ax[i] - b[i]));
    }
    std::cout << "max |A x - b| = " << residual << '\n';

    // The same solution in two triangular steps: L y = b, then L^T x = y.
    std::vector<double> two_step = b;
    if (!linalg::solve_lower_in_place(l_ref, std::span<double>(two_step))
        || !linalg::solve_lower_transpose_in_place(l_ref, std::span<double>(two_step)))
    {
        return 1;
    }
    std::cout << "two triangular solves equal solve_llt_in_place: " << (two_step == x) << '\n';

    // Several right-hand sides at once: solve for the identity to obtain A^-1 and read its diagonal.
    std::vector<double> inverse(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
    {
        inverse[(i * n) + i] = 1.0;
    }
    if (auto r = linalg::solve_llt_in_place(l_ref, MatrixMut<double>(inverse.data(), n, n)); !r)
    {
        return fail("solve_llt_in_place", r.error());
    }
    // [end]

    // [diag]
    std::vector<double> inv_diag(n);
    std::vector<double> scratch(n * n);
    if (auto r = linalg::inv_diag_from_chol_l(
            l_ref, std::span<double>(inv_diag), MatrixMut<double>(scratch.data(), n, n)
        );
        !r)
    {
        return fail("inv_diag_from_chol_l", r.error());
    }
    double diag_error = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        diag_error = std::max(diag_error, std::abs(inv_diag[i] - inverse[(i * n) + i]));
    }
    std::cout << "diag(A^-1) agrees with the explicit inverse to " << diag_error << '\n';

    const double log_det = linalg::log_det_from_l(l_ref).value();
    const double half_log_det = linalg::half_log_det_from_l(l_ref).value();
    double naive = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        naive += 2.0 * std::log(l_ref(i, i));
    }
    std::cout << "log|A| = " << log_det << ", 0.5 log|A| = " << half_log_det << ", naive sum = " << naive << '\n';
    if (residual > 1e-12 || diag_error > 1e-12 || std::abs(log_det - naive) > 1e-12 || two_step != x)
    {
        return 1;
    }
    // [end]

    // [unchecked]
    // After one successful validated call the factor is known to be sound, so the unchecked variants
    // skip validation inside a hot loop. They return no error and require valid views.
    std::vector<double> fast = b;
    linalg::solve_llt_in_place_unchecked(l_ref, std::span<double>(fast));
    std::cout << "unchecked solve identical: " << (fast == x) << '\n';
    if (fast != x)
    {
        return 1;
    }
    // [end]

    // [jitter]
    // A rank-one matrix is positive semi-definite, so an unjittered factorization breaks down.
    constexpr std::size_t k = 4;
    std::vector<double> ones(k * k, 1.0);
    std::vector<double> attempt = ones;
    const auto plain = linalg::cholesky_lower_in_place(MatrixMut<double>(attempt.data(), k, k), CholeskyStage::Fit);
    std::cout << "unjittered factorization of a rank-one matrix: "
              << (plain ? "succeeded" : std::string(plain.error().message())) << '\n';
    if (plain)
    {
        return 1;
    }

    // With an adaptive policy the factorization is retried on A + jI. A zero policy never retries.
    const auto policy = JitterPolicy::adaptive(1e-10, 10.0, 12, 1e-2).value();
    attempt = ones;
    auto used = linalg::cholesky_lower_with_jitter(MatrixMut<double>(attempt.data(), k, k), policy);
    if (!used)
    {
        return fail("cholesky_lower_with_jitter", used.error());
    }
    std::cout << "adaptive jitter policy succeeded with j = " << *used << '\n';

    // The zero-allocation overload takes a caller-owned backup of the lower triangle.
    std::vector<double> backup(k * k);
    attempt = ones;
    auto used_backup = linalg::cholesky_lower_with_jitter(
        MatrixMut<double>(attempt.data(), k, k), MatrixMut<double>(backup.data(), k, k), policy
    );
    if (!used_backup || *used_backup != *used)
    {
        return 1;
    }

    const auto none = JitterPolicy::fixed(0.0).value();
    attempt = ones;
    const auto refused = linalg::cholesky_lower_with_jitter(MatrixMut<double>(attempt.data(), k, k), none);
    std::cout << "fixed(0) policy: " << (refused ? "succeeded" : "CholeskyFailed")
              << ", error code matches: " << (!refused && refused.error().code() == GprErrorCode::CholeskyFailed)
              << '\n';
    if (refused)
    {
        return 1;
    }
    // [end]

    // [dense]
    // Dense helpers on the symmetric matrix A (only the lower triangle is read).
    std::vector<double> shifted = a;
    if (auto r = linalg::add_to_diag(MatrixMut<double>(shifted.data(), n, n), 0.5); !r)
    {
        return fail("add_to_diag", r.error());
    }
    // Mirror the lower triangle into the upper one so that the matrix is explicitly symmetric.
    std::vector<double> lower_only(n * n, 0.0);
    if (!linalg::copy_lower_triangle(MatrixRef<double>(shifted.data(), n, n), MatrixMut<double>(lower_only.data(), n, n))
        || !linalg::symmetrize_lower(MatrixMut<double>(lower_only.data(), n, n)))
    {
        return 1;
    }
    double asymmetry = 0.0;
    for (std::size_t col = 0; col < n; ++col)
    {
        for (std::size_t row = 0; row < n; ++row)
        {
            asymmetry = std::max(asymmetry, std::abs(lower_only[(col * n) + row] - shifted[(row * n) + col]));
        }
    }
    const double frob = linalg::frobenius_lower(
                            MatrixRef<double>(shifted.data(), n, n), MatrixRef<double>(shifted.data(), n, n)
    )
                            .value();
    double frob_full = 0.0;
    for (const double v : shifted)
    {
        frob_full += v * v;
    }
    std::cout << "asymmetry after symmetrize_lower = " << asymmetry << ", <A,A>_F = " << frob
              << " (full sum of squares " << frob_full << ")\n";
    if (asymmetry != 0.0 || std::abs(frob - frob_full) > 1e-10)
    {
        return 1;
    }
    // [end]

    // [parallel]
    // lower_area() counts lower-triangular entries; lower_block_start() splits columns into blocks
    // of near-equal area, which balances the work of lower-triangular loops.
    constexpr std::size_t order = 1000;
    std::cout << "lower_area(1000, 1000) = " << linalg::lower_area(order, order) << " (= 1000 * 1001 / 2)\n";
    std::cout << "4 area-balanced column blocks start at:";
    for (std::size_t block = 0; block <= 4; ++block)
    {
        std::cout << ' ' << linalg::lower_block_start(order, block, 4);
    }
    std::cout << '\n';

    // Visit every column exactly once, whatever the thread count. Each column writes only its own slot.
    std::vector<std::uint8_t> visited(order, 0);
    linalg::parallel_for_lower_blocks(
        order, 0,
        [&visited](std::size_t col_begin, std::size_t col_end)
        {
            for (std::size_t col = col_begin; col < col_end; ++col)
            {
                visited[col] += 1;
            }
        }
    );
    const bool once = std::ranges::all_of(visited, [](std::uint8_t v) { return v == 1; });
    std::cout << "every column visited exactly once: " << once << '\n';

    // A deterministic reduction: per-column values are summed in ascending column order, so the
    // result is bit-identical for any thread count.
    std::vector<double> partials(order);
    auto column_value = [](std::size_t col) { return 1.0 / static_cast<double>((col + 1) * (col + 1)); };
    const auto serial = linalg::parallel_lower_reduce_cols(order, 1, std::span<double>(partials), column_value);
    const auto wide = linalg::parallel_lower_reduce_cols(order, 4, std::span<double>(partials), column_value);
    const auto automatic = linalg::parallel_lower_reduce_cols(order, 0, std::span<double>(partials), column_value);
    if (!serial || !wide || !automatic)
    {
        return 1;
    }
    const bool identical = std::bit_cast<std::uint64_t>(*serial) == std::bit_cast<std::uint64_t>(*wide)
        && std::bit_cast<std::uint64_t>(*serial) == std::bit_cast<std::uint64_t>(*automatic);
    std::cout << "sum of 1/k^2 for k = 1..1000 = " << std::setprecision(12) << *serial
              << ", bit-identical for 1, 4 and automatic threads: " << identical << '\n';
    std::cout << "openmp_max_threads() >= 1: " << (linalg::openmp_max_threads() >= 1) << '\n';
    if (!once || !identical)
    {
        return 1;
    }
    // [end]
    return 0;
}
