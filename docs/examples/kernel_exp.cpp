// SPDX-License-Identifier: Apache-2.0
// How-to and explanation: the two exponential backends (KernelExp::Accurate and KernelExp::FastApprox),
// their measured error, and their effect on a Gram matrix and its gradient.
#include <cxxgpr/kernel/kernel.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace
{

    void check(const std::expected<void, cxxgpr::GprError>& result)
    {
        if (!result)
        {
            std::cerr << "error: " << result.error().message() << '\n';
            std::exit(1);
        }
    }

    bool all_ok = true;

    void expect(bool condition, std::string_view what)
    {
        all_ok = all_ok && condition;
        if (!condition)
        {
            std::cout << "CHECK FAILED: " << what << '\n';
        }
    }

}  // namespace

int main()
{
    using namespace cxxgpr;
    std::cout << std::scientific << std::setprecision(2);

    // [scalar]
    // Relative error of the fast backend against std::exp over the argument range kernels actually use.
    double worst_double = 0.0;
    float worst_float = 0.0F;
    for (int i = 0; i <= 20000; ++i)
    {
        const double x = -0.005 * i;  // 0 down to -100
        const double exact = std::exp(x);
        const double fast = eval_kernel_exp(x, KernelExp::FastApprox);
        worst_double = std::max(worst_double, std::abs(fast - exact) / exact);
        const auto xf = static_cast<float>(x);
        if (xf > -80.0F)  // float exp underflows to subnormals below about -87
        {
            const float exact_f = eval_kernel_exp(xf, KernelExp::Accurate);
            const float fast_f = eval_kernel_exp(xf, KernelExp::FastApprox);
            worst_float = std::max(worst_float, std::abs(fast_f - exact_f) / exact_f);
        }
    }
    // [end]
    std::cout << "max relative error of FastApprox on [-100, 0]: double " << worst_double << ", float " << worst_float
              << '\n';
    expect(worst_double < 1e-8 && worst_double > 1e-12, "double error is the truncation error of degree 7");
    expect(worst_float < 1e-6, "float error stays at single precision level");
    expect(eval_kernel_exp(-0.5, KernelExp::Accurate) == std::exp(-0.5), "Accurate is std::exp");

    // Special values are preserved.
    expect(eval_kernel_exp(-800.0, KernelExp::FastApprox) == 0.0, "underflow gives 0");
    expect(std::isinf(eval_kernel_exp(800.0, KernelExp::FastApprox)), "overflow gives +inf");
    expect(std::isnan(eval_kernel_exp(std::nan(""), KernelExp::FastApprox)), "NaN stays NaN");

    // [jet]
    // ExpJet carries exp(x) and its first two derivatives. Accurate returns the same value three times;
    // FastApprox returns the derivatives of its own polynomial, which differ slightly from exp(x).
    const ExpJet<double> accurate = eval_kernel_exp_jet(-0.3, KernelExp::Accurate);
    const ExpJet<double> fast = eval_kernel_exp_jet(-0.3, KernelExp::FastApprox);
    // [end]
    std::cout << "ExpJet at -0.3: Accurate v = " << accurate.v << " d1 = " << accurate.d1 << " d2 = " << accurate.d2
              << '\n';
    std::cout << "                FastApprox v - exp = " << fast.v - accurate.v << ", d1 - v = " << fast.d1 - fast.v
              << ", d2 - v = " << fast.d2 - fast.v << '\n';
    expect(accurate.v == accurate.d1 && accurate.v == accurate.d2, "Accurate jet");
    expect(fast.d1 != fast.v && std::abs(fast.d1 - fast.v) < 1e-5, "FastApprox derivative is the polynomial derivative");

    // [kernel]
    // The backend is a property of the compiled kernel (Gpr exposes the same switch as with_kernel_exp).
    const KernelSpec spec(RbfKernel::create(1.5).value());
    CompiledKernel<double> kernel = spec.compile<double>();
    std::cout << "default backend is Accurate: " << std::boolalpha << (kernel.kernel_exp() == KernelExp::Accurate)
              << '\n';
    // Eight points on a line and the Gram matrix with the accurate backend.
    const std::array<double, 8> x_data{0.0, 0.7, 1.1, 2.0, 2.4, 3.3, 4.1, 5.0};
    const MatrixRef<double> x(x_data.data(), 8, 1);
    std::array<double, 64> accurate_gram{};
    std::array<double, 64> fast_gram{};
    std::array<double, 64> scratch{};
    std::array<double, 64> grad_fast{};
    std::array<double, 64> grad_accurate{};
    std::array<double, 64> scratch_b{};
    const auto m = [](std::array<double, 64>& v) { return MatrixMut<double>(v.data(), 8, 8); };
    check(kernel.apply_points(x, m(accurate_gram), Triangle::Full, m(scratch)));
    check(kernel.grad_points(x, m(grad_accurate), 0, Triangle::Full, m(scratch), m(scratch_b)));

    kernel.set_kernel_exp(KernelExp::FastApprox);
    check(kernel.apply_points(x, m(fast_gram), Triangle::Full, m(scratch)));
    check(kernel.grad_points(x, m(grad_fast), 0, Triangle::Full, m(scratch), m(scratch_b)));
    // [end]

    double gram_diff = 0.0;
    double grad_diff = 0.0;
    for (std::size_t i = 0; i < 64; ++i)
    {
        gram_diff = std::max(gram_diff, std::abs(fast_gram[i] - accurate_gram[i]));
        grad_diff = std::max(grad_diff, std::abs(grad_fast[i] - grad_accurate[i]));
    }
    std::cout << "max |Gram(FastApprox) - Gram(Accurate)| = " << gram_diff << '\n';
    std::cout << "max |dK(FastApprox) - dK(Accurate)|     = " << grad_diff << '\n';
    expect(kernel.kernel_exp() == KernelExp::FastApprox, "backend switched");
    expect(gram_diff < 1e-8 && gram_diff > 0.0, "Gram error is bounded by the polynomial truncation error");

    // Self-consistency: under FastApprox the analytic gradient is the derivative of the *approximate* Gram matrix,
    // so a central finite difference of that matrix agrees with it more closely than with the accurate gradient.
    const double h = 1e-5;
    std::array<double, 1> theta{};
    auto probe = spec;
    check(probe.get_params(theta));
    std::array<double, 64> plus{};
    std::array<double, 64> minus{};
    const std::array<double, 1> up{theta[0] + h};
    const std::array<double, 1> down{theta[0] - h};
    auto fast_kernel = [&](const std::array<double, 1>& p, std::array<double, 64>& out) {
        check(probe.set_params(p));
        auto k = probe.compile<double>();
        k.set_kernel_exp(KernelExp::FastApprox);
        check(k.apply_points(x, m(out), Triangle::Full, m(scratch)));
    };
    fast_kernel(up, plus);
    fast_kernel(down, minus);
    double fd_vs_fast = 0.0;
    double fd_vs_accurate = 0.0;
    for (std::size_t i = 0; i < 64; ++i)
    {
        const double fd = (plus[i] - minus[i]) / (2.0 * h);
        fd_vs_fast = std::max(fd_vs_fast, std::abs(fd - grad_fast[i]));
        fd_vs_accurate = std::max(fd_vs_accurate, std::abs(fd - grad_accurate[i]));
    }
    std::cout << "finite difference vs FastApprox gradient:  " << fd_vs_fast << '\n';
    std::cout << "finite difference vs Accurate gradient:    " << fd_vs_accurate << '\n';
    expect(fd_vs_fast < 1e-8, "FastApprox gradient is consistent with the FastApprox Gram matrix");

    // Parameter transforms and custom leaves always use std::exp: only kernel evaluation changes.
    // Single precision uses the float polynomial.
    const auto kernel_f = spec.compile<float>();
    expect(kernel_f.kernel_exp() == KernelExp::Accurate, "every compiled kernel starts with the accurate backend");

    return all_ok ? 0 : 1;
}
