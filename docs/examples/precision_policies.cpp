// SPDX-License-Identifier: Apache-2.0
// Compare the four precision policies and the fast exponential on one problem.
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace
{

    constexpr std::size_t n = 200;  // training points
    constexpr std::size_t m = 50;   // query points

    // Everything the comparison needs from one fitted model, widened to double.
    struct Outcome
    {
        std::string name;
        std::size_t factor_bytes = 0;
        std::vector<double> alpha;
        std::vector<double> mean;
        std::vector<double> variance;
        std::string refinement = "none";
    };

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

    // [fit]
    // Factor the system at fixed hyperparameters under policy P, so that the only difference between
    // runs is the arithmetic. optimizer::Fixed performs no optimization iterations.
    template <typename P>
    bool run(
        std::string name, const cxxgpr::KernelSpec& spec, cxxgpr::KernelExp exp_mode, std::span<const double> x,
        std::span<const double> y, std::span<const double> xq, Outcome& out
    )
    {
        using namespace cxxgpr;
        auto trainer = Gpr<optimizer::Fixed, P>(spec).with_noise_variance(2e-5);
        if (!trainer)
        {
            return false;
        }
        auto fitted = trainer->with_kernel_exp(exp_mode).factor(x, n, 1, y);
        if (!fitted)
        {
            std::cerr << name << ": " << fitted.error().message() << '\n';
            return false;
        }
        auto prediction = fitted->predict(xq, m, 1);
        if (!prediction)
        {
            return false;
        }
        out.name = std::move(name);
        out.factor_bytes = n * n * sizeof(typename P::Storage);
        out.alpha.assign(fitted->alpha().begin(), fitted->alpha().end());
        out.mean.assign(prediction->mean.begin(), prediction->mean.end());
        out.variance.assign(prediction->variance.begin(), prediction->variance.end());
        if (const auto& summary = fitted->refinement_summary(); summary)
        {
            out.refinement = std::string(to_string(summary->status)) + ", " + std::to_string(summary->iterations)
                + " corrections";
        }
        return true;
    }
    // [end]

    double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b)
    {
        double worst = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            worst = std::max(worst, std::abs(a[i] - b[i]));
        }
        return worst;
    }

    double max_abs(const std::vector<double>& a)
    {
        double worst = 0.0;
        for (const double v : a)
        {
            worst = std::max(worst, std::abs(v));
        }
        return worst;
    }

}  // namespace

int main()
{
    using namespace cxxgpr;

    // A smooth signal with small noise: the Gram matrix is badly conditioned, which is where
    // single precision starts to lose accuracy.
    std::vector<double> x(n);
    std::vector<double> y(n);
    std::vector<double> xq(m);
    Xoshiro256PlusPlus rng(11);
    for (std::size_t i = 0; i < n; ++i)
    {
        x[i] = 3.0 * static_cast<double>(i) / static_cast<double>(n - 1);
        y[i] = std::sin(2.0 * x[i]) + 0.01 * rng.unit_normal();
    }
    for (std::size_t j = 0; j < m; ++j)
    {
        xq[j] = 0.03 + 2.94 * static_cast<double>(j) / static_cast<double>(m - 1);
    }
    const auto spec = KernelSpec(ConstantKernel::create(1.0).value()) * KernelSpec(RbfKernel::create(2.0).value());

    Outcome reference;
    Outcome single;
    Outcome promote;
    Outcome reeval;
    Outcome fast;
    const KernelExp accurate = KernelExp::Accurate;
    if (!run<DoublePrecision>("DoublePrecision", spec, accurate, x, y, xq, reference)
        || !run<SinglePrecision>("SinglePrecision", spec, accurate, x, y, xq, single)
        || !run<MixedPrecision<PromoteStorage>>("Mixed<PromoteStorage>", spec, accurate, x, y, xq, promote)
        || !run<MixedPrecision<ReevaluateKernel>>("Mixed<ReevaluateKernel>", spec, accurate, x, y, xq, reeval)
        || !run<DoublePrecision>("Double + FastApprox", spec, KernelExp::FastApprox, x, y, xq, fast))
    {
        return 1;
    }

    std::cout << std::left << std::setw(26) << "policy" << std::right << std::setw(12) << "factor bytes"
              << std::setw(14) << "alpha error" << std::setw(12) << "mean error" << std::setw(12) << "var error"
              << "  refinement\n";
    std::cout << std::setprecision(2) << std::scientific;
    for (const Outcome* o : {&reference, &single, &promote, &reeval, &fast})
    {
        std::cout << std::left << std::setw(26) << o->name << std::right << std::setw(12) << o->factor_bytes
                  << std::setw(14) << max_abs_diff(o->alpha, reference.alpha) / max_abs(reference.alpha)
                  << std::setw(12) << max_abs_diff(o->mean, reference.mean) << std::setw(12)
                  << max_abs_diff(o->variance, reference.variance) << "  " << o->refinement << '\n';
    }

    // The refined weights are orders of magnitude closer to the double solution than single precision.
    const double single_alpha = max_abs_diff(single.alpha, reference.alpha) / max_abs(reference.alpha);
    const double reeval_alpha = max_abs_diff(reeval.alpha, reference.alpha) / max_abs(reference.alpha);
    const double promote_alpha = max_abs_diff(promote.alpha, reference.alpha) / max_abs(reference.alpha);
    const bool ok = reeval_alpha < 1e-4 && promote_alpha < 1e-4 && single_alpha > 100.0 * reeval_alpha
        && max_abs_diff(reeval.mean, reference.mean) < max_abs_diff(single.mean, reference.mean)
        && max_abs_diff(fast.mean, reference.mean) < 1e-4;
    return ok ? 0 : 1;
}
