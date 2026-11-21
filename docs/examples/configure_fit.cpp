// SPDX-License-Identifier: Apache-2.0
// How-to: configure how a Gaussian process is trained (noise, jitter, precision, optimizer, buffers).
#include <cxxgpr/core/rng.hpp>
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
#include <cxxgpr/optimizer/optimizer.hpp>
#include <cxxgpr/transform/transform.hpp>

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace
{
    // 30 noisy samples of y = 5 + sin(x) on [0, 6]; the offset 5 motivates StandardizeTarget.
    struct Data
    {
        std::vector<double> x;
        std::vector<double> y;
    };

    Data make_data()
    {
        cxxgpr::Xoshiro256PlusPlus rng(2024U);
        Data d;
        for (std::size_t i = 0; i < 30U; ++i)
        {
            const double x = 6.0 * static_cast<double>(i) / 29.0;
            d.x.push_back(x);
            d.y.push_back(5.0 + std::sin(x) + 0.1 * rng.unit_normal());
        }
        return d;
    }
}  // namespace

int main()
{
    using namespace cxxgpr;
    std::cout << std::fixed << std::setprecision(4);
    const Data data = make_data();
    const MatrixRef<double> x(data.x.data(), data.x.size(), 1U, data.x.size());
    bool ok = true;

    // [kernel]
    const auto kernel = KernelSpec(ConstantKernel::create(1.0).value()) * KernelSpec(RbfKernel::create(1.0).value());
    // [end]

    // [builder]
    const auto base = Gpr<optimizer::Lbfgs>(kernel, optimizer::Lbfgs().with_max_iterations(200U))
                          .with_input_transform(transform::StandardizeInput())
                          .with_target_transform(transform::StandardizeTarget());
    // [end]

    // [noise]
    const auto with_noise = base.with_noise_variance(0.05);
    if (!with_noise)
    {
        std::cerr << with_noise.error().message() << '\n';
        return 1;
    }
    const auto bounded_noise = GaussianLikelihood::create(0.05).and_then(
        [](const GaussianLikelihood& lik) { return lik.with_bounds(Interval::positive(1e-4, 1.0).value()); }
    );
    if (!bounded_noise)
    {
        std::cerr << bounded_noise.error().message() << '\n';
        return 1;
    }
    const auto trainer = with_noise->with_likelihood(*bounded_noise);
    std::cout << "noise variance = " << trainer.likelihood().noise_variance() << " in ("
              << trainer.likelihood().bounds().lo() << ", " << trainer.likelihood().bounds().hi() << ")\n";
    // [end]

    // [reject]
    const auto bad_noise = base.with_noise_variance(-1.0);
    std::cout << "negative noise rejected: " << (!bad_noise && bad_noise.error().code() == GprErrorCode::InvalidNoiseVariance)
              << '\n';
    ok = ok && !bad_noise;
    // [end]

    // [defaults]
    std::cout << "default jitter kind is " << (base.jitter().is_adaptive() ? "adaptive" : "fixed") << ", kernel_exp is "
              << (base.kernel_exp() == KernelExp::Accurate ? "Accurate" : "FastApprox") << ", policy is "
              << (base.fit_policy() == FitPolicy::prefer_speed() ? "prefer_speed" : "other") << '\n';
    // [end]

    // [fit-factor]
    const auto fitted = trainer.fit(x, data.y);
    const auto factored = trainer.factor(x, data.y);
    if (!fitted || !factored)
    {
        std::cerr << "fit or factor failed\n";
        return 1;
    }
    std::cout << "factor: neg-MLL " << factored->neg_log_marginal_likelihood() << ", noise "
              << factored->likelihood().noise_variance() << ", iterations " << factored->optimizer_result().iterations
              << '\n';
    std::cout << "fit:    neg-MLL " << fitted->neg_log_marginal_likelihood() << ", noise "
              << fitted->likelihood().noise_variance() << ", iterations " << fitted->optimizer_result().iterations
              << ", converged " << fitted->optimizer_result().converged << '\n';
    // [end]
    ok = ok && fitted->neg_log_marginal_likelihood() < factored->neg_log_marginal_likelihood();
    ok = ok && std::abs(factored->likelihood().noise_variance() - 0.05) < 1e-12;
    // The data were generated with noise standard deviation 0.1 (variance 0.01) on a unit-variance
    // signal, so the optimized noise variance on standardized targets must be well below the start.
    ok = ok && fitted->likelihood().noise_variance() < 0.05;

    // [jitter]
    const auto jitter = JitterPolicy::adaptive(1e-10, 10.0, 8U, 1e-4);
    if (!jitter)
    {
        std::cerr << jitter.error().message() << '\n';
        return 1;
    }
    const auto jittered = trainer.with_jitter(*jitter).fit(x, data.y);
    std::cout << "accepted factor jitter: " << std::scientific << jittered->factor_jitter() << std::fixed << '\n';
    // [end]
    ok = ok && jittered.has_value();

    // [precision]
    const auto single = Gpr<optimizer::Lbfgs, SinglePrecision>(kernel).with_input_transform(transform::StandardizeInput())
                            .with_target_transform(transform::StandardizeTarget());
    const auto single_fit = single.fit(x, data.y);
    using Mixed = MixedPrecision<PromoteStorage>;
    const auto mixed = Gpr<optimizer::Lbfgs, Mixed>(kernel).with_input_transform(transform::StandardizeInput())
                           .with_target_transform(transform::StandardizeTarget());
    const auto mixed_fit = mixed.fit(x, data.y);
    if (!single_fit || !mixed_fit)
    {
        std::cerr << "precision fit failed\n";
        return 1;
    }
    std::cout << "single neg-MLL " << single_fit->neg_log_marginal_likelihood() << ", mixed neg-MLL "
              << mixed_fit->neg_log_marginal_likelihood() << ", refinement iterations "
              << mixed_fit->refinement_summary()->iterations << '\n';
    // [end]
    ok = ok && std::abs(single_fit->neg_log_marginal_likelihood() - mixed_fit->neg_log_marginal_likelihood()) < 0.5;

    // [optimizer]
    const auto restarts = trainer.with_optimizer(optimizer::Lbfgs().with_max_iterations(200U).with_restarts(3U, 42U));
    const auto restarted = restarts.fit(x, data.y);
    std::cout << "with 3 restarts: neg-MLL " << restarted->neg_log_marginal_likelihood() << '\n';
    // [end]
    ok = ok && restarted.has_value() && restarted->neg_log_marginal_likelihood() <= fitted->neg_log_marginal_likelihood() + 1e-6;

    // [policy]
    const auto lean = trainer.with_prefer_memory();
    const auto custom_policy = trainer.with_distance_cache_policy(DistanceCachePolicy::Cached)
                                   .with_cholesky_buffer(CholeskyBuffer::Reuse)
                                   .with_kernel_exp(KernelExp::FastApprox);
    const auto lean_fit = lean.fit(x, data.y);
    const auto custom_fit = custom_policy.fit(x, data.y);
    std::cout << "prefer_memory neg-MLL " << lean_fit->neg_log_marginal_likelihood() << ", custom policy neg-MLL "
              << custom_fit->neg_log_marginal_likelihood() << '\n';
    // [end]
    ok = ok && std::abs(lean_fit->neg_log_marginal_likelihood() - fitted->neg_log_marginal_likelihood()) < 1e-6;

    // [failure]
    std::vector<double> bad_y = data.y;
    bad_y[3] = std::numeric_limits<double>::quiet_NaN();
    const auto failed = trainer.fit(x, bad_y);
    std::cout << "bad targets: " << (failed ? "fitted" : std::string(failed.error().message())) << '\n';
    const auto again = trainer.fit(x, data.y);
    std::cout << "trainer still usable: " << again.has_value() << ", noise still "
              << trainer.likelihood().noise_variance() << '\n';
    // [end]
    ok = ok && !failed && failed.error().code() == GprErrorCode::NonFiniteInput && again.has_value();

    // [jitter-schedule]
    JitterSchedule schedule = jitter->retry_schedule();
    std::cout << "adaptive retry ladder:";
    std::size_t rungs = 0;
    while (const auto next = schedule.next())
    {
        std::cout << ' ' << std::scientific << *next;
        ++rungs;
    }
    const auto ada = jitter->as_adaptive();
    const auto fixed_policy = JitterPolicy::fixed(1e-8).value();
    std::cout << std::fixed << "\nmax_retries " << ada->max_retries() << ", multiplier " << ada->multiplier()
              << ", fixed retry " << std::scientific << fixed_policy.as_fixed()->jitter() << std::fixed
              << ", fixed has no adaptive part: " << !fixed_policy.as_adaptive().has_value() << '\n';
    // [end]
    ok = ok && rungs == 7U && jitter->is_adaptive() && fixed_policy.is_fixed()
        && jitter->kind() == JitterPolicy::Kind::Adaptive && FixedJitter{1e-8} == *fixed_policy.as_fixed()
        && AdaptiveJitter{1e-10, 10.0, 8U, 1e-4} == *ada;

    // [precision-helpers]
    static_assert(PrecisionPolicyConcept<DoublePrecision> && PrecisionPolicyConcept<Mixed>);
    static_assert(std::same_as<Mixed::Storage, float> && std::same_as<Mixed::Refine, double>);
    static_assert(Mixed::refines_in_f64 && !DoublePrecision::refines_in_f64);
    std::cout << to_precision_string(Mixed::persist_kind()) << " / " << to_residual_string(Mixed::persist_kind())
              << ", reevaluate variant: " << to_residual_string(MixedPrecision<ReevaluateKernel>::persist_kind())
              << ", tag name " << ReevaluateKernel::name() << '\n';
    const std::array<float, 3> weights_f{0.5F, 1.5F, -2.0F};
    std::array<double, 3> weights_d{};
    ok = ok && copy_storage_to_refine<Mixed>(weights_f, weights_d).has_value() && weights_d[2] == -2.0;
    const std::array<double, 4> cross{1.0, 2.0, 3.0, 4.0};  // 2 x 2 column-major k(X, X*)
    const std::array<double, 2> alpha{1.0, -1.0};
    std::array<double, 2> means{};
    ok = ok && compute_storage_means<DoublePrecision>(MatrixRef<double>(cross.data(), 2U, 2U), alpha, means).has_value();
    std::cout << "k^T alpha = " << means[0] << ", " << means[1] << '\n';
    // [end]
    ok = ok && means[0] == -1.0 && means[1] == -1.0;

    std::cout << (ok ? "OK" : "FAILED") << '\n';
    return ok ? 0 : 1;
}
