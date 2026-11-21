// SPDX-License-Identifier: Apache-2.0
// How-to: predict, compute joint covariances, draw posterior samples, and use leave-one-out.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cxxgpr/core/rng.hpp>
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
#include <cxxgpr/optimizer/optimizer.hpp>
#include <iomanip>
#include <iostream>
#include <vector>

int main()
{
    using namespace cxxgpr;
    std::cout << std::fixed << std::setprecision(4);
    bool ok = true;

    // Training data: 25 noisy samples of sin(x) on [0, 6].
    Xoshiro256PlusPlus noise_rng(7U);
    std::vector<double> x_train;
    std::vector<double> y_train;
    for (std::size_t i = 0; i < 25U; ++i)
    {
        const double x = 6.0 * static_cast<double>(i) / 24.0;
        x_train.push_back(x);
        y_train.push_back(std::sin(x) + 0.1 * noise_rng.unit_normal());
    }
    const std::size_t n = x_train.size();

    const auto kernel = KernelSpec(ConstantKernel::create(1.0).value()) * KernelSpec(RbfKernel::create(1.0).value());
    const auto trainer = Gpr<optimizer::Lbfgs>(kernel).with_noise_variance(0.05).value();
    const auto fitted = trainer.fit(x_train, n, 1U, y_train);
    if (!fitted)
    {
        std::cerr << fitted.error().message() << '\n';
        return 1;
    }
    std::cout << "fitted noise variance " << fitted->likelihood().noise_variance() << '\n';

    // [predict]
    const std::array<double, 4> x_query{0.5, 1.5, 3.0, 5.0};
    const auto pred = fitted->predict(x_query, 4U, 1U);  // span, n_rows, n_cols
    if (!pred)
    {
        std::cerr << pred.error().message() << '\n';
        return 1;
    }
    for (std::size_t i = 0; i < pred->mean.size(); ++i)
    {
        std::cout << "x = " << x_query[i] << "  mean = " << pred->mean[i] << "  sd = " << std::sqrt(pred->variance[i])
                  << "  sin(x) = " << std::sin(x_query[i]) << '\n';
    }
    // [end]
    for (std::size_t i = 0; i < 4U; ++i)
    {
        ok = ok && pred->variance[i] >= 0.0
            && std::abs(pred->mean[i] - std::sin(x_query[i])) < 3.0 * std::sqrt(pred->variance[i]);
    }

    // [variance-kind]
    const auto latent = fitted->predict(x_query, 4U, 1U, PredictOptions{.variance_kind = VarianceKind::Latent});
    std::cout << "observation - latent variance = " << pred->variance[0] - latent->variance[0] << " (noise variance "
              << fitted->likelihood().noise_variance() << ")\n";
    // [end]
    ok = ok && std::abs(pred->variance[0] - latent->variance[0] - fitted->likelihood().noise_variance()) < 1e-9;
    ok = ok && latent->variance_kind == VarianceKind::Latent && pred->variance_kind == VarianceKind::Observation;

    // [predict-into]
    Prediction<double> buffer;
    const MatrixRef<double> xq(x_query.data(), 4U, 1U, 4U);
    for (int round = 0; round < 3; ++round)  // after the first call no memory is allocated
    {
        if (auto r = fitted->predict_into(xq, buffer); !r)
        {
            std::cerr << r.error().message() << '\n';
            return 1;
        }
    }
    const auto with = fitted->predict_with(xq, PredictOptions{.variance_kind = VarianceKind::Latent});
    if (auto r = fitted->predict_with_into(xq, PredictOptions{.variance_kind = VarianceKind::Latent}, buffer); !r)
    {
        return 1;
    }
    // [end]
    ok = ok && buffer == *with && buffer.mean == latent->mean;

    // [covariance]
    const auto cov = fitted->predict_covariance(x_query, 4U, 1U, PredictOptions{.variance_kind = VarianceKind::Latent});
    if (!cov)
    {
        std::cerr << cov.error().message() << '\n';
        return 1;
    }
    // Column-major m x m matrix: element (row, col) is covariance[col * n_points + row].
    std::cout << "cov(0, 1) = " << cov->covariance[(1U * cov->n_points) + 0U] << ", var(0) = " << cov->covariance[0]
              << '\n';
    // [end]
    ok = ok && cov->n_points == 4U && cov->covariance.size() == 16U;
    for (std::size_t i = 0; i < 4U; ++i)
    {
        ok = ok && std::abs(cov->covariance[(i * 4U) + i] - latent->variance[i]) < 1e-9;
        for (std::size_t j = 0; j < i; ++j)
        {
            ok = ok && std::abs(cov->covariance[(j * 4U) + i] - cov->covariance[(i * 4U) + j]) < 1e-12;
        }
    }

    // [sample]
    constexpr std::size_t n_draws = 4000U;
    const auto draws =
        fitted->sample(x_query, 4U, 1U, n_draws, /*seed=*/123U, PredictOptions{.variance_kind = VarianceKind::Latent});
    if (!draws)
    {
        std::cerr << draws.error().message() << '\n';
        return 1;
    }
    // Column-major m x n_draws: draw j occupies draws[j * m .. j * m + m).
    double mean0 = 0.0;
    double sq0 = 0.0;
    for (std::size_t j = 0; j < n_draws; ++j)
    {
        const double v = (*draws)[(j * 4U) + 0U];
        mean0 += v;
        sq0 += v * v;
    }
    mean0 /= static_cast<double>(n_draws);
    const double var0 = (sq0 / static_cast<double>(n_draws)) - (mean0 * mean0);
    std::cout << "sample mean " << mean0 << " (posterior " << latent->mean[0] << "), sample variance " << var0
              << " (posterior " << latent->variance[0] << ")\n";
    // [end]
    ok = ok && std::abs(mean0 - latent->mean[0]) < 0.05
        && std::abs(var0 - latent->variance[0]) < 0.2 * latent->variance[0];

    // [seed]
    const auto same = fitted->sample(x_query, 4U, 1U, 8U, 5U);
    const auto again = fitted->sample(x_query, 4U, 1U, 8U, 5U);
    const auto other = fitted->sample(x_query, 4U, 1U, 8U, 6U);
    std::cout << "same seed identical: " << (*same == *again) << ", different seed identical: " << (*same == *other)
              << '\n';
    // [end]
    ok = ok && *same == *again && *same != *other;

    // [jitter]
    const auto strong = JitterPolicy::adaptive(1e-8, 10.0, 6U, 1e-3).value();
    const auto custom = fitted->sample(x_query, 4U, 1U, 8U, 5U, PredictOptions{}, strong);
    ok = ok && custom.has_value();
    // [end]

    // [loo]
    const auto loo = fitted->loo_predict();
    if (!loo)
    {
        std::cerr << loo.error().message() << '\n';
        return 1;
    }
    std::cout << "LOO for training point 10: mean " << loo->mean[10] << ", sd " << std::sqrt(loo->variance[10])
              << ", observed " << y_train[10] << '\n';
    // [end]

    // Cross-check: refit without point 10 at the same hyperparameters and predict it.
    {
        std::vector<double> xs = x_train;
        std::vector<double> ys = y_train;
        xs.erase(xs.begin() + 10);
        ys.erase(ys.begin() + 10);
        const auto frozen = Gpr<optimizer::Fixed>(fitted->kernel_spec()).with_likelihood(fitted->likelihood());
        const auto held_out = frozen.factor(xs, n - 1U, 1U, ys);
        const std::array<double, 1> xq10{x_train[10]};
        const auto p = held_out->predict(xq10, 1U, 1U);
        std::cout << "explicit refit without point 10: mean " << p->mean[0] << ", sd " << std::sqrt(p->variance[0])
                  << '\n';
        ok = ok && std::abs(p->mean[0] - loo->mean[10]) < 1e-6 && std::abs(p->variance[0] - loo->variance[10]) < 1e-6;
    }

    // [errors]
    const std::array<double, 4> wrong{0.0, 1.0, 2.0, 3.0};
    const auto bad = fitted->predict(wrong, 2U, 2U);  // model was trained on one feature
    std::cout << "two-column query: " << bad.error().message() << " (x_dim " << *bad.error().x_dim() << ", expected "
              << *bad.error().expected_dim() << ")\n";
    // [end]
    ok = ok && !bad && bad.error().code() == GprErrorCode::DimensionMismatch;

    // [rng]
    Xoshiro256PlusPlus rng(42U);
    const double u = rng.unit();             // uniform in [0, 1)
    const double g = rng.unit_normal();      // standard normal
    const std::size_t pick = rng.up_to(9U);  // uniform integer in [0, 9]
    const double o = rng.open_unit();        // uniform strictly inside (0, 1)
    const std::uint64_t w = rng.next_u64();  // raw 64-bit word
    SeededRng copy(42U);                     // SeededRng is an alias of Xoshiro256PlusPlus
    std::cout << "u = " << u << ", normal = " << g << ", pick = " << pick << ", open = " << o << ", word = " << w
              << ", fresh generator equals advanced one: " << (copy == rng) << '\n';
    // [end]
    ok = ok && u >= 0.0 && u < 1.0 && pick <= 9U && o > 0.0 && o < 1.0 && !(copy == rng);

    std::cout << (ok ? "OK" : "FAILED") << '\n';
    return ok ? 0 : 1;
}
