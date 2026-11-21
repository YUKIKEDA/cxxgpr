// SPDX-License-Identifier: Apache-2.0
// How-to: fit a Gaussian process on supplied squared distances (here the geodesic distance on a circle),
// predict with uncertainty, draw samples, cross-validate, and handle invalid tables.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cxxgpr/core/rng.hpp>
#include <cxxgpr/gpr/distance_gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
#include <cxxgpr/transform/transform.hpp>
#include <iostream>
#include <numbers>
#include <vector>

namespace
{
    // Squared geodesic (arc-length) distance between two angles on the unit circle.
    double squared_arc(double a, double b)
    {
        const double two_pi = 2.0 * std::numbers::pi;
        double diff = std::fmod(std::abs(a - b), two_pi);
        diff = std::min(diff, two_pi - diff);
        return diff * diff;
    }

    // Column-major table of squared arc lengths: entry (i, j) is the distance between rows[i] and cols[j].
    std::vector<double> arc_table(const std::vector<double>& rows, const std::vector<double>& cols)
    {
        std::vector<double> table(rows.size() * cols.size());
        for (std::size_t j = 0; j < cols.size(); ++j)
        {
            for (std::size_t i = 0; i < rows.size(); ++i)
            {
                table[(j * rows.size()) + i] = squared_arc(rows[i], cols[j]);
            }
        }
        return table;
    }

    double truth(double angle)
    {
        return std::sin(angle) + (0.4 * std::cos(2.0 * angle));
    }
}  // namespace

int main()
{
    // [data]
    // 24 noisy samples of a periodic function at angles spread over the circle, and 5 query angles.
    constexpr std::size_t n = 24;
    std::vector<double> angle(n);
    std::vector<double> y(n);
    cxxgpr::Xoshiro256PlusPlus rng(3);
    for (std::size_t i = 0; i < n; ++i)
    {
        angle[i] = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(n);
        y[i] = truth(angle[i]) + (0.05 * rng.unit_normal());
    }
    const std::vector<double> query_angle{0.3, 1.7, 3.1, 4.4, 6.0};
    const std::size_t m = query_angle.size();
    // [end]

    // [kernel]
    // k(a, b) = c * exp(-arc(a, b) / l): the exponential (Matern 1/2) kernel on the circle, using a table of
    // squared distances instead of coordinates. The slot ties a table to this kernel leaf.
    const cxxgpr::ScalarDistance arc_slot;
    const auto kernel = cxxgpr::DistanceKernel<cxxgpr::DistanceOnly>(cxxgpr::ConstantKernel::create(1.0).value())
        * arc_slot.matern(cxxgpr::MaternKernel::create(1.0, cxxgpr::MaternNu::OneHalf).value());
    // [end]

    // [fit]
    // The tables are bound to the slot and validated once: n x n for training, n x m from training to queries,
    // and m x m between queries (needed only for the joint covariance and for samples).
    const std::vector<double> train_table = arc_table(angle, angle);
    const std::vector<double> cross_table = arc_table(angle, query_angle);
    const std::vector<double> query_table = arc_table(query_angle, query_angle);
    cxxgpr::DistanceSource train;
    cxxgpr::DistanceSource cross;
    cxxgpr::DistanceSource query;
    train.bind_scalar(arc_slot.slot(), cxxgpr::MatrixRef<double>(train_table.data(), n, n));
    cross.bind_scalar(arc_slot.slot(), cxxgpr::MatrixRef<double>(cross_table.data(), n, m));
    query.bind_scalar(arc_slot.slot(), cxxgpr::MatrixRef<double>(query_table.data(), m, m));

    auto trainer = cxxgpr::DistanceGpr<cxxgpr::optimizer::Lbfgs>(kernel).with_noise_variance(0.01);
    if (!trainer)
    {
        std::cerr << trainer.error().message() << '\n';
        return 1;
    }
    auto model = trainer->fit(train, y);
    if (!model)
    {
        std::cerr << model.error().message() << '\n';
        return 1;
    }
    // [end]
    std::cout << "negative log marginal likelihood = " << model->neg_log_marginal_likelihood()
              << ", converged = " << model->optimizer_result().converged
              << ", noise variance = " << model->likelihood().noise_variance() << '\n';

    // [predict]
    auto pred = model->predict(cross);
    if (!pred)
    {
        std::cerr << pred.error().message() << '\n';
        return 1;
    }
    bool ok = true;
    for (std::size_t j = 0; j < m; ++j)
    {
        const double sd = std::sqrt(pred->variance[j]);
        std::cout << "angle " << query_angle[j] << ": mean = " << pred->mean[j] << ", std = " << sd
                  << ", truth = " << truth(query_angle[j]) << '\n';
        ok = ok && std::abs(pred->mean[j] - truth(query_angle[j])) < 4.0 * sd;
    }
    // [end]

    // [covariance]
    auto cov = model->predict_covariance(cross, query);
    auto draws = model->sample(cross, query, 3U, 42U);
    auto loo = model->loo_predict();
    if (!cov || !draws || !loo)
    {
        std::cerr << "covariance, sampling or leave-one-out failed\n";
        return 1;
    }
    // [end]
    double loo_rmse = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        loo_rmse += (loo->mean[i] - truth(angle[i])) * (loo->mean[i] - truth(angle[i]));
    }
    loo_rmse = std::sqrt(loo_rmse / static_cast<double>(n));
    std::cout << "covariance is " << cov->n_points << " x " << cov->n_points << ", " << draws->size() / m
              << " samples, leave-one-out RMSE = " << loo_rmse << '\n';
    ok = ok && cov->n_points == m && draws->size() == 3U * m && loo_rmse < 0.1;
    // The diagonal of the joint covariance equals the predictive variance.
    ok = ok && std::abs(cov->covariance[0] - pred->variance[0]) < 1e-9;

    // [errors]
    // A table that is not symmetric is rejected, unless the caller explicitly allows a repair.
    std::vector<double> skewed = train_table;
    skewed[1] += 1e-6;
    cxxgpr::DistanceSource strict;
    strict.bind_scalar(arc_slot.slot(), cxxgpr::MatrixRef<double>(skewed.data(), n, n));
    const auto rejected = trainer->fit(strict, y);
    cxxgpr::DistanceSource lenient;
    lenient.bind_scalar(arc_slot.slot(), cxxgpr::MatrixRef<double>(skewed.data(), n, n)).tidy(1e-3);
    const auto repaired = trainer->fit(lenient, y);
    // [end]
    std::cout << "asymmetric table: " << (rejected ? "accepted" : "rejected")
              << ", with tidy(): " << (repaired ? "accepted" : "rejected") << '\n';
    ok = ok && !rejected && rejected.error().code() == cxxgpr::GprErrorCode::InvalidDistance && repaired.has_value();
    return ok ? 0 : 1;
}
