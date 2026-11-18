// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/core/build_info.hpp"
#include "cxxgpr/gpr/gpr.hpp"
#include "cxxgpr/kernel/kernel.hpp"

#include <benchmark/benchmark.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace
{

    struct ForresterDataset
    {
        std::vector<double> x_train;
        std::vector<double> y_train;
        std::vector<double> x_query;
        std::size_t n;
        std::size_t m;

        explicit ForresterDataset(std::size_t n_pts = 256U, std::size_t m_pts = 64U) :
            x_train(n_pts),
            y_train(n_pts),
            x_query(m_pts),
            n(n_pts),
            m(m_pts)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                const double x = static_cast<double>(i) / static_cast<double>(n - 1U);
                x_train[i] = x;
                const double t = (6.0 * x) - 2.0;
                y_train[i] = (0.1 * (t * t) * std::sin((12.0 * x) - 4.0)) + (0.01 * std::cos(40.0 * x));
            }
            for (std::size_t j = 0; j < m; ++j)
            {
                x_query[j] = (static_cast<double>(j) + 0.5) / static_cast<double>(m);
            }
        }
    };

    struct ArdSphereDataset
    {
        std::vector<double> x_train;
        std::vector<double> y_train;
        std::size_t n;
        std::size_t d;

        ArdSphereDataset(std::size_t n_pts = 128U, std::size_t d_dims = 8U) :
            x_train(n_pts * d_dims),
            y_train(n_pts),
            n(n_pts),
            d(d_dims)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                double s = 0.0;
                for (std::size_t dim = 0; dim < d; ++dim)
                {
                    const double v = std::sin(0.37 * static_cast<double>((i * (dim + 1U)) + dim));
                    x_train[(dim * n) + i] = v;
                    s += (static_cast<double>(dim + 1U) * 0.25) * v * v;
                }
                y_train[i] = s;
            }
        }
    };

    void bm_build_info(benchmark::State& state)
    {
        for (auto _ : state)
        {
            auto info = cxxgpr::build_info();
            benchmark::DoNotOptimize(info.platform_key);
        }
    }

    void bm_forrester_256_factor_double(benchmark::State& state)
    {
        const ForresterDataset data(256U, 64U);
        const auto x_ref = cxxgpr::MatrixRef<double>(data.x_train.data(), data.n, 1U, data.n);
        const auto spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.25).value());
        const auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(spec).with_noise_variance(1e-3).value();

        for (auto _ : state)
        {
            auto fit = trainer.factor(x_ref, data.y_train);
            benchmark::DoNotOptimize(fit);
        }
    }

    template <cxxgpr::PrecisionPolicyConcept P> void bm_forrester_256_fit_lbfgs(benchmark::State& state)
    {
        const ForresterDataset data(256U, 64U);
        const auto x_ref = cxxgpr::MatrixRef<double>(data.x_train.data(), data.n, 1U, data.n);
        const auto spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.25).value());
        const auto trainer =
            cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs, P>(spec, cxxgpr::optimizer::Lbfgs().with_max_iterations(20U))
                .with_noise_variance(1e-3)
                .value()
                .with_prefer_speed();

        for (auto _ : state)
        {
            auto fit = trainer.fit(x_ref, data.y_train);
            benchmark::DoNotOptimize(fit);
        }
    }

    void bm_forrester_256_predict_into_double(benchmark::State& state)
    {
        const ForresterDataset data(256U, 64U);
        const auto x_ref = cxxgpr::MatrixRef<double>(data.x_train.data(), data.n, 1U, data.n);
        const auto xq_ref = cxxgpr::MatrixRef<double>(data.x_query.data(), data.m, 1U, data.m);
        const auto spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.25).value());
        const auto fit = cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(spec)
                             .with_noise_variance(1e-3)
                             .value()
                             .factor(x_ref, data.y_train)
                             .value();
        cxxgpr::Prediction<double> pred;
        (void)fit.predict_into(xq_ref, pred);

        for (auto _ : state)
        {
            auto res = fit.predict_into(xq_ref, pred);
            benchmark::DoNotOptimize(res);
            benchmark::DoNotOptimize(pred.mean.data());
        }
    }

    void bm_ard_sphere_fit_lbfgs_double(benchmark::State& state)
    {
        const ArdSphereDataset data(128U, 8U);
        const auto x_ref = cxxgpr::MatrixRef<double>(data.x_train.data(), data.n, data.d, data.n);
        const std::vector<double> ls(data.d, 1.0);
        const auto spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(ls).value());
        const auto trainer =
            cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs>(spec, cxxgpr::optimizer::Lbfgs().with_max_iterations(20U))
                .with_noise_variance(1e-2)
                .value()
                .with_prefer_speed();

        for (auto _ : state)
        {
            auto fit = trainer.fit(x_ref, data.y_train);
            benchmark::DoNotOptimize(fit);
        }
    }

    void bm_ard_sphere_fit_fsa_cached(benchmark::State& state)
    {
        const ArdSphereDataset data(128U, 8U);
        const auto x_ref = cxxgpr::MatrixRef<double>(data.x_train.data(), data.n, data.d, data.n);
        const std::vector<double> ls(data.d, 1.0);
        const auto spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(ls).value());
        const auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::FastSimulatedAnnealing>(
            spec, cxxgpr::optimizer::FastSimulatedAnnealing().with_max_iterations(40U).with_seed(7ULL)
        )
                                 .with_noise_variance(1e-2)
                                 .value()
                                 .with_prefer_speed();

        for (auto _ : state)
        {
            auto fit = trainer.fit(x_ref, data.y_train);
            benchmark::DoNotOptimize(fit);
        }
    }

    template <cxxgpr::KernelScalarConcept T, cxxgpr::KernelExp ExpMode> void bm_kernel_exp_eval(benchmark::State& state)
    {
        constexpr std::size_t n = 256U;
        std::vector<T> x(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            x[i] = static_cast<T>(i) / static_cast<T>(n - 1U);
        }
        const auto x_ref = cxxgpr::MatrixRef<T>(x.data(), n, 1U, n);
        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.25).value())
            + cxxgpr::KernelSpec(cxxgpr::MaternKernel::create(0.4, cxxgpr::MaternNu::FiveHalves).value());
        auto ck = spec.compile<T>();
        ck.set_kernel_exp(ExpMode);

        std::vector<T> out(n * n, T{0});
        std::vector<T> scratch(n * n, T{0});
        const auto out_mut = cxxgpr::MatrixMut<T>(out.data(), n, n, n);
        const auto scratch_mut = cxxgpr::MatrixMut<T>(scratch.data(), n, n, n);
        auto nested = ck.make_nested_buffers(n, n);

        for (auto _ : state)
        {
            auto res = ck.apply_points(x_ref, out_mut, cxxgpr::Triangle::Lower, scratch_mut, nested);
            benchmark::DoNotOptimize(res);
            benchmark::DoNotOptimize(out.data());
        }
    }

    void bm_coord_apply_points_double(benchmark::State& state)
    {
        const ArdSphereDataset data(128U, 8U);
        const auto x_ref = cxxgpr::MatrixRef<double>(data.x_train.data(), data.n, data.d, data.n);
        const std::vector<double> ls(data.d, 1.2);
        const auto rbf_ard = cxxgpr::RbfArdKernel::create(ls).value();
        const auto m52 = cxxgpr::MaternKernel::create(1.5, cxxgpr::MaternNu::FiveHalves).value();
        const auto ck = (cxxgpr::KernelSpec(rbf_ard) + cxxgpr::KernelSpec(m52)).compile<double>();

        std::vector<double> out(data.n * data.n, 0.0);
        std::vector<double> scratch(data.n * data.n, 0.0);
        const auto out_mut = cxxgpr::MatrixMut<double>(out.data(), data.n, data.n, data.n);
        const auto scratch_mut = cxxgpr::MatrixMut<double>(scratch.data(), data.n, data.n, data.n);
        auto nested = ck.make_nested_buffers(data.n, data.n);

        for (auto _ : state)
        {
            auto res = ck.apply_points(x_ref, out_mut, cxxgpr::Triangle::Lower, scratch_mut, nested);
            benchmark::DoNotOptimize(res);
            benchmark::DoNotOptimize(out.data());
        }
    }

    void bm_distance_supply_apply_double(benchmark::State& state)
    {
        const ArdSphereDataset data(128U, 8U);
        const std::vector<double> ls(data.d, 1.2);
        const auto rbf_ard = cxxgpr::RbfArdKernel::create(ls).value();
        const auto m52 = cxxgpr::MaternKernel::create(1.5, cxxgpr::MaternNu::FiveHalves).value();

        cxxgpr::ScalarDistance s_slot;
        auto a_slot = cxxgpr::ArdDistance::create(data.d).value();
        const auto dist_spec = a_slot.rbf_ard(rbf_ard).value() + s_slot.matern(m52);
        const auto dist_ck = dist_spec.compile<double>();

        std::vector<double> d_scalar(data.n * data.n, 0.0);
        std::vector<std::vector<double>> d_ard(data.d, std::vector<double>(data.n * data.n, 0.0));
        std::vector<cxxgpr::MatrixRef<double>> ard_refs;
        ard_refs.reserve(data.d);
        for (std::size_t dim = 0; dim < data.d; ++dim)
        {
            for (std::size_t col = 0; col < data.n; ++col)
            {
                for (std::size_t row = 0; row < data.n; ++row)
                {
                    const double diff = data.x_train[(dim * data.n) + row] - data.x_train[(dim * data.n) + col];
                    const double sq = diff * diff;
                    d_ard[dim][(col * data.n) + row] = sq;
                    d_scalar[(col * data.n) + row] += sq;
                }
            }
            ard_refs.emplace_back(d_ard[dim].data(), data.n, data.n, data.n);
        }

        cxxgpr::DistanceSource src;
        src.bind_scalar(s_slot.slot(), cxxgpr::MatrixRef<double>(d_scalar.data(), data.n, data.n, data.n))
            .bind_ard(a_slot.slot(), ard_refs);
        const auto store_res = dist_ck.prepare_square(src);
        if (!store_res.has_value())
        {
            state.SkipWithError("prepare_square failed");
            return;
        }

        std::vector<double> out(data.n * data.n, 0.0);
        std::vector<double> scratch(data.n * data.n, 0.0);
        const auto out_mut = cxxgpr::MatrixMut<double>(out.data(), data.n, data.n, data.n);
        const auto scratch_mut = cxxgpr::MatrixMut<double>(scratch.data(), data.n, data.n, data.n);
        auto nested = dist_ck.make_nested_buffers(data.n, data.n);

        for (auto _ : state)
        {
            auto res = dist_ck.apply(*store_res, std::nullopt, out_mut, cxxgpr::Triangle::Lower, scratch_mut, nested);
            benchmark::DoNotOptimize(res);
            benchmark::DoNotOptimize(out.data());
        }
    }

    void bm_distance_supply_scalar_only_double(benchmark::State& state)
    {
        const ArdSphereDataset data(128U, 8U);
        const auto rbf = cxxgpr::RbfKernel::create(1.2).value();
        const auto m52 = cxxgpr::MaternKernel::create(1.5, cxxgpr::MaternNu::FiveHalves).value();

        cxxgpr::ScalarDistance s_slot;
        const auto dist_spec = s_slot.rbf(rbf) + s_slot.matern(m52);
        const auto dist_ck = dist_spec.compile<double>();

        std::vector<double> d_scalar(data.n * data.n, 0.0);
        for (std::size_t dim = 0; dim < data.d; ++dim)
        {
            for (std::size_t col = 0; col < data.n; ++col)
            {
                for (std::size_t row = 0; row < data.n; ++row)
                {
                    const double diff = data.x_train[(dim * data.n) + row] - data.x_train[(dim * data.n) + col];
                    d_scalar[(col * data.n) + row] += diff * diff;
                }
            }
        }

        cxxgpr::DistanceSource src;
        src.bind_scalar(s_slot.slot(), cxxgpr::MatrixRef<double>(d_scalar.data(), data.n, data.n, data.n));
        const auto store_res = dist_ck.prepare_square(src);
        if (!store_res.has_value())
        {
            state.SkipWithError("prepare_square failed");
            return;
        }

        std::vector<double> out(data.n * data.n, 0.0);
        std::vector<double> scratch(data.n * data.n, 0.0);
        const auto out_mut = cxxgpr::MatrixMut<double>(out.data(), data.n, data.n, data.n);
        const auto scratch_mut = cxxgpr::MatrixMut<double>(scratch.data(), data.n, data.n, data.n);
        auto nested = dist_ck.make_nested_buffers(data.n, data.n);

        for (auto _ : state)
        {
            auto res = dist_ck.apply(*store_res, std::nullopt, out_mut, cxxgpr::Triangle::Lower, scratch_mut, nested);
            benchmark::DoNotOptimize(res);
            benchmark::DoNotOptimize(out.data());
        }
    }

    BENCHMARK(bm_build_info);
    BENCHMARK(bm_forrester_256_factor_double);
    BENCHMARK_TEMPLATE(bm_forrester_256_fit_lbfgs, cxxgpr::DoublePrecision);
    BENCHMARK_TEMPLATE(bm_forrester_256_fit_lbfgs, cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>);
    BENCHMARK(bm_forrester_256_predict_into_double);
    BENCHMARK(bm_ard_sphere_fit_lbfgs_double);
    BENCHMARK(bm_ard_sphere_fit_fsa_cached);
    BENCHMARK_TEMPLATE(bm_kernel_exp_eval, double, cxxgpr::KernelExp::Accurate);
    BENCHMARK_TEMPLATE(bm_kernel_exp_eval, double, cxxgpr::KernelExp::FastApprox);
    BENCHMARK_TEMPLATE(bm_kernel_exp_eval, float, cxxgpr::KernelExp::Accurate);
    BENCHMARK_TEMPLATE(bm_kernel_exp_eval, float, cxxgpr::KernelExp::FastApprox);
    BENCHMARK(bm_coord_apply_points_double);
    BENCHMARK(bm_distance_supply_apply_double);
    BENCHMARK(bm_distance_supply_scalar_only_double);

}  // namespace

int main(int argc, char** argv)
{
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
