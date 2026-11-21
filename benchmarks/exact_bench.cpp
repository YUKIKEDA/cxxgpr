// SPDX-License-Identifier: Apache-2.0
/// @file exact_bench.cpp
/// @brief Google Benchmark suite for Exact GPR fit, factor, prediction, kernel exp, and distance paths.
/// @details Measures build_info overhead, Forrester 1D factorization/L-BFGS/predict_into,
/// repeated predict_into on fitted and loaded models across training and query sizes,
/// 8D ARD sphere L-BFGS and FSA cached optimization, Accurate vs FastApprox KernelExp evaluation,
/// and coordinate vs supplied-distance kernel Gram evaluation.

#include "cxxgpr/core/build_info.hpp"
#include "cxxgpr/gpr/gpr.hpp"
#include "cxxgpr/kernel/kernel.hpp"
#include "cxxgpr/persist/persist.hpp"

#include <benchmark/benchmark.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace
{

    /// @brief Precomputed 1D Forrester + high-frequency perturbation benchmark dataset.
    /// @details Generates n_pts uniformly spaced training points on [0, 1] and m_pts shifted
    /// midpoint query coordinates for Exact GPR factor, fit, and prediction microbenchmarks.
    struct ForresterDataset
    {
        /// @brief Training input coordinates on [0, 1] of length n.
        std::vector<double> x_train;
        /// @brief Training target values evaluated at x_train of length n.
        std::vector<double> y_train;
        /// @brief Query coordinates on (0, 1) of length m.
        std::vector<double> x_query;
        /// @brief Number of training points.
        std::size_t n;
        /// @brief Number of query points.
        std::size_t m;

        /// @brief Populates Forrester training targets and midpoint query grid.
        /// @param[in] n_pts Number of training observations generated on [0, 1].
        /// @param[in] m_pts Number of query coordinates generated at cell midpoints.
        explicit ForresterDataset(std::size_t n_pts = 256U, std::size_t m_pts = 64U) :
            x_train(n_pts),
            y_train(n_pts),
            x_query(m_pts),
            n(n_pts),
            m(m_pts)
        {
            // Sample smooth Forrester baseline with small high-frequency cosine perturbation.
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

    /// @brief Precomputed multi-dimensional anisotropic quadratic sphere benchmark dataset.
    /// @details Synthesizes column-major n_pts x d_dims inputs via deterministic trigonometric
    /// projections and weighted squared-norm targets for ARD kernel and optimizer benchmarks.
    struct ArdSphereDataset
    {
        /// @brief Column-major n x d training input matrix.
        std::vector<double> x_train;
        /// @brief Anisotropic weighted sphere target vector of length n.
        std::vector<double> y_train;
        /// @brief Number of training points.
        std::size_t n;
        /// @brief Input feature dimensionality.
        std::size_t d;

        /// @brief Populates deterministic column-major 8D ARD inputs and weighted sphere targets.
        /// @param[in] n_pts Number of training observations.
        /// @param[in] d_dims Input feature dimensionality.
        ArdSphereDataset(std::size_t n_pts = 128U, std::size_t d_dims = 8U) :
            x_train(n_pts * d_dims),
            y_train(n_pts),
            n(n_pts),
            d(d_dims)
        {
            // Generate column-major inputs with dimension-dependent quadratic target curvature.
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

    /// @brief Benchmarks cxxgpr::build_info() metadata lookup overhead.
    /// @param[in,out] state Google Benchmark iteration state.
    void bm_build_info(benchmark::State& state)
    {
        for (auto _ : state)
        {
            auto info = cxxgpr::build_info();
            benchmark::DoNotOptimize(info.platform_key);
        }
    }

    /// @brief Benchmarks fixed-hyperparameter Cholesky factorization on n = 256 Forrester data.
    /// @param[in,out] state Google Benchmark iteration state.
    void bm_forrester_256_factor_double(benchmark::State& state)
    {
        const ForresterDataset data(256U, 64U);
        const auto x_ref = cxxgpr::MatrixRef<double>(data.x_train.data(), data.n, 1U, data.n);
        const auto spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.25).value());
        const auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(spec).with_noise_variance(1e-3).value();

        // Measure repeated Cholesky factorization without hyperparameter optimization.
        for (auto _ : state)
        {
            auto fit = trainer.factor(x_ref, data.y_train);
            benchmark::DoNotOptimize(fit);
        }
    }

    /// @brief Benchmarks L-BFGS hyperparameter fitting on n = 256 Forrester data for precision P.
    /// @tparam P Precision policy type (DoublePrecision or MixedPrecision).
    /// @param[in,out] state Google Benchmark iteration state.
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

        // Measure up to 20 L-BFGS steps under prefer_speed() workspace policy.
        std::size_t last_iters = 0U;
        for (auto _ : state)
        {
            auto fit = trainer.fit(x_ref, data.y_train);
            if (fit.has_value())
            {
                last_iters = fit->optimizer_result().iterations;
            }
            benchmark::DoNotOptimize(fit);
        }
        state.counters["iters"] = static_cast<double>(last_iters);
    }

    /// @brief Benchmarks fixed-hyperparameter Cholesky factorization on n = 512 Forrester data.
    /// @param[in,out] state Google Benchmark iteration state.
    void bm_forrester_512_factor_double(benchmark::State& state)
    {
        const ForresterDataset data(512U, 64U);
        const auto x_ref = cxxgpr::MatrixRef<double>(data.x_train.data(), data.n, 1U, data.n);
        const auto spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.25).value());
        const auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(spec).with_noise_variance(1e-3).value();

        // Measure repeated Cholesky factorization for n = 512 without hyperparameter optimization.
        for (auto _ : state)
        {
            auto fit = trainer.factor(x_ref, data.y_train);
            benchmark::DoNotOptimize(fit);
        }
    }

    /// @brief Benchmarks fixed-hyperparameter Cholesky factorization on n = 1024 Forrester data.
    /// @param[in,out] state Google Benchmark iteration state.
    void bm_forrester_1024_factor_double(benchmark::State& state)
    {
        const ForresterDataset data(1024U, 64U);
        const auto x_ref = cxxgpr::MatrixRef<double>(data.x_train.data(), data.n, 1U, data.n);
        const auto spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.25).value());
        const auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(spec).with_noise_variance(1e-3).value();

        // Measure large-scale O(n^3) Cholesky factorization for n = 1024.
        for (auto _ : state)
        {
            auto fit = trainer.factor(x_ref, data.y_train);
            benchmark::DoNotOptimize(fit);
        }
    }

    /// @brief Benchmarks fixed-hyperparameter Cholesky factorization on n = 2048 Forrester data.
    /// @param[in,out] state Google Benchmark iteration state.
    void bm_forrester_2048_factor_double(benchmark::State& state)
    {
        const ForresterDataset data(2048U, 64U);
        const auto x_ref = cxxgpr::MatrixRef<double>(data.x_train.data(), data.n, 1U, data.n);
        const auto spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.25).value());
        const auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(spec).with_noise_variance(1e-3).value();

        // Measure very large-scale O(n^3) Cholesky factorization for n = 2048.
        for (auto _ : state)
        {
            auto fit = trainer.factor(x_ref, data.y_train);
            benchmark::DoNotOptimize(fit);
        }
    }

    /// @brief Benchmarks L-BFGS hyperparameter fitting on n = 512 Forrester data for precision P.
    /// @tparam P Precision policy type (DoublePrecision or MixedPrecision).
    /// @param[in,out] state Google Benchmark iteration state.
    template <cxxgpr::PrecisionPolicyConcept P> void bm_forrester_512_fit_lbfgs(benchmark::State& state)
    {
        const ForresterDataset data(512U, 64U);
        const auto x_ref = cxxgpr::MatrixRef<double>(data.x_train.data(), data.n, 1U, data.n);
        const auto spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.25).value());
        const auto trainer =
            cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs, P>(spec, cxxgpr::optimizer::Lbfgs().with_max_iterations(10U))
                .with_noise_variance(1e-3)
                .value()
                .with_prefer_speed();

        // Measure up to 10 L-BFGS iterations at n = 512 and record iteration count.
        std::size_t last_iters = 0U;
        for (auto _ : state)
        {
            auto fit = trainer.fit(x_ref, data.y_train);
            if (fit.has_value())
            {
                last_iters = fit->optimizer_result().iterations;
            }
            benchmark::DoNotOptimize(fit);
        }
        state.counters["iters"] = static_cast<double>(last_iters);
    }

    /// @brief Benchmarks full m x m posterior predict_covariance() on n = 512 training and m = 256 query points.
    /// @param[in,out] state Google Benchmark iteration state.
    void bm_forrester_512_predict_cov_double(benchmark::State& state)
    {
        const ForresterDataset data(512U, 256U);
        const auto x_ref = cxxgpr::MatrixRef<double>(data.x_train.data(), data.n, 1U, data.n);
        const auto xq_ref = cxxgpr::MatrixRef<double>(data.x_query.data(), data.m, 1U, data.m);
        const auto spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.25).value());
        auto fit_res =
            cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(spec).with_noise_variance(1e-3).value().factor(x_ref, data.y_train);
        if (!fit_res.has_value())
        {
            state.SkipWithError("factor failed");
            return;
        }
        const auto& fit = *fit_res;

        // Measure V = L^-1 K_* and Sigma_* = K_** - V^T V at n = 512, m = 256.
        for (auto _ : state)
        {
            auto cov = fit.predict_covariance(xq_ref);
            benchmark::DoNotOptimize(cov);
        }
    }

    /// @brief Benchmarks zero-allocation predict_into() on n = 256 training and m = 64 query points.
    /// @param[in,out] state Google Benchmark iteration state.
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
        // Warm up Prediction output buffers outside the timed loop.
        cxxgpr::Prediction<double> pred;
        (void)fit.predict_into(xq_ref, pred);

        for (auto _ : state)
        {
            auto res = fit.predict_into(xq_ref, pred);
            benchmark::DoNotOptimize(res);
            benchmark::DoNotOptimize(pred.mean.data());
        }
    }

    /// @brief Fitted and loaded n x 3 Constant * Rbf models shared by the repeated-prediction benchmarks.
    /// @details Each training size is factored once; the loaded model is saved with its factor and loaded back,
    /// so a DoublePrecision factor is memory-mapped. The destructor releases the maps before deleting the files.
    struct RepeatedPredictModels
    {
        /// @brief Fixed-hyperparameter fitted model per training size n.
        std::map<std::size_t, cxxgpr::FittedGpr<cxxgpr::optimizer::Fixed, cxxgpr::DoublePrecision>> fitted;
        /// @brief Model loaded from save_with_factor() output per training size n.
        std::map<std::size_t, cxxgpr::persist::LoadedGpr> loaded;
        /// @brief Directories holding the saved models, removed on destruction.
        std::vector<std::filesystem::path> dirs;

        /// @brief Constructs an empty model cache.
        RepeatedPredictModels() = default;
        /// @brief Copying is disabled because the cache owns temporary directories.
        RepeatedPredictModels(const RepeatedPredictModels&) = delete;
        /// @brief Copy assignment is disabled because the cache owns temporary directories.
        /// @return Never returns.
        RepeatedPredictModels& operator=(const RepeatedPredictModels&) = delete;
        /// @brief Moving is disabled because the cache is a function-local singleton.
        RepeatedPredictModels(RepeatedPredictModels&&) = delete;
        /// @brief Move assignment is disabled because the cache is a function-local singleton.
        /// @return Never returns.
        RepeatedPredictModels& operator=(RepeatedPredictModels&&) = delete;

        /// @brief Unmaps the loaded models, then deletes their saved files.
        ~RepeatedPredictModels()
        {
            loaded.clear();
            for (const auto& dir : dirs)
            {
                std::error_code ec;
                std::filesystem::remove_all(dir, ec);
            }
        }

        /// @brief Returns the cache, factoring and saving the model for n on first use.
        /// @param[in] n Training sample count.
        /// @return The cache, or nullptr when factoring, saving, or loading failed.
        static RepeatedPredictModels* ensure(std::size_t n)
        {
            static RepeatedPredictModels models;
            if (models.fitted.contains(n))
            {
                return &models;
            }
            const ArdSphereDataset data(n, 3U);
            const cxxgpr::MatrixRef<double> x(data.x_train.data(), data.n, data.d, data.n);
            const auto spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
                * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.25).value());
            auto fit =
                cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(spec).with_noise_variance(1e-3).value().factor(x, data.y_train);
            if (!fit)
            {
                return nullptr;
            }
            auto dir = std::filesystem::temp_directory_path() / ("cxxgpr_bench_loaded_" + std::to_string(n));
            std::error_code ec;
            std::filesystem::remove_all(dir, ec);
            std::filesystem::create_directories(dir, ec);
            models.dirs.push_back(dir);
            if (!cxxgpr::persist::save_with_factor(*fit, dir))
            {
                return nullptr;
            }
            auto loaded = cxxgpr::persist::LoadedGpr::load(dir);
            if (!loaded)
            {
                return nullptr;
            }
            models.fitted.emplace(n, std::move(*fit));
            models.loaded.emplace(n, std::move(*loaded));
            return &models;
        }
    };

    /// @brief Times repeated predict_into() (mean and diagonal variance) on a warmed-up model.
    /// @details Uses n x 3 sphere inputs with a Constant * Rbf kernel and m query points offset from the training
    /// inputs; the first call outside the timed loop grows the query workspace and output buffers.
    /// @tparam Loaded Times LoadedGpr (memory-mapped factor) when true and FittedGpr otherwise.
    /// @param[in,out] state Benchmark state with training size n and query count m.
    template <bool Loaded> void bm_repeated_predict_into(benchmark::State& state)
    {
        const auto n = static_cast<std::size_t>(state.range(0));
        const auto m = static_cast<std::size_t>(state.range(1));
        auto* models = RepeatedPredictModels::ensure(n);
        if (models == nullptr)
        {
            state.SkipWithError("model setup failed");
            return;
        }
        std::vector<double> xq(m * 3U);
        for (std::size_t j = 0; j < m; ++j)
        {
            for (std::size_t dim = 0; dim < 3U; ++dim)
            {
                xq[(dim * m) + j] = std::sin((0.53 * static_cast<double>((j * (dim + 1U)) + dim)) + 0.5);
            }
        }
        const cxxgpr::MatrixRef<double> xq_ref(xq.data(), m, 3U, m);
        cxxgpr::Prediction<double> pred;
        const auto predict = [&]()
        {
            if constexpr (Loaded)
            {
                return models->loaded.at(n).predict_into(xq_ref, pred);
            }
            else
            {
                return models->fitted.at(n).predict_into(xq_ref, pred);
            }
        };
        if (!predict())
        {
            state.SkipWithError("warm-up prediction failed");
            return;
        }
        for (auto _ : state)
        {
            auto res = predict();
            benchmark::DoNotOptimize(res);
            benchmark::DoNotOptimize(pred.variance.data());
        }
    }

    /// @brief Benchmarks 8D ARD Constant * RbfArd fitting with L-BFGS on n = 128 points.
    /// @param[in,out] state Google Benchmark iteration state.
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

        // Measure 9-parameter ARD MLL + gradient optimization via L-BFGS.
        for (auto _ : state)
        {
            auto fit = trainer.fit(x_ref, data.y_train);
            benchmark::DoNotOptimize(fit);
        }
    }

    /// @brief Benchmarks 32D ARD Constant * RbfArd fitting with L-BFGS on n = 256 points.
    /// @param[in,out] state Google Benchmark iteration state.
    void bm_ard_sphere_highdim_fit_lbfgs_double(benchmark::State& state)
    {
        const ArdSphereDataset data(256U, 32U);
        const auto x_ref = cxxgpr::MatrixRef<double>(data.x_train.data(), data.n, data.d, data.n);
        const std::vector<double> ls(data.d, 1.0);
        const auto spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(ls).value());
        const auto trainer =
            cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs>(spec, cxxgpr::optimizer::Lbfgs().with_max_iterations(10U))
                .with_noise_variance(1e-2)
                .value()
                .with_prefer_speed();

        // Measure 33-parameter 32D ARD MLL + gradient optimization via L-BFGS.
        std::size_t last_iters = 0U;
        for (auto _ : state)
        {
            auto fit = trainer.fit(x_ref, data.y_train);
            if (fit.has_value())
            {
                last_iters = fit->optimizer_result().iterations;
            }
            benchmark::DoNotOptimize(fit);
        }
        state.counters["iters"] = static_cast<double>(last_iters);
    }

    /// @brief Benchmarks 8D ARD fitting with FastSimulatedAnnealing and incremental LeafCache.
    /// @param[in,out] state Google Benchmark iteration state.
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

        // Measure seeded FSA steps exploiting LeafCache and cached distances.
        for (auto _ : state)
        {
            auto fit = trainer.fit(x_ref, data.y_train);
            benchmark::DoNotOptimize(fit);
        }
    }

    /// @brief Benchmarks lower-triangle Gram evaluation for scalar type T and KernelExp mode.
    /// @tparam T Kernel evaluation scalar type (float or double).
    /// @tparam ExpMode Exponential implementation policy (Accurate or FastApprox).
    /// @param[in,out] state Google Benchmark iteration state.
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

        // Preallocate output, scratch, and nested buffers so loop timing isolates kernel math.
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

    /// @brief Benchmarks coordinate-based RbfArd + Matern-5/2 lower-triangle Gram evaluation.
    /// @param[in,out] state Google Benchmark iteration state.
    void bm_coord_apply_points_double(benchmark::State& state)
    {
        const ArdSphereDataset data(128U, 8U);
        const auto x_ref = cxxgpr::MatrixRef<double>(data.x_train.data(), data.n, data.d, data.n);
        const std::vector<double> ls(data.d, 1.2);
        const auto rbf_ard = cxxgpr::RbfArdKernel::create(ls).value();
        const auto m52 = cxxgpr::MaternKernel::create(1.5, cxxgpr::MaternNu::FiveHalves).value();
        const auto ck = (cxxgpr::KernelSpec(rbf_ard) + cxxgpr::KernelSpec(m52)).compile<double>();

        // Preallocate matrices and nested buffers before timing coordinate evaluation.
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

    /// @brief Times lower Gram assembly and complete fixed-parameter training on identical inputs.
    /// @tparam Factor Selects complete training when true and Gram assembly otherwise.
    /// @param[in,out] state Benchmark state with training size and feature count.
    template <bool Factor> void bm_training_coordinates(benchmark::State& state)
    {
        const ArdSphereDataset data(static_cast<std::size_t>(state.range(0)), static_cast<std::size_t>(state.range(1)));
        const cxxgpr::MatrixRef<double> x(data.x_train.data(), data.n, data.d, data.n);
        const auto spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(0.25).value());
        if constexpr (Factor)
        {
            const auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(spec).with_noise_variance(1e-3).value();
            for (auto _ : state)
            {
                auto model = trainer.factor(x, data.y_train);
                if (!model)
                {
                    state.SkipWithError("factor failed");
                    break;
                }
                benchmark::DoNotOptimize(model);
            }
        }
        else
        {
            const auto kernel = spec.compile<double>();
            std::vector<double> gram(data.n * data.n);
            std::vector<double> scratch(data.n * data.n);
            const cxxgpr::MatrixMut<double> out(gram.data(), data.n, data.n, data.n);
            const cxxgpr::MatrixMut<double> tmp(scratch.data(), data.n, data.n, data.n);
            auto nested = kernel.make_nested_buffers(data.n, data.n);
            for (auto _ : state)
            {
                if (!kernel.apply_points(x, out, cxxgpr::Triangle::Lower, tmp, nested))
                {
                    state.SkipWithError("Gram evaluation failed");
                    break;
                }
                benchmark::ClobberMemory();
            }
        }
    }

    /// @brief Benchmarks supplied-distance ARD + scalar DistanceKernel lower-triangle Gram evaluation.
    /// @param[in,out] state Google Benchmark iteration state.
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

        // Precompute scalar and per-dimension squared distances outside the timed region.
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

    /// @brief Benchmarks supplied-distance scalar-only DistanceKernel lower-triangle Gram evaluation.
    /// @param[in,out] state Google Benchmark iteration state.
    void bm_distance_supply_scalar_only_double(benchmark::State& state)
    {
        const ArdSphereDataset data(128U, 8U);
        const auto rbf = cxxgpr::RbfKernel::create(1.2).value();
        const auto m52 = cxxgpr::MaternKernel::create(1.5, cxxgpr::MaternNu::FiveHalves).value();

        cxxgpr::ScalarDistance s_slot;
        const auto dist_spec = s_slot.rbf(rbf) + s_slot.matern(m52);
        const auto dist_ck = dist_spec.compile<double>();

        // Precompute scalar squared Euclidean distances once before benchmarking apply().
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
    BENCHMARK(bm_forrester_512_factor_double);
    BENCHMARK(bm_forrester_1024_factor_double);
    BENCHMARK(bm_forrester_2048_factor_double);
    BENCHMARK_TEMPLATE(bm_forrester_512_fit_lbfgs, cxxgpr::DoublePrecision);
    BENCHMARK_TEMPLATE(bm_forrester_512_fit_lbfgs, cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>);
    BENCHMARK(bm_forrester_512_predict_cov_double);
    BENCHMARK(bm_ard_sphere_fit_lbfgs_double);
    BENCHMARK(bm_ard_sphere_highdim_fit_lbfgs_double);
    BENCHMARK(bm_ard_sphere_fit_fsa_cached);
    BENCHMARK_TEMPLATE(bm_kernel_exp_eval, double, cxxgpr::KernelExp::Accurate);
    BENCHMARK_TEMPLATE(bm_kernel_exp_eval, double, cxxgpr::KernelExp::FastApprox);
    BENCHMARK_TEMPLATE(bm_kernel_exp_eval, float, cxxgpr::KernelExp::Accurate);
    BENCHMARK_TEMPLATE(bm_kernel_exp_eval, float, cxxgpr::KernelExp::FastApprox);
    BENCHMARK(bm_coord_apply_points_double);
    BENCHMARK_TEMPLATE(bm_training_coordinates, false)
        ->ArgsProduct({
            {1024, 2048, 4096},
            {8, 64}
    })
        ->UseRealTime();
    BENCHMARK_TEMPLATE(bm_training_coordinates, true)
        ->ArgsProduct({
            {1024, 2048, 4096},
            {8, 64}
    })
        ->UseRealTime();
    BENCHMARK(bm_distance_supply_apply_double);
    BENCHMARK(bm_distance_supply_scalar_only_double);
    BENCHMARK_TEMPLATE(bm_repeated_predict_into, false)
        ->ArgsProduct({
            {256, 1024, 4096},
            {  1,    8,   64}
    })
        ->UseRealTime();
    BENCHMARK_TEMPLATE(bm_repeated_predict_into, true)
        ->ArgsProduct({
            {256, 1024, 4096},
            {  1,    8,   64}
    })
        ->UseRealTime();

}  // namespace

/// @brief Initializes Google Benchmark and executes all registered Exact GPR benchmarks.
/// @param[in,out] argc Command-line argument count forwarded to benchmark::Initialize().
/// @param[in,out] argv Command-line argument vector forwarded to benchmark::Initialize().
/// @return Zero exit status after benchmark shutdown.
int main(int argc, char** argv)
{
    // Forward CLI flags such as --benchmark_filter and --benchmark_out to Google Benchmark.
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
