// SPDX-License-Identifier: Apache-2.0
// Zero-allocation prediction: warm up once, then predict_into() never touches the heap again.
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>

#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <new>
#include <span>
#include <vector>

// [counter]
// Replace the global allocation functions with counting versions. Because the library is linked
// statically, every std::vector growth inside it is counted too.
namespace
{
    std::atomic<std::size_t> g_allocations{0};
}

void* operator new(std::size_t size)
{
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(size == 0 ? 1 : size))
    {
        return p;
    }
    throw std::bad_alloc{};
}

void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
// [end]

namespace
{

    constexpr std::size_t n_train = 400;
    constexpr std::size_t n_features = 2;

    // FNV-1a over the bit patterns of the results: equal checksums mean bit-identical output.
    std::uint64_t checksum(const cxxgpr::Prediction<double>& p)
    {
        std::uint64_t h = 1469598103934665603ULL;
        for (const auto* v : {&p.mean, &p.variance})
        {
            for (const double x : *v)
            {
                h = (h ^ std::bit_cast<std::uint64_t>(x)) * 1099511628211ULL;
            }
        }
        return h;
    }

}  // namespace

int main()
{
    using namespace cxxgpr;

    // Training data: a smooth function of two features with a little noise, column-major (n x 2).
    Xoshiro256PlusPlus rng(5);
    std::vector<double> x(n_train * n_features);
    std::vector<double> y(n_train);
    for (std::size_t i = 0; i < n_train; ++i)
    {
        x[i] = 6.0 * rng.unit();
        x[n_train + i] = 6.0 * rng.unit();
        y[i] = std::sin(x[i]) + 0.5 * std::cos(2.0 * x[n_train + i]) + 0.05 * rng.unit_normal();
    }

    const auto spec = KernelSpec(ConstantKernel::create(1.0).value())
        * KernelSpec(RbfArdKernel::create(std::vector<double>{1.0, 1.0}).value());
    auto trainer = Gpr<optimizer::Lbfgs>(spec).with_target_transform(transform::StandardizeTarget());
    auto fitted = trainer.with_noise_variance(0.01).value().fit(x, n_train, n_features, y);
    if (!fitted)
    {
        std::cerr << fitted.error().message() << '\n';
        return 1;
    }

    // [warmup]
    // Query batches of 64 points (column-major, 64 x 2).
    auto make_queries = [&rng](std::size_t m)
    {
        std::vector<double> q(m * n_features);
        for (double& v : q)
        {
            v = 6.0 * rng.unit();
        }
        return q;
    };
    const std::vector<double> batch = make_queries(64);

    Prediction<double> out;  // reused for every call
    const std::size_t before_warmup = g_allocations;
    if (auto r = fitted->predict_into(batch, 64, n_features, out); !r)
    {
        std::cerr << r.error().message() << '\n';
        return 1;
    }
    const std::size_t warmup = g_allocations - before_warmup;
    std::cout << "warm-up call allocated " << warmup << " times\n";
    // [end]

    // [steady]
    const std::size_t before_steady = g_allocations;
    for (int repeat = 0; repeat < 1000; ++repeat)
    {
        if (!fitted->predict_into(batch, 64, n_features, out))
        {
            return 1;
        }
    }
    const std::size_t steady = g_allocations - before_steady;
    std::cout << "1000 further predict_into calls allocated " << steady << " times\n";

    // predict() builds a new Prediction each time, so its vectors are allocated again.
    const std::size_t before_predict = g_allocations;
    auto fresh = fitted->predict(batch, 64, n_features);
    std::cout << "one predict() call allocated " << (g_allocations - before_predict) << " times\n";
    // [end]

    // [checksum]
    // Both paths run the same code, so the results are bit-identical.
    if (!fresh)
    {
        return 1;
    }
    const bool same = checksum(out) == checksum(*fresh);
    std::cout << "predict_into equals predict: " << same << '\n';
    std::cout << "checksum of the 64-point prediction: " << std::hex << checksum(out) << std::dec << '\n';
    // [end]

    // [grow]
    // A larger batch grows the workspace once; the next identical call is allocation-free again.
    const std::vector<double> large = make_queries(256);
    const std::size_t before_grow = g_allocations;
    if (!fitted->predict_into(large, 256, n_features, out))
    {
        return 1;
    }
    const std::size_t grow = g_allocations - before_grow;
    const std::size_t before_again = g_allocations;
    if (!fitted->predict_into(large, 256, n_features, out))
    {
        return 1;
    }
    const std::size_t again = g_allocations - before_again;
    std::cout << "larger batch (256 points): first call " << grow << " allocations, second call " << again << '\n';
    // [end]

    const bool ok = warmup > 0 && steady == 0 && again == 0 && same;
    return ok ? 0 : 1;
}
