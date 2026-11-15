// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/core/build_info.hpp"

#include <benchmark/benchmark.h>

namespace {

void bm_build_info(benchmark::State& state) {
    for (auto _ : state) {
        auto info = cxxgpr::build_info();
        benchmark::DoNotOptimize(info.platform_key);
    }
}

BENCHMARK(bm_build_info);

} // namespace

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
