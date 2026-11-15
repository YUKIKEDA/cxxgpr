# SPDX-License-Identifier: Apache-2.0
find_package(benchmark CONFIG REQUIRED)

add_executable(cxxgpr_bench
    exact_bench.cpp
)
cxxgpr_configure_target(cxxgpr_bench)
target_link_libraries(cxxgpr_bench PRIVATE
    cxxgpr::cxxgpr
    cxxgpr::eigen
    benchmark::benchmark
)
