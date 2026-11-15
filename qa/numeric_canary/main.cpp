// SPDX-License-Identifier: Apache-2.0
#include "cxxgpr/core/build_info.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <exception>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

void emit(std::string_view name, double value) {
    std::cout << std::format("{} {:a}\n", name, value);
}

void two_product() {
    constexpr double a = 1.0 + 0x1.0p-27;
    constexpr double b = 1.0 + 0x1.0p-26;
    const double product = a * b;
    const double err_fma = std::fma(a, b, -product);
    emit("two_product.product", product);
    emit("two_product.err_fma", err_fma);
}

void neumaier_sums() {
    constexpr std::array<double, 8> terms = {1e16, 1.0, 0.5, -1e16, 0.25, -1e-16, 0.125, 1e-16};
    double sum = 0.0;
    double comp = 0.0;
    for (const double x : terms) {
        const double t = sum + x;
        if (std::abs(sum) >= std::abs(x)) {
            comp += (sum - t) + x;
        } else {
            comp += (x - t) + sum;
        }
        sum = t;
    }
    emit("neumaier.sum", sum + comp);
}

void exact_gpr_cholesky() {
    constexpr Eigen::Index n = 4;
    const std::array<double, 4> xs = {-1.5, -0.4, 0.7, 1.8};
    const std::array<double, 4> y_vals = {0.9, -0.3, 0.6, -1.1};

    Eigen::MatrixXd cov(n, n);
    Eigen::VectorXd y(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        y(i) = y_vals[static_cast<std::size_t>(i)];
        for (Eigen::Index j = 0; j < n; ++j) {
            const double diff = xs[static_cast<std::size_t>(i)] - xs[static_cast<std::size_t>(j)];
            cov(i, j) = std::exp(-0.5 * diff * diff);
        }
        cov(i, i) += 1e-6;
    }

    const Eigen::LLT<Eigen::MatrixXd, Eigen::Lower> llt(cov);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("Eigen LLT decomposition failed in numeric canary");
    }

    const auto& l = llt.matrixL();
    double half_log_det = 0.0;
    for (Eigen::Index i = 0; i < n; ++i) {
        half_log_det += std::log(l.nestedExpression()(i, i));
    }
    const Eigen::VectorXd alpha = llt.solve(y);

    emit("eigen_llt.half_log_det", half_log_det);
    emit("eigen_llt.alpha0", alpha(0));
    emit("eigen_llt.alpha_norm2", alpha.squaredNorm());
}

void math_library() {
    emit("libm.sin", std::sin(1.2345678901234567));
    emit("libm.cos", std::cos(1.2345678901234567));
    emit("libm.exp", std::exp(-0.75));
    emit("libm.log", std::log(2.5));
    emit("libm.pow", std::pow(1.3, 4.2));
    emit("libm.sqrt", std::sqrt(5.0));
    emit("libm.erfc", std::erfc(1.5));
}

void prng_step() {
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    state ^= state >> 30U;
    state *= 0xbf58476d1ce4e5b9ULL;
    state ^= state >> 27U;
    state *= 0x94d049bb133111ebULL;
    state ^= state >> 31U;
    const std::uint64_t upper53 = state >> 11U;
    const double u01 = static_cast<double>(upper53) * 0x1.0p-53;
    emit("prng.splitmix_u01", u01);
    emit("prng.bitcast_check", std::bit_cast<double>(UINT64_C(0x3ff0000000000001)));
}

int run() {
    try {
        std::cout << "platform " << cxxgpr::build_info().platform_key << '\n';
        two_product();
        neumaier_sums();
        exact_gpr_cholesky();
        math_library();
        prng_step();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

} // namespace

int main() {
    try {
        return run();
    } catch (...) {
        return 1;
    }
}
