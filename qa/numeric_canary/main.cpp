// SPDX-License-Identifier: Apache-2.0
/// @file main.cpp
/// @brief Standalone IEEE-754 numeric canary verifying FMA, Neumaier sums, Eigen LLT, libm, and PRNG.
/// @details Emits hex-float (`%a`) probe outputs across arithmetic, linear algebra, transcendental
/// functions, and SplitMix64 operations so cross-platform baselines detect numerical drift.

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

namespace
{

    /// @brief Prints one named IEEE-754 probe value in exact C++ hex-float notation.
    /// @param name Probe identifier key.
    /// @param value Scalar floating-point result formatted with `std::format("{:a}")`.
    void emit(std::string_view name, double value)
    {
        std::cout << std::format("{} {:a}\n", name, value);
    }

    /// @brief Probes fused multiply-add error-free transformation `fma(a, b, -(a * b))`.
    /// @details Captures the exact rounding residual of a non-trivial binary64 product to verify
    /// hardware/software IEEE-754 FMA behavior.
    void two_product()
    {
        constexpr double a = 1.0 + 0x1.0p-27;
        constexpr double b = 1.0 + 0x1.0p-26;
        // Exact product residual must match 2^(-53) when FMA executes without intermediate rounding.
        const double product = a * b;
        const double err_fma = std::fma(a, b, -product);
        emit("two_product.product", product);
        emit("two_product.err_fma", err_fma);
    }

    /// @brief Probes Neumaier compensated summation across extreme dynamic range cancellation.
    /// @details Accumulates terms spanning +1e16, -1e16, and 1e-16 to confirm compensated rounding
    /// preserves small magnitude summands.
    void neumaier_sums()
    {
        constexpr std::array<double, 8> terms = {1e16, 1.0, 0.5, -1e16, 0.25, -1e-16, 0.125, 1e-16};
        double sum = 0.0;
        double comp = 0.0;
        // Track low-order rounding error separately whenever |sum| and |x| differ sharply.
        for (const double x : terms)
        {
            const double t = sum + x;
            if (std::abs(sum) >= std::abs(x))
            {
                comp += (sum - t) + x;
            }
            else
            {
                comp += (x - t) + sum;
            }
            sum = t;
        }
        emit("neumaier.sum", sum + comp);
    }

    /// @brief Probes Eigen lower Cholesky factorization, log-determinant, and solve on a 4x4 RBF matrix.
    /// @details Forms an isotropic RBF Gram matrix with 1e-6 diagonal jitter, computes Eigen::LLT,
    /// and emits half_log_det, alpha(0), and squared alpha norm as hex-float canary digests.
    void exact_gpr_cholesky()
    {
        constexpr Eigen::Index n = 4;
        const std::array<double, 4> xs = {-1.5, -0.4, 0.7, 1.8};
        const std::array<double, 4> y_vals = {0.9, -0.3, 0.6, -1.1};

        // Populate dense 4x4 RBF Gram matrix with fixed 1e-6 diagonal jitter.
        Eigen::MatrixXd cov(n, n);
        Eigen::VectorXd y(n);
        for (Eigen::Index i = 0; i < n; ++i)
        {
            y(i) = y_vals[static_cast<std::size_t>(i)];
            for (Eigen::Index j = 0; j < n; ++j)
            {
                const double diff = xs[static_cast<std::size_t>(i)] - xs[static_cast<std::size_t>(j)];
                cov(i, j) = std::exp(-0.5 * diff * diff);
            }
            cov(i, i) += 1e-6;
        }

        const Eigen::LLT<Eigen::MatrixXd, Eigen::Lower> llt(cov);
        if (llt.info() != Eigen::Success)
        {
            throw std::runtime_error("Eigen LLT decomposition failed in numeric canary");
        }

        // Accumulate 0.5 * log|A| = sum_i log(L_ii) and solve A * alpha = y.
        const auto& l = llt.matrixL();
        double half_log_det = 0.0;
        for (Eigen::Index i = 0; i < n; ++i)
        {
            half_log_det += std::log(l.nestedExpression()(i, i));
        }
        const Eigen::VectorXd alpha = llt.solve(y);

        emit("eigen_llt.half_log_det", half_log_det);
        emit("eigen_llt.alpha0", alpha(0));
        emit("eigen_llt.alpha_norm2", alpha.squaredNorm());
    }

    /// @brief Probes standard libm transcendental functions at fixed binary64 inputs.
    /// @details Emits sin, cos, exp, log, pow, sqrt, and erfc results in exact hex-float format.
    void math_library()
    {
        // Evaluate standard transcendental primitives used across kernels, bounds, and MLL.
        emit("libm.sin", std::sin(1.2345678901234567));
        emit("libm.cos", std::cos(1.2345678901234567));
        emit("libm.exp", std::exp(-0.75));
        emit("libm.log", std::log(2.5));
        emit("libm.pow", std::pow(1.3, 4.2));
        emit("libm.sqrt", std::sqrt(5.0));
        emit("libm.erfc", std::erfc(1.5));
    }

    /// @brief Probes deterministic SplitMix64 bit mixing, [0, 1) conversion, and std::bit_cast.
    /// @details Verifies 64-bit unsigned mixing shifts, upper-53-bit double conversion, and exact
    /// IEEE-754 representation of `1.0 + 2^(-52)` via std::bit_cast.
    void prng_step()
    {
        // Execute one SplitMix64 mixing step from golden-ratio state seed and project to [0, 1).
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

    /// @brief Executes all numeric canary probes and prints platform build metadata.
    /// @details Emits platform_key and uncommitted_changes status followed by arithmetic, LLT,
    /// libm, and PRNG probes, catching standard exceptions to report failures cleanly.
    /// @return Zero on success, or 1 when an exception is thrown.
    int run()
    {
        try
        {
            // Print environment provenance header before emitting hex-float probe results.
            const auto info = cxxgpr::build_info();
            std::cout << "platform " << info.platform_key << '\n';
            std::cout << "uncommitted_changes " << (info.uncommitted_changes ? "true" : "false") << '\n';
            two_product();
            neumaier_sums();
            exact_gpr_cholesky();
            math_library();
            prng_step();
            return 0;
        }
        catch (const std::exception& error)
        {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }

}  // namespace

/// @brief Program entrypoint invoking the standalone numeric canary probe suite.
/// @details Wraps run() in a catch-all exception guard so non-standard exceptions exit cleanly.
/// @return Exit code 0 on success or 1 on error.
int main()
{
    try
    {
        // Delegate all probe execution and standard-exception logging to run().
        return run();
    }
    catch (...)
    {
        return 1;
    }
}
