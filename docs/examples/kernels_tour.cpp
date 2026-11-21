// SPDX-License-Identifier: Apache-2.0
// How-to: create each of the ten built-in kernel leaves and check its value against a hand-written formula.
#include <array>
#include <cmath>
#include <cstdlib>
#include <cxxgpr/kernel/kernel.hpp>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

namespace
{

    // Unwrap a std::expected or stop the program with the library's error message.
    template <typename T> T unwrap(std::expected<T, cxxgpr::GprError> result)
    {
        if (!result)
        {
            std::cerr << "error: " << result.error().message() << '\n';
            std::exit(1);
        }
        return std::move(*result);
    }

    // Stop the program with the library's error message if an operation without a result failed.
    void check(const std::expected<void, cxxgpr::GprError>& result)
    {
        if (!result)
        {
            std::cerr << "error: " << result.error().message() << '\n';
            std::exit(1);
        }
    }

    // Two points in two dimensions, stored column-major (2 rows, 2 columns):
    //   p0 = (0.5, 1.0), p1 = (1.5, 3.0), so p1 - p0 = (1, 2) and |p1 - p0|^2 = 5.
    constexpr std::array<double, 4> k_points{0.5, 1.5, 1.0, 3.0};

    struct Pair
    {
        double diagonal;  // k(p0, p0)
        double off_diag;  // k(p0, p1)
    };

    // [evaluate-pair]
    // Evaluate one leaf on the two points above and return k(p0, p0) and k(p0, p1).
    Pair evaluate_pair(const cxxgpr::KernelLeaf& leaf)
    {
        const cxxgpr::KernelSpec spec(leaf);
        const auto compiled = spec.compile<double>();
        std::array<double, 4> gram{};
        std::array<double, 4> scratch{};
        const cxxgpr::MatrixRef<double> x(k_points.data(), 2, 2);
        const cxxgpr::MatrixMut<double> out(gram.data(), 2, 2);
        const cxxgpr::MatrixMut<double> work(scratch.data(), 2, 2);
        check(compiled.apply_points(x, out, cxxgpr::Triangle::Full, work));
        return {gram[0], gram[1]};  // out(0, 0) and out(1, 0) in column-major order
    }

    // [end]

    bool all_ok = true;

    void report(const std::string& name, const Pair& got, double expected_diag, double expected_off)
    {
        const bool ok = std::abs(got.diagonal - expected_diag) < 1e-12 && std::abs(got.off_diag - expected_off) < 1e-12;
        all_ok = all_ok && ok;
        std::cout << std::left << std::setw(28) << name << std::right << std::setw(12) << got.diagonal << std::setw(12)
                  << got.off_diag << std::setw(12) << expected_off << (ok ? "   ok" : "   MISMATCH") << '\n';
    }

}  // namespace

int main()
{
    using namespace cxxgpr;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << std::left << std::setw(28) << "kernel" << std::right << std::setw(12) << "k(p0,p0)" << std::setw(12)
              << "k(p0,p1)" << std::setw(12) << "expected" << '\n';

    const double r2 = 5.0;  // squared distance between the two points
    const double r = std::sqrt(r2);

    // [create-leaves]
    const auto rbf = unwrap(RbfKernel::create(1.5));
    const auto rbf_ard = unwrap(RbfArdKernel::create(std::array<double, 2>{1.0, 2.0}));
    const auto m12 = unwrap(MaternKernel::create(1.5, MaternNu::OneHalf));
    const auto m32 = unwrap(MaternKernel::create(1.5, MaternNu::ThreeHalves));
    const auto m52 = unwrap(MaternKernel::create(1.5, MaternNu::FiveHalves));
    const auto m52_ard = unwrap(MaternArdKernel::create(std::array<double, 2>{1.0, 2.0}, MaternNu::FiveHalves));
    const auto periodic = unwrap(PeriodicKernel::create(0.8, 3.0));
    const auto rq = unwrap(RationalQuadraticKernel::create(1.5, 2.0));
    const auto rq_ard = unwrap(RationalQuadraticArdKernel::create(std::array<double, 2>{1.0, 2.0}, 2.0));
    const auto constant = unwrap(ConstantKernel::create(2.0));
    const auto linear = unwrap(LinearKernel::create(0.5));
    const auto white = unwrap(WhiteKernel::create(0.1));
    // [end]

    // Squared-exponential: exp(-r^2 / (2 l^2)).
    report("RbfKernel", evaluate_pair(rbf), 1.0, std::exp(-r2 / (2.0 * 1.5 * 1.5)));

    // ARD: scaled squared distance s = (1/1)^2 + (2/2)^2 = 2, so exp(-s / 2) = exp(-1).
    report("RbfArdKernel", evaluate_pair(rbf_ard), 1.0, std::exp(-1.0));

    // Matern family with l = 1.5.
    report("MaternKernel (1/2)", evaluate_pair(m12), 1.0, std::exp(-r / 1.5));
    const double u3 = std::sqrt(3.0) * r / 1.5;
    report("MaternKernel (3/2)", evaluate_pair(m32), 1.0, (1.0 + u3) * std::exp(-u3));
    const double u5 = std::sqrt(5.0) * r / 1.5;
    report("MaternKernel (5/2)", evaluate_pair(m52), 1.0, (1.0 + u5 + u5 * u5 / 3.0) * std::exp(-u5));

    // ARD Matern-5/2: scaled distance sqrt(s) = sqrt(2).
    const double ua = std::sqrt(5.0) * std::sqrt(2.0);
    report("MaternArdKernel (5/2)", evaluate_pair(m52_ard), 1.0, (1.0 + ua + ua * ua / 3.0) * std::exp(-ua));

    // Periodic: exp(-2 sin^2(pi r / p) / l^2).
    const double s = std::sin(std::numbers::pi * r / 3.0);
    report("PeriodicKernel", evaluate_pair(periodic), 1.0, std::exp(-2.0 * s * s / (0.8 * 0.8)));

    // Rational quadratic: (1 + r^2 / (2 alpha l^2))^(-alpha).
    report("RationalQuadraticKernel", evaluate_pair(rq), 1.0, std::pow(1.0 + r2 / (2.0 * 2.0 * 1.5 * 1.5), -2.0));
    report("RationalQuadraticArdKernel", evaluate_pair(rq_ard), 1.0, std::pow(1.0 + 2.0 / (2.0 * 2.0), -2.0));

    // Constant: k = c everywhere. Linear: k = v * (p0 . p1) with p0 . p1 = 0.75 + 3 = 3.75 and p0 . p0 = 1.25.
    report("ConstantKernel", evaluate_pair(constant), 2.0, 2.0);
    report("LinearKernel", evaluate_pair(linear), 0.5 * 1.25, 0.5 * 3.75);

    // White noise: variance on the diagonal of the training Gram matrix, zero between distinct points.
    report("WhiteKernel", evaluate_pair(white), 0.1, 0.0);

    // Parameters live in log space and are read back through get_params().
    std::array<double, 2> theta{};
    check(periodic.get_params(theta));
    std::cout << "\nPeriodicKernel log-parameters: [" << theta[0] << ", " << theta[1] << "]  (log 0.8, log 3.0)\n";
    all_ok = all_ok && std::abs(theta[0] - std::log(0.8)) < 1e-15 && std::abs(theta[1] - std::log(3.0)) < 1e-15;

    // Every leaf exposes its open bounds; the default is (1e-5, 1e5).
    std::array<Interval, 2> bounds{Interval::default_positive(), Interval::default_positive()};
    check(periodic.bounds_into(bounds));
    std::cout << "default bounds of the period: (" << bounds[1].lo() << ", " << bounds[1].hi() << ")\n";

    // Values outside the interval are rejected with InvalidInterval.
    const auto bad = RbfKernel::create(1e6);
    std::cout << "RbfKernel::create(1e6) -> " << (bad ? "ok" : std::string(bad.error().message())) << '\n';
    all_ok = all_ok && !bad && bad.error().code() == GprErrorCode::InvalidInterval;

    // [custom-bounds]
    // Tighter bounds than the default keep an optimizer away from degenerate lengthscales.
    const Interval narrow = unwrap(Interval::create(0.1, 10.0));
    const auto bounded = unwrap(RbfKernel::create(1.0, narrow));
    const auto too_small = RbfKernel::create(0.05, narrow);  // 0.05 is outside (0.1, 10)
    // [end]
    std::cout << "bounded lengthscale in (" << bounded.interval().lo() << ", " << bounded.interval().hi()
              << "); create(0.05, narrow) -> " << (too_small ? "ok" : std::string(too_small.error().message())) << '\n';
    all_ok = all_ok && !too_small && too_small.error().code() == GprErrorCode::InvalidInterval;

    // A leaf can be updated in log space; a rejected update changes nothing.
    auto mutable_rbf = bounded;
    const std::array<double, 1> good{std::log(2.5)};
    const std::array<double, 1> bad_value{std::log(50.0)};
    check(mutable_rbf.set_params(good));
    const auto refused = mutable_rbf.set_params(bad_value);
    std::cout << "after set_params: lengthscale = " << mutable_rbf.lengthscale() << ", second update "
              << (refused ? "accepted" : "refused") << ", lengthscale still " << mutable_rbf.lengthscale() << '\n';
    all_ok = all_ok && std::abs(mutable_rbf.lengthscale() - 2.5) < 1e-12 && !refused;
    all_ok = all_ok && mutable_rbf != bounded && bounded == unwrap(RbfKernel::create(1.0, narrow));

    return all_ok ? 0 : 1;
}
