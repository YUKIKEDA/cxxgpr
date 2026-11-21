// SPDX-License-Identifier: Apache-2.0
// How-to: write a custom kernel leaf. The kernel k(d2) = 1 / (1 + d2 / (2 l^2)) is a rational quadratic with
// alpha = 1, so the library's own RationalQuadraticKernel serves as an independent reference for the
// value, the gradient and the Hessian.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cxxgpr/kernel/kernel.hpp>
#include <cxxgpr/persist/persist.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

namespace
{

    template <typename T> T unwrap(std::expected<T, cxxgpr::GprError> result)
    {
        if (!result)
        {
            std::cerr << "error: " << result.error().message() << '\n';
            std::exit(1);
        }
        return std::move(*result);
    }

    void check(const std::expected<void, cxxgpr::GprError>& result)
    {
        if (!result)
        {
            std::cerr << "error: " << result.error().message() << '\n';
            std::exit(1);
        }
    }

    // [term]
    // k(d2) = 1 / (1 + d2 / (2 l^2)), parameterised by the single log-hyperparameter theta = log(l).
    template <cxxgpr::KernelScalarConcept T> class InverseQuadraticTerm final : public cxxgpr::KernelTerm<T>
    {
    public:
        static constexpr double k_lo = 0.05;  // open bounds of the lengthscale
        static constexpr double k_hi = 50.0;

        explicit InverseQuadraticTerm(double lengthscale) : lengthscale_(lengthscale) {}

        [[nodiscard]] std::unique_ptr<cxxgpr::KernelTerm<T>> clone() const override
        {
            return std::make_unique<InverseQuadraticTerm<T>>(lengthscale_);
        }

        [[nodiscard]] std::size_t num_params() const noexcept override { return 1; }

        [[nodiscard]] std::expected<void, cxxgpr::GprError> get_params(std::span<double> out) const override
        {
            if (out.size() != 1)
            {
                return std::unexpected(cxxgpr::GprError::length_mismatch("out must have one element"));
            }
            out[0] = std::log(lengthscale_);
            return {};
        }

        // Validate everything first and change state last, so a failed update leaves the term untouched.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> set_params(std::span<const double> params) override
        {
            if (params.size() != 1)
            {
                return std::unexpected(cxxgpr::GprError::length_mismatch("params must have one element"));
            }
            if (!std::isfinite(params[0]))
            {
                return std::unexpected(cxxgpr::GprError::non_finite_input());
            }
            const double value = std::exp(params[0]);
            if (!(value > k_lo && value < k_hi))
            {
                return std::unexpected(cxxgpr::GprError::invalid_interval_out_of_range(value, k_lo, k_hi));
            }
            lengthscale_ = value;
            return {};
        }

        [[nodiscard]] std::expected<void, cxxgpr::GprError> bounds_into(std::span<cxxgpr::Interval> out) const override
        {
            if (out.size() != 1)
            {
                return std::unexpected(cxxgpr::GprError::length_mismatch("out must have one element"));
            }
            auto interval = cxxgpr::Interval::create(k_lo, k_hi);
            if (!interval)
            {
                return std::unexpected(interval.error());
            }
            out[0] = *interval;
            return {};
        }

        [[nodiscard]] std::expected<void, cxxgpr::GprError> apply(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> out, cxxgpr::Triangle uplo
        ) const override
        {
            fill(dist, out, uplo, [this](T w) { return T{1} / (T{1} + w); });
            return {};
        }

        [[nodiscard]] std::expected<void, cxxgpr::GprError> apply_cross(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> out
        ) const override
        {
            return apply(dist, out, cxxgpr::Triangle::Full);
        }

        // The prior variance k(x, x) is 1 for every point because d2 = 0 on the diagonal.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> fill_diag(std::span<T> out) const override
        {
            std::ranges::fill(out, T{1});
            return {};
        }

        // dk/dtheta with w = d2 / (2 l^2) and dw/dtheta = -2 w:  dk/dtheta = 2 w / (1 + w)^2.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> grad(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> d_k, std::size_t param_idx, cxxgpr::Triangle uplo
        ) const override
        {
            if (param_idx != 0)
            {
                return std::unexpected(cxxgpr::GprError::index_out_of_range("param_idx must be 0"));
            }
            fill(dist, d_k, uplo, [](T w) { return (T{2} * w) / ((T{1} + w) * (T{1} + w)); });
            return {};
        }

        // d2k/dtheta2 = 4 w (w - 1) / (1 + w)^3.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> hess(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> d2_k, std::size_t i, std::size_t j, cxxgpr::Triangle uplo
        ) const override
        {
            if (i != 0 || j != 0)
            {
                return std::unexpected(cxxgpr::GprError::index_out_of_range("i and j must be 0"));
            }
            fill(dist, d2_k, uplo, [](T w) {
                return (T{4} * w * (w - T{1})) / ((T{1} + w) * (T{1} + w) * (T{1} + w));
            });
            return {};
        }

        // Optional: a non-empty identifier lets the kernel be saved and loaded.
        [[nodiscard]] std::string_view persist_id() const noexcept override { return "example.inverse_quadratic"; }

        [[nodiscard]] std::expected<std::string, cxxgpr::GprError> persist_state_json() const override
        {
            return "{\"lengthscale\":" + std::to_string(lengthscale_) + "}";
        }

    private:
        // Applies f(w) with w = d2 / (2 l^2) to the requested triangle of dist.
        template <typename F>
        void fill(cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> out, cxxgpr::Triangle uplo, F f) const
        {
            const T inv = T{1} / static_cast<T>(2.0 * lengthscale_ * lengthscale_);
            for (std::size_t col = 0; col < dist.cols(); ++col)
            {
                for (std::size_t row = 0; row < dist.rows(); ++row)
                {
                    if ((uplo == cxxgpr::Triangle::Lower && row < col)
                        || (uplo == cxxgpr::Triangle::Upper && row > col))
                    {
                        continue;
                    }
                    const T d2 = std::max(dist(row, col), T{0});  // squared distances can round to -0
                    out(row, col) = f(d2 * inv);
                }
            }
        }

        double lengthscale_;
    };

    // [end]

    bool all_ok = true;

    void expect(bool condition, std::string_view what)
    {
        all_ok = all_ok && condition;
        if (!condition)
        {
            std::cout << "CHECK FAILED: " << what << '\n';
        }
    }

}  // namespace

int main()
{
    using namespace cxxgpr;

    // [make]
    // Both precisions are constructed from the same arguments; make_custom_kernel pairs them.
    CustomKernel custom = unwrap(make_custom_kernel<InverseQuadraticTerm>(1.5));
    const KernelSpec spec = KernelSpec::custom(custom);
    // [end]
    std::cout << "num_params = " << spec.num_params() << ", persist_id = " << custom.persist_id() << '\n';

    // Three points in two dimensions (column-major, 3 rows, 2 columns).
    const std::array<double, 6> x_data{0.0, 1.0, 0.5, 0.0, 0.5, 2.0};
    const MatrixRef<double> x(x_data.data(), 3, 2);

    // Reference: a rational quadratic with alpha = 1 is the same function of the squared distance.
    const KernelSpec reference(unwrap(RationalQuadraticKernel::create(1.5, 1.0)));

    // [compare]
    const auto mine = spec.compile<double>();
    const auto theirs = reference.compile<double>();
    std::array<double, 9> a{};
    std::array<double, 9> b{};
    std::array<double, 9> s1{};
    std::array<double, 9> s2{};
    std::array<double, 9> s3{};
    const auto mat = [](std::array<double, 9>& v)
    {
        return MatrixMut<double>(v.data(), 3, 3);
    };

    check(mine.apply_points(x, mat(a), Triangle::Full, mat(s1)));
    check(theirs.apply_points(x, mat(b), Triangle::Full, mat(s1)));
    double max_value_diff = 0.0;
    for (std::size_t i = 0; i < 9; ++i)
    {
        max_value_diff = std::max(max_value_diff, std::abs(a[i] - b[i]));
    }

    // Parameter 0 is log(lengthscale) in both kernels.
    check(mine.grad_points(x, mat(a), 0, Triangle::Full, mat(s1), mat(s2)));
    check(theirs.grad_points(x, mat(b), 0, Triangle::Full, mat(s1), mat(s2)));
    double max_grad_diff = 0.0;
    for (std::size_t i = 0; i < 9; ++i)
    {
        max_grad_diff = std::max(max_grad_diff, std::abs(a[i] - b[i]));
    }

    check(mine.hess_points(x, mat(a), 0, 0, Triangle::Full, mat(s1), mat(s2), mat(s3)));
    check(theirs.hess_points(x, mat(b), 0, 0, Triangle::Full, mat(s1), mat(s2), mat(s3)));
    double max_hess_diff = 0.0;
    for (std::size_t i = 0; i < 9; ++i)
    {
        max_hess_diff = std::max(max_hess_diff, std::abs(a[i] - b[i]));
    }
    // [end]
    std::cout << "max |custom - rational quadratic|: value " << max_value_diff << ", gradient " << max_grad_diff
              << ", Hessian " << max_hess_diff << '\n';
    expect(max_value_diff < 1e-15 && max_grad_diff < 1e-15 && max_hess_diff < 1e-15, "custom kernel matches reference");

    // Hand value: points 0 and 1 are (0, 0.5) and (1, 0) with d2 = 1.25, so k = 1 / (1 + 1.25 / 4.5).
    check(mine.apply_points(x, mat(a), Triangle::Full, mat(s1)));
    std::cout << "k(p0, p1) = " << a[1] << " (hand: " << 1.0 / (1.0 + 1.25 / 4.5) << ")\n";
    expect(std::abs(a[1] - 1.0 / (1.0 + 1.25 / 4.5)) < 1e-15, "hand-computed value");

    // The gradient contract is d/d(log theta): check it against a central finite difference of the Gram matrix.
    // [finite-difference]
    auto probe = spec;
    std::array<double, 1> theta{};
    check(probe.get_params(theta));
    const double h = 1e-5;
    std::array<double, 9> plus{};
    std::array<double, 9> minus{};
    const std::array<double, 1> up{theta[0] + h};
    const std::array<double, 1> down{theta[0] - h};
    check(probe.set_params(up));
    check(probe.compile<double>().apply_points(x, mat(plus), Triangle::Full, mat(s1)));
    check(probe.set_params(down));
    check(probe.compile<double>().apply_points(x, mat(minus), Triangle::Full, mat(s1)));
    check(mine.grad_points(x, mat(a), 0, Triangle::Full, mat(s1), mat(s2)));
    double max_fd_diff = 0.0;
    for (std::size_t i = 0; i < 9; ++i)
    {
        max_fd_diff = std::max(max_fd_diff, std::abs(a[i] - (plus[i] - minus[i]) / (2.0 * h)));
    }
    // [end]
    std::cout << "max |gradient - finite difference| = " << max_fd_diff << '\n';
    expect(max_fd_diff < 1e-9, "gradient matches finite differences");

    // The custom leaf composes like any other leaf and takes part in the flat parameter vector.
    const KernelSpec scaled = KernelSpec(unwrap(ConstantKernel::create(3.0))) * spec;
    std::array<double, 2> flat{};
    check(scaled.get_params(flat));
    std::cout << "constant * custom: log-parameters = [" << flat[0] << ", " << flat[1] << "]\n";
    expect(scaled.num_params() == 2 && std::abs(flat[1] - std::log(1.5)) < 1e-15, "composition");

    // Out-of-bounds updates are rejected and leave the leaf unchanged.
    auto copy = spec;
    const std::array<double, 1> too_big{std::log(100.0)};
    const auto rejected = copy.set_params(too_big);
    std::cout << "set_params(log 100) -> " << (rejected ? "ok" : std::string(rejected.error().message())) << '\n';
    expect(!rejected && rejected.error().code() == GprErrorCode::InvalidInterval, "bounds are enforced");

    // Float evaluation uses the paired InverseQuadraticTerm<float>.
    const std::array<float, 6> xf{0.0F, 1.0F, 0.5F, 0.0F, 0.5F, 2.0F};
    const auto mine_f = spec.compile<float>();
    std::array<float, 9> af{};
    std::array<float, 9> sf{};
    check(mine_f.apply_points(
        MatrixRef<float>(xf.data(), 3, 2), MatrixMut<float>(af.data(), 3, 3), Triangle::Full,
        MatrixMut<float>(sf.data(), 3, 3)
    ));
    expect(std::abs(static_cast<double>(af[1]) - 1.0 / (1.0 + 1.25 / 4.5)) < 1e-6, "float evaluation");

    // [persist]
    // Saving a model that contains the kernel needs the factory below to be registered when loading.
    persist::PersistRegistry registry;
    check(registry.register_kernel(
        "example.inverse_quadratic", [](std::string_view json)
    {
        const auto colon = json.find(':');
        const double ell = std::strtod(std::string(json.substr(colon + 1)).c_str(), nullptr);
        return make_custom_kernel<InverseQuadraticTerm>(ell);
    }
    ));
    const std::string state = unwrap(custom.persist_state_json());
    const CustomKernel restored = unwrap(registry.create_kernel(custom.persist_id(), state));
    // [end]
    std::cout << "state = " << state << ", restored parameters equal: " << std::boolalpha << (restored == custom)
              << '\n';
    expect(restored == custom, "registry round trip");

    return all_ok ? 0 : 1;
}
