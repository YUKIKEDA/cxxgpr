// SPDX-License-Identifier: Apache-2.0
// How-to: persist a model that uses a custom kernel and a custom input transform through a PersistRegistry.
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cxxgpr/core/rng.hpp>
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
#include <cxxgpr/persist/persist.hpp>
#include <cxxgpr/transform/transform.hpp>
#include <expected>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace
{
    using cxxgpr::GprError;

    // Parses a state string that is a single JSON number, such as "1.25" or "3".
    template <typename Number> std::expected<Number, GprError> parse_state(std::string_view state)
    {
        Number value{};
        const auto [end, ec] = std::from_chars(state.data(), state.data() + state.size(), value);
        if (ec != std::errc{} || end != state.data() + state.size())
        {
            return std::unexpected(GprError::invalid_config("custom state is not a number"));
        }
        return value;
    }

    // ------------------------------------------------------------------------------------------------------------
    // A custom kernel leaf: k(d2) = exp(-d2 / (2 l^2)) with one trainable length scale l in [0.05, 50].
    // ------------------------------------------------------------------------------------------------------------
    // [kernel]
    template <cxxgpr::KernelScalarConcept T> class GaussTerm final : public cxxgpr::KernelTerm<T>
    {
    public:
        explicit GaussTerm(double length_scale = 1.0) : l_(length_scale) {}

        std::unique_ptr<cxxgpr::KernelTerm<T>> clone() const override { return std::make_unique<GaussTerm<T>>(l_); }

        std::size_t num_params() const noexcept override { return 1U; }

        std::expected<void, GprError> get_params(std::span<double> out) const override
        {
            out[0] = std::log(l_);
            return {};
        }

        std::expected<void, GprError> set_params(std::span<const double> params) override
        {
            const double l = std::exp(params[0]);
            if (!std::isfinite(l) || l < 0.05 || l > 50.0)
            {
                return std::unexpected(GprError::invalid_config("length scale out of bounds"));
            }
            l_ = l;
            return {};
        }

        std::expected<void, GprError> bounds_into(std::span<cxxgpr::Interval> out) const override
        {
            out[0] = cxxgpr::Interval::create(0.05, 50.0).value();
            return {};
        }

        std::expected<void, GprError> apply(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> out, cxxgpr::Triangle uplo
        ) const override
        {
            for (std::size_t col = 0; col < dist.cols(); ++col)
            {
                for (std::size_t row = 0; row < dist.rows(); ++row)
                {
                    if ((uplo == cxxgpr::Triangle::Lower && row < col)
                        || (uplo == cxxgpr::Triangle::Upper && row > col))
                    {
                        continue;
                    }
                    out(row, col) = value(dist(row, col));
                }
            }
            return {};
        }

        std::expected<void, GprError> apply_cross(cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> out) const override
        {
            return apply(dist, out, cxxgpr::Triangle::Full);
        }

        std::expected<void, GprError> fill_diag(std::span<T> out) const override
        {
            std::ranges::fill(out, T{1});
            return {};
        }

        // d k / d log(l) = k * d2 / l^2
        std::expected<void, GprError> grad(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> d_k, std::size_t, cxxgpr::Triangle uplo
        ) const override
        {
            const T inv_l2 = static_cast<T>(1.0 / (l_ * l_));
            for (std::size_t col = 0; col < dist.cols(); ++col)
            {
                for (std::size_t row = 0; row < dist.rows(); ++row)
                {
                    if ((uplo == cxxgpr::Triangle::Lower && row < col)
                        || (uplo == cxxgpr::Triangle::Upper && row > col))
                    {
                        continue;
                    }
                    d_k(row, col) = value(dist(row, col)) * dist(row, col) * inv_l2;
                }
            }
            return {};
        }

        // d2 k / d log(l)^2 = k * (d2^2 / l^4 - 2 d2 / l^2)
        std::expected<void, GprError> hess(
            cxxgpr::MatrixRef<T> dist, cxxgpr::MatrixMut<T> d2_k, std::size_t, std::size_t, cxxgpr::Triangle uplo
        ) const override
        {
            const T inv_l2 = static_cast<T>(1.0 / (l_ * l_));
            for (std::size_t col = 0; col < dist.cols(); ++col)
            {
                for (std::size_t row = 0; row < dist.rows(); ++row)
                {
                    if ((uplo == cxxgpr::Triangle::Lower && row < col)
                        || (uplo == cxxgpr::Triangle::Upper && row > col))
                    {
                        continue;
                    }
                    const T s = dist(row, col) * inv_l2;
                    d2_k(row, col) = value(dist(row, col)) * (s * s - T{2} * s);
                }
            }
            return {};
        }

        // Persistence: a stable id (never "cxxgpr." ...) and the state needed to rebuild the leaf.
        std::string_view persist_id() const noexcept override { return "example.gauss"; }

        std::expected<std::string, GprError> persist_state_json() const override
        {
            return std::format("{}", l_);  // shortest decimal text that round-trips the double exactly
        }

    private:
        T value(T d2) const { return std::exp(-d2 / static_cast<T>(2.0 * l_ * l_)); }

        double l_;
    };

    // [end]

    // ------------------------------------------------------------------------------------------------------------
    // A custom input transform pair: z = log(x). The fitted half only remembers the column count.
    // ------------------------------------------------------------------------------------------------------------
    // [transform]
    class FittedLog final : public cxxgpr::transform::FittedInputTransform
    {
    public:
        explicit FittedLog(std::size_t n_cols) : n_cols_(n_cols) {}

        std::expected<void, GprError> apply(std::span<double> x, std::size_t n_rows, std::size_t n_cols) const override
        {
            return map(x, n_rows, n_cols, true, [](double v) { return std::log(v); });
        }

        std::expected<void, GprError> inverse_apply(
            std::span<double> x, std::size_t n_rows, std::size_t n_cols
        ) const override
        {
            return map(x, n_rows, n_cols, false, [](double v) { return std::exp(v); });
        }

        std::unique_ptr<cxxgpr::transform::FittedInputTransform> clone() const override
        {
            return std::make_unique<FittedLog>(*this);
        }

        std::string_view persist_id() const override { return "example.log.fitted"; }

        std::expected<std::string, GprError> persist_state_json() const override { return std::to_string(n_cols_); }

    private:
        template <typename F>
        std::expected<void, GprError> map(
            std::span<double> x, std::size_t n_rows, std::size_t n_cols, bool positive_only, F f
        ) const
        {
            if (x.empty() || n_rows == 0 || n_cols == 0)
            {
                return std::unexpected(GprError::empty_input());
            }
            if (n_cols != n_cols_)
            {
                return std::unexpected(GprError::dimension_mismatch(n_cols, n_cols_));
            }
            if (x.size() != n_rows * n_cols)
            {
                return std::unexpected(GprError::length_mismatch("x.size() must equal n_rows * n_cols"));
            }
            for (double& v : x)
            {
                if (!std::isfinite(v) || (positive_only && v <= 0.0))
                {
                    return std::unexpected(GprError::non_finite_input());
                }
                v = f(v);
            }
            return {};
        }

        std::size_t n_cols_;
    };

    class UnfittedLog final : public cxxgpr::transform::UnfittedInputTransform
    {
    public:
        std::expected<std::unique_ptr<cxxgpr::transform::FittedInputTransform>, GprError> fit(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols
        ) const override
        {
            if (x.empty() || n_rows == 0 || n_cols == 0 || x.size() != n_rows * n_cols)
            {
                return std::unexpected(GprError::length_mismatch("x must be a non-empty n_rows x n_cols matrix"));
            }
            return std::make_unique<FittedLog>(n_cols);
        }

        std::unique_ptr<cxxgpr::transform::UnfittedInputTransform> clone() const override
        {
            return std::make_unique<UnfittedLog>(*this);
        }

        std::string_view persist_id() const override { return "example.log.unfitted"; }

        std::expected<std::string, GprError> persist_state_json() const override { return "0"; }
    };

    // [end]

    // [registry]
    // One registry describes how to rebuild every custom piece. Register all that the model contains.
    cxxgpr::persist::PersistRegistry make_registry()
    {
        cxxgpr::persist::PersistRegistry registry;

        (void)registry.register_kernel(
            "example.gauss", [](std::string_view state) -> std::expected<cxxgpr::CustomKernel, GprError>
        {
            auto l = parse_state<double>(state);
            if (!l)
            {
                return std::unexpected(std::move(l).error());
            }
            return cxxgpr::make_custom_kernel<GaussTerm>(*l);
        }
        );
        (void)registry.register_unfitted_input(
            "example.log.unfitted",
            [](std::string_view) -> std::expected<std::unique_ptr<cxxgpr::transform::UnfittedInputTransform>, GprError>
        { return std::make_unique<UnfittedLog>(); }
        );
        (void)registry.register_fitted_input(
            "example.log.fitted",
            [](
                std::string_view state
            ) -> std::expected<std::unique_ptr<cxxgpr::transform::FittedInputTransform>, GprError>
        {
            auto n_cols = parse_state<std::size_t>(state);
            if (!n_cols)
            {
                return std::unexpected(std::move(n_cols).error());
            }
            return std::make_unique<FittedLog>(*n_cols);
        }
        );
        return registry;
    }

    // [end]

    struct ScratchDir
    {
        std::filesystem::path path = std::filesystem::temp_directory_path() / "cxxgpr_example_custom_registry";

        ScratchDir()
        {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }

        ~ScratchDir()
        {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }
    };

    bool same_bits(const std::vector<double>& a, const std::vector<double>& b)
    {
        if (a.size() != b.size())
        {
            return false;
        }
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            if (std::bit_cast<std::uint64_t>(a[i]) != std::bit_cast<std::uint64_t>(b[i]))
            {
                return false;
            }
        }
        return true;
    }
}  // namespace

int main()
{
    // Positive inputs on a logarithmic grid; the signal is smooth in log(x).
    constexpr std::size_t n = 24;
    std::vector<double> x(n);
    std::vector<double> y(n);
    cxxgpr::Xoshiro256PlusPlus rng(3);
    for (std::size_t i = 0; i < n; ++i)
    {
        x[i] = std::pow(10.0, 3.0 * static_cast<double>(i) / static_cast<double>(n - 1));
        y[i] = 2.0 + std::sin(1.2 * std::log(x[i])) + 0.02 * rng.unit_normal();
    }

    // [fit]
    auto gauss = cxxgpr::make_custom_kernel<GaussTerm>(0.5);
    if (!gauss)
    {
        std::cerr << gauss.error().message() << '\n';
        return 1;
    }
    const auto kernel = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value()) * cxxgpr::KernelSpec(*gauss);

    cxxgpr::transform::Pipeline features;
    features.then(UnfittedLog{}).then(cxxgpr::transform::StandardizeInput{});

    auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs>(kernel).with_noise_variance(1e-3);
    if (!trainer)
    {
        return 1;
    }
    auto model = trainer->with_input_transform(features)
                     .with_target_transform(cxxgpr::transform::StandardizeTarget{})
                     .fit(x, n, 1U, y);
    if (!model)
    {
        std::cerr << model.error().message() << '\n';
        return 1;
    }
    // [end]

    const ScratchDir scratch;

    // [save-load]
    const auto registry = make_registry();

    // Saving checks that every custom piece has a valid persist_id and can describe its state.
    if (auto r = cxxgpr::persist::save_with_factor(*model, scratch.path, registry); !r)
    {
        std::cerr << "save failed: " << r.error().message() << '\n';
        return 1;
    }

    // Loading without the registry fails: the file mentions ids this process does not know.
    auto without = cxxgpr::persist::LoadedGpr::load(scratch.path);
    const bool unregistered = !without && without.error().persist_error_kind()
        && *without.error().persist_error_kind() == cxxgpr::PersistErrorKind::UnregisteredId;
    std::cout << "load without registry fails with UnregisteredId: " << std::boolalpha << unregistered << '\n';

    // Loading with the registry rebuilds the kernel and the transforms.
    auto loaded = cxxgpr::persist::LoadedGpr::load(scratch.path, registry);
    if (!loaded)
    {
        std::cerr << loaded.error().message() << '\n';
        return 1;
    }
    // [end]

    const std::vector<double> query{2.0, 20.0, 200.0};
    auto before = model->predict(query, query.size(), 1U);
    auto after = loaded->predict(query, query.size(), 1U);
    if (!before || !after)
    {
        return 1;
    }
    const bool identical = same_bits(before->mean, after->mean) && same_bits(before->variance, after->variance);
    std::cout << "predictions after load are bit-identical: " << identical << '\n';

    // [registration-errors]
    auto reserved = make_registry().register_kernel(
        "cxxgpr.mine", [](std::string_view) -> std::expected<cxxgpr::CustomKernel, GprError>
    { return std::unexpected(GprError::empty_input()); }
    );
    auto duplicate = make_registry().register_kernel(
        "example.gauss", [](std::string_view) -> std::expected<cxxgpr::CustomKernel, GprError>
    { return std::unexpected(GprError::empty_input()); }
    );
    std::cout << "reserved prefix rejected: " << !reserved << ", duplicate rejected: " << !duplicate << '\n';
    // [end]

    return (unregistered && identical && !reserved && !duplicate) ? 0 : 1;
}
