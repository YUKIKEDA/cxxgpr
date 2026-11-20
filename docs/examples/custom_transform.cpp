// SPDX-License-Identifier: Apache-2.0
// How-to: write a custom input transform (a logarithmic map) and use it in a pipeline inside a Gpr.
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
#include <cxxgpr/transform/transform.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <expected>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

namespace
{
    using cxxgpr::GprError;
    using cxxgpr::transform::FittedInputTransform;
    using cxxgpr::transform::UnfittedInputTransform;

    // Shared validation: non-empty, tightly packed, expected width, strictly positive finite entries.
    std::expected<void, GprError> check_positive(
        std::span<const double> x, std::size_t n_rows, std::size_t n_cols, std::size_t expected_cols
    )
    {
        if (x.empty() || n_rows == 0 || n_cols == 0)
        {
            return std::unexpected(GprError::empty_input());
        }
        if (n_cols != expected_cols)
        {
            return std::unexpected(GprError::dimension_mismatch(n_cols, expected_cols));
        }
        if (x.size() != n_rows * n_cols)
        {
            return std::unexpected(GprError::length_mismatch("x.size() must equal n_rows * n_cols"));
        }
        for (const double v : x)
        {
            if (!std::isfinite(v))
            {
                return std::unexpected(GprError::non_finite_input());
            }
            if (v <= 0.0)
            {
                return std::unexpected(GprError::invalid_config("LogInput requires strictly positive features"));
            }
        }
        return {};
    }

    // [fitted]
    // The fitted half: applies z = log(x) and inverts with x = exp(z). It only remembers the column count.
    class FittedLogInput final : public FittedInputTransform
    {
    public:
        explicit FittedLogInput(std::size_t n_cols) : n_cols_(n_cols) {}

        std::expected<void, GprError> apply(std::span<double> x, std::size_t n_rows, std::size_t n_cols) const override
        {
            if (auto ok = check_positive(x, n_rows, n_cols, n_cols_); !ok)
            {
                return ok;
            }
            for (double& v : x)
            {
                v = std::log(v);
            }
            return {};
        }

        std::expected<void, GprError> inverse_apply(
            std::span<double> x, std::size_t n_rows, std::size_t n_cols
        ) const override
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
                if (!std::isfinite(v))
                {
                    return std::unexpected(GprError::non_finite_input());
                }
                v = std::exp(v);
            }
            return {};
        }

        std::unique_ptr<FittedInputTransform> clone() const override { return std::make_unique<FittedLogInput>(*this); }

    private:
        std::size_t n_cols_;
    };
    // [end]

    // [unfitted]
    // The unfitted half: validates the training matrix and produces the fitted transform.
    class LogInput final : public UnfittedInputTransform
    {
    public:
        std::expected<std::unique_ptr<FittedInputTransform>, GprError> fit(
            std::span<const double> x, std::size_t n_rows, std::size_t n_cols
        ) const override
        {
            if (auto ok = check_positive(x, n_rows, n_cols, n_cols); !ok)
            {
                return std::unexpected(std::move(ok).error());
            }
            return std::make_unique<FittedLogInput>(n_cols);
        }

        std::unique_ptr<UnfittedInputTransform> clone() const override { return std::make_unique<LogInput>(*this); }
    };
    // [end]
}  // namespace

int main()
{
    // Inputs span four decades; the function is smooth in log(x), not in x.
    const std::size_t n = 25;
    std::vector<double> x(n);
    std::vector<double> y(n);
    auto truth = [](double v) { return 2.0 + std::sin(1.2 * std::log(v)); };
    for (std::size_t i = 0; i < n; ++i)
    {
        x[i] = std::pow(10.0, 3.7 * static_cast<double>(i) / static_cast<double>(n - 1));
        y[i] = truth(x[i]);
    }

    // [use]
    // Log first, then standardize the logged feature.
    cxxgpr::transform::Pipeline features;
    features.then(LogInput{}).then(cxxgpr::transform::StandardizeInput{});

    const auto kernel = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
        * cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0, cxxgpr::Interval::create(0.1, 10.0).value()).value());
    auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs>(kernel).with_noise_variance(1e-4);
    if (!trainer)
    {
        return 1;
    }
    const auto configured = trainer->with_input_transform(features).with_target_transform(
        cxxgpr::transform::StandardizeTarget{}
    );
    auto model = configured.fit(x, n, 1U, y);
    if (!model)
    {
        std::cerr << model.error().message() << '\n';
        return 1;
    }
    // [end]

    const std::array<double, 3> query{3.0, 30.0, 300.0};
    auto pred = model->predict(query, 3U, 1U);
    if (!pred)
    {
        std::cerr << pred.error().message() << '\n';
        return 1;
    }
    bool ok = true;
    for (std::size_t i = 0; i < query.size(); ++i)
    {
        const double err = std::abs(pred->mean[i] - truth(query[i]));
        std::cout << "x = " << query[i] << ": mean = " << pred->mean[i] << ", truth = " << truth(query[i])
                  << ", |error| = " << err << '\n';
        ok = ok && err < 0.05;
    }

    // A non-positive query is rejected by the transform, not silently mapped to NaN.
    const std::array<double, 1> bad{-1.0};
    auto rejected = model->predict(bad, 1U, 1U);
    std::cout << "negative query rejected: " << (!rejected ? "yes" : "no") << '\n';
    ok = ok && !rejected;
    return ok ? 0 : 1;
}
