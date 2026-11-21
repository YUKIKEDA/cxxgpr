// SPDX-License-Identifier: Apache-2.0
// How-to: inspect and react to errors returned through std::expected<T, GprError>.
#include <array>
#include <cmath>
#include <cstddef>
#include <cxxgpr/core/error.hpp>
#include <cxxgpr/core/interval.hpp>
#include <cxxgpr/core/likelihood.hpp>
#include <cxxgpr/gpr/gpr.hpp>
#include <cxxgpr/kernel/kernel.hpp>
#include <cxxgpr/linalg/jitter.hpp>
#include <cxxgpr/optimizer/optimizer.hpp>
#include <cxxgpr/persist/persist.hpp>
#include <expected>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
    using cxxgpr::GprErrorCode;

    // [name]
    // The library does not ship a code-to-string function; a switch without a default keeps this
    // helper complete (the compiler warns when a new enumerator is added).
    constexpr std::string_view code_name(GprErrorCode code) noexcept
    {
        switch (code)
        {
            case GprErrorCode::EmptyInput:
                return "EmptyInput";
            case GprErrorCode::SizeOverflow:
                return "SizeOverflow";
            case GprErrorCode::LengthMismatch:
                return "LengthMismatch";
            case GprErrorCode::DimensionMismatch:
                return "DimensionMismatch";
            case GprErrorCode::IndexOutOfRange:
                return "IndexOutOfRange";
            case GprErrorCode::InsufficientData:
                return "InsufficientData";
            case GprErrorCode::NonFiniteInput:
                return "NonFiniteInput";
            case GprErrorCode::NonFiniteKernelValue:
                return "NonFiniteKernelValue";
            case GprErrorCode::InvalidHyperparameter:
                return "InvalidHyperparameter";
            case GprErrorCode::InvalidNoiseVariance:
                return "InvalidNoiseVariance";
            case GprErrorCode::InvalidInterval:
                return "InvalidInterval";
            case GprErrorCode::InvalidConfig:
                return "InvalidConfig";
            case GprErrorCode::InvalidDistance:
                return "InvalidDistance";
            case GprErrorCode::DistanceSlot:
                return "DistanceSlot";
            case GprErrorCode::UnsupportedKernelOperation:
                return "UnsupportedKernelOperation";
            case GprErrorCode::CholeskyFailed:
                return "CholeskyFailed";
            case GprErrorCode::NonPositiveDefiniteMatrix:
                return "NonPositiveDefiniteMatrix";
            case GprErrorCode::OptimizationNotConverged:
                return "OptimizationNotConverged";
            case GprErrorCode::WorkspaceTooSmall:
                return "WorkspaceTooSmall";
            case GprErrorCode::PersistFailed:
                return "PersistFailed";
            case GprErrorCode::UnsupportedPersistVersion:
                return "UnsupportedPersistVersion";
        }
        return "unknown";
    }

    // [end]

    // [report]
    void report(std::string_view what, const cxxgpr::GprError& error)
    {
        std::cout << what << " -> " << code_name(error.code()) << ": " << error.message() << '\n';
    }
    // [end]

    // [classify]
    enum class Reaction
    {
        FixCaller,    // a precondition was violated: correct the input and call again
        Retry,        // numerical trouble: change jitter, noise or precision and try again
        Environment,  // I/O or format problem: report to the user
    };

    constexpr Reaction classify(GprErrorCode code) noexcept
    {
        switch (code)
        {
            case GprErrorCode::CholeskyFailed:
            case GprErrorCode::NonPositiveDefiniteMatrix:
            case GprErrorCode::NonFiniteKernelValue:
            case GprErrorCode::OptimizationNotConverged:
                return Reaction::Retry;
            case GprErrorCode::PersistFailed:
            case GprErrorCode::UnsupportedPersistVersion:
                return Reaction::Environment;
            default:
                return Reaction::FixCaller;
        }
    }

    // [end]

    // [give-up]
    // Custom optimizers and kernels report failures with the same factories the library uses.
    struct GiveUp
    {
        static constexpr bool uses_change_indices = false;

        [[nodiscard]] std::expected<cxxgpr::optimizer::OptimizerResult, cxxgpr::GprError> optimize(
            cxxgpr::optimizer::TwiceDifferentiable&, std::span<const double>
        ) const
        {
            return std::unexpected(cxxgpr::GprError::optimization_not_converged(3U));
        }
    };

    // [end]
}  // namespace

int main()
{
    using namespace cxxgpr;
    bool ok = true;

    std::vector<double> x;
    std::vector<double> y;
    for (std::size_t i = 0; i < 20U; ++i)
    {
        x.push_back(static_cast<double>(i) / 19.0);
        y.push_back(std::sin(6.0 * x.back()));
    }
    const auto kernel = KernelSpec(ConstantKernel::create(1.0).value()) * KernelSpec(RbfKernel::create(0.3).value());
    const auto trainer = Gpr<optimizer::Lbfgs>(kernel).with_noise_variance(0.01).value();

    // [check]
    const auto fitted = trainer.fit(x, x.size(), 1U, y);
    if (!fitted)
    {
        report("fit", fitted.error());
        return 1;
    }
    // From here on, *fitted is a valid model.
    // [end]

    // [bad-inputs]
    const auto empty = trainer.fit(std::span<const double>{}, 0U, 1U, std::span<const double>{});
    const auto short_y = trainer.fit(x, x.size(), 1U, std::span<const double>(y).first(5U));
    std::vector<double> y_nan = y;
    y_nan[2] = std::numeric_limits<double>::infinity();
    const auto non_finite = trainer.fit(x, x.size(), 1U, y_nan);
    const std::array<double, 2> two_cols{0.1, 0.2};
    const auto wrong_dim = fitted->predict(two_cols, 1U, 2U);
    report("empty fit", empty.error());
    report("short targets", short_y.error());
    report("infinite target", non_finite.error());
    report("2-column query", wrong_dim.error());
    // [end]
    ok = ok && empty.error().code() == GprErrorCode::EmptyInput
        && short_y.error().code() == GprErrorCode::LengthMismatch
        && non_finite.error().code() == GprErrorCode::NonFiniteInput
        && wrong_dim.error().code() == GprErrorCode::DimensionMismatch;

    // [accessors]
    if (const auto& e = wrong_dim.error(); e.code() == GprErrorCode::DimensionMismatch)
    {
        std::cout << "got " << *e.x_dim() << " columns, expected " << *e.expected_dim() << '\n';
    }
    // [end]
    ok = ok && *wrong_dim.error().x_dim() == 2U && *wrong_dim.error().expected_dim() == 1U;

    // [validators]
    const auto no_noise = trainer.with_noise_variance(0.0);
    const auto bad_jitter = JitterPolicy::fixed(-1.0);
    const auto bad_length = RbfKernel::create(-1.0);
    const auto bad_interval = Interval::create(2.0, 1.0);
    const auto outside = BoundedParam::create(5.0, Interval::create(0.0, 1.0).value());
    report("noise 0", no_noise.error());
    report("negative jitter", bad_jitter.error());
    report("negative lengthscale", bad_length.error());
    report("reversed interval", bad_interval.error());
    report("value outside interval", outside.error());
    // [end]
    ok = ok && no_noise.error().code() == GprErrorCode::InvalidNoiseVariance
        && bad_jitter.error().code() == GprErrorCode::InvalidConfig
        && bad_length.error().code() == GprErrorCode::InvalidInterval
        && bad_interval.error().interval_error_kind() == IntervalErrorKind::InvalidBounds
        && outside.error().interval_error_kind() == IntervalErrorKind::OutOfRange
        && *outside.error().interval_value() == 5.0 && *bad_interval.error().interval_lo() == 2.0;

    // [index]
    std::array<double, 2> diag{};
    const auto index_err = GaussianLikelihood::create(0.1).value().noise_grad_diag(diag, 3U);
    report("noise gradient index 3", index_err.error());
    // [end]
    ok = ok && index_err.error().code() == GprErrorCode::IndexOutOfRange;

    // [cholesky]
    // Single precision, a huge constant kernel and 2e-5 noise: float rounding swamps the noise term.
    std::vector<double> xs;
    std::vector<double> ys;
    for (std::size_t i = 0; i < 300U; ++i)
    {
        xs.push_back(static_cast<double>(i) / 299.0);
        ys.push_back(std::sin(3.0 * xs.back()));
    }
    const auto smooth = KernelSpec(ConstantKernel::create(5e4).value());
    const auto fragile = Gpr<optimizer::Fixed, SinglePrecision>(smooth).with_noise_variance(2e-5).value();
    const auto strict = fragile.fit(xs, xs.size(), 1U, ys);
    if (!strict)
    {
        const auto& e = strict.error();
        std::cout << code_name(e.code()) << ": last jitter " << *e.jitter() << ", matrix order " << *e.matrix_size()
                  << ", stage " << (*e.cholesky_stage() == CholeskyStage::Fit ? "Fit" : "Predict") << '\n';
    }
    const auto ladder = JitterPolicy::adaptive(1e-3, 10.0, 6U, 10.0).value();
    const auto recovered = fragile.with_jitter(ladder).fit(xs, xs.size(), 1U, ys);
    if (recovered)
    {
        std::cout << "adaptive jitter recovered, accepted jitter " << recovered->factor_jitter() << '\n';
    }
    // [end]
    ok = ok && !strict && strict.error().code() == GprErrorCode::CholeskyFailed && recovered.has_value()
        && recovered->factor_jitter() > 0.0;

    // [and-then]
    const auto chained = trainer.with_noise_variance(0.02).and_then([&](const auto& configured)
    { return configured.fit(x, x.size(), 1U, y); });
    const auto chained_bad = trainer.with_noise_variance(-0.02).and_then([&](const auto& configured)
    { return configured.fit(x, x.size(), 1U, y); });
    const auto mean_at_half = chained
                                  .and_then([](const auto& model)
    {
        const std::array<double, 1> q{0.5};
        return model.predict(q, 1U, 1U);
    }).transform([](const Prediction<double>& p) { return p.mean[0]; });
    std::cout << "chained mean at 0.5: " << mean_at_half.value_or(-99.0)
              << ", failing chain stops at: " << code_name(chained_bad.error().code()) << '\n';
    // [end]
    ok = ok && mean_at_half.has_value() && std::abs(*mean_at_half - std::sin(3.0)) < 0.1
        && chained_bad.error().code() == GprErrorCode::InvalidNoiseVariance;

    // [or-else]
    const auto lenient = trainer.with_noise_variance(-1.0).or_else(
        [&](const GprError&) -> std::expected<Gpr<optimizer::Lbfgs>, GprError> { return trainer; }
    );
    // [end]
    ok = ok && lenient.has_value();

    // [value-throws]
    bool threw = false;
    try
    {
        (void)bad_length.value();  // value() on an error throws std::bad_expected_access<GprError>
    }
    catch (const std::bad_expected_access<GprError>& ex)
    {
        threw = true;
        std::cout << "value() threw; carried code " << code_name(ex.error().code()) << '\n';
    }
    // [end]
    ok = ok && threw;

    // [factories]
    const auto distance_error = GprError::invalid_distance(
        std::nullopt, std::nullopt, std::pair<std::size_t, std::size_t>{1U, 2U}, "negative entry"
    )
                                    .with_slot_if_unset(2U)
                                    .with_dim_if_unset(0U);
    std::cout << "invalid_distance: " << distance_error.message() << " | slot " << *distance_error.distance_slot()
              << ", dim " << *distance_error.distance_dim() << ", pair (" << distance_error.distance_pair()->first
              << ", " << distance_error.distance_pair()->second << "), reason \"" << *distance_error.distance_reason()
              << "\"\n";
    const GprError slot_error = GprError::distance_slot(SlotErrorKind::Missing, 1U);
    std::cout << "distance_slot: " << slot_error.message() << '\n';
    std::cout << "size_overflow: " << GprError::size_overflow().message() << '\n';
    std::cout << "insufficient_data: " << GprError::insufficient_data(2U, 5U).message() << '\n';
    std::cout << "non_finite_kernel_value: " << GprError::non_finite_kernel_value().message() << '\n';
    std::cout << "non_positive_definite_matrix: " << GprError::non_positive_definite_matrix().message() << '\n';
    std::cout << "unsupported_kernel_operation: " << GprError::unsupported_kernel_operation("no coordinates").message()
              << '\n';
    std::cout << "workspace_too_small: " << GprError::workspace_too_small().message() << '\n';
    std::cout << "invalid_hyperparameter: " << GprError::invalid_hyperparameter("scale must be positive").message()
              << '\n';
    std::cout << "length_mismatch: " << GprError::length_mismatch("expected 3 values").message() << '\n';
    std::cout << "index_out_of_range: " << GprError::index_out_of_range("leaf 4 of 3").message() << '\n';
    std::cout << "persist_failed: " << GprError::persist_failed(PersistErrorKind::Config, "bad json").message() << '\n';
    std::cout << "unsupported_persist_version: " << GprError::unsupported_persist_version(2U, 1U).message() << '\n';
    std::cout << "invalid_config: " << GprError::invalid_config("tolerance must be non-negative").message() << '\n';
    std::cout << "optimization_not_converged: " << GprError::optimization_not_converged(100U).message() << '\n';
    std::cout << "empty_input: " << GprError::empty_input().message() << '\n';
    std::cout << "non_finite_input: " << GprError::non_finite_input().message() << '\n';
    std::cout << "dimension_mismatch: " << GprError::dimension_mismatch(2U, 3U).message() << '\n';
    std::cout << "cholesky_failed: " << GprError::cholesky_failed(1e-6, 10U, CholeskyStage::Predict).message() << '\n';
    std::cout << "invalid_interval_bounds: " << GprError::invalid_interval_bounds(2.0, 1.0).message() << '\n';
    std::cout << "invalid_interval_out_of_range: " << GprError::invalid_interval_out_of_range(5.0, 0.0, 1.0).message()
              << '\n';
    std::cout << "invalid_noise_variance: " << GprError::invalid_noise_variance("must be positive").message() << '\n';
    // [end]
    ok = ok && *distance_error.distance_slot() == 2U && *distance_error.distance_dim() == 0U
        && *slot_error.slot_error_kind() == SlotErrorKind::Missing && *slot_error.distance_slot() == 1U
        && GprError::empty_input() == GprError::empty_input() && GprError::empty_input() != GprError::non_finite_input()
        && *GprError::insufficient_data(2U, 5U).n_points() == 2U
        && *GprError::insufficient_data(2U, 5U).min_points() == 5U
        && *GprError::optimization_not_converged(100U).iterations() == 100U
        && *GprError::unsupported_persist_version(2U, 1U).found_version() == 2U
        && *GprError::unsupported_persist_version(2U, 1U).supported_version() == 1U
        && !GprError::empty_input().distance_reason().has_value();

    // [custom-error]
    const auto gave_up = Gpr<GiveUp>(kernel).fit(x, x.size(), 1U, y);
    std::cout << "custom optimizer error passes through: " << code_name(gave_up.error().code()) << " after "
              << *gave_up.error().iterations() << " iterations\n";
    // [end]
    ok = ok && gave_up.error().code() == GprErrorCode::OptimizationNotConverged;

    // [persist]
    const auto missing = persist::LoadedGpr::load("this-directory-does-not-exist");
    std::cout << "load -> " << code_name(missing.error().code()) << ", kind "
              << (*missing.error().persist_error_kind() == PersistErrorKind::Io ? "Io" : "other") << '\n';
    // [end]
    ok = ok && missing.error().code() == GprErrorCode::PersistFailed
        && classify(missing.error().code()) == Reaction::Environment
        && classify(strict.error().code()) == Reaction::Retry && classify(empty.error().code()) == Reaction::FixCaller;

    std::cout << (ok ? "OK" : "FAILED") << '\n';
    return ok ? 0 : 1;
}
