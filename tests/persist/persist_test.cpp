// SPDX-License-Identifier: Apache-2.0
// Spec: FMT-01, FMT-02, FMT-03, FMT-04, FMT-05, FMT-06, FMT-08
/// @file persist_test.cpp
/// @brief Unit and integration tests for GPR model persistence, mmap loading, and registry hooks.
/// @details Verifies save_with_factor() mmap roundtrip and bit-identical predictions/LOO, save()
/// refactor-on-load parity, into_fitted() precision checks, MixedPrecision<ReevaluateKernel> and
/// CustomKernel PersistRegistry roundtrips, and negative validation against corrupt or unsupported
/// config.json and model.safetensors artifacts.

#include "cxxgpr/persist/persist.hpp"

#include "cxxgpr/gpr/gpr.hpp"
#include "cxxgpr/kernel/kernel.hpp"
#include "cxxgpr/transform/transform.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "allocation_audit.hpp"
#include "small_stack.hpp"
#include "span_shape_contract.hpp"

namespace
{

    /// @brief RAII temporary directory guard creating an isolated per-test directory under temp_directory_path.
    /// @details Removes any leftover directory on construction and cleans up recursively on destruction
    /// so parallel CTest runs never collide on persistence artifacts.
    struct TempDirGuard
    {
        /// @brief Filesystem path of the managed temporary directory.
        std::filesystem::path path;

        /// @brief Creates a fresh temporary directory named with prefix `cxxgpr_test_`.
        /// @param[in] name Unique test identifier suffix appended to the directory name.
        explicit TempDirGuard(const std::string& name)
        {
            path = std::filesystem::temp_directory_path() / ("cxxgpr_test_" + name);
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
            std::filesystem::create_directories(path, ec);
        }

        /// @brief Recursively deletes the temporary directory while suppressing cleanup exceptions.
        ~TempDirGuard()
        {
            std::error_code ec;
            try
            {
                std::filesystem::remove_all(path, ec);
            }
            catch (...)
            {
                ec.clear();
            }
        }
    };

    /// @brief Double-precision custom scaled exponential kernel implementing state JSON persistence.
    /// @details Evaluates k(d2) = scale_ * exp(-d2) and emits serialized scale_ state via
    /// persist_state_json() under persist ID `"user.custom_scale"`.
    class CustomScaleKernel final : public cxxgpr::KernelTerm<double>
    {
    public:
        /// @brief Constructs the double-precision custom kernel with initial scale factor.
        /// @param[in] scale Positive kernel output scale parameter.
        /// @param[in] state_text When set, persist_state_json() returns this text instead of the decimal scale.
        /// @param[in] id Persistence identifier reported by persist_id().
        explicit CustomScaleKernel(
            double scale, std::optional<std::string> state_text = std::nullopt, std::string id = "user.custom_scale"
        ) :
            scale_(scale),
            state_text_(std::move(state_text)),
            id_(std::move(id))
        {
        }

        /// @brief Clones this custom kernel preserving its scale_ value.
        /// @return Owning unique_ptr copy of this CustomScaleKernel.
        [[nodiscard]] std::unique_ptr<cxxgpr::KernelTerm<double>> clone() const override
        {
            return std::make_unique<CustomScaleKernel>(scale_, state_text_, id_);
        }

        /// @brief Returns the number of trainable hyperparameters (1 for scale_).
        /// @return Hyperparameter count (always 1).
        [[nodiscard]] std::size_t num_params() const noexcept override { return 1U; }

        /// @brief Writes log(scale_) into out[0].
        /// @param[out] out Destination span of length 1 receiving log-transformed scale.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> get_params(std::span<double> out) const override
        {
            out[0] = std::log(scale_);
            return {};
        }

        /// @brief Updates scale_ from log-parameter params[0].
        /// @param[in] params Input span containing log(scale) at index 0.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> set_params(std::span<const double> params) override
        {
            scale_ = std::exp(params[0]);
            return {};
        }

        /// @brief Populates default positive bounds for the scale parameter.
        /// @param[out] out Destination Interval span of length 1.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> bounds_into(std::span<cxxgpr::Interval> out) const override
        {
            out[0] = cxxgpr::Interval::default_positive();
            return {};
        }

        /// @brief Evaluates scale_ * exp(-d2) over the requested square matrix triangle.
        /// @param[in] dist Pairwise squared Euclidean distance matrix.
        /// @param[out] out Output kernel Gram matrix buffer.
        /// @param[in] uplo Triangle mask controlling which entries are written.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> apply(
            cxxgpr::MatrixRef<double> dist, cxxgpr::MatrixMut<double> out, cxxgpr::Triangle uplo
        ) const override
        {
            const std::size_t n = dist.rows();
            for (std::size_t col = 0; col < n; ++col)
            {
                const std::size_t rb = (uplo == cxxgpr::Triangle::Lower) ? col : 0U;
                const std::size_t re = (uplo == cxxgpr::Triangle::Upper) ? (col + 1U) : n;
                for (std::size_t row = rb; row < re; ++row)
                {
                    out(row, col) = scale_ * std::exp(-dist(row, col));
                }
            }
            return {};
        }

        /// @brief Evaluates rectangular cross-covariance scale_ * exp(-d2).
        /// @param[in] dist Rectangular pairwise squared Euclidean distance matrix.
        /// @param[out] out Output cross-covariance matrix buffer.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> apply_cross(
            cxxgpr::MatrixRef<double> dist, cxxgpr::MatrixMut<double> out
        ) const override
        {
            for (std::size_t col = 0; col < dist.cols(); ++col)
            {
                for (std::size_t row = 0; row < dist.rows(); ++row)
                {
                    out(row, col) = scale_ * std::exp(-dist(row, col));
                }
            }
            return {};
        }

        /// @brief Fills the prior kernel diagonal with scale_.
        /// @param[out] out Destination diagonal span of length n.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> fill_diag(std::span<double> out) const override
        {
            std::ranges::fill(out, scale_);
            return {};
        }

        /// @brief Computes derivative dk / d(log scale), which equals k itself.
        /// @param[in] dist Pairwise squared Euclidean distance matrix.
        /// @param[out] d_k Output gradient matrix buffer.
        /// @param[in] param_idx Local parameter index (unused because num_params() == 1).
        /// @param[in] uplo Triangle mask controlling which entries are written.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> grad(
            cxxgpr::MatrixRef<double> dist,
            cxxgpr::MatrixMut<double> d_k,
            [[maybe_unused]] std::size_t param_idx,
            cxxgpr::Triangle uplo
        ) const override
        {
            return apply(dist, d_k, uplo);
        }

        /// @brief Computes second derivative d^2 k / d(log scale)^2, which equals k itself.
        /// @param[in] dist Pairwise squared Euclidean distance matrix.
        /// @param[out] d2_k Output Hessian matrix buffer.
        /// @param[in] i First local parameter index (unused because num_params() == 1).
        /// @param[in] j Second local parameter index (unused because num_params() == 1).
        /// @param[in] uplo Triangle mask controlling which entries are written.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> hess(
            cxxgpr::MatrixRef<double> dist,
            cxxgpr::MatrixMut<double> d2_k,
            [[maybe_unused]] std::size_t i,
            [[maybe_unused]] std::size_t j,
            cxxgpr::Triangle uplo
        ) const override
        {
            return apply(dist, d2_k, uplo);
        }

        /// @brief Returns the persistence identifier matching CustomScaleKernelFloat and PersistRegistry.
        /// @return View of the identifier given at construction (`"user.custom_scale"` by default).
        [[nodiscard]] std::string_view persist_id() const noexcept override { return id_; }

        /// @brief Serializes the state for persistence.
        /// @return The state text given at construction, or else the decimal string representation of scale_.
        [[nodiscard]] std::expected<std::string, cxxgpr::GprError> persist_state_json() const override
        {
            return state_text_.has_value() ? *state_text_ : std::to_string(scale_);
        }

    private:
        /// @brief Positive output scale factor in linear space.
        double scale_;
        /// @brief State text persisted verbatim in place of the decimal scale, when set.
        std::optional<std::string> state_text_;
        /// @brief Persistence identifier.
        std::string id_;
    };

    /// @brief Single-precision companion of CustomScaleKernel required by CustomKernel::create().
    /// @details Evaluates float-precision k(d2) = scale_ * exp(-d2) and shares persist ID
    /// `"user.custom_scale"` with CustomScaleKernel.
    class CustomScaleKernelFloat final : public cxxgpr::KernelTerm<float>
    {
    public:
        /// @brief Constructs the single-precision custom kernel with initial scale factor.
        /// @param[in] scale Positive kernel output scale parameter.
        /// @param[in] id Persistence identifier reported by persist_id().
        explicit CustomScaleKernelFloat(float scale, std::string id = "user.custom_scale") :
            scale_(scale),
            id_(std::move(id))
        {
        }

        /// @brief Clones this float custom kernel preserving its scale_ value.
        /// @return Owning unique_ptr copy of this CustomScaleKernelFloat.
        [[nodiscard]] std::unique_ptr<cxxgpr::KernelTerm<float>> clone() const override
        {
            return std::make_unique<CustomScaleKernelFloat>(scale_, id_);
        }

        /// @brief Returns the number of trainable hyperparameters (1 for scale_).
        /// @return Hyperparameter count (always 1).
        [[nodiscard]] std::size_t num_params() const noexcept override { return 1U; }

        /// @brief Writes log(scale_) cast to double into out[0].
        /// @param[out] out Destination span of length 1 receiving log-transformed scale.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> get_params(std::span<double> out) const override
        {
            out[0] = std::log(static_cast<double>(scale_));
            return {};
        }

        /// @brief Updates float scale_ from log-parameter params[0].
        /// @param[in] params Input span containing log(scale) at index 0.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> set_params(std::span<const double> params) override
        {
            scale_ = static_cast<float>(std::exp(params[0]));
            return {};
        }

        /// @brief Populates default positive bounds for the scale parameter.
        /// @param[out] out Destination Interval span of length 1.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> bounds_into(std::span<cxxgpr::Interval> out) const override
        {
            out[0] = cxxgpr::Interval::default_positive();
            return {};
        }

        /// @brief Evaluates float scale_ * exp(-d2) over the requested square matrix triangle.
        /// @param[in] dist Pairwise squared Euclidean distance matrix.
        /// @param[out] out Output kernel Gram matrix buffer.
        /// @param[in] uplo Triangle mask controlling which entries are written.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> apply(
            cxxgpr::MatrixRef<float> dist, cxxgpr::MatrixMut<float> out, cxxgpr::Triangle uplo
        ) const override
        {
            const std::size_t n = dist.rows();
            for (std::size_t col = 0; col < n; ++col)
            {
                const std::size_t rb = (uplo == cxxgpr::Triangle::Lower) ? col : 0U;
                const std::size_t re = (uplo == cxxgpr::Triangle::Upper) ? (col + 1U) : n;
                for (std::size_t row = rb; row < re; ++row)
                {
                    out(row, col) = scale_ * std::exp(-dist(row, col));
                }
            }
            return {};
        }

        /// @brief Evaluates float rectangular cross-covariance scale_ * exp(-d2).
        /// @param[in] dist Rectangular pairwise squared Euclidean distance matrix.
        /// @param[out] out Output cross-covariance matrix buffer.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> apply_cross(
            cxxgpr::MatrixRef<float> dist, cxxgpr::MatrixMut<float> out
        ) const override
        {
            for (std::size_t col = 0; col < dist.cols(); ++col)
            {
                for (std::size_t row = 0; row < dist.rows(); ++row)
                {
                    out(row, col) = scale_ * std::exp(-dist(row, col));
                }
            }
            return {};
        }

        /// @brief Fills the prior float kernel diagonal with scale_.
        /// @param[out] out Destination diagonal span of length n.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> fill_diag(std::span<float> out) const override
        {
            std::ranges::fill(out, scale_);
            return {};
        }

        /// @brief Computes derivative dk / d(log scale) in single precision.
        /// @param[in] dist Pairwise squared Euclidean distance matrix.
        /// @param[out] d_k Output gradient matrix buffer.
        /// @param[in] param_idx Local parameter index (unused because num_params() == 1).
        /// @param[in] uplo Triangle mask controlling which entries are written.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> grad(
            cxxgpr::MatrixRef<float> dist,
            cxxgpr::MatrixMut<float> d_k,
            [[maybe_unused]] std::size_t param_idx,
            cxxgpr::Triangle uplo
        ) const override
        {
            return apply(dist, d_k, uplo);
        }

        /// @brief Computes second derivative d^2 k / d(log scale)^2 in single precision.
        /// @param[in] dist Pairwise squared Euclidean distance matrix.
        /// @param[out] d2_k Output Hessian matrix buffer.
        /// @param[in] i First local parameter index (unused because num_params() == 1).
        /// @param[in] j Second local parameter index (unused because num_params() == 1).
        /// @param[in] uplo Triangle mask controlling which entries are written.
        /// @return Empty expected on success.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> hess(
            cxxgpr::MatrixRef<float> dist,
            cxxgpr::MatrixMut<float> d2_k,
            [[maybe_unused]] std::size_t i,
            [[maybe_unused]] std::size_t j,
            cxxgpr::Triangle uplo
        ) const override
        {
            return apply(dist, d2_k, uplo);
        }

        /// @brief Returns the persistence identifier shared with CustomScaleKernel.
        /// @return View of the identifier given at construction (`"user.custom_scale"` by default).
        [[nodiscard]] std::string_view persist_id() const noexcept override { return id_; }

    private:
        /// @brief Positive output scale factor stored in float precision.
        float scale_;
        /// @brief Persistence identifier.
        std::string id_;
    };

    /// @brief Fitted input transform that changes nothing and persists a caller-supplied identifier and state text.
    class TextFittedInput final : public cxxgpr::transform::FittedInputTransform
    {
    public:
        /// @brief Constructs the transform.
        /// @param[in] id Persistence identifier reported by persist_id().
        /// @param[in] state State text returned by persist_state_json().
        TextFittedInput(std::string id, std::string state) : id_(std::move(id)), state_(std::move(state)) {}

        /// @brief Leaves the inputs unchanged.
        /// @param[in,out] x Column-major inputs, left as they are.
        /// @param[in] n_rows Number of rows of x.
        /// @param[in] n_cols Number of columns of x.
        /// @return Empty expected.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> apply(
            [[maybe_unused]] std::span<double> x,
            [[maybe_unused]] std::size_t n_rows,
            [[maybe_unused]] std::size_t n_cols
        ) const override
        {
            return {};
        }

        /// @brief Leaves the inputs unchanged.
        /// @param[in,out] x Column-major inputs, left as they are.
        /// @param[in] n_rows Number of rows of x.
        /// @param[in] n_cols Number of columns of x.
        /// @return Empty expected.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> inverse_apply(
            [[maybe_unused]] std::span<double> x,
            [[maybe_unused]] std::size_t n_rows,
            [[maybe_unused]] std::size_t n_cols
        ) const override
        {
            return {};
        }

        /// @brief Clones this transform.
        /// @return Owning copy.
        [[nodiscard]] std::unique_ptr<cxxgpr::transform::FittedInputTransform> clone() const override
        {
            return std::make_unique<TextFittedInput>(id_, state_);
        }

        /// @brief Returns the identifier given at construction.
        /// @return View of the identifier.
        [[nodiscard]] std::string_view persist_id() const override { return id_; }

        /// @brief Returns the state text given at construction.
        /// @return The state text.
        [[nodiscard]] std::expected<std::string, cxxgpr::GprError> persist_state_json() const override
        {
            return state_;
        }

    private:
        /// @brief Persistence identifier.
        std::string id_;
        /// @brief Persisted state text.
        std::string state_;
    };

    /// @brief Unfitted input transform persisting its own text and fitting to a TextFittedInput.
    class TextUnfittedInput final : public cxxgpr::transform::UnfittedInputTransform
    {
    public:
        /// @brief Constructs the transform.
        /// @param[in] id Persistence identifier reported by persist_id().
        /// @param[in] state State text returned by persist_state_json().
        /// @param[in] fitted The fitted transform that fit() returns a copy of.
        TextUnfittedInput(std::string id, std::string state, TextFittedInput fitted) :
            id_(std::move(id)),
            state_(std::move(state)),
            fitted_(std::move(fitted))
        {
        }

        /// @brief Returns a copy of the fitted transform given at construction.
        /// @param[in] x Column-major training inputs, ignored.
        /// @param[in] n_rows Number of rows of x.
        /// @param[in] n_cols Number of columns of x.
        /// @return Owning fitted transform.
        [[nodiscard]] std::expected<std::unique_ptr<cxxgpr::transform::FittedInputTransform>, cxxgpr::GprError> fit(
            [[maybe_unused]] std::span<const double> x,
            [[maybe_unused]] std::size_t n_rows,
            [[maybe_unused]] std::size_t n_cols
        ) const override
        {
            return fitted_.clone();
        }

        /// @brief Clones this transform.
        /// @return Owning copy.
        [[nodiscard]] std::unique_ptr<cxxgpr::transform::UnfittedInputTransform> clone() const override
        {
            return std::make_unique<TextUnfittedInput>(id_, state_, fitted_);
        }

        /// @brief Returns the identifier given at construction.
        /// @return View of the identifier.
        [[nodiscard]] std::string_view persist_id() const override { return id_; }

        /// @brief Returns the state text given at construction.
        /// @return The state text.
        [[nodiscard]] std::expected<std::string, cxxgpr::GprError> persist_state_json() const override
        {
            return state_;
        }

    private:
        /// @brief Persistence identifier.
        std::string id_;
        /// @brief Persisted state text.
        std::string state_;
        /// @brief Result of fit().
        TextFittedInput fitted_;
    };

    /// @brief Fitted target transform that changes nothing and persists a caller-supplied identifier and state text.
    class TextFittedTarget final : public cxxgpr::transform::FittedTargetTransform
    {
    public:
        /// @brief Constructs the transform.
        /// @param[in] id Persistence identifier reported by persist_id().
        /// @param[in] state State text returned by persist_state_json().
        TextFittedTarget(std::string id, std::string state) : id_(std::move(id)), state_(std::move(state)) {}

        /// @brief Leaves the targets unchanged.
        /// @param[in,out] y Targets, left as they are.
        /// @return Empty expected.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> transform(
            [[maybe_unused]] std::span<double> y
        ) const override
        {
            return {};
        }

        /// @brief Leaves the means unchanged.
        /// @param[in,out] mean Predictive means, left as they are.
        /// @return Empty expected.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> inverse_transform_mean(
            [[maybe_unused]] std::span<double> mean
        ) const override
        {
            return {};
        }

        /// @brief Leaves the variances unchanged.
        /// @param[in,out] var Predictive variances, left as they are.
        /// @return Empty expected.
        [[nodiscard]] std::expected<void, cxxgpr::GprError> inverse_transform_variance(
            [[maybe_unused]] std::span<double> var
        ) const override
        {
            return {};
        }

        /// @brief Clones this transform.
        /// @return Owning copy.
        [[nodiscard]] std::unique_ptr<cxxgpr::transform::FittedTargetTransform> clone() const override
        {
            return std::make_unique<TextFittedTarget>(id_, state_);
        }

        /// @brief Returns the identifier given at construction.
        /// @return View of the identifier.
        [[nodiscard]] std::string_view persist_id() const override { return id_; }

        /// @brief Returns the state text given at construction.
        /// @return The state text.
        [[nodiscard]] std::expected<std::string, cxxgpr::GprError> persist_state_json() const override
        {
            return state_;
        }

    private:
        /// @brief Persistence identifier.
        std::string id_;
        /// @brief Persisted state text.
        std::string state_;
    };

    /// @brief Unfitted target transform persisting its own text and fitting to a TextFittedTarget.
    class TextUnfittedTarget final : public cxxgpr::transform::UnfittedTargetTransform
    {
    public:
        /// @brief Constructs the transform.
        /// @param[in] id Persistence identifier reported by persist_id().
        /// @param[in] state State text returned by persist_state_json().
        /// @param[in] fitted The fitted transform that fit() returns a copy of.
        TextUnfittedTarget(std::string id, std::string state, TextFittedTarget fitted) :
            id_(std::move(id)),
            state_(std::move(state)),
            fitted_(std::move(fitted))
        {
        }

        /// @brief Returns a copy of the fitted transform given at construction.
        /// @param[in] y Training targets, ignored.
        /// @return Owning fitted transform.
        [[nodiscard]] std::expected<std::unique_ptr<cxxgpr::transform::FittedTargetTransform>, cxxgpr::GprError> fit(
            [[maybe_unused]] std::span<const double> y
        ) const override
        {
            return fitted_.clone();
        }

        /// @brief Clones this transform.
        /// @return Owning copy.
        [[nodiscard]] std::unique_ptr<cxxgpr::transform::UnfittedTargetTransform> clone() const override
        {
            return std::make_unique<TextUnfittedTarget>(id_, state_, fitted_);
        }

        /// @brief Returns the identifier given at construction.
        /// @return View of the identifier.
        [[nodiscard]] std::string_view persist_id() const override { return id_; }

        /// @brief Returns the state text given at construction.
        /// @return The state text.
        [[nodiscard]] std::expected<std::string, cxxgpr::GprError> persist_state_json() const override
        {
            return state_;
        }

    private:
        /// @brief Persistence identifier.
        std::string id_;
        /// @brief Persisted state text.
        std::string state_;
        /// @brief Result of fit().
        TextFittedTarget fitted_;
    };

    /// @brief One text per custom component of a model: the kernel and the four transforms.
    /// @details Used both for the state texts and for the persistence identifiers of the five writer sites.
    struct ComponentTexts
    {
        /// @brief Text of the custom kernel leaf.
        std::string kernel;
        /// @brief Text of the unfitted input transform.
        std::string x_unfitted;
        /// @brief Text of the fitted input transform.
        std::string x_fitted;
        /// @brief Text of the unfitted target transform.
        std::string y_unfitted;
        /// @brief Text of the fitted target transform.
        std::string y_fitted;

        /// @brief Compares all five texts.
        /// @return True when every text is equal.
        [[nodiscard]] bool operator==(const ComponentTexts&) const = default;
    };

    /// @brief Identifiers used by the five custom components unless a test replaces one.
    /// @return The default identifiers.
    [[nodiscard]] ComponentTexts default_component_ids()
    {
        return ComponentTexts{
            .kernel = "user.custom_scale",
            .x_unfitted = "user.text.x_unfitted",
            .x_fitted = "user.text.x_fitted",
            .y_unfitted = "user.text.y_unfitted",
            .y_fitted = "user.text.y_fitted",
        };
    }

    /// @brief Factors a three-point model whose kernel and four transforms are all custom components.
    /// @param[in] ids Persistence identifier of each component.
    /// @param[in] states State text of each component.
    /// @param[in] y_train Three training targets.
    /// @return The factored model.
    [[nodiscard]] auto factor_all_custom_model(
        const ComponentTexts& ids, const ComponentTexts& states, const std::array<double, 3>& y_train
    )
    {
        const std::array<double, 3> x_train{0.0, 1.0, 2.0};
        const cxxgpr::KernelSpec spec(
            cxxgpr::CustomKernel::create(
                std::make_unique<CustomScaleKernel>(2.5, states.kernel, ids.kernel),
                std::make_unique<CustomScaleKernelFloat>(2.5F, ids.kernel)
            )
                .value()
        );
        return cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(spec)
            .with_input_transform(
                TextUnfittedInput(ids.x_unfitted, states.x_unfitted, TextFittedInput(ids.x_fitted, states.x_fitted))
            )
            .with_target_transform(
                TextUnfittedTarget(ids.y_unfitted, states.y_unfitted, TextFittedTarget(ids.y_fitted, states.y_fitted))
            )
            .factor(x_train, 3U, 1U, y_train);
    }

    /// @brief Registers factories for the five default identifiers that record the state text they receive.
    /// @param[in,out] reg Registry receiving the factories.
    /// @param[out] received Destination of the state texts; must outlive reg.
    void register_recording_factories(cxxgpr::persist::PersistRegistry& reg, ComponentTexts& received)
    {
        const ComponentTexts ids = default_component_ids();
        ASSERT_TRUE(
            reg.register_kernel(
                   ids.kernel,
                   [&received](std::string_view state) -> std::expected<cxxgpr::CustomKernel, cxxgpr::GprError>
        {
            received.kernel = std::string(state);
            return cxxgpr::CustomKernel::create(
                std::make_unique<CustomScaleKernel>(2.5), std::make_unique<CustomScaleKernelFloat>(2.5F)
            );
        }
            ).has_value()
        );
        ASSERT_TRUE(
            reg.register_unfitted_input(
                   ids.x_unfitted,
                   [&received, ids](std::string_view state)
                       -> std::expected<std::unique_ptr<cxxgpr::transform::UnfittedInputTransform>, cxxgpr::GprError>
        {
            received.x_unfitted = std::string(state);
            return std::make_unique<TextUnfittedInput>(
                ids.x_unfitted, std::string(state), TextFittedInput(ids.x_fitted, "")
            );
        }
            ).has_value()
        );
        ASSERT_TRUE(
            reg.register_fitted_input(
                   ids.x_fitted,
                   [&received, ids](std::string_view state)
                       -> std::expected<std::unique_ptr<cxxgpr::transform::FittedInputTransform>, cxxgpr::GprError>
        {
            received.x_fitted = std::string(state);
            return std::make_unique<TextFittedInput>(ids.x_fitted, std::string(state));
        }
            ).has_value()
        );
        ASSERT_TRUE(
            reg.register_unfitted_target(
                   ids.y_unfitted,
                   [&received, ids](std::string_view state)
                       -> std::expected<std::unique_ptr<cxxgpr::transform::UnfittedTargetTransform>, cxxgpr::GprError>
        {
            received.y_unfitted = std::string(state);
            return std::make_unique<TextUnfittedTarget>(
                ids.y_unfitted, std::string(state), TextFittedTarget(ids.y_fitted, "")
            );
        }
            ).has_value()
        );
        ASSERT_TRUE(
            reg.register_fitted_target(
                   ids.y_fitted,
                   [&received, ids](std::string_view state)
                       -> std::expected<std::unique_ptr<cxxgpr::transform::FittedTargetTransform>, cxxgpr::GprError>
        {
            received.y_fitted = std::string(state);
            return std::make_unique<TextFittedTarget>(ids.y_fitted, std::string(state));
        }
            ).has_value()
        );
    }

    /// @brief Reads a whole file as bytes.
    /// @param[in] path File to read.
    /// @return The file contents (empty when the file cannot be opened).
    [[nodiscard]] std::string read_file_bytes(const std::filesystem::path& path)
    {
        std::ifstream ifs(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>()};
    }

    /// @brief Lists the file names directly under a directory, sorted.
    /// @param[in] dir Directory to list.
    /// @return Sorted entry names.
    [[nodiscard]] std::vector<std::string> list_dir(const std::filesystem::path& dir)
    {
        std::vector<std::string> names;
        for (const auto& entry : std::filesystem::directory_iterator(dir))
        {
            names.push_back(entry.path().filename().string());
        }
        std::ranges::sort(names);
        return names;
    }

    /// @brief Verifies save_with_factor() mmap roundtrip and save() refactor-on-load parity.
    /// @details Saves an L-BFGS model fitted with input/target transforms, checks bit-exact
    /// mmap predictions/LOO and typed extraction rejection/success, then checks refactor-on-load
    /// accuracy when saved without factors.
    TEST(PersistModelsTest, SaveWithFactorMmapAndSaveRefactorRoundtrip)
    {
        // Spec: FMT-01, FMT-02, FMT-03, FMT-04, FMT-05
        const auto spec =
            (cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.2).value())
                * cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(std::array<double, 2>{0.9, 1.3}).value()))
            + cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.02).value());

        const std::size_t n = 6;
        const std::size_t d = 2;
        const std::vector<double> x_train{
            -1.0, -0.4, 0.1, 0.6, 1.2, 1.7,  // col 0
            0.3, -0.7, 0.5, -0.2, 0.9, -0.5  // col 1
        };
        const std::vector<double> y_train{5.2, 4.1, 6.0, 5.8, 7.1, 4.9};
        const std::vector<double> x_query{-0.5, 0.8, 0.0, 0.4};

        const auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Lbfgs>(spec)
                                 .with_input_transform(cxxgpr::transform::StandardizeInput())
                                 .with_target_transform(cxxgpr::transform::StandardizeTarget())
                                 .with_noise_variance(0.05)
                                 .value();
        auto fitted = trainer.fit(x_train, n, d, y_train);
        ASSERT_TRUE(fitted.has_value());

        const auto xq_ref = cxxgpr::MatrixRef<double>(x_query.data(), 2U, d, 2U);
        auto orig_pred = fitted->predict(xq_ref);
        auto orig_loo = fitted->loo_predict();
        ASSERT_TRUE(orig_pred.has_value());
        ASSERT_TRUE(orig_loo.has_value());

        // 1. save_with_factor -> mmap F64 factor L.
        const TempDirGuard dir_factor("factor_roundtrip");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fitted, dir_factor.path).has_value());
        EXPECT_TRUE(std::filesystem::exists(dir_factor.path / "config.json"));
        EXPECT_TRUE(std::filesystem::exists(dir_factor.path / "model.safetensors"));

        {
            auto loaded = cxxgpr::persist::LoadedGpr::load(dir_factor.path);
            ASSERT_TRUE(loaded.has_value());
            EXPECT_EQ(loaded->n(), n);
            EXPECT_EQ(loaded->d(), d);
            EXPECT_EQ(loaded->precision_kind(), cxxgpr::PersistPrecisionKind::Double);
            EXPECT_TRUE(loaded->has_factor());
            EXPECT_TRUE(loaded->has_mmap_factor());

            auto loaded_pred = loaded->predict(xq_ref);
            auto loaded_loo = loaded->loo_predict();
            ASSERT_TRUE(loaded_pred.has_value());
            ASSERT_TRUE(loaded_loo.has_value());
            for (std::size_t j = 0; j < 2U; ++j)
            {
                EXPECT_EQ(
                    std::bit_cast<std::uint64_t>(loaded_pred->mean[j]), std::bit_cast<std::uint64_t>(orig_pred->mean[j])
                );
                EXPECT_EQ(
                    std::bit_cast<std::uint64_t>(loaded_pred->variance[j]),
                    std::bit_cast<std::uint64_t>(orig_pred->variance[j])
                );
            }
            for (std::size_t i = 0; i < n; ++i)
            {
                EXPECT_EQ(
                    std::bit_cast<std::uint64_t>(loaded_loo->mean[i]), std::bit_cast<std::uint64_t>(orig_loo->mean[i])
                );
            }

            // Typed extraction and wrong-precision rejection.
            auto wrong_typed = loaded->into_fitted<cxxgpr::optimizer::Lbfgs, cxxgpr::SinglePrecision>();
            ASSERT_FALSE(wrong_typed.has_value());
            EXPECT_EQ(wrong_typed.error().code(), cxxgpr::GprErrorCode::PersistFailed);
            EXPECT_EQ(wrong_typed.error().persist_error_kind(), cxxgpr::PersistErrorKind::WrongModel);

            auto typed = loaded->into_fitted<cxxgpr::optimizer::Lbfgs, cxxgpr::DoublePrecision>();
            ASSERT_TRUE(typed.has_value());
            EXPECT_NEAR(typed->neg_log_marginal_likelihood(), fitted->neg_log_marginal_likelihood(), 1e-12);
            auto typed_pred = typed->predict(xq_ref);
            ASSERT_TRUE(typed_pred.has_value());
            EXPECT_EQ(typed_pred->mean, orig_pred->mean);
        }

        // 2. save (has_factor = false) -> reconstructs L and alpha on load.
        const TempDirGuard dir_nofactor("nofactor_roundtrip");
        ASSERT_TRUE(cxxgpr::persist::save(*fitted, dir_nofactor.path).has_value());
        auto loaded_nf = cxxgpr::persist::LoadedGpr::load(dir_nofactor.path);
        ASSERT_TRUE(loaded_nf.has_value());
        EXPECT_FALSE(loaded_nf->has_factor());
        EXPECT_FALSE(loaded_nf->has_mmap_factor());
        auto pred_nf = loaded_nf->predict(xq_ref);
        ASSERT_TRUE(pred_nf.has_value());
        for (std::size_t j = 0; j < 2U; ++j)
        {
            EXPECT_NEAR(pred_nf->mean[j], orig_pred->mean[j], 1e-11);
            EXPECT_NEAR(pred_nf->variance[j], orig_pred->variance[j], 1e-11);
        }
        auto typed_nf = loaded_nf->into_fitted<cxxgpr::optimizer::Lbfgs, cxxgpr::DoublePrecision>();
        ASSERT_TRUE(typed_nf.has_value());
        EXPECT_NEAR(typed_nf->neg_log_marginal_likelihood(), fitted->neg_log_marginal_likelihood(), 1e-12);
    }

    /// @brief Verifies MixedPrecision<ReevaluateKernel> and CustomKernel PersistRegistry persistence.
    /// @details Confirms MixedReevaluate precision metadata and non-mmap load behavior, registry
    /// validation of empty/reserved/duplicate IDs, UnregisteredId rejection without registry, and
    /// predictive parity when loaded with a registered CustomKernel factory.
    TEST(PersistModelsTest, PrecisionVariantsAndRegistryRoundtrip)
    {
        // Spec: FMT-05, FMT-06
        const std::size_t n = 4;
        const std::size_t d = 1;
        const std::vector<double> x_train{0.0, 0.5, 1.0, 1.5};
        const std::vector<double> y_train{1.0, 1.2, 0.9, 0.5};
        const std::array<double, 1> xq{0.75};
        const auto xq_ref = cxxgpr::MatrixRef<double>(xq.data(), 1U, 1U, 1U);

        // MixedPrecision<ReevaluateKernel> roundtrip.
        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
        const auto mp_trainer =
            cxxgpr::Gpr<cxxgpr::optimizer::Fixed, cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>(spec);
        auto mp_fit = mp_trainer.factor(x_train, n, d, y_train);
        ASSERT_TRUE(mp_fit.has_value());

        const TempDirGuard dir_mp("mixed_reeval");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*mp_fit, dir_mp.path).has_value());
        auto mp_loaded = cxxgpr::persist::LoadedGpr::load(dir_mp.path);
        ASSERT_TRUE(mp_loaded.has_value());
        EXPECT_EQ(mp_loaded->precision_kind(), cxxgpr::PersistPrecisionKind::MixedReevaluate);
        EXPECT_FALSE(mp_loaded->has_mmap_factor());
        EXPECT_NEAR(mp_loaded->predict(xq_ref)->mean[0], mp_fit->predict(xq_ref)->mean[0], 1e-12);

        // CustomKernel + PersistRegistry validation & roundtrip.
        cxxgpr::persist::PersistRegistry reg;
        EXPECT_EQ(
            reg.register_kernel("", [](std::string_view) { return std::unexpected(cxxgpr::GprError::empty_input()); })
                .error()
                .persist_error_kind(),
            cxxgpr::PersistErrorKind::InvalidPersistId
        );
        EXPECT_EQ(
            reg.register_kernel(
                   "cxxgpr.reserved", [](std::string_view) { return std::unexpected(cxxgpr::GprError::empty_input()); }
            )
                .error()
                .persist_error_kind(),
            cxxgpr::PersistErrorKind::InvalidPersistId
        );

        auto make_custom = [](double scale)
        {
            return cxxgpr::CustomKernel::create(
                std::make_unique<CustomScaleKernel>(scale),
                std::make_unique<CustomScaleKernelFloat>(static_cast<float>(scale))
            )
                .value();
        };
        ASSERT_TRUE(
            reg.register_kernel(
                   "user.custom_scale",
                   [&](std::string_view state) -> std::expected<cxxgpr::CustomKernel, cxxgpr::GprError>
        {
            const double s = std::stod(std::string(state));
            return make_custom(s);
        }
            ).has_value()
        );
        // Duplicate registration must fail with InvalidPersistId.
        EXPECT_EQ(
            reg.register_kernel("user.custom_scale", [&](std::string_view) { return make_custom(1.0); })
                .error()
                .persist_error_kind(),
            cxxgpr::PersistErrorKind::InvalidPersistId
        );

        const cxxgpr::KernelSpec custom_spec(make_custom(2.5));
        const auto c_fit = cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(custom_spec).factor(x_train, n, d, y_train);
        ASSERT_TRUE(c_fit.has_value());

        const TempDirGuard dir_custom("custom_kernel");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*c_fit, dir_custom.path, reg).has_value());

        // Loading without registration fails with UnregisteredId.
        auto unreg_load = cxxgpr::persist::LoadedGpr::load(dir_custom.path);
        ASSERT_FALSE(unreg_load.has_value());
        EXPECT_EQ(unreg_load.error().persist_error_kind(), cxxgpr::PersistErrorKind::UnregisteredId);

        // Loading with registry succeeds.
        auto reg_load = cxxgpr::persist::LoadedGpr::load(dir_custom.path, reg);
        ASSERT_TRUE(reg_load.has_value());
        EXPECT_NEAR(reg_load->predict(xq_ref)->mean[0], c_fit->predict(xq_ref)->mean[0], 1e-12);
    }

    /// @brief Verifies custom state text that is stored as a JSON string comes back byte for byte at every site.
    /// @details Covers the five writer sites (kernel leaf, unfitted and fitted input transform, unfitted and fitted
    /// target transform) through both save() and save_with_factor(). A state that is itself a JSON string literal
    /// (`"abc"`, quotes included) used to come back without its quotes, because the loader hands any string value
    /// back as it is; text that is not JSON takes the same storage path.
    TEST(PersistModelsTest, CustomStateStoredAsStringRoundTripsVerbatimAtEverySite)
    {
        // Spec: FMT-06, FMT-03
        // Every case derives a different text per site from a tag, so a text handed to the wrong factory fails too.
        using MakeText = std::string (*)(const std::string&);
        const std::array<MakeText, 8> cases{
            // A JSON string literal.
            [](const std::string& tag) { return "\"" + tag + "\""; },
            // A JSON string literal surrounded by whitespace, which the JSON grammar allows.
            [](const std::string& tag) { return "  \"" + tag + "\"\n"; },
            // A JSON string literal containing escapes: the text is "<tag>\"é\n" with the backslashes kept.
            [](const std::string& tag) { return "\"" + tag + R"(\"é\n")"; },
            // Text that is not JSON.
            [](const std::string& tag) { return tag + " raw"; },
            // Text that starts like JSON but does not parse.
            [](const std::string& tag) { return "{" + tag; },
            // Non-ASCII UTF-8 text that is not JSON (U+72B6 U+614B).
            [](const std::string& tag) { return tag + " \xE7\x8A\xB6\xE6\x85\x8B"; },
            // The empty text.
            []([[maybe_unused]] const std::string& tag) { return std::string(); },
            // Text that is not JSON and opens more brackets than the nesting limit: as a string it nests nothing.
            [](const std::string& tag) { return std::string(200, '[') + tag; },
        };
        const std::array<double, 3> y_train{0.1, 0.9, 0.4};

        for (std::size_t c = 0; c < cases.size(); ++c)
        {
            for (const bool with_factor : {false, true})
            {
                SCOPED_TRACE("case " + std::to_string(c) + (with_factor ? " save_with_factor" : " save"));
                const ComponentTexts sent{
                    .kernel = cases[c]("kernel"),
                    .x_unfitted = cases[c]("x_unfitted"),
                    .x_fitted = cases[c]("x_fitted"),
                    .y_unfitted = cases[c]("y_unfitted"),
                    .y_fitted = cases[c]("y_fitted"),
                };
                const auto fit = factor_all_custom_model(default_component_ids(), sent, y_train);
                ASSERT_TRUE(fit.has_value());

                ComponentTexts received{
                    .kernel = "unset",
                    .x_unfitted = "unset",
                    .x_fitted = "unset",
                    .y_unfitted = "unset",
                    .y_fitted = "unset"
                };
                cxxgpr::persist::PersistRegistry reg;
                register_recording_factories(reg, received);
                const TempDirGuard dir("custom_state_verbatim");
                const auto saved = with_factor ? cxxgpr::persist::save_with_factor(*fit, dir.path, reg)
                                               : cxxgpr::persist::save(*fit, dir.path, reg);
                ASSERT_TRUE(saved.has_value());
                const auto loaded = cxxgpr::persist::LoadedGpr::load(dir.path, reg);
                ASSERT_TRUE(loaded.has_value());
                EXPECT_EQ(received.kernel, sent.kernel);
                EXPECT_EQ(received.x_unfitted, sent.x_unfitted);
                EXPECT_EQ(received.x_fitted, sent.x_fitted);
                EXPECT_EQ(received.y_unfitted, sent.y_unfitted);
                EXPECT_EQ(received.y_fitted, sent.y_fitted);
            }
        }
    }

    /// @brief Verifies custom state that is a JSON value other than a string is stored as that value.
    /// @details Such a state is not handed back verbatim: the factory receives the compact re-serialization of
    /// the parsed value. The expected texts are what the format specification documents, one per kind of change.
    TEST(PersistModelsTest, CustomStateThatIsNonStringJsonIsHandedBackReserialized)
    {
        // Spec: FMT-06, FMT-03
        struct Case
        {
            /// @brief Text returned by persist_state_json().
            std::string sent;
            /// @brief Text the factory must receive.
            std::string received;
        };

        const std::array<Case, 12> cases{
            Case{               .sent = "2.500000",                    .received = "2.5"}, // trailing zeros dropped
            Case{                    .sent = "1e2",                  .received = "100.0"}, // exponent form normalized
            Case{                     .sent = "-0",                      .received = "0"}, // sign of integer zero lost
            Case{                   .sent = "-0.0",                   .received = "-0.0"},
            Case{                    .sent = "0.1",                    .received = "0.1"}, // shortest round-trip form
            // An integer beyond 64 bits becomes the nearest double: digits are lost.
            Case{.sent = "12345678901234567890123", .received = "1.2345678901234568e+22"},
            Case{      .sent = R"({"b":1, "a":2})",       .received = R"({"a":2,"b":1})"}, // keys sorted, spaces dropped
            Case{       .sent = R"({"a":1,"a":2})",             .received = R"({"a":2})"}, // last duplicate key wins
            Case{              .sent = "[1, 2 ,3]",                .received = "[1,2,3]"},
            Case{                 .sent = " true ",                   .received = "true"},
            Case{                   .sent = "null",                   .received = "null"},
            Case{         .sent = R"({"s":"A\/"})",          .received = R"({"s":"A/"})"}, // escapes inside normalized
        };
        const std::array<double, 3> y_train{0.1, 0.9, 0.4};

        for (const auto& c : cases)
        {
            SCOPED_TRACE(c.sent);
            const ComponentTexts sent{
                .kernel = c.sent, .x_unfitted = c.sent, .x_fitted = c.sent, .y_unfitted = c.sent, .y_fitted = c.sent
            };
            const auto fit = factor_all_custom_model(default_component_ids(), sent, y_train);
            ASSERT_TRUE(fit.has_value());
            ComponentTexts received;
            cxxgpr::persist::PersistRegistry reg;
            register_recording_factories(reg, received);
            const TempDirGuard dir("custom_state_reserialized");
            ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit, dir.path, reg).has_value());
            ASSERT_TRUE(cxxgpr::persist::LoadedGpr::load(dir.path, reg).has_value());
            EXPECT_EQ(received.kernel, c.received);
            EXPECT_EQ(received.x_unfitted, c.received);
            EXPECT_EQ(received.x_fitted, c.received);
            EXPECT_EQ(received.y_unfitted, c.received);
            EXPECT_EQ(received.y_fitted, c.received);
        }
    }

    /// @brief Negative control: custom text that cannot be written into UTF-8 JSON makes save fail as a value.
    /// @details config.json is UTF-8, so state text that has to be stored as a string, and every persistence
    /// identifier, must be valid UTF-8. The JSON serializer throws on anything else; save() and save_with_factor()
    /// must instead return an error, write nothing, and leave an existing directory byte for byte as it was.
    TEST(PersistModelsTest, SaveRejectsCustomTextThatIsNotValidUtf8WithoutTouchingFiles)
    {
        // Spec: FMT-06, FMT-08, FMT-02
        const std::array<double, 3> y_train{0.1, 0.9, 0.4};
        const ComponentTexts good_states{
            .kernel = "\"k\"",
            .x_unfitted = "\"xu\"",
            .x_fitted = "\"xf\"",
            .y_unfitted = "\"yu\"",
            .y_fitted = "\"yf\""
        };
        const TempDirGuard dir("invalid_utf8_existing");
        const TempDirGuard fresh_parent("invalid_utf8_fresh");
        const std::filesystem::path fresh = fresh_parent.path / "model";
        {
            const auto base = factor_all_custom_model(default_component_ids(), good_states, y_train);
            ASSERT_TRUE(base.has_value());
            ASSERT_TRUE(cxxgpr::persist::save_with_factor(*base, dir.path).has_value());
        }
        const std::string config_before = read_file_bytes(dir.path / "config.json");
        const std::string tensors_before = read_file_bytes(dir.path / "model.safetensors");
        ASSERT_FALSE(config_before.empty());
        ASSERT_FALSE(tensors_before.empty());

        auto expect_rejected =
            [&](const ComponentTexts& ids, const ComponentTexts& states, cxxgpr::PersistErrorKind expected_kind)
        {
            // Different targets, so that the model differs from the one already in dir.
            const auto fit = factor_all_custom_model(ids, states, std::array<double, 3>{0.7, -0.2, 0.3});
            ASSERT_TRUE(fit.has_value());
            for (const bool with_factor : {false, true})
            {
                for (const auto& target : {dir.path, fresh})
                {
                    const auto saved = with_factor ? cxxgpr::persist::save_with_factor(*fit, target)
                                                   : cxxgpr::persist::save(*fit, target);
                    ASSERT_FALSE(saved.has_value());
                    EXPECT_EQ(saved.error().code(), cxxgpr::GprErrorCode::PersistFailed);
                    EXPECT_EQ(saved.error().persist_error_kind(), expected_kind);
                }
            }
            EXPECT_EQ(list_dir(dir.path), (std::vector<std::string>{"config.json", "model.safetensors"}));
            EXPECT_EQ(read_file_bytes(dir.path / "config.json"), config_before);
            EXPECT_EQ(read_file_bytes(dir.path / "model.safetensors"), tensors_before);
            EXPECT_FALSE(std::filesystem::exists(fresh));
        };

        const std::array<std::string ComponentTexts::*, 5> sites{
            &ComponentTexts::kernel, &ComponentTexts::x_unfitted, &ComponentTexts::x_fitted,
            &ComponentTexts::y_unfitted, &ComponentTexts::y_fitted
        };
        // Bytes that are not UTF-8: a stray 0xFF 0xFE, a truncated three-byte sequence, an overlong encoding,
        // an encoded surrogate, and an invalid byte inside what looks like a JSON string literal.
        const std::array<std::string, 5> bad_texts{"\xff\xfe", "abc\xe3\x81", "\xc0\xaf", "\xed\xa0\x80", "\"\xff\""};
        for (std::size_t s = 0; s < sites.size(); ++s)
        {
            for (std::size_t b = 0; b < bad_texts.size(); ++b)
            {
                SCOPED_TRACE("state: site " + std::to_string(s) + ", text " + std::to_string(b));
                ComponentTexts states = good_states;
                states.*sites[s] = bad_texts[b];
                expect_rejected(default_component_ids(), states, cxxgpr::PersistErrorKind::NotPersistable);
            }
            SCOPED_TRACE("identifier: site " + std::to_string(s));
            ComponentTexts ids = default_component_ids();
            ids.*sites[s] = "user.\xff";
            expect_rejected(ids, good_states, cxxgpr::PersistErrorKind::InvalidPersistId);
        }

        // The same identifier rule applies at registration time.
        const auto direct = cxxgpr::persist::validate_persist_id("user.\xff");
        ASSERT_FALSE(direct.has_value());
        EXPECT_EQ(direct.error().persist_error_kind(), cxxgpr::PersistErrorKind::InvalidPersistId);
        EXPECT_TRUE(cxxgpr::persist::validate_persist_id("user.\xE7\x8A\xB6").has_value());
        cxxgpr::persist::PersistRegistry reg;
        const std::array<std::expected<void, cxxgpr::GprError>, 5> registrations{
            reg.register_kernel(
                "user.\xff", [](std::string_view) -> std::expected<cxxgpr::CustomKernel, cxxgpr::GprError>
        { return std::unexpected(cxxgpr::GprError::empty_input()); }
            ),
            reg.register_unfitted_input(
                "user.\xff",
                [](std::string_view)
                    -> std::expected<std::unique_ptr<cxxgpr::transform::UnfittedInputTransform>, cxxgpr::GprError>
        { return std::unexpected(cxxgpr::GprError::empty_input()); }
            ),
            reg.register_fitted_input(
                "user.\xff",
                [](std::string_view)
                    -> std::expected<std::unique_ptr<cxxgpr::transform::FittedInputTransform>, cxxgpr::GprError>
        { return std::unexpected(cxxgpr::GprError::empty_input()); }
            ),
            reg.register_unfitted_target(
                "user.\xff",
                [](std::string_view)
                    -> std::expected<std::unique_ptr<cxxgpr::transform::UnfittedTargetTransform>, cxxgpr::GprError>
        { return std::unexpected(cxxgpr::GprError::empty_input()); }
            ),
            reg.register_fitted_target(
                "user.\xff",
                [](std::string_view)
                    -> std::expected<std::unique_ptr<cxxgpr::transform::FittedTargetTransform>, cxxgpr::GprError>
        { return std::unexpected(cxxgpr::GprError::empty_input()); }
            ),
        };
        for (const auto& registration : registrations)
        {
            ASSERT_FALSE(registration.has_value());
            EXPECT_EQ(registration.error().persist_error_kind(), cxxgpr::PersistErrorKind::InvalidPersistId);
        }
    }

    /// @brief Negative control: an optional top-level string field of the wrong JSON type is a Config error.
    /// @details `precision`, `residual`, `math`, and `distance_cache` may be omitted, but when present they must be
    /// strings. A number, boolean, array, or null there used to raise the JSON library's type error out of load().
    TEST(PersistModelsTest, LoadRejectsOptionalStringFieldsOfTheWrongTypeWithConfig)
    {
        // Spec: FMT-03, FMT-08
        const std::array<double, 2> x_train{0.0, 1.0};
        const std::array<double, 2> y_train{0.5, 1.5};
        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
        const auto fit = cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(spec).factor(x_train, 2U, 1U, y_train);
        ASSERT_TRUE(fit.has_value());
        const TempDirGuard dir("optional_field_type");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit, dir.path).has_value());
        const std::string good = read_file_bytes(dir.path / "config.json");

        struct Case
        {
            /// @brief Text present in the saved config.json.
            std::string from;
            /// @brief Replacement giving one field a non-string value.
            std::string to;
        };

        const std::array<Case, 4> cases{
            Case{     .from = R"("precision": "double")",                          .to = R"("precision": 1)"},
            Case{     .from = R"("precision": "double")", .to = R"("precision": "double", "residual": null)"},
            Case{        .from = R"("math": "accurate")",                            .to = R"("math": true)"},
            Case{.from = R"("distance_cache": "always")",            .to = R"("distance_cache": ["always"])"},
        };
        for (const auto& c : cases)
        {
            SCOPED_TRACE(c.to);
            const auto pos = good.find(c.from);
            ASSERT_NE(pos, std::string::npos);
            std::string bad = good;
            bad.replace(pos, c.from.size(), c.to);
            {
                std::ofstream ofs(dir.path / "config.json", std::ios::binary | std::ios::trunc);
                ofs << bad;
            }
            const auto loaded = cxxgpr::persist::LoadedGpr::load(dir.path);
            ASSERT_FALSE(loaded.has_value());
            EXPECT_EQ(loaded.error().code(), cxxgpr::GprErrorCode::PersistFailed);
            EXPECT_EQ(loaded.error().persist_error_kind(), cxxgpr::PersistErrorKind::Config);
        }
    }

    /// @brief Factors a four-point double-precision RBF model with fixed hyperparameters.
    /// @param[in] length_scale RBF length scale.
    /// @param[in] x_train Four training inputs (one feature).
    /// @param[in] y_train Four training targets.
    /// @return The factored model.
    [[nodiscard]] auto factor_rbf_model(
        double length_scale, const std::array<double, 4>& x_train, const std::array<double, 4>& y_train
    )
    {
        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(length_scale).value());
        return cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(spec).factor(x_train, 4U, 1U, y_train);
    }

    /// @brief Verifies saving over a directory whose factor a live LoadedGpr has mapped never corrupts that model.
    /// @details save replaces files by renaming, never by rewriting in place. Where the platform lets a mapped file
    /// be renamed over, the save succeeds, the live model keeps answering from the old bytes, and a new load sees
    /// the new model. Windows refuses the rename: the save fails with Io and the directory keeps the old model.
    TEST(PersistModelsTest, SaveOverAMappedFactorLeavesTheLoadedModelIntact)
    {
        // Spec: FMT-02, FMT-05
        const auto fit_old = factor_rbf_model(1.0, {0.0, 0.5, 1.0, 1.5}, {1.0, 1.2, 0.9, 0.5});
        const auto fit_new = factor_rbf_model(0.4, {0.1, 0.7, 1.1, 1.9}, {-0.3, 0.8, 0.2, 1.4});
        ASSERT_TRUE(fit_old.has_value());
        ASSERT_TRUE(fit_new.has_value());
        const std::array<double, 2> xq{0.75, 1.3};
        const auto xq_ref = cxxgpr::MatrixRef<double>(xq.data(), 2U, 1U, 2U);
        const auto pred_old = fit_old->predict(xq_ref);
        const auto pred_new = fit_new->predict(xq_ref);
        const auto loo_old = fit_old->loo_predict();
        ASSERT_TRUE(pred_old.has_value());
        ASSERT_TRUE(pred_new.has_value());
        ASSERT_TRUE(loo_old.has_value());
        ASSERT_NE(pred_old->mean, pred_new->mean);
        ASSERT_NE(pred_old->variance, pred_new->variance);

        const TempDirGuard dir("save_over_mapped");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit_old, dir.path).has_value());
        {
            auto loaded = cxxgpr::persist::LoadedGpr::load(dir.path);
            ASSERT_TRUE(loaded.has_value());
            ASSERT_TRUE(loaded->has_mmap_factor());

            const auto resaved = cxxgpr::persist::save_with_factor(*fit_new, dir.path);
#ifdef _WIN32
            ASSERT_FALSE(resaved.has_value());
            EXPECT_EQ(resaved.error().code(), cxxgpr::GprErrorCode::PersistFailed);
            EXPECT_EQ(resaved.error().persist_error_kind(), cxxgpr::PersistErrorKind::Io);
            const auto& pred_on_disk = pred_old;
#else
            ASSERT_TRUE(resaved.has_value());
            const auto& pred_on_disk = pred_new;
#endif
            EXPECT_EQ(list_dir(dir.path), (std::vector<std::string>{"config.json", "model.safetensors"}));

            // The live model still answers from the factor it mapped (the variance and LOO read L).
            const auto pred_live = loaded->predict(xq_ref);
            const auto loo_live = loaded->loo_predict();
            ASSERT_TRUE(pred_live.has_value());
            ASSERT_TRUE(loo_live.has_value());
            EXPECT_EQ(pred_live->mean, pred_old->mean);
            EXPECT_EQ(pred_live->variance, pred_old->variance);
            EXPECT_EQ(loo_live->mean, loo_old->mean);
            EXPECT_EQ(loo_live->variance, loo_old->variance);

            // A new load reads what the directory now holds.
            auto reloaded = cxxgpr::persist::LoadedGpr::load(dir.path);
            ASSERT_TRUE(reloaded.has_value());
            const auto pred_reloaded = reloaded->predict(xq_ref);
            ASSERT_TRUE(pred_reloaded.has_value());
            EXPECT_EQ(pred_reloaded->mean, pred_on_disk->mean);
            EXPECT_EQ(pred_reloaded->variance, pred_on_disk->variance);
        }
    }

    /// @brief Negative control: when config.json cannot be replaced, save reports Io and leaves no temporary file.
    /// @details The two files are replaced one after the other, model.safetensors first. This test makes the second
    /// replacement fail and pins down what the format specification states about that case: the error is Io, the
    /// temporary files are gone, config.json is untouched, and model.safetensors is already the new one.
    TEST(PersistModelsTest, SaveReportsIoWhenConfigJsonCannotBeReplaced)
    {
        // Spec: FMT-02, FMT-08
        const auto fit_old = factor_rbf_model(1.0, {0.0, 0.5, 1.0, 1.5}, {1.0, 1.2, 0.9, 0.5});
        const auto fit_new = factor_rbf_model(0.4, {0.1, 0.7, 1.1, 1.9}, {-0.3, 0.8, 0.2, 1.4});
        ASSERT_TRUE(fit_old.has_value());
        ASSERT_TRUE(fit_new.has_value());
        const TempDirGuard dir("config_not_replaceable");
        const TempDirGuard reference("config_not_replaceable_ref");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit_old, dir.path).has_value());
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit_new, reference.path).has_value());
        const std::string tensors_old = read_file_bytes(dir.path / "model.safetensors");
        const std::string tensors_new = read_file_bytes(reference.path / "model.safetensors");
        ASSERT_NE(tensors_old, tensors_new);

        // A file cannot be renamed over a directory, so a directory named config.json blocks the second rename.
        std::error_code ec;
        ASSERT_TRUE(std::filesystem::remove(dir.path / "config.json", ec));
        ASSERT_TRUE(std::filesystem::create_directory(dir.path / "config.json", ec));

        const auto saved = cxxgpr::persist::save_with_factor(*fit_new, dir.path);
        ASSERT_FALSE(saved.has_value());
        EXPECT_EQ(saved.error().code(), cxxgpr::GprErrorCode::PersistFailed);
        EXPECT_EQ(saved.error().persist_error_kind(), cxxgpr::PersistErrorKind::Io);
        EXPECT_EQ(list_dir(dir.path), (std::vector<std::string>{"config.json", "model.safetensors"}));
        EXPECT_TRUE(std::filesystem::is_directory(dir.path / "config.json"));
        EXPECT_EQ(read_file_bytes(dir.path / "model.safetensors"), tensors_new);
    }

    /// @brief Negative control: save refuses a model whose config.json would be nested deeper than load accepts.
    /// @details The loader rejects a config.json nested more than 128 levels deep. save() must apply the same
    /// limit to the document it is about to write, so that it never writes a directory that cannot be loaded, and
    /// must bound the nesting of custom state before handing it to the recursive JSON serializer. Each case checks
    /// the deepest model that still round-trips and the shallowest one that is refused.
    TEST(PersistModelsTest, SaveRejectsAConfigNestedDeeperThanLoadAccepts)
    {
        // Spec: FMT-02, FMT-08
        const std::array<double, 3> x_train{0.0, 1.0, 2.0};
        const std::array<double, 3> y_train{0.1, 0.9, 0.4};
        const TempDirGuard parent("save_nesting_limit");
        const std::filesystem::path dir = parent.path / "model";

        // Saves into a fresh directory and checks either a loadable result or a Config error that wrote nothing.
        auto expect_outcome = [&](const auto& fit, const cxxgpr::persist::PersistRegistry& reg, bool accepted)
        {
            ASSERT_TRUE(fit.has_value());
            std::error_code ec;
            std::filesystem::remove_all(dir, ec);
            const auto saved = cxxgpr::persist::save_with_factor(*fit, dir, reg);
            if (accepted)
            {
                ASSERT_TRUE(saved.has_value());
                EXPECT_TRUE(cxxgpr::persist::LoadedGpr::load(dir, reg).has_value());
                return;
            }
            ASSERT_FALSE(saved.has_value());
            EXPECT_EQ(saved.error().code(), cxxgpr::GprErrorCode::PersistFailed);
            EXPECT_EQ(saved.error().persist_error_kind(), cxxgpr::PersistErrorKind::Config);
            EXPECT_FALSE(std::filesystem::exists(dir));
        };
        const cxxgpr::persist::PersistRegistry no_registry;
        const auto rbf = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
        const auto white = cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.01).value());
        const auto constant = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.1).value());

        // Below the root object, a composite kernel node adds two levels ({"sum": {"left": ...}}) and the leaf adds
        // three ({"rbf": {"length_scale": {"value": ...}}}): a left-deep chain of m composites reaches depth
        // 2m + 4, so m = 62 gives 128 and m = 63 gives 130.
        for (const bool product : {false, true})
        {
            for (const std::size_t m : {std::size_t{62}, std::size_t{63}})
            {
                SCOPED_TRACE(std::string(product ? "product" : "sum") + " chain of " + std::to_string(m));
                auto spec = rbf;
                for (std::size_t i = 0; i < m; ++i)
                {
                    spec = product ? spec * constant : spec + white;
                }
                expect_outcome(
                    cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(spec).factor(x_train, 3U, 1U, y_train), no_registry, m == 62U
                );
            }
        }

        // A pipeline adds three levels ({"pipeline": {"steps": [...]}}) below the root object: m nested pipelines
        // reach depth 3m + 1, so m = 42 gives 127 and m = 43 gives 130.
        for (const std::size_t m : {std::size_t{42}, std::size_t{43}})
        {
            SCOPED_TRACE("pipelines nested " + std::to_string(m));
            cxxgpr::transform::Pipeline input_pipeline;
            cxxgpr::transform::TargetPipeline target_pipeline;
            for (std::size_t i = 1; i < m; ++i)
            {
                input_pipeline = cxxgpr::transform::Pipeline().then(input_pipeline);
                target_pipeline = cxxgpr::transform::TargetPipeline().then(target_pipeline);
            }
            const cxxgpr::Gpr<cxxgpr::optimizer::Fixed> base(rbf);
            expect_outcome(
                base.with_input_transform(input_pipeline).factor(x_train, 3U, 1U, y_train), no_registry, m == 42U
            );
            expect_outcome(
                base.with_target_transform(target_pipeline).factor(x_train, 3U, 1U, y_train), no_registry, m == 42U
            );
        }

        // Custom state sits three levels below the root ({"kernel": {"custom": {"state": ...}}}, and likewise for
        // each transform): state nested k levels reaches depth k + 3, so k = 125 gives 128 and k = 126 gives 129.
        // The last depth would overflow the stack inside the JSON serializer if it were not refused beforehand.
        const std::array<std::string ComponentTexts::*, 5> sites{
            &ComponentTexts::kernel, &ComponentTexts::x_unfitted, &ComponentTexts::x_fitted,
            &ComponentTexts::y_unfitted, &ComponentTexts::y_fitted
        };
        ComponentTexts received;
        cxxgpr::persist::PersistRegistry reg;
        register_recording_factories(reg, received);
        for (std::size_t s = 0; s < sites.size(); ++s)
        {
            for (const std::size_t k : {std::size_t{125}, std::size_t{126}, std::size_t{1'000'000}})
            {
                SCOPED_TRACE("site " + std::to_string(s) + ", state nested " + std::to_string(k));
                ComponentTexts states{
                    .kernel = "1", .x_unfitted = "1", .x_fitted = "1", .y_unfitted = "1", .y_fitted = "1"
                };
                states.*sites[s] = std::string(k, '[') + std::string(k, ']');
                expect_outcome(factor_all_custom_model(default_component_ids(), states, y_train), reg, k == 125U);
            }
        }
    }

    /// @brief Property: save and load of the deepest accepted models, and save of models refused for nesting, run on
    /// a thread with a small stack.
    /// @details The config.json codec recurses once per kernel composite and per transform pipeline level. The
    /// models are fitted on the main thread; every save, load, and destruction of a loaded model runs on the small
    /// stack. The deepest accepted chains are the ones of SaveRejectsAConfigNestedDeeperThanLoadAccepts. Of the
    /// refused models, the kernel chain drives the kernel encoder to its own recursion limit of 128 levels, and the
    /// pipelines are the shallowest ones the document check refuses. Deeper pipelines are not fitted here, since
    /// fitting them recurses through the transforms on the main thread.
    TEST(PersistModelsTest, SaveAndLoadOfTheDeepestAcceptedModelsFitInASmallStack)
    {
        // Spec: FMT-02, FMT-08
        const std::array<double, 3> x_train{0.0, 1.0, 2.0};
        const std::array<double, 3> y_train{0.1, 0.9, 0.4};
        const TempDirGuard parent("deep_model_small_stack");
        const cxxgpr::persist::PersistRegistry no_registry;
        const auto rbf = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
        const auto white = cxxgpr::KernelSpec(cxxgpr::WhiteKernel::create(0.01).value());
        const auto constant = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.1).value());
        const cxxgpr::Gpr<cxxgpr::optimizer::Fixed> base(rbf);

        // Left-deep chains of m composites reach depth 2m + 4: 62 is accepted, while 129 composites exceed the
        // kernel encoder's own recursion limit of 128.
        auto chain = [&](std::size_t m, bool product)
        {
            auto spec = rbf;
            for (std::size_t i = 0; i < m; ++i)
            {
                spec = product ? spec * constant : spec + white;
            }
            return cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(spec).factor(x_train, 3U, 1U, y_train);
        };
        // m nested pipelines reach depth 3m + 1: 42 is accepted and 43 is refused.
        auto pipelines = [](std::size_t m)
        {
            cxxgpr::transform::Pipeline input;
            cxxgpr::transform::TargetPipeline target;
            for (std::size_t i = 1; i < m; ++i)
            {
                input = cxxgpr::transform::Pipeline().then(input);
                target = cxxgpr::transform::TargetPipeline().then(target);
            }
            return std::pair{std::move(input), std::move(target)};
        };
        const auto [input42, target42] = pipelines(42U);
        const auto [input43, target43] = pipelines(43U);
        const auto sum62 = chain(62U, false);
        const auto product62 = chain(62U, true);
        const auto sum129 = chain(129U, false);
        const auto input_deep = base.with_input_transform(input42).factor(x_train, 3U, 1U, y_train);
        const auto target_deep = base.with_target_transform(target42).factor(x_train, 3U, 1U, y_train);
        const auto input_refused = base.with_input_transform(input43).factor(x_train, 3U, 1U, y_train);
        const auto target_refused = base.with_target_transform(target43).factor(x_train, 3U, 1U, y_train);
        for (const auto* fit :
            {&sum62, &product62, &sum129, &input_deep, &target_deep, &input_refused, &target_refused})
        {
            ASSERT_TRUE(fit->has_value());
        }

        // Outcomes are recorded on the small stack and checked on the main thread.
        std::array<bool, 4> saved{};
        std::array<bool, 4> loaded{};
        std::array<std::optional<cxxgpr::GprError>, 3> refused;
        const bool ran = cxxgpr::test_support::run_with_stack_size(
            cxxgpr::test_support::deep_model_stack_bytes(), [&]
        {
            const std::array accepted{&*sum62, &*product62, &*input_deep, &*target_deep};
            for (std::size_t i = 0; i < accepted.size(); ++i)
            {
                const auto dir = parent.path / ("accepted_" + std::to_string(i));
                saved.at(i) = cxxgpr::persist::save_with_factor(*accepted.at(i), dir, no_registry).has_value();
                loaded.at(i) = cxxgpr::persist::LoadedGpr::load(dir, no_registry).has_value();
            }
            const std::array too_deep{&*sum129, &*input_refused, &*target_refused};
            for (std::size_t i = 0; i < too_deep.size(); ++i)
            {
                auto result =
                    cxxgpr::persist::save_with_factor(*too_deep.at(i), parent.path / ("refused_" + std::to_string(i)));
                if (!result.has_value())
                {
                    refused.at(i) = std::move(result).error();
                }
            }
        }
        );
        ASSERT_TRUE(ran);
        for (std::size_t i = 0; i < saved.size(); ++i)
        {
            SCOPED_TRACE("accepted model " + std::to_string(i));
            EXPECT_TRUE(saved.at(i));
            EXPECT_TRUE(loaded.at(i));
        }
        for (std::size_t i = 0; i < refused.size(); ++i)
        {
            SCOPED_TRACE("refused model " + std::to_string(i));
            const auto& error = refused.at(i);
            if (!error.has_value())
            {
                ADD_FAILURE() << "save accepted a model nested deeper than load accepts";
                continue;
            }
            EXPECT_EQ(error->code(), cxxgpr::GprErrorCode::PersistFailed);
            EXPECT_EQ(error->persist_error_kind(), cxxgpr::PersistErrorKind::Config);
        }
    }

    /// @brief Verifies the raw-span overload of LoadedGpr::predict against the span shape contract.
    /// @details Runs the shared shape cases (short span, long span, overflowing dimensions, empty span with a null
    /// or non-null pointer, zero dimension) on a loaded model, checks that a wrong length is reported before a
    /// wrong feature count, and that the loaded model predicts bitwise identically afterwards.
    TEST(PersistModelsTest, LoadedGprSpanPredictRejectsLengthsThatDisagreeWithDimensions)
    {
        // Spec: REQ-03, DES-02
        using cxxgpr::test_support::error_code_of;
        const std::vector<double> x_train{0.0, 1.0};
        const std::vector<double> y_train{0.5, 1.5};
        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
        const auto fit = cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(spec).factor(x_train, 2U, 1U, y_train);
        ASSERT_TRUE(fit.has_value());
        const TempDirGuard dir("span_shape_contract");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit, dir.path).has_value());
        auto loaded = cxxgpr::persist::LoadedGpr::load(dir.path);
        ASSERT_TRUE(loaded.has_value());

        const std::vector<double> query{0.5, 1.5};
        const auto before = loaded->predict(query, 2U, 1U);
        ASSERT_TRUE(before.has_value());

        cxxgpr::test_support::expect_span_shape_contract(
            [&](std::span<const double> xs, std::size_t n_rows, std::size_t n_cols)
        { return error_code_of(loaded->predict(xs, n_rows, n_cols)); }, query, 2U, 1U
        );

        // Three values declared as 2 x 2: the length is reported before the feature count.
        const std::vector<double> q3{0.5, 1.5, 2.5};
        EXPECT_EQ(error_code_of(loaded->predict(q3, 2U, 2U)), std::optional{cxxgpr::GprErrorCode::LengthMismatch});

        const auto after = loaded->predict(query, 2U, 1U);
        ASSERT_TRUE(after.has_value());
        EXPECT_EQ(*after, *before);
    }

    /// @brief Returns the byte offset of element (row, col) of the stored column-major `l` tensor in a model file.
    /// @param[in] file Path of a `model.safetensors` file written by save_with_factor().
    /// @param[in] n Training sample count (the factor is n x n).
    /// @param[in] element_size Size in bytes of one stored factor element (8 for F64, 4 for F32).
    /// @param[in] row Row index of the element.
    /// @param[in] col Column index of the element.
    /// @return Absolute byte offset of the element, or 0 when the header cannot be parsed.
    std::size_t stored_factor_offset(
        const std::filesystem::path& file, std::size_t n, std::size_t element_size, std::size_t row, std::size_t col
    )
    {
        std::ifstream in(file, std::ios::binary);
        std::uint64_t header_len = 0;
        in.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
        std::string header(static_cast<std::size_t>(header_len), '\0');
        in.read(header.data(), static_cast<std::streamsize>(header.size()));
        const auto key = header.find("\"l\":");
        const auto offsets = (key == std::string::npos) ? key : header.find("\"data_offsets\":[", key);
        if (!in || offsets == std::string::npos)
        {
            return 0U;
        }
        const auto begin = static_cast<std::size_t>(std::stoull(header.substr(offsets + 16U)));
        return sizeof(header_len) + static_cast<std::size_t>(header_len) + begin + (((col * n) + row) * element_size);
    }

    /// @brief Overwrites element (row, col) of the stored `l` tensor in place with value in storage precision.
    /// @tparam Storage Stored factor scalar type (double for F64, float for F32).
    /// @param[in] file Path of a `model.safetensors` file written by save_with_factor().
    /// @param[in] n Training sample count (the factor is n x n).
    /// @param[in] row Row index of the element.
    /// @param[in] col Column index of the element.
    /// @param[in] value Replacement value.
    /// @return True when the element was located and rewritten.
    template <typename Storage>
    bool overwrite_stored_factor(
        const std::filesystem::path& file, std::size_t n, std::size_t row, std::size_t col, Storage value
    )
    {
        const std::size_t position = stored_factor_offset(file, n, sizeof(Storage), row, col);
        if (position == 0U || position + sizeof(Storage) > std::filesystem::file_size(file))
        {
            return false;
        }
        // Rewrite in place (no truncation) so an existing memory map of the file observes the change.
        std::fstream io(file, std::ios::binary | std::ios::in | std::ios::out);
        io.seekp(static_cast<std::streamoff>(position));
        io.write(reinterpret_cast<const char*>(&value), sizeof(Storage));
        io.flush();
        return static_cast<bool>(io);
    }

    /// @brief Verifies a corrupt Cholesky factor is rejected at load, because predictions no longer rescan it.
    /// @details LoadedGpr predictions skip the per-call factor validation and rely on load having checked every
    /// element for finiteness and the diagonal for positivity. A NaN below the diagonal and a zero diagonal entry
    /// in the stored `l` tensor must therefore fail `LoadedGpr::load` instead of reaching a prediction, for the
    /// memory-mapped F64 factor of DoublePrecision and the copied F32 factors of SinglePrecision and both
    /// MixedPrecision policies.
    TEST(PersistModelsTest, LoadRejectsCorruptCholeskyFactorBecausePredictionsDoNotRescanIt)
    {
        // Spec: FMT-02, FMT-05, DES-16
        const std::vector<double> x_train{0.0, 1.0, 2.0, 3.0};
        const std::vector<double> y_train{0.1, 0.9, 0.4, 0.7};
        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());

        const auto check_policy = [&]<typename P>(const std::string& tag)
        {
            using Storage = P::Storage;
            const auto fit = cxxgpr::Gpr<cxxgpr::optimizer::Fixed, P>(spec).factor(x_train, 4U, 1U, y_train);
            ASSERT_TRUE(fit.has_value()) << tag;

            const auto corrupt = [&](const std::string& name, std::size_t row, std::size_t col, Storage value)
            {
                const TempDirGuard dir(name + "_" + tag);
                ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit, dir.path).has_value());
                ASSERT_TRUE(overwrite_stored_factor<Storage>(dir.path / "model.safetensors", 4U, row, col, value));
                const auto loaded = cxxgpr::persist::LoadedGpr::load(dir.path);
                EXPECT_FALSE(loaded.has_value()) << name << " " << tag;
            };
            corrupt("corrupt_l_nan_below_diagonal", 2U, 1U, std::numeric_limits<Storage>::quiet_NaN());
            corrupt("corrupt_l_zero_diagonal", 1U, 1U, Storage{0});

            // The unmodified model still loads and predicts.
            const TempDirGuard clean("clean_factor_for_rescan_test_" + tag);
            ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit, clean.path).has_value());
            const auto loaded = cxxgpr::persist::LoadedGpr::load(clean.path);
            ASSERT_TRUE(loaded.has_value()) << tag;
            EXPECT_EQ(loaded->has_mmap_factor(), (std::same_as<Storage, double>)) << tag;
            const std::vector<double> query{0.5, 1.5};
            EXPECT_TRUE(loaded->predict(query, 2U, 1U).has_value()) << tag;
        };
        check_policy.template operator()<cxxgpr::DoublePrecision>("double");
        check_policy.template operator()<cxxgpr::SinglePrecision>("single");
        check_policy.template operator()<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>("mixed_promote");
        check_policy.template operator()<cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>("mixed_reevaluate");
    }

    /// @brief Verifies a factor rewritten in place after load makes predictions fail instead of collapsing variance.
    /// @details Rewriting the memory-mapped `model.safetensors` while a LoadedGpr maps it violates the mmap contract,
    /// and predictions do not re-check the factor. A NaN or zero written onto the diagonal after load must not be
    /// turned into a clamped zero latent variance (which would report only the noise variance); the non-finite
    /// latent variance is reported as NonFiniteKernelValue. Runs only where the map observes in-place writes (Linux).
    TEST(PersistModelsTest, FactorRewrittenAfterLoadFailsPredictionInsteadOfCollapsingVariance)
    {
        // Spec: FMT-05, DES-16
#ifndef __linux__
        GTEST_SKIP() << "an in-place write is observed through a private file mapping only on Linux";
#else
        const std::vector<double> x_train{0.0, 1.0, 2.0, 3.0};
        const std::vector<double> y_train{0.1, 0.9, 0.4, 0.7};
        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
        const auto fit = cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(spec).factor(x_train, 4U, 1U, y_train);
        ASSERT_TRUE(fit.has_value());
        const std::vector<double> query{0.5, 1.5};

        for (const double value : {std::numeric_limits<double>::quiet_NaN(), 0.0})
        {
            const TempDirGuard dir(std::isnan(value) ? "rewrite_after_load_nan" : "rewrite_after_load_zero");
            ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit, dir.path).has_value());
            const auto loaded = cxxgpr::persist::LoadedGpr::load(dir.path);
            ASSERT_TRUE(loaded.has_value());
            ASSERT_TRUE(loaded->has_mmap_factor());
            ASSERT_TRUE(loaded->predict(query, 2U, 1U).has_value());

            ASSERT_TRUE(overwrite_stored_factor<double>(dir.path / "model.safetensors", 4U, 1U, 1U, value));
            const auto pred = loaded->predict(query, 2U, 1U);
            ASSERT_FALSE(pred.has_value()) << value;
            EXPECT_EQ(pred.error().code(), cxxgpr::GprErrorCode::NonFiniteKernelValue) << value;
        }
#endif
    }

    /// @brief Negative controls verifying persistence load rejects missing, corrupt, and foreign artifacts.
    /// @details Checks Io errors on missing directories, Config errors on malformed or deeply nested
    /// JSON, UnsupportedPersistVersion, WrongModel rejections for sgpr/ldlt/distance configs, and
    /// Tensor errors on truncated or deeply nested safetensors headers.
    TEST(PersistModelsTest, NegativeControlsRejectCorruptAndUnsupportedFiles)
    {
        // Spec: FMT-01, FMT-02, FMT-08
        // Ensure loading from a non-existent directory fails with PersistErrorKind::Io.
        const TempDirGuard missing_dir("does_not_exist");
        std::error_code ec;
        std::filesystem::remove_all(missing_dir.path, ec);
        auto err_io = cxxgpr::persist::LoadedGpr::load(missing_dir.path);
        ASSERT_FALSE(err_io.has_value());
        EXPECT_EQ(err_io.error().persist_error_kind(), cxxgpr::PersistErrorKind::Io);

        const std::vector<double> x_train{0.0, 1.0};
        const std::vector<double> y_train{0.5, 1.5};
        const auto spec = cxxgpr::KernelSpec(cxxgpr::RbfKernel::create(1.0).value());
        const auto fit = cxxgpr::Gpr<cxxgpr::optimizer::Fixed>(spec).factor(x_train, 2U, 1U, y_train);
        ASSERT_TRUE(fit.has_value());

        // Helper lambda to mutate config.json and expect specific error.
        auto check_bad_config = [&](const std::string& name, const std::string& json_text,
                                    cxxgpr::GprErrorCode expected_code,
                                    std::optional<cxxgpr::PersistErrorKind> expected_pk)
        {
            const TempDirGuard dir("bad_" + name);
            ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit, dir.path).has_value());
            {
                std::ofstream ofs(dir.path / "config.json", std::ios::binary | std::ios::trunc);
                ofs << json_text;
            }
            auto res = cxxgpr::persist::LoadedGpr::load(dir.path);
            ASSERT_FALSE(res.has_value());
            EXPECT_EQ(res.error().code(), expected_code);
            if (expected_pk.has_value())
            {
                EXPECT_EQ(res.error().persist_error_kind(), *expected_pk);
            }
        };

        check_bad_config(
            "syntax", "{not valid json", cxxgpr::GprErrorCode::PersistFailed, cxxgpr::PersistErrorKind::Config
        );
        {
            std::string deep_json;
            for (std::size_t i = 0; i < 130U; ++i)
            {
                deep_json += "{\"a\":";
            }
            deep_json += "1";
            for (std::size_t i = 0; i < 130U; ++i)
            {
                deep_json += "}";
            }
            check_bad_config(
                "deep_nesting", deep_json, cxxgpr::GprErrorCode::PersistFailed, cxxgpr::PersistErrorKind::Config
            );
        }
        check_bad_config(
            "version", R"({"format_version": 99, "factor_kind": "llt"})",
            cxxgpr::GprErrorCode::UnsupportedPersistVersion, std::nullopt
        );
        check_bad_config(
            "wrong_model", R"({"format_version": 1, "model": "sgpr", "factor_kind": "llt"})",
            cxxgpr::GprErrorCode::PersistFailed, cxxgpr::PersistErrorKind::WrongModel
        );
        check_bad_config(
            "ldlt", R"({"format_version": 1, "factor_kind": "ldlt"})", cxxgpr::GprErrorCode::PersistFailed,
            cxxgpr::PersistErrorKind::WrongModel
        );
        check_bad_config(
            "distance", R"({"format_version": 1, "distance": {}, "factor_kind": "llt"})",
            cxxgpr::GprErrorCode::PersistFailed, cxxgpr::PersistErrorKind::WrongModel
        );

        // Corrupt safetensors header -> PersistFailed(Tensor).
        const TempDirGuard dir_st("bad_safetensors");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit, dir_st.path).has_value());
        {
            std::ofstream ofs(dir_st.path / "model.safetensors", std::ios::binary | std::ios::trunc);
            const std::uint64_t bad_len = 999999ULL;
            ofs.write(reinterpret_cast<const char*>(&bad_len), sizeof(bad_len));
        }
        auto err_st = cxxgpr::persist::LoadedGpr::load(dir_st.path);
        ASSERT_FALSE(err_st.has_value());
        EXPECT_EQ(err_st.error().persist_error_kind(), cxxgpr::PersistErrorKind::Tensor);

        // Deeply nested JSON in safetensors header -> PersistFailed(Tensor).
        const TempDirGuard dir_st_deep("bad_safetensors_deep");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit, dir_st_deep.path).has_value());
        {
            std::string deep_hdr;
            for (std::size_t i = 0; i < 130U; ++i)
            {
                deep_hdr += "{\"a\":";
            }
            deep_hdr += "1";
            for (std::size_t i = 0; i < 130U; ++i)
            {
                deep_hdr += "}";
            }
            while ((deep_hdr.size() % 8U) != 0U)
            {
                deep_hdr.push_back(' ');
            }
            const std::uint64_t hdr_len = deep_hdr.size();
            std::ofstream ofs(dir_st_deep.path / "model.safetensors", std::ios::binary | std::ios::trunc);
            ofs.write(reinterpret_cast<const char*>(&hdr_len), sizeof(hdr_len));
            ofs.write(deep_hdr.data(), static_cast<std::streamsize>(deep_hdr.size()));
        }
        auto err_st_deep = cxxgpr::persist::LoadedGpr::load(dir_st_deep.path);
        ASSERT_FALSE(err_st_deep.has_value());
        EXPECT_EQ(err_st_deep.error().persist_error_kind(), cxxgpr::PersistErrorKind::Tensor);
    }

    /// @brief Checks that query batch sizes may shrink and grow in any order on one LoadedGpr.
    /// @details Fits a 2-feature model under precision policy P with fitted input and target transforms,
    /// saves it, and loads it several times. One loaded model sees an 8-point batch first; the following
    /// 3-point batch must match bitwise a loaded model that never saw the larger batch, for predict,
    /// predict_covariance, and sample, and returning to 8 points must match a model that never saw the
    /// smaller batch. Finally audits that predict_into() allocates nothing for a smaller batch after a larger one.
    /// @tparam P Precision policy of the saved model.
    /// @param[in] dir_name Unique temporary directory suffix for this policy.
    /// @param[in] with_factor Whether the factor is stored (save_with_factor) or rebuilt on load (save).
    template <cxxgpr::PrecisionPolicyConcept P>
    void verify_loaded_query_batch_may_shrink(const std::string& dir_name, bool with_factor)
    {
        const std::size_t n = 20;
        const std::size_t d = 2;
        std::vector<double> x(n * d);
        std::vector<double> y(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            x[i] = 0.3 * static_cast<double>(i);
            x[n + i] = 0.25 * static_cast<double>((i * 5U) % 7U);
            y[i] = 4.0 + std::sin(x[i]) + (0.5 * x[n + i]);
        }
        const std::vector<double> big{0.1, 0.5, 0.9, 1.3, 1.7, 2.1, 2.5, 2.9, 0.2, 1.4, 0.6, 1.0, 0.3, 1.2, 0.8, 0.4};
        const std::vector<double> small{0.2, 1.1, 2.3, 0.7, 0.1, 1.3};
        const auto big_ref = cxxgpr::MatrixRef<double>(big.data(), 8U, d, 8U);
        const auto small_ref = cxxgpr::MatrixRef<double>(small.data(), 3U, d, 3U);

        const auto spec = cxxgpr::KernelSpec(cxxgpr::ConstantKernel::create(1.0).value())
            * cxxgpr::KernelSpec(cxxgpr::RbfArdKernel::create(std::array<double, 2>{1.0, 1.5}).value());
        const auto trainer = cxxgpr::Gpr<cxxgpr::optimizer::Fixed, P>(spec)
                                 .with_input_transform(cxxgpr::transform::StandardizeInput())
                                 .with_target_transform(cxxgpr::transform::StandardizeTarget())
                                 .with_noise_variance(1e-3);
        ASSERT_TRUE(trainer.has_value());
        const auto fitted = trainer->factor(x, n, d, y);
        ASSERT_TRUE(fitted.has_value());

        const TempDirGuard dir(dir_name);
        if (with_factor)
        {
            ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fitted, dir.path).has_value());
        }
        else
        {
            ASSERT_TRUE(cxxgpr::persist::save(*fitted, dir.path).has_value());
        }
        // Every load() yields an independent model with an empty query workspace.
        const auto load = [&]
        {
            return cxxgpr::persist::LoadedGpr::load(dir.path);
        };

        // predict: large -> small -> large on one loaded model.
        {
            auto used = load();
            auto never_big = load();
            auto never_small = load();
            ASSERT_TRUE(used.has_value());
            ASSERT_TRUE(never_big.has_value());
            ASSERT_TRUE(never_small.has_value());
            ASSERT_TRUE(used->predict(big_ref).has_value());
            auto shrunk = used->predict(small_ref);
            auto shrunk_expected = never_big->predict(small_ref);
            ASSERT_TRUE(shrunk.has_value()) << shrunk.error().message();
            ASSERT_TRUE(shrunk_expected.has_value());
            EXPECT_EQ(shrunk->mean, shrunk_expected->mean);
            EXPECT_EQ(shrunk->variance, shrunk_expected->variance);

            auto regrown = used->predict(big_ref);
            auto regrown_expected = never_small->predict(big_ref);
            ASSERT_TRUE(regrown.has_value()) << regrown.error().message();
            ASSERT_TRUE(regrown_expected.has_value());
            EXPECT_EQ(regrown->mean, regrown_expected->mean);
            EXPECT_EQ(regrown->variance, regrown_expected->variance);
        }

        // predict_covariance: large -> small -> large.
        {
            auto used = load();
            auto never_big = load();
            auto never_small = load();
            ASSERT_TRUE(used.has_value());
            ASSERT_TRUE(never_big.has_value());
            ASSERT_TRUE(never_small.has_value());
            ASSERT_TRUE(used->predict_covariance(big_ref).has_value());
            auto shrunk = used->predict_covariance(small_ref);
            auto shrunk_expected = never_big->predict_covariance(small_ref);
            ASSERT_TRUE(shrunk.has_value()) << shrunk.error().message();
            ASSERT_TRUE(shrunk_expected.has_value());
            EXPECT_EQ(shrunk->mean, shrunk_expected->mean);
            EXPECT_EQ(shrunk->covariance, shrunk_expected->covariance);

            auto regrown = used->predict_covariance(big_ref);
            auto regrown_expected = never_small->predict_covariance(big_ref);
            ASSERT_TRUE(regrown.has_value()) << regrown.error().message();
            ASSERT_TRUE(regrown_expected.has_value());
            EXPECT_EQ(regrown->mean, regrown_expected->mean);
            EXPECT_EQ(regrown->covariance, regrown_expected->covariance);
        }

        // sample with a fixed seed: large -> small -> large.
        {
            auto used = load();
            auto never_big = load();
            auto never_small = load();
            ASSERT_TRUE(used.has_value());
            ASSERT_TRUE(never_big.has_value());
            ASSERT_TRUE(never_small.has_value());
            ASSERT_TRUE(used->sample(big_ref, 2U, 7U).has_value());
            auto shrunk = used->sample(small_ref, 2U, 7U);
            auto shrunk_expected = never_big->sample(small_ref, 2U, 7U);
            ASSERT_TRUE(shrunk.has_value()) << shrunk.error().message();
            ASSERT_TRUE(shrunk_expected.has_value());
            EXPECT_EQ(*shrunk, *shrunk_expected);

            auto regrown = used->sample(big_ref, 2U, 7U);
            auto regrown_expected = never_small->sample(big_ref, 2U, 7U);
            ASSERT_TRUE(regrown.has_value()) << regrown.error().message();
            ASSERT_TRUE(regrown_expected.has_value());
            EXPECT_EQ(*regrown, *regrown_expected);
        }

        // predict_into: after the larger batch sized the workspace and the output, a smaller batch and a
        // return to the larger batch perform no heap allocation.
        {
            auto used = load();
            auto never_big = load();
            ASSERT_TRUE(used.has_value());
            ASSERT_TRUE(never_big.has_value());
            cxxgpr::Prediction<double> out;
            ASSERT_TRUE(used->predict_into(big_ref, out).has_value());
            {
                const cxxgpr::test_support::NoAllocationScope no_alloc;
                const auto res = used->predict_into(small_ref, out);
                const std::size_t allocations = no_alloc.delta_allocations();
                ASSERT_TRUE(res.has_value());
                EXPECT_EQ(allocations, 0U);
            }
            auto shrunk_expected = never_big->predict(small_ref);
            ASSERT_TRUE(shrunk_expected.has_value());
            EXPECT_EQ(out.mean, shrunk_expected->mean);
            EXPECT_EQ(out.variance, shrunk_expected->variance);
            {
                const cxxgpr::test_support::NoAllocationScope no_alloc;
                const auto res = used->predict_into(big_ref, out);
                const std::size_t allocations = no_alloc.delta_allocations();
                ASSERT_TRUE(res.has_value());
                EXPECT_EQ(allocations, 0U);
            }
        }
    }

    /// @brief Verifies a LoadedGpr accepts a smaller query batch after a larger one.
    /// @details Covers all four precision policies with a stored factor, plus the rebuild-on-load path,
    /// for predict, predict_covariance, sample, and allocation-free predict_into.
    TEST(PersistModelsTest, LoadedQueryBatchSizeMayShrinkAfterLargerBatch)
    {
        // Spec: REQ-07, REQ-10, REQ-11, REQ-12, DES-05, DES-16, DES-17, DES-18, FMT-05
        {
            SCOPED_TRACE("DoublePrecision, stored factor");
            verify_loaded_query_batch_may_shrink<cxxgpr::DoublePrecision>("shrink_f64", true);
        }
        {
            SCOPED_TRACE("DoublePrecision, factor rebuilt on load");
            verify_loaded_query_batch_may_shrink<cxxgpr::DoublePrecision>("shrink_f64_refactor", false);
        }
        {
            SCOPED_TRACE("SinglePrecision");
            verify_loaded_query_batch_may_shrink<cxxgpr::SinglePrecision>("shrink_f32", true);
        }
        {
            SCOPED_TRACE("MixedPrecision<PromoteStorage>");
            verify_loaded_query_batch_may_shrink<cxxgpr::MixedPrecision<cxxgpr::PromoteStorage>>(
                "shrink_mixed_promote", true
            );
        }
        {
            SCOPED_TRACE("MixedPrecision<ReevaluateKernel>");
            verify_loaded_query_batch_may_shrink<cxxgpr::MixedPrecision<cxxgpr::ReevaluateKernel>>(
                "shrink_mixed_reevaluate", true
            );
        }
    }

    namespace
    {

        /// @brief Extracts the string value of `save_id` from serialized config.json text.
        /// @param[in] json_text Serialized config.json contents.
        /// @return Extracted 32-character value if found, or empty string otherwise.
        [[nodiscard]] std::string extract_json_save_id(const std::string& json_text)
        {
            const std::string needle = "\"save_id\": \"";
            const std::size_t pos = json_text.find(needle);
            if (pos == std::string::npos)
            {
                return {};
            }
            const std::size_t start = pos + needle.size();
            const std::size_t end = json_text.find('"', start);
            if (end == std::string::npos)
            {
                return {};
            }
            return json_text.substr(start, end - start);
        }

        /// @brief Replaces or removes the `"save_id": "..."` line in pretty-printed config.json text.
        /// @param[in] json_text Serialized config.json contents.
        /// @param[in] replacement Replacement JSON value token (e.g. `123` or `"bad"`), or empty to erase the key.
        /// @return Modified config.json text.
        [[nodiscard]] std::string mutate_config_save_id(const std::string& json_text, const std::string& replacement)
        {
            const std::size_t key_pos = json_text.find("\"save_id\":");
            if (key_pos == std::string::npos)
            {
                return json_text;
            }
            const std::size_t line_start = json_text.rfind('\n', key_pos);
            const std::size_t line_end = json_text.find('\n', key_pos);
            if (replacement.empty())
            {
                const bool has_trailing_comma =
                    (line_end != std::string::npos && key_pos < line_end && json_text[line_end - 1U] == ',');
                std::string out = json_text;
                if (line_end != std::string::npos)
                {
                    out.erase(line_start, line_end - line_start);
                }
                if (!has_trailing_comma && line_start != std::string::npos && line_start > 0U
                    && out[line_start - 1U] == ',')
                {
                    out.erase(line_start - 1U, 1U);
                }
                return out;
            }
            const std::size_t val_start = json_text.find(':', key_pos) + 2U;
            const std::size_t comma_or_nl = json_text.find_first_of(",\r\n", val_start);
            std::string out = json_text;
            out.replace(val_start, comma_or_nl - val_start, replacement);
            return out;
        }

        /// @brief Reads and mutates the raw JSON header string of a model.safetensors file, preserving 8-byte padding.
        /// @param[in] path Filesystem path to model.safetensors.
        /// @param[in] mutate Callback transforming the unpadded safetensors header JSON string.
        void rewrite_safetensors_header_str(
            const std::filesystem::path& path, const std::function<std::string(std::string)>& mutate
        )
        {
            const std::string raw = read_file_bytes(path);
            ASSERT_GE(raw.size(), 8U);
            std::uint64_t hlen = 0U;
            std::memcpy(&hlen, raw.data(), sizeof(hlen));
            ASSERT_LE(8U + hlen, raw.size());
            std::string hstr = mutate(raw.substr(8U, static_cast<std::size_t>(hlen)));
            while (!hstr.empty() && hstr.back() == ' ')
            {
                hstr.pop_back();
            }
            const std::size_t padded = (hstr.size() + 7U) & ~std::size_t{7U};
            hstr.append(padded - hstr.size(), ' ');
            const auto new_hlen = static_cast<std::uint64_t>(hstr.size());
            std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
            ASSERT_TRUE(ofs.good());
            ofs.write(reinterpret_cast<const char*>(&new_hlen), sizeof(new_hlen));
            ofs.write(hstr.data(), static_cast<std::streamsize>(hstr.size()));
            ofs.write(raw.data() + 8U + hlen, static_cast<std::streamsize>(raw.size() - (8U + hlen)));
            ofs.flush();
            ASSERT_TRUE(ofs.good());
        }

        /// @brief Reads the raw JSON header string from a model.safetensors file.
        /// @param[in] path Filesystem path to model.safetensors.
        /// @return Raw header JSON text.
        [[nodiscard]] std::string read_safetensors_header_str(const std::filesystem::path& path)
        {
            const std::string raw = read_file_bytes(path);
            std::uint64_t hlen = 0U;
            std::memcpy(&hlen, raw.data(), sizeof(hlen));
            return raw.substr(8U, static_cast<std::size_t>(hlen));
        }

        /// @brief Returns true if s is exactly 32 lowercase hexadecimal digits.
        /// @param[in] s Candidate identifier string.
        /// @return True when s has length 32 and all characters match `[0-9a-f]`.
        [[nodiscard]] bool is_hex32_lower(const std::string& s) noexcept
        {
            return s.size() == 32U && std::ranges::all_of(s, [](unsigned char c) {
                return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
            });
        }

    }  // namespace

    /// @brief Verifies FMT-09 save_id determinism, format, and agreement between config.json and model.safetensors.
    TEST(PersistModelsTest, SaveAndLoadEmitDeterministicMatchingSaveId)
    {
        // Spec: FMT-02, FMT-03, FMT-04, FMT-09
        const auto fit_a1 = factor_rbf_model(1.0, {0.0, 0.5, 1.0, 1.5}, {1.0, 1.2, 0.9, 0.5});
        const auto fit_a2 = factor_rbf_model(1.0, {0.0, 0.5, 1.0, 1.5}, {1.0, 1.2, 0.9, 0.5});
        const auto fit_b = factor_rbf_model(0.6, {0.0, 0.5, 1.0, 1.5}, {1.0, 1.2, 0.9, 0.5});
        ASSERT_TRUE(fit_a1.has_value());
        ASSERT_TRUE(fit_a2.has_value());
        ASSERT_TRUE(fit_b.has_value());

        const TempDirGuard dir_a1("save_id_det_a1");
        const TempDirGuard dir_a2("save_id_det_a2");
        const TempDirGuard dir_b("save_id_det_b");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit_a1, dir_a1.path).has_value());
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit_a2, dir_a2.path).has_value());
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit_b, dir_b.path).has_value());

        const std::string cfg_a1_text = read_file_bytes(dir_a1.path / "config.json");
        EXPECT_EQ(cfg_a1_text, read_file_bytes(dir_a2.path / "config.json"));
        EXPECT_EQ(
            read_file_bytes(dir_a1.path / "model.safetensors"), read_file_bytes(dir_a2.path / "model.safetensors")
        );

        const std::string id_a = extract_json_save_id(cfg_a1_text);
        EXPECT_TRUE(is_hex32_lower(id_a));

        const std::string hdr_a = read_safetensors_header_str(dir_a1.path / "model.safetensors");
        EXPECT_NE(hdr_a.find("\"__metadata__\":{\"cxxgpr.save_id\":\"" + id_a + "\"}"), std::string::npos);

        const std::string id_b = extract_json_save_id(read_file_bytes(dir_b.path / "config.json"));
        EXPECT_TRUE(is_hex32_lower(id_b));
        EXPECT_NE(id_b, id_a);

        const auto loaded = cxxgpr::persist::LoadedGpr::load(dir_a1.path);
        ASSERT_TRUE(loaded.has_value());
    }

    /// @brief Negative controls for FMT-09 pairing identifier mismatch, one-sided presence, legacy load, and syntax.
    TEST(PersistModelsTest, RejectsMismatchedOneSidedLegacyAndMalformedSaveIds)
    {
        // Spec: FMT-02, FMT-03, FMT-04, FMT-08, FMT-09
        const auto fit_old = factor_rbf_model(1.0, {0.0, 0.5, 1.0, 1.5}, {1.0, 1.2, 0.9, 0.5});
        const auto fit_new = factor_rbf_model(0.4, {0.1, 0.7, 1.1, 1.9}, {-0.3, 0.8, 0.2, 1.4});
        const auto fit_other_n = factor_rbf_model(0.4, {0.1, 0.7, 1.1}, {-0.3, 0.8, 0.2});
        ASSERT_TRUE(fit_old.has_value());
        ASSERT_TRUE(fit_new.has_value());
        ASSERT_TRUE(fit_other_n.has_value());

        const TempDirGuard dir_old("save_id_old");
        const TempDirGuard dir_new("save_id_new");
        const TempDirGuard dir_other("save_id_other_n");
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit_old, dir_old.path).has_value());
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit_new, dir_new.path).has_value());
        ASSERT_TRUE(cxxgpr::persist::save_with_factor(*fit_other_n, dir_other.path).has_value());

        const std::string cfg_old = read_file_bytes(dir_old.path / "config.json");
        const std::string id_old = extract_json_save_id(cfg_old);
        ASSERT_TRUE(is_hex32_lower(id_old));

        // Mismatched pair (same shapes/dtypes, e.g., interrupted save or failed 2nd rename) -> PersistFailed(Tensor).
        {
            const TempDirGuard mix("save_id_mix_same");
            std::filesystem::copy_file(dir_old.path / "config.json", mix.path / "config.json");
            std::filesystem::copy_file(dir_new.path / "model.safetensors", mix.path / "model.safetensors");
            const auto res = cxxgpr::persist::LoadedGpr::load(mix.path);
            ASSERT_FALSE(res.has_value());
            EXPECT_EQ(res.error().code(), cxxgpr::GprErrorCode::PersistFailed);
            EXPECT_EQ(res.error().persist_error_kind(), cxxgpr::PersistErrorKind::Tensor);
        }

        // Pairing check runs before tensor shape check even when shapes also differ.
        {
            const TempDirGuard mix("save_id_mix_diff_n");
            std::filesystem::copy_file(dir_old.path / "config.json", mix.path / "config.json");
            std::filesystem::copy_file(dir_other.path / "model.safetensors", mix.path / "model.safetensors");
            const auto res = cxxgpr::persist::LoadedGpr::load(mix.path);
            ASSERT_FALSE(res.has_value());
            EXPECT_EQ(res.error().code(), cxxgpr::GprErrorCode::PersistFailed);
            EXPECT_EQ(res.error().persist_error_kind(), cxxgpr::PersistErrorKind::Tensor);
            EXPECT_NE(res.error().message().find("different saves"), std::string_view::npos);
        }

        auto strip_metadata = [](std::string hdr)
        {
            const std::size_t pos = hdr.find("\"__metadata__\":{");
            if (pos != std::string::npos)
            {
                const std::size_t end = hdr.find("},", pos);
                if (end != std::string::npos)
                {
                    hdr.erase(pos, (end + 2U) - pos);
                }
            }
            return hdr;
        };

        // Legacy pair (neither carries an identifier) loads cleanly.
        {
            const TempDirGuard leg("save_id_legacy");
            std::filesystem::copy_file(dir_old.path / "config.json", leg.path / "config.json");
            std::filesystem::copy_file(dir_old.path / "model.safetensors", leg.path / "model.safetensors");
            {
                std::ofstream ofs(leg.path / "config.json", std::ios::binary | std::ios::trunc);
                ofs << mutate_config_save_id(cfg_old, "");
            }
            rewrite_safetensors_header_str(leg.path / "model.safetensors", strip_metadata);
            EXPECT_TRUE(cxxgpr::persist::LoadedGpr::load(leg.path).has_value());
        }

        // One-sided pairs (only config.json or only model.safetensors carries an identifier) are rejected.
        {
            const TempDirGuard only_cfg("save_id_only_cfg");
            std::filesystem::copy_file(dir_old.path / "config.json", only_cfg.path / "config.json");
            std::filesystem::copy_file(dir_old.path / "model.safetensors", only_cfg.path / "model.safetensors");
            rewrite_safetensors_header_str(only_cfg.path / "model.safetensors", strip_metadata);
            const auto res = cxxgpr::persist::LoadedGpr::load(only_cfg.path);
            ASSERT_FALSE(res.has_value());
            EXPECT_EQ(res.error().code(), cxxgpr::GprErrorCode::PersistFailed);
            EXPECT_EQ(res.error().persist_error_kind(), cxxgpr::PersistErrorKind::Tensor);
        }
        {
            const TempDirGuard only_st("save_id_only_st");
            std::filesystem::copy_file(dir_old.path / "config.json", only_st.path / "config.json");
            std::filesystem::copy_file(dir_old.path / "model.safetensors", only_st.path / "model.safetensors");
            {
                std::ofstream ofs(only_st.path / "config.json", std::ios::binary | std::ios::trunc);
                ofs << mutate_config_save_id(cfg_old, "");
            }
            const auto res = cxxgpr::persist::LoadedGpr::load(only_st.path);
            ASSERT_FALSE(res.has_value());
            EXPECT_EQ(res.error().code(), cxxgpr::GprErrorCode::PersistFailed);
            EXPECT_EQ(res.error().persist_error_kind(), cxxgpr::PersistErrorKind::Tensor);
        }

        // Malformed config.json save_id -> PersistFailed(Config).
        for (const std::string bad_val : {
                 std::string("123"),
                 std::string("\"0123456789abcdef0123456789abcde\""),
                 std::string("\"0123456789ABCDEF0123456789ABCDEF\""),
                 std::string("\"0123456789abcdef0123456789abcdeg\""),
             })
        {
            const TempDirGuard bad_cfg("save_id_bad_cfg");
            std::filesystem::copy_file(dir_old.path / "config.json", bad_cfg.path / "config.json");
            std::filesystem::copy_file(dir_old.path / "model.safetensors", bad_cfg.path / "model.safetensors");
            {
                std::ofstream ofs(bad_cfg.path / "config.json", std::ios::binary | std::ios::trunc);
                ofs << mutate_config_save_id(cfg_old, bad_val);
            }
            const auto res = cxxgpr::persist::LoadedGpr::load(bad_cfg.path);
            ASSERT_FALSE(res.has_value());
            EXPECT_EQ(res.error().code(), cxxgpr::GprErrorCode::PersistFailed);
            EXPECT_EQ(res.error().persist_error_kind(), cxxgpr::PersistErrorKind::Config);
        }

        // Malformed __metadata__ / cxxgpr.save_id -> PersistFailed(Tensor), while extra metadata keys are ignored.
        {
            const TempDirGuard extra_meta("save_id_extra_meta");
            std::filesystem::copy_file(dir_old.path / "config.json", extra_meta.path / "config.json");
            std::filesystem::copy_file(dir_old.path / "model.safetensors", extra_meta.path / "model.safetensors");
            const std::string meta_orig = "\"__metadata__\":{\"cxxgpr.save_id\":\"" + id_old + "\"}";
            rewrite_safetensors_header_str(extra_meta.path / "model.safetensors", [&](std::string hdr) {
                const std::size_t pos = hdr.find(meta_orig);
                hdr.replace(
                    pos, meta_orig.size(),
                    "\"__metadata__\":{\"cxxgpr.save_id\":\"" + id_old + "\",\"other_tool\":\"ignored\"}"
                );
                return hdr;
            });
            EXPECT_TRUE(cxxgpr::persist::LoadedGpr::load(extra_meta.path).has_value());

            rewrite_safetensors_header_str(extra_meta.path / "model.safetensors", [&](std::string hdr) {
                const std::string target = "\"other_tool\":\"ignored\"";
                hdr.replace(hdr.find(target), target.size(), "\"other_tool\":42");
                return hdr;
            });
            const auto non_str = cxxgpr::persist::LoadedGpr::load(extra_meta.path);
            ASSERT_FALSE(non_str.has_value());
            EXPECT_EQ(non_str.error().persist_error_kind(), cxxgpr::PersistErrorKind::Tensor);

            rewrite_safetensors_header_str(extra_meta.path / "model.safetensors", [&](std::string hdr) {
                hdr.replace(hdr.find(id_old), id_old.size(), "0123456789ABCDEF0123456789ABCDEF");
                return hdr;
            });
            const auto upper = cxxgpr::persist::LoadedGpr::load(extra_meta.path);
            ASSERT_FALSE(upper.has_value());
            EXPECT_EQ(upper.error().persist_error_kind(), cxxgpr::PersistErrorKind::Tensor);
        }
    }

}  // namespace
