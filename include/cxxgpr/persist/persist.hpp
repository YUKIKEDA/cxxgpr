// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/likelihood.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/gpr/gpr.hpp"
#include "cxxgpr/kernel/custom.hpp"
#include "cxxgpr/kernel/distance_cache.hpp"
#include "cxxgpr/kernel/exp.hpp"
#include "cxxgpr/kernel/spec.hpp"
#include "cxxgpr/linalg/jitter.hpp"
#include "cxxgpr/precision/policy.hpp"
#include "cxxgpr/transform/input.hpp"
#include "cxxgpr/transform/target.hpp"
#include "cxxgpr/workspace/buffers.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cxxgpr::persist {

/// Persistence directory format version implemented by this release.
inline constexpr std::uint32_t FORMAT_VERSION = 1U;

/// Reserved persist_id prefix prohibited for user-defined kernels and transforms.
inline constexpr std::string_view RESERVED_PREFIX = "cxxgpr.";

/// Factory callback reconstructing a CustomKernel from its serialized JSON state string.
using CustomKernelFactory =
    std::function<std::expected<CustomKernel, GprError>(std::string_view state_json)>;

/// Factory callback reconstructing an UnfittedInputTransform from its serialized JSON state.
using UnfittedInputFactory =
    std::function<std::expected<std::unique_ptr<transform::UnfittedInputTransform>, GprError>(
        std::string_view state_json)>;

/// Factory callback reconstructing a FittedInputTransform from its serialized JSON state.
using FittedInputFactory =
    std::function<std::expected<std::unique_ptr<transform::FittedInputTransform>, GprError>(
        std::string_view state_json)>;

/// Factory callback reconstructing an UnfittedTargetTransform from its serialized JSON state.
using UnfittedTargetFactory =
    std::function<std::expected<std::unique_ptr<transform::UnfittedTargetTransform>, GprError>(
        std::string_view state_json)>;

/// Factory callback reconstructing a FittedTargetTransform from its serialized JSON state.
using FittedTargetFactory =
    std::function<std::expected<std::unique_ptr<transform::FittedTargetTransform>, GprError>(
        std::string_view state_json)>;

/// Validates a user-supplied persist_id against empty and reserved prefix rules.
[[nodiscard]] std::expected<void, GprError> validate_persist_id(std::string_view persist_id);

/// Registry mapping user-defined persist_id strings to deserialization factories across 5
/// extension categories.
class PersistRegistry {
public:
    /// Registers a custom kernel leaf factory under persist_id.
    [[nodiscard]] std::expected<void, GprError> register_kernel(std::string persist_id,
                                                                CustomKernelFactory factory);

    /// Registers a custom unfitted input transform factory under persist_id.
    [[nodiscard]] std::expected<void, GprError>
    register_unfitted_input(std::string persist_id, UnfittedInputFactory factory);

    /// Registers a custom fitted input transform factory under persist_id.
    [[nodiscard]] std::expected<void, GprError> register_fitted_input(std::string persist_id,
                                                                      FittedInputFactory factory);

    /// Registers a custom unfitted target transform factory under persist_id.
    [[nodiscard]] std::expected<void, GprError>
    register_unfitted_target(std::string persist_id, UnfittedTargetFactory factory);

    /// Registers a custom fitted target transform factory under persist_id.
    [[nodiscard]] std::expected<void, GprError> register_fitted_target(std::string persist_id,
                                                                       FittedTargetFactory factory);

    /// Looks up a registered custom kernel factory by persist_id.
    [[nodiscard]] std::expected<CustomKernel, GprError>
    create_kernel(std::string_view persist_id, std::string_view state_json) const;

    /// Looks up a registered custom unfitted input transform factory by persist_id.
    [[nodiscard]] std::expected<std::unique_ptr<transform::UnfittedInputTransform>, GprError>
    create_unfitted_input(std::string_view persist_id, std::string_view state_json) const;

    /// Looks up a registered custom fitted input transform factory by persist_id.
    [[nodiscard]] std::expected<std::unique_ptr<transform::FittedInputTransform>, GprError>
    create_fitted_input(std::string_view persist_id, std::string_view state_json) const;

    /// Looks up a registered custom unfitted target transform factory by persist_id.
    [[nodiscard]] std::expected<std::unique_ptr<transform::UnfittedTargetTransform>, GprError>
    create_unfitted_target(std::string_view persist_id, std::string_view state_json) const;

    /// Looks up a registered custom fitted target transform factory by persist_id.
    [[nodiscard]] std::expected<std::unique_ptr<transform::FittedTargetTransform>, GprError>
    create_fitted_target(std::string_view persist_id, std::string_view state_json) const;

private:
    std::unordered_map<std::string, CustomKernelFactory> kernels_;
    std::unordered_map<std::string, UnfittedInputFactory> unfitted_inputs_;
    std::unordered_map<std::string, FittedInputFactory> fitted_inputs_;
    std::unordered_map<std::string, UnfittedTargetFactory> unfitted_targets_;
    std::unordered_map<std::string, FittedTargetFactory> fitted_targets_;
};

/// Type-erased Exact GPR model loaded from disk (Double/Single/MixedPromote/MixedReevaluate),
/// supporting zero-copy read-only mmap for F64 Cholesky factors and precision-agnostic prediction.
class LoadedGpr {
public:
    /// Loads and validates a persisted Exact GPR directory (config.json + model.safetensors).
    [[nodiscard]] static std::expected<LoadedGpr, GprError>
    load(const std::filesystem::path& dir, const PersistRegistry& registry = {});

    LoadedGpr(const LoadedGpr&) = default;
    LoadedGpr& operator=(const LoadedGpr&) = default;
    LoadedGpr(LoadedGpr&&) noexcept = default;
    LoadedGpr& operator=(LoadedGpr&&) noexcept = default;
    ~LoadedGpr();

    /// Returns the number of training samples n.
    [[nodiscard]] std::size_t n() const noexcept;

    /// Returns the number of training features d.
    [[nodiscard]] std::size_t d() const noexcept;

    /// Returns the precision policy tag stored in config.json.
    [[nodiscard]] PersistPrecisionKind precision_kind() const noexcept;

    /// Returns true if the model was loaded from a directory with has_factor == true.
    [[nodiscard]] bool has_factor() const noexcept;

    /// Returns true if the F64 Cholesky factor l is backed directly by a read-only mmap view.
    [[nodiscard]] bool has_mmap_factor() const noexcept;

    /// Returns the accepted Cholesky jitter offset.
    [[nodiscard]] double factor_jitter() const noexcept;

    /// Returns the restored kernel specification.
    [[nodiscard]] const KernelSpec& kernel_spec() const noexcept;

    /// Returns the restored Gaussian observation noise likelihood.
    [[nodiscard]] const GaussianLikelihood& likelihood() const noexcept;

    /// Returns the restored jitter policy.
    [[nodiscard]] const JitterPolicy& jitter() const noexcept;

    /// Returns the restored exponential evaluation backend.
    [[nodiscard]] KernelExp kernel_exp() const noexcept;

    /// Returns the restored distance cache policy.
    [[nodiscard]] DistanceCachePolicy distance_cache_policy() const noexcept;

    /// Computes posterior mean and diagonal variance promoted to double precision.
    [[nodiscard]] std::expected<Prediction<double>, GprError>
    predict(MatrixRef<double> xs, const PredictOptions& options = {}) const;

    /// Computes posterior mean and diagonal variance from a raw column-major span.
    [[nodiscard]] std::expected<Prediction<double>, GprError>
    predict(std::span<const double> xs, std::size_t n_rows, std::size_t n_cols,
            const PredictOptions& options = {}) const {
        return predict(MatrixRef<double>(xs.data(), n_rows, n_cols, n_rows), options);
    }

    /// Computes posterior mean and diagonal variance into out (double-promoted).
    [[nodiscard]] std::expected<void, GprError>
    predict_into(MatrixRef<double> xs, Prediction<double>& out,
                 const PredictOptions& options = {}) const;

    /// Computes full m x m posterior covariance matrix (double-promoted).
    [[nodiscard]] std::expected<PredictiveCovariance<double>, GprError>
    predict_covariance(MatrixRef<double> xs, const PredictOptions& options = {}) const;

    /// Draws n_draws deterministic posterior samples (double-promoted).
    [[nodiscard]] std::expected<std::vector<double>, GprError>
    sample(MatrixRef<double> xs, std::size_t n_draws, std::uint64_t seed,
           const PredictOptions& options = {},
           std::optional<JitterPolicy> jitter = std::nullopt) const;

    /// Computes closed-form Leave-One-Out prediction on the original target scale
    /// (double-promoted).
    [[nodiscard]] std::expected<Prediction<double>, GprError>
    loo_predict(const PredictOptions& options = {}) const;

    /// Extracts a typed FittedGpr<O, P, KernelSpec> using Retain policy when P matches
    /// precision_kind(); returns PersistFailed(WrongModel) on precision mismatch.
    template <typename O = optimizer::Lbfgs, PrecisionPolicyConcept P = DoublePrecision>
    [[nodiscard]] std::expected<FittedGpr<O, P, KernelSpec>, GprError>
    into_fitted(O optimizer = O{}) const;

private:
    struct Impl;
    explicit LoadedGpr(std::shared_ptr<const Impl> impl) noexcept : impl_(std::move(impl)) {}

    template <PrecisionPolicyConcept P>
    [[nodiscard]] std::expected<detail::FittedPayload<P>, GprError>
    extract_payload(std::unique_ptr<transform::UnfittedInputTransform>& out_unfitted_in,
                    std::unique_ptr<transform::UnfittedTargetTransform>& out_unfitted_tgt) const;

    std::shared_ptr<const Impl> impl_;
};

template <typename O, PrecisionPolicyConcept P>
std::expected<FittedGpr<O, P, KernelSpec>, GprError> LoadedGpr::into_fitted(O optimizer) const {
    if (P::persist_kind() != precision_kind()) {
        return std::unexpected(GprError::persist_failed(
            PersistErrorKind::WrongModel,
            "requested precision policy does not match persisted model precision"));
    }
    std::unique_ptr<transform::UnfittedInputTransform> unfitted_in;
    std::unique_ptr<transform::UnfittedTargetTransform> unfitted_tgt;
    auto payload = extract_payload<P>(unfitted_in, unfitted_tgt);
    if (!payload.has_value()) {
        return std::unexpected(std::move(payload).error());
    }
    return FittedGpr<O, P, KernelSpec>(std::move(*payload), std::move(optimizer), jitter(),
                                       std::move(unfitted_in), std::move(unfitted_tgt),
                                       kernel_exp(),
                                       FitPolicy{distance_cache_policy(), CholeskyBuffer::Retain});
}

namespace detail {

struct SaveInput {
    std::size_t n_train{0};
    std::size_t n_features{0};
    bool has_factor{false};
    PersistPrecisionKind precision_kind{PersistPrecisionKind::Double};
    KernelExp kernel_exp{KernelExp::Accurate};
    DistanceCachePolicy distance_cache{DistanceCachePolicy::Cached};
    double factor_jitter{0.0};
    const KernelSpec& kernel_spec;
    const GaussianLikelihood& likelihood;
    const JitterPolicy& jitter;
    const transform::UnfittedInputTransform& unfitted_input;
    const transform::FittedInputTransform& fitted_input;
    const transform::UnfittedTargetTransform& unfitted_target;
    const transform::FittedTargetTransform& fitted_target;
    std::span<const double> x_raw;
    std::span<const double> y_raw;
    std::span<const double> l_f64;
    std::span<const float> l_f32;
    std::span<const double> alpha_f64;
    std::span<const float> alpha_f32;
};

[[nodiscard]] std::expected<void, GprError> save_model_impl(const SaveInput& input,
                                                            const std::filesystem::path& dir,
                                                            const PersistRegistry& registry);

} // namespace detail

/// Atomically persists a fitted model without factor tensors (has_factor = false).
template <typename O, PrecisionPolicyConcept P, typename K>
[[nodiscard]] std::expected<void, GprError> save(const FittedGpr<O, P, K>& model,
                                                 const std::filesystem::path& dir,
                                                 const PersistRegistry& registry = {}) {
    const detail::SaveInput input{
        .n_train = model.n_train(),
        .n_features = model.n_features(),
        .has_factor = false,
        .precision_kind = P::persist_kind(),
        .kernel_exp = model.kernel_exp(),
        .distance_cache = model.fit_policy().distance_cache,
        .factor_jitter = model.factor_jitter(),
        .kernel_spec = model.kernel_spec(),
        .likelihood = model.likelihood(),
        .jitter = model.jitter(),
        .unfitted_input = model.unfitted_input_transform(),
        .fitted_input = model.input_transform(),
        .unfitted_target = model.unfitted_target_transform(),
        .fitted_target = model.target_transform(),
        .x_raw = model.x_raw(),
        .y_raw = model.y_raw(),
        .l_f64 = {},
        .l_f32 = {},
        .alpha_f64 = {},
        .alpha_f32 = {},
    };
    return detail::save_model_impl(input, dir, registry);
}

/// Atomically persists a fitted model including factor L and alpha tensors (has_factor = true).
template <typename O, PrecisionPolicyConcept P, typename K>
[[nodiscard]] std::expected<void, GprError> save_with_factor(const FittedGpr<O, P, K>& model,
                                                             const std::filesystem::path& dir,
                                                             const PersistRegistry& registry = {}) {
    const auto l_ref = model.l_factor();
    const std::size_t n = model.n_train();
    std::span<const double> l_f64;
    std::span<const float> l_f32;
    if constexpr (std::same_as<typename P::Storage, double>) {
        l_f64 = std::span<const double>(l_ref.data(), n * n);
    } else {
        l_f32 = std::span<const float>(l_ref.data(), n * n);
    }
    std::span<const double> alpha_f64;
    std::span<const float> alpha_f32;
    if constexpr (std::same_as<typename P::Refine, double>) {
        alpha_f64 = model.alpha();
    } else {
        alpha_f32 = model.alpha();
    }
    const detail::SaveInput input{
        .n_train = n,
        .n_features = model.n_features(),
        .has_factor = true,
        .precision_kind = P::persist_kind(),
        .kernel_exp = model.kernel_exp(),
        .distance_cache = model.fit_policy().distance_cache,
        .factor_jitter = model.factor_jitter(),
        .kernel_spec = model.kernel_spec(),
        .likelihood = model.likelihood(),
        .jitter = model.jitter(),
        .unfitted_input = model.unfitted_input_transform(),
        .fitted_input = model.input_transform(),
        .unfitted_target = model.unfitted_target_transform(),
        .fitted_target = model.target_transform(),
        .x_raw = model.x_raw(),
        .y_raw = model.y_raw(),
        .l_f64 = l_f64,
        .l_f32 = l_f32,
        .alpha_f64 = alpha_f64,
        .alpha_f32 = alpha_f32,
    };
    return detail::save_model_impl(input, dir, registry);
}

} // namespace cxxgpr::persist
