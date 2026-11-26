// SPDX-License-Identifier: Apache-2.0
/// @file persist.hpp
/// @brief Model persistence API for saving and loading Exact GPR checkpoints.
/// @details Provides directory-based checkpoint serialization (config.json and model.safetensors),
/// custom extension factory registration via PersistRegistry, zero-copy mmap loading of F64 Cholesky
/// factors via LoadedGpr, and precision-checked extraction into typed FittedGpr models.

#pragma once

#include "cxxgpr/core/checked.hpp"
#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/likelihood.hpp"
#include "cxxgpr/core/views.hpp"
#include "cxxgpr/gpr/distance_gpr.hpp"
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
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

/// @defgroup cxxgpr_persist Model Persistence and Extension Registry
/// @brief Directory-based serialization (`config.json` + `model.safetensors`) and mmap loading.
/// @{

/// @brief Model serialization, deserialization, and custom extension registry APIs.
namespace cxxgpr::persist
{

    /// @brief Persistence directory format version implemented by this release.
    inline constexpr std::uint32_t FORMAT_VERSION = 1U;

    /// @brief Reserved persist_id prefix prohibited for user-defined kernels and transforms.
    inline constexpr std::string_view RESERVED_PREFIX = "cxxgpr.";

    /// @brief Factory callback reconstructing a CustomKernel from its serialized JSON state string.
    using CustomKernelFactory = std::function<std::expected<CustomKernel, GprError>(std::string_view state_json)>;

    /// @brief Factory callback reconstructing an UnfittedInputTransform from its serialized JSON state.
    using UnfittedInputFactory = std::function<
        std::expected<std::unique_ptr<transform::UnfittedInputTransform>, GprError>(std::string_view state_json)>;

    /// @brief Factory callback reconstructing a FittedInputTransform from its serialized JSON state.
    using FittedInputFactory = std::function<
        std::expected<std::unique_ptr<transform::FittedInputTransform>, GprError>(std::string_view state_json)>;

    /// @brief Factory callback reconstructing an UnfittedTargetTransform from its serialized JSON state.
    using UnfittedTargetFactory = std::function<
        std::expected<std::unique_ptr<transform::UnfittedTargetTransform>, GprError>(std::string_view state_json)>;

    /// @brief Factory callback reconstructing a FittedTargetTransform from its serialized JSON state.
    using FittedTargetFactory = std::function<
        std::expected<std::unique_ptr<transform::FittedTargetTransform>, GprError>(std::string_view state_json)>;

    /// @brief Validates a user-supplied persist_id against the empty, reserved prefix, and encoding rules.
    /// @details Rejects empty strings, identifiers beginning with RESERVED_PREFIX ("cxxgpr."), and identifiers
    /// that are not valid UTF-8 (config.json is UTF-8 JSON, so such an identifier could not be written into it).
    /// @param[in] persist_id Candidate extension identifier string.
    /// @return Empty expected on success, or PersistFailed(InvalidPersistId) when invalid.
    /// @retval GprErrorCode::PersistFailed When persist_id is empty, uses the reserved prefix, or is not valid
    /// UTF-8.
    [[nodiscard]] std::expected<void, GprError> validate_persist_id(std::string_view persist_id);

    /// @brief Registry mapping user-defined persist_id strings to deserialization factories across 5
    /// extension categories.
    /// @details Enables saving and loading custom kernels as well as unfitted and fitted input/target
    /// transforms without coupling persistence internals to user types.
    class PersistRegistry
    {
    public:
        /// @brief Registers a custom kernel leaf factory under persist_id.
        /// @details Rejects invalid or already registered identifiers before inserting into kernels_.
        /// @param[in] persist_id Unique user extension identifier outside the reserved prefix.
        /// @param[in] factory Deserialization callback accepting serialized state JSON.
        /// @return Empty expected on success, or PersistFailed(InvalidPersistId) on validation/duplicate failure.
        /// @retval GprErrorCode::PersistFailed When persist_id is invalid or already registered.
        [[nodiscard]] std::expected<void, GprError> register_kernel(
            std::string persist_id, CustomKernelFactory factory
        );

        /// @brief Registers a custom unfitted input transform factory under persist_id.
        /// @details Rejects invalid or already registered identifiers before inserting into unfitted_inputs_.
        /// @param[in] persist_id Unique user extension identifier outside the reserved prefix.
        /// @param[in] factory Deserialization callback accepting serialized state JSON.
        /// @return Empty expected on success, or PersistFailed(InvalidPersistId) on validation/duplicate failure.
        /// @retval GprErrorCode::PersistFailed When persist_id is invalid or already registered.
        [[nodiscard]] std::expected<void, GprError> register_unfitted_input(
            std::string persist_id, UnfittedInputFactory factory
        );

        /// @brief Registers a custom fitted input transform factory under persist_id.
        /// @details Rejects invalid or already registered identifiers before inserting into fitted_inputs_.
        /// @param[in] persist_id Unique user extension identifier outside the reserved prefix.
        /// @param[in] factory Deserialization callback accepting serialized state JSON.
        /// @return Empty expected on success, or PersistFailed(InvalidPersistId) on validation/duplicate failure.
        /// @retval GprErrorCode::PersistFailed When persist_id is invalid or already registered.
        [[nodiscard]] std::expected<void, GprError> register_fitted_input(
            std::string persist_id, FittedInputFactory factory
        );

        /// @brief Registers a custom unfitted target transform factory under persist_id.
        /// @details Rejects invalid or already registered identifiers before inserting into unfitted_targets_.
        /// @param[in] persist_id Unique user extension identifier outside the reserved prefix.
        /// @param[in] factory Deserialization callback accepting serialized state JSON.
        /// @return Empty expected on success, or PersistFailed(InvalidPersistId) on validation/duplicate failure.
        /// @retval GprErrorCode::PersistFailed When persist_id is invalid or already registered.
        [[nodiscard]] std::expected<void, GprError> register_unfitted_target(
            std::string persist_id, UnfittedTargetFactory factory
        );

        /// @brief Registers a custom fitted target transform factory under persist_id.
        /// @details Rejects invalid or already registered identifiers before inserting into fitted_targets_.
        /// @param[in] persist_id Unique user extension identifier outside the reserved prefix.
        /// @param[in] factory Deserialization callback accepting serialized state JSON.
        /// @return Empty expected on success, or PersistFailed(InvalidPersistId) on validation/duplicate failure.
        /// @retval GprErrorCode::PersistFailed When persist_id is invalid or already registered.
        [[nodiscard]] std::expected<void, GprError> register_fitted_target(
            std::string persist_id, FittedTargetFactory factory
        );

        /// @brief Looks up and invokes a registered custom kernel factory by persist_id.
        /// @details Returns PersistFailed(UnregisteredId) if no matching kernel factory is registered.
        /// @param[in] persist_id Identifier read from persisted config.json.
        /// @param[in] state_json Serialized JSON state string passed to the factory.
        /// @return Reconstructed CustomKernel or GprError on missing registration or factory failure.
        /// @retval GprErrorCode::PersistFailed When persist_id is unregistered or factory validation fails.
        [[nodiscard]] std::expected<CustomKernel, GprError> create_kernel(
            std::string_view persist_id, std::string_view state_json
        ) const;

        /// @brief Looks up and invokes a registered custom unfitted input transform factory by persist_id.
        /// @details Returns PersistFailed(UnregisteredId) if no matching factory is registered.
        /// @param[in] persist_id Identifier read from persisted config.json.
        /// @param[in] state_json Serialized JSON state string passed to the factory.
        /// @return Reconstructed UnfittedInputTransform pointer or GprError on failure.
        /// @retval GprErrorCode::PersistFailed When persist_id is unregistered or factory validation fails.
        [[nodiscard]] std::expected<std::unique_ptr<transform::UnfittedInputTransform>, GprError> create_unfitted_input(
            std::string_view persist_id, std::string_view state_json
        ) const;

        /// @brief Looks up and invokes a registered custom fitted input transform factory by persist_id.
        /// @details Returns PersistFailed(UnregisteredId) if no matching factory is registered.
        /// @param[in] persist_id Identifier read from persisted config.json.
        /// @param[in] state_json Serialized JSON state string passed to the factory.
        /// @return Reconstructed FittedInputTransform pointer or GprError on failure.
        /// @retval GprErrorCode::PersistFailed When persist_id is unregistered or factory validation fails.
        [[nodiscard]] std::expected<std::unique_ptr<transform::FittedInputTransform>, GprError> create_fitted_input(
            std::string_view persist_id, std::string_view state_json
        ) const;

        /// @brief Looks up and invokes a registered custom unfitted target transform factory by persist_id.
        /// @details Returns PersistFailed(UnregisteredId) if no matching factory is registered.
        /// @param[in] persist_id Identifier read from persisted config.json.
        /// @param[in] state_json Serialized JSON state string passed to the factory.
        /// @return Reconstructed UnfittedTargetTransform pointer or GprError on failure.
        /// @retval GprErrorCode::PersistFailed When persist_id is unregistered or factory validation fails.
        [[nodiscard]] std::expected<std::unique_ptr<transform::UnfittedTargetTransform>, GprError>
            create_unfitted_target(std::string_view persist_id, std::string_view state_json) const;

        /// @brief Looks up and invokes a registered custom fitted target transform factory by persist_id.
        /// @details Returns PersistFailed(UnregisteredId) if no matching factory is registered.
        /// @param[in] persist_id Identifier read from persisted config.json.
        /// @param[in] state_json Serialized JSON state string passed to the factory.
        /// @return Reconstructed FittedTargetTransform pointer or GprError on failure.
        /// @retval GprErrorCode::PersistFailed When persist_id is unregistered or factory validation fails.
        [[nodiscard]] std::expected<std::unique_ptr<transform::FittedTargetTransform>, GprError> create_fitted_target(
            std::string_view persist_id, std::string_view state_json
        ) const;

    private:
        /// @brief Registered custom kernel leaf factories keyed by persist_id.
        std::unordered_map<std::string, CustomKernelFactory> kernels_;
        /// @brief Registered custom unfitted input transform factories keyed by persist_id.
        std::unordered_map<std::string, UnfittedInputFactory> unfitted_inputs_;
        /// @brief Registered custom fitted input transform factories keyed by persist_id.
        std::unordered_map<std::string, FittedInputFactory> fitted_inputs_;
        /// @brief Registered custom unfitted target transform factories keyed by persist_id.
        std::unordered_map<std::string, UnfittedTargetFactory> unfitted_targets_;
        /// @brief Registered custom fitted target transform factories keyed by persist_id.
        std::unordered_map<std::string, FittedTargetFactory> fitted_targets_;
    };

    /// @brief Type-erased Exact GPR model loaded from disk (Double/Single/MixedPromote/MixedReevaluate),
    /// supporting zero-copy read-only mmap for F64 Cholesky factors and precision-agnostic prediction.
    /// @details Holds immutable checkpoint state via std::shared_ptr<const Impl> so copies share mapped
    /// factor backing storage while remaining ready for double-promoted inference or typed extraction.
    class LoadedGpr
    {
    public:
        /// @brief Loads and validates a persisted Exact GPR directory (config.json + model.safetensors).
        /// @details Parses configuration, memory-maps model.safetensors, validates tensor shapes and
        /// finiteness, and either binds or reconstructs the Cholesky factor depending on has_factor.
        /// @param[in] dir Directory containing config.json and model.safetensors.
        /// @param[in] registry Optional custom extension registry for custom kernels and transforms.
        /// @return Validated LoadedGpr handle or GprError on I/O, schema, version, or factor failure.
        /// @retval GprErrorCode::PersistFailed When directory files, JSON, tensors, or registry lookup fail.
        /// @retval GprErrorCode::UnsupportedPersistVersion When artifact format_version is unsupported.
        /// @retval GprErrorCode::CholeskyFailed When factor reconstruction fails when has_factor is false.
        [[nodiscard]] static std::expected<LoadedGpr, GprError> load(
            const std::filesystem::path& dir, const PersistRegistry& registry = {}
        );

        /// @brief Copy-constructs a LoadedGpr sharing the immutable internal checkpoint implementation.
        /// @details Increments the shared ownership handle without copying mapped tensors.
        LoadedGpr(const LoadedGpr&) = default;

        /// @brief Copy-assigns a LoadedGpr sharing the immutable internal checkpoint implementation.
        /// @details Increments the shared ownership handle without copying mapped tensors.
        /// @return Reference to this LoadedGpr.
        LoadedGpr& operator=(const LoadedGpr&) = default;

        /// @brief Move-constructs a LoadedGpr transferring shared checkpoint ownership.
        /// @details Transfers the std::shared_ptr<const Impl> without releasing mapped buffers.
        LoadedGpr(LoadedGpr&&) noexcept = default;

        /// @brief Move-assigns a LoadedGpr transferring shared checkpoint ownership.
        /// @details Transfers the std::shared_ptr<const Impl> without releasing mapped buffers.
        /// @return Reference to this LoadedGpr.
        LoadedGpr& operator=(LoadedGpr&&) noexcept = default;

        /// @brief Destroys this LoadedGpr handle, releasing mapped resources when the last reference drops.
        ~LoadedGpr();

        /// @brief Returns the number of training samples n.
        /// @return Training sample count n.
        [[nodiscard]] std::size_t n() const noexcept;

        /// @brief Returns the number of training features d.
        /// @return Training feature dimension d.
        [[nodiscard]] std::size_t d() const noexcept;

        /// @brief Returns the precision policy tag stored in config.json.
        /// @return Persisted precision and residual evaluation mode.
        [[nodiscard]] PersistPrecisionKind precision_kind() const noexcept;

        /// @brief Returns true if the model was loaded from a directory with has_factor == true.
        /// @return True when precomputed L and alpha tensors were loaded from model.safetensors.
        [[nodiscard]] bool has_factor() const noexcept;

        /// @brief Returns true if the F64 Cholesky factor l is backed directly by a read-only mmap view.
        /// @return True when DoublePrecision loaded its Cholesky factor directly via zero-copy mmap.
        [[nodiscard]] bool has_mmap_factor() const noexcept;

        /// @brief Returns the accepted Cholesky jitter offset.
        /// @return Diagonal jitter scalar recorded during factorization.
        [[nodiscard]] double factor_jitter() const noexcept;

        /// @brief Returns the restored kernel specification.
        /// @return Constant reference to the deserialized KernelSpec.
        [[nodiscard]] const KernelSpec& kernel_spec() const noexcept;

        /// @brief Returns the restored Gaussian observation noise likelihood.
        /// @return Constant reference to the deserialized GaussianLikelihood.
        [[nodiscard]] const GaussianLikelihood& likelihood() const noexcept;

        /// @brief Returns the restored jitter policy.
        /// @return Constant reference to the deserialized JitterPolicy.
        [[nodiscard]] const JitterPolicy& jitter() const noexcept;

        /// @brief Returns the restored exponential evaluation backend.
        /// @return KernelExp mode configured in the checkpoint.
        [[nodiscard]] KernelExp kernel_exp() const noexcept;

        /// @brief Returns the restored distance cache policy.
        /// @return DistanceCachePolicy configured in the checkpoint.
        [[nodiscard]] DistanceCachePolicy distance_cache_policy() const noexcept;

        /// @brief Computes posterior mean and diagonal variance promoted to double precision.
        /// @details Validates xs, applies the restored input transform, evaluates posterior moments under
        /// the persisted precision policy, and applies the inverse target transform on the original scale.
        /// @param[in] xs Column-major query feature matrix (m x d).
        /// @param[in] options Optional prediction controls (e.g., noiseless vs noisy variance).
        /// @return Double-precision Prediction on the original target scale, or GprError on failure.
        /// @retval GprErrorCode::EmptyInput When xs is empty or null.
        /// @retval GprErrorCode::DimensionMismatch When xs.cols() != d().
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
        /// @retval GprErrorCode::NonFiniteKernelValue When a latent predictive variance is NaN or +/-Inf.
        [[nodiscard]] std::expected<Prediction<double>, GprError> predict(
            MatrixRef<double> xs, const PredictOptions& options = {}
        ) const;

        /// @brief Computes posterior mean and diagonal variance promoted to double precision on float queries.
        /// @param[in] xs Column-major single-precision query feature matrix (m x d).
        /// @param[in] options Optional prediction controls.
        /// @return Double-precision Prediction on the original target scale, or GprError on failure.
        [[nodiscard]] std::expected<Prediction<double>, GprError> predict(
            MatrixRef<float> xs, const PredictOptions& options = {}
        ) const;

        /// @brief Computes posterior mean and diagonal variance from a raw column-major span.
        /// @details Checks the length of xs against n_rows * n_cols, wraps it as a MatrixRef<double> with leading
        /// dimension n_rows, and delegates to predict().
        /// @param[in] xs Contiguous column-major query data of length n_rows * n_cols.
        /// @param[in] n_rows Number of query points m.
        /// @param[in] n_cols Number of features per query point (must match d()).
        /// @param[in] options Optional prediction controls.
        /// @return Double-precision Prediction on the original target scale, or GprError on failure.
        [[nodiscard]] std::expected<Prediction<double>, GprError> predict(
            std::span<const double> xs, std::size_t n_rows, std::size_t n_cols, const PredictOptions& options = {}
        ) const
        {
            auto view = cxxgpr::detail::matrix_view_from_span(xs, n_rows, n_cols);
            if (!view.has_value())
            {
                return std::unexpected(std::move(view).error());
            }
            return predict(*view, options);
        }

        /// @brief Computes posterior mean and diagonal variance from a raw single-precision span.
        /// @param[in] xs Contiguous column-major float query data of length n_rows * n_cols.
        /// @param[in] n_rows Number of query points m.
        /// @param[in] n_cols Number of features per query point (must match d()).
        /// @param[in] options Optional prediction controls.
        /// @return Double-precision Prediction on the original target scale, or GprError on failure.
        [[nodiscard]] std::expected<Prediction<double>, GprError> predict(
            std::span<const float> xs, std::size_t n_rows, std::size_t n_cols, const PredictOptions& options = {}
        ) const
        {
            auto view = cxxgpr::detail::matrix_view_from_span(xs, n_rows, n_cols);
            if (!view.has_value())
            {
                return std::unexpected(std::move(view).error());
            }
            return predict(*view, options);
        }

        /// @brief Computes posterior mean and diagonal variance into out (double-promoted).
        /// @details Reuses internal query scratch buffers and writes results into out on the original scale.
        /// @param[in] xs Column-major query feature matrix (m x d).
        /// @param[out] out Destination prediction struct populated on success.
        /// @param[in] options Optional prediction controls.
        /// @return Empty expected on success, or GprError on validation or numerical failure.
        [[nodiscard]] std::expected<void, GprError> predict_into(
            MatrixRef<double> xs, Prediction<double>& out, const PredictOptions& options = {}
        ) const;

        /// @brief Computes posterior mean and diagonal variance into out for single-precision query features.
        /// @param[in] xs Column-major float query feature matrix (m x d).
        /// @param[out] out Destination prediction struct populated on success.
        /// @param[in] options Optional prediction controls.
        /// @return Empty expected on success, or GprError on validation or numerical failure.
        [[nodiscard]] std::expected<void, GprError> predict_into(
            MatrixRef<float> xs, Prediction<double>& out, const PredictOptions& options = {}
        ) const;

        /// @brief Computes full m x m posterior covariance matrix (double-promoted).
        /// @details Transforms query coordinates, evaluates full posterior covariance under the persisted
        /// precision policy, and maps both mean and covariance back to the original target scale.
        /// @param[in] xs Column-major query feature matrix (m x d).
        /// @param[in] options Optional prediction controls.
        /// @return Double-precision PredictiveCovariance or GprError on validation or numerical failure.
        [[nodiscard]] std::expected<PredictiveCovariance<double>, GprError> predict_covariance(
            MatrixRef<double> xs, const PredictOptions& options = {}
        ) const;

        /// @brief Computes full m x m posterior covariance matrix for single-precision query features.
        /// @param[in] xs Column-major float query feature matrix (m x d).
        /// @param[in] options Optional prediction controls.
        /// @return Double-precision PredictiveCovariance or GprError on validation or numerical failure.
        [[nodiscard]] std::expected<PredictiveCovariance<double>, GprError> predict_covariance(
            MatrixRef<float> xs, const PredictOptions& options = {}
        ) const;

        /// @brief Computes full m x m posterior covariance from a raw float span.
        /// @param[in] xs Contiguous column-major float query data of length n_rows * n_cols.
        /// @param[in] n_rows Number of query points m.
        /// @param[in] n_cols Number of features per query point (must match d()).
        /// @param[in] options Optional prediction controls.
        /// @return Double-precision PredictiveCovariance or GprError on validation or numerical failure.
        [[nodiscard]] std::expected<PredictiveCovariance<double>, GprError> predict_covariance(
            std::span<const float> xs, std::size_t n_rows, std::size_t n_cols, const PredictOptions& options = {}
        ) const
        {
            auto view = cxxgpr::detail::matrix_view_from_span(xs, n_rows, n_cols);
            if (!view.has_value())
            {
                return std::unexpected(std::move(view).error());
            }
            return predict_covariance(*view, options);
        }

        /// @brief Draws n_draws deterministic posterior samples (double-promoted).
        /// @details Evaluates posterior draws using seed and maps each draw back to the original target scale.
        /// @param[in] xs Column-major query feature matrix (m x d).
        /// @param[in] n_draws Number of posterior function draws to generate.
        /// @param[in] seed Deterministic PRNG seed for sampling.
        /// @param[in] options Optional prediction controls.
        /// @param[in] jitter Optional override jitter policy for predictive covariance Cholesky stabilization.
        /// @return Column-major sample matrix (m x n_draws) promoted to double, or GprError on failure.
        [[nodiscard]] std::expected<std::vector<double>, GprError> sample(
            MatrixRef<double> xs,
            std::size_t n_draws,
            std::uint64_t seed,
            const PredictOptions& options = {},
            std::optional<JitterPolicy> jitter = std::nullopt
        ) const;

        /// @brief Draws n_draws deterministic posterior samples on single-precision query features.
        /// @param[in] xs Column-major float query feature matrix (m x d).
        /// @param[in] n_draws Number of posterior function draws to generate.
        /// @param[in] seed Deterministic PRNG seed for sampling.
        /// @param[in] options Optional prediction controls.
        /// @param[in] jitter Optional override jitter policy for predictive covariance Cholesky stabilization.
        /// @return Column-major sample matrix (m x n_draws) promoted to double, or GprError on failure.
        [[nodiscard]] std::expected<std::vector<double>, GprError> sample(
            MatrixRef<float> xs,
            std::size_t n_draws,
            std::uint64_t seed,
            const PredictOptions& options = {},
            std::optional<JitterPolicy> jitter = std::nullopt
        ) const;

        /// @brief Draws n_draws deterministic posterior samples from a raw single-precision query span.
        /// @param[in] xs Contiguous column-major float query data of length n_rows * n_cols.
        /// @param[in] n_rows Number of query points m.
        /// @param[in] n_cols Number of features per query point (must match d()).
        /// @param[in] n_draws Number of posterior function draws to generate.
        /// @param[in] seed Deterministic PRNG seed for sampling.
        /// @param[in] options Optional prediction controls.
        /// @param[in] jitter Optional predictive covariance jitter override.
        /// @return Column-major sample matrix (m x n_draws) promoted to double, or GprError on failure.
        [[nodiscard]] std::expected<std::vector<double>, GprError> sample(
            std::span<const float> xs,
            std::size_t n_rows,
            std::size_t n_cols,
            std::size_t n_draws,
            std::uint64_t seed,
            const PredictOptions& options = {},
            std::optional<JitterPolicy> jitter = std::nullopt
        ) const
        {
            auto view = cxxgpr::detail::matrix_view_from_span(xs, n_rows, n_cols);
            if (!view.has_value())
            {
                return std::unexpected(std::move(view).error());
            }
            return sample(*view, n_draws, seed, options, std::move(jitter));
        }

        /// @brief Computes closed-form Leave-One-Out prediction on the original target scale
        /// (double-promoted).
        /// @details Uses cached training Cholesky factor L and weights alpha to compute analytical LOO
        /// moments before applying inverse target transforms.
        /// @param[in] options Optional prediction controls.
        /// @return Double-precision LOO Prediction on the original target scale, or GprError on failure.
        /// @retval GprErrorCode::NonFiniteKernelValue When inverse target transform encounters non-finite outputs.
        [[nodiscard]] std::expected<Prediction<double>, GprError> loo_predict(const PredictOptions& options = {}) const;

        /// @brief Extracts a typed FittedGpr<O, P, KernelSpec> using Retain policy when P matches
        /// precision_kind(); returns PersistFailed(WrongModel) on precision mismatch.
        /// @details Copies restored tensors into an independent FittedPayload<P> paired with cloned
        /// unfitted/fitted transforms so the resulting FittedGpr is completely decoupled from mmap lifetimes.
        /// @tparam O Optimizer type associated with the returned FittedGpr for refit/into_trainer.
        /// @tparam P Precision policy type that must match precision_kind().
        /// @param[in] optimizer Optimizer instance stored on the extracted FittedGpr.
        /// @return Typed FittedGpr configured with CholeskyBuffer::Retain on precision match, or
        /// PersistFailed(WrongModel) on mismatch.
        /// @retval GprErrorCode::PersistFailed When P::persist_kind() != precision_kind().
        template <typename O = optimizer::Lbfgs, PrecisionPolicyConcept P = DoublePrecision>
        [[nodiscard]] std::expected<FittedGpr<O, P, KernelSpec>, GprError> into_fitted(O optimizer = O{}) const;

    private:
        /// @brief Opaque pimpl holding checkpoint configuration, mapped safetensors, and typed state.
        struct Impl;

        /// @brief Private constructor wrapping a validated shared checkpoint implementation.
        /// @param[in] impl Shared immutable checkpoint payload.
        explicit LoadedGpr(std::shared_ptr<const Impl> impl) noexcept : impl_(std::move(impl)) {}

        /// @brief Materializes an owned FittedPayload<P> and clones unfitted transforms from impl_.
        /// @tparam P Target precision policy matching impl_->precision_kind.
        /// @param[out] out_unfitted_in Output pointer populated with cloned unfitted input transform.
        /// @param[out] out_unfitted_tgt Output pointer populated with cloned unfitted target transform.
        /// @return Owned FittedPayload<P> or GprError on extraction failure.
        template <PrecisionPolicyConcept P>
        [[nodiscard]] std::expected<cxxgpr::detail::FittedPayload<P>, GprError> extract_payload(
            std::unique_ptr<transform::UnfittedInputTransform>& out_unfitted_in,
            std::unique_ptr<transform::UnfittedTargetTransform>& out_unfitted_tgt
        ) const;

        /// @brief Shared immutable checkpoint state loaded from disk.
        std::shared_ptr<const Impl> impl_;
    };

    /// @cond CXXGPR_OUT_OF_CLASS_DEFINITION
    /// @brief Out-of-class definition of LoadedGpr::into_fitted(); the declaration carries the public docs.
    /// @tparam O Optimizer type attached to the returned FittedGpr.
    /// @tparam P Precision policy matching precision_kind().
    /// @param[in] optimizer Optimizer configuration forwarded to the returned FittedGpr.
    /// @return Typed FittedGpr, or GprError on precision mismatch.
    /// @retval GprErrorCode::PersistFailed When P::persist_kind() does not match precision_kind().
    template <typename O, PrecisionPolicyConcept P>
    std::expected<FittedGpr<O, P, KernelSpec>, GprError> LoadedGpr::into_fitted(O optimizer) const
    {
        // Reject extraction if the requested precision policy differs from the checkpoint's stored mode.
        if (P::persist_kind() != precision_kind())
        {
            return std::unexpected(
                GprError::persist_failed(
                    PersistErrorKind::WrongModel, "requested precision policy does not match persisted model precision"
                )
            );
        }
        // Extract owned copies of tensors and transforms so FittedGpr outlives any mmap backing.
        std::unique_ptr<transform::UnfittedInputTransform> unfitted_in;
        std::unique_ptr<transform::UnfittedTargetTransform> unfitted_tgt;
        auto payload = extract_payload<P>(unfitted_in, unfitted_tgt);
        if (!payload.has_value())
        {
            return std::unexpected(std::move(payload).error());
        }
        return FittedGpr<O, P, KernelSpec>(
            std::move(*payload), std::move(optimizer), jitter(), std::move(unfitted_in), std::move(unfitted_tgt),
            kernel_exp(), FitPolicy{distance_cache_policy(), CholeskyBuffer::Retain}
        );
    }

    /// @endcond

    /// @brief Internal persistence helpers bridging typed FittedGpr accessors to disk writers.
    namespace detail
    {

        /// @brief Type-erased view of all configuration and tensor spans required to save a model.
        /// @details Avoids template bloat in persist.cpp by borrowing raw data spans and factor/alpha spans
        /// selected according to the caller's precision policy.
        struct SaveInput
        {
            /// @brief Number of training samples n.
            std::size_t n_train{0};
            /// @brief Number of training feature dimensions d.
            std::size_t n_features{0};
            /// @brief True when l and alpha tensors are persisted alongside raw training data.
            bool has_factor{false};
            /// @brief Precision policy tag identifying storage and refinement scalar types.
            PersistPrecisionKind precision_kind{PersistPrecisionKind::Double};
            /// @brief Exponential evaluation backend mode.
            KernelExp kernel_exp{KernelExp::Accurate};
            /// @brief Distance cache policy retained for factor reconstruction.
            DistanceCachePolicy distance_cache{DistanceCachePolicy::Cached};
            /// @brief Accepted diagonal jitter offset added during final Cholesky factorization.
            double factor_jitter{0.0};
            /// @brief Fitted kernel specification to serialize into config.json.
            const KernelSpec& kernel_spec;
            /// @brief Fitted Gaussian observation noise likelihood to serialize.
            const GaussianLikelihood& likelihood;
            /// @brief Cholesky jitter stabilization policy to serialize.
            const JitterPolicy& jitter;
            /// @brief Unfitted input transform configuration retained for refit/into_trainer.
            const transform::UnfittedInputTransform& unfitted_input;
            /// @brief Fitted input transform state applied to training/query coordinates.
            const transform::FittedInputTransform& fitted_input;
            /// @brief Unfitted target transform configuration retained for refit/into_trainer.
            const transform::UnfittedTargetTransform& unfitted_target;
            /// @brief Fitted target transform state used for inverse scaling.
            const transform::FittedTargetTransform& fitted_target;
            /// @brief Raw column-major training features in double precision (n * d).
            std::span<const double> x_raw;
            /// @brief Raw training targets in double precision (n).
            std::span<const double> y_raw;
            /// @brief Double-precision Cholesky factor L span (empty unless Storage == double and has_factor).
            std::span<const double> l_f64;
            /// @brief Single-precision Cholesky factor L span (empty unless Storage == float and has_factor).
            std::span<const float> l_f32;
            /// @brief Double-precision prediction weight vector alpha span (empty unless Refine == double).
            std::span<const double> alpha_f64;
            /// @brief Single-precision prediction weight vector alpha span (empty unless Refine == float).
            std::span<const float> alpha_f32;
        };

        /// @brief Validates inputs and writes config.json and model.safetensors atomically into dir.
        /// @details Stages temporary files inside dir and atomically renames them into place on success.
        /// @param[in] input Borrowed model configuration and raw/factor tensor spans.
        /// @param[in] dir Destination directory path created if absent.
        /// @param[in] registry Custom extension registry (currently unused during save; reserved for symmetry).
        /// @return Empty expected on success, or GprError on validation, serialization, or I/O failure.
        /// @retval GprErrorCode::EmptyInput When n_train or n_features is zero.
        /// @retval GprErrorCode::NonFiniteInput When jitter or any persisted tensor entry is non-finite.
        /// @retval GprErrorCode::PersistFailed When serialization, directory creation, writing, or rename fails.
        [[nodiscard]] std::expected<void, GprError> save_model_impl(
            const SaveInput& input, const std::filesystem::path& dir, const PersistRegistry& registry
        );

    }  // namespace detail

    /// @brief Atomically persists a fitted model without factor tensors (has_factor = false).
    /// @details Serializes configuration, transforms, and raw (x, y) data so LoadedGpr reconstructs the
    /// Cholesky factor upon loading, trading load time for smaller disk footprint.
    /// @tparam O Optimizer type held by model.
    /// @tparam P Precision policy used by model.
    /// @tparam K Kernel specification type (KernelSpec).
    /// @param[in] model Fitted GPR model to persist.
    /// @param[in] dir Target checkpoint directory path.
    /// @param[in] registry Optional custom extension registry.
    /// @return Empty expected on success, or GprError on non-persistable components or I/O failure.
    /// @retval GprErrorCode::PersistFailed When custom kernels/transforms are non-persistable or I/O fails.
    template <typename O, PrecisionPolicyConcept P, typename K>
    [[nodiscard]] std::expected<void, GprError> save(
        const FittedGpr<O, P, K>& model, const std::filesystem::path& dir, const PersistRegistry& registry = {}
    )
    {
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

    /// @brief Atomically persists a fitted model including factor L and alpha tensors (has_factor = true).
    /// @details Persists raw data alongside L (F64 or F32) and alpha (F64 or F32) matching precision policy P
    /// so LoadedGpr can map F64 factors directly with zero-copy mmap.
    /// @tparam O Optimizer type held by model.
    /// @tparam P Precision policy used by model.
    /// @tparam K Kernel specification type (KernelSpec).
    /// @param[in] model Fitted GPR model to persist.
    /// @param[in] dir Target checkpoint directory path.
    /// @param[in] registry Optional custom extension registry.
    /// @return Empty expected on success, or GprError on non-persistable components or I/O failure.
    /// @retval GprErrorCode::PersistFailed When custom kernels/transforms are non-persistable or I/O fails.
    template <typename O, PrecisionPolicyConcept P, typename K>
    [[nodiscard]] std::expected<void, GprError> save_with_factor(
        const FittedGpr<O, P, K>& model, const std::filesystem::path& dir, const PersistRegistry& registry = {}
    )
    {
        const auto l_ref = model.l_factor();
        const std::size_t n = model.n_train();
        // Bind the Cholesky factor span matching P::Storage (F64 vs F32).
        std::span<const double> l_f64;
        std::span<const float> l_f32;
        if constexpr (std::same_as<typename P::Storage, double>)
        {
            l_f64 = std::span<const double>(l_ref.data(), n * n);
        }
        else
        {
            l_f32 = std::span<const float>(l_ref.data(), n * n);
        }
        // Bind the prediction weight span matching P::Refine (F64 vs F32).
        std::span<const double> alpha_f64;
        std::span<const float> alpha_f32;
        if constexpr (std::same_as<typename P::Refine, double>)
        {
            alpha_f64 = model.alpha();
        }
        else
        {
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

    /// @brief Library-internal helpers of the supplied-distance persistence.
    namespace detail
    {

        /// @brief Library-internal bridge giving the persistence layer access to the state of a FittedDistanceGpr.
        struct DistanceModelAccess
        {
            /// @brief Returns the immutable fitted state of a model.
            /// @tparam O Optimizer type.
            /// @tparam P Precision policy.
            /// @tparam C Distance mode.
            /// @param[in] model Fitted model.
            /// @return Shared fitted state.
            template <typename O, PrecisionPolicyConcept P, DistanceModeConcept C>
            [[nodiscard]] static const std::shared_ptr<const cxxgpr::detail::DistanceFittedPayload<P, C>>& payload(
                const FittedDistanceGpr<O, P, C>& model
            ) noexcept
            {
                return model.payload_;
            }

            /// @brief Returns the jitter policy of a model.
            /// @tparam O Optimizer type.
            /// @tparam P Precision policy.
            /// @tparam C Distance mode.
            /// @param[in] model Fitted model.
            /// @return Jitter policy.
            template <typename O, PrecisionPolicyConcept P, DistanceModeConcept C>
            [[nodiscard]] static const JitterPolicy& jitter(const FittedDistanceGpr<O, P, C>& model) noexcept
            {
                return model.jitter_;
            }

            /// @brief Returns the exponential mode of a model.
            /// @tparam O Optimizer type.
            /// @tparam P Precision policy.
            /// @tparam C Distance mode.
            /// @param[in] model Fitted model.
            /// @return Kernel exponential mode.
            template <typename O, PrecisionPolicyConcept P, DistanceModeConcept C>
            [[nodiscard]] static KernelExp kernel_exp(const FittedDistanceGpr<O, P, C>& model) noexcept
            {
                return model.kernel_exp_;
            }

            /// @brief Returns the unfitted coordinate transform of a model.
            /// @tparam O Optimizer type.
            /// @tparam P Precision policy.
            /// @tparam C Distance mode.
            /// @param[in] model Fitted model.
            /// @return Unfitted input transform (identity for a DistanceOnly model).
            template <typename O, PrecisionPolicyConcept P, DistanceModeConcept C>
            [[nodiscard]] static const transform::UnfittedInputTransform& unfitted_input(
                const FittedDistanceGpr<O, P, C>& model
            ) noexcept
            {
                return *model.input_transform_;
            }

            /// @brief Returns the unfitted target transform of a model.
            /// @tparam O Optimizer type.
            /// @tparam P Precision policy.
            /// @tparam C Distance mode.
            /// @param[in] model Fitted model.
            /// @return Unfitted target transform.
            template <typename O, PrecisionPolicyConcept P, DistanceModeConcept C>
            [[nodiscard]] static const transform::UnfittedTargetTransform& unfitted_target(
                const FittedDistanceGpr<O, P, C>& model
            ) noexcept
            {
                return *model.target_transform_;
            }

            /// @brief Builds a fitted model from restored parts.
            /// @tparam O Optimizer type.
            /// @tparam P Precision policy.
            /// @tparam C Distance mode.
            /// @param[in] payload Restored fitted state.
            /// @param[in] optimizer Optimizer kept for refit and into_trainer.
            /// @param[in] jitter Jitter policy.
            /// @param[in] input_transform Unfitted coordinate transform.
            /// @param[in] target_transform Unfitted target transform.
            /// @param[in] kernel_exp Exponential mode.
            /// @return The fitted model.
            template <typename O, PrecisionPolicyConcept P, DistanceModeConcept C>
            [[nodiscard]] static FittedDistanceGpr<O, P, C> make(
                std::shared_ptr<const cxxgpr::detail::DistanceFittedPayload<P, C>> payload,
                O optimizer,
                JitterPolicy jitter,
                std::unique_ptr<transform::UnfittedInputTransform> input_transform,
                std::unique_ptr<transform::UnfittedTargetTransform> target_transform,
                KernelExp kernel_exp
            )
            {
                return FittedDistanceGpr<O, P, C>(
                    std::move(payload), std::move(optimizer), jitter, std::move(input_transform),
                    std::move(target_transform), kernel_exp
                );
            }
        };

        /// @brief Borrowed description of a supplied-distance model to persist.
        struct DistanceSaveInput
        {
            /// @brief Number of training points.
            std::size_t n_train{0};
            /// @brief Coordinate dimension (0 for a DistanceOnly model).
            std::size_t n_features{0};
            /// @brief Whether the factor and weights are written.
            bool has_factor{false};
            /// @brief Precision policy kind.
            PersistPrecisionKind precision_kind{PersistPrecisionKind::Double};
            /// @brief Exponential mode.
            KernelExp kernel_exp{KernelExp::Accurate};
            /// @brief Accepted diagonal jitter.
            double factor_jitter{0.0};
            /// @brief Kernel of the model (its mode selects DistanceOnly or WithPoints).
            std::variant<const DistanceKernel<DistanceOnly>*, const DistanceKernel<WithPoints>*> kernel;
            /// @brief Validated training tables (the scalar selects the stored precision).
            std::variant<const CheckedDistanceStore<double>*, const CheckedDistanceStore<float>*> store;
            /// @brief Likelihood with the fitted noise variance.
            const GaussianLikelihood& likelihood;
            /// @brief Jitter policy.
            const JitterPolicy& jitter;
            /// @brief Unfitted coordinate transform.
            const transform::UnfittedInputTransform& unfitted_input;
            /// @brief Fitted coordinate transform, or null for a DistanceOnly model.
            const transform::FittedInputTransform* fitted_input;
            /// @brief Unfitted target transform.
            const transform::UnfittedTargetTransform& unfitted_target;
            /// @brief Fitted target transform.
            const transform::FittedTargetTransform& fitted_target;
            /// @brief Raw training coordinates (empty for DistanceOnly).
            std::span<const double> x_raw;
            /// @brief Raw training targets.
            std::span<const double> y_raw;
            /// @brief Cholesky factor in double (empty unless double storage with a factor).
            std::span<const double> l_f64;
            /// @brief Cholesky factor in float (empty unless float storage with a factor).
            std::span<const float> l_f32;
            /// @brief Weights in double (empty unless double refinement with a factor).
            std::span<const double> alpha_f64;
            /// @brief Weights in float (empty unless float refinement with a factor).
            std::span<const float> alpha_f32;
        };

        /// @brief Validates and atomically writes the checkpoint of a supplied-distance model.
        /// @param[in] input Borrowed model description.
        /// @param[in] dir Destination directory.
        /// @param[in] registry Registry of custom kernels and transforms.
        /// @return Empty expected on success, or GprError on validation, serialization or I/O failure.
        [[nodiscard]] std::expected<void, GprError> save_distance_model_impl(
            const DistanceSaveInput& input, const std::filesystem::path& dir, const PersistRegistry& registry
        );

        /// @brief Builds the borrowed description of a fitted supplied-distance model and writes it.
        /// @tparam O Optimizer type.
        /// @tparam P Precision policy.
        /// @tparam C Distance mode.
        /// @param[in] model Fitted model.
        /// @param[in] dir Destination directory.
        /// @param[in] registry Registry of custom kernels and transforms.
        /// @param[in] with_factor Whether the factor and weights are written.
        /// @return Empty expected on success, or GprError.
        template <typename O, PrecisionPolicyConcept P, DistanceModeConcept C>
        [[nodiscard]] std::expected<void, GprError> save_distance_model(
            const FittedDistanceGpr<O, P, C>& model,
            const std::filesystem::path& dir,
            const PersistRegistry& registry,
            bool with_factor
        )
        {
            if (!model.retains_distances())
            {
                return std::unexpected(
                    GprError::persist_failed(
                        PersistErrorKind::NotPersistable,
                        "supplied-distance model trained without retained distances cannot be saved"
                    )
                );
            }
            const auto& payload = *DistanceModelAccess::payload(model);
            std::span<const double> l_f64;
            std::span<const float> l_f32;
            std::span<const double> alpha_f64;
            std::span<const float> alpha_f32;
            if (with_factor)
            {
                if constexpr (std::same_as<typename P::Storage, double>)
                {
                    l_f64 = payload.l_factor.view();
                }
                else
                {
                    l_f32 = payload.l_factor.view();
                }
                if constexpr (std::same_as<typename P::Refine, double>)
                {
                    alpha_f64 = payload.alpha;
                }
                else
                {
                    alpha_f32 = payload.alpha;
                }
            }
            const DistanceSaveInput input{
                .n_train = payload.n_train,
                .n_features = payload.n_features,
                .has_factor = with_factor,
                .precision_kind = P::persist_kind(),
                .kernel_exp = DistanceModelAccess::kernel_exp(model),
                .factor_jitter = payload.factor_jitter,
                .kernel = &payload.kernel,
                .store = payload.store.get(),
                .likelihood = payload.likelihood,
                .jitter = DistanceModelAccess::jitter(model),
                .unfitted_input = DistanceModelAccess::unfitted_input(model),
                .fitted_input = payload.input_transform.get(),
                .unfitted_target = DistanceModelAccess::unfitted_target(model),
                .fitted_target = *payload.target_transform,
                .x_raw = payload.x_raw,
                .y_raw = payload.y_raw,
                .l_f64 = l_f64,
                .l_f32 = l_f32,
                .alpha_f64 = alpha_f64,
                .alpha_f32 = alpha_f32,
            };
            return save_distance_model_impl(input, dir, registry);
        }

    }  // namespace detail

    /// @brief Saves a supplied-distance model without its factor.
    /// @details Writes config.json and model.safetensors atomically. The factor is recomputed on load.
    /// @tparam O Optimizer type.
    /// @tparam P Precision policy.
    /// @tparam C Distance mode.
    /// @param[in] model Fitted supplied-distance model.
    /// @param[in] dir Destination directory, created if needed.
    /// @param[in] registry Registry that names custom kernels and transforms.
    /// @return Empty expected on success, or GprError.
    /// @retval GprErrorCode::PersistFailed When a custom component is not persistable or writing fails.
    template <typename O, PrecisionPolicyConcept P, DistanceModeConcept C>
    [[nodiscard]] std::expected<void, GprError> save(
        const FittedDistanceGpr<O, P, C>& model, const std::filesystem::path& dir, const PersistRegistry& registry = {}
    )
    {
        return detail::save_distance_model(model, dir, registry, false);
    }

    /// @brief Saves a supplied-distance model together with its factor and weights.
    /// @details Loading such a checkpoint restores the exact fitted state, so predictions are bit-identical.
    /// @tparam O Optimizer type.
    /// @tparam P Precision policy.
    /// @tparam C Distance mode.
    /// @param[in] model Fitted supplied-distance model.
    /// @param[in] dir Destination directory, created if needed.
    /// @param[in] registry Registry that names custom kernels and transforms.
    /// @return Empty expected on success, or GprError.
    /// @retval GprErrorCode::PersistFailed When a custom component is not persistable or writing fails.
    template <typename O, PrecisionPolicyConcept P, DistanceModeConcept C>
    [[nodiscard]] std::expected<void, GprError> save_with_factor(
        const FittedDistanceGpr<O, P, C>& model, const std::filesystem::path& dir, const PersistRegistry& registry = {}
    )
    {
        return detail::save_distance_model(model, dir, registry, true);
    }

    /// @brief A supplied-distance model restored from disk.
    /// @details Holds the fitted state of one of the four precision policies. The loaded kernel owns freshly issued
    /// slots: take them from kernel().slots() to bind the query tables. Copies own independent query buffers.
    /// @tparam C Distance mode the checkpoint was saved with.
    template <DistanceModeConcept C> class LoadedDistanceGpr
    {
    public:
        /// @brief Loads a supplied-distance checkpoint.
        /// @param[in] dir Directory holding config.json and model.safetensors.
        /// @param[in] registry Registry that recreates custom kernels and transforms.
        /// @return The loaded model, or GprError.
        /// @retval GprErrorCode::PersistFailed When the files are unreadable, malformed, or hold another kind of model.
        /// @retval GprErrorCode::UnsupportedPersistVersion When the format version is not supported.
        /// @retval GprErrorCode::InvalidDistance When a stored distance table is invalid.
        [[nodiscard]] static std::expected<LoadedDistanceGpr, GprError> load(
            const std::filesystem::path& dir, const PersistRegistry& registry = {}
        );

        /// @brief Copy-constructs a loaded model with independent query buffers.
        /// @param[in] other Model to copy.
        LoadedDistanceGpr(const LoadedDistanceGpr& other);

        /// @brief Copy-assigns a loaded model with independent query buffers.
        /// @param[in] other Model to copy.
        /// @return This model.
        LoadedDistanceGpr& operator=(const LoadedDistanceGpr& other);

        /// @brief Move-constructs a loaded model.
        /// @param[in] other Model to move from.
        LoadedDistanceGpr(LoadedDistanceGpr&& other) noexcept;

        /// @brief Move-assigns a loaded model.
        /// @param[in] other Model to move from.
        /// @return This model.
        LoadedDistanceGpr& operator=(LoadedDistanceGpr&& other) noexcept;

        /// @brief Destroys the loaded model.
        ~LoadedDistanceGpr();

        /// @brief Returns the number of training points.
        /// @return Training point count.
        [[nodiscard]] std::size_t n() const noexcept;

        /// @brief Returns the coordinate dimension (0 for a DistanceOnly model).
        /// @return Coordinate dimension.
        [[nodiscard]] std::size_t d() const noexcept;

        /// @brief Returns the stored precision policy.
        /// @return Precision kind.
        [[nodiscard]] PersistPrecisionKind precision_kind() const noexcept;

        /// @brief Returns whether the checkpoint held the factor and weights.
        /// @return True when the factor was stored and not recomputed.
        [[nodiscard]] bool has_factor() const noexcept;

        /// @brief Returns whether the Cholesky factor is read directly from a read-only memory map of the file.
        /// @details True for a double-precision checkpoint written with save_with_factor(). The file must not be
        /// rewritten in place while this object or a copy of it exists; replacing it through save() is safe.
        /// into_fitted() copies the factor, so the returned model does not depend on the file.
        /// @return True when the factor is memory-mapped, false when it is owned.
        [[nodiscard]] bool has_mmap_factor() const noexcept;

        /// @brief Returns the accepted diagonal jitter of the factor.
        /// @return Jitter offset.
        [[nodiscard]] double factor_jitter() const;

        /// @brief Returns the kernel with its freshly issued slots.
        /// @return Kernel carrying the fitted hyperparameters.
        [[nodiscard]] const DistanceKernel<C>& kernel() const;

        /// @brief Returns the likelihood with the fitted noise variance.
        /// @return Gaussian likelihood.
        [[nodiscard]] const GaussianLikelihood& likelihood() const;

        /// @brief Returns the jitter policy.
        /// @return Jitter policy.
        [[nodiscard]] const JitterPolicy& jitter() const;

        /// @brief Returns the exponential mode.
        /// @return Kernel exponential mode.
        [[nodiscard]] KernelExp kernel_exp() const;

        /// @brief Predicts the mean and variance at the queries of a DistanceOnly model.
        /// @param[in] cross Training-to-query tables bound to the slots of kernel().
        /// @param[in] options Variance selection.
        /// @return Prediction in double, or GprError.
        [[nodiscard]] std::expected<Prediction<double>, GprError> predict(
            const DistanceSource& cross, const PredictOptions& options = {}
        ) const
            requires std::same_as<C, DistanceOnly>;

        /// @brief Predicts the mean and variance at the queries of a WithPoints model.
        /// @param[in] cross Training-to-query tables bound to the slots of kernel().
        /// @param[in] xq Query coordinates, m x d.
        /// @param[in] options Variance selection.
        /// @return Prediction in double, or GprError.
        [[nodiscard]] std::expected<Prediction<double>, GprError> predict(
            const DistanceSource& cross, MatrixRef<double> xq, const PredictOptions& options = {}
        ) const
            requires std::same_as<C, WithPoints>;

        /// @brief Predicts the mean and variance at float query coordinates of a WithPoints model.
        /// @param[in] cross Training-to-query tables bound to the slots of kernel().
        /// @param[in] xq Single-precision query coordinates, m x d.
        /// @param[in] options Variance selection.
        /// @return Prediction in double, or GprError.
        [[nodiscard]] std::expected<Prediction<double>, GprError> predict(
            const DistanceSource& cross, MatrixRef<float> xq, const PredictOptions& options = {}
        ) const
            requires std::same_as<C, WithPoints>;

        /// @brief Computes the joint predictive covariance of a DistanceOnly model.
        /// @param[in] cross Training-to-query tables.
        /// @param[in] query Query-to-query tables.
        /// @param[in] options Variance selection.
        /// @return Mean and covariance in double, or GprError.
        [[nodiscard]] std::expected<PredictiveCovariance<double>, GprError> predict_covariance(
            const DistanceSource& cross, const DistanceSource& query, const PredictOptions& options = {}
        ) const
            requires std::same_as<C, DistanceOnly>;

        /// @brief Computes the joint predictive covariance of a WithPoints model.
        /// @param[in] cross Training-to-query tables.
        /// @param[in] query Query-to-query tables.
        /// @param[in] xq Query coordinates, m x d.
        /// @param[in] options Variance selection.
        /// @return Mean and covariance in double, or GprError.
        [[nodiscard]] std::expected<PredictiveCovariance<double>, GprError> predict_covariance(
            const DistanceSource& cross,
            const DistanceSource& query,
            MatrixRef<double> xq,
            const PredictOptions& options = {}
        ) const
            requires std::same_as<C, WithPoints>;

        /// @brief Computes the joint predictive covariance of a WithPoints model on float coordinates.
        /// @param[in] cross Training-to-query tables.
        /// @param[in] query Query-to-query tables.
        /// @param[in] xq Single-precision query coordinates, m x d.
        /// @param[in] options Variance selection.
        /// @return Mean and covariance in double, or GprError.
        [[nodiscard]] std::expected<PredictiveCovariance<double>, GprError> predict_covariance(
            const DistanceSource& cross,
            const DistanceSource& query,
            MatrixRef<float> xq,
            const PredictOptions& options = {}
        ) const
            requires std::same_as<C, WithPoints>;

        /// @brief Draws joint posterior samples of a DistanceOnly model.
        /// @param[in] cross Training-to-query tables.
        /// @param[in] query Query-to-query tables.
        /// @param[in] n_draws Number of samples.
        /// @param[in] seed Seed of the deterministic generator.
        /// @param[in] options Variance selection.
        /// @return Samples in double, m values per draw, or GprError.
        [[nodiscard]] std::expected<std::vector<double>, GprError> sample(
            const DistanceSource& cross,
            const DistanceSource& query,
            std::size_t n_draws,
            std::uint64_t seed,
            const PredictOptions& options = {}
        ) const
            requires std::same_as<C, DistanceOnly>;

        /// @brief Draws joint posterior samples of a WithPoints model.
        /// @param[in] cross Training-to-query tables.
        /// @param[in] query Query-to-query tables.
        /// @param[in] xq Query coordinates, m x d.
        /// @param[in] n_draws Number of samples.
        /// @param[in] seed Seed of the deterministic generator.
        /// @param[in] options Variance selection.
        /// @return Samples in double, m values per draw, or GprError.
        [[nodiscard]] std::expected<std::vector<double>, GprError> sample(
            const DistanceSource& cross,
            const DistanceSource& query,
            MatrixRef<double> xq,
            std::size_t n_draws,
            std::uint64_t seed,
            const PredictOptions& options = {}
        ) const
            requires std::same_as<C, WithPoints>;

        /// @brief Draws joint posterior samples of a WithPoints model on float coordinates.
        /// @param[in] cross Training-to-query tables.
        /// @param[in] query Query-to-query tables.
        /// @param[in] xq Single-precision query coordinates, m x d.
        /// @param[in] n_draws Number of samples.
        /// @param[in] seed Seed of the deterministic generator.
        /// @param[in] options Variance selection.
        /// @return Samples in double, m values per draw, or GprError.
        [[nodiscard]] std::expected<std::vector<double>, GprError> sample(
            const DistanceSource& cross,
            const DistanceSource& query,
            MatrixRef<float> xq,
            std::size_t n_draws,
            std::uint64_t seed,
            const PredictOptions& options = {}
        ) const
            requires std::same_as<C, WithPoints>;

        /// @brief Computes leave-one-out predictions at the training points.
        /// @param[in] options Variance selection.
        /// @return Leave-one-out prediction in double, or GprError.
        [[nodiscard]] std::expected<Prediction<double>, GprError> loo_predict(const PredictOptions& options = {}) const;

        /// @brief Extracts a typed FittedDistanceGpr that shares the restored state.
        /// @tparam O Optimizer type kept for refit and into_trainer.
        /// @tparam P Precision policy; must match precision_kind().
        /// @param[in] optimizer Optimizer to keep.
        /// @return The fitted model, or GprError when P does not match the stored precision.
        /// @retval GprErrorCode::PersistFailed When the requested precision policy differs from the stored one.
        template <typename O = optimizer::Lbfgs, PrecisionPolicyConcept P = DoublePrecision>
        [[nodiscard]] std::expected<FittedDistanceGpr<O, P, C>, GprError> into_fitted(O optimizer = O{}) const
        {
            if (P::persist_kind() != precision_kind())
            {
                return std::unexpected(
                    GprError::persist_failed(
                        PersistErrorKind::WrongModel,
                        "requested precision policy does not match persisted model precision"
                    )
                );
            }
            return detail::DistanceModelAccess::make<O, P, C>(
                fitted_payload<P>(), std::move(optimizer), jitter(), unfitted_input_clone(), unfitted_target_clone(),
                kernel_exp()
            );
        }

    private:
        /// @brief Pimpl state holding the fitted model of the stored precision.
        struct Impl;

        /// @brief Wraps a restored state.
        /// @param[in] impl Restored state.
        explicit LoadedDistanceGpr(std::unique_ptr<Impl> impl) noexcept;

        /// @brief Returns the shared fitted state for precision policy P.
        /// @tparam P Precision policy matching precision_kind().
        /// @return Shared fitted state.
        template <PrecisionPolicyConcept P>
        [[nodiscard]] std::shared_ptr<const cxxgpr::detail::DistanceFittedPayload<P, C>> fitted_payload() const;

        /// @brief Returns a copy of the unfitted coordinate transform.
        /// @return Cloned transform.
        [[nodiscard]] std::unique_ptr<transform::UnfittedInputTransform> unfitted_input_clone() const;

        /// @brief Returns a copy of the unfitted target transform.
        /// @return Cloned transform.
        [[nodiscard]] std::unique_ptr<transform::UnfittedTargetTransform> unfitted_target_clone() const;

        /// @brief Owning pointer to the restored state.
        std::unique_ptr<Impl> impl_;
    };

}  // namespace cxxgpr::persist

/// @}
