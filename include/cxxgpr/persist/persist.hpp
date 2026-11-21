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

    /// @brief Validates a user-supplied persist_id against empty and reserved prefix rules.
    /// @details Rejects empty strings and identifiers beginning with RESERVED_PREFIX ("cxxgpr.").
    /// @param[in] persist_id Candidate extension identifier string.
    /// @return Empty expected on success, or PersistFailed(InvalidPersistId) when invalid.
    /// @retval GprErrorCode::PersistFailed When persist_id is empty or uses the reserved prefix.
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
        [[nodiscard]] std::expected<Prediction<double>, GprError> predict(
            MatrixRef<double> xs, const PredictOptions& options = {}
        ) const;

        /// @brief Computes posterior mean and diagonal variance from a raw column-major span.
        /// @details Wraps the raw column-major query buffer into a MatrixRef<double> and delegates to predict().
        /// @param[in] xs Contiguous column-major query data of length n_rows * n_cols.
        /// @param[in] n_rows Number of query points m.
        /// @param[in] n_cols Number of features per query point (must match d()).
        /// @param[in] options Optional prediction controls.
        /// @return Double-precision Prediction on the original target scale, or GprError on failure.
        /// @retval GprErrorCode::EmptyInput When xs has zero elements or null data.
        /// @retval GprErrorCode::LengthMismatch When xs.size() != n_rows * n_cols.
        /// @retval GprErrorCode::DimensionMismatch When n_cols != d().
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
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

        /// @brief Computes posterior mean and diagonal variance into out (double-promoted).
        /// @details Reuses internal query scratch buffers and writes results into out on the original scale.
        /// @param[in] xs Column-major query feature matrix (m x d).
        /// @param[out] out Destination prediction struct populated on success.
        /// @param[in] options Optional prediction controls.
        /// @return Empty expected on success, or GprError on validation or numerical failure.
        /// @retval GprErrorCode::EmptyInput When xs is empty or null.
        /// @retval GprErrorCode::DimensionMismatch When xs.cols() != d().
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
        [[nodiscard]] std::expected<void, GprError> predict_into(
            MatrixRef<double> xs, Prediction<double>& out, const PredictOptions& options = {}
        ) const;

        /// @brief Computes full m x m posterior covariance matrix (double-promoted).
        /// @details Transforms query coordinates, evaluates full posterior covariance under the persisted
        /// precision policy, and maps both mean and covariance back to the original target scale.
        /// @param[in] xs Column-major query feature matrix (m x d).
        /// @param[in] options Optional prediction controls.
        /// @return Double-precision PredictiveCovariance or GprError on validation or numerical failure.
        /// @retval GprErrorCode::EmptyInput When xs is empty or null.
        /// @retval GprErrorCode::DimensionMismatch When xs.cols() != d().
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
        [[nodiscard]] std::expected<PredictiveCovariance<double>, GprError> predict_covariance(
            MatrixRef<double> xs, const PredictOptions& options = {}
        ) const;

        /// @brief Draws n_draws deterministic posterior samples (double-promoted).
        /// @details Evaluates posterior draws using seed and maps each draw back to the original target scale.
        /// @param[in] xs Column-major query feature matrix (m x d).
        /// @param[in] n_draws Number of posterior function draws to generate.
        /// @param[in] seed Deterministic PRNG seed for sampling.
        /// @param[in] options Optional prediction controls.
        /// @param[in] jitter Optional override jitter policy for predictive covariance Cholesky stabilization.
        /// @return Column-major sample matrix (m x n_draws) promoted to double, or GprError on failure.
        /// @retval GprErrorCode::EmptyInput When xs is empty or null.
        /// @retval GprErrorCode::DimensionMismatch When xs.cols() != d().
        /// @retval GprErrorCode::NonFiniteInput When xs contains NaN or Inf.
        /// @retval GprErrorCode::CholeskyFailed When predictive covariance factorization fails.
        [[nodiscard]] std::expected<std::vector<double>, GprError> sample(
            MatrixRef<double> xs,
            std::size_t n_draws,
            std::uint64_t seed,
            const PredictOptions& options = {},
            std::optional<JitterPolicy> jitter = std::nullopt
        ) const;

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
        [[nodiscard]] std::expected<detail::FittedPayload<P>, GprError> extract_payload(
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

}  // namespace cxxgpr::persist

/// @}
