// SPDX-License-Identifier: Apache-2.0
/// @file config_json.hpp
/// @brief Internal serializer and parser declarations for persisted config.json metadata.
/// @details Defines ParsedConfig and the JSON serialization/deserialization entry points bridging
/// SaveInput and PersistRegistry to the on-disk Exact GPR config.json schema.

#pragma once

#include "cxxgpr/core/error.hpp"
#include "cxxgpr/core/likelihood.hpp"
#include "cxxgpr/kernel/distance_cache.hpp"
#include "cxxgpr/kernel/exp.hpp"
#include "cxxgpr/kernel/spec.hpp"
#include "cxxgpr/linalg/jitter.hpp"
#include "cxxgpr/persist/persist.hpp"
#include "cxxgpr/precision/policy.hpp"
#include "cxxgpr/transform/input.hpp"
#include "cxxgpr/transform/target.hpp"

#include <cstddef>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

/// @brief Internal persistence helpers for Exact GPR serialization and loading.
namespace cxxgpr::persist::detail
{

    /// @brief Validated configuration metadata parsed from a persisted config.json file.
    /// @details Holds model dimensions, factor metadata, precision policy, kernel tree, likelihood,
    /// jitter policy, and both unfitted and fitted input/target transforms.
    struct ParsedConfig
    {
        /// @brief Number of training samples n.
        std::size_t n{0};
        /// @brief Number of input feature dimensions d.
        std::size_t d{0};
        /// @brief True when model.safetensors stores precomputed Cholesky factor l and weight vector alpha.
        bool has_factor{false};
        /// @brief Accepted diagonal jitter offset recorded during the persisted factorization.
        double factor_jitter{0.0};
        /// @brief Precision and residual evaluation mode encoded in config.json.
        PersistPrecisionKind precision_kind{PersistPrecisionKind::Double};
        /// @brief Exponential evaluation backend mode (Accurate or FastApprox).
        KernelExp kernel_exp{KernelExp::Accurate};
        /// @brief Distance matrix caching policy for training factor reconstruction.
        DistanceCachePolicy distance_cache{DistanceCachePolicy::Cached};
        /// @brief Deserialized kernel expression specification with hyperparameters and bounds.
        std::optional<KernelSpec> kernel_spec;
        /// @brief Deserialized Gaussian observation noise likelihood with bounds.
        std::optional<GaussianLikelihood> likelihood;
        /// @brief Deserialized Cholesky stabilization jitter policy.
        JitterPolicy jitter;
        /// @brief Reconstructed unfitted input transform for refit and into_trainer workflows.
        std::unique_ptr<transform::UnfittedInputTransform> x_unfitted;
        /// @brief Reconstructed fitted input transform applied to training and query features.
        std::unique_ptr<transform::FittedInputTransform> x_transform;
        /// @brief Reconstructed unfitted target transform for refit and into_trainer workflows.
        std::unique_ptr<transform::UnfittedTargetTransform> y_unfitted;
        /// @brief Reconstructed fitted target transform used for inverse prediction scaling.
        std::unique_ptr<transform::FittedTargetTransform> y_transform;
    };

    /// @brief Serializes model metadata, kernel tree, likelihood, jitter, and transforms into config.json text.
    /// @details Encodes bounded parameters with open intervals and delegates custom kernel/transform state
    /// serialization while enforcing nesting depth and persist_id rules.
    /// @param input Borrowed model configuration and transform references to serialize.
    /// @return Formatted JSON string on success, or GprError if serialization or validation fails.
    /// @retval GprErrorCode::PersistFailed When custom components are not persistable or exceed depth limits.
    [[nodiscard]] std::expected<std::string, GprError> serialize_config_json(const SaveInput& input);

    /// @brief Reads, validates, and parses a persisted config.json file into ParsedConfig.
    /// @details Enforces MAX_JSON_NESTING_DEPTH prior to parsing, verifies format_version and factor_kind,
    /// rejects incompatible distance-only or non-Exact checkpoints, and reconstructs custom extensions via registry.
    /// @param path Filesystem path to config.json.
    /// @param registry Registry supplying factories for user-defined custom kernels and transforms.
    /// @return Populated ParsedConfig on success, or GprError on I/O, schema, version, or registry failure.
    /// @retval GprErrorCode::PersistFailed When file I/O, JSON schema, model kind, or registry lookup fails.
    /// @retval GprErrorCode::UnsupportedPersistVersion When format_version does not match FORMAT_VERSION.
    [[nodiscard]] std::expected<ParsedConfig, GprError> parse_config_json(
        const std::filesystem::path& path, const PersistRegistry& registry
    );

}  // namespace cxxgpr::persist::detail
