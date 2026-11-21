// SPDX-License-Identifier: Apache-2.0
/// @file save_id.hpp
/// @brief Content-addressed pairing identifier calculation and validation for persisted checkpoints.
/// @details Computes the 32-lowercase-hex-digit SHA-256 content identifier shared between config.json
/// (`save_id`) and model.safetensors (`__metadata__["cxxgpr.save_id"]`), validates identifier syntax,
/// and enforces save-pair matching on load.

#pragma once

#include "cxxgpr/core/error.hpp"
#include "persist/safetensors.hpp"

#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>

/// @brief Internal persistence helpers for Exact GPR serialization and loading.
namespace cxxgpr::persist::detail
{

    /// @brief Returns true if text is a valid 32-lowercase-hex-digit pairing identifier.
    /// @param[in] text Candidate identifier string.
    /// @return True iff text has length 32 and every character is in `[0-9a-f]`.
    [[nodiscard]] bool is_valid_save_id(std::string_view text) noexcept;

    /// @brief Computes the pairing identifier for a checkpoint from config.json (without save_id) and tensors.
    /// @details Hashes the domain tag `cxxgpr.save_id.v1\0`, length-prefixed config JSON bytes, and length-prefixed
    /// tensor descriptors and raw payloads in storage offset order using SHA-256, returning the first 16 bytes
    /// as 32 lowercase hexadecimal digits.
    /// @param[in] config_json_without_id Serialized config.json bytes without the `save_id` top-level key.
    /// @param[in] tensors Ordered tensor descriptors and raw payloads in model.safetensors offset order.
    /// @return 32-character lowercase hexadecimal pairing identifier.
    [[nodiscard]] std::string compute_save_id(
        std::string_view config_json_without_id, std::span<const TensorWriteView> tensors
    );

    /// @brief Verifies that config.json and model.safetensors pairing identifiers agree on load.
    /// @details Succeeds when both identifiers are absent (legacy checkpoints) or both are present and equal.
    /// Rejects mismatched or one-sided identifiers with PersistFailed(Tensor).
    /// @param[in] config_id Optional validated `save_id` read from config.json.
    /// @param[in] tensor_id Optional validated `cxxgpr.save_id` read from model.safetensors `__metadata__`.
    /// @return Empty expected on agreement, or PersistFailed(Tensor) when files come from different saves.
    /// @retval GprErrorCode::PersistFailed When pairing identifiers differ or only one file carries an identifier.
    [[nodiscard]] std::expected<void, GprError> check_save_id_pairing(
        const std::optional<std::string>& config_id, const std::optional<std::string>& tensor_id
    );

}  // namespace cxxgpr::persist::detail
